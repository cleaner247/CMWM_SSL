# File purpose: CMWM_SSL main student/teacher ViT-FSQ learner.
import torch
import torch.nn as nn
import torch.nn.functional as F

from CMWM_SSL.models.ViT import ViT
from CMWM_SSL.models.VQ import SequenceFSQ



class SSL_FSQ_Learner_NoPredictor(nn.Module):
    """
    【无 Predictor 版本】
    依靠 FSQ 模块在 Teacher 端的几何操作（Rotate->Mean->Rotate back）
    与 Student 端的原始输出之间的不对称性来驱动学习。
    """
    def __init__(self,
                 encoder_config: dict,
                 teacher_momentum: float = 0.996):
        super().__init__()

        self.teacher_momentum = teacher_momentum
        self.d_out = encoder_config['d_out'] # 每个token的维度
        self.device = encoder_config['device']
        self.geometry_type = encoder_config.get('geometry_type', 'se2_yaw')
        self.beta = encoder_config.get('beta', 1.0)
        if self.geometry_type != 'se2_yaw':
            raise ValueError('This CMWM_SSL upload only supports x,y,yaw labels with geometry_type=se2_yaw')

        # 1. Student Encoder
        self.student_encoder = self._build_vit(encoder_config)

        # 2. Teacher Encoder
        self.teacher_encoder = self._build_vit(encoder_config)

        # 3. FSQ 模块 (Teacher 专用，作为几何滤波器)
        num_tokens = encoder_config['img_height'] * encoder_config['img_width'] // (encoder_config['p_in_encoder']) ** 2 #16
        self.fsq_layer = SequenceFSQ(
            num=num_tokens,
            d=self.d_out,
            random_group=encoder_config['random_group'],
            active_ratio=encoder_config['active_ratio'],
            device=self.device
        )

        # 【注意】这里移除了 self.predictor
        # Student 将被迫直接输出符合 Teacher 分布特征的表示

        # 初始化 Teacher
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False

    def _build_vit(self, config):
        return ViT(
            image_size=(config['img_height'], config['img_width']),
            p_in=config['p_in_encoder'],
            p_out=1,
            dim=config['dim'],
            depth=config['depth'],
            heads=config['dim'] // 64,
            mlp_dim=config['dim'] * 4,
            dim_in=4,
            dim_out=config['d_out'],
            dim_head=64,
            patch_in_out=(True, False),
            use_temporal=False,
            causal=True
        )

    def _format_latent(self, x):
        """
        Student 侧的预处理：
        虽然没有 Predictor，但我们仍需确保 Student 的输出
        位于单位圆上（归一化），以符合 FSQ 的几何定义。
        """
        B, S, num, d = x.shape
        group_size = 2
        n_rot = (num * d) // group_size

        # 变形为坐标对
        x = x.view(B, S, n_rot, group_size)

        # 强制归一化：Student 必须学会输出模长为 1 的向量对
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

        # 还原形状
        x = x.view(B, S, num, d)
        return x

    def forward_student(self, x, label=None):
        """
        Student 端：ViT -> Normalize
        直接输出归一化后的 Latent，没有 Predictor 层的映射。
        """
        # [B, S, num, d]
        feat = self.student_encoder(x)

        # [B, S, num, d] (几何归一化后)
        feat_norm = self._format_latent(feat)

        return feat_norm

    @torch.no_grad()
    def forward_teacher(self, x, label):
        """
        Teacher 端：ViT -> FSQ Rotation (Filter) -> Reshape
        """
        B, S, _, _, _ = x.shape

        # 1. 提取特征
        feat = self.teacher_encoder(x)

        # 2. FSQ 旋转处理 (引入非对称性)
        # 这里的处理包含了利用 label 进行的 rotate 和 mean
        x_st, _ = self.fsq_layer(feat, label)

        # 3. 还原形状 [B, S, n_rot, 2] -> [B, S, num, d]
        x_st = x_st.view(B, S, self.fsq_layer.num, self.fsq_layer.d)

        return x_st.detach()

    @torch.no_grad()
    def update_moving_average(self):
        for s_params, t_params in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            old_weight, up_weight = t_params.data, s_params.data
            t_params.data = old_weight * self.teacher_momentum + up_weight * (1 - self.teacher_momentum)

    def loss_function(self, student_out, teacher_out):
        """SSL commitment loss between student features and stopgrad teacher target."""
        mse_loss = F.mse_loss(student_out, teacher_out)
        return {
            'loss': mse_loss * self.beta,
            'Commitment_Loss': mse_loss,
        }
