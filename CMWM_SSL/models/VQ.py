# File purpose: CMWM_SSL FSQ/rotation module used by the teacher-side geometric filter.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SequenceFSQ(nn.Module):
    def __init__(self, num, d, random_group=False, active_ratio=[2,2,1,1], device='cpu'):
        super().__init__()
        assert (num * d) % 2 == 0, "num * d 必须是偶数"
        assert len(active_ratio) == 4, "active_ratio 必须包含4个元素 [x, y, r, fixed]"

        self.device = device
        self.num = num  # 16 token数
        self.d = d  # 768
        self.n_rot = (num * d) // 2  # 768*16 角度有 768*8
        self.random_group = random_group
        self.active_ratio = active_ratio

        # 根据划分模式创建theta和region_index
        if random_group:
            full_theta, region_index = self._create_token_based_partition()

        else:
            full_theta, region_index = self._create_global_partition()

        self.register_buffer('theta', full_theta)
        self.register_buffer('region_index', region_index)

    def _calculate_partition_lengths(self, total_length):
        """计算各区域的长度

        Args:
            total_length: 要划分的总长度

        Returns:
            tuple: (len_x, len_y, len_r, len_fixed)
        """
        total_ratio = sum(self.active_ratio)

        len_x = int(total_length * self.active_ratio[0] / total_ratio)
        len_y = int(total_length * self.active_ratio[1] / total_ratio)
        len_r = int(total_length * self.active_ratio[2] / total_ratio)
        len_fixed = total_length - len_x - len_y - len_r

        return len_x, len_y, len_r, len_fixed

    def _create_theta_and_index(self, len_x, len_y, len_r, len_fixed):
        """根据各区域长度创建theta和region_index

        Args:
            len_x, len_y, len_r, len_fixed: 各区域的长度

        Returns:
            tuple: (theta, region_index)
        """
        # 创建theta
        theta_x = self._create_grouped_theta(len_x, num_groups=8)
        theta_y = self._create_grouped_theta(len_y, num_groups=8)
        theta_r = torch.ones(len_r, device=self.device)
        theta_fixed = torch.zeros(len_fixed, device=self.device)

        theta = torch.cat([theta_x, theta_y, theta_r, theta_fixed], dim=0)

        # 创建索引映射: 0-x区域, 1-y区域, 2-r区域, 3-fixed区域
        index_x = torch.zeros(len_x, dtype=torch.long, device=self.device)
        index_y = torch.ones(len_y, dtype=torch.long, device=self.device)
        index_r = torch.full((len_r,), 2, dtype=torch.long, device=self.device)
        index_fixed = torch.full((len_fixed,), 3, dtype=torch.long, device=self.device)

        region_index = torch.cat([index_x, index_y, index_r, index_fixed], dim=0)

        return theta, region_index

    def _create_global_partition(self):
        """在整个latent上划分"""
        len_x, len_y, len_r, len_fixed = self._calculate_partition_lengths(self.n_rot)

        print(f"Global split - x:{len_x}, y:{len_y}, r:{len_r}, fixed:{len_fixed}")

        theta, region_index = self._create_theta_and_index(len_x, len_y, len_r, len_fixed)

        return theta, region_index

    def _create_token_based_partition(self):
        """在每个token内部划分"""
        token_dim = self.d // 2  # 每个token的n_rot维度数

        len_x, len_y, len_r, len_fixed = self._calculate_partition_lengths(token_dim)

        print(f"Per-token split - x:{len_x}, y:{len_y}, r:{len_r}, fixed:{len_fixed}")

        # 为单个token创建theta和index
        token_theta, token_index = self._create_theta_and_index(len_x, len_y, len_r, len_fixed)

        # 复制num次
        full_theta = token_theta.repeat(self.num)
        region_index = token_index.repeat(self.num)

        return full_theta, region_index

    def _create_grouped_theta(self, length, num_groups=8):
        """创建分组的theta值"""
        if length == 0:
            return torch.tensor([], device=self.device)

        group_size = math.ceil(length / num_groups)
        group_indices = torch.arange(length, device=self.device, dtype=torch.long) // group_size
        group_indices = torch.clamp(group_indices, 0, num_groups - 1)

        # 创建值映射
        group_values = torch.ones(num_groups, device=self.device)
        sqrt_e_recip = 1.0 / math.sqrt(math.e)
        powers = torch.arange(1, num_groups, device=self.device)
        group_values[1:] = sqrt_e_recip ** powers

        theta = group_values[group_indices]
        return theta

    def forward(self, x, y):
        """
        x: (B, S, num, d)
        y: (B, S, 3) - 3个维度分别对应 [x_change, y_change, r_change]
        """
        B, S, num, d = x.shape
        assert num == self.num and d == self.d
        assert y.shape[-1] == 3, "y的最后一维必须是3 [x_change, y_change, r_change]"

        x = x.view(B, S, self.n_rot, 2)  # => (B, S, n_rot, 2)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

        # 根据 region_index 扩展 y 到 (B, S, n_rot)
        # region_index: 0-x, 1-y, 2-r, 3-fixed
        y_expanded = torch.zeros(B, S, self.n_rot, device=x.device)
        for i in range(3):  # 只处理前3个区域
            mask = (self.region_index == i)
            y_expanded[:, :, mask] = y[:, :, i:i+1]

        # === Step 1: 逆时针旋转 ===
        angles = y_expanded * self.theta.view(1, 1, -1)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x1, x2 = x[..., 0], x[..., 1]
        x_rot = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        # === Step 2: 均值后归一化 ===
        x_ = x_rot.mean(dim=1)
        x_ = x_ / (torch.norm(x_, dim=-1, keepdim=True) + 1e-8)

        # === Step 3: 顺时针旋转回来 S 次 ===
        x_ = x_.unsqueeze(1).expand(-1, S, -1, -1)
        angles_inv = -angles
        cos_inv = torch.cos(angles_inv)
        sin_inv = torch.sin(angles_inv)

        x1_, x2_ = x_[..., 0], x_[..., 1]
        x_restored = torch.stack([
            x1_ * cos_inv - x2_ * sin_inv,
            x1_ * sin_inv + x2_ * cos_inv
        ], dim=-1)
        x_restored = x_restored / (torch.norm(x_restored, dim=-1, keepdim=True) + 1e-8)

        # === Step 4: 计算 commit_loss + straight-through estimator ===
        commit_loss = F.mse_loss(x_restored.detach(), x, reduction='mean')
        x_st = x + (x_restored - x).detach()

        return x_st, commit_loss

    def test_forward(self, x_init, y_init, y_new):
        """
        测试阶段的前向传播
        Args:
            x_init: [B x S x num x d] 初始的潜在表示
            y_init: [B x S x 3] 初始位置标签
            y_new: [B x S_new x 3] 新的位置标签
        Returns:
            quantized: [B x S_new x num x d] 调整后的量化表示
        """
        B, S, num, d = x_init.shape
        _, S_new, _ = y_new.shape
        assert y_init.shape[-1] == 3 and y_new.shape[-1] == 3

        x = x_init.view(B, S, self.n_rot, 2)
        x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-8)

        # 扩展 y_init 和 y_new
        y_init_expanded = torch.zeros(B, S, self.n_rot, device=x.device)
        y_new_expanded = torch.zeros(B, S_new, self.n_rot, device=x.device)

        for i in range(3):
            mask = (self.region_index == i)
            y_init_expanded[:, :, mask] = y_init[:, :, i:i+1]
            y_new_expanded[:, :, mask] = y_new[:, :, i:i+1]

        # === Step 1: 逆时针旋转 (初始) ===
        angles_init = y_init_expanded * self.theta.view(1, 1, -1)
        cos_init = torch.cos(angles_init)
        sin_init = torch.sin(angles_init)

        x1, x2 = x[..., 0], x[..., 1]
        x_rot_init = torch.stack([
            x1 * cos_init - x2 * sin_init,
            x1 * sin_init + x2 * cos_init
        ], dim=-1)

        # === Step 2: 均值后归一化 ===
        x_ = x_rot_init.mean(dim=1)
        x_ = x_ / (torch.norm(x_, dim=-1, keepdim=True) + 1e-8)

        # === Step 3: 顺时针旋转回来 (新位置) ===
        x_ = x_.unsqueeze(1).expand(-1, S_new, -1, -1)
        angles_new = -y_new_expanded * self.theta.view(1, 1, -1)
        cos_new = torch.cos(angles_new)
        sin_new = torch.sin(angles_new)

        x1_, x2_ = x_[..., 0], x_[..., 1]
        x_restored = torch.stack([
            x1_ * cos_new - x2_ * sin_new,
            x1_ * sin_new + x2_ * cos_new
        ], dim=-1)
        x_restored = x_restored / (torch.norm(x_restored, dim=-1, keepdim=True) + 1e-8)

        return x_restored
