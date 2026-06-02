# File purpose: CMWM_SSL single-process decoder training entry from SSL features to VAE latents.
"""
Train Decoder for SSL-FSQ Model
使用训练好的 Teacher 网络作为 Encoder，训练一个 Decoder 来重建 VAE latent
模仿 train_decoder_multi.py 的训练流程
"""
import os
import sys
import time
import argparse
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data_load.data_latents_test import get_recon_loaders as get_test_recon_loaders
from CMWM_SSL.models.SSL_fsq import SSL_FSQ_Learner_NoPredictor
from CMWM_SSL.models.ViT import ViT
from first_stage_model.load_first_stage_model import load_first_stage_model
from util import count_params


def get_decoder_loaders(config):
    loader_name = getattr(config, 'data_loader', 'data_latents_test')
    if loader_name == 'data_latents_test':
        loader_bundle = get_test_recon_loaders(config)
    else:
        raise ValueError(f'Unsupported data_loader for x,y,yaw CMWM_SSL decoder: {loader_name}')

    if len(loader_bundle) == 3:
        train_loader, test_loader, _ = loader_bundle
    else:
        train_loader, test_loader = loader_bundle
    return train_loader, test_loader


def configure_cuda_linalg():
    backend = os.environ.get("CMWM_SSL_CUDA_LINALG_BACKEND")
    if not backend:
        return
    try:
        torch.backends.cuda.preferred_linalg_library(backend)
        print(f"Using CUDA linalg backend: {torch.backends.cuda.preferred_linalg_library()}")
    except Exception as exc:
        print(f"Warning: failed to set CUDA linalg backend {backend!r}: {exc}")


def infer_encoder_arch_from_state_dict(state_dict):
    """
    Infer minimal encoder architecture info from a ViT checkpoint state_dict.
    Returns dict with keys: dim, depth.
    """
    if 'pos_embedding' not in state_dict:
        raise ValueError("Missing key 'pos_embedding' in encoder state_dict")

    pos_shape = state_dict['pos_embedding'].shape
    if len(pos_shape) != 3:
        raise ValueError(f"Unexpected pos_embedding shape: {pos_shape}")

    inferred_dim = pos_shape[-1]

    layer_indices = []
    layer_pat = re.compile(r'^transformer\.layers\.(\d+)\.')
    for key in state_dict.keys():
        m = layer_pat.match(key)
        if m:
            layer_indices.append(int(m.group(1)))

    if not layer_indices:
        raise ValueError("Cannot infer transformer depth from checkpoint keys")
    inferred_depth = max(layer_indices) + 1

    return {
        'dim': inferred_dim,
        'depth': inferred_depth,
    }


class SSLDecoder(nn.Module):
    """
    Decoder: 将 SSL encoder 输出的 [B, S, 16, 768] 重建为 VAE latent [B, S, 4, 16, 16]
    """
    def __init__(
        self,
        input_dim=768,
        model_dim=768,
        output_dim=4,
        img_size=16,
        p_in=4,
        depth=6,
        heads=None,
        mlp_ratio=4.0,
    ):
        super().__init__()
        # Input: [B, 16, 768] (16 tokens, each 768 dim)
        # Target: [B, 4, 16, 16]

        # ViT Decoder
        if heads is None:
            heads = max(1, model_dim // 64)
        mlp_dim = int(model_dim * mlp_ratio)

        self.decoder = ViT(
            image_size=(img_size, img_size),  # 16x16
            p_in=p_in,     # 16 / 4 = 4 -> 4x4 grid -> 16 tokens
            dim=model_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_in=input_dim,
            dim_out=output_dim,  # 4
            p_out=p_in,    # Output patch size = 4
            patch_in_out=(False, True),  # input is seq, output is image-like
            use_temporal=False
        )

    def forward(self, x):
        """
        Args:
            x: [B, 16, 768] or [B, S, 16, 768]
        Returns:
            [B, 4, 16, 16] or [B, S, 4, 16, 16]
        """
        if x.dim() == 3:
            # [B, 16, 768] -> [B, 1, 16, 768]
            x = x.unsqueeze(1)
            x = self.decoder(x)  # [B, 1, 4, 16, 16]
            return x.squeeze(1)  # [B, 4, 16, 16]
        else:
            # [B, S, 16, 768]
            B, S, N, D = x.shape
            x = x.view(B * S, N, D).unsqueeze(1)  # [B*S, 1, 16, 768]
            x = self.decoder(x)  # [B*S, 1, 4, 16, 16]
            x = x.squeeze(1)  # [B*S, 4, 16, 16]
            return x.view(B, S, 4, 16, 16)


def train_one_epoch(model, teacher_model, optimizer, train_loader, epoch, device):
    model.train()
    teacher_model.eval()

    total_loss = 0
    batch_count = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue

        obs, pos, _ = batch
        obs = obs.to(device, non_blocking=True)  # [B, S, 4, 16, 16]
        pos = pos.to(device, non_blocking=True)

        with torch.no_grad():
            # Get SSL features from teacher
            # obs: [B, S, 4, 16, 16]
            # ssl_latent: [B, S, 16, d_out]
            ssl_latent = teacher_model.forward_teacher(obs, pos)

        # Flatten sequences for frame-wise decoding
        B, S, N, D = ssl_latent.shape
        ssl_input = ssl_latent.reshape(B * S, N, D)
        vae_target = obs.reshape(B * S, 4, 16, 16)

        optimizer.zero_grad()

        # Forward
        pred = model(ssl_input)  # [B*S, 4, 16, 16]

        # Loss
        loss = F.mse_loss(pred, vae_target)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if (batch_idx + 1) % 100 == 0:
            end_time = time.time()
            print(f"[Epoch {epoch} | Batch {batch_idx + 1}] "
                  f"Loss: {loss.item():.6f} "
                  f"(Time: {end_time - start_time:.2f}s)")
            start_time = end_time

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    return avg_loss


def save_model(model, optimizer, scheduler, epoch, loss, loss_history, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'decoder_epoch_{epoch}.pth')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'loss_history': loss_history,
    }, model_path)
    print(f"Model saved to: {model_path}")


@torch.no_grad()
def visualize(model, teacher_model, loader, first_stage_decoder, device, epoch, save_dir):
    model.eval()
    teacher_model.eval()

    if first_stage_decoder:
        first_stage_decoder.eval()

    # Get a batch
    try:
        batch = next(iter(loader))
        while batch is None:
            batch = next(iter(loader))
    except StopIteration:
        return

    obs, pos, _ = batch

    # Pick first sequence
    obs = obs[0:1].to(device)  # [1, S, 4, 16, 16]
    pos = pos[0:1].to(device)

    # Pass through teacher
    ssl_latent = teacher_model.forward_teacher(obs, pos)  # [1, S, 16, d_out]

    # Flatten
    B, S, N, D = ssl_latent.shape
    ssl_input = ssl_latent.reshape(B * S, N, D)
    vae_gt = obs.reshape(B * S, 4, 16, 16)

    # Take a few frames (e.g. first 8)
    num_frames = min(8, S)
    indices = torch.linspace(0, S - 1, num_frames).long().to(device)

    ssl_subset = ssl_input[indices]
    vae_gt_subset = vae_gt[indices]

    vae_pred = model(ssl_subset)

    mse = F.mse_loss(vae_pred, vae_gt_subset).item()
    print(f"Visualization MSE (Latent, frames={num_frames}): {mse:.6f}")

    if first_stage_decoder is None:
        return

    scale = 0.22765929

    imgs_gt = first_stage_decoder.decode(vae_gt_subset / scale)
    imgs_pred = first_stage_decoder.decode(vae_pred / scale)

    imgs_gt = (imgs_gt + 1) / 2
    imgs_pred = (imgs_pred + 1) / 2
    imgs_gt = torch.clamp(imgs_gt, 0, 1)
    imgs_pred = torch.clamp(imgs_pred, 0, 1)

    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 2, 4))
    if num_frames == 1:
        axes = axes[:, None]

    for i in range(num_frames):
        # GT
        axes[0, i].imshow(imgs_gt[i].cpu().permute(1, 2, 0).numpy())
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("GT")

        # Pred
        axes[1, i].imshow(imgs_pred[i].cpu().permute(1, 2, 0).numpy())
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Pred")

    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'vis_epoch_{epoch}.png'))
    plt.close()
    print(f"Visualization saved to {save_dir}/vis_epoch_{epoch}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='CMWM_SSL/config/cmwm_L.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--resume', type=str, default=None, help='Path to decoder checkpoint to resume from')
    parser.add_argument('--encoder_ckpt', type=str,
                        default='/home/xiangyuanpeng/BrainVQ/CMWM_SSL/results/models/cmwm_L_right/ssl_cmwm_L_scand_seqlen128_depth10_dim768_mse_80.pth',
                        help='Path to trained SSL encoder checkpoint')
    parser.add_argument('--decoder_depth', type=int, default=None, help='Decoder ViT depth')
    parser.add_argument('--decoder_dim', type=int, default=None, help='Decoder hidden dim (default follows encoder dim)')
    parser.add_argument('--decoder_heads', type=int, default=None, help='Decoder attention heads (default decoder_dim//64)')
    parser.add_argument('--decoder_mlp_ratio', type=float, default=4.0, help='Decoder MLP ratio')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for AdamW')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Scheduler minimum learning rate')
    parser.add_argument('--reset_optimizer', action='store_true',
                        help='When resuming, do not load optimizer/scheduler states and start a fresh optimization schedule')
    parser.add_argument('--frame_step', type=int, default=4, help='frame_step for data_latents_test dataloader')
    parser.add_argument('--results_subdir', type=str, default='decoder_right', help='Subdir under CMWM_SSL/results to save decoder checkpoints')
    parser.add_argument('--save_interval', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--encoder_dim', type=int, default=None, help='Override SSL encoder dim for loading encoder_ckpt')
    parser.add_argument('--encoder_depth', type=int, default=None, help='Override SSL encoder depth for loading encoder_ckpt')
    parser.add_argument('--encoder_d_out', type=int, default=None, help='Override SSL encoder output dim (default follows inferred encoder dim)')
    args = parser.parse_args()

    configure_cuda_linalg()

    # Load config
    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = OmegaConf.load(config_path)

    print(f"Loading config from {config_path}")

    # Device
    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else config.device)
    print(f"Device: {device}")

    # Results directories
    results_dir = os.path.join(PROJECT_ROOT, f'CMWM_SSL/results/{args.results_subdir}')
    img_dir = os.path.join(results_dir, 'images')

    # Load Data
    if not hasattr(config, 'frame_step'):
        config.frame_step = args.frame_step
    train_loader, test_loader = get_decoder_loaders(config)

    if not os.path.isfile(args.encoder_ckpt):
        print(f"Error: Teacher checkpoint not found at {args.encoder_ckpt}")
        return

    checkpoint = torch.load(args.encoder_ckpt, map_location="cpu")
    ckpt_state = None
    if 'teacher_state_dict' in checkpoint:
        ckpt_state = checkpoint['teacher_state_dict']
    elif 'model_state_dict' in checkpoint:
        ckpt_state = checkpoint['model_state_dict']
    else:
        print("Warning: Could not find state dict in checkpoint")
        return

    inferred_arch = infer_encoder_arch_from_state_dict(ckpt_state)
    encoder_dim = args.encoder_dim if args.encoder_dim is not None else inferred_arch['dim']
    encoder_depth = args.encoder_depth if args.encoder_depth is not None else inferred_arch['depth']
    encoder_d_out = args.encoder_d_out if args.encoder_d_out is not None else encoder_dim

    # Build Encoder config (can differ from decoder config)
    encoder_config = {
        'd_out': encoder_d_out,
        'beta': getattr(config.model.params, 'beta', 1.0),
        'img_height': config.model.params.img_height,
        'img_width': config.model.params.img_width,
        'dim': encoder_dim,
        'depth': encoder_depth,
        'p_in_encoder': config.model.params.p_in_encoder,
        'random_group': config.model.params.random_group,
        'active_ratio': config.model.params.active_ratio,
        'geometry_type': 'se2_yaw',
        'pose_dim': 3,
        'device': device
    }

    # Load Teacher SSL Model
    print("Loading SSL Teacher model...")
    print(f"Encoder arch -> dim: {encoder_dim}, depth: {encoder_depth}, d_out: {encoder_d_out}")
    teacher_model = SSL_FSQ_Learner_NoPredictor(encoder_config).to(device)

    if 'teacher_state_dict' in checkpoint:
        teacher_model.teacher_encoder.load_state_dict(checkpoint['teacher_state_dict'])
        teacher_model.student_encoder.load_state_dict(checkpoint['student_state_dict'])
        print(f"Loaded SSL checkpoint from: {args.encoder_ckpt}")
    elif 'model_state_dict' in checkpoint:
        teacher_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded SSL checkpoint (model_state_dict) from: {args.encoder_ckpt}")
    else:
        print("Warning: Could not find state dict in checkpoint")
        return

    # Freeze encoder
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Decoder Model
    print("Creating Decoder...")
    decoder_dim = args.decoder_dim if args.decoder_dim is not None else config.model.params.dim
    decoder_depth = args.decoder_depth if args.decoder_depth is not None else config.model.params.depth
    model = SSLDecoder(
        input_dim=encoder_d_out,  # can differ from decoder dim (e.g., 384 -> 768)
        model_dim=decoder_dim,
        output_dim=4,
        img_size=config.model.params.img_height,  # 16
        p_in=config.model.params.p_in_encoder,  # 4
        depth=decoder_depth,
        heads=args.decoder_heads,
        mlp_ratio=args.decoder_mlp_ratio,
    ).to(device)
    print(f"Decoder arch -> input_dim: {encoder_d_out}, model_dim: {decoder_dim}, depth: {decoder_depth}")
    count_params(model)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Resume
    start_epoch = 1
    loss_history = []

    resume_ckpt = None
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading decoder checkpoint from {args.resume}")
            resume_ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(resume_ckpt['model_state_dict'])
            if not args.reset_optimizer and 'optimizer_state_dict' in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            loss_history = resume_ckpt.get('loss_history', [])
            start_epoch = resume_ckpt['epoch'] + 1
            if args.reset_optimizer:
                print(f"Resuming model weights from epoch {resume_ckpt['epoch']} with fresh optimizer/scheduler")
            else:
                print(f"Resuming full state from epoch {start_epoch}")
        else:
            print(f"Warning: Checkpoint {args.resume} not found, starting from scratch")

    # Scheduler
    remaining_epochs = max(1, args.epochs - start_epoch + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs if args.reset_optimizer else args.epochs,
        eta_min=args.min_lr,
        last_epoch=-1 if args.reset_optimizer else (start_epoch - 2 if start_epoch > 1 else -1)
    )
    if resume_ckpt is not None and (not args.reset_optimizer):
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])

    # Load VAE Decoder for visualization
    print("Loading VAE for visualization...")
    first_stage_decoder = None
    try:
        first_stage_model = load_first_stage_model(
            config.first_stage_config_path,
            config.first_stage_ckpt_path,
            device
        )
        first_stage_decoder = first_stage_model
        first_stage_decoder.eval()
        print("VAE loaded.")
    except Exception as e:
        print(f"Failed to load VAE: {e}")

    print(f"\nStarting training...")
    print(f"Models will be saved to: {results_dir}")
    print(f"Visualizations will be saved to: {img_dir}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n==== Epoch {epoch} ====")

        # Regenerate sequences for variability
        if hasattr(train_loader.dataset, 'regenerate_sequences'):
            train_loader.dataset.regenerate_sequences(seed=getattr(config, "seed", 3407) + epoch)
        elif hasattr(train_loader.dataset, 'generate_sequences'):
            train_loader.dataset.generate_sequences(getattr(train_loader.dataset, 'seqlen', None))

        # Train
        loss = train_one_epoch(model, teacher_model, optimizer, train_loader, epoch, device)
        loss_history.append(loss)
        print(f"Epoch {epoch} Avg Loss: {loss:.6f}")

        # Step scheduler
        scheduler.step()

        # Save and visualize
        if epoch % args.save_interval == 0 or epoch == 1:
            save_model(model, optimizer, scheduler, epoch, loss, loss_history, results_dir)
            visualize(model, teacher_model, test_loader, first_stage_decoder, device, epoch, img_dir)

    print("\nTraining completed!")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Decoder Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_curve.png'))
    plt.close()
    print(f"Loss curve saved to {results_dir}/loss_curve.png")


if __name__ == "__main__":
    main()
