# File purpose: CMWM_SSL single-process student/teacher SSL-FSQ training entry.
import os
import sys
import time
import argparse
import torch
from omegaconf import OmegaConf
import math
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from data_load.data_latent_random import get_recon_loaders as get_random_recon_loaders
from data_load.data_latents_test import get_recon_loaders as get_test_recon_loaders
from CMWM_SSL.models.SSL_fsq import SSL_FSQ_Learner_NoPredictor
from util import count_params


def get_train_loaders(config):
    loader_name = getattr(config, 'data_loader', 'data_latent_random')

    if loader_name == 'data_latents_test':
        loader_bundle = get_test_recon_loaders(config)
    elif loader_name == 'data_latent_random':
        loader_bundle = get_random_recon_loaders(config)
    else:
        raise ValueError(f'Unsupported data_loader for x,y,yaw CMWM_SSL: {loader_name}')

    if len(loader_bundle) == 3:
        train_loader, test_loader, _ = loader_bundle
    else:
        train_loader, test_loader = loader_bundle

    return train_loader, test_loader


def refresh_train_sequences(train_loader):
    dataset = train_loader.dataset

    if hasattr(dataset, 'regenerate_sequences'):
        dataset.regenerate_sequences()
        return

    if hasattr(dataset, 'generate_sequences'):
        dataset.generate_sequences(getattr(dataset, 'seqlen', None))
        return

    print('Warning: dataset does not support sequence refresh, reusing current sequences.')

def train_one_epoch(model, optimizer, train_loader, epoch, device, max_steps=None):
    model.train()
    total_loss = 0
    batch_count = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        if max_steps is not None and batch_idx >= max_steps:
            break
        if batch is None:
            continue

        obs, pos, _ = batch
        obs = obs.to(device)
        pos = pos.to(device) # Teacher 需要 pos (label)

        optimizer.zero_grad()

        # 1. Student Forward (注意：Student 内部忽略 pos)
        # Student 试图预测 Teacher 经过旋转处理后的结果
        student_pred = model.forward_student(obs, pos)

        # 2. Teacher Forward (Teacher 使用 pos 进行 FSQ 旋转)
        with torch.no_grad():
            teacher_target = model.forward_teacher(obs, pos)

        # 3. Segment-wise SSL commitment loss
        loss_dict = model.loss_function(student_pred, teacher_target)
        loss = loss_dict['loss']

        loss.backward()
        optimizer.step()

        # 4. EMA Update
        model.update_moving_average()

        total_loss += loss.item()
        batch_count += 1

        if (batch_idx + 1) % 100 == 0:
            end_time = time.time()
            print(f"[Epoch {epoch} | Batch {batch_idx + 1}] "
                  f"Loss: {loss.item():.6f} "
                  f"(Time: {end_time - start_time:.2f}s)")
            start_time = end_time

    if batch_count == 0:
        raise RuntimeError('No valid batches were produced by the dataloader.')

    avg_loss = total_loss / batch_count
    return avg_loss

def save_model(model, optimizer, scheduler, config, epoch, loss, loss_history):
    save_dir = os.path.join(PROJECT_ROOT, config.model_dir)
    os.makedirs(save_dir, exist_ok=True)

    # 保存 Student 和 Teacher
    model_path = os.path.join(save_dir, f'ssl_{config.model_name}_{epoch}.pth')

    torch.save({
        'epoch': epoch,
        'student_state_dict': model.student_encoder.state_dict(),
        'teacher_state_dict': model.teacher_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'loss_history': loss_history
    }, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='CMWM_SSL/config/cmwm_L.yaml')
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--resume_epoch', type=int, default=None, help='Epoch to resume training from')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Optional maximum batches per epoch for smoke tests')
    parser.add_argument('--debug', action='store_true',
                       help='Run a short smoke-test style training loop')
    args = parser.parse_args()

    config_path = os.path.join(PROJECT_ROOT, args.config)
    config = OmegaConf.load(config_path)

    device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else config.device)
    print(f"Device: {device}")

    # Load Data
    train_loader, test_loader = get_train_loaders(config)

    # Config
    encoder_config = {
        'd_out': config.model.params.d_out,
        'beta': getattr(config.model.params, 'beta', 1.0),
        'img_height': config.model.params.img_height,
        'img_width': config.model.params.img_width,
        'dim': config.model.params.dim,
        'depth': config.model.params.depth,
        'p_in_encoder': config.model.params.p_in_encoder,
        'random_group': config.model.params.random_group,
        'active_ratio': config.model.params.active_ratio,
        'geometry_type': 'se2_yaw',
        'pose_dim': 3,
        'device': device
    }

    # Model
    print("Creating Asymmetric SSL-FSQ Model...")
    model = SSL_FSQ_Learner_NoPredictor(encoder_config).to(device)
    count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)

    start_epoch = 1
    loss_history = []

    if args.resume_epoch is not None:
        save_dir = os.path.join(PROJECT_ROOT, config.model_dir)
        model_path = os.path.join(save_dir, f'ssl_{config.model_name}_{args.resume_epoch}.pth')
        if os.path.exists(model_path):
            print(f"Loading checkpoint from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            model.student_encoder.load_state_dict(checkpoint['student_state_dict'])
            model.teacher_encoder.load_state_dict(checkpoint['teacher_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                # If scheduler state is missing, fast-forward it
                print("Scheduler state not found")
                #for _ in range(args.resume_epoch):
                #    scheduler.step()

            if 'loss_history' in checkpoint:
                loss_history = checkpoint['loss_history']

            start_epoch = args.resume_epoch + 1
        else:
            print(f"Checkpoint {model_path} not found! Starting from scratch.")

    for epoch in range(start_epoch, config.num_epochs + 1):
        current_m = 1.0 - (1.0 - 0.996) * (math.cos(math.pi * epoch / config.num_epochs) + 1) / 2
        model.teacher_momentum = current_m
        print(f"\n==== Epoch {epoch} ====")
        refresh_train_sequences(train_loader)
        loss = train_one_epoch(
            model, optimizer, train_loader, epoch, device,
            max_steps=args.max_steps if args.max_steps is not None else (20 if args.debug else None)
        )
        loss_history.append(loss)
        print(f"Epoch {epoch} Avg Loss: {loss:.6f}")
        scheduler.step()
        if epoch % config.save_interval == 0:
            save_model(model, optimizer, scheduler, config, epoch, loss, loss_history)
        if args.debug:
            break
