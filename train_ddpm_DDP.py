"""
DDPM + DDIM training (DDP) for seismic interpolation.

Conditioning: time t + masked image x_cond (no explicit mask channel).
Model learns to predict noise ε(x_t, t, x_cond) added during forward diffusion.

Usage:
    python train_ddpm_DDP.py --batch_size 4 --epochs 200 --data_path /path/to/patches/label_256

    python train_ddpm_DDP.py --resume ./results/260521_DDPM/checkpoints/checkpoint_epoch_10.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from model import UNet
from dataset import seis_dataset
from diffusion_utils import (cosine_beta_schedule, compute_alphas, q_sample,
                              ddim_sample)
from utils import split_dataset, seismic
import os
import argparse
import datetime
import signal
import atexit
import sys
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

DATE = "260521_DDPM_384_192"
TIMESTEPS = 1000          # total diffusion steps
SAMPLING_STEPS = 50      # DDIM sampling steps (must divide TIMESTEPS evenly-ish)


def Args():
    parser = argparse.ArgumentParser(description="Train DDPM + DDIM (DDP)")

    parser.add_argument('--data_path', type=str, default="/home/ShareData/gtx/5Ddagang/label_384_192/")
    parser.add_argument('--dim', type=int, default=64, help='Base dimensionality of the UNet model')
    parser.add_argument('--img_size', type=int, nargs=2, default=[384, 192])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mask_ratio_range', type=float, nargs=2, default=(0.3, 0.7),
                        metavar=('LOW', 'HIGH'))
    parser.add_argument('--mask_mode', type=str, default='random',
                        choices=['random', 'uniform', 'large_gap'])

    parser.add_argument('--output_dir_png', type=str, default=f"./results/{DATE}/generated_pngs")
    parser.add_argument('--output_dir_npy', type=str, default=f"./results/{DATE}/generated_npys")
    parser.add_argument('--model_path', type=str, default=f"./results/{DATE}/checkpoints/")

    parser.add_argument('--resume', type=str, default=None, help="Path to resume checkpoint")
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision (AMP) training')

    parser.add_argument('--dist_backend', default='nccl', type=str)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)

    return parser.parse_args()


def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    dist.init_process_group(
        backend=args.dist_backend, rank=rank, world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    torch.cuda.set_device(rank)


def cleanup():
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def train(rank, world_size, args):
    setup(rank, world_size, args)
    is_master = (rank == 0)

    # ── signal handling ──
    stop_requested = False

    def sig_handler(signum, frame):
        nonlocal stop_requested
        if stop_requested:
            if is_master:
                print(f"\n[Rank {rank}] Force exit.")
            cleanup()
            sys.exit(1)
        stop_requested = True
        if is_master:
            print(f"\n[Rank {rank}] Received signal {signum}. Finishing current batch... "
                  "(Ctrl+C again to force exit)")

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    atexit.register(cleanup)

    device = torch.device(f"cuda:{rank}")
    train_loader = None
    val_loader = None
    log_file = None

    try:
        # ── noise schedule ──
        betas = cosine_beta_schedule(TIMESTEPS).to(device)
        sched = compute_alphas(betas)
        alphas_cumprod = sched["alphas_cumprod"]  # (T,), on device

        if is_master:
            os.makedirs(args.output_dir_png, exist_ok=True)
            os.makedirs(args.output_dir_npy, exist_ok=True)
            os.makedirs(args.model_path, exist_ok=True)
            os.makedirs(os.path.join(args.model_path, "vis"), exist_ok=True)
            log_path = os.path.join(args.model_path, "training.log")
            log_file = open(log_path, "a")

        # ── data ──
        train_clean_paths, val_clean_paths = split_dataset(args.data_path, train_ratio=0.9)

        train_dataset = seis_dataset(clean_files=train_clean_paths, data_shape=args.img_size,
                                     mask_ratio_range=args.mask_ratio_range,
                                     mask_mode=args.mask_mode)
        val_dataset   = seis_dataset(clean_files=val_clean_paths,   data_shape=args.img_size,
                                     mask_ratio_range=args.mask_ratio_range,
                                     mask_mode=args.mask_mode)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=True)

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=8, pin_memory=True, persistent_workers=False, prefetch_factor=4,
            pin_memory_device=f'cuda:{rank}' if torch.cuda.is_available() else None,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler,
            num_workers=8, pin_memory=True, persistent_workers=False, prefetch_factor=4,
            pin_memory_device=f'cuda:{rank}' if torch.cuda.is_available() else None,
        )

        # ── model (cond_channels=1: only masked image, no mask) ──
        model = UNet(
            dim=args.dim, image_size=args.img_size[0],
            channels=1, out_dim=1, cond_channels=1,
            self_attention=False,
        ).to(device)

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[rank], output_device=rank)

        if is_master:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Diffusion timesteps: {TIMESTEPS}, DDIM sampling steps: {SAMPLING_STEPS}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = GradScaler() if args.use_amp else None

        # ── resume ──
        start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=f"cuda:{rank}")
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if args.use_amp and scaler is not None:
                scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
            start_epoch = checkpoint['epoch'] + 1
            if is_master:
                print(f"Resuming from epoch {start_epoch}")

        # ── training loop ──
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)
            model.train()
            local_batch_loss_sum = 0

            iterator = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}",
                            disable=not is_master)

            for x_cond, x0, mask in iterator:
                # Dataset returns (masked, clean, mask). For DDPM we drop mask, use x_cond only.
                x0 = x0.to(device, non_blocking=True)
                x_cond = x_cond.to(device, non_blocking=True)
                B = x0.size(0)

                optimizer.zero_grad()

                # Sample random timestep per batch element
                t = torch.randint(0, TIMESTEPS, (B,), device=device)
                t_norm = t.float() / TIMESTEPS  # [0, 1) for sinusoidal embedding

                # Forward diffusion: add noise to CLEAN image
                x_t, noise = q_sample(x0, t, alphas_cumprod)

                if args.use_amp:
                    with autocast():
                        noise_pred = model(x_t, t_norm, x_cond)
                        loss = F.mse_loss(noise_pred, noise)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    noise_pred = model(x_t, t_norm, x_cond)
                    loss = F.mse_loss(noise_pred, noise)
                    loss.backward()
                    optimizer.step()

                local_batch_loss_sum += loss.item()

                if is_master:
                    iterator.set_postfix({'loss': loss.item()})

                if stop_requested:
                    break

            scheduler.step()

            if stop_requested:
                if is_master:
                    print("Interrupted. Saving checkpoint before exit...")
                break

            # All-reduce training loss
            total_loss_tensor = torch.tensor(local_batch_loss_sum, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = total_loss_tensor.item() / (len(train_loader) * world_size)
            current_lr = scheduler.get_last_lr()[0]

            if is_master:
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.6e}")

            # ── validation ──
            model.eval()
            local_val_loss_sum = 0

            with torch.no_grad():
                for val_cond, val_clean, val_mask in val_loader:
                    val_clean = val_clean.to(device, non_blocking=True)
                    val_cond = val_cond.to(device, non_blocking=True)
                    B_val = val_clean.size(0)

                    t_val = torch.randint(0, TIMESTEPS, (B_val,), device=device)
                    t_val_norm = t_val.float() / TIMESTEPS

                    x_t_val, noise_val = q_sample(val_clean, t_val, alphas_cumprod)

                    if args.use_amp:
                        with autocast():
                            noise_pred_val = model(x_t_val, t_val_norm, val_cond)
                            loss_val = F.mse_loss(noise_pred_val, noise_val)
                    else:
                        noise_pred_val = model(x_t_val, t_val_norm, val_cond)
                        loss_val = F.mse_loss(noise_pred_val, noise_val)

                    local_val_loss_sum += loss_val.item()

                    if stop_requested:
                        break

            if stop_requested:
                break

            val_loss_tensor = torch.tensor(local_val_loss_sum, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)

            if is_master:
                print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}")
                log_file.write(f"Epoch {epoch+1}/{args.epochs} | "
                               f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
                               f"LR: {current_lr:.6e}\n")
                log_file.flush()

                # ── save checkpoint + DDIM sample ──
                if (epoch + 1) % 10 == 0:
                    checkpoint_path = os.path.join(args.model_path, f"checkpoint_epoch_{epoch+1}.pth")
                    checkpoint_data = {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'alphas_cumprod': alphas_cumprod.cpu(),
                    }
                    if args.use_amp and scaler is not None:
                        checkpoint_data['scaler_state_dict'] = scaler.state_dict()
                    torch.save(checkpoint_data, checkpoint_path)

                    # DDIM sampling visualization
                    try:
                        temp_iter = iter(val_loader)
                        sample_cond, sample_clean, _ = next(temp_iter)
                        sample_cond = sample_cond[:1].to(device)
                        sample_clean = sample_clean[:1].to(device)

                        denoised = ddim_sample(
                            model, sample_cond,
                            TIMESTEPS, SAMPLING_STEPS, alphas_cumprod, device,
                        )

                        pred_img = denoised[0].cpu().numpy().squeeze()
                        gt_img = sample_clean[0].cpu().numpy().squeeze()
                        masked_img = sample_cond[0].cpu().numpy().squeeze()

                        # Symmetric vmin/vmax from all three images (percentile-based)
                        all_vals = np.concatenate([masked_img.ravel(),
                                                   pred_img.ravel(),
                                                   gt_img.ravel()])
                        vmax = max(abs(np.percentile(all_vals, 1)),
                                   abs(np.percentile(all_vals, 99)))
                        vmin = -vmax

                        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                        im0 = axes[0].imshow(masked_img, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
                        axes[0].set_title('Masked (condition)')
                        im1 = axes[1].imshow(pred_img, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
                        axes[1].set_title(f'DDIM Restored E{epoch+1}')
                        im2 = axes[2].imshow(gt_img, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
                        axes[2].set_title('Ground Truth')
                        for ax in axes: ax.axis('off')

                        # Shared colorbar
                        fig.subplots_adjust(right=0.92)
                        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
                        fig.colorbar(im2, cax=cbar_ax)

                        plt.savefig(os.path.join(args.output_dir_png, f"epoch{epoch+1}_sample.png"),
                                    bbox_inches='tight')
                        plt.close(fig)

                        np.save(os.path.join(args.output_dir_npy, f"epoch{epoch+1}_pred.npy"), pred_img)
                        print(f"  -> Saved checkpoint and DDIM sample.")
                    except Exception as e:
                        print(f"Visualization error: {e}")

        if is_master:
            torch.save(model.module.state_dict(), f"{args.model_path}/ddpm_model_final.pth")
            log_file.write("Training finished.\n")
            log_file.close()
            print("Training finished.")

    finally:
        try:
            del train_loader
            del val_loader
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass
        cleanup()


if __name__ == "__main__":
    args = Args()

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ngpus_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    else:
        ngpus_per_node = torch.cuda.device_count()

    print(f"Found {ngpus_per_node} GPUs for training.")
    args.world_size = ngpus_per_node

    try:
        mp.spawn(train, args=(args.world_size, args),
                 nprocs=args.world_size, join=True)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C. Waiting for child processes to release GPU resources "
              "(max 60s timeout)...")
