import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from dataset import seis_dataset   
from model import UNet
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import json

# -----------------------------
# 可视化与保存工具
# -----------------------------
def _to_numpy(img_tensor):
    # 兼容处理：如果还在GPU上先转CPU
    return img_tensor.detach().squeeze().float().cpu().numpy()

def save_sample_fig(noisy, vpred, clean, epoch, save_dir="./checkpoints/samples"):
    os.makedirs(save_dir, exist_ok=True)
    noisy_img = _to_numpy(noisy)
    vpred_img = _to_numpy(vpred)
    clean_img = _to_numpy(clean)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # 这里的 Noisy 实际上是 Masked 输入
    axes[0].imshow(noisy_img, cmap='gray');  axes[0].set_title("Masked Input"); axes[0].axis("off")
    axes[1].imshow(vpred_img, cmap='gray');  axes[1].set_title(f"Restored (Epoch {epoch})"); axes[1].axis("off")
    axes[2].imshow(clean_img, cmap='gray');  axes[2].set_title("Clean Ground Truth"); axes[2].axis("off")
    path = os.path.join(save_dir, f"epoch_{epoch:03d}_sample.png")
    plt.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  -> 可视化保存: {path}")

# -----------------------------
# 训练主循环
# -----------------------------
def train(model, train_loader, val_loader, optimizer, device, epochs=30, checkpoint_dir="./checkpoints"):
    model.train()
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/vis", exist_ok=True)
    os.makedirs(f"{checkpoint_dir}/samples", exist_ok=True)

    log_path = os.path.join(checkpoint_dir, "loss_lr.log")

    for epoch in range(epochs):
        # ----------- 训练阶段 -----------
        model.train()
        total_train_loss = 0.0

        # dataset 返回的是 (masked, clean)，且已经在 GPU 上
        for x_cond, x0 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
            # 注意：dataset.py 已经将数据放到了 device 上，这里可以直接用
            # 但为了保险起见（防止 device 字符串不一致），保留 .to(device) 是安全的
            x0 = x0.to(device)
            x_cond = x_cond.to(device)
            B = x0.size(0)

            t_model = torch.rand(B, device=device)          # [B]
            t_cf = t_model.unsqueeze(1)                     # [B,1]

            # 计算 flow matching 目标
            x_t, v_star, _ = compute_xt_and_velocity(x0, t_cf)
            
            # 模型输入：当前噪声状态 x_t, 时间 t, 条件 x_cond (masked image)
            v_pred = model(x_t, t_model, x_cond)

            loss = F.mse_loss(v_pred, v_star)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0: # 每5个epoch保存一次权重
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"  -> 已保存定期检查点: {checkpoint_path}")

        # ----------- 验证阶段 -----------
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_cond, val_clean in val_loader:
                val_clean = val_clean.to(device)
                val_cond = val_cond.to(device)
                B_val = val_clean.size(0)

                t_model_val = torch.rand(B_val, device=device)
                t_cf_val = t_model_val.unsqueeze(1)

                x_t_val, v_star_val, _ = compute_xt_and_velocity(val_clean, t_cf_val)
                v_pred_val = model(x_t_val, t_model_val, val_cond)

                val_loss = F.mse_loss(v_pred_val, v_star_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # ----------- 保存日志 -----------
        log_message = (f"Epoch {epoch+1}/{epochs} -> "
                        f"Train Loss: {avg_train_loss:.6f} | "
                        f"Val Loss: {avg_val_loss:.6f} | "
                        f"LR: {current_lr:.6e}\n")
        print("\n" + log_message.strip())
        with open(log_path, "a") as log_file:
            log_file.write(log_message)

        # ----------- 可视化矢量场 -----------
        # 使用最后一个 batch 的数据进行矢量场可视化
        visualize_vector_field(v_star, v_pred, epoch + 1, save_dir=f"{checkpoint_dir}/vis")

        # ----------- 验证采样可视化 (去噪/插值过程) -----------
        try:
            # 获取一个验证集样本
            sample_masked, sample_clean = next(iter(val_loader))
            sample_masked = sample_masked[0:1].to(device)
            sample_clean = sample_clean[0:1].to(device)

            # 执行采样：输入是 Masked 图像作为 Condition
            denoised_image = denoise_sample(model, sample_masked, steps=50, device=device, save_velocity=False)

            save_sample_fig(
                noisy=sample_masked,
                vpred=denoised_image,
                clean=sample_clean,
                epoch=epoch+1,
                save_dir=f"{checkpoint_dir}/samples"
            )
        except StopIteration:
            pass

# -----------------------------
# main
# -----------------------------
def main(args):
    DATA_PATH = args.data_path
    IMAGE_SHAPE = (256, 256)
    
    # 1. 确定设备 (必须在 Dataset 初始化前确定，因为 Dataset 内部需要它)
    device_str = f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using Device: {device}")

    # 2. 数据集切分 (适配新的 utils.split_dataset，返回两个 clean 路径列表)
    train_clean_paths, val_clean_paths = split_dataset(DATA_PATH, data_shape=IMAGE_SHAPE)
    print(f"Train files: {len(train_clean_paths)}, Val files: {len(val_clean_paths)}")

    # 3. 实例化数据集 (适配新的 seis_dataset)
    # 注意：传入 device，且只传入 clean_files
    train_dataset = seis_dataset(clean_files=train_clean_paths, data_shape=IMAGE_SHAPE,
                                 mask_ratio_range=args.mask_ratio_range, mask_mode=args.mask_mode)
    val_dataset   = seis_dataset(clean_files=val_clean_paths,   data_shape=IMAGE_SHAPE,
                                 mask_ratio_range=args.mask_ratio_range, mask_mode=args.mask_mode)

    # 4. DataLoader 设置 (关键修改)
    # 由于 dataset.__getitem__ 返回的是 GPU 上的 Tensor，
    # num_workers 必须设置为 0，否则多进程无法处理 CUDA Tensor。
    # pin_memory 设为 False，因为数据已经在 GPU 上了。
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=8, pin_memory=False)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                                               num_workers=8, pin_memory=False)

    # 5. 模型初始化
    model = UNet(
                dim=64,            
                image_size=IMAGE_SHAPE[0],
                channels=1,
                out_dim=1,
                self_attention=False
            ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    CHECKPOINTS_PATH = "./checkpoints"
    train(model, train_loader, val_loader, optimizer, device, epochs=args.epochs, checkpoint_dir=CHECKPOINTS_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train conditional flow matching interpolation")
    parser.add_argument('--device_id', type=int, default=2, help='CUDA device ID')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--data_path', type=str, default='/home/data/gtx/Geo_data/5D_interpolation/251031/patches/label_256', help='Number of training epochs')

    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    # 新增掩码比例参数
    parser.add_argument('--mask_ratio_range', type=float, nargs=2, default=(0.3, 0.7),
                        metavar=('LOW', 'HIGH'), help='Ratio of columns to mask (drop)')
    parser.add_argument('--mask_mode', type=str, default='random',
                        choices=['random', 'uniform', 'large_gap'],
                        help='Masking pattern: random, uniform, or large_gap (contiguous block)')

    args = parser.parse_args()
    main(args)