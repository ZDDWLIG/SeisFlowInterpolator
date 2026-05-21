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
from utils import * # 确保这里包含 compute_xt_and_velocity, denoise_sample, split_dataset
import os
import argparse
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler 
import warnings
warnings.filterwarnings("ignore")

# 根据你的服务器情况修改 GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" 

DATE = "260520_FM_128" # 修改标识符以区分

def Args():
    parser = argparse.ArgumentParser(description="Train Conditional Flow Matching (DDP)")

    parser.add_argument('--data_path', type=str, default="/home/data/gtx/Geo_data/5D_interpolation/251031/patches/label_256")
    parser.add_argument('--dim', type=int, default=128, help='Base dimensionality of the UNet model')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256])
    parser.add_argument('--batch_size', type=int, default=4) # 单卡 batch size
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--mask_ratio_range', type=float, nargs=2, default=(0.3, 0.7),
                        metavar=('LOW', 'HIGH'))
    parser.add_argument('--mask_mode', type=str, default='random',
                        choices=['random', 'uniform', 'large_gap'],
                        help='Masking pattern')

    # 输出路径配置
    parser.add_argument('--output_dir_png', type=str, default=f"./results/{DATE}/generated_pngs")
    parser.add_argument('--output_dir_npy', type=str, default=f"./results/{DATE}/generated_npys")
    parser.add_argument('--model_path', type=str, default=f"./results/{DATE}/checkpoints/")

    parser.add_argument('--resume', type=str, default=None, help="Path to resume checkpoint")

    # DDP 参数
    parser.add_argument('--dist_backend', default='nccl', type=str)
    # 下面这几个参数由 mp.spawn 自动处理，通常不需要手动传
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)

    return parser.parse_args()

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357' # 防止端口冲突，稍微改一下
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    dist.init_process_group(backend=args.dist_backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, args):
    setup(rank, world_size, args)
    is_master = (rank == 0)
    
    try:
        device = torch.device(f"cuda:{rank}")

        if is_master:
            os.makedirs(args.output_dir_png, exist_ok=True)
            os.makedirs(args.output_dir_npy, exist_ok=True)
            os.makedirs(args.model_path, exist_ok=True)
            os.makedirs(os.path.join(args.model_path, "vis"), exist_ok=True)
            log_path = os.path.join(args.model_path, "training.log")
            log_file = open(log_path, "a")
        else:
            log_file = None

        # 1. 数据集准备
        # 注意：split_dataset 需要根据你的 utils 实现调整返回值
        train_clean_paths, val_clean_paths = split_dataset(args.data_path,train_ratio=0.9)
        
        # ⚠️ 重要提示：为了使用 num_workers > 0，dataset.__getitem__ 必须返回 CPU Tensor
        # 如果你的 Dataset 还是返回 GPU Tensor，请将 num_workers 改为 0
        train_dataset = seis_dataset(clean_files=train_clean_paths, data_shape=args.img_size,
                                     mask_ratio_range=args.mask_ratio_range,
                                     mask_mode=args.mask_mode)
        val_dataset   = seis_dataset(clean_files=val_clean_paths,   data_shape=args.img_size,
                                     mask_ratio_range=args.mask_ratio_range,
                                     mask_mode=args.mask_mode)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=8,           # ⚠️ 如果报错 pickle error，请改为 0
            pin_memory=True,         # ⚠️ 如果 Dataset 返回 GPU 数据，请改为 False
            persistent_workers=True, # ⚠️ 如果 num_workers=0，请改为 False
            prefetch_factor=4,       # ⚠️ 如果 num_workers=0，请删除此参数
            pin_memory_device=f'cuda:{rank}' if torch.cuda.is_available() else None
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory_device=f'cuda:{rank}' if torch.cuda.is_available() else None
        )

        # 2. 模型初始化
        model = UNet(
            dim=args.dim, 
            image_size=args.img_size[0], 
            channels=1, 
            out_dim=1,
            self_attention=False
        ).to(device)

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # 推荐用于 DDP
        model = DDP(model, device_ids=[rank], output_device=rank)

        if is_master:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Trainable parameters: {trainable_params:,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        
        # 混合精度训练Scaler
        scaler = GradScaler() 

        # 3. 断点续训逻辑
        start_epoch = 0
        if args.resume and os.path.isfile(args.resume):
            # map_location 确保加载到当前进程的 GPU
            checkpoint = torch.load(args.resume, map_location=f"cuda:{rank}") 
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint.get('scaler_state_dict', {}))
            start_epoch = checkpoint['epoch'] + 1
            if is_master:
                print(f"Resuming from epoch {start_epoch}")

        # 4. 训练循环
        for epoch in range(start_epoch, args.epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            local_batch_loss_sum = 0 
            
            # 只有 master 进程显示进度条
            iterator = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{args.epochs}", disable=not is_master)

            for x_cond, x0 in iterator:
                # 数据搬运到 GPU
                x0 = x0.to(device, non_blocking=True)
                x_cond = x_cond.to(device, non_blocking=True)
                B = x0.size(0)

                optimizer.zero_grad()

                # --- Flow Matching 核心逻辑 ---
                # 生成随机时间 t
                t_model = torch.rand(B, device=device)
                t_cf = t_model.unsqueeze(1)

                # 计算加噪后的状态 x_t 和目标速度 v_star
                x_t, v_star, _ = compute_xt_and_velocity(x0, t_cf)

                with autocast():
                    # 模型预测速度 v_pred
                    v_pred = model(x_t, t_model, x_cond)
                    loss = F.mse_loss(v_pred, v_star)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                local_batch_loss_sum += loss.item()
                
                if is_master:
                    iterator.set_postfix({'loss': loss.item()})

            scheduler.step()
            
            # 汇总多卡 Loss 用于日志
            total_loss_tensor = torch.tensor(local_batch_loss_sum, device=device)
            dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM) # 使用 all_reduce 方便所有卡都能计算平均值
            avg_train_loss = total_loss_tensor.item() / (len(train_loader) * world_size)
            current_lr = scheduler.get_last_lr()[0]
            
            if is_master:
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f} | LR: {current_lr:.6e}")
            
            # --- 验证阶段 ---
            model.eval()
            local_val_loss_sum = 0
            
            with torch.no_grad():
                for val_cond, val_clean in val_loader:
                    val_clean = val_clean.to(device, non_blocking=True)
                    val_cond = val_cond.to(device, non_blocking=True)
                    B_val = val_clean.size(0)

                    t_model_val = torch.rand(B_val, device=device)
                    t_cf_val = t_model_val.unsqueeze(1)
                    
                    x_t_val, v_star_val, _ = compute_xt_and_velocity(val_clean, t_cf_val)
                    
                    with autocast():
                        v_pred_val = model(x_t_val, t_model_val, val_cond)
                        loss_val = F.mse_loss(v_pred_val, v_star_val)

                    local_val_loss_sum += loss_val.item()

            # 汇总验证 Loss
            val_loss_tensor = torch.tensor(local_val_loss_sum, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            avg_val_loss = val_loss_tensor.item() / (len(val_loader) * world_size)

            if is_master:
                print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.6f}")
                log_file.write(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.6e}\n")
                log_file.flush()

                # --- 保存模型与可视化采样 (仅 Master) ---
                if (epoch + 1) % 10 == 0:
                    # 保存 Checkpoint
                    checkpoint_path = os.path.join(args.model_path, f"checkpoint_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(), 
                    }, checkpoint_path)

                    # 采样可视化 (使用 denoise_sample)
                    # 抓取一个 Batch 进行采样
                    try:
                        # 创建一个新的 iterator 避免干扰 DataLoader 状态
                        temp_iter = iter(val_loader)
                        sample_masked, sample_clean = next(temp_iter)
                        sample_masked = sample_masked[:1].to(device)
                        sample_clean = sample_clean[:1].to(device)

                        # ⚠️ 这里使用 Flow Matching 的 ODE 采样
                        # 注意：model 需要传入 module 因为 model 现在是 DDP 包装的
                        # 但其实 DDP 包装的 model 也可以直接 call，只要 denoise_sample 内部是 model(x, t, cond)
                        denoised_image = denoise_sample(model, sample_masked, steps=50, device=device, save_velocity=False)

                        # 转换数据用于绘图
                        pred_img = denoised_image[0].cpu().numpy().squeeze()
                        gt_img = sample_clean[0].cpu().numpy().squeeze()
                        masked_img = sample_masked[0].cpu().numpy().squeeze()
                        
                        # 简单绘图保存
                        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                        axes[0].imshow(masked_img, cmap='gray'); axes[0].set_title('Masked')
                        axes[1].imshow(pred_img, cmap='gray'); axes[1].set_title(f'Restored E{epoch+1}')
                        axes[2].imshow(gt_img, cmap='gray'); axes[2].set_title('Ground Truth')
                        for ax in axes: ax.axis('off')
                        
                        plt.savefig(os.path.join(args.output_dir_png, f"epoch{epoch+1}_sample.png"), bbox_inches='tight')
                        plt.close(fig)

                        # 保存 numpy 数据
                        np.save(os.path.join(args.output_dir_npy, f"epoch{epoch+1}_pred.npy"), pred_img)
                        print(f"  -> Saved checkoint and visual samples.")
                        
                    except Exception as e:
                        print(f"Visualization error: {e}")

        if is_master:
            torch.save(model.module.state_dict(), f"{args.model_path}/cfm_model_final.pth")
            log_file.write("Training finished.\n")
            log_file.close()
            print("Training finished.")

    finally:
        cleanup()

if __name__ == "__main__":
    args = Args()
    
    # 自动检测可见 GPU 数量
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ngpus_per_node = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    else:
        ngpus_per_node = torch.cuda.device_count()

    print(f"Found {ngpus_per_node} GPUs for training.")
    args.world_size = ngpus_per_node

    # 使用 spawn 启动多进程
    mp.spawn(train,
             args=(args.world_size, args),
             nprocs=args.world_size,
             join=True)