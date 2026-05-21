import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap


def seismic(iop=1):
    """
    Seismic colormap for Python
    
    Parameters:
        iop: int
            1 = min brown, zero white, max black
            2 = min red, zero white, max black
            3 = min blue, zero white, max red
            4 = custom (less used)
    
    Returns:
        M: ListedColormap
            Matplotlib colormap object
    """
    N = 40
    L = 40
    size_total = 128

    if iop == 1:
        u1 = np.concatenate([0.5*np.ones(N), np.linspace(0.5,1,size_total-N), 
                             np.linspace(1,0,size_total-N), np.zeros(N)])
        u2 = np.concatenate([0.25*np.ones(N), np.linspace(0.25,1,size_total-N),
                             np.linspace(1,0,size_total-N), np.zeros(N)])
        u3 = np.concatenate([np.zeros(N), np.linspace(0,1,size_total-N),
                             np.linspace(1,0,size_total-N), np.zeros(N)])
    elif iop == 2:
        u1 = np.concatenate([np.ones(N), np.linspace(1,1,size_total-N),
                             np.linspace(1,0,size_total-N), np.zeros(N)])
        u2 = np.concatenate([np.zeros(N), np.linspace(0,1,size_total-N),
                             np.linspace(1,0,size_total-N), np.zeros(N)])
        u3 = np.concatenate([np.zeros(N), np.linspace(0,1,size_total-N),
                             np.linspace(1,0,size_total-N), np.zeros(N)])
    elif iop == 3:
        u1 = np.concatenate([np.zeros(N), np.linspace(0,1,size_total-N-L//2), 
                             np.ones(L), np.linspace(1,0.5,size_total-L//2)])
        u2 = np.concatenate([np.zeros(N), np.linspace(0,1,size_total-N-L//2),
                             np.ones(L), np.linspace(1,0,size_total-N-L//2), np.zeros(N)])
        u3 = np.concatenate([np.linspace(0.5,1,size_total-L//2), np.ones(L),
                             np.linspace(1,0,size_total-N-L//2), np.zeros(N)])
    elif iop == 4:
        u1 = np.concatenate([np.linspace(1,1,128), np.linspace(1,0,128)])
        u2 = np.concatenate([np.linspace(0,1,128), np.linspace(1,0,128)])
        u3 = np.concatenate([np.linspace(0,1,128), np.linspace(1,1,128)])
    else:
        raise ValueError("iop must be 1,2,3 or 4")

    # 合并为 (128,3) 数组
    M = np.vstack([u1, u2, u3]).T

    return ListedColormap(M)

def sigma_t(t, sigma_min=0.01, sigma_max=1.0):
    return sigma_min * (sigma_max / sigma_min) ** t

def compute_xt_and_velocity(x0, t, sigma_min=0.01, sigma_max=1.0):
    noise = torch.randn_like(x0)
    sigma = sigma_t(t, sigma_min, sigma_max).view(-1, 1, 1, 1)
    x_t = x0 + sigma * noise
    v_star = (x0 - x_t) / sigma
    return x_t, v_star, noise

def visualize_vector_field(v_star, v_pred, epoch, save_dir="./vis"):
    """
    可视化 v_star 和 v_pred (白色背景)
    - 如果 C=1，则用梯度构造伪矢量场并画箭头
    - 如果 C>=2，则直接画矢量场
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    os.makedirs(save_dir, exist_ok=True)
    v_star_np = v_star.detach().cpu().numpy()
    v_pred_np = v_pred.detach().cpu().numpy()

    # 只取第一个样本
    v_gt = v_star_np[0]  # [C, H, W]
    v_pd = v_pred_np[0]

    # --- 创建图窗和坐标轴，并设置图窗背景色 ---
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), facecolor='white')

    if v_gt.shape[0] == 1:
        # --- 单通道情况：在灰度图上绘制向量 ---
        img_gt = v_gt[0]
        img_pd = v_pd[0]

        Gy_gt, Gx_gt = np.gradient(img_gt)
        Gy_pd, Gx_pd = np.gradient(img_pd)

        skip = 8
        Y, X = np.mgrid[0:img_gt.shape[0]:skip, 0:img_gt.shape[1]:skip]

        # 绘制左图
        axs[0].imshow(img_gt, cmap='gray')
        axs[0].quiver(X, Y, Gx_gt[::skip, ::skip], Gy_gt[::skip, ::skip],
                      color='red', scale=50)
        axs[0].set_title("v_star (gradient arrows)")

        # 绘制右图
        axs[1].imshow(img_pd, cmap='gray')
        axs[1].quiver(X, Y, Gx_pd[::skip, ::skip], Gy_pd[::skip, ::skip],
                      color='blue', scale=50)
        axs[1].set_title("v_pred (gradient arrows)")

    elif v_gt.shape[0] >= 2:
        # --- 多通道情况：直接绘制向量场 ---
        skip = 4
        Y, X = np.mgrid[0:v_gt.shape[1]:skip, 0:v_gt.shape[2]:skip]
        U_gt, V_gt = v_gt[0, ::skip, ::skip], v_gt[1, ::skip, ::skip]
        U_pd, V_pd = v_pd[0, ::skip, ::skip], v_pd[1, ::skip, ::skip]

        # 绘制左图
        axs[0].quiver(X, Y, U_gt, V_gt, color='blue')
        axs[0].set_title("v_star")

        # 绘制右图
        axs[1].quiver(X, Y, U_pd, V_pd, color='green')
        axs[1].set_title("v_pred")

    # --- 统一设置所有子图的样式 ---
    for ax in axs:
        ax.set_facecolor('white') # 设置坐标轴背景色
        ax.axis('off')            # 关闭坐标轴

    plt.tight_layout()
    # --- 保存图像，并再次确认背景色 ---
    plt.savefig(f"{save_dir}/vector_field_epoch{epoch:03d}.png", facecolor='white', bbox_inches='tight')
    plt.close(fig)



# def denoise_sample(model, x_cond, steps=100, device='cpu'):
#     model.eval()
    
#     # 从需要去噪的图像开始
#     x_t = x_cond.clone()
#     # x_t = torch.randn_like(x_cond)

    
#     # 时间步从 1 到 0
#     time_steps = torch.linspace(1, 0, steps + 1, device=device)
    
#     with torch.no_grad():
#         for i in tqdm(range(steps), desc="Denoising"):
#             t_now = time_steps[i]
#             t_next = time_steps[i+1]
            
#             # 准备模型输入，注意 t 的形状是 [B]
#             t_model = t_now.expand(x_t.shape[0])
            
#             # 预测速度场 (v_pred ≈ -noise)
#             v_pred = model(x_t, t_model, x_cond)
            

#             sigma_now = sigma_t(t_now.view(-1, 1, 1, 1))
#             sigma_next = sigma_t(t_next.view(-1, 1, 1, 1))

#             # 预测的干净图像 x0
#             x0_pred = x_t + sigma_now * v_pred
            
#             # 引导方向
#             direction_from_x0 = x_t - x0_pred

#             # 更新到下一个时间步 (基于DDIM的确定性采样)
#             x_t = x0_pred + direction_from_x0 * (sigma_next / sigma_now)

#     return x_t # 假设数据范围是[-1, 1]


def denoise_sample(model, x_cond, steps=100, device='cpu', save_velocity=True, save_dir='./velocity_fields'):
    model.eval()
    
    # 初始化
    x_t = x_cond.clone()
    time_steps = torch.linspace(1, 0, steps + 1, device=device)
    velocity_list = []

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(steps), desc="Denoising"):
            t_now = time_steps[i]
            t_next = time_steps[i+1]
            t_model = t_now.expand(x_t.shape[0])
            
            # 预测速度场
            v_pred = model(x_t, t_model, x_cond)

            sigma_now = sigma_t(t_now.view(-1, 1, 1, 1))
            sigma_next = sigma_t(t_next.view(-1, 1, 1, 1))

            x0_pred = x_t + sigma_now * v_pred
            direction_from_x0 = x_t - x0_pred
            x_t = x0_pred + direction_from_x0 * (sigma_next / sigma_now)
            
            # 记录速度场
            if save_velocity:
                velocity_list.append(v_pred.cpu().numpy())

    if save_velocity:
        velocity_array = np.stack(velocity_list, axis=0)  # shape: [steps, B, C, H, W]
        np.save(os.path.join(save_dir, 'velocity_fields.npy'), velocity_array)
        print(f"✅ 已保存速度场序列到 {save_dir}/velocity_fields.npy")

    return x_t


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import *
from dataset import *
import math
import os
import glob

import os

def split_dataset(data_path, train_ratio=0.9):
    """
    适配只有一个文件夹，里面全是 clean patch 文件的情况。
    返回训练集 clean 路径列表和验证集 clean 路径列表。
    """
    # 获取所有 patch 文件
    all_files = sorted([f for f in os.listdir(data_path) if f.endswith(".npy")])
    num_total = len(all_files)

    if num_total == 0:
        raise ValueError(f"No .npy files found in {data_path}")

    num_train = int(num_total * train_ratio)

    # 划分
    train_files = all_files[:num_train]
    val_files = all_files[num_train:]

    # 构建完整路径
    train_clean_paths = [os.path.join(data_path, f) for f in train_files]
    val_clean_paths = [os.path.join(data_path, f) for f in val_files]

    # 返回格式仍保持 (train_clean_paths, val_clean_paths)
    return train_clean_paths, val_clean_paths

import numpy as np
import matplotlib.pyplot as plt

def plot_npys(arrays, titles=None, figsize=(12, 4), cmap=seismic(2)):
    """
    可视化多个 npy 数组，每个数组一个子图。

    参数：
        arrays: list[np.ndarray]  直接传入 numpy 数组
        titles: list[str]         每个子图的标题（可选）
        figsize: tuple            整个画布大小
        cmap: str                 颜色图
    """

    num = len(arrays)
    # if titles is None:
    #     titles = [f"array {i}" for i in range(num)]

    plt.figure(figsize=figsize)

    for i, arr in enumerate(arrays):
        std = np.std(arr)
        vmin, vmax = -2 * std, 2 * std

        plt.subplot(1, num, i + 1)
        # plt.title(titles[i])
        plt.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar()

    plt.tight_layout()
    plt.show()


def save_sample_images(noisy, pred, clean, epoch, save_dir="./checkpoints/samples"):
    """保存含噪 / 预测 / 干净数据对比图"""
    os.makedirs(save_dir, exist_ok=True)

    noisy_img = noisy.squeeze().cpu().numpy()
    pred_img = pred.squeeze().cpu().numpy()
    clean_img = clean.squeeze().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(noisy_img, cmap='gray')
    axes[0].set_title("Noisy Input")
    axes[0].axis("off")

    axes[1].imshow(pred_img, cmap='gray')
    axes[1].set_title(f"Prediction (Epoch {epoch})")
    axes[1].axis("off")

    axes[2].imshow(clean_img, cmap='gray')
    axes[2].set_title("Ground Truth (Clean)")
    axes[2].axis("off")

    save_path = os.path.join(save_dir, f"epoch_{epoch}_sample.png")
    plt.savefig(save_path)
    plt.close(fig)
    print(f"  -> 样本图像已保存至 {save_path}")

def normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data, dtype=np.float32)
    scaled_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
    return scaled_data.astype(np.float32), (min_val, max_val)


def denormalization(scaled_data, min_max):
    min_val, max_val = min_max
    if max_val == min_val:
        return np.full_like(scaled_data, min_val, dtype=np.float32)
    original_data = (scaled_data + 1) / 2 * (max_val - min_val) + min_val
    return original_data.astype(np.float32)

def align_range(source, target):
    s_mean, s_std = source.mean(), source.std()
    t_mean, t_std = target.mean(), target.std()

    aligned = (source - s_mean) / (s_std + 1e-8) * t_std + t_mean
    return aligned


