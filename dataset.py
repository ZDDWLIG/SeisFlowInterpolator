import os
import numpy as np
import torch
from torch.utils import data
import random


class seis_dataset(data.Dataset):
    """
    Interpolation dataset with three masking modes:

    - random:      Randomly placed missing columns (default).
    - uniform:     Evenly-spaced missing columns.
    - large_gap:   One contiguous block of missing columns.

    Mask ratio is randomly sampled from mask_ratio_range for each sample.

    Parameters:
        clean_files: list of .npy file paths
        data_shape: (C, H, W)
        mask_ratio_range: (min_ratio, max_ratio) tuple
        mask_mode: one of "random", "uniform", "large_gap"
    """
    def __init__(self, clean_files, data_shape, mask_ratio_range=(0.3, 0.7),
                 mask_mode="random"):
        self.clean_files = clean_files
        self.data_shape = data_shape
        self.mask_ratio_range = mask_ratio_range
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.clean_files)

    def __getitem__(self, idx):
        clean_np = np.load(self.clean_files[idx]) # (H, W)
        clean = torch.from_numpy(clean_np).float().unsqueeze(0)  # (1, H, W)

        masked = clean.clone()
        _, H, W = masked.shape

        min_r, max_r = self.mask_ratio_range
        mask_ratio = np.random.uniform(min_r, max_r)
        num_mask = max(1, int(W * mask_ratio))

        # 构建显式 mask: 1=缺失, 0=已知
        mask = torch.zeros_like(masked)

        if self.mask_mode == "random":
            mask_cols = np.random.choice(W, num_mask, replace=False)
            masked[:, :, mask_cols] = 0
            mask[:, :, mask_cols] = 1

        elif self.mask_mode == "uniform":
            # Evenly-space mask columns across the full width
            step = W / num_mask
            positions = np.arange(0, W, step)
            # Add small random jitter within each bin
            jitter = np.random.uniform(0, step, size=num_mask)
            mask_cols = np.clip((positions + jitter).astype(int), 0, W - 1)
            mask_cols = np.unique(mask_cols)
            masked[:, :, mask_cols] = 0
            mask[:, :, mask_cols] = 1

        elif self.mask_mode == "large_gap":
            # One contiguous block of missing columns
            gap_start = np.random.randint(0, W - num_mask)
            masked[:, :, gap_start:gap_start + num_mask] = 0
            mask[:, :, gap_start:gap_start + num_mask] = 1

        else:
            raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

        return masked, clean, mask
