"""
Visualize .npy patches using the same dataset pipeline as training.

Shows what the model actually sees: the raw patch, the masked input (x_cond),
the mask, and the clean target (x0), all normalized identically to training.

Usage:
    python visualize_patches.py --data_path /path/to/patches/label_256

    python visualize_patches.py --data_path /path/to/patches/label_256 \
        --mask_mode large_gap --num_samples 8

    python visualize_patches.py --data_path /path/to/patches/label_256 \
        --save_dir ./vis_output --save  # save to files instead of showing
"""
import argparse
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from dataset import seis_dataset
from utils import seismic


def parse_args():
    p = argparse.ArgumentParser(description="Visualize dataset patches")
    p.add_argument("--data_path", type=str, required=True,
                   help="Directory containing .npy patch files")
    p.add_argument("--num_samples", type=int, default=20,
                   help="Number of patches to visualize")
    p.add_argument("--mask_mode", type=str, default="random",
                   choices=["random", "uniform", "large_gap"],
                   help="Masking pattern to apply")
    p.add_argument("--mask_ratio_range", type=float, nargs=2, default=(0.3, 0.7),
                   metavar=("LOW", "HIGH"))
    p.add_argument("--img_size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible masking")
    p.add_argument("--save_dir", type=str, default=None,
                   help="If set, save images to this directory instead of displaying")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Collect files ──
    all_files = sorted([
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path)
        if f.endswith(".npy")
    ])
    print(f"Found {len(all_files)} .npy files in {args.data_path}")

    if len(all_files) == 0:
        print("No .npy files found.")
        return

    n_show = min(args.num_samples, len(all_files))
    chosen = random.sample(all_files, n_show)

    # ── Build dataset with same logic as training ──
    ds = seis_dataset(
        clean_files=chosen,
        data_shape=args.img_size,
        mask_ratio_range=args.mask_ratio_range,
        mask_mode=args.mask_mode,
    )

    # ── Plot ──
    fig, axes = plt.subplots(n_show, 3, figsize=(10, 3.2 * n_show))
    if n_show == 1:
        axes = axes[None, :]  # make 2D for uniform indexing

    col_titles = ["Clean (x₀)", "Masked (x_cond)", "Mask"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=13, fontweight="bold")

    cmap = seismic(2)

    for i in range(n_show):
        masked, clean, mask = ds[i]  # all (1, H, W) tensors

        clean_np = clean.squeeze().numpy()
        masked_np = masked.squeeze().numpy()
        mask_np = mask.squeeze().numpy()

        # Shared color range from clean data
        std = np.std(clean_np)
        vmin, vmax = -2 * std, 2 * std

        # Clean
        axes[i, 0].imshow(clean_np, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        axes[i, 0].set_ylabel(os.path.basename(chosen[i]), fontsize=9)

        # Masked
        axes[i, 1].imshow(masked_np, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # Mask overlay (red where masked, transparent elsewhere)
        axes[i, 2].imshow(clean_np, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        axes[i, 2].imshow(
            np.ma.masked_where(mask_np < 0.5, mask_np),
            cmap="Reds", alpha=0.6, aspect="auto",
        )

        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(args.save_dir, f"patches_{args.mask_mode}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
