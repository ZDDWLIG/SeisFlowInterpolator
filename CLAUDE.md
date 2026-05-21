# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Conditional Flow Matching model for seismic data interpolation — restoring missing seismic traces (columns) from 2D patches. The model is a U-Net conditioned on a masked seismic image; it predicts the velocity field v_star = (x0 - x_t) / σ that transports noise toward the clean data, then uses an ODE sampler (DDIM-like) for deterministic inference.

## Environment

- Python 3 with PyTorch 2.4.1+cu118, CUDA 11.8
- Key packages: `numpy`, `matplotlib`, `tqdm`
- Two GPUs visible for DDP training (`CUDA_VISIBLE_DEVICES=0,1`)

## Architecture

```
dataset.py     — seis_dataset: loads .npy patches, supports 3 masking modes, normalizes by masked max
model.py       — UNet with ResnetBlock + SinusoidalPositionEmbeddings + optional AttentionBlock
utils.py       — Flow Matching math (sigma_t, compute_xt_and_velocity), ODE sampler (denoise_sample),
                 visualization (visualize_vector_field, plot_npys), data split, seismic colormap
train.py       — Single-GPU training loop
train_DDP.py   — Multi-GPU DDP training with AMP (autocast + GradScaler), DistributedSampler
sgy_to_patches.py — Convert SEG-Y files to .npy patches via inline/crossline grouping + sliding window
```

### Model details

- **Input channels**: 2 (x_t concatenated with condition image)
- **Output channels**: 1 (predicted velocity field)
- Base dim = 64, dim multipliers = (1, 2, 4, 8), GroupNorm with 8 groups
- Time embedding: sinusoidal → MLP(dim → dim*4 → dim*4)
- Self-attention at the bottleneck is disabled by default (`self_attention=False`)
- Condition is injected via channel concatenation at the input (not cross-attention)

### Flow Matching math

- σ(t) = σ_min * (σ_max / σ_min)^t, with σ_min=0.01, σ_max=1.0
- x_t = x_0 + σ·ε  (noise schedule, not interpolation)
- Target velocity v* = (x_0 - x_t) / σ  (points from noisy back to clean)
- Inference: DDIM-like ODE stepping from t=1→0, x_{t+1} = x̂_0 + (x_t - x̂_0) * (σ_{t+1} / σ_t)

### Data pipeline

- Input: `.npy` files in a single directory, shape (W, H), loaded and transposed to (H, W)
- SEG-Y preprocessing: `sgy_to_patches.py` groups traces by inline or crossline, sorts by the other spatial dimension + offset, then cuts patches via sliding window
- Three masking modes (configured with `--mask_mode`):
  - `random` — randomly placed missing columns
  - `uniform` — evenly-spaced missing columns with jitter
  - `large_gap` — one contiguous block of missing columns
- Mask ratio uniformly sampled from `--mask_ratio_range` (default `(0.3, 0.7)`)
- Normalization: divide both masked and clean by the masked tensor's max absolute value
- Train/val split: 90/10 by default, using first N files for train

## Commands

```bash
# Single-GPU training
python train.py --device_id 2 --batch_size 4 --epochs 200 --data_path /path/to/patches/label_256

# Multi-GPU DDP training
python train_DDP.py --batch_size 4 --epochs 200 --data_path /path/to/patches/label_256

# Resume from checkpoint (DDP only)
python train_DDP.py --resume ./results/251107_FM/checkpoints/checkpoint_epoch_10.pth
```

Key flags for both scripts:
- `--mask_ratio_range` (tuple, default `(0.3, 0.7)`) — range for column masking ratio
- `--mask_mode` (str, default `random`, choices: `random`/`uniform`/`large_gap`)
- `--lr` (float, default `1e-4`)

```bash
# Convert SEG-Y to training patches
python sgy_to_patches.py \
    --sgy_path /path/to/file.sgy \
    --out_dir /path/to/patches/label_256 \
    --group_by inline --patch_size 256 --stride 128
```

## Important caveats

- `dataset.py` returns **CPU** tensors (line 141 loads to CPU, line 144 `.float().unsqueeze(0)` stays on CPU). Despite a comment in `train.py` saying "dataset already on GPU," the current implementation does NOT move data to GPU. Training scripts call `.to(device)` explicitly — this is correct.
- If modifying `seis_dataset.__getitem__` to return GPU tensors, you must set `num_workers=0` in the DataLoader.
- `train_DDP.py` hardcodes `CUDA_VISIBLE_DEVICES="0,1"` at the top of the file (line 23), overriding any external environment variable.
- `utils.py` duplicates imports from `model` and `dataset` (lines 219-229) — these are unused and can be removed.
- The `compute_xt_and_velocity` signature accepts `sigma_min`/`sigma_max` with defaults — these should match the defaults used in `sigma_t` and `denoise_sample` (all use 0.01, 1.0).
- Checkpoints are ~350 MB each.
