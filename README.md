# FM4Interpolation

Conditional Flow Matching for seismic data interpolation — restoring missing seismic traces from 2D patches.

## Overview

The model uses a U-Net conditioned on a masked seismic image to predict the velocity field that transports noise toward clean data. A DDIM-like ODE sampler performs deterministic inference.

- **Input**: 2 channels (noisy sample + masked condition)
- **Output**: 1 channel (predicted velocity field)
- **Training**: Flow Matching loss on the velocity field
- **Inference**: ODE integration from noise to clean data

## Architecture

| File | Purpose |
|------|---------|
| `dataset.py` | Data loader with 3 masking modes |
| `model.py` | U-Net with ResNet blocks + sinusoidal time embeddings |
| `utils.py` | Flow Matching math, ODE sampler, visualization |
| `train.py` | Single-GPU training loop |
| `train_DDP.py` | Multi-GPU DDP training with AMP |
| `sgy_to_patches.py` | SEG-Y to `.npy` patch conversion |

### Masking Modes

- `random` — randomly placed missing columns
- `uniform` — evenly-spaced missing columns with jitter
- `large_gap` — one contiguous block of missing columns

## Requirements

- Python 3
- PyTorch 2.4.1+cu118, CUDA 11.8
- numpy, matplotlib, tqdm

## Usage

### Convert SEG-Y to training patches

```bash
python sgy_to_patches.py \
    --sgy_path /path/to/file.sgy \
    --out_dir /path/to/patches/label_256 \
    --group_by inline --patch_size 256 --stride 128
```

### Training

```bash
# Single-GPU
python train.py --device_id 2 --batch_size 4 --epochs 200 --data_path /path/to/patches/label_256

# Multi-GPU DDP
python train_DDP.py --batch_size 4 --epochs 200 --data_path /path/to/patches/label_256

# Resume from checkpoint (DDP only)
python train_DDP.py --resume ./results/251107_FM/checkpoints/checkpoint_epoch_10.pth
```

Key flags: `--mask_mode` (random/uniform/large_gap), `--mask_ratio_range` (default 0.3, 0.7), `--lr` (default 1e-4).

## Flow Matching

σ(t) = σ_min · (σ_max / σ_min)^t, where σ_min=0.01, σ_max=1.0. The target velocity v* = (x₀ - x_t) / σ points from the noisy sample back to clean data.
