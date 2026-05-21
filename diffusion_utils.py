"""
Diffusion (DDPM) training utilities and DDIM sampler for seismic interpolation.

Condition: time t + masked image x_cond (no explicit mask channel).
Model:  x_t, x_cond  →  predicted noise ε.
"""
import torch
import torch.nn.functional as F
import numpy as np


# ── Noise schedules ──

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (Improved DDPM). Better than linear for seismic data."""
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(betas, max=0.999)


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear schedule (original DDPM)."""
    return torch.linspace(beta_start, beta_end, timesteps)


def compute_alphas(betas):
    """betas -> alphas, alphas_cumprod, etc."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
    }


# ── Forward diffusion (q-sample) ──

def q_sample(x0, t, alphas_cumprod, noise=None):
    """x_t = sqrt(a_cum[t]) * x0 + sqrt(1 - a_cum[t]) * noise"""
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_alpha = alphas_cumprod[t] ** 0.5
    sqrt_one_minus_alpha = (1.0 - alphas_cumprod[t]) ** 0.5
    while sqrt_alpha.dim() < x0.dim():
        sqrt_alpha = sqrt_alpha.unsqueeze(-1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
    return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise, noise


# ── DDIM sampler ──

@torch.no_grad()
def ddim_sample(model, x_cond, timesteps, sampling_timesteps,
                alphas_cumprod, device, eta=0.0):
    """
    DDIM reverse process for conditional generation (inpainting).

    Args:
        model: UNet(cond_channels=1) predicting noise ε from (x_t, t, x_cond)
        x_cond: (B, 1, H, W) masked seismic image (condition)
        timesteps: total diffusion steps (e.g. 1000)
        sampling_timesteps: number of DDIM steps (e.g. 100)
        alphas_cumprod: shape (timesteps,)
        eta: 0 = deterministic DDIM, 1 = stochastic DDPM
    Returns:
        (B, 1, H, W) denoised image
    """
    B, _, H, W = x_cond.shape

    # Start from pure noise, same shape as x_cond
    img = torch.randn(B, 1, H, W, device=device)

    # Sub-sequence of timesteps (evenly spaced, decreasing)
    step_indices = torch.linspace(timesteps - 1, 0, sampling_timesteps,
                                  dtype=torch.long, device=device)

    for i in range(sampling_timesteps):
        t_idx = step_indices[i]                       # current index
        t_prev = step_indices[i + 1] if i < sampling_timesteps - 1 else 0

        t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
        t_normalized = t_batch.float() / timesteps    # [0, 1] for time embedding

        # Predict noise
        noise_pred = model(img, t_normalized, x_cond)

        # Predicted x_0
        alpha_t = alphas_cumprod[t_idx]
        alpha_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0, device=device)
        sqrt_alpha_t = alpha_t ** 0.5
        sqrt_one_minus_alpha_t = (1 - alpha_t) ** 0.5

        x0_pred = (img - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t

        # Direction to x_t
        dir_xt = (1 - alpha_prev) ** 0.5 * noise_pred

        # Random noise (eta=0 -> deterministic)
        if eta > 0:
            sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)) ** 0.5
            noise = torch.randn_like(img)
        else:
            sigma = 0.0
            noise = 0.0

        img = (alpha_prev ** 0.5) * x0_pred + dir_xt + sigma * noise

    return img
