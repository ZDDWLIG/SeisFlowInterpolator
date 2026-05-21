import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
import glob


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm(h)
        time_bias = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_bias
        h = self.act(h)
        return self.conv2(h)

import torch
import torch.nn as nn
from functools import partial


# helper functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# Residual block with a normalization layer and activation
class ResnetBlock(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim_in),
            nn.SiLU(),
            nn.Conv2d(dim_in, dim_out, 3, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )

        self.res_conv = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = h + time_emb[:, :, None, None]

        h = self.block2(h)
        return h + self.res_conv(x)

# U-Net structure with self-attention and wider channels
class UNet(nn.Module):
    def __init__(self,
                 dim,
                 image_size,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 channels=1,
                 with_time_emb=True,
                 self_attention=False,
                 resnet_block_groups=8):
        super().__init__()

        # determine dimensions
        self.channels = channels
        # init_dim = default(init_dim, dim // 3 * 2)
        init_dim = default(init_dim, dim) 

        self.init_conv = nn.Conv2d(channels + 1, init_dim, 7, padding=3) # input channel becomes 2 (x, cond)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                nn.Conv2d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = AttentionBlock(mid_dim) if self_attention else nn.Identity()
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2, 1) if not is_last else nn.Identity()
            ]))

        self.final_res_block = ResnetBlock(dim, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, out_dim or channels, 1)

    def forward(self, x, t, cond):
        x = torch.cat([x, cond], dim=1)
        
        t = self.time_mlp(t) if exists(self.time_mlp) else None

        x = self.init_conv(x)
        h = []

        for resnet1, resnet2, downsample in self.downs:
            x = resnet1(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet1, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, t)
            x = resnet2(x, t)
            x = upsample(x)
        
        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
    
