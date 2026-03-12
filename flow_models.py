"""
flow_models.py
==============
Model architecture for conditional flow matching.

Exports
-------
  CustomPad          : reflect (lat) + circular (lon) padding
  ConvResBlockSingle : single-conv residual block with GroupNorm + Mish
  SinusoidalEmbedding: sinusoidal time embedding projected to bottleneck dim
  Unet6R             : 6-level U-Net velocity field predictor, parameterized
                       by base_channels for easy scaling

Scaling guide (base_channels)
------------------------------
  base=2  : dev/MPS — fast iteration, minimal capacity
  base=4  : mid — good for GPU training runs
  base=8  : large — upper bound, comparable to serious climate emulators

Channel progression (standard pyramid: narrow input, wide bottleneck)
  c1  = base * 1    (full resolution,   128x192)
  c2  = base * 2    (64x96)
  c4  = base * 4    (32x48)
  c8  = base * 8    (16x24)
  c16 = base * 16   (8x12)
  c32 = base * 32   (4x6)
  c64 = base * 64   (2x3  — bottleneck)

The time embedding out_dim is always c64 regardless of base_channels,
so it is injected additively at the bottleneck without a dimension mismatch.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomPad(nn.Module):
    """
    Reflect padding in latitude (height) and circular padding in longitude
    (width), matching the topology of a global lat/lon field.
    """
    def __init__(self, pad_height, pad_width):
        super().__init__()
        self.pad_height = pad_height
        self.pad_width  = pad_width

    def forward(self, x):
        x = F.pad(x, (0, 0, self.pad_height, self.pad_height), mode='reflect')
        x = F.pad(x, (self.pad_width, self.pad_width, 0, 0),   mode='circular')
        return x


class ConvResBlockSingle(nn.Module):
    """
    Residual block with a single conv layer: CONV -> NORM -> ACT -> DROPOUT.
    Skip connection with 1x1 conv if in_ch != out_ch.
    """
    def __init__(self, in_ch, out_ch, k_size=3, p_drop=0.0, gn_groups=1):
        super().__init__()
        pad        = (k_size - 1) // 2
        self.pad   = CustomPad(pad, pad)
        self.conv1 = nn.Conv2d(in_ch, out_ch, k_size, padding=0)
        self.gn1   = nn.GroupNorm(num_groups=gn_groups, num_channels=out_ch)
        self.act   = nn.Mish(inplace=True)
        self.dp    = nn.Dropout2d(p_drop) if p_drop else nn.Identity()
        self.skip  = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        y = self.act(self.gn1(self.conv1(self.pad(x))))
        y = self.dp(y)
        return self.act(y + self.skip(x))


class SinusoidalEmbedding(nn.Module):
    """
    Maps scalar t in [0, 1] to a sinusoidal feature vector of length `dim`,
    then projects to `out_dim` for additive injection at the bottleneck.
    """
    def __init__(self, dim=64, out_dim=8):
        super().__init__()
        assert dim % 2 == 0
        self.dim  = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, out_dim * 4),
            nn.Mish(),
            nn.Linear(out_dim * 4, out_dim),
        )

    def forward(self, t):
        # t : (B,) in [0, 1]
        half  = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (np.log(10000) / (half - 1))
        )
        args = t[:, None] * freqs[None, :]
        emb  = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.proj(emb)                                          # (B, out_dim)


class Unet6R(nn.Module):
    """
    6-level U-Net for predicting the conditional flow matching velocity field.

    Channel width scales as powers of 2 from base_channels at the input up to
    base_channels * 64 at the bottleneck (standard pyramid). This concentrates
    parameters where spatial resolution is low and representational richness
    is needed most.

    Inputs
    ------
      xt   : (B, 1, H, W)  interpolated field  (1-t)*x0 + t*x1
      clim : (B, 1, H, W)  conditioning climatology (normalized)
      t    : (B,)           flow time in [0, 1]

    xt and clim are concatenated -> (B, 2, H, W) before the encoder.
    Time t is sinusoidally embedded and injected additively at the bottleneck.

    Output
    ------
      (B, 1, H, W)  predicted velocity field  v ≈ x1 - x0

    Parameters
    ----------
    base_channels : int
        Controls overall model size. Channel counts at each level are
        base * [1, 2, 4, 8, 16, 32, 64]. Recommended values:
          2  — dev/MPS (fast iteration)
          4  — mid (GPU training)
          8  — large (upper bound)
    """
    def __init__(self, input_channels=2, output_channels=1,
                 base_channels=4, kernel_size=3, p_drop=0.0, gn_groups=1):
        super().__init__()
        k  = kernel_size
        b  = base_channels
        c1, c2, c4, c8, c16, c32, c64 = (b, b*2, b*4, b*8, b*16, b*32, b*64)

        # time embedding projects to bottleneck channel count
        self.t_emb = SinusoidalEmbedding(dim=64, out_dim=c64)

        # encoder
        self.enc1  = ConvResBlockSingle(input_channels, c1,  k, p_drop, gn_groups)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = ConvResBlockSingle(c1,  c2,  k, p_drop, gn_groups)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3  = ConvResBlockSingle(c2,  c4,  k, p_drop, gn_groups)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4  = ConvResBlockSingle(c4,  c8,  k, p_drop, gn_groups)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5  = ConvResBlockSingle(c8,  c16, k, p_drop, gn_groups)
        self.pool5 = nn.MaxPool2d(2)
        self.enc6  = ConvResBlockSingle(c16, c32, k, p_drop, gn_groups)
        self.pool6 = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = ConvResBlockSingle(c32, c64, k, p_drop, gn_groups)

        # decoder
        self.upconv1 = nn.ConvTranspose2d(c64, c32, 2, stride=2)
        self.dec1    = ConvResBlockSingle(c32 + c32, c32, k, p_drop, gn_groups)
        self.upconv2 = nn.ConvTranspose2d(c32, c16, 2, stride=2)
        self.dec2    = ConvResBlockSingle(c16 + c16, c16, k, p_drop, gn_groups)
        self.upconv3 = nn.ConvTranspose2d(c16, c8,  2, stride=2)
        self.dec3    = ConvResBlockSingle(c8  + c8,  c8,  k, p_drop, gn_groups)
        self.upconv4 = nn.ConvTranspose2d(c8,  c4,  2, stride=2)
        self.dec4    = ConvResBlockSingle(c4  + c4,  c4,  k, p_drop, gn_groups)
        self.upconv5 = nn.ConvTranspose2d(c4,  c2,  2, stride=2)
        self.dec5    = ConvResBlockSingle(c2  + c2,  c2,  k, p_drop, gn_groups)
        self.upconv6 = nn.ConvTranspose2d(c2,  c1,  2, stride=2)
        self.dec6    = ConvResBlockSingle(c1  + c1,  c1,  k, p_drop, gn_groups)

        self.final_conv = nn.Conv2d(c1, output_channels, 1)

    def forward(self, xt, clim, t):
        x = torch.cat([xt, clim], dim=1)   # (B, 2, H, W)

        orig_h, orig_w = x.shape[2], x.shape[3]
        pad_h = (64 - orig_h % 64) % 64
        pad_w = (64 - orig_w % 64) % 64
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_w//2, pad_w - pad_w//2,
                          pad_h//2, pad_h - pad_h//2), mode='reflect')

        x1 = self.enc1(x);   x1p = self.pool1(x1)
        x2 = self.enc2(x1p); x2p = self.pool2(x2)
        x3 = self.enc3(x2p); x3p = self.pool3(x3)
        x4 = self.enc4(x3p); x4p = self.pool4(x4)
        x5 = self.enc5(x4p); x5p = self.pool5(x5)
        x6 = self.enc6(x5p); x6p = self.pool6(x6)

        b  = self.bottleneck(x6p)
        b  = b + self.t_emb(t)[:, :, None, None]   # inject time embedding

        u1 = self.upconv1(b);  d1 = self.dec1(torch.cat([u1, x6], dim=1))
        u2 = self.upconv2(d1); d2 = self.dec2(torch.cat([u2, x5], dim=1))
        u3 = self.upconv3(d2); d3 = self.dec3(torch.cat([u3, x4], dim=1))
        u4 = self.upconv4(d3); d4 = self.dec4(torch.cat([u4, x3], dim=1))
        u5 = self.upconv5(d4); d5 = self.dec5(torch.cat([u5, x2], dim=1))
        u6 = self.upconv6(d5); d6 = self.dec6(torch.cat([u6, x1], dim=1))

        out = self.final_conv(d6)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, pad_h//2 : orig_h + pad_h//2,
                            pad_w//2 : orig_w + pad_w//2]
        return out
