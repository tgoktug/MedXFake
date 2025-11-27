# ddpm_synthesis.py
# DDPM tabanlı medikal görüntü SENTEZ modeli (sıfırdan üretim)

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Sabitler
# ---------------------------------------------------------
IMG_SIZE   = 128
CHANNELS   = 1
T_STEPS    = 1000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Sinüzoidal Zaman Embedding
# ---------------------------------------------------------
def sinusoidal_time_embedding(t, dim):
    """
    t: [B] (0..T-1 arası int)
    dim: embedding dimension
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=device).float() / half
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

# ---------------------------------------------------------
# ConvBlock (ORİJİNAL İSİMLERLE!)
# ---------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.act   = nn.SiLU()
        self.bn1   = nn.BatchNorm2d(out_c)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.time_mlp = nn.Linear(time_emb_dim, out_c)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.act(h)
        return h

# ---------------------------------------------------------
# UNet (ORİJİNAL trial_ddpm_synthesis.py ile birebir)
# ---------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, img_channels=1, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim*4),
            nn.SiLU(),
            nn.Linear(time_emb_dim*4, time_emb_dim)
        )

        self.down1 = ConvBlock(img_channels, 64,  time_emb_dim)
        self.down2 = ConvBlock(64,         128, time_emb_dim)
        self.down3 = ConvBlock(128,        256, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        self.bot  = ConvBlock(256, 256, time_emb_dim)

        self.up3  = ConvBlock(256+256, 128, time_emb_dim)
        self.up2  = ConvBlock(128+128, 64,  time_emb_dim)
        self.up1  = ConvBlock(64+64,   64,  time_emb_dim)

        self.final = nn.Conv2d(64, img_channels, 1)

    def forward(self, x, t):
        # t: [B]
        t_emb = sinusoidal_time_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Down
        d1 = self.down1(x,              t_emb)   # [B, 64, 128,128]
        d2 = self.down2(self.pool(d1),  t_emb)   # [B,128,  64, 64]
        d3 = self.down3(self.pool(d2),  t_emb)   # [B,256,  32, 32]

        # Bottleneck
        b  = self.bot(self.pool(d3),    t_emb)   # [B,256,  16, 16]

        # Up
        u3 = F.interpolate(b, scale_factor=2, mode="nearest")
        u3 = self.up3(torch.cat([u3, d3], dim=1), t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)

        out = self.final(u1)
        return out

# Bu ismi interface’te kullanmak istersen diye alias:
UNetSynthesis = UNet

# ---------------------------------------------------------
# Diffusion Zaman Parametreleri (ORİJİNAL FORM)
# ---------------------------------------------------------
betas = torch.linspace(1e-4, 0.02, T_STEPS)           # linear schedule
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

betas      = betas.to(DEVICE)
alphas     = alphas.to(DEVICE)
alphas_cumprod      = alphas_cumprod.to(DEVICE)
alphas_cumprod_prev = alphas_cumprod_prev.to(DEVICE)

sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
sqrt_one_minus_cumprod  = torch.sqrt(1 - alphas_cumprod)
sqrt_recip_alphas       = torch.sqrt(1.0 / alphas)
posterior_variance      = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
posterior_variance = posterior_variance.to(DEVICE)

# ---------------------------------------------------------
# Yardımcı: [-1,1] → [0,1]
# ---------------------------------------------------------
def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2

# ---------------------------------------------------------
# DDPM Sampling Fonksiyonu (ORİJİNALLE UYUMLU)
# ---------------------------------------------------------
@torch.no_grad()
def ddpm_sample(model, n_samples, img_size=IMG_SIZE, channels=CHANNELS, device=DEVICE):
    """
    x_T ~ N(0,I) den başlayıp, T_STEPS adımda geriye giderek n_samples görüntü üretir.
    Çıktı: [B,C,H,W], [0,1] aralığında tensor (CPU)
    """
    model.eval()
    x = torch.randn(n_samples, channels, img_size, img_size, device=device)  # x_T

    for t_step in reversed(range(T_STEPS)):
        t_batch = torch.full((n_samples,), t_step, device=device, dtype=torch.long)

        # gürültü tahmini
        pred_noise = model(x, t_batch)

        beta_t       = betas[t_step]
        alpha_t      = alphas[t_step]
        alpha_cum_t  = alphas_cumprod[t_step]

        if t_step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_cum_t)) * pred_noise
            ) + torch.sqrt(beta_t) * noise

    return denorm(x).cpu()
