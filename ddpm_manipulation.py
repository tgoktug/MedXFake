# ddpm_manipulation.py
# DDPM tabanlı maske ile inpainting / manipülasyon modeli

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image

# ======================================================================
# DEVICE & SABİTLER
# ======================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = device          # interface ile uyum için alias
IMG_SIZE = 128
T = 1000                 # diffusion adım sayısı (T_STEPS gibi)

print("DDPM Manipulation Device:", device)

# ======================================================================
# SINUSOIDAL TIME EMBEDDING
# ======================================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B]
        dev = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=dev)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=dev) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# ======================================================================
# RESIDUAL BLOCK
# ======================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.SiLU()
        )

        self.res_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        # x: [B,C,H,W], t_emb: [B, time_emb_dim]
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + time_emb
        h = self.block2(h)
        return h + self.res_conv(x)


# ======================================================================
# UNet ARCHITECTURE
# ======================================================================
class UNet(nn.Module):
    def __init__(self, img_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # down
        self.conv0 = ResidualBlock(img_channels, base_channels, time_emb_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)
        self.conv1 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)
        self.conv2 = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # bottleneck
        self.bot = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # up
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1)
        self.deconv1 = ResidualBlock(base_channels * 4, base_channels * 2, time_emb_dim)
        self.up0 = nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1)
        self.deconv0 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)

        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        # t: [B] (int zaman index)
        t_emb = self.time_mlp(t)  # [B, time_emb_dim]

        x0 = self.conv0(x, t_emb)                      # [B, base,   H,   W]
        x1 = self.conv1(self.down1(x0), t_emb)         # [B, 2base, H/2, W/2]
        x2 = self.conv2(self.down2(x1), t_emb)         # [B, 4base, H/4, W/4]

        b = self.bot(x2, t_emb)                        # [B, 4base, H/4, W/4]

        u1 = self.up1(b)                               # [B, 2base, H/2, W/2]
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.deconv1(u1, t_emb)

        u0 = self.up0(u1)                              # [B, base, H, W]
        u0 = torch.cat([u0, x0], dim=1)
        u0 = self.deconv0(u0, t_emb)

        return self.final_conv(u0)


# API ile uyum için alias
UNetManip = UNet


# ======================================================================
# DIFFUSION PARAMETERS
# ======================================================================
betas = torch.linspace(1e-4, 0.02, T, device=device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


def get_xt_from_x0(x0, noise):
    """
    Orijinal görüntü x0 ve noise ile, istenen t zamanı için x_t üretmek üzere
    callable döner.
    """
    def xt(t_scalar):
        alpha_bar = alphas_cumprod[t_scalar].view(1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    return xt


# ======================================================================
# DDPM INPAINTING (insert_tumor_ddpm)
# ======================================================================
@torch.no_grad()
def insert_tumor_ddpm(model, x_in, mask, num_steps=T):
    """
    x_in : [B,1,H,W]  (orijinal görüntü, değer aralığı [-1,1])
    mask : [B,1,H,W]  (1 = yeniden üretilecek (tümör) bölgesi, 0 = korunacak bölge)
    num_steps : diffusion ters adım sayısı (T ile aynı olabilir)
    """
    model.eval()
    B, C, H, W = x_in.shape

    # Orijinal görüntünün forward noised versiyonu arka plan için
    noise_bg = torch.randn_like(x_in)
    xt_from_x0 = get_xt_from_x0(x_in, noise_bg)

    # Başlangıç: saf noise
    x = torch.randn_like(x_in).to(x_in.device)

    for t in reversed(range(num_steps)):
        t_batch = torch.full((B,), t, device=x_in.device, dtype=torch.long)
        eps_theta = model(x, t_batch)

        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_cum = alphas_cumprod[t]

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        # DDPM ters geçiş
        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_cum)) * eps_theta
        ) + torch.sqrt(beta_t) * noise

        # inpainting: maske DIŞI -> orijinal görüntünün aynı t'deki hali
        x_orig_t = xt_from_x0(t)
        x = mask * x + (1 - mask) * x_orig_t

    # x hala [-1,1] aralığında; normalizasyonu dışarıda yapabilirsin
    return x


# API ile uyum için: ddpm_inpaint ismiyle de erişilsin
@torch.no_grad()
def ddpm_inpaint(model, x_in, mask, num_steps=T):
    """
    insert_tumor_ddpm'in sarmalayıcısı; çıkışı [0,1] aralığına çevirir.
    """
    x = insert_tumor_ddpm(model, x_in, mask, num_steps=num_steps)
    return ((x + 1) / 2).clamp(0, 1).cpu()


# ======================================================================
# (İsteğe bağlı) Yardımcı fonksiyonlar: tek görüntü yükleme
# ======================================================================
def load_image_gray(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("L")
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = arr * 2 - 1  # [0,1] -> [-1,1]
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
