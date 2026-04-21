"""
Modified DDPM for 1D malware embeddings, as described in the paper.

Key modifications vs standard DDPM:
  1. U-Net uses Conv1D / Dropout1d instead of 2D equivalents.
  2. Forward diffusion runs only to T//2 (not full T) — the paper found this
     prevents the model from struggling to reconstruct from fully-destroyed samples.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1D U-Net ────────────────────────────────────────────────────────────────

class SinusoidalPE(nn.Module):
    """Sinusoidal timestep positional encoding, projected to `dim`."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.proj(emb)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)
        self.drop = nn.Dropout1d(dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + x


class UNet1D(nn.Module):
    """
    Lightweight 1D U-Net.
    Input/output shape: (B, 1, embed_dim) — the embedding is treated as a
    single-channel 1D signal of length `embed_dim`.
    """

    def __init__(self, embed_dim: int = 104, time_dim: int = 64, base_ch: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_emb = SinusoidalPE(time_dim)

        ch = base_ch
        self.enc1 = ResBlock1D(ch, time_dim)
        self.enc2 = ResBlock1D(ch * 2, time_dim)
        self.down1 = nn.Conv1d(ch, ch * 2, kernel_size=3, stride=2, padding=1)

        self.mid = ResBlock1D(ch * 2, time_dim)

        self.up1 = nn.ConvTranspose1d(ch * 2, ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock1D(ch, time_dim)

        self.in_conv = nn.Conv1d(1, ch, kernel_size=3, padding=1)
        self.out_conv = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        h = self.in_conv(x)          # (B, ch, L)
        e1 = self.enc1(h, t_emb)     # (B, ch, L)
        e2 = self.enc2(self.down1(e1), t_emb)  # (B, 2ch, L/2)
        m = self.mid(e2, t_emb)      # (B, 2ch, L/2)
        d1 = self.up1(m)             # (B, ch, L)

        # crop d1 to match e1 if sizes differ (due to odd lengths)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = d1[:, :, : e1.shape[-1]]

        d1 = self.dec1(d1 + e1, t_emb)
        return self.out_conv(d1)


# ── Diffusion process ────────────────────────────────────────────────────────

class MalwareDiffusion(nn.Module):
    """
    DDPM trained on 104-dim malware embeddings.

    Forward schedule: linear beta from beta_start to beta_end.
    Training uses T_half = T // 2 (paper modification).
    """

    def __init__(
        self,
        embed_dim: int = 104,
        T: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        time_dim: int = 64,
        base_ch: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.T = T
        self.T_half = T // 2

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())

        self.unet = UNet1D(embed_dim=embed_dim, time_dim=time_dim, base_ch=base_ch)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sqrt_ab * x0 + sqrt_1mab * noise

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Training step: sample t uniformly from [1, T_half], add noise,
        predict noise with U-Net, return MSE loss.
        """
        B = x0.shape[0]
        t = torch.randint(1, self.T_half + 1, (B,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t - 1, noise)
        pred = self.unet(x_t, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Reverse diffusion (Algorithm 1 from paper).
        Start from x_{T_half} ~ N(0, I), denoise back to x_0.
        """
        x = torch.randn(n, 1, self.embed_dim, device=device)
        for step in range(self.T_half, 0, -1):
            t_tensor = torch.full((n,), step, device=device, dtype=torch.long)
            eps = self.unet(x, t_tensor)

            alpha_t = self.alphas[step - 1]
            alpha_bar_t = self.alpha_bar[step - 1]
            beta_t = self.betas[step - 1]
            sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[step - 1]

            coef = (1 - alpha_t) / sqrt_one_minus_ab
            x = (1 / alpha_t.sqrt()) * (x - coef * eps)

            if step > 1:
                sigma = beta_t.sqrt()
                z = torch.randn_like(x)
                x = x + sigma * z

        return x.squeeze(1)  # (n, embed_dim)
