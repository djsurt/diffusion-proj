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
    def __init__(self, channels: int, time_dim: int, dropout: float = 0.01):
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


class AttentionBlock1D(nn.Module):
    """Multi-head self-attention over sequence length."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        H = self.num_heads
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        q = q.reshape(B, H, C // H, L)
        k = k.reshape(B, H, C // H, L)
        v = v.reshape(B, H, C // H, L)
        scale = (C // H) ** -0.5
        attn = (q.transpose(-2, -1) @ k) * scale           # (B, H, L, L)
        attn = attn.softmax(dim=-1)
        out = v @ attn.transpose(-2, -1)                    # (B, H, C/H, L)
        out = out.reshape(B, C, L)
        return x + self.proj(out)


class UNet1D(nn.Module):
    """
    3-level 1D U-Net with attention at the bottleneck (paper Table 6 alignment).
    Input/output shape: (B, 1, embed_dim).
    """

    def __init__(self, embed_dim: int = 104, time_dim: int = 64, base_ch: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.time_emb = SinusoidalPE(time_dim)

        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4

        self.in_conv = nn.Conv1d(1, c1, kernel_size=3, padding=1)
        self.enc1 = ResBlock1D(c1, time_dim)
        self.down1 = nn.Conv1d(c1, c2, kernel_size=3, stride=2, padding=1)

        self.enc2 = ResBlock1D(c2, time_dim)
        self.down2 = nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1)

        self.mid1 = ResBlock1D(c3, time_dim)
        self.attn = AttentionBlock1D(c3)
        self.mid2 = ResBlock1D(c3, time_dim)

        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResBlock1D(c2, time_dim)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResBlock1D(c1, time_dim)

        self.out_conv = nn.Conv1d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)

        h = self.in_conv(x)                              # (B, c1, L)
        e1 = self.enc1(h, t_emb)                         # (B, c1, L)
        e2 = self.enc2(self.down1(e1), t_emb)            # (B, c2, L/2)

        m = self.down2(e2)                               # (B, c3, L/4)
        m = self.mid1(m, t_emb)
        m = self.attn(m)
        m = self.mid2(m, t_emb)

        d2 = self.up2(m)                                 # (B, c2, L/2)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = d2[:, :, : e2.shape[-1]]
        d2 = self.dec2(d2 + e2, t_emb)

        d1 = self.up1(d2)                                # (B, c1, L)
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
        base_ch: int = 32,
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
    def sample(
        self,
        n: int,
        device: torch.device,
        x0_real: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Reverse diffusion (Algorithm 1 from paper).

        Paper approach: sample n real embeddings, forward-diffuse to T_half,
        then reverse-diffuse back. This is SDEdit-style augmentation — the
        synthetic samples are noisy perturbations of real samples, which
        preserves the real data's variance structure.

        If x0_real is None: fall back to starting from N(0, I) (non-paper behavior,
        kept for tests on untrained models).
        """
        if x0_real is not None:
            if x0_real.dim() == 2:
                x0_real = x0_real.unsqueeze(1)  # (N, 1, D)
            idx = torch.randint(0, x0_real.shape[0], (n,), device=x0_real.device)
            x0 = x0_real[idx].to(device)
            t_start = torch.full((n,), self.T_half - 1, device=device, dtype=torch.long)
            x = self.q_sample(x0, t_start)
        else:
            x = torch.randn(n, 1, self.embed_dim, device=device)

        for step in range(self.T_half, 0, -1):
            t_tensor = torch.full((n,), step, device=device, dtype=torch.long)
            eps = self.unet(x, t_tensor)

            alpha_t = self.alphas[step - 1]
            beta_t = self.betas[step - 1]
            sqrt_one_minus_ab = self.sqrt_one_minus_alpha_bar[step - 1]

            coef = (1 - alpha_t) / sqrt_one_minus_ab
            x = (1 / alpha_t.sqrt()) * (x - coef * eps)

            if step > 1:
                sigma = beta_t.sqrt()
                z = torch.randn_like(x)
                x = x + sigma * z

        return x.squeeze(1)  # (n, embed_dim)
