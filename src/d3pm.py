"""
AbsorbingD3PM — Discrete Denoising Diffusion Probabilistic Model with absorbing
(mask) transitions for malware opcode sequences (D3PM paper §3.2, NeurIPS 2021).

Schedule: β_t = 1/(T − t + 1)  →  ã_t = (T − t) / T  (linearly decreasing).

Denoiser: transformer encoder that predicts p̃_θ(x₀ | xₜ) at every position.

Loss: L_vb + λ·L_CE  (λ=0.01 per paper §6 for absorbing-state text).

Posterior (closed form for absorbing state):
  q(x_{t-1}=k | x_t=MASK, x_0=k) = 1/t
  q(x_{t-1}=MASK | x_t=MASK, x_0=k) = (t-1)/t
  → KL = (1/t)·log(1/p̃_θ(k)) + ((t-1)/t)·log((t-1)/(t-1+p̃_θ(MASK)))
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Transformer denoiser ─────────────────────────────────────────────────────


class _TransformerDenoiser(nn.Module):
    """
    Compact transformer encoder (pre-LN) that predicts logits over x₀
    for each sequence position, conditioned on the noisy input xₜ and timestep t.
    """

    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        max_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal positional encoding for timestep t → (B, d_model)."""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None].float() * freqs[None]      # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, d_model)

    def forward(
        self,
        x_t: torch.Tensor,         # (B, L) token indices
        t: torch.Tensor,            # (B,) timesteps
        pad_mask: torch.Tensor | None,  # (B, L) True = PAD
    ) -> torch.Tensor:              # (B, L, vocab_size)
        B, L = x_t.shape
        pos = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(x_t) + self.pos_emb(pos)          # (B, L, d)
        x = x + self.time_mlp(self._sinusoidal(t)).unsqueeze(1)  # inject time
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.head(self.norm(x))                         # (B, L, V)


# ── Main D3PM model ──────────────────────────────────────────────────────────


class AbsorbingD3PM(nn.Module):
    """
    D3PM with absorbing-state (MASK) transitions for discrete opcode sequences.

    Vocabulary layout expected from d3pm_data.Vocabulary:
      0 … K-1 : regular opcodes
      K        : [MASK]   (vocab.mask_idx)
      K+1      : [PAD]    (vocab.pad_idx)
      vocab_size = K + 2
    """

    def __init__(
        self,
        vocab_size: int,
        mask_idx: int,
        pad_idx: int,
        T: int = 500,
        max_len: int = 512,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_ff: int = 512,
        dropout: float = 0.1,
        lambda_ce: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx
        self.T = T
        self.max_len = max_len
        self.lambda_ce = lambda_ce

        # ã_t = (T − t) / T  for t = 0, …, T
        t_vals = torch.arange(T + 1, dtype=torch.float32)
        self.register_buffer("alpha_bar", (T - t_vals) / T)   # (T+1,)

        self.denoiser = _TransformerDenoiser(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            max_len=max_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )

    # ── forward diffusion ────────────────────────────────────────────────────

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Sample xₜ ~ q(xₜ | x₀).
        Each non-PAD token is independently masked with probability t/T.
        """
        ab = self.alpha_bar[t][:, None]               # (B, 1) broadcast over L
        mask_prob = (1.0 - ab).expand_as(x0.float())
        should_mask = torch.bernoulli(mask_prob).bool()
        should_mask = should_mask & (x0 != self.pad_idx)   # never mask PAD
        return torch.where(should_mask, torch.full_like(x0, self.mask_idx), x0)

    # ── training ─────────────────────────────────────────────────────────────

    def forward(
        self,
        x0: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Training step.
        x0:       (B, L) integer token indices (true opcode sequence)
        pad_mask: (B, L) bool, True = PAD position
        Returns scalar hybrid loss  L_vb + λ·L_CE.
        """
        B = x0.shape[0]
        t = torch.randint(1, self.T + 1, (B,), device=x0.device)
        x_t = self.q_sample(x0, t)

        # Only compute loss at masked (non-PAD) positions
        is_masked = (x_t == self.mask_idx)
        if pad_mask is not None:
            is_masked = is_masked & ~pad_mask

        logits = self.denoiser(x_t, t, pad_mask)      # (B, L, V)

        if not is_masked.any():
            return (logits * 0.0).sum()                # zero loss, keeps graph

        flat_logits = logits[is_masked]                # (N, V)
        x0_m = x0[is_masked]                          # (N,) true tokens
        t_m = t.unsqueeze(1).expand_as(x0)[is_masked].float()  # (N,)

        # ── VLB: KL(q(x_{t-1}|x_t=MASK, x_0) || p_θ(x_{t-1}|x_t=MASK)) ──
        probs = flat_logits.softmax(dim=-1)            # (N, V)

        p_x0 = probs.gather(1, x0_m.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
        p_msk = probs[:, self.mask_idx].clamp(min=1e-8)  # clamp prevents log(0) in KL denominator

        # KL x₀ term: (1/t) · log(1 / p̃_θ(x₀))
        kl_x0 = (1.0 / t_m) * (-p_x0.log())

        # KL mask term: ((t-1)/t) · log((t-1) / (t-1 + p̃_θ(MASK)))
        # Zero when t=1 to avoid log(0)
        kl_msk = torch.where(
            t_m > 1,
            ((t_m - 1) / t_m) * torch.log(
                (t_m - 1).clamp(min=1e-8) / (t_m - 1 + p_msk).clamp(min=1e-8)
            ),
            torch.zeros_like(t_m),
        )

        vlb_loss = (kl_x0 + kl_msk).mean()

        # ── Auxiliary CE loss: -log p̃_θ(x₀ | xₜ) at masked positions ────────
        ce_loss = F.cross_entropy(flat_logits, x0_m)

        return vlb_loss + self.lambda_ce * ce_loss

    # ── sampling ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        n: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Reverse diffusion: start from all-MASK at t=T, iteratively denoise to x₀.

        Posterior at masked positions:
          p_θ(x_{t-1}=k  | x_t=MASK) = p̃_θ(k)  / t          (k ≠ MASK)
          p_θ(x_{t-1}=MASK | x_t=MASK) = (t-1)/t + p̃_θ(MASK)/t

        Returns (n, seq_len) integer tensor of generated opcode indices.
        """
        x = torch.full((n, seq_len), self.mask_idx, device=device, dtype=torch.long)

        for step in range(self.T, 0, -1):
            t_vec = torch.full((n,), step, device=device, dtype=torch.long)
            logits = self.denoiser(x, t_vec, pad_mask=None)   # (n, L, V)
            probs_x0 = logits.softmax(dim=-1)                 # (n, L, V)

            is_masked = (x == self.mask_idx)
            if not is_masked.any():
                break

            if step == 1:
                # Final step: sample x₀ directly; never output MASK or PAD
                probs_x0[..., self.mask_idx] = 0.0
                probs_x0[..., self.pad_idx] = 0.0
                probs_x0 = probs_x0 / probs_x0.sum(-1, keepdim=True).clamp(min=1e-8)
                flat = probs_x0[is_masked]                    # (N_msk, V)
            else:
                t_val = float(step)
                posterior = probs_x0 / t_val                  # (n, L, V)
                posterior[..., self.mask_idx] += (t_val - 1) / t_val
                posterior[..., self.pad_idx] = 0.0
                posterior = posterior / posterior.sum(-1, keepdim=True).clamp(min=1e-8)
                flat = posterior[is_masked]                   # (N_msk, V)

            sampled = torch.multinomial(flat, 1).squeeze(1)
            x = x.clone()
            x[is_masked] = sampled

        return x    # (n, seq_len) — integer opcode indices
