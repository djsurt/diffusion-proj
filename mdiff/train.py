"""Unified training CLI for all diffusion variants.

Usage:
    python -m mdiff.train --variant ddpm --families zeroaccess winwebsec zbot \\
        [--epochs 200] [--batch 64] [--T 1000]

    python -m mdiff.train --variant d3pm --families zeroaccess \\
        [--epochs 100] [--batch 8] [--T 500] [--max-len 2048]

The --variant flag picks the runner. Each runner reads the same args.Namespace
and ignores fields it doesn't use, so adding a third variant is one switch case.
"""

import argparse
from pathlib import Path

import torch


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a diffusion model on malware opcode data.")
    p.add_argument("--variant", required=True, choices=["ddpm", "d3pm"],
                   help="ddpm = continuous DDPM on W2V embeddings; d3pm = discrete absorbing D3PM.")
    p.add_argument("--families", nargs="+", required=True,
                   help="One or more malware family names (subdirectories of --malicia).")
    p.add_argument("--malicia", type=Path, default=Path("malicia"))
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p.add_argument("--max-files", type=int, default=None,
                   help="Cap files per family (useful for fast testing)")

    # ── DDPM-specific defaults ─────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=None,
                   help="Default: 200 for ddpm, 100 for d3pm.")
    p.add_argument("--batch", type=int, default=None,
                   help="Default: 64 for ddpm, 8 for d3pm.")
    p.add_argument("--T", type=int, default=None,
                   help="Default: 1000 for ddpm, 500 for d3pm.")
    p.add_argument("--lr", type=float, default=None,
                   help="Default: 2e-4 for ddpm, 1e-3 for d3pm.")
    p.add_argument("--embed-dim", type=int, default=104,
                   help="Word2Vec embedding dim (DDPM only).")

    # ── D3PM-specific ──────────────────────────────────────────────────────
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dim-ff", type=int, default=512)
    p.add_argument("--lambda-ce", type=float, default=0.01)
    p.add_argument("--no-chunked", action="store_true",
                   help="Disable chunked training (truncate to first --max-len opcodes).")
    p.add_argument("--min-chunk", type=int, default=32)

    return p


def _apply_variant_defaults(args: argparse.Namespace) -> None:
    """Fill in variant-specific defaults for any flag the user left unset."""
    if args.variant == "ddpm":
        defaults = {"epochs": 200, "batch": 64, "T": 1000, "lr": 2e-4}
    else:  # d3pm
        defaults = {"epochs": 100, "batch": 8, "T": 500, "lr": 1e-3}
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def main() -> None:
    args = _build_parser().parse_args()
    _apply_variant_defaults(args)

    device = _device()
    print(f"Variant: {args.variant} | Device: {device}")

    if args.variant == "ddpm":
        from mdiff.models.ddpm.runner import train_families
    else:
        from mdiff.models.d3pm.runner import train_families
    train_families(args, device)


if __name__ == "__main__":
    main()
