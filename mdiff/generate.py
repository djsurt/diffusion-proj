"""Unified generation CLI for all diffusion variants.

Usage:
    python -m mdiff.generate --variant ddpm --family zeroaccess --n 500
    python -m mdiff.generate --variant d3pm --family zeroaccess --n 200
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
    p = argparse.ArgumentParser(description="Generate synthetic samples with a trained diffusion model.")
    p.add_argument("--variant", required=True, choices=["ddpm", "d3pm"])
    p.add_argument("--family", required=True)
    p.add_argument("--n", type=int, default=None,
                   help="Default: 100 for ddpm, 200 for d3pm.")
    p.add_argument("--malicia", type=Path, default=Path("malicia"))
    p.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    p.add_argument("--synthetic", type=Path, default=Path("synthetic"))
    p.add_argument("--T", type=int, default=None,
                   help="Default: 1000 for ddpm, 500 for d3pm.")
    p.add_argument("--embed-dim", type=int, default=104)

    # ── D3PM-specific ──────────────────────────────────────────────────────
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--chunks-per-file", type=int, default=0,
                   help="(D3PM only) Chunks per synthetic file. 0 = match avg real file length.")
    p.add_argument("--batch", type=int, default=64,
                   help="(D3PM only) Sampling batch size.")
    return p


def _apply_variant_defaults(args: argparse.Namespace) -> None:
    if args.variant == "ddpm":
        defaults = {"n": 100, "T": 1000}
    else:
        defaults = {"n": 200, "T": 500}
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)


def main() -> None:
    args = _build_parser().parse_args()
    _apply_variant_defaults(args)

    device = _device()
    print(f"Variant: {args.variant} | Device: {device}")

    if args.variant == "ddpm":
        from mdiff.models.ddpm.runner import generate_family
    else:
        from mdiff.models.d3pm.runner import generate_family
    generate_family(args, device)


if __name__ == "__main__":
    main()
