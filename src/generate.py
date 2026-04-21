"""
Load trained diffusion models and generate synthetic embeddings.

Usage:
    python src/generate.py --checkpoints checkpoints/ --family zeroaccess --n 100 --out synthetic/
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from diffusion import MalwareDiffusion


def generate(
    checkpoint: Path,
    embed_dim: int,
    n: int,
    T: int = 1000,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    model = MalwareDiffusion(embed_dim=embed_dim, T=T).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    samples = model.sample(n, device)
    return samples.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--family", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=104)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("synthetic"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = args.checkpoints / f"{args.family}_diffusion.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt}")

    print(f"Generating {args.n} samples for '{args.family}'...")
    samples = generate(ckpt, args.embed_dim, args.n, args.T, device)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.family}_synthetic.npy"
    np.save(out_path, samples)
    print(f"Saved {samples.shape} -> {out_path}")


if __name__ == "__main__":
    main()
