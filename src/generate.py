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
    x0_real: np.ndarray | None = None,
) -> np.ndarray:
    model = MalwareDiffusion(embed_dim=embed_dim, T=T).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    x0_tensor = None
    if x0_real is not None:
        x0_tensor = torch.tensor(x0_real, dtype=torch.float32, device=device)
    samples = model.sample(n, device, x0_real=x0_tensor)
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    ckpt = args.checkpoints / f"{args.family}_diffusion.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint found: {ckpt}")

    real_path = args.checkpoints / f"{args.family}_embeddings.npy"
    if not real_path.exists():
        raise FileNotFoundError(f"No real embeddings found: {real_path}")
    x0_real = np.load(real_path)
    print(f"Loaded {x0_real.shape} real embeddings as seeds")

    print(f"Generating {args.n} samples for '{args.family}'...")
    samples = generate(ckpt, args.embed_dim, args.n, args.T, device, x0_real=x0_real)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.family}_synthetic.npy"
    np.save(out_path, samples)
    print(f"Saved {samples.shape} -> {out_path}")


if __name__ == "__main__":
    main()
