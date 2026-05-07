"""Train + generate orchestration for the continuous DDPM variant.

Per-family layout under `checkpoints/<family>/`:
  ddpm.pt              — trained weights
  ddpm_embeddings.npy  — real W2V embeddings (DDPM coordinate system)
  ddpm_losses.json     — per-epoch loss curve

Per-family layout under `synthetic/<family>/`:
  ddpm.npy             — generated embeddings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mdiff.data import load_family_opcodes, build_family_embeddings
from mdiff.models.ddpm.model import MalwareDiffusion
from mdiff import paths


def _train_one(
    embeddings: np.ndarray,
    family: str,
    family_dir: Path,
    *,
    epochs: int,
    batch_size: int,
    T: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    embed_dim = embeddings.shape[1]
    X = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = MalwareDiffusion(embed_dim=embed_dim, T=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses: list[float] = []
    bar = tqdm(range(epochs), desc=family, leave=False)
    for _ in bar:
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = model(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        bar.set_postfix(loss=f"{avg:.4f}")

    torch.save(model.state_dict(), family_dir / paths.DDPM_MODEL)
    (family_dir / paths.DDPM_LOSSES).write_text(json.dumps(losses))
    return losses


def train_families(args: argparse.Namespace, device: torch.device) -> None:
    print("Loading opcode sequences...")
    corpus = load_family_opcodes(
        args.malicia, families=args.families, max_files_per_family=args.max_files,
    )
    print(f"Families: {list(corpus.keys())}")

    print("Building Word2Vec embeddings (per family)...")
    embeddings, _ = build_family_embeddings(corpus, dim=args.embed_dim)

    args.checkpoints.mkdir(parents=True, exist_ok=True)
    for family, emb in embeddings.items():
        fd = paths.family_ckpt_dir(args.checkpoints, family)
        np.save(fd / paths.DDPM_EMBEDDINGS, emb)

    print("Training DDPM models...")
    for family, emb in embeddings.items():
        print(f"\n[{family}] {emb.shape[0]} samples, embedding dim={emb.shape[1]}")
        fd = paths.family_ckpt_dir(args.checkpoints, family)
        losses = _train_one(
            emb, family, fd,
            epochs=args.epochs,
            batch_size=min(args.batch, emb.shape[0]),
            T=args.T,
            lr=args.lr,
            device=device,
        )
        print(f"  Final loss: {losses[-1]:.4f}")

    print("\nAll families trained.")


def generate_family(args: argparse.Namespace, device: torch.device) -> None:
    family = args.family
    fd_ckpt = paths.family_ckpt_dir(args.checkpoints, family)
    ckpt = fd_ckpt / paths.DDPM_MODEL
    real_path = fd_ckpt / paths.DDPM_EMBEDDINGS
    for p in (ckpt, real_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}  — run training first.")

    x0_real = np.load(real_path)
    print(f"Loaded {x0_real.shape} real embeddings as seeds")

    model = MalwareDiffusion(embed_dim=args.embed_dim, T=args.T).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    x0_tensor = torch.tensor(x0_real, dtype=torch.float32, device=device)
    print(f"Generating {args.n} samples for '{family}'...")
    samples = model.sample(args.n, device, x0_real=x0_tensor).cpu().numpy()

    fd_synth = paths.family_synth_dir(args.synthetic, family)
    out_path = fd_synth / paths.DDPM_SYNTHETIC
    np.save(out_path, samples)
    print(f"Saved {samples.shape} -> {out_path}")
