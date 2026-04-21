"""
Train a MalwareDiffusion model for each malware family.

Usage:
    python src/train.py --malicia malicia/ --out checkpoints/ [--families zeroaccess zbot]
                        [--epochs 200] [--batch 64] [--T 1000]
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from data_loader import load_family_opcodes
from embeddings import build_family_embeddings
from diffusion import MalwareDiffusion


def train_family(
    embeddings: np.ndarray,
    family: str,
    out_dir: Path,
    epochs: int = 200,
    batch_size: int = 64,
    T: int = 1000,
    lr: float = 2e-4,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    embed_dim = embeddings.shape[1]
    X = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(1)  # (N, 1, D)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = MalwareDiffusion(embed_dim=embed_dim, T=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    bar = tqdm(range(epochs), desc=f"{family}", leave=False)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{family}_diffusion.pt")
    (out_dir / f"{family}_losses.json").write_text(json.dumps(losses))
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--malicia", type=Path, default=Path("malicia"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints"))
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--embed-dim", type=int, default=104)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading opcode sequences...")
    corpus = load_family_opcodes(
        args.malicia, families=args.families, max_files_per_family=args.max_files
    )
    print(f"Families: {list(corpus.keys())}")

    print("Building Word2Vec embeddings (per family)...")
    embeddings, w2v_models = build_family_embeddings(corpus, dim=args.embed_dim)

    # Save embeddings and models for later use
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "embeddings.npy", {k: v for k, v in embeddings.items()})
    with open(args.out / "w2v_models.pkl", "wb") as f:
        pickle.dump(w2v_models, f)
    # Also save as individual arrays
    for family, emb in embeddings.items():
        np.save(args.out / f"{family}_embeddings.npy", emb)
    print(f"Embeddings saved to {args.out}/")

    print("Training diffusion models...")
    for family, emb in embeddings.items():
        print(f"\n[{family}] {emb.shape[0]} samples, embedding dim={emb.shape[1]}")
        losses = train_family(
            emb, family, args.out,
            epochs=args.epochs,
            batch_size=min(args.batch, emb.shape[0]),
            T=args.T,
            lr=args.lr,
            device=device,
        )
        print(f"  Final loss: {losses[-1]:.4f}")

    print("\nAll families trained.")


if __name__ == "__main__":
    main()
