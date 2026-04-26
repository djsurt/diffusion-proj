"""
Train AbsorbingD3PM on a single malware family's opcode sequences.

Usage:
    python src/d3pm_train.py --family zeroaccess --malicia malicia/ \\
        --out checkpoints/ [--epochs 100] [--T 500] [--max-len 512]
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_family_opcodes
from d3pm_data import Vocabulary, OpcodeDataset, OpcodeChunkedDataset
from d3pm import AbsorbingD3PM


def train(
    sequences: list[list[str]],
    family: str,
    out_dir: Path,
    *,
    T: int = 500,
    max_len: int = 512,
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 4,
    dim_ff: int = 512,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    lambda_ce: float = 0.1,
    chunked: bool = True,
    min_chunk: int = 32,
    device: torch.device,
) -> list[float]:
    # ── vocabulary ────────────────────────────────────────────────────────────
    vocab = Vocabulary.from_sequences(sequences)
    print(f"  Vocabulary size: {vocab.size}  (opcodes: {vocab.size - 2})")

    if chunked:
        dataset = OpcodeChunkedDataset(sequences, vocab, max_len=max_len, min_chunk=min_chunk)
        avg_len = sum(len(s) for s in sequences) / max(len(sequences), 1)
        print(f"  Chunked dataset: {len(dataset)} chunks of up to {max_len} tokens "
              f"(avg file len: {avg_len:.0f})")
    else:
        dataset = OpcodeDataset(sequences, vocab, max_len=max_len)
        print(f"  Truncated dataset: {len(dataset)} files (first {max_len} tokens each)")

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)),
                        shuffle=True, drop_last=False)

    # ── model ─────────────────────────────────────────────────────────────────
    model = AbsorbingD3PM(
        vocab_size=vocab.size,
        mask_idx=vocab.mask_idx,
        pad_idx=vocab.pad_idx,
        T=T,
        max_len=max_len,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_ff=dim_ff,
        lambda_ce=lambda_ce,
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── training loop ─────────────────────────────────────────────────────────
    losses: list[float] = []
    bar = tqdm(range(epochs), desc=family, leave=False)
    for _ in bar:
        epoch_loss = 0.0
        for batch in loader:
            tokens = batch["tokens"].to(device)       # (B, L)
            pad_mask = batch["pad_mask"].to(device)   # (B, L)
            loss = model(tokens, pad_mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        losses.append(avg)
        scheduler.step()
        bar.set_postfix(loss=f"{avg:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / f"{family}_d3pm.pt")
    vocab.save(out_dir / f"{family}_d3pm_vocab.pkl")
    (out_dir / f"{family}_d3pm_losses.json").write_text(json.dumps(losses))
    print(f"  Saved checkpoint and vocab to {out_dir}/")

    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AbsorbingD3PM on a malware family.")
    parser.add_argument("--family",   required=True, help="Family name (e.g. zeroaccess)")
    parser.add_argument("--malicia",  type=Path, default=Path("malicia"))
    parser.add_argument("--out",      type=Path, default=Path("checkpoints"))
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--T",        type=int, default=500)
    parser.add_argument("--max-len",  type=int, default=512)
    parser.add_argument("--d-model",  type=int, default=128)
    parser.add_argument("--nhead",    type=int, default=4)
    parser.add_argument("--layers",   type=int, default=4)
    parser.add_argument("--dim-ff",   type=int, default=512)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--lambda-ce", type=float, default=0.1)
    parser.add_argument("--no-chunked", action="store_true",
                        help="Disable chunked training (fall back to first-max_len truncation).")
    parser.add_argument("--min-chunk", type=int, default=32,
                        help="Drop chunked windows shorter than this many opcodes.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Cap files per family (useful for fast testing)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print(f"Loading opcode sequences for '{args.family}'...")
    corpus = load_family_opcodes(
        args.malicia, families=[args.family],
        max_files_per_family=args.max_files,
    )
    if args.family not in corpus:
        raise SystemExit(f"Family '{args.family}' not found in {args.malicia}")

    sequences = corpus[args.family]
    print(f"Files: {len(sequences)}")

    losses = train(
        sequences=sequences,
        family=args.family,
        out_dir=args.out,
        T=args.T,
        max_len=args.max_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        lambda_ce=args.lambda_ce,
        chunked=not args.no_chunked,
        min_chunk=args.min_chunk,
        device=device,
    )
    print(f"Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
