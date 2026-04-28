"""
Generate synthetic opcode sequences with a trained AbsorbingD3PM model,
then convert them to Word2Vec embeddings in the SAME W2V space and scale
range as the (recomputed) real reference.

Usage:
    python src/d3pm_generate.py --family zeroaccess --checkpoints checkpoints/ \\
        --n 200 --out synthetic/
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d3pm_data import Vocabulary
from d3pm import AbsorbingD3PM
from data_loader import load_family_opcodes
from embeddings import train_family_word2vec, file_embedding, scale_to_range


def _load_model(
    checkpoint: Path,
    vocab: Vocabulary,
    T: int,
    max_len: int,
    device: torch.device,
) -> AbsorbingD3PM:
    """Reconstruct model from checkpoint. Architecture inferred from state_dict shapes."""
    sd = torch.load(checkpoint, map_location=device, weights_only=False)
    d_model = sd["denoiser.token_emb.weight"].shape[1]
    layer_keys = [k for k in sd if k.startswith("denoiser.transformer.layers.")]
    num_layers = max(int(k.split(".")[3]) for k in layer_keys) + 1 if layer_keys else 4
    ff_keys = [k for k in sd if "linear1.weight" in k and "transformer" in k]
    dim_ff = sd[ff_keys[0]].shape[0] if ff_keys else 512

    model = AbsorbingD3PM(
        vocab_size=vocab.size,
        mask_idx=vocab.mask_idx,
        pad_idx=vocab.pad_idx,
        T=T,
        max_len=max_len,
        d_model=d_model,
        nhead=4,
        num_layers=num_layers,
        dim_ff=dim_ff,
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def generate_sequences(
    model: AbsorbingD3PM,
    vocab: Vocabulary,
    n_files: int,
    chunks_per_file: int,
    seq_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> list[list[str]]:
    """Generate n_files synthetic 'files', each = concatenation of chunks_per_file chunks."""
    total_chunks = n_files * chunks_per_file
    decoded_chunks: list[list[str]] = []
    for start in range(0, total_chunks, batch_size):
        bs = min(batch_size, total_chunks - start)
        tokens = model.sample(bs, seq_len, device)  # (bs, seq_len)
        for row in tokens:
            decoded_chunks.append(vocab.decode(row))
        print(f"  generated {min(start + bs, total_chunks)}/{total_chunks} chunks")

    files: list[list[str]] = []
    for i in range(n_files):
        flat: list[str] = []
        for c in decoded_chunks[i * chunks_per_file : (i + 1) * chunks_per_file]:
            flat.extend(c)
        files.append(flat)
    return files


def _load_or_train_w2v(
    family: str,
    checkpoints: Path,
    real_seqs: list[list[str]],
    embed_dim: int,
):
    """Load saved W2V for `family` if present in d3pm_w2v.pkl; else train and save."""
    pkl = checkpoints / "d3pm_w2v.pkl"
    store: dict = {}
    if pkl.exists():
        with open(pkl, "rb") as fh:
            store = pickle.load(fh)
        if family in store:
            print(f"  Loaded W2V for '{family}' from {pkl}")
            return store[family]

    print(f"  Training fresh W2V on {len(real_seqs)} real sequences (dim={embed_dim})...")
    model = train_family_word2vec(real_seqs, dim=embed_dim)
    store[family] = model
    with open(pkl, "wb") as fh:
        pickle.dump(store, fh)
    print(f"  Saved W2V to {pkl}")
    return model


def _embed_files(w2v, files: list[list[str]]) -> np.ndarray:
    return np.stack([file_embedding(w2v, seq) for seq in files])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic opcode sequences with AbsorbingD3PM and embed them.")
    parser.add_argument("--family",      required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--malicia",     type=Path, default=Path("malicia"))
    parser.add_argument("--n",           type=int, default=200,
                        help="Number of synthetic 'files' to generate.")
    parser.add_argument("--chunks-per-file", type=int, default=0,
                        help="Chunks per synthetic file. 0 = auto match avg real file length.")
    parser.add_argument("--T",           type=int, default=500)
    parser.add_argument("--max-len",     type=int, default=2048)
    parser.add_argument("--embed-dim",   type=int, default=104)
    parser.add_argument("--out",         type=Path, default=Path("synthetic"))
    parser.add_argument("--batch",       type=int, default=64)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    vocab_path = args.checkpoints / f"{args.family}_d3pm_vocab.pkl"
    ckpt_path  = args.checkpoints / f"{args.family}_d3pm.pt"
    for p in (vocab_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}  — run d3pm_train.py first.")

    print(f"Loading vocab and checkpoint for '{args.family}'...")
    vocab = Vocabulary.load(vocab_path)
    model = _load_model(ckpt_path, vocab, args.T, args.max_len, device)
    print(f"  Vocab size: {vocab.size}")

    # ── load real sequences (needed for W2V + real reference embeddings) ────
    print(f"Loading real sequences for '{args.family}'...")
    corpus = load_family_opcodes(args.malicia, families=[args.family])
    if args.family not in corpus:
        raise SystemExit(f"Family '{args.family}' not found in {args.malicia}")
    real_seqs = corpus[args.family]
    avg_real_len = sum(len(s) for s in real_seqs) / len(real_seqs)
    print(f"  Real files: {len(real_seqs)}  (avg length: {avg_real_len:.0f} opcodes)")

    chunks_per_file = args.chunks_per_file or max(1, math.ceil(avg_real_len / args.max_len))
    print(f"  chunks_per_file: {chunks_per_file}  → ~{chunks_per_file * args.max_len} tokens / synth file")

    # ── W2V (load or train; persist alongside D3PM checkpoints) ─────────────
    w2v = _load_or_train_w2v(args.family, args.checkpoints, real_seqs, args.embed_dim)

    # ── real reference embeddings in this W2V space ─────────────────────────
    print("Embedding real sequences in this W2V space...")
    real_raw = _embed_files(w2v, real_seqs)
    real_scaled = scale_to_range(real_raw)
    real_ref_path = args.checkpoints / f"{args.family}_d3pm_real_embeddings.npy"
    np.save(real_ref_path, real_scaled)
    print(f"  Saved real reference {real_scaled.shape} → {real_ref_path}")

    # ── generate synthetic files ────────────────────────────────────────────
    print(f"Generating {args.n} synthetic files (chunks={chunks_per_file} × seq_len={args.max_len})...")
    synth_files = generate_sequences(
        model, vocab,
        n_files=args.n,
        chunks_per_file=chunks_per_file,
        seq_len=args.max_len,
        device=device,
        batch_size=args.batch,
    )
    avg_synth_len = sum(len(s) for s in synth_files) / len(synth_files)
    print(f"  Synthetic files: {len(synth_files)}  (avg length: {avg_synth_len:.0f} opcodes)")

    # ── synthetic embeddings (same W2V, scaled with real reference min/max) ─
    print("Embedding synthetic files in the same W2V space...")
    synth_raw = _embed_files(w2v, synth_files)
    synth_scaled = scale_to_range(synth_raw, ref=real_raw)

    args.out.mkdir(parents=True, exist_ok=True)

    # save sequences as text (one file per synthetic file)
    seq_dir = args.out / f"{args.family}_d3pm_sequences"
    seq_dir.mkdir(exist_ok=True)
    for i, seq in enumerate(synth_files):
        (seq_dir / f"seq_{i:05d}.txt").write_text("\n".join(seq))

    emb_path = args.out / f"{args.family}_d3pm_synthetic.npy"
    np.save(emb_path, synth_scaled)
    print(f"Saved synthetic embeddings {synth_scaled.shape} → {emb_path}")
    print(f"Saved {len(synth_files)} sequences → {seq_dir}/")


if __name__ == "__main__":
    main()
