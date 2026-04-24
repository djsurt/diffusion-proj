"""
Generate synthetic opcode sequences with a trained AbsorbingD3PM model,
then convert them to Word2Vec embeddings for comparison with the continuous baseline.

Usage:
    python src/d3pm_generate.py --family zeroaccess --checkpoints checkpoints/ \\
        --n 200 --out synthetic/
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d3pm_data import Vocabulary
from d3pm import AbsorbingD3PM
from embeddings import train_family_word2vec, file_embedding, scale_to_range


def _load_model(
    checkpoint: Path,
    vocab: Vocabulary,
    T: int,
    max_len: int,
    device: torch.device,
) -> AbsorbingD3PM:
    """Reconstruct model from checkpoint. Architecture params are stored in state_dict."""
    # Infer architecture from checkpoint shapes
    sd = torch.load(checkpoint, map_location=device, weights_only=False)

    # Extract d_model from the token embedding weight
    d_model = sd["denoiser.token_emb.weight"].shape[1]
    # Extract num_layers: look for highest layer index in keys
    layer_keys = [k for k in sd if k.startswith("denoiser.transformer.layers.")]
    num_layers = max(int(k.split(".")[3]) for k in layer_keys) + 1 if layer_keys else 4
    # Extract nhead from the first self-attention key's shape
    nhead_keys = [k for k in sd if "self_attn.in_proj_weight" in k]
    if nhead_keys:
        proj_w = sd[nhead_keys[0]]  # shape (3*d_model, d_model)
        # nhead can't be inferred from weights alone; default to 4
        nhead = 4
    else:
        nhead = 4
    # dim_ff from the ffn first linear
    ff_keys = [k for k in sd if "linear1.weight" in k and "transformer" in k]
    dim_ff = sd[ff_keys[0]].shape[0] if ff_keys else 512

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
    ).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model


def generate_sequences(
    model: AbsorbingD3PM,
    vocab: Vocabulary,
    n: int,
    seq_len: int,
    device: torch.device,
    batch_size: int = 64,
) -> list[list[str]]:
    """Generate n opcode sequences, returned as lists of opcode strings."""
    all_seqs: list[list[str]] = []
    for start in range(0, n, batch_size):
        bs = min(batch_size, n - start)
        tokens = model.sample(bs, seq_len, device)  # (bs, seq_len)
        for row in tokens:
            all_seqs.append(vocab.decode(row))
    return all_seqs


def sequences_to_embeddings(
    sequences: list[list[str]],
    w2v_model=None,
    real_sequences: list[list[str]] | None = None,
    embed_dim: int = 104,
) -> np.ndarray:
    """
    Embed each generated sequence as the mean of its opcode vectors, scaled to [-1, 1].

    Uses w2v_model if provided (preferred — keeps same embedding space as real data).
    Falls back to training a new W2V on real+synthetic if w2v_model is None.
    """
    if w2v_model is None:
        train_seqs = (real_sequences or []) + sequences
        w2v_model = train_family_word2vec(train_seqs, dim=embed_dim)
    raw = np.stack([file_embedding(w2v_model, seq) for seq in sequences])
    return scale_to_range(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic opcode sequences with AbsorbingD3PM.")
    parser.add_argument("--family",      required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--malicia",     type=Path, default=Path("malicia"))
    parser.add_argument("--n",           type=int, default=200)
    parser.add_argument("--T",           type=int, default=500)
    parser.add_argument("--max-len",     type=int, default=512)
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

    vocab_path = args.checkpoints / f"{args.family}_d3pm_vocab.pkl"
    ckpt_path  = args.checkpoints / f"{args.family}_d3pm.pt"
    for p in (vocab_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}  — run d3pm_train.py first.")

    print(f"Loading vocab and checkpoint for '{args.family}'...")
    vocab = Vocabulary.load(vocab_path)
    model = _load_model(ckpt_path, vocab, args.T, args.max_len, device)
    print(f"  Vocab size: {vocab.size}")

    # Try to load the saved W2V model from continuous DDPM training (same embedding space)
    w2v_model = None
    real_seqs: list[list[str]] | None = None
    w2v_pkl = args.checkpoints / "w2v_models.pkl"
    if w2v_pkl.exists():
        with open(w2v_pkl, "rb") as fh:
            w2v_dict = pickle.load(fh)
        w2v_model = w2v_dict.get(args.family)
        if w2v_model:
            print(f"  Reusing saved W2V model for '{args.family}' (same embedding space as real data)")
    if w2v_model is None:
        try:
            from data_loader import load_family_opcodes
            corpus = load_family_opcodes(args.malicia, families=[args.family])
            real_seqs = corpus.get(args.family)
            print(f"  Falling back: training new W2V on {len(real_seqs or [])} real sequences")
        except Exception:
            pass

    print(f"Generating {args.n} sequences (seq_len={args.max_len})...")
    seqs = generate_sequences(model, vocab, args.n, args.max_len, device, args.batch)
    non_empty = [s for s in seqs if s]
    print(f"  Non-empty sequences: {len(non_empty)} / {args.n}")

    print("Converting to Word2Vec embeddings...")
    embeddings = sequences_to_embeddings(non_empty or seqs, w2v_model, real_seqs, args.embed_dim)

    args.out.mkdir(parents=True, exist_ok=True)

    # Save sequences as text (one file per sequence)
    seq_dir = args.out / f"{args.family}_d3pm_sequences"
    seq_dir.mkdir(exist_ok=True)
    for i, seq in enumerate(seqs):
        (seq_dir / f"seq_{i:05d}.txt").write_text("\n".join(seq))

    # Save embeddings (compatible with evaluate.py naming convention)
    emb_path = args.out / f"{args.family}_d3pm_synthetic.npy"
    np.save(emb_path, embeddings)
    print(f"Saved embeddings {embeddings.shape} → {emb_path}")
    print(f"Saved {len(seqs)} sequences → {seq_dir}/")


if __name__ == "__main__":
    main()
