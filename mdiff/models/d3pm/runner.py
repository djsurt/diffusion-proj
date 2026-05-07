"""Train + generate orchestration for the discrete D3PM variant.

Per-family layout under `checkpoints/<family>/`:
  d3pm.pt                    — trained weights
  d3pm_vocab.pkl             — Vocabulary (opcode↔idx, MASK/PAD)
  d3pm_losses.json           — per-epoch loss curve
  d3pm_w2v.pkl               — Word2Vec model used to embed sequences (separate from DDPM W2V)
  d3pm_real_embeddings.npy   — real W2V embeddings in *this* W2V space

Per-family layout under `synthetic/<family>/`:
  d3pm.npy                   — generated embeddings (same W2V space as the real ref)
  d3pm_sequences/seq_*.txt   — raw generated opcode sequences
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mdiff.data import (
    load_family_opcodes,
    Vocabulary,
    OpcodeDataset,
    OpcodeChunkedDataset,
    train_family_word2vec,
    file_embedding,
    scale_to_range,
)
from mdiff.models.d3pm.model import AbsorbingD3PM
from mdiff import paths


# ── training ─────────────────────────────────────────────────────────────────


def _train_one(
    sequences: list[list[str]],
    family: str,
    family_dir: Path,
    *,
    T: int,
    max_len: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    dim_ff: int,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_ce: float,
    chunked: bool,
    min_chunk: int,
    device: torch.device,
) -> list[float]:
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

    losses: list[float] = []
    bar = tqdm(range(epochs), desc=family, leave=False)
    for _ in bar:
        epoch_loss = 0.0
        for batch in loader:
            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)
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

    torch.save(model.state_dict(), family_dir / paths.D3PM_MODEL)
    vocab.save(family_dir / paths.D3PM_VOCAB)
    (family_dir / paths.D3PM_LOSSES).write_text(json.dumps(losses))
    print(f"  Saved checkpoint and vocab to {family_dir}/")

    return losses


def train_families(args: argparse.Namespace, device: torch.device) -> None:
    for family in args.families:
        print(f"\nLoading opcode sequences for '{family}'...")
        corpus = load_family_opcodes(
            args.malicia, families=[family], max_files_per_family=args.max_files,
        )
        if family not in corpus:
            raise SystemExit(f"Family '{family}' not found in {args.malicia}")
        sequences = corpus[family]
        print(f"Files: {len(sequences)}")

        fd = paths.family_ckpt_dir(args.checkpoints, family)
        losses = _train_one(
            sequences=sequences,
            family=family,
            family_dir=fd,
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
        print(f"Final loss for '{family}': {losses[-1]:.4f}")


# ── generation ───────────────────────────────────────────────────────────────


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


def _sample_files(
    model: AbsorbingD3PM,
    vocab: Vocabulary,
    n_files: int,
    chunks_per_file: int,
    seq_len: int,
    device: torch.device,
    batch_size: int,
) -> list[list[str]]:
    """Generate n_files synthetic 'files', each = concatenation of chunks_per_file chunks."""
    total_chunks = n_files * chunks_per_file
    decoded_chunks: list[list[str]] = []
    for start in range(0, total_chunks, batch_size):
        bs = min(batch_size, total_chunks - start)
        tokens = model.sample(bs, seq_len, device)
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


def _load_or_train_w2v(family_dir: Path, real_seqs: list[list[str]], embed_dim: int):
    """Load saved W2V from `family_dir/d3pm_w2v.pkl` if present, else train and save."""
    pkl = family_dir / paths.D3PM_W2V
    if pkl.exists():
        with open(pkl, "rb") as fh:
            print(f"  Loaded W2V from {pkl}")
            return pickle.load(fh)

    print(f"  Training fresh W2V on {len(real_seqs)} real sequences (dim={embed_dim})...")
    model = train_family_word2vec(real_seqs, dim=embed_dim)
    with open(pkl, "wb") as fh:
        pickle.dump(model, fh)
    print(f"  Saved W2V to {pkl}")
    return model


def _embed_files(w2v, files: list[list[str]]) -> np.ndarray:
    return np.stack([file_embedding(w2v, seq) for seq in files])


def generate_family(args: argparse.Namespace, device: torch.device) -> None:
    family = args.family
    fd_ckpt = paths.family_ckpt_dir(args.checkpoints, family)
    vocab_path = fd_ckpt / paths.D3PM_VOCAB
    ckpt_path = fd_ckpt / paths.D3PM_MODEL
    for p in (vocab_path, ckpt_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}  — run training first.")

    print(f"Loading vocab and checkpoint for '{family}'...")
    vocab = Vocabulary.load(vocab_path)
    model = _load_model(ckpt_path, vocab, args.T, args.max_len, device)
    print(f"  Vocab size: {vocab.size}")

    print(f"Loading real sequences for '{family}'...")
    corpus = load_family_opcodes(args.malicia, families=[family])
    if family not in corpus:
        raise SystemExit(f"Family '{family}' not found in {args.malicia}")
    real_seqs = corpus[family]
    avg_real_len = sum(len(s) for s in real_seqs) / len(real_seqs)
    print(f"  Real files: {len(real_seqs)}  (avg length: {avg_real_len:.0f} opcodes)")

    chunks_per_file = args.chunks_per_file or max(1, math.ceil(avg_real_len / args.max_len))
    print(f"  chunks_per_file: {chunks_per_file}  → ~{chunks_per_file * args.max_len} tokens / synth file")

    w2v = _load_or_train_w2v(fd_ckpt, real_seqs, args.embed_dim)

    print("Embedding real sequences in this W2V space...")
    real_raw = _embed_files(w2v, real_seqs)
    real_scaled = scale_to_range(real_raw)
    real_ref_path = fd_ckpt / paths.D3PM_REAL_EMBEDDINGS
    np.save(real_ref_path, real_scaled)
    print(f"  Saved real reference {real_scaled.shape} → {real_ref_path}")

    print(f"Generating {args.n} synthetic files (chunks={chunks_per_file} × seq_len={args.max_len})...")
    synth_files = _sample_files(
        model, vocab,
        n_files=args.n,
        chunks_per_file=chunks_per_file,
        seq_len=args.max_len,
        device=device,
        batch_size=args.batch,
    )
    avg_synth_len = sum(len(s) for s in synth_files) / len(synth_files)
    print(f"  Synthetic files: {len(synth_files)}  (avg length: {avg_synth_len:.0f} opcodes)")

    print("Embedding synthetic files in the same W2V space...")
    synth_raw = _embed_files(w2v, synth_files)
    synth_scaled = scale_to_range(synth_raw, ref=real_raw)

    fd_synth = paths.family_synth_dir(args.synthetic, family)
    seq_dir = fd_synth / paths.D3PM_SEQ_DIR
    seq_dir.mkdir(exist_ok=True)
    for i, seq in enumerate(synth_files):
        (seq_dir / f"seq_{i:05d}.txt").write_text("\n".join(seq))

    emb_path = fd_synth / paths.D3PM_SYNTHETIC
    np.save(emb_path, synth_scaled)
    print(f"Saved synthetic embeddings {synth_scaled.shape} → {emb_path}")
    print(f"Saved {len(synth_files)} sequences → {seq_dir}/")
