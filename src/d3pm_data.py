"""
Vocabulary and dataset utilities for D3PM discrete diffusion on opcode sequences.

Special token layout (per family vocabulary):
  index 0 … K-1  : unique opcodes (sorted for determinism)
  index K         : [MASK]  (absorbing token, used during diffusion)
  index K+1       : [PAD]   (batch-padding, ignored in loss)
  vocab_size = K + 2
"""

import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class Vocabulary:
    """Bidirectional opcode ↔ integer mapping with [MASK] and [PAD] tokens."""

    MASK_TOKEN = "[MASK]"
    PAD_TOKEN = "[PAD]"

    def __init__(self) -> None:
        self._tok2idx: dict[str, int] = {}
        self._idx2tok: list[str] = []

    # ── construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_sequences(cls, sequences: list[list[str]]) -> "Vocabulary":
        """Build from a list of opcode sequences (one list per file)."""
        vocab = cls()
        opcodes = sorted({op for seq in sequences for op in seq})
        vocab._idx2tok = opcodes + [cls.MASK_TOKEN, cls.PAD_TOKEN]
        vocab._tok2idx = {tok: i for i, tok in enumerate(vocab._idx2tok)}
        return vocab

    # ── special-token indices ─────────────────────────────────────────────────

    @property
    def mask_idx(self) -> int:
        return self._tok2idx[self.MASK_TOKEN]

    @property
    def pad_idx(self) -> int:
        return self._tok2idx[self.PAD_TOKEN]

    @property
    def size(self) -> int:
        return len(self._idx2tok)

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(self, seq: list[str], max_len: int) -> torch.Tensor:
        """Tokenise → truncate to max_len → right-pad with PAD."""
        indices = [self._tok2idx.get(op, self.mask_idx) for op in seq[:max_len]]
        indices += [self.pad_idx] * (max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices: torch.Tensor) -> list[str]:
        """Integer tensor → opcode strings, skipping MASK and PAD."""
        special = {self.mask_idx, self.pad_idx}
        return [self._idx2tok[i] for i in indices.tolist() if i not in special]

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        with open(path, "rb") as fh:
            return pickle.load(fh)


class OpcodeDataset(Dataset):
    """Tokenised + padded opcode sequences for D3PM training.

    Truncates each file to the first `max_len` opcodes. Use OpcodeChunkedDataset
    to instead train on every opcode in every file.
    """

    def __init__(
        self,
        sequences: list[list[str]],
        vocab: Vocabulary,
        max_len: int,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.data = [vocab.encode(seq, max_len) for seq in sequences]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]                        # (max_len,)
        pad_mask = tokens == self.vocab.pad_idx        # True = PAD (ignored in loss)
        return {"tokens": tokens, "pad_mask": pad_mask}


class OpcodeChunkedDataset(Dataset):
    """Splits each sequence into non-overlapping `max_len` chunks (no truncation).

    Files longer than `max_len` produce multiple chunks; the final partial chunk
    is right-padded with PAD. By default partial chunks shorter than `min_chunk`
    opcodes are dropped (they have very few signal tokens after padding).
    """

    def __init__(
        self,
        sequences: list[list[str]],
        vocab: Vocabulary,
        max_len: int,
        min_chunk: int = 32,
    ) -> None:
        self.vocab = vocab
        self.max_len = max_len
        chunks: list[torch.Tensor] = []
        for seq in sequences:
            for start in range(0, len(seq), max_len):
                piece = seq[start : start + max_len]
                if len(piece) < min_chunk:
                    continue
                chunks.append(vocab.encode(piece, max_len))
        self.data = chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tokens = self.data[idx]
        pad_mask = tokens == self.vocab.pad_idx
        return {"tokens": tokens, "pad_mask": pad_mask}
