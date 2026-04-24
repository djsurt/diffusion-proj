"""
Unit tests for the D3PM pipeline.

Tests cover:
  - Vocabulary building, encoding, decoding
  - Forward diffusion (q_sample) invariants
  - Model forward pass (loss is scalar, finite, decreasing)
  - Sampling shape and token validity
  - Sequence-level evaluation utilities

Uses a tiny corpus of synthetic opcode sequences so the suite runs in < 30s.

Run: python tests/test_d3pm.py
"""

import sys
import unittest
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from d3pm_data import Vocabulary, OpcodeDataset
from d3pm import AbsorbingD3PM
from d3pm_evaluate import opcode_freq_kl, ngram_overlap, edit_distance_stats

# ── tiny synthetic corpus ─────────────────────────────────────────────────────
OPCODES = ["mov", "push", "pop", "call", "ret", "xor", "add", "sub", "jmp", "nop"]
RNG = np.random.default_rng(42)

def _make_seqs(n: int = 30, min_len: int = 20, max_len: int = 80) -> list[list[str]]:
    seqs = []
    for _ in range(n):
        length = int(RNG.integers(min_len, max_len))
        seqs.append([RNG.choice(OPCODES) for _ in range(length)])
    return seqs

SEQS = _make_seqs(30)

FAST_T = 50
FAST_MAX_LEN = 64
EMBED_DIM = 16   # tiny for speed

# ── Vocabulary tests ──────────────────────────────────────────────────────────

class TestVocabulary(unittest.TestCase):

    def setUp(self):
        self.vocab = Vocabulary.from_sequences(SEQS)

    def test_size(self):
        # unique opcodes + MASK + PAD
        expected = len({op for seq in SEQS for op in seq}) + 2
        self.assertEqual(self.vocab.size, expected)

    def test_special_indices_distinct(self):
        self.assertNotEqual(self.vocab.mask_idx, self.vocab.pad_idx)

    def test_encode_shape(self):
        tokens = self.vocab.encode(SEQS[0], max_len=FAST_MAX_LEN)
        self.assertEqual(tokens.shape, (FAST_MAX_LEN,))
        self.assertEqual(tokens.dtype, torch.long)

    def test_encode_pads_short_seq(self):
        short = ["mov", "push"]
        tokens = self.vocab.encode(short, max_len=10)
        pad_count = (tokens == self.vocab.pad_idx).sum().item()
        self.assertEqual(pad_count, 8)

    def test_encode_truncates_long_seq(self):
        long_seq = ["mov"] * 200
        tokens = self.vocab.encode(long_seq, max_len=FAST_MAX_LEN)
        self.assertEqual(len(tokens), FAST_MAX_LEN)
        # No PAD tokens since the sequence is longer than max_len
        self.assertFalse((tokens == self.vocab.pad_idx).any())

    def test_decode_roundtrip(self):
        seq = SEQS[0][:FAST_MAX_LEN]
        tokens = self.vocab.encode(seq, max_len=FAST_MAX_LEN)
        decoded = self.vocab.decode(tokens)
        self.assertEqual(decoded, seq)

    def test_decode_strips_pad(self):
        tokens = self.vocab.encode(["mov"], max_len=5)
        decoded = self.vocab.decode(tokens)
        self.assertEqual(decoded, ["mov"])

    def test_mask_not_in_decode(self):
        # Manually create a tensor with MASK
        t = torch.tensor([self.vocab.mask_idx, 0, self.vocab.pad_idx])
        decoded = self.vocab.decode(t)
        self.assertNotIn(Vocabulary.MASK_TOKEN, decoded)
        self.assertNotIn(Vocabulary.PAD_TOKEN, decoded)

    def test_save_load(self):
        import tempfile, pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        self.vocab.save(path)
        loaded = Vocabulary.load(path)
        self.assertEqual(loaded.size, self.vocab.size)
        self.assertEqual(loaded.mask_idx, self.vocab.mask_idx)
        self.assertEqual(loaded.pad_idx, self.vocab.pad_idx)


# ── OpcodeDataset tests ───────────────────────────────────────────────────────

class TestOpcodeDataset(unittest.TestCase):

    def setUp(self):
        self.vocab = Vocabulary.from_sequences(SEQS)
        self.ds = OpcodeDataset(SEQS, self.vocab, max_len=FAST_MAX_LEN)

    def test_length(self):
        self.assertEqual(len(self.ds), len(SEQS))

    def test_item_keys(self):
        item = self.ds[0]
        self.assertIn("tokens", item)
        self.assertIn("pad_mask", item)

    def test_item_shapes(self):
        item = self.ds[0]
        self.assertEqual(item["tokens"].shape, (FAST_MAX_LEN,))
        self.assertEqual(item["pad_mask"].shape, (FAST_MAX_LEN,))

    def test_pad_mask_dtype(self):
        item = self.ds[0]
        self.assertEqual(item["pad_mask"].dtype, torch.bool)


# ── AbsorbingD3PM tests ───────────────────────────────────────────────────────

class TestD3PMModel(unittest.TestCase):

    def setUp(self):
        self.vocab = Vocabulary.from_sequences(SEQS)
        self.model = AbsorbingD3PM(
            vocab_size=self.vocab.size,
            mask_idx=self.vocab.mask_idx,
            pad_idx=self.vocab.pad_idx,
            T=FAST_T,
            max_len=FAST_MAX_LEN,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_ff=64,
        )
        self.device = torch.device("cpu")

    def _batch(self, n: int = 4):
        ds = OpcodeDataset(SEQS[:n], self.vocab, max_len=FAST_MAX_LEN)
        tokens = torch.stack([ds[i]["tokens"] for i in range(n)])
        pad_mask = torch.stack([ds[i]["pad_mask"] for i in range(n)])
        return tokens, pad_mask

    def test_forward_scalar_loss(self):
        tokens, pad_mask = self._batch()
        loss = self.model(tokens, pad_mask)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_loss_finite(self):
        tokens, pad_mask = self._batch()
        loss = self.model(tokens, pad_mask)
        self.assertTrue(torch.isfinite(loss))

    def test_q_sample_shape(self):
        tokens, _ = self._batch()
        t = torch.randint(1, FAST_T + 1, (4,))
        x_t = self.model.q_sample(tokens, t)
        self.assertEqual(x_t.shape, tokens.shape)

    def test_q_sample_only_masks(self):
        """q_sample can only MASK tokens — it must not change unmasked tokens to other opcodes."""
        tokens, _ = self._batch()
        t = torch.randint(1, FAST_T + 1, (4,))
        x_t = self.model.q_sample(tokens, t)
        mask_idx = self.vocab.mask_idx
        # For each position: x_t[i,j] is either the original token OR the MASK
        same_or_masked = (x_t == tokens) | (x_t == mask_idx)
        self.assertTrue(same_or_masked.all())

    def test_q_sample_pad_stays_pad(self):
        """PAD tokens must never become MASK."""
        tokens, _ = self._batch()
        # Force at-least a few PAD positions
        tokens[:, -10:] = self.vocab.pad_idx
        t = torch.full((4,), FAST_T, dtype=torch.long)  # maximum masking
        x_t = self.model.q_sample(tokens, t)
        pad_positions = tokens == self.vocab.pad_idx
        # All original PAD positions should still be PAD
        self.assertTrue((x_t[pad_positions] == self.vocab.pad_idx).all())

    def test_alpha_bar_boundary(self):
        self.assertAlmostEqual(self.model.alpha_bar[0].item(), 1.0, places=5)
        self.assertAlmostEqual(self.model.alpha_bar[FAST_T].item(), 0.0, places=5)

    def test_sample_shape(self):
        samples = self.model.sample(n=5, seq_len=FAST_MAX_LEN, device=self.device)
        self.assertEqual(samples.shape, (5, FAST_MAX_LEN))

    def test_sample_no_mask_token(self):
        """Generated samples should not contain MASK tokens."""
        samples = self.model.sample(n=4, seq_len=FAST_MAX_LEN, device=self.device)
        self.assertFalse((samples == self.vocab.mask_idx).any())

    def test_sample_valid_vocab(self):
        """All generated tokens must be valid vocab indices."""
        samples = self.model.sample(n=4, seq_len=FAST_MAX_LEN, device=self.device)
        self.assertTrue((samples >= 0).all())
        self.assertTrue((samples < self.vocab.size).all())

    def test_sample_finite(self):
        samples = self.model.sample(n=4, seq_len=FAST_MAX_LEN, device=self.device)
        self.assertTrue(torch.isfinite(samples.float()).all())


# ── Training convergence (mini smoke test) ────────────────────────────────────

class TestD3PMTraining(unittest.TestCase):

    def test_loss_decreases(self):
        vocab = Vocabulary.from_sequences(SEQS)
        model = AbsorbingD3PM(
            vocab_size=vocab.size,
            mask_idx=vocab.mask_idx,
            pad_idx=vocab.pad_idx,
            T=FAST_T,
            max_len=FAST_MAX_LEN,
            d_model=32,
            nhead=2,
            num_layers=2,
            dim_ff=64,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        ds = OpcodeDataset(SEQS, vocab, max_len=FAST_MAX_LEN)
        tokens = torch.stack([ds[i]["tokens"] for i in range(len(ds))])
        pad_mask = torch.stack([ds[i]["pad_mask"] for i in range(len(ds))])

        EPOCHS = 20
        losses = []
        for _ in range(EPOCHS):
            loss = model(tokens, pad_mask)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())

        first_half = np.mean(losses[: EPOCHS // 2])
        second_half = np.mean(losses[EPOCHS // 2 :])
        self.assertLess(second_half, first_half, "Loss should decrease over training")


# ── Sequence evaluation tests ─────────────────────────────────────────────────

class TestSequenceEvaluation(unittest.TestCase):

    def setUp(self):
        self.real = _make_seqs(20)
        self.synthetic = _make_seqs(20)

    def test_kl_nonneg(self):
        kl = opcode_freq_kl(self.real, self.synthetic)
        self.assertGreaterEqual(kl["kl_real_vs_synth"], 0)
        self.assertGreaterEqual(kl["kl_synth_vs_real"], 0)

    def test_kl_identical_is_zero(self):
        kl = opcode_freq_kl(self.real, self.real)
        self.assertAlmostEqual(kl["symmetric_kl"], 0.0, places=3)

    def test_ngram_f1_in_range(self):
        result = ngram_overlap(self.real, self.synthetic)
        for n in (1, 2, 3):
            f1 = result[f"{n}gram_f1"]
            self.assertGreaterEqual(f1, 0.0)
            self.assertLessEqual(f1, 1.0)

    def test_ngram_identical_is_one(self):
        result = ngram_overlap(self.real, self.real)
        for n in (1, 2, 3):
            self.assertAlmostEqual(result[f"{n}gram_f1"], 1.0, places=5)

    def test_edit_distance_keys(self):
        result = edit_distance_stats(self.real, self.synthetic, n_pairs=20)
        self.assertIn("real_vs_synthetic", result)
        self.assertIn("real_vs_real", result)
        self.assertIn("synth_vs_synth", result)

    def test_edit_distance_nonneg(self):
        result = edit_distance_stats(self.real, self.synthetic, n_pairs=20)
        for key in ("real_vs_synthetic", "real_vs_real", "synth_vs_synth"):
            self.assertGreaterEqual(result[key]["mean"], 0.0)


if __name__ == "__main__":
    import time
    start = time.time()
    unittest.main(verbosity=2, exit=False)
    print(f"\nTotal time: {time.time() - start:.1f}s")
