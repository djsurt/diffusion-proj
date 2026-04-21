"""
Quick end-to-end test of the full diffusion pipeline on a small subset.

Uses 3 small families (harebot, smarthdd, securityshield) with reduced
epochs and T so the suite finishes in ~5 minutes.

Run: python tests/test_pipeline.py
"""

import sys
import os
import time
import unittest
from pathlib import Path

import numpy as np
import torch

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_family_opcodes
from embeddings import build_family_embeddings, scale_to_range
from diffusion import MalwareDiffusion

MALICIA = Path(__file__).parent.parent / "malicia"
# Small families that load fast
TEST_FAMILIES = ["harebot", "smarthdd", "securityshield"]
FAST_EPOCHS = 15
FAST_T = 200
EMBED_DIM = 104
MAX_FILES = 20  # cap per family for speed


class TestDataLoader(unittest.TestCase):
    def test_loads_specified_families(self):
        corpus = load_family_opcodes(MALICIA, families=TEST_FAMILIES, max_files_per_family=5)
        self.assertGreater(len(corpus), 0)
        for family in corpus:
            self.assertIn(family, TEST_FAMILIES)

    def test_each_file_has_opcodes(self):
        corpus = load_family_opcodes(MALICIA, families=["harebot"], max_files_per_family=5)
        for seqs in corpus.values():
            for seq in seqs:
                self.assertIsInstance(seq, list)
                self.assertGreater(len(seq), 0)

    def test_max_files_cap(self):
        corpus = load_family_opcodes(MALICIA, families=["harebot"], max_files_per_family=3)
        for seqs in corpus.values():
            self.assertLessEqual(len(seqs), 3)


class TestEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = load_family_opcodes(
            MALICIA, families=TEST_FAMILIES, max_files_per_family=MAX_FILES
        )

    def test_embedding_shape(self):
        embeddings, _ = build_family_embeddings(self.corpus, dim=EMBED_DIM)
        for family, emb in embeddings.items():
            n = len(self.corpus[family])
            self.assertEqual(emb.shape, (n, EMBED_DIM), f"{family}: wrong shape")

    def test_embedding_range(self):
        embeddings, _ = build_family_embeddings(self.corpus, dim=EMBED_DIM)
        for family, emb in embeddings.items():
            self.assertLessEqual(emb.max(), 1.0 + 1e-5, f"{family}: max > 1")
            self.assertGreaterEqual(emb.min(), -1.0 - 1e-5, f"{family}: min < -1")

    def test_separate_w2v_per_family(self):
        _, models = build_family_embeddings(self.corpus, dim=EMBED_DIM)
        model_ids = [id(m) for m in models.values()]
        self.assertEqual(len(model_ids), len(set(model_ids)), "Word2Vec models must be distinct per family")

    def test_scale_to_range(self):
        arr = np.random.randn(50, 10).astype(np.float32)
        scaled = scale_to_range(arr)
        self.assertAlmostEqual(scaled.min(), -1.0, places=5)
        self.assertAlmostEqual(scaled.max(),  1.0, places=5)


class TestDiffusionModel(unittest.TestCase):
    def setUp(self):
        self.model = MalwareDiffusion(embed_dim=EMBED_DIM, T=FAST_T)
        self.device = torch.device("cpu")

    def test_forward_loss_shape(self):
        x = torch.randn(8, 1, EMBED_DIM)
        loss = self.model(x)
        self.assertEqual(loss.shape, torch.Size([]))  # scalar
        self.assertFalse(torch.isnan(loss))

    def test_q_sample_shape(self):
        x0 = torch.randn(4, 1, EMBED_DIM)
        t = torch.randint(0, FAST_T // 2, (4,))
        xt = self.model.q_sample(x0, t)
        self.assertEqual(xt.shape, x0.shape)

    def test_sample_shape(self):
        samples = self.model.sample(5, self.device)
        self.assertEqual(samples.shape, (5, EMBED_DIM))

    def test_t_half_constraint(self):
        self.assertEqual(self.model.T_half, FAST_T // 2)

    def test_sample_in_reasonable_range(self):
        samples = self.model.sample(10, self.device)
        # untrained model samples may be noisy, but should not explode
        self.assertTrue(torch.isfinite(samples).all())


class TestEndToEnd(unittest.TestCase):
    """
    Light training run (FAST_EPOCHS) on a single small family.
    Verifies loss decreases and generation produces valid output.
    """

    def test_training_loss_decreases(self):
        corpus = load_family_opcodes(MALICIA, families=["harebot"], max_files_per_family=MAX_FILES)
        embeddings, _ = build_family_embeddings(corpus, dim=EMBED_DIM)
        emb = embeddings["harebot"]

        X = torch.tensor(emb, dtype=torch.float32).unsqueeze(1)
        model = MalwareDiffusion(embed_dim=EMBED_DIM, T=FAST_T)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)

        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(X), batch_size=min(16, len(X)), shuffle=True)

        losses = []
        for _ in range(FAST_EPOCHS):
            epoch_loss = 0.0
            for (batch,) in loader:
                loss = model(batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss / len(loader))

        first_third = np.mean(losses[: FAST_EPOCHS // 3])
        last_third = np.mean(losses[-FAST_EPOCHS // 3 :])
        self.assertLess(last_third, first_third, "Loss should decrease over training")

    def test_generated_samples_finite(self):
        corpus = load_family_opcodes(MALICIA, families=["harebot"], max_files_per_family=MAX_FILES)
        embeddings, _ = build_family_embeddings(corpus, dim=EMBED_DIM)
        emb = embeddings["harebot"]

        X = torch.tensor(emb, dtype=torch.float32).unsqueeze(1)
        model = MalwareDiffusion(embed_dim=EMBED_DIM, T=FAST_T)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)

        for _ in range(5):  # minimal training just to ensure model is runnable
            loss = model(X)
            opt.zero_grad(); loss.backward(); opt.step()

        samples = model.sample(10, torch.device("cpu"))
        self.assertEqual(samples.shape, (10, EMBED_DIM))
        self.assertTrue(torch.isfinite(samples).all())


if __name__ == "__main__":
    start = time.time()
    unittest.main(verbosity=2, exit=False)
    print(f"\nTotal time: {time.time() - start:.1f}s")
