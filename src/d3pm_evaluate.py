"""
Sequence-level evaluation metrics for D3PM generated opcode sequences.

Metrics:
  1. Opcode-frequency KL divergence  — real vs synthetic unigram distributions
  2. N-gram precision / recall / F1   — 1, 2, 3 grams
  3. Edit-distance distribution       — sampled pairwise Levenshtein distances

All functions accept lists of opcode sequences (list[list[str]]).
"""

import random
from collections import Counter
from typing import Sequence

import numpy as np


# ── 1. Opcode frequency KL divergence ───────────────────────────────────────


def opcode_freq_kl(
    real: list[list[str]],
    synthetic: list[list[str]],
    eps: float = 1e-10,
) -> dict[str, float]:
    """
    Compare unigram opcode frequency distributions via KL divergence.

    Returns KL(real || synthetic) and KL(synthetic || real).
    Lower is better (0 = identical distribution).
    """
    real_counts = Counter(op for seq in real for op in seq)
    synth_counts = Counter(op for seq in synthetic for op in seq)

    vocab = sorted(set(real_counts) | set(synth_counts))
    real_total = sum(real_counts.values()) or 1
    synth_total = sum(synth_counts.values()) or 1

    p = np.array([real_counts.get(v, 0) / real_total for v in vocab], dtype=float)
    q = np.array([synth_counts.get(v, 0) / synth_total for v in vocab], dtype=float)

    p = p + eps; p /= p.sum()
    q = q + eps; q /= q.sum()

    kl_rs = float(np.sum(p * np.log(p / q)))
    kl_sr = float(np.sum(q * np.log(q / p)))

    return {
        "kl_real_vs_synth": kl_rs,
        "kl_synth_vs_real": kl_sr,
        "symmetric_kl": (kl_rs + kl_sr) / 2,
        "vocab_real": len(real_counts),
        "vocab_synth": len(synth_counts),
        "vocab_overlap": len(set(real_counts) & set(synth_counts)),
    }


# ── 2. N-gram overlap ────────────────────────────────────────────────────────


def _ngrams(seq: list[str], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def ngram_overlap(
    real: list[list[str]],
    synthetic: list[list[str]],
    ns: Sequence[int] = (1, 2, 3),
) -> dict[str, float]:
    """
    Corpus-level n-gram precision, recall, F1 between real and synthetic corpora.

    Precision = fraction of synthetic n-grams that appear in real.
    Recall    = fraction of real n-grams that appear in synthetic.
    """
    results: dict[str, float] = {}
    for n in ns:
        real_ngrams = sum((_ngrams(s, n) for s in real), Counter())
        synth_ngrams = sum((_ngrams(s, n) for s in synthetic), Counter())

        real_types = set(real_ngrams)
        synth_types = set(synth_ngrams)

        overlap = real_types & synth_types

        precision = len(overlap) / max(len(synth_types), 1)
        recall = len(overlap) / max(len(real_types), 1)
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else 0.0)

        results[f"{n}gram_precision"] = float(precision)
        results[f"{n}gram_recall"] = float(recall)
        results[f"{n}gram_f1"] = float(f1)

    return results


# ── 3. Edit distance ─────────────────────────────────────────────────────────


def _levenshtein(a: list[str], b: list[str], max_dist: int = 200) -> int:
    """Token-level Levenshtein distance, capped at max_dist for speed."""
    a = a[:max_dist]
    b = b[:max_dist]
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def edit_distance_stats(
    real: list[list[str]],
    synthetic: list[list[str]],
    n_pairs: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """
    Sampled pairwise edit distances:
      - real vs synthetic  (quality)
      - real vs real       (baseline diversity)
      - synthetic vs synthetic (synthetic diversity)
    """
    rng = random.Random(seed)

    def _sample_pairs(A: list[list[str]], B: list[list[str]], k: int) -> list[int]:
        pairs = [(rng.choice(A), rng.choice(B)) for _ in range(k)]
        return [_levenshtein(a, b) for a, b in pairs]

    rs_dists = _sample_pairs(real, synthetic, n_pairs)
    rr_dists = _sample_pairs(real, real, n_pairs)
    ss_dists = _sample_pairs(synthetic, synthetic, n_pairs)

    def _stats(dists: list[int]) -> dict:
        arr = np.array(dists, dtype=float)
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "std": float(arr.std())}

    return {
        "real_vs_synthetic": _stats(rs_dists),
        "real_vs_real":      _stats(rr_dists),
        "synth_vs_synth":    _stats(ss_dists),
    }


# ── Full sequence evaluation ─────────────────────────────────────────────────


def evaluate_sequences(
    family: str,
    real: list[list[str]],
    synthetic: list[list[str]],
    n_edit_pairs: int = 200,
) -> dict:
    """
    Run all three sequence-level metrics and print a summary.
    """
    print(f"\n[{family}] Sequence-level evaluation")
    print(f"  real={len(real)} seqs, synthetic={len(synthetic)} seqs")

    freq_kl = opcode_freq_kl(real, synthetic)
    print(f"  Frequency KL (symmetric): {freq_kl['symmetric_kl']:.4f}")
    print(f"  Vocab overlap: {freq_kl['vocab_overlap']}/{freq_kl['vocab_real']} "
          f"real opcodes present in synthetic")

    ngrams = ngram_overlap(real, synthetic)
    for n in (1, 2, 3):
        print(f"  {n}-gram F1: {ngrams[f'{n}gram_f1']:.3f}  "
              f"(P={ngrams[f'{n}gram_precision']:.3f} R={ngrams[f'{n}gram_recall']:.3f})")

    edit = edit_distance_stats(real, synthetic, n_pairs=n_edit_pairs)
    rs = edit["real_vs_synthetic"]
    rr = edit["real_vs_real"]
    print(f"  Edit distance  real↔synth: {rs['mean']:.1f} ± {rs['std']:.1f}")
    print(f"  Edit distance  real↔real:  {rr['mean']:.1f} ± {rr['std']:.1f}")

    return {
        "family": family,
        "freq_kl": freq_kl,
        "ngram_overlap": ngrams,
        "edit_distance": edit,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json, sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_family_opcodes

    parser = argparse.ArgumentParser(
        description="Sequence-level evaluation of D3PM generated opcodes.")
    parser.add_argument("--family",   required=True)
    parser.add_argument("--malicia",  type=Path, default=Path("malicia"))
    parser.add_argument("--synth-dir", type=Path, default=Path("synthetic"),
                        help="Directory containing <family>_d3pm_sequences/")
    parser.add_argument("--out",      type=Path, default=Path("eval_results"))
    parser.add_argument("--n-pairs",  type=int, default=200)
    args = parser.parse_args()

    corpus = load_family_opcodes(args.malicia, families=[args.family])
    if args.family not in corpus:
        raise SystemExit(f"Family '{args.family}' not found in {args.malicia}")
    real_seqs = corpus[args.family]

    seq_dir = args.synth_dir / f"{args.family}_d3pm_sequences"
    if not seq_dir.exists():
        raise SystemExit(f"No synthetic sequences at {seq_dir}  — run d3pm_generate.py first")

    synth_seqs = [
        p.read_text().splitlines()
        for p in sorted(seq_dir.glob("seq_*.txt"))
    ]
    synth_seqs = [s for s in synth_seqs if s]

    report = evaluate_sequences(args.family, real_seqs, synth_seqs, args.n_pairs)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.family}_d3pm_seq_eval.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved → {out_path}")
