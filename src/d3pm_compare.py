"""
Side-by-side: D3PM (chunked, in fresh W2V space) vs continuous DDPM
(numbers pulled from existing eval_results/full_report.json).

Both methods report metric "deviation from each method's own real reference",
so the comparison is fair across W2V coordinate systems.

Usage:
    python src/d3pm_compare.py --family zeroaccess
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import binary_classification, similarity_scores


def _eval_d3pm(real: np.ndarray, synth: np.ndarray) -> dict:
    return {
        "binary_classification": binary_classification(real, synth),
        "cosine_similarity":     similarity_scores(real, synth),
    }


def _continuous_from_report(report_path: Path, family: str) -> dict | None:
    """Pull the continuous DDPM numbers for `family` from a previous eval report."""
    if not report_path.exists():
        return None
    blob = json.loads(report_path.read_text())
    pf = blob.get("per_family", {})
    if family in pf:
        return pf[family]
    if set(blob.keys()) >= {"binary_classification", "cosine_similarity"}:
        return blob   # single-family report
    return None


def _print_row(label: str, d3pm_val: float | str, cont_val: float | str, target: str) -> None:
    print(f"  {label:<26} | {d3pm_val:<10} | {cont_val:<10} | {target}")


def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family",      default="zeroaccess")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--synthetic",   type=Path, default=Path("synthetic"))
    parser.add_argument("--out",         type=Path, default=Path("eval_results"))
    parser.add_argument("--continuous-report",
                        type=Path, default=Path("eval_results/full_report.json"),
                        help="Existing continuous DDPM eval report to pull numbers from.")
    args = parser.parse_args()

    real_path = args.checkpoints / f"{args.family}_d3pm_real_embeddings.npy"
    d3pm_path = args.synthetic   / f"{args.family}_d3pm_synthetic.npy"
    for p in (real_path, d3pm_path):
        if not p.exists():
            raise SystemExit(f"Missing required file: {p}")

    real = np.load(real_path)
    d3pm = np.load(d3pm_path)
    print(f"family={args.family}")
    print(f"  real ref  : {real.shape}  → {real_path}")
    print(f"  D3PM synth: {d3pm.shape}  → {d3pm_path}")

    # ── D3PM (computed now in the fresh W2V space) ──────────────────────────
    d3pm_metrics = _eval_d3pm(real, d3pm)

    # ── continuous DDPM (read from existing report) ─────────────────────────
    cont_metrics = _continuous_from_report(args.continuous_report, args.family)
    if cont_metrics is None:
        print(f"\n[warn] No continuous DDPM entry for '{args.family}' in "
              f"{args.continuous_report}; only D3PM numbers will be shown.")
        cont_metrics = {"binary_classification": {}, "cosine_similarity": {}}

    # ── side-by-side print ──────────────────────────────────────────────────
    print(f"\n── {args.family}  D3PM (this run)  vs  continuous DDPM (existing eval) ──\n")
    print(f"  {'metric':<26} | {'D3PM':<10} | {'continuous':<10} | target")
    print(f"  {'-'*26}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    d_clf = d3pm_metrics["binary_classification"]
    c_clf = cont_metrics.get("binary_classification", {})
    for clf_name in ("RF", "SVM", "MLP"):
        _print_row(f"binary F1 ({clf_name})",
                   _fmt(d_clf.get(clf_name, "—")),
                   _fmt(c_clf.get(clf_name, "—")),
                   "≈ 0.5")

    d_sim = d3pm_metrics["cosine_similarity"]
    c_sim = cont_metrics.get("cosine_similarity", {})
    _print_row("cos real↔synth median",
               _fmt(d_sim.get("real_vs_synthetic_median", "—")),
               _fmt(c_sim.get("real_vs_synthetic_median", "—")),
               "≈ real↔real")
    _print_row("cos real↔real  median",
               _fmt(d_sim.get("baseline_real_vs_real_median", "—")),
               _fmt(c_sim.get("baseline_real_vs_real_median", "—")),
               "—")
    _print_row("cos deviation",
               _fmt(d_sim.get("deviation", "—")),
               _fmt(c_sim.get("deviation", "—")),
               "≤ 0.035")

    # ── persist a small JSON ────────────────────────────────────────────────
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{args.family}_d3pm_vs_continuous.json"
    out.write_text(json.dumps({
        "family": args.family,
        "d3pm_real_embeddings": str(real_path),
        "d3pm_synthetic":       str(d3pm_path),
        "continuous_source":    str(args.continuous_report),
        "d3pm":       d3pm_metrics,
        "continuous": cont_metrics,
    }, indent=2))
    print(f"\nReport saved → {out}")


if __name__ == "__main__":
    main()
