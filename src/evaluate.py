"""
Evaluation metrics from the paper (all 6):
  1. Binary classification  (real vs synthetic, per family) — F1 goal ~0.5
  2. t-SNE visualization    (per family)
  3. Synthetic-only training — train on synthetic, test on real (multiclass)
  4. Threshold-based augmentation — augment low-F1 families, compare multiclass F1
  5. Fidelity score          — train on real, test on synthetic (multiclass)
  6. Cosine similarity       (per family)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ── helpers ─────────────────────────────────────────────────────────────────

def _classifiers():
    return [
        ("RF",  RandomForestClassifier(n_estimators=100, random_state=42)),
        ("SVM", SVC(kernel="rbf", random_state=42)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                              random_state=42)),
    ]


def _stack(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Stack family dict into (X, y_int, family_names)."""
    families = sorted(d.keys())
    X = np.concatenate([d[f] for f in families])
    y = np.concatenate([np.full(len(d[f]), i) for i, f in enumerate(families)])
    return X, y.astype(int), families


# ── metric 1: binary classification (per family) ────────────────────────────

def binary_classification(real: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    """
    Stratified 70/30 split of (real=1, synthetic=0).
    Ideal F1 ≈ 0.5 means the classifier cannot distinguish real from synthetic.
    """
    X = np.concatenate([real, synthetic])
    y = np.array([1] * len(real) + [0] * len(synthetic))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=42)
    results = {}
    for name, clf in _classifiers():
        clf.fit(X_tr, y_tr)
        results[name] = f1_score(y_te, clf.predict(X_te), zero_division=0)
    return results


# ── metric 2: t-SNE ─────────────────────────────────────────────────────────

def plot_tsne(real: np.ndarray, synthetic: np.ndarray, family: str,
              save_path: str | None = None) -> None:
    combined = np.concatenate([real, synthetic])
    labels = ["real"] * len(real) + ["synthetic"] * len(synthetic)
    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) - 1),
                random_state=42)
    reduced = tsne.fit_transform(combined)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in [("real", "green"), ("synthetic", "orange")]:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(reduced[idx, 0], reduced[idx, 1], label=label,
                   c=color, alpha=0.6, s=20)
    ax.set_title(f"t-SNE: {family}")
    ax.legend()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


# ── metric 3: synthetic-only training ───────────────────────────────────────

def synthetic_only_training(real_dict: dict[str, np.ndarray],
                             synthetic_dict: dict[str, np.ndarray]) -> dict:
    """
    Train multiclass classifier exclusively on synthetic data,
    test on real data. Mirrors Table 8 from the paper.
    Ideal: F1 matches the real-data baseline (~0.87).
    """
    families = sorted(set(real_dict) & set(synthetic_dict))
    X_synth = np.concatenate([synthetic_dict[f] for f in families])
    y_synth = np.concatenate([np.full(len(synthetic_dict[f]), i)
                               for i, f in enumerate(families)])
    X_real  = np.concatenate([real_dict[f] for f in families])
    y_real  = np.concatenate([np.full(len(real_dict[f]), i)
                               for i, f in enumerate(families)])
    results = {}
    for name, clf in _classifiers():
        clf.fit(X_synth, y_synth)
        pred = clf.predict(X_real)
        results[name] = float(f1_score(y_real, pred,
                                        average="weighted", zero_division=0))
    return {"families": families, "scores": results}


# ── metric 4: threshold-based augmentation ──────────────────────────────────

def threshold_augmentation(real_dict: dict[str, np.ndarray],
                            synthetic_dict: dict[str, np.ndarray],
                            threshold: float = 0.95) -> dict:
    """
    Replicates the paper's threshold-based augmentation (Table 9).

    Steps:
      1. Stratified 70/30 split on real data.
      2. Baseline: train RF on real_train, evaluate per-family F1 on real_test.
      3. Families with per-family F1 < threshold get their training sets
         augmented with synthetic samples (equal to real_train count for that
         family, capped by available synthetic data).
      4. Retrain and report before/after overall weighted F1.
    """
    families = sorted(set(real_dict) & set(synthetic_dict))
    X_all = np.concatenate([real_dict[f] for f in families])
    y_all = np.concatenate([np.full(len(real_dict[f]), i)
                             for i, f in enumerate(families)])

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3,
                                               stratify=y_all, random_state=42)

    baseline_scores = {}
    augmented_scores = {}

    for clf_name, clf_base in _classifiers():
        # ── baseline ──────────────────────────────────────────────────────
        clf_base.fit(X_tr, y_tr)
        pred_base = clf_base.predict(X_te)
        baseline_scores[clf_name] = float(
            f1_score(y_te, pred_base, average="weighted", zero_division=0))

        per_class = f1_score(y_te, pred_base, average=None, zero_division=0)

        # ── find families needing augmentation ────────────────────────────
        X_aug, y_aug = X_tr.copy(), y_tr.copy()
        for i, fam in enumerate(families):
            if per_class[i] < threshold:
                synth = synthetic_dict[fam]
                # take as many synthetic samples as we have real train samples
                n_real_train = int((y_tr == i).sum())
                n_to_add = min(n_real_train, len(synth))
                idx = np.random.default_rng(42).integers(0, len(synth), n_to_add)
                X_aug = np.concatenate([X_aug, synth[idx]])
                y_aug = np.concatenate([y_aug, np.full(n_to_add, i)])

        # ── augmented ─────────────────────────────────────────────────────
        from sklearn.base import clone
        clf_aug = clone(clf_base)
        clf_aug.fit(X_aug, y_aug)
        pred_aug = clf_aug.predict(X_te)
        augmented_scores[clf_name] = float(
            f1_score(y_te, pred_aug, average="weighted", zero_division=0))

    augmented_families = [
        fam for i, fam in enumerate(families)
        if f1_score(y_te, clf_base.predict(X_te), average=None,
                    zero_division=0)[i] < threshold
    ]

    return {
        "families": families,
        "threshold": threshold,
        "augmented_families": augmented_families,
        "baseline_weighted_f1":  baseline_scores,
        "augmented_weighted_f1": augmented_scores,
    }


# ── metric 5: fidelity score ─────────────────────────────────────────────────

def fidelity_score(real_dict: dict[str, np.ndarray],
                   synthetic_dict: dict[str, np.ndarray]) -> dict:
    """
    Train multiclass classifier on real data, test on synthetic data.
    Mirrors Table 10 from the paper.
    High F1 means synthetic samples look like the correct family.
    """
    families = sorted(set(real_dict) & set(synthetic_dict))
    X_real  = np.concatenate([real_dict[f] for f in families])
    y_real  = np.concatenate([np.full(len(real_dict[f]), i)
                               for i, f in enumerate(families)])
    X_synth = np.concatenate([synthetic_dict[f] for f in families])
    y_synth = np.concatenate([np.full(len(synthetic_dict[f]), i)
                               for i, f in enumerate(families)])
    results = {}
    for name, clf in _classifiers():
        clf.fit(X_real, y_real)
        pred = clf.predict(X_synth)
        results[name] = float(f1_score(y_synth, pred,
                                        average="weighted", zero_division=0))
    return {"families": families, "scores": results}


# ── metric 6: cosine similarity (per family) ────────────────────────────────

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def similarity_scores(real: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    """Median cosine similarity: real vs synthetic vs real-vs-real baseline."""
    rs_sim = cosine_similarity_matrix(real, synthetic)
    rr_sim = cosine_similarity_matrix(real, real)
    mask = ~np.eye(len(real), dtype=bool)
    return {
        "real_vs_synthetic_median":   float(np.median(rs_sim)),
        "baseline_real_vs_real_median": float(np.median(rr_sim[mask])),
        "deviation": float(abs(np.median(rs_sim) - np.median(rr_sim[mask]))),
    }


# ── per-family evaluation (metrics 1, 2, 6) ──────────────────────────────────

def evaluate_family(family: str, real: np.ndarray, synthetic: np.ndarray,
                    tsne_dir: str | None = None) -> dict:
    print(f"\n[{family}] real={len(real)}, synthetic={len(synthetic)}")

    clf_scores = binary_classification(real, synthetic)
    print(f"  1. Binary classification F1 (goal ~0.5): {clf_scores}")

    sim = similarity_scores(real, synthetic)
    print(f"  6. Cosine similarity deviation: {sim['deviation']:.4f}  "
          f"(paper threshold ≤0.035 for best families)")

    if tsne_dir:
        save_path = f"{tsne_dir}/{family}_tsne.png"
        plot_tsne(real, synthetic, family, save_path=save_path)
        print(f"  2. t-SNE saved → {save_path}")

    return {"binary_classification": clf_scores, "cosine_similarity": sim}


# ── full multi-family evaluation ─────────────────────────────────────────────

def run_all_metrics(real_dict: dict[str, np.ndarray],
                    synthetic_dict: dict[str, np.ndarray],
                    tsne_dir: str | None = None,
                    threshold: float = 0.95) -> dict:
    """Run all 6 paper metrics for a set of families."""
    report: dict = {"per_family": {}}

    # metrics 1, 2, 6 — per family
    for fam in sorted(real_dict):
        if fam not in synthetic_dict:
            print(f"  [skip {fam}] no synthetic data found")
            continue
        report["per_family"][fam] = evaluate_family(
            fam, real_dict[fam], synthetic_dict[fam], tsne_dir=tsne_dir)

    shared = sorted(set(real_dict) & set(synthetic_dict))
    if len(shared) >= 2:
        print(f"\n── Multiclass metrics (families: {shared}) ──")

        # metric 3
        print("\n  3. Synthetic-only training (train synth / test real):")
        m3 = synthetic_only_training(real_dict, synthetic_dict)
        print(f"     {m3['scores']}")
        report["synthetic_only_training"] = m3

        # metric 4
        print(f"\n  4. Threshold-based augmentation (t={threshold}):")
        m4 = threshold_augmentation(real_dict, synthetic_dict, threshold)
        print(f"     Augmented families: {m4['augmented_families']}")
        print(f"     Baseline  F1 (RF): {m4['baseline_weighted_f1']['RF']:.4f}")
        print(f"     Augmented F1 (RF): {m4['augmented_weighted_f1']['RF']:.4f}")
        report["threshold_augmentation"] = m4

        # metric 5
        print("\n  5. Fidelity score (train real / test synth):")
        m5 = fidelity_score(real_dict, synthetic_dict)
        print(f"     {m5['scores']}")
        report["fidelity_score"] = m5

    else:
        print(f"\n  [metrics 3–5 skipped] need ≥2 families with synthetic data "
              f"(found: {shared})")

    return report


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diffusion model quality (all 6 paper metrics).")
    parser.add_argument("--families", nargs="+", default=None,
                        help="families to evaluate (default: all in checkpoints/)")
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--synthetic",   type=Path, default=Path("synthetic"))
    parser.add_argument("--out",         type=Path, default=Path("eval_results"))
    parser.add_argument("--threshold",   type=float, default=0.95)
    parser.add_argument("--variant",     choices=["continuous", "d3pm"],
                        default="continuous",
                        help="continuous: read {f}_embeddings.npy + {f}_synthetic.npy "
                             "(continuous DDPM W2V). d3pm: read {f}_d3pm_real_embeddings.npy + "
                             "{f}_d3pm_synthetic.npy (D3PM W2V — keeps real and synth in the "
                             "same coordinate system).")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.variant == "d3pm":
        real_suffix = "_d3pm_real_embeddings.npy"
        synth_suffix = "_d3pm_synthetic.npy"
    else:
        real_suffix = "_embeddings.npy"
        synth_suffix = "_synthetic.npy"

    # discover families
    if args.families:
        families = args.families
    else:
        families = [p.name.removesuffix(real_suffix)
                    for p in args.checkpoints.glob(f"*{real_suffix}")]
        families = [f for f in families
                    if (args.synthetic / f"{f}{synth_suffix}").exists()]

    if not families:
        raise SystemExit(
            f"No families found for variant '{args.variant}'. "
            f"Looked for {real_suffix} in {args.checkpoints} and "
            f"{synth_suffix} in {args.synthetic}.")

    print(f"Evaluating families ({args.variant}): {families}")

    real_dict  = {f: np.load(args.checkpoints / f"{f}{real_suffix}")
                  for f in families}
    synth_dict = {f: np.load(args.synthetic / f"{f}{synth_suffix}")
                  for f in families
                  if (args.synthetic / f"{f}{synth_suffix}").exists()}

    report = run_all_metrics(real_dict, synth_dict,
                              tsne_dir=str(args.out),
                              threshold=args.threshold)

    out_path = args.out / "full_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved → {out_path}")


if __name__ == "__main__":
    main()
