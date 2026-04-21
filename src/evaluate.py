"""
Evaluation metrics from the paper:
  1. Binary classification (real vs synthetic) — F1 score targeting ~0.5
  2. t-SNE visualization
  3. Cosine similarity (real vs synthetic cross-comparison)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def binary_classification(
    real: np.ndarray, synthetic: np.ndarray
) -> dict[str, float]:
    """
    Train on synthetic, test on real (and vice versa).
    Returns F1 scores for RF, SVM, MLP.
    """
    X = np.concatenate([real, synthetic])
    y = np.array([1] * len(real) + [0] * len(synthetic))

    n_train = len(X) // 2
    X_train, y_train = synthetic, np.zeros(len(synthetic))
    X_test, y_test = real, np.ones(len(real))

    results = {}
    for name, clf in [
        ("RF", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("SVM", SVC(kernel="rbf", random_state=42)),
        ("MLP", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)),
    ]:
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        results[name] = f1_score(y_test, pred, zero_division=0)
    return results


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Full cross-comparison cosine similarity between rows of A and B."""
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A_norm @ B_norm.T


def similarity_scores(real: np.ndarray, synthetic: np.ndarray) -> dict[str, float]:
    """Median cosine similarity: real vs synthetic, and baseline (real vs real)."""
    rs_sim = cosine_similarity_matrix(real, synthetic)
    rr_sim = cosine_similarity_matrix(real, real)
    # exclude diagonal for baseline
    mask = ~np.eye(len(real), dtype=bool)
    return {
        "real_vs_synthetic_median": float(np.median(rs_sim)),
        "baseline_real_vs_real_median": float(np.median(rr_sim[mask])),
        "deviation": float(abs(np.median(rs_sim) - np.median(rr_sim[mask]))),
    }


def plot_tsne(
    real: np.ndarray,
    synthetic: np.ndarray,
    family: str,
    save_path: str | None = None,
) -> None:
    combined = np.concatenate([real, synthetic])
    labels = ["real"] * len(real) + ["synthetic"] * len(synthetic)

    tsne = TSNE(n_components=2, perplexity=min(30, len(combined) - 1), random_state=42)
    reduced = tsne.fit_transform(combined)

    colors = {"real": "green", "synthetic": "orange"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for label in ["real", "synthetic"]:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(reduced[idx, 0], reduced[idx, 1], label=label,
                   c=colors[label], alpha=0.6, s=20)
    ax.set_title(f"t-SNE: {family}")
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def evaluate_family(
    family: str,
    real: np.ndarray,
    synthetic: np.ndarray,
    tsne_dir: str | None = None,
) -> dict:
    print(f"\n[{family}] real={len(real)}, synthetic={len(synthetic)}")

    clf_scores = binary_classification(real, synthetic)
    print(f"  Binary classification F1 (goal ~0.5): {clf_scores}")

    sim = similarity_scores(real, synthetic)
    print(f"  Cosine similarity: {sim}")

    if tsne_dir:
        import os
        os.makedirs(tsne_dir, exist_ok=True)
        plot_tsne(real, synthetic, family, save_path=f"{tsne_dir}/{family}_tsne.png")
        print(f"  t-SNE saved to {tsne_dir}/{family}_tsne.png")

    return {"binary_classification": clf_scores, "cosine_similarity": sim}
