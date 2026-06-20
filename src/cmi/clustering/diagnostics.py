"""
Clustering Diagnostics
=======================
Tools for evaluating cluster quality and stability:
- Elbow plot (inertia vs K)
- Silhouette analysis (per-K silhouette scores)
- Cluster stability score (centroid drift over time)
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from cmi.clustering.temporal import ClusterTimeline

logger = logging.getLogger(__name__)

# Matplotlib style
sns.set_theme(style="darkgrid", palette="viridis")


def elbow_plot(
    fused_matrix: np.ndarray,
    k_range: range = range(2, 12),
    random_state: int = 42,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """
    Plot inertia (within-cluster sum of squares) vs K for elbow method.

    Returns the figure if no axes were provided.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(fused_matrix)
        inertias.append(kmeans.inertia_)
        logger.debug("K=%d, inertia=%.2f", k, kmeans.inertia_)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(list(k_range), inertias, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.set_title("Elbow Method — Optimal K Selection", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    return fig


def silhouette_analysis(
    fused_matrix: np.ndarray,
    k_range: range = range(2, 12),
    random_state: int = 42,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure | None, dict[int, float]]:
    """
    Compute and plot silhouette scores for each K.

    Returns (figure_or_None, {K: score} dict).
    """
    scores: dict[int, float] = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(fused_matrix)
        score = silhouette_score(fused_matrix, labels)
        scores[k] = score
        logger.debug("K=%d, silhouette=%.4f", k, score)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(list(scores.keys()), list(scores.values()), color=sns.color_palette("viridis", len(scores)))
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Silhouette Analysis — Cluster Cohesion", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    best_k = max(scores, key=scores.get)  # type: ignore[arg-type]
    logger.info("Best K by silhouette: %d (score=%.4f)", best_k, scores[best_k])

    return fig, scores


def cluster_stability_score(timeline: ClusterTimeline) -> pd.DataFrame:
    """
    Measure how stable each cluster's identity is over time
    by computing the mean centroid drift velocity per cluster.

    Low drift = stable identity. High drift = morphing archetype.
    """
    import pandas as pd

    if not timeline.drift_velocities:
        logger.warning("No drift velocities available — need ≥2 months of data")
        return pd.DataFrame()

    rows = []
    for month, drifts in timeline.drift_velocities.items():
        for cluster_id, drift_val in enumerate(drifts):
            rows.append({
                "month": month,
                "cluster_id": cluster_id,
                "drift_velocity": drift_val,
            })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("cluster_id")["drift_velocity"]
        .agg(["mean", "std", "max"])
        .round(4)
    )
    summary.columns = ["mean_drift", "std_drift", "max_drift"]
    summary = summary.sort_values("mean_drift", ascending=False)

    logger.info("Cluster stability summary:\n%s", summary)
    return summary


def find_optimal_k(
    fused_matrix: np.ndarray,
    k_range: tuple[int, int] = (4, 30),
    random_state: int = 42,
) -> int:
    """
    Two-step process to find the optimal cluster number:
    1. Use yellowbrick KElbowVisualizer to auto-detect the best elbow K.
    2. Search locally around that elbow (elbow +/- 2) by silhouette score.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from yellowbrick.cluster import KElbowVisualizer

    logger.info("Step 1: Running yellowbrick KElbowVisualizer to auto-detect elbow value...")
    model = KMeans(random_state=random_state, n_init=10)
    visualizer = KElbowVisualizer(model, k=k_range, timings=False, locate_elbow=True, force_model=True)
    
    # We catch/suppress warnings for matplotlib plotting in non-interactive environments
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        visualizer.fit(fused_matrix)

    elbow_k = visualizer.elbow_value_

    if elbow_k is None:
        logger.warning("Yellowbrick failed to detect an elbow value. Defaulting to K=6.")
        elbow_k = 6
    else:
        logger.info("Yellowbrick auto-detected elbow K = %d", elbow_k)

    # Step 2: Search locally around the elbow (elbow_k +/- 2) by silhouette score
    search_min = max(k_range[0], elbow_k - 2)
    search_max = min(k_range[1], elbow_k + 2)
    local_range = range(search_min, search_max + 1)

    logger.info("Step 2: Searching local range k=%s by silhouette score...", list(local_range))
    best_score = -1.0
    best_k = elbow_k

    for k in local_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(fused_matrix)
        score = silhouette_score(fused_matrix, labels)
        logger.info("Local search k=%d, silhouette_score=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k

    logger.info("Optimal cluster count selected: K = %d (silhouette_score=%.4f)", best_k, best_score)
    return best_k
