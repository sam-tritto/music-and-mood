"""
Temporal Clustering with Hungarian Alignment
==============================================
Clusters songs into "vibe archetypes" per monthly time window, then uses
the Hungarian algorithm (linear_sum_assignment) to maintain consistent
cluster identities across time.

The label-switching problem: K-Means assigns arbitrary labels each run.
"High-Energy Melancholic Pop" might be Cluster 2 in January but Cluster 5
in February. The Hungarian algorithm solves this by finding the optimal
1-to-1 mapping between old and new centroids based on Euclidean distance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from cmi.config import K_CLUSTERS

logger = logging.getLogger(__name__)


@dataclass
class ClusterTimeline:
    """Container for temporal clustering results."""

    # Per-track labels aligned to consistent cluster IDs
    labels: np.ndarray

    # Centroids per month: {year_month: np.ndarray of shape (K, D)}
    centroids_by_month: dict[str, np.ndarray] = field(default_factory=dict)

    # Volumetric shares: DataFrame with columns = cluster IDs, index = months
    volumetric_shares: pd.DataFrame | None = None

    # Centroid drift velocity between consecutive months
    drift_velocities: dict[str, np.ndarray] = field(default_factory=dict)

    # Month labels for each row (aligned with self.labels)
    month_labels: np.ndarray | None = None

    # The K used
    k: int = K_CLUSTERS


def align_clusters(
    centroids_t0: np.ndarray,
    centroids_t1: np.ndarray,
) -> dict[int, int]:
    """
    Use the Hungarian algorithm to find the optimal 1-to-1 mapping
    between cluster centroids at T0 and T1.

    Parameters
    ----------
    centroids_t0 : (K, D) centroid matrix from previous time window
    centroids_t1 : (K, D) centroid matrix from current time window

    Returns
    -------
    dict mapping T1 label → T0 label (for remapping)
    """
    cost_matrix = cdist(centroids_t0, centroids_t1, metric="euclidean")
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind = T0 indices, col_ind = T1 indices
    # We want: T1 label → T0 label
    return {int(col): int(row) for row, col in zip(row_ind, col_ind)}


def cluster_temporal_windows(
    fused_matrix: np.ndarray,
    months: np.ndarray | pd.Series,
    k: int = K_CLUSTERS,
    random_state: int = 42,
    streams: np.ndarray | pd.Series | None = None,
) -> ClusterTimeline:
    """
    Perform temporal clustering with Hungarian alignment.

    1. Segment data into monthly windows
    2. K-Means on first month → baseline centroids
    3. For each subsequent month: cluster independently, then remap labels
       via Hungarian algorithm to maintain identity continuity

    Parameters
    ----------
    fused_matrix : (N, D) scaled feature matrix
    months : array of period labels (e.g., pd.Period or str) per row
    k : number of clusters
    random_state : for K-Means reproducibility
    streams : optional stream counts for volumetric share weighting

    Returns
    -------
    ClusterTimeline with aligned labels, centroid trajectories, and shares
    """
    unique_months = sorted(pd.Series(months).unique())
    logger.info(
        "Clustering %d tracks across %d monthly windows (K=%d)",
        len(fused_matrix), len(unique_months), k,
    )

    all_labels = np.full(len(fused_matrix), -1, dtype=int)
    centroids_by_month: dict[str, np.ndarray] = {}
    drift_velocities: dict[str, np.ndarray] = {}
    prev_centroids: np.ndarray | None = None

    for i, month in enumerate(unique_months):
        mask = np.array(months == month)
        month_data = fused_matrix[mask]
        month_key = str(month)

        if len(month_data) < k:
            # Not enough data points — use fewer clusters or skip
            effective_k = max(2, len(month_data))
            logger.warning(
                "Month %s has only %d tracks (< K=%d). Using K=%d.",
                month_key, len(month_data), k, effective_k,
            )
        else:
            effective_k = k

        # Run K-Means
        kmeans = KMeans(
            n_clusters=effective_k,
            random_state=random_state,
            n_init=10,
            max_iter=300,
        )
        raw_labels = kmeans.fit_predict(month_data)
        current_centroids = kmeans.cluster_centers_

        if prev_centroids is not None and effective_k == k:
            # Hungarian alignment: remap current labels to match previous
            mapping = align_clusters(prev_centroids, current_centroids)
            aligned_labels = np.array([mapping.get(l, l) for l in raw_labels])

            # Reorder centroids to match aligned labels
            aligned_centroids = np.zeros_like(current_centroids)
            for new_label, old_label in mapping.items():
                aligned_centroids[old_label] = current_centroids[new_label]

            # Compute drift velocity
            drift = np.linalg.norm(
                aligned_centroids - prev_centroids, axis=1
            )
            drift_velocities[month_key] = drift

            current_centroids = aligned_centroids
            raw_labels = aligned_labels

        all_labels[mask] = raw_labels
        centroids_by_month[month_key] = current_centroids
        prev_centroids = current_centroids

    # Compute volumetric shares
    shares = _compute_volumetric_shares(all_labels, months, k, streams)

    timeline = ClusterTimeline(
        labels=all_labels,
        centroids_by_month=centroids_by_month,
        volumetric_shares=shares,
        drift_velocities=drift_velocities,
        month_labels=np.array(months),
        k=k,
    )

    logger.info("Temporal clustering complete: %d months tracked", len(unique_months))
    return timeline


def _compute_volumetric_shares(
    labels: np.ndarray,
    months: np.ndarray | pd.Series,
    k: int,
    streams: np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Compute the monthly volumetric share (%) of each cluster.

    If streams are provided, weight by stream count.
    Otherwise, weight by track count.
    """
    df = pd.DataFrame({
        "month": months,
        "cluster": labels,
        "weight": streams if streams is not None else 1,
    })

    # Pivot: rows = months, columns = cluster IDs
    pivot = df.pivot_table(
        values="weight",
        index="month",
        columns="cluster",
        aggfunc="sum",
        fill_value=0,
    )

    # Normalize to percentages
    shares = pivot.div(pivot.sum(axis=1), axis=0) * 100

    # Ensure all K clusters have columns
    for c in range(k):
        if c not in shares.columns:
            shares[c] = 0.0

    shares = shares[sorted(shares.columns)]
    shares.columns = [f"cluster_{c}" for c in shares.columns]

    return shares
