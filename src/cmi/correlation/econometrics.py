"""
Econometric Correlation Analysis
==================================
Maps cluster volumetric share trajectories against the Misery Index to
identify which musical "vibe archetypes" are reactive to economic stress.

Methods:
- Rolling Pearson & Spearman correlation
- Granger causality testing (does economic pain *precede* music shifts?)
- Cluster classification: reactive / stable / counter-cyclical
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

from cmi.clustering.temporal import ClusterTimeline

logger = logging.getLogger(__name__)


def compute_volumetric_shares(timeline: ClusterTimeline) -> pd.DataFrame:
    """
    Convenience wrapper — returns the timeline's volumetric shares DataFrame.
    Ensures month index is datetime-compatible for merging with FRED data.
    """
    if timeline.volumetric_shares is None:
        raise ValueError("ClusterTimeline has no volumetric shares computed")

    shares = timeline.volumetric_shares.copy()

    # Convert Period index to Timestamp if needed
    if hasattr(shares.index, "to_timestamp"):
        shares.index = shares.index.to_timestamp()

    return shares


def rolling_correlation(
    cluster_shares: pd.DataFrame,
    misery_index: pd.DataFrame,
    window: int = 6,
) -> pd.DataFrame:
    """
    Compute rolling Pearson and Spearman correlations between each cluster's
    volumetric share and the Misery Index.

    Parameters
    ----------
    cluster_shares : DataFrame, index=date, columns=cluster_0..cluster_K
    misery_index : DataFrame with 'date' and 'misery_index' columns
    window : rolling window size in months

    Returns
    -------
    DataFrame with rolling correlations per cluster
    """
    # Merge on date
    misery = misery_index.set_index("date")["misery_index"]
    merged = cluster_shares.join(misery, how="inner")

    if len(merged) < window:
        logger.warning(
            "Only %d overlapping months (need %d for window). "
            "Returning static correlation instead.",
            len(merged), window,
        )
        window = max(3, len(merged))

    results = {}
    cluster_cols = [c for c in merged.columns if c.startswith("cluster_")]

    for col in cluster_cols:
        pearson_rolling = (
            merged[col]
            .rolling(window)
            .corr(merged["misery_index"])
        )
        results[f"{col}_pearson"] = pearson_rolling

    result_df = pd.DataFrame(results, index=merged.index)
    logger.info(
        "Rolling correlation computed: %d months, %d clusters, window=%d",
        len(result_df), len(cluster_cols), window,
    )
    return result_df


def static_correlation(
    cluster_shares: pd.DataFrame,
    misery_index: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute overall (non-rolling) Pearson and Spearman correlations
    between each cluster's volumetric share and the Misery Index.

    Returns a summary table with columns: pearson_r, pearson_p, spearman_r, spearman_p
    """
    misery = misery_index.set_index("date")["misery_index"]
    merged = cluster_shares.join(misery, how="inner")

    cluster_cols = [c for c in merged.columns if c.startswith("cluster_")]
    rows = []

    for col in cluster_cols:
        valid = merged[[col, "misery_index"]].dropna()
        if len(valid) < 5:
            continue

        pr, pp = stats.pearsonr(valid[col], valid["misery_index"])
        sr, sp = stats.spearmanr(valid[col], valid["misery_index"])

        rows.append({
            "cluster": col,
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 4),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 4),
        })

    result = pd.DataFrame(rows)
    logger.info("Static correlations:\n%s", result.to_string())
    return result


def granger_causality(
    cluster_shares: pd.DataFrame,
    misery_index: pd.DataFrame,
    max_lag: int = 3,
) -> pd.DataFrame:
    """
    Run Granger causality tests: does the Misery Index Granger-cause
    shifts in cluster volumetric shares?

    Parameters
    ----------
    cluster_shares : monthly volumetric share DataFrame
    misery_index : DataFrame with 'date' and 'misery_index'
    max_lag : maximum lag order to test

    Returns
    -------
    DataFrame with columns: cluster, lag, f_stat, p_value
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    misery = misery_index.set_index("date")["misery_index"]
    merged = cluster_shares.join(misery, how="inner").dropna()

    cluster_cols = [c for c in merged.columns if c.startswith("cluster_")]
    rows = []

    for col in cluster_cols:
        test_data = merged[[col, "misery_index"]].values

        if len(test_data) < max_lag + 3:
            logger.warning(
                "Not enough data for Granger test on %s (need %d, have %d)",
                col, max_lag + 3, len(test_data),
            )
            continue

        try:
            results = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            for lag, result in results.items():
                f_test = result[0]["ssr_ftest"]
                rows.append({
                    "cluster": col,
                    "lag": lag,
                    "f_stat": round(f_test[0], 4),
                    "p_value": round(f_test[1], 4),
                })
        except Exception as e:
            logger.warning("Granger test failed for %s: %s", col, e)

    result = pd.DataFrame(rows)
    logger.info("Granger causality tests complete: %d results", len(result))
    return result


def classify_clusters(
    static_corr: pd.DataFrame,
    threshold: float = 0.4,
) -> pd.DataFrame:
    """
    Classify clusters based on their correlation with the Misery Index.

    Categories:
    - reactive: |pearson_r| >= threshold, positive correlation
      (grows when economy worsens)
    - counter_cyclical: |pearson_r| >= threshold, negative correlation
      (grows when economy improves)
    - stable: |pearson_r| < threshold
      (insensitive to economic conditions)
    """
    def _classify(row):
        r = row["pearson_r"]
        if abs(r) < threshold:
            return "stable"
        elif r > 0:
            return "reactive"
        else:
            return "counter_cyclical"

    classified = static_corr.copy()
    classified["economic_sensitivity"] = classified.apply(_classify, axis=1)

    logger.info("Cluster classification:\n%s", classified[["cluster", "economic_sensitivity"]].to_string())
    return classified
