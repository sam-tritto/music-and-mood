"""
Audio Feature Extraction & Normalization
==========================================
Extracts the 7 core Spotify audio features from the chart DataFrame and
provides scaling utilities for downstream clustering.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from cmi.config import AUDIO_FEATURES

logger = logging.getLogger(__name__)


def extract_audio_matrix(
    df: pd.DataFrame,
    features: list[str] = AUDIO_FEATURES,
) -> np.ndarray:
    """
    Extract audio features as a dense numpy matrix.

    Parameters
    ----------
    df : DataFrame containing the audio feature columns
    features : list of column names to extract

    Returns
    -------
    np.ndarray of shape (n_tracks, n_features)
    """
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning("Missing audio features (will be zero-filled): %s", missing)

    matrix = df[available].fillna(0).values.astype(np.float64)

    # Add zero columns for any missing features to maintain consistent dimensions
    if missing:
        zeros = np.zeros((matrix.shape[0], len(missing)))
        matrix = np.hstack([matrix, zeros])

    logger.info("Audio matrix shape: %s", matrix.shape)
    return matrix


def scale_audio(
    matrix: np.ndarray,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Apply StandardScaler to audio features.

    Parameters
    ----------
    matrix : raw audio feature matrix
    scaler : pre-fitted scaler (if None, fits a new one)

    Returns
    -------
    (scaled_matrix, fitted_scaler)
    """
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)
        logger.info("Fitted new StandardScaler on audio features")
    else:
        scaled = scaler.transform(matrix)
        logger.info("Applied existing StandardScaler to audio features")

    return scaled, scaler


def audio_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for audio features.
    Useful for notebook display and cluster profiling.
    """
    available = [f for f in AUDIO_FEATURES if f in df.columns]
    return df[available].describe().round(3)
