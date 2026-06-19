"""
Multimodal Feature Fusion
==========================
Combines audio features (7D) with compressed lyric embeddings (10D)
into a single fused feature matrix for clustering.

The key challenge: raw lyric embeddings (768D) would completely dominate
the 7 audio features in any distance metric. UMAP compression down to
~10D preserves the macro-semantic structure (isolation vs. partying vs.
heartbreak) while keeping feature spaces balanced.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from cmi.config import EMBEDDING_DIM_TARGET

logger = logging.getLogger(__name__)


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = EMBEDDING_DIM_TARGET,
    random_state: int = 42,
    **umap_kwargs,
) -> np.ndarray:
    """
    Reduce high-dimensional lyric embeddings via UMAP.

    Parameters
    ----------
    embeddings : (n_tracks, 768) matrix from gemini-embedding-001
    n_components : target dimensionality (default 10)
    random_state : for reproducibility
    **umap_kwargs : additional UMAP parameters

    Returns
    -------
    np.ndarray of shape (n_tracks, n_components)
    """
    logger.info(
        "Reducing embeddings from %dD → %dD via UMAP",
        embeddings.shape[1],
        n_components,
    )

    reducer = UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=min(15, embeddings.shape[0] - 1),
        min_dist=0.1,
        metric="cosine",
        **umap_kwargs,
    )
    reduced = reducer.fit_transform(embeddings)

    logger.info("Reduced embedding shape: %s", reduced.shape)
    return reduced


def fuse_features(
    audio_matrix: np.ndarray,
    reduced_embeddings: np.ndarray | None = None,
) -> tuple[np.ndarray, StandardScaler]:
    """
    Concatenate scaled audio features + reduced lyric embeddings.

    If reduced_embeddings is None, returns just the scaled audio matrix
    (useful for tracks without lyrics).

    Parameters
    ----------
    audio_matrix : (n, 7) raw audio features
    reduced_embeddings : (n, 10) UMAP-compressed lyric embeddings, or None

    Returns
    -------
    (fused_matrix, scaler) — the fused matrix has mean=0, var=1 per feature
    """
    if reduced_embeddings is not None:
        if audio_matrix.shape[0] != reduced_embeddings.shape[0]:
            raise ValueError(
                f"Row count mismatch: audio={audio_matrix.shape[0]}, "
                f"embeddings={reduced_embeddings.shape[0]}"
            )
        combined = np.hstack([audio_matrix, reduced_embeddings])
        logger.info(
            "Fused matrix: %d audio + %d lyric = %dD total",
            audio_matrix.shape[1],
            reduced_embeddings.shape[1],
            combined.shape[1],
        )
    else:
        combined = audio_matrix
        logger.info(
            "Audio-only fusion (no lyrics): %dD", combined.shape[1]
        )

    scaler = StandardScaler()
    fused = scaler.fit_transform(combined)

    logger.info("Fused & scaled matrix shape: %s", fused.shape)
    return fused, scaler
