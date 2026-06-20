"""
Timbral & Harmonic Audio Analysis via Spotify API
==================================================
Taps into Spotify's raw Audio Analysis API endpoint to compute segment-level
music-theory-driven complexity features:

1. Harmonic Entropy: Measure of chord predictability and complexity based on
   relative chroma pitches.
2. Timbral Variance (Sonic Roughness): Standard deviation/variance of the 12
   timbre coefficients over time as a proxy for production density.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from cmi.config import (
    AUDIO_ANALYSIS_CACHE,
    SPOTIPY_CLIENT_ID,
    SPOTIPY_CLIENT_SECRET,
)

logger = logging.getLogger(__name__)


def _init_spotify():
    """Initialize Spotipy client using Client Credentials flow if available."""
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    if not SPOTIPY_CLIENT_ID or not SPOTIPY_CLIENT_SECRET:
        logger.warning(
            "Spotify credentials not set (SPOTIPY_CLIENT_ID/SECRET). "
            "Live Audio Analysis requests will be skipped, using default baselines."
        )
        return None

    try:
        auth_manager = SpotifyClientCredentials(
            client_id=SPOTIPY_CLIENT_ID,
            client_secret=SPOTIPY_CLIENT_SECRET,
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        logger.error("Failed to initialize Spotify client: %s", e)
        return None


def calculate_complexity(analysis_data: dict) -> dict[str, float]:
    """
    Compute Harmonic Entropy and Timbral Variance from raw Spotify segment lists.

    Parameters
    ----------
    analysis_data : raw JSON dictionary returned by Spotify's audio_analysis API

    Returns
    -------
    dict with 'harmonic_entropy' and 'timbral_variance' floats
    """
    segments = analysis_data.get("segments", [])
    if not segments:
        return {"harmonic_entropy": 0.5, "timbral_variance": 0.0}

    pitches_list = []
    timbre_list = []

    for seg in segments:
        p = seg.get("pitches")
        t = seg.get("timbre")
        if p and len(p) == 12:
            pitches_list.append(p)
        if t and len(t) == 12:
            timbre_list.append(t)

    # 1. Calculate Harmonic Entropy (Shannon entropy of normalized chroma)
    if pitches_list:
        pitches = np.array(pitches_list, dtype=np.float64)
        # Normalize each row to create a probability distribution over the 12 pitch classes
        row_sums = pitches.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1e-9, row_sums)
        p_probs = pitches / row_sums

        # Shannon Entropy per segment: -sum(p_i * log2(p_i))
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_per_segment = -np.sum(p_probs * np.log2(p_probs + 1e-9), axis=1)

        # Average entropy across all segments in the track
        avg_entropy = float(np.mean(entropy_per_segment))
    else:
        avg_entropy = 0.5

    # 2. Calculate Timbral Variance (average variance across the 12 MFCC timbre coefficients)
    if timbre_list:
        timbres = np.array(timbre_list, dtype=np.float64)
        # Compute variance of each of the 12 coefficients over time
        variances = np.var(timbres, axis=0)
        # Average variance to represent overall production roughness/texture density
        avg_variance = float(np.mean(variances))
    else:
        avg_variance = 100.0

    return {
        "harmonic_entropy": round(avg_entropy, 4),
        "timbral_variance": round(avg_variance, 4),
    }


def fetch_audio_complexity_batch(
    tracks_df: pd.DataFrame,
    cache_dir: Path = AUDIO_ANALYSIS_CACHE,
    delay_seconds: float = 1.0,
) -> pd.DataFrame:
    """
    Fetch raw Spotify Audio Analysis for unique tracks and calculate complexity metrics.

    Includes local file caching and an offline baseline fallback.

    Parameters
    ----------
    tracks_df : DataFrame with 'track_id', 'title', and 'artist' columns
    cache_dir : path to JSON file cache directory
    delay_seconds : API rate-limit delay in seconds

    Returns
    -------
    pd.DataFrame with ['track_id', 'harmonic_entropy', 'timbral_variance']
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sp = _init_spotify()

    unique_tracks = tracks_df.drop_duplicates(subset=["track_id"]).copy()
    
    entropies = []
    variances = []

    # If no client, fall back to offline baseline defaults immediately
    if sp is None:
        logger.info("Using offline baseline fallback for all tracks (no credentials).")
        unique_tracks["harmonic_entropy"] = 0.5
        unique_tracks["timbral_variance"] = 100.0
        return unique_tracks[["track_id", "harmonic_entropy", "timbral_variance"]]

    for _, row in tqdm(
        unique_tracks.iterrows(),
        total=len(unique_tracks),
        desc="Analyzing audio complexity",
    ):
        track_id = str(row["track_id"])
        
        # Guard against placeholder / empty track IDs
        if not track_id or track_id.startswith("SongID") or len(track_id) < 15:
            # Baseline values for invalid IDs
            entropies.append(0.5)
            variances.append(100.0)
            continue

        cache_file = cache_dir / f"{track_id}.json"

        # Check local cache first
        if cache_file.exists():
            try:
                analysis = json.loads(cache_file.read_text())
                metrics = calculate_complexity(analysis)
                entropies.append(metrics["harmonic_entropy"])
                variances.append(metrics["timbral_variance"])
                continue
            except Exception as e:
                logger.warning("Failed to load cached analysis for %s: %s. Re-fetching...", track_id, e)

        # Live fetch from Spotify
        try:
            analysis = sp.audio_analysis(track_id)
            
            # Cache the raw JSON data
            cache_file.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
            
            metrics = calculate_complexity(analysis)
            entropies.append(metrics["harmonic_entropy"])
            variances.append(metrics["timbral_variance"])
            
            # Rate limiting
            time.sleep(delay_seconds)
        except Exception as e:
            logger.warning("Spotify audio analysis failed for track %s: %s", track_id, e)
            # Offline baseline defaults for failures
            entropies.append(0.5)
            variances.append(100.0)

    unique_tracks["harmonic_entropy"] = entropies
    unique_tracks["timbral_variance"] = variances

    return unique_tracks[["track_id", "harmonic_entropy", "timbral_variance"]]
