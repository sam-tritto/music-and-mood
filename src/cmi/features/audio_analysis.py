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
    DATA_RAW,
)

logger = logging.getLogger(__name__)


def _get_itunes_preview_url(title: str, artist: str) -> str | None:
    """Query iTunes Search API to retrieve a track's preview URL."""
    import urllib.parse
    import requests
    query = f"{title} {artist}"
    url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&limit=1&entity=musicTrack"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                preview_url = results[0].get("previewUrl")
                if preview_url:
                    logger.info("Found iTunes preview URL for '%s' by '%s'", title, artist)
                    return preview_url
    except Exception as e:
        logger.warning("iTunes search failed for '%s' by '%s': %s", title, artist, e)
    return None


def _extract_complexity_from_url(preview_url: str) -> dict[str, float] | None:
    """Download audio snippet and compute Harmonic Entropy and Timbral Variance using librosa."""
    import tempfile
    import requests
    import librosa
    import numpy as np

    try:
        response = requests.get(preview_url, timeout=15)
        if response.status_code != 200:
            logger.warning("Failed to download audio preview from %s", preview_url)
            return None

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = Path(temp_file.name)

        try:
            # Load audio (CoreAudio on macOS decodes M4A automatically)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
                warnings.filterwarnings("ignore", category=FutureWarning)
                y, sr = librosa.load(temp_path, sr=None)

            # 1. Calculate Harmonic Entropy (Shannon entropy of normalized chroma)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma = chroma.T
            row_sums = chroma.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1e-9, row_sums)
            p_probs = chroma / row_sums

            with np.errstate(divide="ignore", invalid="ignore"):
                entropy_per_frame = -np.sum(p_probs * np.log2(p_probs + 1e-9), axis=1)

            avg_entropy = float(np.mean(entropy_per_frame))

            # 2. Calculate Timbral Variance (average variance across first 12 MFCCs)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
            variances = np.var(mfccs, axis=1)
            avg_variance = float(np.mean(variances))

            return {
                "harmonic_entropy": round(avg_entropy, 4),
                "timbral_variance": round(avg_variance, 4),
            }
        finally:
            if temp_path.exists():
                temp_path.unlink()
    except Exception as e:
        logger.warning("Failed to extract complexity from preview %s: %s", preview_url, e)
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
    delay_seconds: float = 0.5,
) -> pd.DataFrame:
    """
    Fetch audio analysis features for unique tracks.
    Uses local cache, pre-compiled CSV lookup, iTunes Search API fallback,
    and librosa-based offline feature extraction.

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

    unique_tracks = tracks_df.drop_duplicates(subset=["track_id"]).copy()

    # Load pre-compiled audio features database for quick CSV preview lookup
    db_path = DATA_RAW / "Hot 100 Audio Features.csv"
    preview_lookup = {}
    if db_path.exists():
        try:
            logger.info("Loading pre-compiled Spotify preview URLs from %s", db_path.name)
            db_df = pd.read_csv(db_path, usecols=["spotify_track_id", "spotify_track_preview_url"])
            db_df = db_df.dropna(subset=["spotify_track_id", "spotify_track_preview_url"])
            preview_lookup = dict(zip(db_df["spotify_track_id"], db_df["spotify_track_preview_url"]))
        except Exception as e:
            logger.warning("Could not load preview lookup database: %s", e)

    entropies = []
    variances = []

    for _, row in tqdm(
        unique_tracks.iterrows(),
        total=len(unique_tracks),
        desc="Analyzing audio complexity (local librosa)",
    ):
        track_id = str(row["track_id"])
        title = str(row.get("title", ""))
        artist = str(row.get("artist", ""))

        # Guard against placeholder / empty / invalid track IDs
        if not track_id or len(track_id) != 22 or not track_id.isalnum():
            entropies.append(0.5)
            variances.append(100.0)
            continue

        cache_file = cache_dir / f"{track_id}.json"

        # 1. Check local cache first
        if cache_file.exists():
            try:
                cache_data = json.loads(cache_file.read_text())
                if "harmonic_entropy" in cache_data and "timbral_variance" in cache_data:
                    entropies.append(cache_data["harmonic_entropy"])
                    variances.append(cache_data["timbral_variance"])
                    continue
                metrics = calculate_complexity(cache_data)
                entropies.append(metrics["harmonic_entropy"])
                variances.append(metrics["timbral_variance"])
                continue
            except Exception as e:
                logger.warning("Failed to load cached analysis for %s: %s. Re-fetching...", track_id, e)

        # 2. Not cached: Resolve preview URL (CSV lookup -> iTunes API)
        preview_url = preview_lookup.get(track_id)
        is_itunes = False

        if not preview_url:
            preview_url = _get_itunes_preview_url(title, artist)
            is_itunes = True

        metrics = None
        if preview_url:
            # 3. Extract complexity from preview using librosa
            metrics = _extract_complexity_from_url(preview_url)
            if is_itunes:
                time.sleep(delay_seconds)

        if metrics:
            cache_data = {
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "harmonic_entropy": metrics["harmonic_entropy"],
                "timbral_variance": metrics["timbral_variance"],
                "source": "local_librosa",
            }
            entropies.append(metrics["harmonic_entropy"])
            variances.append(metrics["timbral_variance"])
        else:
            cache_data = {
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "harmonic_entropy": 0.5,
                "timbral_variance": 100.0,
                "source": "local_librosa_failed_fallback",
            }
            entropies.append(0.5)
            variances.append(100.0)

        # Write to cache
        try:
            cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning("Failed to write cache for %s: %s", track_id, e)

    unique_tracks["harmonic_entropy"] = entropies
    unique_tracks["timbral_variance"] = variances

    return unique_tracks[["track_id", "harmonic_entropy", "timbral_variance"]]
