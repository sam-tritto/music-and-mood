"""
Spotify Charts Data Loader
===========================
Loads and cleans the Kaggle "Spotify Top 200" dataset (pre-compiled with
audio features). Handles column normalization, date parsing, and deduplication.

Since the Spotify audio-features API was restricted for new apps in Nov 2024,
we rely on pre-compiled Kaggle datasets that were built before the lockdown.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cmi.config import AUDIO_FEATURES, DATE_END, DATE_START, REGION

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name normalization map (common Kaggle dataset variations)
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: dict[str, list[str]] = {
    "track_id": ["uri", "spotify_id", "id", "track_uri", "spotify_uri"],
    "title": ["track_name", "song", "name", "track_title", "song_name"],
    "artist": ["artist_name", "artist_names", "artists", "artist(s)"],
    "date": ["week", "chart_date", "release_date", "day"],
    "streams": ["stream", "plays", "count", "total_streams"],
    "rank": ["position", "chart_position", "chart_rank"],
    "danceability": [],
    "energy": [],
    "valence": [],
    "acousticness": [],
    "tempo": ["bpm"],
    "speechiness": [],
    "instrumentalness": [],
    "key": [],
    "mode": [],
    "loudness": [],
    "liveness": [],
    "duration_ms": ["duration"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various Kaggle column names to our canonical names."""
    col_lower = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=col_lower)

    for canonical, aliases in _COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: canonical})
                logger.info("Mapped column '%s' → '%s'", alias, canonical)
                break

    return df


def _clean_track_id(df: pd.DataFrame) -> pd.DataFrame:
    """Extract bare Spotify track ID from URI if needed."""
    if "track_id" in df.columns:
        # spotify:track:XXXX → XXXX
        df["track_id"] = (
            df["track_id"]
            .astype(str)
            .str.replace("spotify:track:", "", regex=False)
            .str.strip()
        )
    return df


def load_charts(
    csv_path: str | Path,
    region: str = REGION,
    date_start: str = DATE_START,
    date_end: str = DATE_END,
) -> pd.DataFrame:
    """
    Load and clean a Kaggle Spotify Top 200 CSV.

    Parameters
    ----------
    csv_path : path to the CSV file
    region : filter to this region (e.g. 'us'). Set to None to keep all.
    date_start / date_end : ISO date strings for the time range filter.

    Returns
    -------
    pd.DataFrame with columns:
        track_id, title, artist, date, year_month, streams, rank,
        + AUDIO_FEATURES (danceability, energy, valence, ...)
    """
    logger.info("Loading chart data from %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Raw shape: %s", df.shape)

    # Normalize column names
    df = _normalize_columns(df)
    df = _clean_track_id(df)

    # --- Region filter ---
    region_col = None
    for candidate in ("region", "country", "chart"):
        if candidate in df.columns:
            region_col = candidate
            break

    if region_col and region:
        df[region_col] = df[region_col].astype(str).str.strip().str.lower()
        df = df[df[region_col] == region.lower()].copy()
        logger.info("Filtered to region='%s': %d rows", region, len(df))

    # --- Date parsing & filtering ---
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df = df[(df["date"] >= date_start) & (df["date"] <= date_end)].copy()
        df["year_month"] = df["date"].dt.to_period("M")
        logger.info("Date-filtered to %s – %s: %d rows", date_start, date_end, len(df))

    # --- Streams to numeric ---
    if "streams" in df.columns:
        df["streams"] = pd.to_numeric(df["streams"], errors="coerce").fillna(0).astype(int)

    # --- Validate audio features exist ---
    missing_features = [f for f in AUDIO_FEATURES if f not in df.columns]
    if missing_features:
        logger.warning(
            "Missing audio features in dataset: %s. "
            "These will be NaN — you may need a different Kaggle source.",
            missing_features,
        )

    # --- Deduplicate: keep one row per (track_id, date) ---
    id_col = "track_id" if "track_id" in df.columns else "title"
    if "date" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col, "date"], keep="first")
        logger.info("Deduplicated: %d → %d rows", before, len(df))

    # --- Sort ---
    sort_cols = [c for c in ["date", "rank"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    logger.info("Final chart data shape: %s", df.shape)
    return df


def get_unique_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique tracks from chart data for lyrics fetching.

    Returns a deduplicated DataFrame with one row per unique track,
    keeping the first occurrence's metadata and the mean audio features.
    """
    id_col = "track_id" if "track_id" in df.columns else "title"
    meta_cols = [c for c in [id_col, "title", "artist"] if c in df.columns]
    audio_cols = [c for c in AUDIO_FEATURES if c in df.columns]

    # Group: keep first meta, mean audio features
    meta = df.groupby(id_col)[meta_cols].first()
    audio = df.groupby(id_col)[audio_cols].mean()

    unique = meta.join(audio).reset_index(drop=id_col in meta_cols)
    logger.info("Unique tracks: %d", len(unique))
    return unique
