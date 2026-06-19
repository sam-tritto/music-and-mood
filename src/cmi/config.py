"""
Centralized configuration for the Cultural Misery Index pipeline.

Loads API keys from .env, defines paths and constants used across all modules.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GENIUS_ACCESS_TOKEN: str = os.getenv("GENIUS_ACCESS_TOKEN", "")
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_RAW: Path = _PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED: Path = _PROJECT_ROOT / "data" / "processed"
LYRICS_CACHE: Path = DATA_RAW / "lyrics_cache"

# Ensure directories exist
for _dir in (DATA_RAW, DATA_PROCESSED, LYRICS_CACHE):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Audio feature columns (from Spotify / Kaggle dataset)
# ---------------------------------------------------------------------------
AUDIO_FEATURES: list[str] = [
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "tempo",
    "speechiness",
    "instrumentalness",
]

# ---------------------------------------------------------------------------
# Clustering constants
# ---------------------------------------------------------------------------
K_CLUSTERS: int = 6
EMBEDDING_DIM_TARGET: int = 10  # UMAP target dims for lyric embeddings
TIME_WINDOW: str = "MS"  # Monthly Start — pandas offset alias

# ---------------------------------------------------------------------------
# Model names (Google GenAI)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-004")
NARRATIVE_MODEL: str = os.getenv("NARRATIVE_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Region & date range
# ---------------------------------------------------------------------------
REGION: str = "us"
DATE_START: str = "2018-01-01"
DATE_END: str = "2024-12-31"
