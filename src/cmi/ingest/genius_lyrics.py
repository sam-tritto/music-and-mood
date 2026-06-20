"""
Genius Lyrics Fetcher
=====================
Fetches full song lyrics via the Genius API (using lyricsgenius) with
disk-based JSON caching, rate limiting, and lyrics cleaning.

The Genius API itself doesn't serve lyrics — lyricsgenius scrapes them
from the Genius web page after finding the song via the API. This means
requests are slower and may occasionally fail on anti-scraping measures.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from cmi.config import GENIUS_ACCESS_TOKEN, LYRICS_CACHE

logger = logging.getLogger(__name__)

# Delay between Genius requests to avoid rate limiting / IP blocks
_REQUEST_DELAY_SECONDS: float = 1.5


def _cache_key(title: str, artist: str) -> str:
    """Generate a filesystem-safe cache key from title + artist."""
    raw = f"{title.lower().strip()}|{artist.lower().strip()}"
    return hashlib.md5(raw.encode()).hexdigest()


def _clean_lyrics(raw_lyrics: str | None) -> str | None:
    """
    Clean raw Genius lyrics text:
    - Strip section headers like [Chorus], [Verse 1], etc.
    - Remove the trailing "...Lyrics" and "Embed" junk
    - Normalize whitespace
    """
    if not raw_lyrics:
        return None

    text = raw_lyrics

    # Remove section headers: [Chorus], [Verse 1], [Bridge], etc.
    text = re.sub(r"\[.*?\]", "", text)

    # Remove common Genius footer artifacts
    text = re.sub(r"\d*Embed$", "", text.strip())
    text = re.sub(r"You might also like", "", text)

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # If we stripped everything, return None
    if len(text) < 20:
        return None

    return text


def _fetch_single(
    genius_client,
    title: str,
    artist: str,
    cache_dir: Path,
) -> str | None:
    """Fetch lyrics for a single track, using disk cache."""
    key = _cache_key(title, artist)
    cache_file = cache_dir / f"{key}.json"

    # Check cache first
    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return data.get("lyrics")

    # Fetch from Genius
    try:
        song = genius_client.search_song(title, artist)
        raw_lyrics = song.lyrics if song else None
    except Exception as e:
        logger.warning("Genius fetch failed for '%s' by '%s': %s", title, artist, e)
        raw_lyrics = None

    cleaned = _clean_lyrics(raw_lyrics)

    # Cache the result (even if None, to avoid re-fetching failures)
    cache_data = {
        "title": title,
        "artist": artist,
        "lyrics": cleaned,
        "raw_length": len(raw_lyrics) if raw_lyrics else 0,
    }
    cache_file.write_text(json.dumps(cache_data, ensure_ascii=False, indent=2))

    return cleaned


def fetch_lyrics_batch(
    tracks_df: pd.DataFrame,
    cache_dir: Path = LYRICS_CACHE,
    access_token: str = GENIUS_ACCESS_TOKEN,
    delay: float = _REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """
    Fetch lyrics for all unique tracks in the DataFrame.

    Parameters
    ----------
    tracks_df : DataFrame with 'title' and 'artist' columns
    cache_dir : directory for JSON lyrics cache files
    access_token : Genius API client access token
    delay : seconds to wait between API requests

    Returns
    -------
    DataFrame with an added 'lyrics' column (str or NaN)
    """
    import lyricsgenius

    if not access_token:
        raise ValueError(
            "GENIUS_ACCESS_TOKEN is not set. "
            "Add it to your .env file (see .env.example)."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Genius client
    genius = lyricsgenius.Genius(
        access_token,
        remove_section_headers=False,  # We do our own cleaning
        retries=3,
    )
    genius.verbose = False

    results: list[str | None] = []

    for _, row in tqdm(
        tracks_df.iterrows(),
        total=len(tracks_df),
        desc="Fetching lyrics",
    ):
        title = str(row.get("title", ""))
        artist = str(row.get("artist", ""))

        if not title or not artist:
            results.append(None)
            continue

        lyrics = _fetch_single(genius, title, artist, cache_dir)
        results.append(lyrics)

        # Rate limiting
        time.sleep(delay)

    out = tracks_df.copy()
    out["lyrics"] = results

    found = sum(1 for r in results if r is not None)
    logger.info(
        "Lyrics fetched: %d / %d tracks (%.1f%% hit rate)",
        found,
        len(results),
        100 * found / max(len(results), 1),
    )

    return out
