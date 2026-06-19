"""
Lyric Embedding via Google Gemini
==================================
Generates dense vector embeddings for song lyrics using Google's
gemini-embedding-001 model via the google-genai SDK.

Embeddings capture the semantic meaning of lyrics in a high-dimensional
space — songs about similar themes (isolation, partying, heartbreak) will
cluster together even if they use completely different words.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from tqdm import tqdm

from cmi.config import EMBEDDING_MODEL, GOOGLE_API_KEY
from cmi.utils.cost_estimator import estimate_embedding_cost

logger = logging.getLogger(__name__)

# Google recommends max ~100 texts per embed_content call
_BATCH_SIZE: int = 100

# Delay between batches to avoid rate limiting
_BATCH_DELAY_SECONDS: float = 0.5

# Max retries on transient errors
_MAX_RETRIES: int = 3


def _init_client():
    """Lazy-initialize the Google GenAI client."""
    from google import genai

    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/apikey "
            "and add it to your .env file."
        )
    return genai.Client(api_key=GOOGLE_API_KEY)


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    """
    Truncate text to stay within embedding model token limits.
    gemini-embedding-001 handles ~2048 tokens; rough char estimate.
    """
    if len(text) > max_chars:
        logger.debug("Truncating text from %d to %d chars", len(text), max_chars)
        return text[:max_chars]
    return text


def embed_lyrics(
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = _BATCH_SIZE,
) -> np.ndarray:
    """
    Embed a list of lyric texts into dense vectors.

    Parameters
    ----------
    texts : list of lyric strings (NaN/empty entries should be pre-filtered)
    model : Google embedding model name
    batch_size : number of texts per API call

    Returns
    -------
    np.ndarray of shape (n_texts, embedding_dim)
        embedding_dim is 768 for gemini-embedding-001
    """
    # Estimate cost locally first
    est = estimate_embedding_cost(texts, model)
    print(f"💰 [Cost Estimate] Embedding {len(texts)} tracks via {model}:")
    print(f"   Est. Tokens: {est['estimated_tokens']:,}")
    print(f"   Est. Cost:   ${est['estimated_cost_usd']:.6f} USD")
    logger.info("Est. embedding cost: $%f", est['estimated_cost_usd'])

    client = _init_client()

    # Pre-process: truncate long texts
    processed = [_truncate_text(t) for t in texts]

    all_embeddings: list[list[float]] = []
    n_batches = (len(processed) + batch_size - 1) // batch_size

    for i in tqdm(range(n_batches), desc="Embedding lyrics"):
        batch = processed[i * batch_size : (i + 1) * batch_size]

        for attempt in range(_MAX_RETRIES):
            try:
                result = client.models.embed_content(
                    model=model,
                    contents=batch,
                )
                # result.embeddings is a list of Embedding objects
                for emb in result.embeddings:
                    all_embeddings.append(emb.values)
                break
            except Exception as e:
                if attempt < _MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "Embedding batch %d failed (attempt %d/%d): %s. "
                        "Retrying in %ds...",
                        i, attempt + 1, _MAX_RETRIES, e, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Embedding batch %d failed after %d retries: %s",
                        i, _MAX_RETRIES, e,
                    )
                    # Fill with zeros for failed batches
                    dim = len(all_embeddings[0]) if all_embeddings else 768
                    for _ in batch:
                        all_embeddings.append([0.0] * dim)

        # Rate limiting between batches
        if i < n_batches - 1:
            time.sleep(_BATCH_DELAY_SECONDS)

    matrix = np.array(all_embeddings, dtype=np.float64)
    logger.info("Embedding matrix shape: %s", matrix.shape)
    return matrix
