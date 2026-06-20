"""
GenAI Narrative Engine
=======================
Transforms raw cluster analytics into compelling cultural analysis using
Google Gemini. Instead of just plotting charts, we generate a structured
data payload summarizing each cluster's musical nature, semantic themes,
and economic affinity — then prompt the LLM to write the cultural digest.

This is the "so what?" layer that turns numbers into stories.
"""

from __future__ import annotations

import json
import logging
from collections import Counter

import numpy as np
import pandas as pd

from cmi.config import GOOGLE_API_KEY, NARRATIVE_MODEL, AUDIO_FEATURES, DATA_PROCESSED
from cmi.utils.cost_estimator import estimate_narrative_cost

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (the persona and task definition)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an economic sociologist and cultural critic with deep expertise in \
musicology and behavioral economics. Your job is to analyze data payloads \
representing shifting musical trends and explain how they reflect the \
collective emotional state of society in response to macroeconomic pressure.

You write with the precision of an academic but the clarity of a journalist. \
You use vivid, evocative language to describe sonic and lyrical qualities. \
You ground every claim in the data provided — never fabricate statistics.
"""

USER_PROMPT_TEMPLATE = """\
I am providing a JSON payload for a tracked multi-modal music cluster \
during a specific economic period.

```json
{payload_json}
```

Please provide:

1. **Cluster Name**: A creative, descriptive, human-readable name for this \
cluster based on its audio and lyric profile (e.g., "Neon Escapism", \
"Bedroom Confessional", "Rage Therapy").

2. **Sonic Profile** (1 sentence): Describe what this music *sounds* like \
based on the audio features.

3. **Lyrical Landscape** (1 sentence): Describe the thematic territory of \
the lyrics based on the top keywords and semantic properties.

4. **Economic Response** (2-3 sentences): Detail how this cluster responded \
to the shifting Misery Index. Explain what the sonic and semantic properties \
tell us about consumer coping mechanisms during this economic environment.

Format your response as structured markdown with the headers above.
"""


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


def _describe_audio_profile(centroid: np.ndarray) -> dict[str, str]:
    """
    Convert a cluster centroid into a human-readable audio profile.
    Maps feature values to qualitative descriptors.
    """
    descriptions = {}
    feature_names = AUDIO_FEATURES

    for i, feature in enumerate(feature_names):
        if i >= len(centroid):
            break
        val = centroid[i]

        if feature == "tempo":
            # Tempo is on a different scale (60-200+ BPM)
            if val < 90:
                descriptions[feature] = f"slow ({val:.0f} BPM)"
            elif val < 120:
                descriptions[feature] = f"moderate ({val:.0f} BPM)"
            elif val < 140:
                descriptions[feature] = f"upbeat ({val:.0f} BPM)"
            else:
                descriptions[feature] = f"fast ({val:.0f} BPM)"
        else:
            # 0-1 scale features
            if val < 0.3:
                level = "low"
            elif val < 0.6:
                level = "moderate"
            else:
                level = "high"
            descriptions[feature] = f"{level} ({val:.2f})"

    return descriptions


def _extract_lyric_keywords(
    lyrics_series: pd.Series,
    top_n: int = 10,
) -> list[str]:
    """
    Extract the most common meaningful words from a collection of lyrics.
    Very basic — for the tutorial, this is sufficient. A production system
    would use TF-IDF or keyword extraction models.
    """
    # Common English stopwords to filter out
    stopwords = {
        "the", "a", "an", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "with", "you", "me", "my",
        "your", "we", "they", "he", "she", "i", "am", "are", "was",
        "were", "be", "been", "do", "did", "have", "has", "had", "will",
        "would", "could", "should", "can", "may", "might", "shall",
        "that", "this", "these", "those", "what", "which", "who", "whom",
        "where", "when", "why", "how", "all", "each", "every", "both",
        "few", "more", "most", "other", "some", "no", "nor", "too",
        "very", "just", "about", "up", "out", "so", "if", "than",
        "then", "there", "here", "now", "also", "only", "its", "his",
        "her", "our", "their", "them", "us", "him", "as", "from", "by",
        "like", "oh", "yeah", "ya", "ay", "ooh", "uh", "ah", "la",
        "na", "da", "got", "get", "go", "come", "let", "make", "take",
        "know", "want", "dont", "im", "aint", "gonna", "wanna", "gotta",
    }

    word_counts: Counter = Counter()

    for text in lyrics_series.dropna():
        words = str(text).lower().split()
        words = [w.strip(".,!?()[]\"'") for w in words]
        words = [w for w in words if len(w) > 2 and w not in stopwords and w.isalpha()]
        word_counts.update(words)

    return [word for word, _ in word_counts.most_common(top_n)]


def build_cluster_payload(
    cluster_id: int,
    cluster_data: pd.DataFrame,
    centroid: np.ndarray,
    temporal_window: str,
    misery_correlation: float,
    volume_share_delta: str,
    lyrics_column: str = "lyrics",
) -> dict:
    """
    Assemble the structured JSON payload for a single cluster.

    Parameters
    ----------
    cluster_id : the cluster index
    cluster_data : DataFrame of tracks in this cluster during the window
    centroid : the cluster centroid (in original feature space)
    temporal_window : human-readable time period label
    misery_correlation : Pearson r between this cluster's share and Misery Index
    volume_share_delta : human-readable change description
    lyrics_column : column name containing lyrics text

    Returns
    -------
    dict suitable for JSON serialization and LLM prompting
    """
    audio_profile = _describe_audio_profile(centroid)

    lyric_keywords = []
    if lyrics_column in cluster_data.columns:
        lyric_keywords = _extract_lyric_keywords(cluster_data[lyrics_column])

    payload = {
        "cluster_id": f"Cluster_{cluster_id}",
        "temporal_window": temporal_window,
        "n_tracks": len(cluster_data),
        "audio_profile": audio_profile,
        "top_lyric_keywords": lyric_keywords,
        "economic_metrics": {
            "misery_index_correlation": round(misery_correlation, 3),
            "volume_share_delta": volume_share_delta,
        },
    }

    # Add representative tracks if available
    if "title" in cluster_data.columns and "artist" in cluster_data.columns:
        top_tracks = (
            cluster_data.head(5)[["title", "artist"]]
            .apply(lambda r: f"{r['title']} — {r['artist']}", axis=1)
            .tolist()
        )
        payload["representative_tracks"] = top_tracks

    return payload


def generate_narrative(
    payload: dict,
    model: str = NARRATIVE_MODEL,
) -> str:
    """
    Send a cluster payload to Gemini and get back the cultural analysis.

    Parameters
    ----------
    payload : structured cluster data dict
    model : Gemini model name

    Returns
    -------
    Generated narrative text (markdown formatted)
    """
    from google.genai import types

    payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
    user_prompt = USER_PROMPT_TEMPLATE.format(payload_json=payload_json)

    # Estimate cost locally first
    est = estimate_narrative_cost(
        prompt=user_prompt,
        system_instruction=SYSTEM_PROMPT,
        max_output_tokens=1024,
        model=model
    )
    print(f"💰 [Cost Estimate] Generating narrative for {payload.get('cluster_id', 'unknown')} via {model}:")
    print(f"   Est. Input Tokens:  {est['estimated_input_tokens']:,}")
    print(f"   Est. Output Tokens: {est['estimated_output_tokens']:,} (approx. baseline)")
    print(f"   Est. Cost:          ${est['estimated_cost_usd']:.6f} USD")
    logger.info("Est. narrative cost: $%f", est['estimated_cost_usd'])

    client = _init_client()

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=1024,
        ),
        contents=user_prompt,
    )

    narrative = response.text or "(No response generated)"
    logger.info(
        "Generated narrative for %s: %d chars",
        payload.get("cluster_id", "unknown"),
        len(narrative),
    )
    return narrative


def generate_full_report(
    payloads: list[dict],
    model: str = NARRATIVE_MODEL,
) -> str:
    """
    Generate narratives for all cluster payloads and compile into
    a full markdown report.

    Parameters
    ----------
    payloads : list of cluster payload dicts
    model : Gemini model name

    Returns
    -------
    Full markdown report string
    """
    sections = [
        "# 🎵 Cultural Misery Index Report\n\n"
        "*Multimodal analysis of music streaming trends "
        "and macroeconomic stress indicators*\n\n---\n"
    ]

    for payload in payloads:
        cluster_id = payload.get("cluster_id", "Unknown")
        logger.info("Generating narrative for %s...", cluster_id)

        narrative = generate_narrative(payload, model=model)
        sections.append(f"\n## {cluster_id}\n\n{narrative}\n\n---\n")

    report = "\n".join(sections)
    logger.info("Full report generated: %d sections, %d chars", len(payloads), len(report))
    return report


# ---------------------------------------------------------------------------
# Brief Narrative Generator prompts and function
# ---------------------------------------------------------------------------
BRIEF_SYSTEM_PROMPT = """\
You are an expert musicologist, economic sociologist, and cultural critic. \
Your task is to analyze structured data payloads representing music clusters \
and provide a concise, catchy name and a brief summary for each.
"""

BRIEF_USER_PROMPT_TEMPLATE = """\
I am providing a JSON payload representing a music cluster.

```json
{payload_json}
```

Please analyze this cluster and return a JSON object with the following fields:
1. **name**: A thoughtful, creative, and catchy name for this cluster (e.g. "Bedroom Melancholia", "Escapist Neon Rave").
2. **description**: A concise 1-3 sentence description of the cluster. The description must capture the musical style (based on audio features), lyrical themes (based on top keywords), and the cultural/emotional vibe. The description must be exactly 1 to 3 sentences long.

Return ONLY a valid JSON object. Do not include markdown formatting or extra text outside the JSON.
"""


def generate_brief_descriptions(
    payloads: list[dict],
    model: str = NARRATIVE_MODEL,
    use_cache: bool = True,
) -> list[dict]:
    """
    Generate brief descriptions (catchy name + 1-3 sentence description) for all clusters.
    Prints the total estimated cost for all clusters combined *before* calling the Gemini API.
    Uses a local cache file to avoid duplicate API requests.

    Parameters
    ----------
    payloads : list of cluster payload dicts
    model : Gemini model name (default: gemini-2.5-flash)
    use_cache : if True, load existing descriptions from disk to skip API calls

    Returns
    -------
    list of dicts containing cluster_id, name, and description
    """
    from google.genai import types

    cache_path = DATA_PROCESSED / "brief_descriptions_cache.json"
    cache = {}
    if use_cache and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            logger.info("Loaded %d descriptions from cache file: %s", len(cache), cache_path)
        except Exception as e:
            logger.warning("Failed to load brief descriptions cache: %s", e)

    # 1. Total cost estimation before running API calls (only for uncached payloads)
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    uncached_payloads = []

    for payload in payloads:
        cluster_id = payload.get("cluster_id", "Unknown")
        if use_cache and cluster_id in cache:
            continue
        
        uncached_payloads.append(payload)
        payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
        user_prompt = BRIEF_USER_PROMPT_TEMPLATE.format(payload_json=payload_json)
        
        # Estimate cost using the cost_estimator
        est = estimate_narrative_cost(
            prompt=user_prompt,
            system_instruction=BRIEF_SYSTEM_PROMPT,
            max_output_tokens=2048,
            model=model
        )
        total_input_tokens += est["estimated_input_tokens"]
        total_output_tokens += est["estimated_output_tokens"]
        total_cost += est["estimated_cost_usd"]

    print("\n💰 ==================================================")
    print(f"💰 [Pre-Execution Cost Estimate] Model: {model}")
    print(f"💰 Uncached Clusters to Process: {len(uncached_payloads)} / {len(payloads)}")
    print(f"💰 Total Expected Input Tokens:  {total_input_tokens:,}")
    print(f"💰 Total Expected Output Tokens: {total_output_tokens:,} (approx. baseline + reasoning)")
    print(f"💰 Total Estimated Cost:         ${total_cost:.6f} USD")
    print("💰 ==================================================\n")

    client = None
    results = []

    for payload in payloads:
        cluster_id = payload.get("cluster_id", "Unknown")
        if use_cache and cluster_id in cache:
            results.append({
                "cluster_id": cluster_id,
                "name": cache[cluster_id]["name"],
                "description": cache[cluster_id]["description"]
            })
            continue

        print(f"🤖 Generating brief description for {cluster_id}...")
        if client is None:
            client = _init_client()

        payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
        user_prompt = BRIEF_USER_PROMPT_TEMPLATE.format(payload_json=payload_json)

        try:
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=BRIEF_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                ),
                contents=user_prompt,
            )
            resp_text = response.text or "{}"
        except Exception as e:
            logger.error("Gemini API call failed for %s: %s", cluster_id, e)
            resp_text = "{}"

        parsed_success = False
        parsed_data = {}
        
        # Try parsing JSON
        try:
            parsed_data = json.loads(resp_text)
            parsed_success = True
        except Exception:
            # Fallback parsing
            clean_text = resp_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            clean_text = clean_text.strip()
            try:
                parsed_data = json.loads(clean_text)
                parsed_success = True
            except Exception:
                pass

        if parsed_success:
            name = parsed_data.get("name", f"Unnamed Cluster {cluster_id}")
            description = parsed_data.get("description", "No description generated.")
        else:
            name = f"Cluster {cluster_id}"
            description = resp_text if resp_text != "{}" else "API generation failed."

        results.append({
            "cluster_id": cluster_id,
            "name": name,
            "description": description,
        })

        # Save to cache immediately if successful
        if parsed_success or (resp_text != "{}"):
            cache[cluster_id] = {
                "name": name,
                "description": description
            }
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.warning("Failed to write brief descriptions cache: %s", e)

    return results
