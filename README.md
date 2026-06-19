# 🎵 The Cultural Misery Index (CMI)

**Multimodal Semantic Audio Clustering × Macroeconomic Shock Correlation × GenAI Narrative Engine**

> *Does music reveal the collective psyche of a nation under economic stress?*

This project fuses three data streams — **Spotify streaming trends**, **Genius lyric semantics**, and **FRED macroeconomic indicators** — to track how American music tastes shift in response to economic pressure. Instead of static genres, we cluster songs purely on their auditory "vibe" and lyrical themes, then measure whether clusters like "High-Energy Melancholic Pop" expand during inflation spikes.

## Architecture

```
[FRED API: Misery Index] ───┐
                            ▼
[Kaggle: Audio Features] ──► [Feature Fusion] ──► [Temporal Clustering] ──► [Econometric Mapping] ──► [GenAI Narrator]
[Genius API: Lyrics]     ──►  (UMAP/PCA)           (Hungarian Algo)          (Rolling Correlation)      (Gemini 3.5 Flash)
```

## Pipeline

| Phase | What | How |
|-------|------|-----|
| **1. Ingestion** | Spotify charts, lyrics, economic data | Kaggle CSV + Genius API + FRED API |
| **2. Feature Fusion** | Combine audio (7D) + lyric semantics (768D→10D) | Gemini Embeddings + UMAP + StandardScaler |
| **3. Temporal Clustering** | Monthly K-Means with identity tracking | Hungarian Algorithm (linear_sum_assignment) |
| **4. Econometric Analysis** | Map cluster dynamics vs. Misery Index | Rolling Pearson/Spearman + Granger Causality |
| **5. Narrative Engine** | AI-generated cultural digest | Gemini 3.5 Flash |

## Quick Start

```bash
# 1. Set up the environment
uv sync

# 2. Configure API keys
cp .env.example .env
# Edit .env with your keys (Google, Genius, FRED)

# 3. Download Spotify data from Kaggle
# First: set up Kaggle credentials (see scripts/download_kaggle_data.py)
uv run python scripts/download_kaggle_data.py

# 4. Open the tutorial notebook
uv run jupyter notebook music_and_mood_trends.ipynb
```

## API Keys (all free tier)

| Service | Get Key | Used For |
|---------|---------|----------|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com/apikey) | Lyric embeddings + narrative generation |
| Genius | [genius.com/api-clients](https://genius.com/api-clients) | Song lyrics |
| FRED | [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) | Unemployment + CPI data |
| Kaggle | [kaggle.com/settings](https://www.kaggle.com/settings) | Dataset download (optional) |

## Project Structure

```
src/cmi/
├── config.py              # API keys, paths, constants
├── ingest/
│   ├── spotify_charts.py  # Kaggle CSV loader
│   ├── genius_lyrics.py   # Genius API lyrics fetcher
│   └── fred_macro.py      # FRED API → Misery Index
├── features/
│   ├── audio.py           # Audio feature extraction
│   ├── embeddings.py      # Gemini lyric embeddings
│   └── fusion.py          # UMAP + concatenation + scaling
├── clustering/
│   ├── temporal.py        # Monthly K-Means + Hungarian alignment
│   └── diagnostics.py     # Elbow, silhouette, stability
├── correlation/
│   └── econometrics.py    # Rolling correlation + Granger causality
└── narrative/
    └── engine.py          # Gemini narrative generation
```

## Key Concepts

- **The Misery Index**: Arthur Okun's formula — Unemployment Rate + Year-over-Year Inflation. When it spikes, people are hurting.
- **Hungarian Algorithm**: Solves the label-switching problem in temporal clustering by finding optimal 1-to-1 centroid mappings across time windows.
- **Multimodal Fusion**: Audio features (what it *sounds* like) + lyric embeddings (what it *says*) — because Hey Ya! sounds happy but is actually about a failing relationship.
