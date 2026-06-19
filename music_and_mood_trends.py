# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🎵 The Cultural Misery Index
#
# **Multimodal Semantic Audio Clustering × Macroeconomic Shock Correlation × GenAI Narrative Engine**
#
# > *Does music reveal the collective psyche of a nation under economic stress?*
#
# This notebook fuses three data streams — **Spotify streaming trends**, **Genius lyric
# semantics**, and **FRED macroeconomic indicators** — to track how American music tastes
# shift in response to economic pressure.
#
# Instead of static genres, we cluster songs purely on their auditory "vibe" and lyrical
# themes, then measure whether clusters like "High-Energy Melancholic Pop" expand during
# inflation spikes or unemployment surges.
#
# ---
#
# ## Pipeline Overview
#
# ```
# [FRED API: Misery Index] ───┐
#                             ▼
# [Kaggle: Audio Features] ──► [Feature Fusion] ──► [Temporal Clustering] ──► [Econometric Mapping] ──► [GenAI Narrator]
# [Genius API: Lyrics]     ──►  (UMAP/PCA)           (Hungarian Algo)          (Rolling Correlation)      (Gemini 3.5 Flash)
# ```

# %% [markdown]
# ---
# ## 0. Setup & Configuration

# %%
# Core imports
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from IPython.display import display, Markdown

# Suppress noisy warnings for cleaner notebook output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="umap")

# Logging — set to INFO so we see pipeline progress
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# Matplotlib / Seaborn style
sns.set_theme(style="darkgrid", palette="viridis", font_scale=1.1)
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 100

print("✅ Core imports loaded")

# %%
# CMI pipeline imports
from cmi.config import (
    AUDIO_FEATURES, DATA_RAW, DATA_PROCESSED, K_CLUSTERS,
    REGION, DATE_START, DATE_END, EMBEDDING_DIM_TARGET,
)
from cmi.ingest.spotify_charts import load_charts, get_unique_tracks
from cmi.ingest.genius_lyrics import fetch_lyrics_batch
from cmi.ingest.fred_macro import fetch_misery_index
from cmi.features.audio import extract_audio_matrix, scale_audio, audio_feature_summary
from cmi.features.embeddings import embed_lyrics
from cmi.features.fusion import reduce_embeddings, fuse_features
from cmi.clustering.temporal import cluster_temporal_windows, ClusterTimeline
from cmi.clustering.diagnostics import elbow_plot, silhouette_analysis, cluster_stability_score
from cmi.correlation.econometrics import (
    compute_volumetric_shares, rolling_correlation,
    static_correlation, granger_causality, classify_clusters,
)
from cmi.narrative.engine import build_cluster_payload, generate_narrative, generate_full_report

print("✅ CMI pipeline modules loaded")
print(f"   Region: {REGION}")
print(f"   Date range: {DATE_START} → {DATE_END}")
print(f"   Cluster count (K): {K_CLUSTERS}")
print(f"   Audio features: {AUDIO_FEATURES}")

# %% [markdown]
# ---
# ## 1. Data Ingestion
#
# We pull from three independent data sources, all joined on a temporal axis
# (monthly intervals, 2018–2024):
#
# 1. **Spotify Charts** (Kaggle) — weekly Top 200 tracks with audio features
# 2. **Genius** — full lyric sheets for semantic analysis
# 3. **FRED** — unemployment + CPI → Okun's Misery Index

# %% [markdown]
# ### 1.1 Spotify Chart Data (Kaggle)
#
# Since Spotify restricted their audio features API for new apps in November 2024,
# we use a pre-compiled Kaggle dataset that was built before the lockdown.
#
# > **First time?** Run `uv run python scripts/download_kaggle_data.py` to download the data,
# > or manually place a Spotify charts CSV in `data/raw/`.

# %%
# Find the chart CSV in data/raw/
csv_files = list(DATA_RAW.glob("*.csv"))
print(f"📂 Found {len(csv_files)} CSV files in {DATA_RAW}:")
for f in csv_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"   {f.name} ({size_mb:.1f} MB)")

# %%
# Load the chart data — update the filename if your CSV is named differently
# The loader handles various Kaggle column name conventions automatically
CHART_CSV = DATA_RAW / "charts.csv"  # ← Adjust this filename to match your download

charts_df = load_charts(CHART_CSV, region=REGION, date_start=DATE_START, date_end=DATE_END)
print(f"\n📊 Chart data loaded: {charts_df.shape}")
display(charts_df.head(10))

# %%
# Quick data quality check
print("Column dtypes:")
print(charts_df.dtypes)
print(f"\nDate range: {charts_df['date'].min()} → {charts_df['date'].max()}")
print(f"Unique tracks: {charts_df['title'].nunique() if 'title' in charts_df.columns else 'N/A'}")
print(f"Unique months: {charts_df['year_month'].nunique() if 'year_month' in charts_df.columns else 'N/A'}")

# Audio feature coverage
for feat in AUDIO_FEATURES:
    if feat in charts_df.columns:
        null_pct = charts_df[feat].isna().mean() * 100
        print(f"   {feat}: {null_pct:.1f}% null")

# %%
# Audio feature distributions
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
available_feats = [f for f in AUDIO_FEATURES if f in charts_df.columns]

for i, feat in enumerate(available_feats):
    ax = axes.flat[i]
    charts_df[feat].dropna().hist(bins=50, ax=ax, color=sns.color_palette("viridis")[i % 6], alpha=0.8)
    ax.set_title(feat.title(), fontweight="bold")
    ax.set_xlabel("")

# Hide unused subplots
for j in range(len(available_feats), len(axes.flat)):
    axes.flat[j].set_visible(False)

plt.suptitle("Audio Feature Distributions — US Top 200 Tracks", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.2 FRED Macroeconomic Data — The Misery Index
#
# Arthur Okun's **Misery Index** = Unemployment Rate + Year-over-Year Inflation.
# When it spikes, people are hurting. We use this as our measure of economic stress.
#
# > **Key events in our window:**
# > - 2020 Q1–Q2: COVID-19 shock (unemployment spike to 14.7%)
# > - 2021–2022: Post-COVID inflation surge
# > - 2022 H2: Fed rate hikes, inflation cooling

# %%
# Fetch economic data from FRED
misery_df = fetch_misery_index()
print(f"📈 Misery Index: {len(misery_df)} monthly observations")
display(misery_df.head(10))

# %%
# Plot the Misery Index with key events annotated
fig, ax = plt.subplots(figsize=(16, 6))

ax.plot(misery_df["date"], misery_df["misery_index"], linewidth=2.5, color="#e74c3c", label="Misery Index")
ax.fill_between(misery_df["date"], misery_df["misery_index"], alpha=0.15, color="#e74c3c")

# Component breakdown
ax.plot(misery_df["date"], misery_df["unemployment_rate"], "--", linewidth=1.5, color="#3498db", alpha=0.7, label="Unemployment Rate")
ax.plot(misery_df["date"], misery_df["yoy_inflation"], "--", linewidth=1.5, color="#f39c12", alpha=0.7, label="YoY Inflation")

# Annotate key events
ax.axvline(pd.Timestamp("2020-03-01"), color="gray", linestyle=":", alpha=0.5)
ax.text(pd.Timestamp("2020-04-01"), ax.get_ylim()[1] * 0.9, "COVID-19\nShock", fontsize=9, color="gray")

ax.axvline(pd.Timestamp("2022-06-01"), color="gray", linestyle=":", alpha=0.5)
ax.text(pd.Timestamp("2022-07-01"), ax.get_ylim()[1] * 0.9, "Peak\nInflation", fontsize=9, color="gray")

ax.set_title("Okun's Misery Index — US Economy (2018–2024)", fontsize=16, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("Index Value", fontsize=12)
ax.legend(loc="upper left", fontsize=11)
ax.grid(True, alpha=0.3)
sns.despine()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1.3 Genius Lyrics
#
# We fetch full lyric sheets for every unique track in our chart data.
# The Genius API is rate-limited, so this step caches results to disk
# and uses a 1.5-second delay between requests.
#
# > ⏱️ **First run warning**: Fetching lyrics for ~2,000+ unique tracks takes
# > approximately 1-2 hours due to rate limiting. Results are cached in
# > `data/raw/lyrics_cache/` so subsequent runs are instant.

# %%
# Get unique tracks for lyrics fetching
unique_tracks = get_unique_tracks(charts_df)
print(f"🎤 Unique tracks to fetch lyrics for: {len(unique_tracks)}")
display(unique_tracks.head(10))

# %%
# Fetch lyrics (cached — only hits API for uncached tracks)
# ⚠️ SLOW on first run (~1.5s per track). Comment this cell if you want to skip lyrics.
tracks_with_lyrics = fetch_lyrics_batch(unique_tracks)

lyrics_found = tracks_with_lyrics["lyrics"].notna().sum()
print(f"\n✅ Lyrics found: {lyrics_found} / {len(tracks_with_lyrics)} ({100*lyrics_found/len(tracks_with_lyrics):.1f}%)")

# Preview a sample
sample = tracks_with_lyrics[tracks_with_lyrics["lyrics"].notna()].head(3)
for _, row in sample.iterrows():
    print(f"\n🎵 {row['title']} — {row['artist']}")
    print(f"   {row['lyrics'][:200]}...")

# %% [markdown]
# ---
# ## 2. Feature Engineering — Multimodal Fusion
#
# This is where the magic happens. We combine two fundamentally different
# representations of each song:
#
# 1. **Audio features** (7D): What it *sounds* like — tempo, energy, valence, etc.
# 2. **Lyric embeddings** (768D → 10D): What it *says* — dense semantic vectors
#
# The challenge: raw 768D embeddings would dominate the 7 audio features in any
# distance metric. We use **UMAP** to compress lyrics to ~10 dimensions, preserving
# macro themes (isolation vs. partying vs. heartbreak) without overwhelming the audio.

# %% [markdown]
# ### 2.1 Lyric Embedding via Google Gemini

# %%
# Filter to tracks that have lyrics
tracks_with_text = tracks_with_lyrics[tracks_with_lyrics["lyrics"].notna()].copy()
lyrics_list = tracks_with_text["lyrics"].tolist()

print(f"📝 Embedding {len(lyrics_list)} lyric texts via gemini-embedding-001...")
lyric_embeddings = embed_lyrics(lyrics_list)
print(f"   Embedding matrix: {lyric_embeddings.shape}")

# %% [markdown]
# ### 2.2 Dimensionality Reduction (UMAP)

# %%
# UMAP: 768D → 10D
reduced_lyrics = reduce_embeddings(lyric_embeddings, n_components=EMBEDDING_DIM_TARGET)
print(f"📉 Reduced lyrics: {reduced_lyrics.shape}")

# Visualize the 2D UMAP projection for intuition
umap_2d = reduce_embeddings(lyric_embeddings, n_components=2)

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(
    umap_2d[:, 0], umap_2d[:, 1],
    c=tracks_with_text["valence"].values if "valence" in tracks_with_text.columns else "steelblue",
    cmap="RdYlGn", alpha=0.6, s=15,
)
ax.set_title("2D UMAP of Lyric Embeddings\n(colored by Valence: Red=Sad, Green=Happy)", fontsize=14, fontweight="bold")
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
plt.colorbar(scatter, label="Valence", ax=ax)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 2.3 Audio Feature Extraction & Fusion

# %%
# Extract and scale audio features for tracks WITH lyrics
audio_matrix = extract_audio_matrix(tracks_with_text)
print(f"🔊 Audio matrix: {audio_matrix.shape}")

# Fuse: audio (7D) + reduced lyrics (10D) → 17D, StandardScaled
fused_matrix, fused_scaler = fuse_features(audio_matrix, reduced_lyrics)
print(f"🔗 Fused matrix: {fused_matrix.shape}")
print(f"   Feature dimensions: {audio_matrix.shape[1]} audio + {reduced_lyrics.shape[1]} lyric = {fused_matrix.shape[1]} total")

# %% [markdown]
# ---
# ## 3. Temporal Clustering — Tracking Vibe Archetypes Over Time
#
# Instead of clustering all songs at once (which would hide temporal dynamics),
# we cluster within **monthly time windows** and use the **Hungarian Algorithm**
# to maintain consistent cluster identities across months.
#
# The label-switching problem: K-Means assigns arbitrary labels each run.
# "High-Energy Melancholic Pop" might be Cluster 2 in January but Cluster 5
# in February. The Hungarian algorithm finds the optimal 1-to-1 mapping between
# old and new centroids based on Euclidean distance.

# %% [markdown]
# ### 3.1 Optimal K Selection

# %%
# Elbow plot + silhouette analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

elbow_plot(fused_matrix, k_range=range(2, 12), ax=ax1)
_, sil_scores = silhouette_analysis(fused_matrix, k_range=range(2, 12), ax=ax2)

plt.suptitle("Cluster Count Diagnostics", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

best_k = max(sil_scores, key=sil_scores.get)
print(f"\n🏆 Best K by silhouette score: {best_k} (score: {sil_scores[best_k]:.4f})")
print(f"   Using K={K_CLUSTERS} as specified in config")

# %% [markdown]
# ### 3.2 Temporal Clustering with Hungarian Alignment

# %%
# We need month labels aligned to the fused matrix
# The fused matrix corresponds to tracks_with_text (tracks that have lyrics)
month_labels = tracks_with_text["year_month"].values if "year_month" in tracks_with_text.columns else None

# If we don't have year_month from the unique tracks, derive it from charts_df
# This maps each track to its most common month
if month_labels is None or len(month_labels) != len(fused_matrix):
    # Use the chart date for temporal windowing
    # For unique tracks, assign the month of their peak chart position
    if "date" in tracks_with_text.columns:
        month_labels = pd.to_datetime(tracks_with_text["date"]).dt.to_period("M").values
    else:
        # Fallback: generate monthly labels from charts_df
        print("⚠️ No date column in unique tracks — using charts_df for temporal mapping")
        # Create a mapping of track → most common month
        id_col = "track_id" if "track_id" in charts_df.columns else "title"
        track_months = charts_df.groupby(id_col)["year_month"].first()
        month_labels = tracks_with_text[id_col].map(track_months).values

print(f"📅 Month labels: {len(month_labels)} entries, {pd.Series(month_labels).nunique()} unique months")

# %%
# Run temporal clustering
streams = tracks_with_text["streams"].values if "streams" in tracks_with_text.columns else None

timeline = cluster_temporal_windows(
    fused_matrix=fused_matrix,
    months=month_labels,
    k=K_CLUSTERS,
    streams=streams,
)

print(f"\n✅ Temporal clustering complete:")
print(f"   Months tracked: {len(timeline.centroids_by_month)}")
print(f"   Label distribution: {pd.Series(timeline.labels).value_counts().to_dict()}")

# %%
# Cluster stability analysis
stability = cluster_stability_score(timeline)
print("📊 Cluster Stability (lower drift = more stable identity):")
display(stability)

# %% [markdown]
# ### 3.3 Cluster Evolution Visualization

# %%
# Volumetric shares over time — the core visualization
if timeline.volumetric_shares is not None:
    shares = timeline.volumetric_shares.copy()

    # Plotly area chart for interactive exploration
    fig = go.Figure()

    colors = px.colors.qualitative.Set2[:K_CLUSTERS]
    for i, col in enumerate(shares.columns):
        fig.add_trace(go.Scatter(
            x=shares.index.astype(str),
            y=shares[col],
            mode="lines",
            name=col.replace("cluster_", "Cluster "),
            stackgroup="one",
            line=dict(width=0.5),
            fillcolor=colors[i % len(colors)],
        ))

    fig.update_layout(
        title="Cluster Volumetric Shares Over Time (% of Top 200 Streams)",
        xaxis_title="Month",
        yaxis_title="Share (%)",
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()

# %%
# Centroid drift over time — how fast are cluster identities changing?
if timeline.drift_velocities:
    drift_df = pd.DataFrame(timeline.drift_velocities).T
    drift_df.columns = [f"Cluster {i}" for i in range(drift_df.shape[1])]

    fig = px.line(
        drift_df,
        title="Centroid Drift Velocity — How Fast Are 'Vibes' Shifting?",
        labels={"value": "Euclidean Drift", "index": "Month"},
        template="plotly_dark",
        height=400,
    )
    fig.show()

# %% [markdown]
# ---
# ## 4. Econometric Analysis — Music vs. The Economy
#
# Now we have stable clusters tracked over time. The key question:
# **Do certain musical "vibes" expand or contract in response to economic stress?**
#
# We test this with:
# - **Rolling Pearson/Spearman Correlation** — ongoing relationship strength
# - **Granger Causality** — does economic pain *precede* music shifts?

# %% [markdown]
# ### 4.1 Merge Cluster Dynamics with Misery Index

# %%
# Get cluster volumetric shares as a time series
cluster_shares = compute_volumetric_shares(timeline)
print(f"📊 Cluster shares: {cluster_shares.shape}")
display(cluster_shares.head())

# %%
# Static (overall) correlation between each cluster and the Misery Index
static_corr = static_correlation(cluster_shares, misery_df)
print("\n📐 Static Correlations (Cluster Share vs. Misery Index):")
display(static_corr)

# %%
# Classify clusters by economic sensitivity
classified = classify_clusters(static_corr)
print("\n🏷️ Cluster Economic Sensitivity Classification:")
display(classified[["cluster", "pearson_r", "economic_sensitivity"]])

# %% [markdown]
# ### 4.2 Rolling Correlation — How the Relationship Evolves

# %%
# Rolling correlation (6-month window)
rolling_corr = rolling_correlation(cluster_shares, misery_df, window=6)

fig = px.line(
    rolling_corr,
    title="Rolling 6-Month Correlation: Cluster Shares vs. Misery Index",
    labels={"value": "Pearson r", "index": "Month"},
    template="plotly_dark",
    height=450,
)
fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3)
fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3)
fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3)
fig.show()

# %% [markdown]
# ### 4.3 Granger Causality — Does Economic Pain Predict Music Shifts?

# %%
# Granger causality test (up to 3-month lag)
granger_results = granger_causality(cluster_shares, misery_df, max_lag=3)
print("🔬 Granger Causality Results (p < 0.05 = significant):")
display(granger_results)

# Highlight significant results
significant = granger_results[granger_results["p_value"] < 0.05]
if len(significant) > 0:
    print(f"\n🎯 Significant Granger-causal relationships found:")
    for _, row in significant.iterrows():
        print(f"   Misery Index → {row['cluster']} (lag={row['lag']}, p={row['p_value']:.4f})")
else:
    print("\n   No significant Granger-causal relationships at p < 0.05")

# %% [markdown]
# ### 4.4 The Cultural Misery Index Dashboard

# %%
# Overlay: Cluster shares + Misery Index on the same timeline
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Cluster Volumetric Shares (%)", "Misery Index"),
    row_heights=[0.65, 0.35],
)

# Top panel: stacked cluster shares
colors = px.colors.qualitative.Set2[:K_CLUSTERS]
for i, col in enumerate(cluster_shares.columns):
    fig.add_trace(
        go.Scatter(
            x=cluster_shares.index,
            y=cluster_shares[col],
            name=col.replace("cluster_", "Cluster "),
            stackgroup="one",
            line=dict(width=0.5),
            fillcolor=colors[i % len(colors)],
        ),
        row=1, col=1,
    )

# Bottom panel: Misery Index
fig.add_trace(
    go.Scatter(
        x=misery_df["date"],
        y=misery_df["misery_index"],
        name="Misery Index",
        line=dict(color="#e74c3c", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(231, 76, 60, 0.15)",
    ),
    row=2, col=1,
)

fig.update_layout(
    title="🎵 The Cultural Misery Index Dashboard",
    template="plotly_dark",
    height=700,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    showlegend=True,
)
fig.show()

# %% [markdown]
# ---
# ## 5. GenAI Narrative Engine — The Story Behind the Data
#
# Instead of just plotting charts, we generate a structured data payload for each
# cluster and prompt **Gemini 3.5 Flash** to write the cultural digest.
#
# The LLM acts as an "economic sociologist and cultural critic," interpreting
# what the sonic and semantic properties tell us about consumer coping mechanisms.

# %% [markdown]
# ### 5.1 Build Cluster Payloads

# %%
# Build payloads for each cluster during key economic periods
payloads = []

# Identify key periods from the Misery Index
# (we'll pick the highest and lowest misery periods for contrast)
misery_df_sorted = misery_df.sort_values("misery_index", ascending=False)
high_misery_period = misery_df_sorted.head(6)  # Top 6 months of economic stress
low_misery_period = misery_df_sorted.tail(6)   # Bottom 6 months of economic calm

print("📊 High-stress economic periods:")
display(high_misery_period[["date", "misery_index", "unemployment_rate", "yoy_inflation"]])

print("\n📊 Low-stress economic periods:")
display(low_misery_period[["date", "misery_index", "unemployment_rate", "yoy_inflation"]])

# %%
# Build a payload for each cluster
for cluster_id in range(K_CLUSTERS):
    # Get data for this cluster
    mask = timeline.labels == cluster_id
    cluster_data = tracks_with_text[mask].copy() if sum(mask) > 0 else pd.DataFrame()

    if len(cluster_data) == 0:
        continue

    # Get centroid (from the last available month)
    last_month = sorted(timeline.centroids_by_month.keys())[-1]
    centroids = timeline.centroids_by_month[last_month]
    centroid = centroids[cluster_id] if cluster_id < len(centroids) else np.zeros(fused_matrix.shape[1])

    # Get correlation
    cluster_col = f"cluster_{cluster_id}"
    corr_row = static_corr[static_corr["cluster"] == cluster_col]
    corr_val = corr_row["pearson_r"].values[0] if len(corr_row) > 0 else 0.0

    # Volume share delta
    if timeline.volumetric_shares is not None:
        shares = timeline.volumetric_shares[cluster_col]
        if len(shares) > 1:
            first_half = shares.iloc[:len(shares)//2].mean()
            second_half = shares.iloc[len(shares)//2:].mean()
            delta = second_half - first_half
            delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}% shift (early → late period)"
        else:
            delta_str = "Insufficient data"
    else:
        delta_str = "No volumetric data"

    payload = build_cluster_payload(
        cluster_id=cluster_id,
        cluster_data=cluster_data,
        centroid=centroid,
        temporal_window=f"{DATE_START[:4]}–{DATE_END[:4]} (Full Period)",
        misery_correlation=corr_val,
        volume_share_delta=delta_str,
    )
    payloads.append(payload)

print(f"📦 Built {len(payloads)} cluster payloads")

# Preview one
import json
print("\nSample payload:")
print(json.dumps(payloads[0], indent=2))

# %% [markdown]
# ### 5.2 Generate Cultural Narratives

# %%
# Generate the full Cultural Misery Index report
print("🤖 Generating narratives via Gemini 3.5 Flash...\n")
report = generate_full_report(payloads)

# Display the report as rendered markdown
display(Markdown(report))

# %%
# Save the report to disk
report_path = DATA_PROCESSED / "cultural_misery_index_report.md"
report_path.write_text(report)
print(f"💾 Report saved to: {report_path}")

# %% [markdown]
# ---
# ## 6. Key Findings & Interpretation
#
# The Cultural Misery Index reveals how music acts as a collective emotional
# barometer. Key patterns to look for:
#
# - **Reactive clusters** (positive Misery correlation): These "vibes" expand
#   when people are stressed — potentially serving as escapism or emotional processing
#
# - **Counter-cyclical clusters** (negative Misery correlation): These contract
#   during stress — perhaps too frivolous or optimistic for hard times
#
# - **Stable clusters** (low correlation): Enduring musical archetypes that
#   persist regardless of economic conditions
#
# - **The COVID Signature**: The 2020 unemployment spike should produce a
#   visible discontinuity in cluster dynamics — look for rapid shifts in
#   the dashboard around March–June 2020
#
# - **The Inflation Wave**: 2021–2022 saw sustained economic anxiety — does
#   this correlate with a gradual drift in dominant clusters?

# %%
# Final summary table
print("📋 Cluster Summary:")
summary_rows = []
for payload in payloads:
    row = {
        "Cluster": payload["cluster_id"],
        "Tracks": payload["n_tracks"],
        "Misery Correlation": payload["economic_metrics"]["misery_index_correlation"],
        "Volume Delta": payload["economic_metrics"]["volume_share_delta"],
        "Top Keywords": ", ".join(payload.get("top_lyric_keywords", [])[:5]),
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
display(summary_df)

print("\n🎵 The Cultural Misery Index analysis is complete.")
print("   Run the notebook from top to bottom to reproduce all results.")
