"""
patch_cluster_viz.py
====================
Fixes the section 5.3 cell (id: a12c2461) so it shows top-level CLUSTER
correlations instead of sub-cluster ones.

Uses:
  - static_corr          : DataFrame with cluster pearson_r values
  - cluster_name_mapping : dict {cluster_id (int) -> catchy_name}
  - brief_descriptions   : list of dicts with cluster narrative info
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "music_and_mood_trends.ipynb"
TARGET_CELL_ID = "a12c2461"

NEW_SOURCE = [
    "# 1. Build cluster name mapping from generated brief descriptions\n",
    "cluster_name_mapping = {d[\"cluster_id\"]: d[\"name\"] for d in brief_descriptions}\n",
    "\n",
    "# 2. Build a tidy correlation DataFrame from static_corr\n",
    "#    static_corr has columns: cluster, pearson_r, pearson_p, spearman_r, spearman_p\n",
    "cluster_corr_df = static_corr[[\"cluster\", \"pearson_r\"]].copy()\n",
    "cluster_corr_df[\"cluster_id\"] = cluster_corr_df[\"cluster\"].str.replace(\"cluster_\", \"\").astype(int)\n",
    "cluster_corr_df[\"Catchy Name\"] = cluster_corr_df[\"cluster_id\"].map(cluster_name_mapping)\n",
    "cluster_corr_df[\"Label\"] = cluster_corr_df.apply(\n",
    "    lambda r: f\"{r['Catchy Name']} (Cluster {r['cluster_id']})\"\n",
    "    if pd.notna(r[\"Catchy Name\"]) else f\"Cluster {r['cluster_id']}\",\n",
    "    axis=1,\n",
    ")\n",
    "cluster_corr_df = cluster_corr_df.rename(columns={\"pearson_r\": \"Misery Correlation (r)\"})\n",
    "cluster_corr_df = cluster_corr_df.sort_values(\"Misery Correlation (r)\")\n",
    "\n",
    "# 3. Visualize cluster correlations to the Misery Index\n",
    "fig = px.bar(\n",
    "    cluster_corr_df,\n",
    "    x=\"Label\",\n",
    "    y=\"Misery Correlation (r)\",\n",
    "    color=\"Misery Correlation (r)\",\n",
    "    color_continuous_scale=px.colors.diverging.RdBu_r,\n",
    "    title=\"Cluster Correlation to Okun's Misery Index (r)\",\n",
    "    labels={\"Misery Correlation (r)\": \"Pearson r\", \"Label\": \"Cluster (Vibe)\"},\n",
    "    template=\"plotly_dark\",\n",
    "    height=450,\n",
    ")\n",
    "fig.update_layout(\n",
    "    coloraxis_showscale=False,\n",
    "    xaxis_tickangle=-45,\n",
    "    margin=dict(b=160),\n",
    "    title_x=0.5,\n",
    "    title_xanchor=\"center\",\n",
    ")\n",
    "fig.show()\n",
]

# ── Patch ─────────────────────────────────────────────────────────────────────
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

patched = False
for cell in nb["cells"]:
    if cell.get("id") == TARGET_CELL_ID:
        cell["source"] = NEW_SOURCE
        cell["outputs"] = []          # clear stale sub-cluster output
        cell["execution_count"] = None
        patched = True
        print(f"✅  Patched cell {TARGET_CELL_ID}")
        break

if not patched:
    print(f"❌  Cell {TARGET_CELL_ID!r} not found!")
else:
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"💾  Notebook saved: {NOTEBOOK}")
