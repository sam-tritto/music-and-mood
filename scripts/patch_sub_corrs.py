"""
patch_sub_corrs.py
==================
Inserts the missing sub-cluster misery-index correlation cell into
music_and_mood_trends.ipynb.

The empty cell with id "b26613ba" sits between section 5.1 and 5.2.
We replace its empty source with the logic that builds:
  - sub_vol_shares  / sub_shares_ts
  - sub_corrs       (dict: compound_id → pearson_r)
  - sub_deltas      (dict: compound_id → delta string)
  - corr_df         (DataFrame used by the visualisation cells below)
"""

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "music_and_mood_trends.ipynb"
TARGET_CELL_ID = "b26613ba"

NEW_SOURCE = [
    "# ── Sub-Cluster Misery-Index Correlations ──────────────────────────────────\n",
    "# Build a monthly volumetric-share time-series for each sub-cluster,\n",
    "# then compute static Pearson correlations against the Misery Index.\n",
    "#\n",
    "# Outputs\n",
    "# -------\n",
    "# sub_vol_shares  : DataFrame  – rows=year_month, cols=compound_id (int)\n",
    "# sub_shares_ts   : same but with a Timestamp index (used by visualisations)\n",
    "# sub_corrs       : dict  { compound_id (int) → pearson_r (float) }\n",
    "# sub_deltas      : dict  { compound_id (int) → delta_str }\n",
    "# corr_df         : DataFrame  – 'Sub-Cluster ID' + 'Misery Correlation (r)'\n",
    "\n",
    "from scipy import stats as _stats\n",
    "\n",
    "# ── 0. Derive a month series aligned to tracks_with_text ─────────────────────\n",
    "# Mirror the same logic used to build month_labels above:\n",
    "# prefer year_month → date → charts_df mapping.\n",
    "_twt = tracks_with_text.copy()\n",
    "\n",
    "if \"year_month\" in _twt.columns:\n",
    "    _twt[\"_month\"] = _twt[\"year_month\"]\n",
    "elif \"date\" in _twt.columns:\n",
    "    _twt[\"_month\"] = pd.to_datetime(_twt[\"date\"]).dt.to_period(\"M\")\n",
    "else:\n",
    "    # Fallback: map via charts_df (same as how month_labels was built)\n",
    "    _id_col = \"track_id\" if \"track_id\" in charts_df.columns else \"title\"\n",
    "    _track_months = charts_df.groupby(_id_col)[\"year_month\"].first()\n",
    "    _twt[\"_month\"] = _twt[_id_col].map(_track_months)\n",
    "\n",
    "# Drop any rows where we couldn't determine a month\n",
    "_twt = _twt.dropna(subset=[\"_month\"])\n",
    "\n",
    "# ── 1. Build monthly pivot ────────────────────────────────────────────────────\n",
    "_pivot = (\n",
    "    _twt\n",
    "    .groupby([\"_month\", \"sub_cluster\"])\n",
    "    .size()\n",
    "    .unstack(fill_value=0)\n",
    ")\n",
    "# Normalise to percentage shares\n",
    "sub_vol_shares = _pivot.div(_pivot.sum(axis=1), axis=0) * 100\n",
    "sub_vol_shares = sub_vol_shares[sorted(sub_vol_shares.columns)]\n",
    "\n",
    "# ── 2. Convert Period index → Timestamp for date-range slicing ────────────────\n",
    "sub_shares_ts = sub_vol_shares.copy()\n",
    "if hasattr(sub_shares_ts.index, \"to_timestamp\"):\n",
    "    sub_shares_ts.index = sub_shares_ts.index.to_timestamp()\n",
    "else:\n",
    "    sub_shares_ts.index = pd.to_datetime(sub_shares_ts.index)\n",
    "\n",
    "# ── 3. Align with the Misery Index date range ─────────────────────────────────\n",
    "_misery_dates = pd.to_datetime(misery_df[\"date\"])\n",
    "_start = max(sub_shares_ts.index.min(), _misery_dates.min())\n",
    "_end   = min(sub_shares_ts.index.max(), _misery_dates.max())\n",
    "\n",
    "_sub_filtered = sub_shares_ts.loc[_start:_end]\n",
    "_mis_filtered = misery_df[\n",
    "    (_misery_dates >= _start) & (_misery_dates <= _end)\n",
    "].set_index(\"date\")[\"misery_index\"]\n",
    "_mis_filtered.index = pd.to_datetime(_mis_filtered.index)\n",
    "\n",
    "# ── 4. Pearson r per sub-cluster ──────────────────────────────────────────────\n",
    "sub_corrs  = {}\n",
    "sub_deltas = {}\n",
    "_corr_rows = []\n",
    "\n",
    "for compound_id in sorted(sub_vol_shares.columns):\n",
    "    parent_id = compound_id // 10\n",
    "    sub_id    = compound_id % 10\n",
    "    label     = f\"Sub-cluster {parent_id}.{sub_id}\"\n",
    "\n",
    "    # Align series on the shared date index\n",
    "    share_series = _sub_filtered[compound_id].dropna()\n",
    "    joined = share_series.to_frame(\"share\").join(\n",
    "        _mis_filtered.rename(\"misery\"), how=\"inner\"\n",
    "    ).dropna()\n",
    "\n",
    "    if len(joined) >= 5:\n",
    "        r, _ = _stats.pearsonr(joined[\"share\"], joined[\"misery\"])\n",
    "    else:\n",
    "        r = 0.0\n",
    "\n",
    "    sub_corrs[compound_id] = round(r, 4)\n",
    "    _corr_rows.append({\"Sub-Cluster ID\": label, \"Misery Correlation (r)\": round(r, 4)})\n",
    "\n",
    "    # Volume-share delta: mean of first half vs second half of the full series\n",
    "    full_series = sub_shares_ts[compound_id].dropna()\n",
    "    if len(full_series) > 1:\n",
    "        mid = len(full_series) // 2\n",
    "        delta = full_series.iloc[mid:].mean() - full_series.iloc[:mid].mean()\n",
    "        sub_deltas[compound_id] = f\"{'+' if delta > 0 else ''}{delta:.1f}% shift (early \\u2192 late period)\"\n",
    "    else:\n",
    "        sub_deltas[compound_id] = \"Insufficient data\"\n",
    "\n",
    "# ── 5. Summary DataFrame for the visualisation cells below ───────────────────\n",
    "corr_df = pd.DataFrame(_corr_rows)\n",
    "sub_corr_df = corr_df  # alias used by some downstream cells\n",
    "\n",
    "print(f\"\\u2705 Sub-cluster correlations computed for {len(sub_corrs)} sub-clusters\")\n",
    "print(f\"   Date overlap: {_start.strftime('%Y-%m')} \\u2192 {_end.strftime('%Y-%m')}\")\n",
    "display(corr_df.sort_values(\"Misery Correlation (r)\", ascending=False).reset_index(drop=True))\n",
]

# ── Patch ────────────────────────────────────────────────────────────────────
nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

patched = False
for cell in nb["cells"]:
    if cell.get("id") == TARGET_CELL_ID:
        if cell["source"]:
            print(f"⚠️  Cell {TARGET_CELL_ID} already has source — overwriting.")
        cell["source"] = NEW_SOURCE
        cell["outputs"] = []
        cell["execution_count"] = None
        patched = True
        print(f"✅  Patched cell {TARGET_CELL_ID}")
        break

if not patched:
    print(f"❌  Cell with id={TARGET_CELL_ID!r} not found in notebook!")
else:
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"💾  Notebook saved: {NOTEBOOK}")
