"""
FRED Macroeconomic Data Fetcher
================================
Fetches unemployment (UNRATE) and CPI (CPIAUCSL) from the Federal Reserve
Economic Data API, then engineers Arthur Okun's Misery Index:

    Misery Index = Unemployment Rate + Year-over-Year CPI % Change

This gives us a single scalar measuring economic pain that we correlate
against music cluster dynamics.
"""

from __future__ import annotations

import logging

import pandas as pd
from fredapi import Fred

from cmi.config import DATE_END, DATE_START, FRED_API_KEY

logger = logging.getLogger(__name__)


def fetch_misery_index(
    start: str = DATE_START,
    end: str = DATE_END,
    api_key: str = FRED_API_KEY,
) -> pd.DataFrame:
    """
    Fetch UNRATE and CPIAUCSL from FRED, compute the Misery Index.

    Parameters
    ----------
    start / end : ISO date strings for the observation window.
    api_key : FRED API key (from .env)

    Returns
    -------
    pd.DataFrame with columns:
        date, unemployment_rate, cpi, yoy_inflation, misery_index
    Monthly frequency, indexed by date.
    """
    if not api_key:
        raise ValueError(
            "FRED_API_KEY is not set. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "and add it to your .env file."
        )

    fred = Fred(api_key=api_key)
    logger.info("Fetching UNRATE from FRED (%s to %s)...", start, end)
    unrate = fred.get_series(
        "UNRATE",
        observation_start=start,
        observation_end=end,
    )

    logger.info("Fetching CPIAUCSL from FRED (%s to %s)...", start, end)
    cpi = fred.get_series(
        "CPIAUCSL",
        observation_start=start,
        observation_end=end,
    )

    # Build DataFrame
    df = pd.DataFrame({
        "unemployment_rate": unrate,
        "cpi": cpi,
    })

    # Resample to monthly (UNRATE is already monthly, CPI is monthly)
    df.index = pd.to_datetime(df.index)
    df = df.resample("MS").first().ffill()

    # Year-over-Year inflation: % change in CPI vs 12 months prior
    df["yoy_inflation"] = df["cpi"].pct_change(periods=12) * 100

    # Okun's Misery Index = Unemployment + YoY Inflation
    df["misery_index"] = df["unemployment_rate"] + df["yoy_inflation"]

    # Clean up
    df = df.dropna(subset=["misery_index"])
    df = df.reset_index()
    df = df.rename(columns={"index": "date"})

    logger.info(
        "Misery Index computed: %d months, range %.1f – %.1f",
        len(df),
        df["misery_index"].min(),
        df["misery_index"].max(),
    )

    return df
