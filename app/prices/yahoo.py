#!/usr/bin/env python3
# pip install yfinance pandas numpy python-dateutil
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional
import numpy as np
import pandas as pd
import yfinance as yf

def _utc_midnight(dt: Optional[datetime] = None) -> datetime:
    now = dt or datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)

def get_last_n_days_excluding_today_yf(
    tickers: Iterable[str],
    n: int = 3,
) -> pd.DataFrame:
    """
    Returns daily OC + adj_close/volume for the last `n` fully completed UTC days,
    excluding today. Columns: ['date','ticker','open','close','adj_close','volume']
    """
    end = _utc_midnight()                 # exclusive (today), so today is excluded
    start = end - timedelta(days=n)       # inclusive start
    tickers = list(tickers)
    if not tickers:
        return pd.DataFrame(columns=["date","ticker","open","close","adj_close","volume"])

    raw = yf.download(
        tickers=" ".join(tickers),
        start=start,
        end=end,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
    )

    # Normalize to tidy DF
    if isinstance(raw.columns, pd.MultiIndex):
        frames: List[pd.DataFrame] = []
        for t in tickers:
            if t not in raw.columns.levels[0]:
                continue
            df_t = raw[t].copy()
            if df_t.empty:
                continue
            df_t.index = pd.DatetimeIndex(df_t.index).tz_localize(None)
            df_t = df_t.reset_index().rename(columns={
                "Date":"date", "Open":"open",
                "Close":"close", "Adj Close":"adj_close", "Volume":"volume"
            })
            df_t["ticker"] = t
            frames.append(df_t[["date","ticker","open","close","adj_close","volume"]])
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["date","ticker","open","close","adj_close","volume"]
        )
    else:
        # Single ticker
        t = tickers[0]
        out = raw.copy()
        if out.empty:
            return pd.DataFrame(columns=["date","ticker","open","close","adj_close","volume"])
        out.index = pd.DatetimeIndex(out.index).tz_localize(None)
        out = out.reset_index().rename(columns={
            "Date":"date", "Open":"open",
            "Close":"close", "Adj Close":"adj_close", "Volume":"volume"
        })
        out["ticker"] = t
        out = out[["date","ticker","open","close","adj_close","volume"]]

    if out.empty:
        return out

    # Keep only the exact dates we want (same as original)
    valid_dates = pd.date_range(start=start.date(), end=(end.date() - timedelta(days=3)), freq="D").date
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out = out[out["date"].isin(valid_dates)].copy()

    # Sanitize numerics (same behavior as original, just without high/low)
    numeric = ["open","close","adj_close","volume"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    # original logic required these fields to exist
    out.dropna(subset=["open","close","volume"], inplace=True)

    # Sort
    out.sort_values(["ticker","date"], inplace=True, ignore_index=True)
    return out
