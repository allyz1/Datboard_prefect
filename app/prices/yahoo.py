import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from typing import Iterable, List

def _today_utc_date():
    return datetime.now(timezone.utc).date()

def get_last_n_days_excluding_today_yf(tickers: Iterable[str], n: int = 3) -> pd.DataFrame:
    tickers = list(tickers)
    cols = ["date","ticker","open","close","adj_close","volume"]
    if not tickers:
        return pd.DataFrame(columns=cols)

    extra = max(7, n + 3)
    period_str = f"{n + extra}d"

    def _norm_single(df: pd.DataFrame, t: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        out = df.reset_index().rename(columns={
            "Date":"date","Open":"open","Close":"close","Adj Close":"adj_close","Volume":"volume"
        })
        out["ticker"] = t
        out["date"] = pd.to_datetime(out["date"]).dt.date
        out = out[out["date"] < _today_utc_date()]
        for c in ["open","close","adj_close","volume"]:
            if c in out: out[c] = pd.to_numeric(out[c], errors="coerce")
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        for c in cols:
            if c not in out.columns: out[c] = np.nan
        return out[cols].sort_values("date").tail(n)

    # Try batched first (without repair)
    raw = yf.download(
        tickers=" ".join(tickers),
        period=period_str,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
        timeout=30,
        # repair=False  # default
    )

    frames: List[pd.DataFrame] = []
    if isinstance(raw.columns, pd.MultiIndex):
        present = set(raw.columns.get_level_values(0))
        for t in tickers:
            df_t = raw[t] if t in present else pd.DataFrame()
            frames.append(_norm_single(df_t, t))
    else:
        # If batched failed (e.g., empty or single block), try per-ticker fallback
        if raw.empty and len(tickers) > 1:
            for t in tickers:
                try:
                    solo = yf.download(
                        tickers=t, period=period_str, interval="1d",
                        auto_adjust=False, actions=False, threads=False,
                        progress=False, timeout=30
                    )
                except Exception:
                    solo = pd.DataFrame()
                frames.append(_norm_single(solo, t))
        else:
            t = tickers[0]
            frames.append(_norm_single(raw, t))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
    return out.sort_values(["ticker","date"], ignore_index=True)



