# app/prices/coinbase_daily.py
from datetime import datetime, timedelta, timezone
import time
import requests
import pandas as pd
import numpy as np

BASE = "https://api.exchange.coinbase.com"
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD"]
GRANULARITY = 86400  # 1 day
HEADERS = {"User-Agent": "allyz-coinbase-daily/1.2"}

def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _fetch(product_id: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """
    Fetch daily candles in [start_utc, end_utc) for one product.
    Returns columns: date, product_id, open, high, low, close, volume
    """
    url = f"{BASE}/products/{product_id}/candles"
    params = {"granularity": GRANULARITY, "start": _iso_z(start_utc), "end": _iso_z(end_utc)}

    for attempt in range(5):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json() or []
            if not isinstance(data, list) or not data:
                return pd.DataFrame(columns=["date","product_id","open","high","low","close","volume"])
            df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
            df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
            df = df.sort_values("date").reset_index(drop=True)
            df["product_id"] = product_id
            return df[["date","product_id","open","high","low","close","volume"]]
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()

    return pd.DataFrame(columns=["date","product_id","open","high","low","close","volume"])

def get_last_n_days_excluding_today(n: int = 3, products = PRODUCTS) -> pd.DataFrame:
    """
    Returns daily candles for the last `n` fully completed UTC days (excludes today).
    For n=3 and now=2025-09-22 UTC, returns dates: 2025-09-21, 2025-09-20, 2025-09-19.
    """
    now_utc = datetime.now(timezone.utc)
    today_midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    end = today_midnight_utc                      # exclusive (this is "today", excluded)
    start = end - timedelta(days=n)               # inclusive start for last n full days

    frames = []
    for p in products:
        df = _fetch(p, start, end)                # window covers exactly the n desired days
        if df.empty:
            continue
        # Keep only the exact date range, just in case
        valid_dates = pd.date_range(start=start.date(), end=(end.date() - timedelta(days=1)), freq="D").date
        df = df[df["date"].isin(valid_dates)].copy()

        # Basic completeness + JSON-safety
        must = ["open","high","low","close","volume"]
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=must)
        df = df.where(pd.notnull(df), None)       # NaN -> None for JSON payloads

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date","product_id","open","high","low","close","volume"])

    out = pd.concat(frames, ignore_index=True)
    # enforce order & sort
    out = out[["date","product_id","open","high","low","close","volume"]]
    out = out.sort_values(["date","product_id"]).reset_index(drop=True)
    return out
