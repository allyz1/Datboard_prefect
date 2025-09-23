# app/prices/coinbase_daily.py
from datetime import datetime, timedelta, timezone
import time
import requests
import pandas as pd
import numpy as np

BASE = "https://api.exchange.coinbase.com"
# You can now list base assets only:
PRODUCTS = ["BTC", "ETH", "SOL"]
GRANULARITY = 86400  # 1 day
HEADERS = {"User-Agent": "allyz-coinbase-daily/1.4"}

def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def _endpoint_id(asset_or_pair: str) -> str:
    """
    Accepts 'BTC' or 'BTC-USD' and returns a valid Coinbase product id for the API.
    We default USD if not provided.
    """
    return asset_or_pair if "-" in asset_or_pair else f"{asset_or_pair}-USD"

def _base_asset(asset_or_pair: str) -> str:
    """'BTC-USD' -> 'BTC', 'BTC' -> 'BTC'"""
    return asset_or_pair.split("-")[0].upper()

def _fetch(product: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """
    Fetch daily candles in [start_utc, end_utc) for one product.
    Returns columns: date, product_id (base asset), open, close
    """
    pid = _endpoint_id(product)
    url = f"{BASE}/products/{pid}/candles"
    params = {"granularity": GRANULARITY, "start": _iso_z(start_utc), "end": _iso_z(end_utc)}

    for attempt in range(5):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json() or []
            if not isinstance(data, list) or not data:
                return pd.DataFrame(columns=["date", "product_id", "open", "close"])
            df = pd.DataFrame(data, columns=["time","low","high","open","close","volume"])
            df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.date
            df = df.sort_values("date").reset_index(drop=True)

            # keep only what we need, and rename product_id to base asset
            df["product_id"] = _base_asset(product)
            df = df[["date","product_id","open","close"]]
            return df

        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()

    return pd.DataFrame(columns=["date", "product_id", "open", "close"])

def get_last_n_days_excluding_today(n: int = 3, products = PRODUCTS) -> pd.DataFrame:
    """
    Returns daily candles for the last `n` fully completed UTC days (excludes today).
    For n=3 and now=2025-09-22 UTC, returns dates: 2025-09-21, 2025-09-20, 2025-09-19.
    """
    now_utc = datetime.now(timezone.utc)
    today_midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    end = today_midnight_utc
    start = end - timedelta(days=n)

    valid_dates = pd.date_range(start=start.date(), end=(end.date() - timedelta(days=1)), freq="D").date

    frames = []
    for p in products:
        df = _fetch(p, start, end)
        if df.empty:
            continue
        # precise date filter
        df = df[df["date"].isin(valid_dates)].copy()

        # numeric coercion + drop incomplete
        must = ["open","close"]
        df[must] = df[must].apply(pd.to_numeric, errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=must, inplace=True)

        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["date","product_id","open","close"])

    out = pd.concat(frames, ignore_index=True)

    # final sanitize
    out[["open","close"]] = out[["open","close"]].apply(pd.to_numeric, errors="coerce")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.dropna(subset=["open","close"], inplace=True)

    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["product_id"] = out["product_id"].astype(str)

    out = out[["date","product_id","open","close"]].sort_values(["date","product_id"]).reset_index(drop=True)
    assert not out.isnull().any().any(), "NaN detected after sanitization."
    return out
