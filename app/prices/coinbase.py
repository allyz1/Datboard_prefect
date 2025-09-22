#!/usr/bin/env python3
# Requires: pip install requests pandas
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd
import time

BASE = "https://api.exchange.coinbase.com"
PRODUCTS = ["BTC-USD", "ETH-USD", "SOL-USD"]
GRANULARITY = 86400  # 1 day
HEADERS = {"User-Agent": "allyz-coinbase-daily/1.0"}

def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

def fetch_daily_window(product_id: str, start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    """
    Fetch daily candles for [start_utc, end_utc) at 1D granularity.
    Coinbase returns rows as [time, low, high, open, close, volume] in reverse-chronological order.
    """
    params = {
        "granularity": GRANULARITY,
        "start": _iso_z(start_utc),
        "end": _iso_z(end_utc),
    }
    url = f"{BASE}/products/{product_id}/candles"

    for attempt in range(5):
        r = requests.get(url, params=params, headers=HEADERS, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if not data:
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

    raise RuntimeError(f"Failed to fetch candles for {product_id}")

def get_yesterday_daily(products=PRODUCTS) -> pd.DataFrame:
    """
    Returns one row per product for the last fully completed UTC day ("yesterday" in UTC).
    This avoids partial candles and is ideal for a once-per-day Prefect flow.
    """
    now_utc = datetime.now(timezone.utc)
    # Yesterday as a closed UTC day
    y_start = (now_utc.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1))
    y_end   = y_start + timedelta(days=1)

    frames = []
    for p in products:
        # Window precisely covers yesterday -> yields exactly one daily candle if available
        df = fetch_daily_window(p, y_start, y_end)
        # Ensure only yesterday's date
        if not df.empty:
            df = df[df["date"] == y_start.date()]
        frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=["date","product_id","open","high","low","close","volume"])

if __name__ == "__main__":
    df = get_yesterday_daily()
    print(df)
