#!/usr/bin/env python3
# polygon_recent_multi.py
# Fetch last 3 days (UTC, through yesterday) of daily bars for a list of tickers.
# Returns a single DataFrame with only: ticker, date, open, high, low, close, volume, transactions, vwap.
# Prints a per-ticker debug summary. No CSV writes.

import os, time, json
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timezone, timedelta, date
from typing import Any, List

# ========= CONFIG =========
POLYGON_BASE = "https://api.polygon.io"
API_KEY = os.getenv("POLYGON_API_KEY", "BZAKLfIESg1ZfdJ0ldE49tplxBcj3jcW")  # ← prefer env var in prod
TICKERS = ["MSTR","CEP","SMLR","NAKA","BMNR","SBET","ETHZ","BTCS","SQNS","BTBT","DFDV","UPXI"]  # ← edit as needed
ADJUSTED = True     # adjusted prices
TIMEOUT = 20
MAX_RETRIES = 3
SAVE_JSON_DEBUG = False  # set True if you want raw Polygon JSON pages

# ========= DATES: last 3 days (UTC, through yesterday) =========
def _today_utc_date() -> date:
    return datetime.now(timezone.utc).date()

def _iso_ymd(d: date) -> str:
    return d.strftime("%Y-%m-%d")

today = _today_utc_date()
end_date = today - timedelta(days=1)        # through yesterday (UTC)
start_date = end_date - timedelta(days=2)   # last 3 calendar days (excl. today)

# Safety
if end_date < start_date:
    start_date = end_date

# ========= Utils =========
def _safe_request_id(req_id: Any) -> str:
    if req_id is None:
        return ""
    if isinstance(req_id, (str, int, float)):
        return str(req_id)
    try:
        return json.dumps(req_id, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(req_id)

def _save_json_debug(payload: dict, ticker: str, page_idx: int, start_d: date, end_d: date) -> None:
    if not SAVE_JSON_DEBUG:
        return
    dbg_dir = os.path.join("data", "debug")
    os.makedirs(dbg_dir, exist_ok=True)
    p = os.path.join(dbg_dir, f"polygon_raw_{ticker}_{_iso_ymd(start_d)}_to_{_iso_ymd(end_d)}_page{page_idx}.json")
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[debug] wrote raw JSON → {p}")
    except Exception as e:
        print(f"[warn] failed to write debug JSON: {e}")

# ========= Core fetch (pagination) =========
def fetch_polygon_daily_range_one_ticker(
    ticker: str,
    start_d: date,
    end_d: date,
    api_key: str,
    adjusted: bool = True,
    max_retries: int = 3,
    timeout: int = 20,
) -> pd.DataFrame:
    cols_out = ["ticker","date","open","high","low","close","volume","transactions","vwap"]

    url = (
        f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{_iso_ymd(start_d)}/{_iso_ymd(end_d)}"
        f"?adjusted={'true' if adjusted else 'false'}&sort=asc&limit=50000&apiKey={api_key}"
    )

    frames: List[pd.DataFrame] = []
    backoff = 1.0
    page = 1

    while url:
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=timeout)
                if r.status_code == 429:
                    time.sleep(backoff); backoff = min(backoff * 2, 8.0)
                    continue
                r.raise_for_status()
                payload = r.json()
                _save_json_debug(payload, ticker, page, start_d, end_d)

                results = payload.get("results", []) or []

                if results:
                    df = pd.DataFrame(results)

                    # Ensure expected keys exist for robust renaming/conversion
                    for col in ["T","t","o","h","l","c","v","n","vw"]:
                        if col not in df.columns:
                            df[col] = np.nan

                    # Normalize to date (UTC)
                    df["datetime_utc"] = pd.to_datetime(df["t"], unit="ms", utc=True)
                    df["date"] = df["datetime_utc"].dt.date

                    # Rename to human-readable columns
                    df.rename(columns={
                        "T":"ticker","o":"open","h":"high","l":"low",
                        "c":"close","v":"volume","n":"transactions","vw":"vwap"
                    }, inplace=True)

                    # Guarantee ticker even if Polygon omits "T"
                    df["ticker"] = ticker

                    # Numeric coercion
                    for c in ["open","high","low","close","volume","transactions","vwap"]:
                        df[c] = pd.to_numeric(df[c], errors="coerce")

                    frames.append(df[cols_out])

                # Pagination
                next_url = payload.get("next_url")
                if next_url and "apiKey=" not in next_url:
                    sep = "&" if "?" in next_url else "?"
                    next_url = f"{next_url}{sep}apiKey={api_key}"
                url = next_url
                page += 1
                break

            except requests.RequestException as e:
                print(f"[error] {ticker} attempt {attempt+1}/{max_retries}: {e}")
                if attempt == max_retries - 1:
                    url = None
                else:
                    time.sleep(backoff); backoff = min(backoff * 2, 8.0)

    if not frames:
        return pd.DataFrame(columns=cols_out)

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["date"], ignore_index=True)

# ========= Main =========
if __name__ == "__main__":
    if not API_KEY.strip():
        raise SystemExit("Missing POLYGON_API_KEY")

    print(f"[info] Range: {_iso_ymd(start_date)} → {_iso_ymd(end_date)} (UTC) adjusted={ADJUSTED}")
    all_frames: List[pd.DataFrame] = []

    for tk in TICKERS:
        df_t = fetch_polygon_daily_range_one_ticker(
            tk, start_date, end_date, api_key=API_KEY, adjusted=ADJUSTED,
            max_retries=MAX_RETRIES, timeout=TIMEOUT
        )
        # === per-ticker debug statement ===
        if df_t.empty:
            print(f"[result] {tk}: 0 rows")
        else:
            dmin = df_t['date'].min(); dmax = df_t['date'].max()
            print(f"[result] {tk}: rows={len(df_t)} dates={dmin}..{dmax}")
            # optional quick peek:
            print(df_t.tail(min(3, len(df_t))).to_string(index=False))
        all_frames.append(df_t)

    combined = pd.concat(all_frames, ignore_index=True) if any(not f.empty for f in all_frames) else pd.DataFrame(
        columns=["ticker","date","open","high","low","close","volume","transactions","vwap"]
    )

    # Final debug summary
    print(f"[summary] total rows={len(combined)} tickers={len(TICKERS)} "
          f"range={_iso_ymd(start_date)}→{_iso_ymd(end_date)}")

    # The script "returns" the DataFrame for importers; running as a script just prints it.
    # If you want to import: from polygon_recent_multi import combined
