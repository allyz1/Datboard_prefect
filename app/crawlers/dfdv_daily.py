#!/usr/bin/env python3
# dfdv_holdings_df.py
# Build a pandas DataFrame: date, ticker, asset, total_holdings
# Source: https://defidevcorp.com/api/dashboard/purchases

from typing import Any, Dict, List, Union, Optional
import requests
import pandas as pd
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

URL = "https://defidevcorp.com/api/dashboard/purchases"

# Browser-y UA; omit 'br' to avoid brotli dependency
BASE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
    "Referer": "https://defidevcorp.com/?tab=history-purchases",
}

# Source column names
DATE_COL = "date"          # e.g., "2025-09-03"
AMOUNT_COL = "sol_amount"  # numeric SOL purchased that day

def _session_with_retries(total=5, backoff=0.5, timeout=30) -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=total,
        connect=total, read=total, status=total,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    s.request_timeout = timeout
    return s

def _fetch_json(s: requests.Session, url: str = URL) -> Dict[str, Any]:
    r = s.get(url, headers=BASE_HEADERS, timeout=getattr(s, "request_timeout", 30))
    r.raise_for_status()
    return r.json()

def _unwrap_records(payload: Union[Dict[str, Any], List[Any]]) -> List[Dict[str, Any]]:
    # API shape: {"success": true, "data": [ ... ]}
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [x if isinstance(x, dict) else {"value": x} for x in payload["data"]]
    if isinstance(payload, list):
        return [x if isinstance(x, dict) else {"value": x} for x in payload]
    return [payload]

def _to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    try:
        return pd.json_normalize(records, max_level=2)
    except Exception:
        return pd.DataFrame(records)

def _add_total_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Compute running total over time (by transaction), then reduce to daily last."""
    if df.empty:
        return pd.DataFrame(columns=[DATE_COL, AMOUNT_COL, "total_holdings"])

    if not {DATE_COL, AMOUNT_COL}.issubset(df.columns):
        missing = [c for c in (DATE_COL, AMOUNT_COL) if c not in df.columns]
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    out = df.copy()

    # Parse/sort; keep timezone-naive
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", utc=True).dt.tz_convert(None)
    out[AMOUNT_COL] = pd.to_numeric(out[AMOUNT_COL], errors="coerce")

    out = out.sort_values([DATE_COL, AMOUNT_COL], kind="mergesort")

    # Running total of SOL purchased across rows
    out["total_holdings"] = out[AMOUNT_COL].fillna(0).cumsum()

    # Collapse to one row per calendar day: take the LAST cumulative value that day
    out["day"] = out[DATE_COL].dt.date.astype("string")
    daily = (out.dropna(subset=["day"])
               .groupby("day", as_index=False)["total_holdings"]
               .last()
               .sort_values("day"))

    daily = daily.rename(columns={"day": "date"})
    return daily[["date", "total_holdings"]]

def get_dfdv_holdings_df(ticker: str = "DFDV", asset: str = "SOL") -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - date (YYYY-MM-DD, string)
      - ticker (constant, upper)
      - asset  (constant, upper)
      - total_holdings (float, cumulative SOL by day)
    """
    s = _session_with_retries()
    payload = _fetch_json(s)
    records = _unwrap_records(payload)
    if not records:
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    df = _to_dataframe(records)
    daily = _add_total_holdings(df)

    # Final table-ready shape
    result = pd.DataFrame({
        "date": daily["date"].astype("string"),
        "ticker": str(ticker).upper(),
        "asset": str(asset).upper(),
        "total_holdings": pd.to_numeric(daily["total_holdings"], errors="coerce").astype(float),
    })
    return result

# Example local test:
if __name__ == "__main__":
    print(get_dfdv_holdings_df().head())
