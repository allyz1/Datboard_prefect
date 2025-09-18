#!/usr/bin/env python3
# sequans_holdings_df.py
# Returns a pandas DataFrame (date, ticker, asset, total_holdings)
# built from Swan Bitcoin Research API purchase history.

from typing import Any, Dict, List, Optional
from datetime import datetime
import requests
import pandas as pd


BASE = "https://api.research.swanbitcoin.com/rest/v1"
# Public anon key (you can rotate/override if needed)
_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJ3em5mZWd5Z25mZGhrcGd0ZWpmIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE0MDcwNDAsImV4cCI6MjA2Njk4MzA0MH0."
    "p5G3JtT_R2-JacqILYS0y-NTAQpzrtuhWEH-y7-yt6I"
)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Accept": "application/json",
        "apikey": _ANON_KEY,
        "Authorization": f"Bearer {_ANON_KEY}",
        "User-Agent": "SequansBTC/df-builder",
    })
    return s


def _get_first(keys: List[str], row: Dict[str, Any]) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _to_yyyy_mm_dd(value) -> str:
    """Return 'YYYY-MM-DD' or '' if missing/unparseable."""
    if not value:
        return ""
    s = str(value).strip()
    if s.endswith("Z"):  # iso Z -> offset
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).date().isoformat()
    except Exception:
        return s[:10] if len(s) >= 10 and s[4] == "-" and s[7] == "-" else ""


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _fetch_company_by_ticker(s: requests.Session, ticker: str) -> Dict[str, Any]:
    r = s.get(f"{BASE}/companies", params={"select": "*", "ticker": f"ilike.{ticker}"}, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        raise ValueError(f"No company found for ticker={ticker}")
    exact = [row for row in rows if str(row.get("ticker", "")).upper() == ticker]
    return exact[0] if exact else rows[0]


def _fetch_purchase_history(s: requests.Session, company_id: str) -> List[Dict[str, Any]]:
    r = s.get(f"{BASE}/purchase_history",
              params={"select": "*", "company_id": f"eq.{company_id}", "order": "date.asc"},
              timeout=30)
    r.raise_for_status()
    return r.json()


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm: List[Dict[str, Any]] = []
    for row in rows:
        date_raw = _get_first(["date", "trade_date", "executed_at"], row)
        btc_raw = _get_first(["amount", "amount_btc", "btc", "bitcoin_amount"], row)
        usd_raw = _get_first(["total_spent", "usd_amount", "amount_usd", "usd"], row)
        price_raw = _get_first(["price", "price_per_btc", "avg_price", "usd_per_btc"], row)

        date_iso = _to_yyyy_mm_dd(date_raw)
        btc = _to_float(btc_raw)
        usd_total = _to_float(usd_raw)
        if usd_total and usd_total > 10_000_000:  # likely cents -> dollars
            usd_total = usd_total / 100.0
        price = _to_float(price_raw)

        if not price and btc:
            price = usd_total / btc if btc else 0.0
        if not usd_total and btc and price:
            usd_total = price * btc

        norm.append({"date": date_iso, "btc": btc, "usd": usd_total, "price": price})
    return norm


def get_sequans_holdings_df(ticker: str = "SQNS", asset: str = "BTC") -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      - date (YYYY-MM-DD)
      - ticker (constant, upper)
      - asset  (constant, upper)
      - total_holdings (cumulative BTC by date)

    Note: Multiple purchases on the same day are aggregated into one daily row.
    """
    t = ticker.upper()
    a = asset.upper()
    s = _session()

    company = _fetch_company_by_ticker(s, t)
    company_id = company.get("id") or company.get("company_id")
    if not company_id:
        raise ValueError("Company record missing 'id'")

    purchases = _fetch_purchase_history(s, company_id)
    if not purchases:
        # Return empty DF with the right columns
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    norm = _normalize_rows(purchases)

    # Build DataFrame, aggregate by calendar day, then cumulative sum
    df = pd.DataFrame(norm)
    # Ensure dates are valid & sorted
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    df = df.dropna(subset=["date"])  # drop rows with bad date parsing
    daily = (df.groupby("date", as_index=False)["btc"]
               .sum()
               .sort_values("date"))
    daily["total_holdings"] = daily["btc"].cumsum()

    result = pd.DataFrame({
        "date": daily["date"].astype("string"),
        "ticker": t,
        "asset": a,
        "total_holdings": daily["total_holdings"].astype(float),
    })

    return result


# quick manual run:
if __name__ == "__main__":
    out = get_sequans_holdings_df("SQNS", "BTC")
    print(out.head())
