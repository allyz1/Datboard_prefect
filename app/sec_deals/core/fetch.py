# sec_deals/core/fetch.py
#!/usr/bin/env python3
"""
Lightweight SEC fetch helpers:
- get(): polite HTTP with retries
- resolve_cik(): ticker -> 10-digit CIK via company_tickers.json
- fetch_filings(): issuer's recent filings JSON -> normalized DataFrame
- list_filing_files(): filing index.json -> files DataFrame (with URLs)

All timestamps returned as naive (no tz) pandas Timestamps for simplicity.
"""

from __future__ import annotations
import time, re, sys
from typing import Dict, Any, Optional
from urllib.parse import urljoin

import requests
import pandas as pd

# -------------------------
# SEC-polite config
# -------------------------
USER_AGENT = "Ally Zach <ally@panteracapital.com>"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
TICKER_MAP_URL   = "https://www.sec.gov/files/company_tickers.json"

# -------------------------
# HTTP helper
# -------------------------
def get(url: str, *, max_retries: int = 3, backoff: float = 1.25) -> requests.Response:
    """
    Polite GET with basic retry semantics, including 429 handling.
    Sleeps a short period between requests to avoid hammering the SEC.
    """
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                # rate-limited: sleep and retry
                sleep_s = 1.5 * (attempt + 1)
                time.sleep(sleep_s)
                continue
            resp.raise_for_status()
            # politeness delay after successful call
            time.sleep(0.25)
            return resp
        except Exception as e:
            last_err = e
            # brief backoff and try again
            time.sleep(backoff * (attempt + 1))
    # If we exhausted retries, raise the last error
    if last_err:
        raise last_err
    raise RuntimeError(f"GET failed unexpectedly for {url}")

# -------------------------
# Ticker -> CIK
# -------------------------
def resolve_cik(ticker: str) -> str:
    """
    Resolve e.g. 'BMNR' -> '0000########' (10-digit, zero-padded CIK).
    """
    j = get(TICKER_MAP_URL).json()
    t_up = ticker.strip().upper()
    for item in j.values():
        if str(item.get("ticker", "")).upper() == t_up:
            return f"{int(item['cik_str']):010d}"
    raise RuntimeError(f"Ticker not found in SEC map: {ticker}")

def _cik_no_zeros(cik_10: str) -> str:
    return str(int(cik_10))

def _acc_nodash(acc: str) -> str:
    return str(acc).replace("-", "")

# -------------------------
# Filings list
# -------------------------
def fetch_filings(cik_10: str) -> pd.DataFrame:
    """
    Load the issuer 'recent' filings JSON and normalize:
    - datetimes to pandas (naive)
    - 'form' to string
    - accession year derived from accessionNumber (e.g. '2025' from '...-25-...')
    Returns possibly-empty DataFrame with at least:
      ['accessionNumber','filingDate','reportDate','acceptanceDateTime',
       'form','primaryDocument','acc_year']
    """
    url = f"{BASE_SUBMISSIONS}/CIK{cik_10}.json"
    j = get(url).json()
    recent = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df

    # Normalize datetime-like fields to naive timestamps
    for col in ("reportDate", "filingDate", "acceptanceDateTime"):
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.tz_localize(None)

    # Strings
    if "form" in df.columns:
        df["form"] = df["form"].astype(str)
    if "primaryDocument" in df.columns:
        df["primaryDocument"] = df["primaryDocument"].astype(str)

    # Accession-year from "0000950170-25-047219" -> 2025
    if "accessionNumber" in df.columns:
        def year_from_acc(s: str) -> Optional[int]:
            m = re.search(r"^\d{10}-(\d{2})-\d{6}$", str(s))
            return 2000 + int(m.group(1)) if m else None
        df["acc_year"] = df["accessionNumber"].apply(year_from_acc)
    else:
        df["acc_year"] = None

    return df

# -------------------------
# Filing files (index.json)
# -------------------------
def filing_base_url(cik_10: str, accession: str) -> str:
    """
    Base directory for a filing's artifacts on EDGAR.
    """
    return f"https://www.sec.gov/Archives/edgar/data/{_cik_no_zeros(cik_10)}/{_acc_nodash(accession)}/"

def list_filing_files(cik_10: str, accession: str) -> pd.DataFrame:
    """
    Query the filing's index.json and return a DataFrame of files with absolute URLs.
    Columns: ['name','type','size','last-modified','url','base_url']
    """
    base = filing_base_url(cik_10, accession)
    j = get(urljoin(base, "index.json")).json()
    items = j.get("directory", {}).get("item", []) or []
    df = pd.DataFrame(items)
    if df.empty:
        return df
    df["url"] = df["name"].apply(lambda n: urljoin(base, str(n)))
    df["base_url"] = base
    # normalize column names
    if "last-modified" not in df.columns and "lastModified" in df.columns:
        df.rename(columns={"lastModified": "last-modified"}, inplace=True)
    return df

# -------------------------
# Simple CLI smoke test
# -------------------------
if __name__ == "__main__":
    """
    Quick manual test with BMNR:
      python -m sec_deals.core.fetch
    """
    test_tkr = "BMNR"
    try:
        cik = resolve_cik(test_tkr)
        print(f"[resolve_cik] {test_tkr} -> {cik}")
    except Exception as e:
        print(f"CIK lookup failed: {e}", file=sys.stderr)
        sys.exit(1)

    filings = fetch_filings(cik)
    if filings.empty:
        print("No recent filings returned.")
        sys.exit(0)

    print(f"[fetch_filings] rows={len(filings)} cols={list(filings.columns)}")
    # Show a few recent rows
    show_cols = ["filingDate","form","accessionNumber","primaryDocument","acc_year"]
    print(filings[show_cols].tail(8).to_string(index=False))

    # Try first filing with a primaryDocument
    row = filings.dropna(subset=["accessionNumber"]).tail(1).iloc[0]
    acc = row["accessionNumber"]
    files_df = list_filing_files(cik, acc)
    print(f"\n[list_filing_files] accession={acc} rows={len(files_df)}")
    if not files_df.empty:
        print(files_df.head(10)[["name","type","size","url"]].to_string(index=False))
