#!/usr/bin/env python3
# SOL_SEC_holdings.py
# Extract ONLY SOL holdings snapshots from SEC filings and return a DataFrame
# with columns: date (filingDate), ticker, total_holdings
#
# Requires: requests, pandas, lxml
#   pip install requests pandas lxml

import argparse
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urljoin

import requests
import pandas as pd
from lxml import etree



# =========================
# CONFIG
# =========================
USER_AGENT = "HoldingsExtractor/1.0 (contact: your-email@example.com)"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
TICKER_MAP_URL   = "https://www.sec.gov/files/company_tickers.json"

# =========================
# HTTP helper (polite)
# =========================
def get(url: str) -> requests.Response:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 429:
        time.sleep(1.5)
        resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.25)  # be nice to SEC
    return resp

# =========================
# Ticker -> CIK mapping
# =========================
def resolve_cik(ticker: str) -> str:
    data = get(TICKER_MAP_URL).json()
    for item in data.values():
        if str(item.get("ticker","")).upper() == ticker.upper():
            return f"{int(item['cik_str']):010d}"
    raise RuntimeError(f"Ticker not found: {ticker}")

# =========================
# Filings list
# =========================
def fetch_filings(cik: str) -> pd.DataFrame:
    url = f"{BASE_SUBMISSIONS}/CIK{cik}.json"
    j = get(url).json()
    recent = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df

    # Normalize datetimes to naive
    for col in ("reportDate","filingDate","acceptanceDateTime"):
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.tz_localize(None)

    df["form"] = df["form"].astype(str)

    # Pull accession year from "0001213900-25-034374" -> 2025
    if "accessionNumber" in df.columns:
        def acc_year(s: str) -> Optional[int]:
            m = re.match(r"^\d{10}-(\d{2})-\d{6}$", str(s))
            if not m: return None
            return 2000 + int(m.group(1))
        df["accessionYear"] = df["accessionNumber"].apply(acc_year)

    return df

def cik_no_zeros(cik_10: str) -> str:
    return str(int(cik_10))

def accession_nodash(acc: str) -> str:
    return acc.replace("-", "")

def filing_base_url(cik_10: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros(cik_10)}/{accession_nodash(accession)}/"

def list_filing_files(cik_10: str, accession: str) -> pd.DataFrame:
    base = filing_base_url(cik_10, accession)
    j = get(urljoin(base, "index.json")).json()
    df = pd.DataFrame(j.get("directory", {}).get("item", []))
    if df.empty:
        return df
    df["url"] = df["name"].apply(lambda n: urljoin(base, n))
    return df

def is_html_name(name: str) -> bool:
    n = str(name).lower()
    return n.endswith((".htm",".html",".xhtml"))

def candidate_html_urls(files_df: pd.DataFrame, primary_doc: str, max_docs: int = 6) -> List[str]:
    if files_df.empty:
        return []
    df = files_df[files_df["name"].apply(is_html_name)].copy()
    if df.empty:
        return []
    prim = (primary_doc or "").lower()

    # prefer primary, then common filing/exhibit names (8-K/10-K/10-Q and ex99/press)
    PREFERRED_TOKENS = ("8-k", "8k", "10-k", "10k", "10-q", "10q", "ex99", "ex-99", "press")

    def rank(name: str) -> Tuple[int,int,int]:
        n = name.lower()
        r0 = 0 if n == prim else 1
        r1 = 0 if any(tok in n for tok in PREFERRED_TOKENS) else 1
        r2 = 0 if n.endswith((".xhtml",".htm",".html")) else 1
        return (r0, r1, r2)

    df["rank_tuple"] = df["name"].apply(rank)
    df = df.sort_values(["rank_tuple","name"])
    return df["url"].head(max_docs).tolist()

# =========================
# Text extraction
# =========================
def html_to_text_bytes_first(url: str) -> str:
    content = get(url).content  # BYTES
    parser = etree.HTMLParser(recover=True, huge_tree=True)
    try:
        root = etree.fromstring(content, parser=parser)
    except Exception:
        root = etree.HTML(content, parser=parser)
    text = " ".join(root.itertext())
    # normalize whitespace and curly quotes/NBSPs
    text = (text or "").replace("\u00a0", " ").replace("\u2019", "'").replace("\u2018", "'")
    return re.sub(r"\s+", " ", text).strip()

# =========================
# Numeric helpers
# =========================
def _num_with_unit_to_float(text: str) -> Optional[float]:
    if not text: return None
    m = re.match(r"\s*([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?\s*$", str(text), flags=re.I)
    if not m: return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "million": val *= 1_000_000
    elif unit == "billion": val *= 1_000_000_000
    return val

# =========================
# SOL holdings patterns (only!)
# =========================
SOL_UNIT      = r"(?:SOL|Solana|tokens?)"
NUMW          = r"[0-9][\d,]*(?:\.\d+)?(?:\s*(?:million|billion))?"
MONTHS_RX     = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_TOKEN    = rf"{MONTHS_RX}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}"
QUAL_APPROX   = r"(?:about|approximately|around|over|more\s+than|at\s+least|in\s+excess\s+of)"

# “As of DATE, holds/held [over] NUM SOL”
AS_OF_HOLDS_SOL_RE = re.compile(
    rf"\bAs of\s+(?P<date>{DATE_TOKEN})[, ]+(?:[^.]{{0,200}}?)?\b(?:holds?|held)\s+(?:{QUAL_APPROX}\s*)?(?P<sol>{NUMW})\s+(?:{SOL_UNIT})\b",
    re.I,
)

# “SOL holdings of NUM SOL” / “holdings of SOL totaling NUM SOL”
SOL_HOLDINGS_OF_RE = re.compile(
    rf"\bSOL(?:ana)?(?:\s+and\s+SOL\s+equivalents)?\s+holdings?\s+(?:of|total(?:ing)?\s+of|total(?:ing)?:?)\s+(?P<sol>{NUMW})\s+(?:{SOL_UNIT})\b",
    re.I,
)

# “now holds/holds [over] NUM SOL”
NOW_HOLDS_SOL_RE = re.compile(
    rf"\b(?:now\s+holds|holds|now\s+totals?|totals?)\s+(?:{QUAL_APPROX}\s*)?(?P<sol>{NUMW})\s+(?:{SOL_UNIT})\b",
    re.I,
)

# “holdings exceed NUM SOL”
EXCEEDS_SOL_RE = re.compile(
    rf"\b(?:holdings?|SOL\s+holdings?|Solana\s+holdings?)\s+(?:exceed|exceeds|exceeding)\s+(?P<sol>{NUMW})\s+(?:{SOL_UNIT})\b",
    re.I,
)

# “total SOL of NUM tokens”
TOTAL_SOL_OF_RE = re.compile(
    rf"\b(?:total\s+SOL|SOL\s+total)\s+(?:of|:)\s+(?P<sol>{NUMW})\s+(?:{SOL_UNIT})\b",
    re.I,
)
PURCHASES_TOTALLED_AS_HOLDINGS_RE = re.compile(
    rf"\b(?:initial\s+)?(?:liquid\s+)?(?:solana\s+token\s+)?purchases?\s+total(?:ed|s)?\s+"
    rf"(?:{QUAL_APPROX}\s*)?(?P<sol>{NUMW})(?!\s*\$)[^\.]{{0,200}}?\bper\s+(?:SOL|Solana)\b",
    re.I,
)

def extract_sol_holdings_snapshots(text: str) -> List[float]:
    """
    Return a list of numeric SOL holdings values found in the text.
    (Ignores purchases, costs, and averages.)
    """
    t = re.sub(r"\s+", " ", text or "")
    values: List[float] = []

    patterns = [
        AS_OF_HOLDS_SOL_RE,
        SOL_HOLDINGS_OF_RE,
        NOW_HOLDS_SOL_RE,
        EXCEEDS_SOL_RE,
        TOTAL_SOL_OF_RE,
        PURCHASES_TOTALLED_AS_HOLDINGS_RE,  # <— add this line
    ]

    used_spans: List[Tuple[int,int]] = []

    def overlaps(a: Tuple[int,int], b: Tuple[int,int]) -> bool:
        return not (a[1] <= b[0] or a[0] >= b[1])

    for rx in patterns:
        for m in rx.finditer(t):
            span = (m.start(), m.end())
            if any(overlaps(span, u) for u in used_spans):
                continue
            sol_s = m.groupdict().get("sol")
            v = _num_with_unit_to_float(sol_s) if sol_s else None
            if v is not None:
                values.append(v)
                used_spans.append(span)

    # de-dup by rounded value
    out, seen = [], set()
    for v in values:
        key = round(float(v), 8)
        if key in seen: 
            continue
        seen.add(key)
        out.append(float(v))
    return out

# =========================
# Runner that returns a DataFrame
# =========================
def run_holdings_extract(
    ticker: str,
    year: int = 2025,
    forms: str = "8-K,10-Q,10-K,10-Q/A,10-K/A,6-K,424B5,424B3",
    limit: int = 200,
    max_docs: int = 6,
    since_hours: int = 24,  # NEW: only check filings accepted in the last N hours (set 0 to disable)
) -> pd.DataFrame:
    """
    Crawl selected filings for a ticker and return a DataFrame:
      columns = ['date', 'ticker', 'total_holdings']

    If since_hours > 0:
      - Filters by acceptanceDateTime >= now - since_hours (ignores 'year')
    Else:
      - Falls back to accessionYear == year

    'date' is the filingDate (YYYY-MM-DD). For days with multiple matches,
    keeps the max total_holdings.
    """
    forms_upper = {f.strip().upper() for f in forms.split(",") if f.strip()}

    cik = resolve_cik(ticker)
    filings = fetch_filings(cik)
    if filings.empty:
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    # Build the base mask for forms
    mask_forms = filings["form"].str.upper().isin(forms_upper)

    if since_hours and since_hours > 0:
        now_utc = pd.Timestamp.utcnow().tz_localize(None)
        cutoff  = now_utc - pd.Timedelta(hours=since_hours)
        mask_recent = filings["acceptanceDateTime"] >= cutoff
        subset = filings[mask_forms & mask_recent].copy()
    else:
        subset = filings[mask_forms & (filings["accessionYear"] == year)].copy()

    if subset.empty:
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    # Sort newest first; cap by 'limit'
    sort_cols = ["acceptanceDateTime", "filingDate"]
    subset = subset.sort_values(sort_cols, ascending=[False, False]).head(limit)

    rows: List[Dict[str, Any]] = []

    for _, row in subset.iterrows():
        accession = row.get("accessionNumber")
        primary   = (row.get("primaryDocument") or "")
        filing_dt = row.get("filingDate")
        date_iso  = filing_dt.date().isoformat() if pd.notna(filing_dt) else ""

        try:
            files = list_filing_files(cik, accession)
            html_urls = candidate_html_urls(files, primary, max_docs=max_docs)
        except Exception:
            continue

        for u in html_urls:
            try:
                text = html_to_text_bytes_first(u)
            except Exception:
                continue

            holdings_list = extract_sol_holdings_snapshots(text)
            for h in holdings_list:
                rows.append({"date": date_iso, "ticker": ticker, "asset": "SOL", "total_holdings": float(h)})

    df = pd.DataFrame(rows, columns=["date","ticker","asset","total_holdings"])
    if df.empty:
        return df

    # Deduplicate by date/ticker: keep the max holdings reported that day
    df = (df.sort_values(["date","total_holdings"])
            .groupby(["date","ticker"], as_index=False)
            .last())

    return df
def _parse_ticker_list_arg(tickers: Union[str, List[str]]) -> List[str]:
    if isinstance(tickers, str):
        parts = re.split(r"[,\s]+", tickers.strip())
        return [p.upper() for p in parts if p]
    return [str(t).upper() for t in tickers if str(t).strip()]

def get_sec_sol_holdings_df(
    tickers: Union[str, List[str]] = "HSDT,FORD",
    hours_back: int = 24,
    forms: tuple[str, ...] = ("8-K","10-Q","10-K","10-Q/A","10-K/A","6-K","424B5","424B3"),
    verbose: bool = False,
    limit: int = 200,
    max_docs: int = 6,
) -> pd.DataFrame:
    """
    Mirror of your BTC/ETH helpers:
      - tickers: comma/space-separated string or list
      - hours_back: last N hours by acceptanceDateTime
      - forms: tuple of forms
    Returns DataFrame with columns: date, ticker, asset, total_holdings
    """
    symbols = _parse_ticker_list_arg(tickers)
    frames: List[pd.DataFrame] = []
    forms_csv = ",".join(forms)

    for tk in symbols:
        try:
            df_t = run_holdings_extract(
                ticker=tk,
                year=0,                # ignored when since_hours > 0
                forms=forms_csv,
                limit=limit,
                max_docs=max_docs,
                since_hours=hours_back # <-- map to acceptance window
            )
        except Exception as e:
            if verbose:
                print(f"[SOL SEC] {tk}: {e}")
            continue

        if isinstance(df_t, pd.DataFrame) and not df_t.empty:
            frames.append(df_t)

    if not frames:
        return pd.DataFrame(columns=["date","ticker","asset","total_holdings"])

    out = pd.concat(frames, ignore_index=True)

    # Normalize + (light) dedup safety: keep max holdings per date/ticker
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    out["ticker"] = out["ticker"].str.upper()
    out["total_holdings"] = pd.to_numeric(out["total_holdings"], errors="coerce")
    out = (out.sort_values(["date","ticker","total_holdings"])
             .groupby(["date","ticker"], as_index=False)
             .last())
    # ensure column order
    out = out[["date","ticker","asset","total_holdings"]]
    return out


# =========================
# CLI (prints the DataFrame; does NOT save files)
# =========================
def _parse_ticker_list(s: str) -> List[str]:
    import re as _re
    parts = _re.split(r"[,\s]+", s.strip())
    return [p.upper() for p in parts if p]

def main():
    ap = argparse.ArgumentParser(
        description="Extract SOL holdings snapshots; outputs a DataFrame (date, ticker, total_holdings)."
    )
    ap.add_argument("--tickers", default="HSDT,FORD",
                    help="Comma- or space-separated tickers, e.g. 'HSDT,FORD' or 'HSDT FORD'.")
    ap.add_argument("--year", type=int, default=2025,
                    help="Used only if --since-hours is 0.")
    ap.add_argument("--forms", default="8-K,10-Q,10-K,10-Q/A,10-K/A,6-K,424B5,424B3")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-docs", type=int, default=6)
    ap.add_argument("--since-hours", type=int, default=24,
                    help="Only check filings accepted in the last N hours (default 24). Set 0 to disable.")
    args = ap.parse_args()

    tickers = _parse_ticker_list(args.tickers)
    if not tickers:
        raise SystemExit("No tickers provided. Use --tickers HSDT,FORD (or similar).")

    dfs = []
    for t in tickers:
        df_t = run_holdings_extract(
            ticker=t,
            year=args.year,
            forms=args.forms,
            limit=args.limit,
            max_docs=args.max_docs,
            since_hours=args.since_hours,  # NEW
        )
        if not df_t.empty:
            dfs.append(df_t)

    out = (pd.concat(dfs, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
           if dfs else pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"]))

    with pd.option_context("display.max_rows", 200, "display.width", 120):
        print(out)

if __name__ == "__main__":
    main()