#!/usr/bin/env python3
# sec_eth_events_and_holdings.py  (simplified outputs)
# Extracts ETH purchase/intent events and ETH holdings snapshots from SEC filings.
# Outputs two files with the SAME columns and only keeps filingDate as the date field.
# Drops: accessionNumber, flag, cik, doc_index, event_index, snippet, as_of
# pip install requests pandas lxml

import argparse, json, time, re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
import pandas as pd
from urllib.parse import urljoin
from lxml import etree

# =========================
# CONFIG
# =========================
USER_AGENT = "Ally Zach <ally@panteracapital.com>"
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
    time.sleep(0.25)
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
    return re.sub(r"\s+", " ", text).strip()

# =========================
# Shared parsing helpers
# =========================
def _money_to_float_with_unit(text: str) -> Optional[float]:
    if not text: return None
    m = re.search(r"\$\s*([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?", text, flags=re.I)
    if not m: return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "million":
        val *= 1_000_000
    elif unit == "billion":
        val *= 1_000_000_000
    return val

def _num_with_unit_to_float(text: str) -> Optional[float]:
    if not text: return None
    m = re.match(r"\s*([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?\s*$", str(text), flags=re.I)
    if not m: return None
    val = float(m.group(1).replace(",", ""))
    unit = (m.group(2) or "").lower()
    if unit == "million":
        val *= 1_000_000
    elif unit == "billion":
        val *= 1_000_000_000
    return val

def _extract_wallet_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://\S+", text)
    keep = []
    for u in urls:
        u = u.rstrip(").,]}>\"'")
        if any(dom in u for dom in (
            "etherscan.io/address/", "etherscan.io/tx/",
            "beaconcha.in/validator/", "beaconscan.com/validator/",
            "ethplorer.io/address/", "blockchair.com/ethereum/address/",
        )):
            keep.append(u)
    return list(dict.fromkeys(keep))

# =========================
# (1) ETH PURCHASE / INTENT EVENTS
# =========================
ETH_UNIT  = r"(?:ETH|Ether|Ethereum|tokens?)"
NUMW      = r"[0-9][\d,]*(?:\.\d+)?(?:\s*(?:million|billion))?"
MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?)"
DATE_TOKEN = rf"{MONTHS_RX}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}"

PURCHASE_VERBS = r"(?:purchase|purchased|acquire|acquired|buy|bought|use(?:d)?\s+.*?proceeds\s+to\s+(?:purchase|acquire|buy))"

ETH_COMPLETED_RE = re.compile(
    rf"\b(?:has\s+)?(?:{PURCHASE_VERBS})\s+(?:approximately\s+|about\s+)?({NUMW})\s+(?:{ETH_UNIT})\b",
    re.I
)

ETH_INTENT_RE = re.compile(
    rf"\b(?:intends?|expect(?:s|ed)?|plan(?:s|ned)?)\s+(?:to\s+)?(?:use|allocate)\s+(?:all|100%|a\s+portion|the\s+net\s+proceeds|proceeds)\s+(?:[^.]{{0,120}}?)\s+(?:to\s+)?(?:purchase|acquire|buy)\s+(?:{ETH_UNIT})\b",
    re.I
)

USD_PHRASE_RE = re.compile(
    r"(?:aggregate\s+(?:purchase\s+)?price|aggregate\s+cost|net\s+proceeds|proceeds)\s*(?:of|=|equal\s+to|amount(?:ing)?\s+to)?\s*\$[0-9\.,]+(?:\s*(?:million|billion))?",
    re.I,
)

AVG_ETH_PRICE_RE = re.compile(
    r"(?:average|avg\.?)\s+(?:price|purchase\s+price)\s+(?:per\s+)?(?:ETH|Ether|Ethereum)\s*(?:of|=)?\s*\$[0-9\.,]+",
    re.I,
)

PRICE_PER_ETH_RE = re.compile(
    r"\$\s*([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?\s+per\s+(?:ETH|Ether|Ethereum)\b",
    re.I,
)

def extract_eth_purchase_intent_events(text: str) -> List[Dict[str, Any]]:
    t = re.sub(r"\s+", " ", text)
    events: List[Dict[str, Any]] = []

    # Completed-ish
    for m in ETH_COMPLETED_RE.finditer(t):
        start, end = max(0, m.start()-420), min(len(t), m.end()+700)
        window = t[start:end]
        eth_units = _num_with_unit_to_float(m.group(1))
        usd = None
        avg = None

        mu = USD_PHRASE_RE.search(window)
        if mu:
            usd = _money_to_float_with_unit(mu.group(0))

        ma = AVG_ETH_PRICE_RE.search(window)
        if ma:
            avg = _money_to_float_with_unit(ma.group(0))
        if avg is None:
            mp = PRICE_PER_ETH_RE.search(window)
            if mp:
                avg = _money_to_float_with_unit(mp.group(0))

        if avg is None and (usd is not None) and eth_units:
            avg = usd / eth_units if eth_units else None
        if usd is None and (avg is not None) and eth_units:
            usd = avg * eth_units

        wallets = _extract_wallet_urls(window)

        events.append({
            "kind": "completed",
            "eth": eth_units,
            "usd": usd,
            "avg_usd_per_eth": avg,
            "wallet_urls": wallets,
        })

    # Intent
    for m in ETH_INTENT_RE.finditer(t):
        start, end = max(0, m.start()-340), min(len(t), m.end()+560)
        window = t[start:end]
        usd = None
        mu = USD_PHRASE_RE.search(window)
        if mu:
            usd = _money_to_float_with_unit(mu.group(0))
        wallets = _extract_wallet_urls(window)
        events.append({
            "kind": "intent",
            "eth": None,
            "usd": usd,
            "avg_usd_per_eth": None,
            "wallet_urls": wallets,
        })

    # de-dup rough windows
    uniq, seen = [], set()
    for e in events:
        key = (e["kind"], round(e["eth"] or -1, 8) if isinstance(e["eth"], (int,float)) else None, round(e["usd"] or -1, 2) if isinstance(e["usd"], (int,float)) else None)
        if key in seen: 
            continue
        seen.add(key)
        uniq.append(e)
    return uniq

# =========================
# (2) ETH HOLDINGS SNAPSHOTS
# =========================
ANY_DATE_NEARBY_RE = re.compile(DATE_TOKEN, re.I)

AS_OF_HOLDS_ETH_RE = re.compile(
    rf"As of\s+(?P<date>{DATE_TOKEN})[, ]+(?:[^.]{{0,200}}?)?\b(?:holds?|held)\s+(?P<eth>{NUMW})\s+(?:ETH|Ether|Ethereum|tokens?)\b",
    re.I,
)

ETH_HOLDINGS_OF_RE = re.compile(
    rf"\bETH(?:ereum)?(?:\s+and\s+ETH\s+equivalents)?\s+holdings?\s+(?:of|total(?:ing)?\s+of|total(?:ing)?:?)\s+(?P<eth>{NUMW})\s+(?:ETH|Ether|Ethereum|tokens?)\b",
    re.I,
)

NOW_HOLDS_ETH_RE = re.compile(
    rf"\b(?:now\s+holds|holds|now\s+totals?|totals?)\s+(?P<eth>{NUMW})\s+(?:ETH|Ether|Ethereum|tokens?)\b",
    re.I,
)

TOTAL_NO_UNIT_PER_ETH_RE = re.compile(
    rf"\b(?:holdings?\s+total(?:s)?|now\s+total(?:s)?|total(?:s)?)\s+(?P<eth>{NUMW})\b[^.]{0,120}?\bper\s+(?:ETH|Ether|Ethereum)\b",
    re.I,
)

EXCEEDS_ETH_RE = re.compile(
    rf"\b(?:holdings?|ETH\s+holdings?|Ethereum\s+holdings?)\s+(?:exceed|exceeds|exceeding)\s+(?P<eth>{NUMW})\s+(?:ETH|Ether|Ethereum|tokens?)\b",
    re.I,
)

EXCEEDS_ETH_NO_UNIT_RE = re.compile(
    rf"\bETH(?:ereum)?\s+holdings?\s+(?:exceed|exceeds|exceeding)\s+(?P<eth>{NUMW})\b(?!\s*\$)",
    re.I,
)

COMPRISED_OF_ETH_RE = re.compile(
    rf"\b(?:are|is)\s+comprised\s+of\s+(?P<eth>{NUMW})\s+(?:ETH|Ether|Ethereum|tokens?)\b",
    re.I,
)

TOTAL_ETH_OF_RE = re.compile(
    rf"\b(?:total\s+ETH|ETH\s+total)\s+(?:of|:)\s+(?P<eth>{NUMW})\s+(?:tokens?|ETH|Ether|Ethereum)\b",
    re.I,
)

MARKET_VALUE_RE = re.compile(
    r"market\s+value\s+of\s*\$([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?",
    re.I,
)

AVG_PRICE_ETH_EXPL_RE = re.compile(
    r"(?:average|avg\.?)\s+(?:purchase\s+)?price\s*(?:per\s+ETH|per\s+Ether|per\s+Ethereum)?\s*(?:of|=)?\s*\$([0-9][\d,]*(?:\.\d+)?)",
    re.I,
)

PRICE_PER_ETH_RE2 = re.compile(
    r"\$\s*([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?\s+per\s+(?:ETH|Ether|Ethereum)\b",
    re.I,
)

ETH_HOLDINGS_VALUE_ONLY_RE = re.compile(
    r"\bETH(?:ereum)?\s+holdings?\s+(?:exceed|exceeds|exceeding|of|valued\s+at)\s*(?:approximately|about)?\s*\$([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?\b",
    re.I,
)

def _normalize_ordinal_date(s: str) -> str:
    s = re.sub(r"(\d{1,2})(st|nd|rd|th)", r"\1", s.strip())
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(rf"({MONTHS_RX})\s+(\d{{1,2}})\s+(\d{{4}})", r"\1 \2, \3", s, flags=re.I)
    return s

def _parse_mdy_to_iso(s: str) -> Optional[str]:
    if not s: return None
    s = _normalize_ordinal_date(s)
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    return None

def _scan_enrichers(window: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (usd_cost_total, avg_usd_per_eth, market_value_usd) if found in the window.
    """
    usd_cost_total = None
    avg_usd = None
    market_value = None

    # Market value
    mc = MARKET_VALUE_RE.search(window)
    if mc:
        market_value = _money_to_float_with_unit(mc.group(0))

    # Aggregate cost
    ac = re.search(
        r"(?:acquired\s+for\s+an\s+aggregate(?:\s+purchase\s+price)?\s+of|"
        r"aggregate\s+(?:purchase\s+)?price\s+of|aggregate\s+of)"
        r"\s*\$([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?",
        window,
        flags=re.I,
    )
    if ac:
        usd_cost_total = _money_to_float_with_unit(ac.group(0))

    # Average price per ETH
    ap = AVG_PRICE_ETH_EXPL_RE.search(window)
    if ap:
        try:
            avg_usd = float(ap.group(1).replace(",", ""))
        except Exception:
            avg_usd = None

    if avg_usd is None:
        mp = PRICE_PER_ETH_RE2.search(window)
        if mp:
            avg_usd = _money_to_float_with_unit(mp.group(0))

    return usd_cost_total, avg_usd, market_value


def extract_eth_holdings_snapshots(text: str, filing_date_iso: str) -> List[Dict[str, Any]]:
    t = re.sub(r"\s+", " ", text)
    hits: List[Dict[str, Any]] = []
    used_spans = []

    def add_hit(eth_s: Optional[str], pos: int):
        eth = _num_with_unit_to_float(eth_s) if eth_s is not None else None
        start = max(0, pos - 480)
        end   = min(len(t), pos + 760)
        window = t[start:end]
        usd_cost_total, avg_usd, market_value = _scan_enrichers(window)
        hits.append({
            "eth": eth,
            "usd_cost_total": usd_cost_total,
            "avg_usd_per_eth": avg_usd,
            "market_value_usd": market_value,
        })

    patterns = [
        AS_OF_HOLDS_ETH_RE,
        ETH_HOLDINGS_OF_RE,
        NOW_HOLDS_ETH_RE,
        TOTAL_NO_UNIT_PER_ETH_RE,
        EXCEEDS_ETH_RE,
        EXCEEDS_ETH_NO_UNIT_RE,
        COMPRISED_OF_ETH_RE,
        TOTAL_ETH_OF_RE,
    ]
    for rx in patterns:
        for m in rx.finditer(t):
            span = (m.start(), m.end())
            if any(not (span[1] <= s or span[0] >= e) for s, e in used_spans):
                continue
            add_hit(m.groupdict().get("eth"), m.start())
            used_spans.append(span)

    for m in ETH_HOLDINGS_VALUE_ONLY_RE.finditer(t):
        span = (m.start(), m.end())
        if any(not (span[1] <= s or span[0] >= e) for s, e in used_spans):
            continue
        # Value-only (no ETH count) — captured in _scan_enrichers via window
        add_hit(None, m.start())
        used_spans.append(span)

    return hits

# =========================
# Driver
# =========================
def main():
    ap = argparse.ArgumentParser(description="Extract ETH purchases/intent events and ETH holdings snapshots from SEC filings (8-K, 10-Q, 10-K, amendments).")
    ap.add_argument("--ticker", default="BTBT", help="Ticker symbol (default: BTBT)")
    ap.add_argument("--year", type=int, default=2025, help="Accession year (default: 2025)")
    ap.add_argument("--limit", type=int, default=200, help="Max recent filings to consider (default: 200)")
    ap.add_argument("--max-docs", type=int, default=6, help="Max HTML docs per filing to fetch (default: 6)")
    ap.add_argument(
        "--forms",
        default="8-K,10-Q,10-K,10-Q/A,10-K/A",
        help="Comma-separated list of forms to include (default: 8-K,10-Q,10-K,10-Q/A,10-K/A)"
    )
    ap.add_argument("--outdir", default="data", help="Output directory (default: data)")
    args = ap.parse_args()

    forms_upper = {f.strip().upper() for f in args.forms.split(",") if f.strip()}

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    cik = resolve_cik(args.ticker)
    print(f"CIK for {args.ticker}: {cik}")

    filings = fetch_filings(cik)
    if filings.empty:
        raise SystemExit("No filings returned.")

    subset = filings[
        filings["form"].str.upper().isin(forms_upper) & (filings["accessionYear"] == args.year)
    ].copy()

    subset = subset.sort_values(["filingDate","acceptanceDateTime"], na_position="last").tail(args.limit)
    if subset.empty:
        raise SystemExit(f"No {', '.join(sorted(forms_upper))} filings found for {args.ticker} in {args.year}.")

    print(f"Scanning {len(subset)} filings ({', '.join(sorted(forms_upper))}) for ETH purchases/intent AND holdings snapshots…")

    rows_purchases: List[Dict[str, Any]] = []
    rows_holdings:  List[Dict[str, Any]] = []

    for _, row in subset.iterrows():
        accession = row.get("accessionNumber")
        primary   = (row.get("primaryDocument") or "")
        filing_dt = row.get("filingDate")
        filing_date_iso = filing_dt.date().isoformat() if pd.notna(filing_dt) else ""
        form_val  = str(row.get("form") or "").upper()

        try:
            files = list_filing_files(cik, accession)
            html_urls = candidate_html_urls(files, primary, max_docs=args.max_docs)
        except Exception as e:
            print(f"  Skipping index.json for {accession}: {e}")
            continue

        for doc_idx, u in enumerate(html_urls):
            try:
                text = html_to_text_bytes_first(u)
            except Exception as e:
                print(f"    Error fetching {u}: {e}")
                continue

            # (1) ETH purchases/intents
            try:
                events = extract_eth_purchase_intent_events(text)
            except Exception as e:
                print(f"    Error parsing ETH purchases in {u}: {e}")
                events = []

            for ev in events:
                rows_purchases.append({
                    "filingDate": filing_dt,
                    "ticker": args.ticker,
                    "year": args.year,
                    "form": form_val,
                    "source_url": u,
                    "kind": ev.get("kind"),  # completed | intent
                    "eth": ev.get("eth"),
                    "usd": ev.get("usd"),
                    "avg_usd_per_eth": ev.get("avg_usd_per_eth"),
                    "usd_cost_total": None,
                    "market_value_usd": None,
                    "wallet_urls": " | ".join(ev.get("wallet_urls") or []),
                })

            # (2) ETH Holdings snapshots
            try:
                holds = extract_eth_holdings_snapshots(text, filing_date_iso=filing_date_iso)
            except Exception as e:
                print(f"    Error parsing ETH holdings in {u}: {e}")
                holds = []

            for hv in holds:
                rows_holdings.append({
                    "filingDate": filing_dt,
                    "ticker": args.ticker,
                    "year": args.year,
                    "form": form_val,
                    "source_url": u,
                    "kind": "holdings",
                    "eth": hv.get("eth"),
                    "usd": None,
                    "avg_usd_per_eth": hv.get("avg_usd_per_eth"),
                    "usd_cost_total": hv.get("usd_cost_total"),
                    "market_value_usd": hv.get("market_value_usd"),
                    "wallet_urls": None,
                })

    # =========================
    # Save + print: Purchases/Intents (unified columns; dedup by filingDate date)
    # =========================
    KEEP_COLS = [
        "filingDate","ticker","year","form","source_url","kind",
        "eth","usd","avg_usd_per_eth","usd_cost_total","market_value_usd","wallet_urls"
    ]

    if rows_purchases:
        dfp = pd.DataFrame(rows_purchases)

        # Derive any missing avg if possible
        m = dfp["avg_usd_per_eth"].isna() & dfp["usd"].notna() & dfp["eth"].notna() & (dfp["eth"] != 0)
        dfp.loc[m, "avg_usd_per_eth"] = dfp.loc[m, "usd"] / dfp.loc[m, "eth"]

        # Dedup: one row per calendar date of filingDate; completed > intent, then larger eth/usd/avg
        dfp["_date_key"] = pd.to_datetime(dfp["filingDate"]).dt.date
        dfp["_is_completed"] = (dfp["kind"].str.lower() == "completed").astype(int)
        dfp["_eth_num"] = pd.to_numeric(dfp["eth"], errors="coerce").fillna(-1)
        dfp["_usd_num"] = pd.to_numeric(dfp["usd"], errors="coerce").fillna(-1)
        dfp["_avg_num"] = pd.to_numeric(dfp["avg_usd_per_eth"], errors="coerce").fillna(-1)

        dfp = dfp.sort_values(
            by=["_date_key","_is_completed","_eth_num","_usd_num","_avg_num"],
            ascending=[True, False, False, False, False]
        ).drop_duplicates(subset=["_date_key"], keep="first")

        dfp = dfp[KEEP_COLS].reset_index(drop=True)

        # Save
        csv_path = outdir / f"{args.ticker.lower()}_eth_purchases_{args.year}.csv"
        dfp.to_csv(csv_path, index=False)

        ndjson_path = outdir / f"{args.ticker.lower()}_eth_purchases_{args.year}.ndjson"
        with ndjson_path.open("w", encoding="utf-8") as f:
            for _, r in dfp.iterrows():
                rec = {k: (r[k].isoformat() if hasattr(r[k], "isoformat") else r[k]) for k in dfp.columns}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nSaved ETH purchases/intents CSV -> {csv_path.resolve()}")
        print(f"Saved ETH purchases/intents NDJSON -> {ndjson_path.resolve()}")
        print("\nETH purchases/intents (deduped by filingDate):")
        view_cols = ["filingDate","form","kind","eth","usd","avg_usd_per_eth","source_url"]
        def fmtn(x):
            if isinstance(x, (int,float)):
                return f"{x:,.8f}" if abs(x) < 1000 else f"{x:,.2f}"
            return x
        print(dfp[view_cols].to_string(index=False, float_format=fmtn))
    else:
        print("\nNo ETH purchase/intent events found.")

    # =========================
    # Save + print: Holdings Snapshots (unified columns; dedup by filingDate date)
    # =========================
    if rows_holdings:
        dfh = pd.DataFrame(rows_holdings)

        # Dedup: one row per calendar date of filingDate; higher ETH > higher market value > higher cost
        dfh["_date_key"] = pd.to_datetime(dfh["filingDate"]).dt.date
        dfh["_eth_num"]  = pd.to_numeric(dfh["eth"], errors="coerce").fillna(-1)
        dfh["_mv_num"]   = pd.to_numeric(dfh["market_value_usd"], errors="coerce").fillna(-1)
        dfh["_cost_num"] = pd.to_numeric(dfh["usd_cost_total"], errors="coerce").fillna(-1)

        dfh = dfh.sort_values(
            by=["_date_key","_eth_num","_mv_num","_cost_num"],
            ascending=[True, False, False, False]
        ).drop_duplicates(subset=["_date_key"], keep="first")

        dfh = dfh[KEEP_COLS].reset_index(drop=True)

        # Save
        csv_path_h = outdir / f"{args.ticker.lower()}_eth_holdings_{args.year}.csv"
        dfh.to_csv(csv_path_h, index=False)

        ndjson_path_h = outdir / f"{args.ticker.lower()}_eth_holdings_{args.year}.ndjson"
        with ndjson_path_h.open("w", encoding="utf-8") as f:
            for _, r in dfh.iterrows():
                rec = {k: (r[k].isoformat() if hasattr(r[k], "isoformat") else r[k]) for k in dfh.columns}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nSaved ETH holdings CSV -> {csv_path_h.resolve()}")
        print(f"Saved ETH holdings NDJSON -> {ndjson_path_h.resolve()}")

        print("\nETH Holdings snapshots (deduped by filingDate):")
        show = dfh[["filingDate","form","kind","eth","usd_cost_total","avg_usd_per_eth","market_value_usd","source_url"]].copy()

        def fmtnum(x):
            if pd.isna(x): return ""
            return f"{x:,.8f}" if isinstance(x, (int,float)) and abs(x) < 1000 else (f"{x:,.2f}" if isinstance(x, (int,float)) else x)

        for c in ["eth","usd_cost_total","avg_usd_per_eth","market_value_usd"]:
            if c in show.columns:
                show[c] = show[c].apply(fmtnum)

        print(show.to_string(index=False))
    else:
        print("\nNo ETH holdings snapshots found.")

if __name__ == "__main__":
    main()
