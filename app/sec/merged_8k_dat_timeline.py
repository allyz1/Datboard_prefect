#!/usr/bin/env python3
# sec_btc_events_and_holdings_merged.py
# Combines:
#  - STRICT completed BTC purchase extraction (with wallet_urls)
#  - BTC holdings "as-of" snapshots (date, BTC, cost/avg/market value)
# Now supports scanning 8-K, 10-Q, 10-K and amendments (10-Q/A, 10-K/A).
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
    PREFERRED_TOKENS = ("8-k", "8k", "10-k", "10k", "10-q", "10q", "ex99", "press")

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
    content = get(url).content  # BYTES (prevents encoding decl error)
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
    """
    Parse amounts like "$458,700,000" or "$147.5 million" / "$1.2 billion".
    Returns a float in USD, or None.
    """
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

def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None

def _extract_wallet_urls(text: str) -> List[str]:
    urls = re.findall(r"https?://\S+", text)
    keep = []
    for u in urls:
        u = u.rstrip(").,]}>\"'")
        if any(dom in u for dom in (
            "blockchain.com/explorer/addresses/btc/",
            "mempool.space/address/",
            "blockstream.info/address/",
        )):
            keep.append(u)
    # dedupe preserving order
    return list(dict.fromkeys(keep))

# =========================
# (1) STRICT COMPLETED PURCHASE EVENTS (from sec_btc_events_from_8k.py)
# =========================
PURCHASE_RE = re.compile(r"tether\s+(?:has\s+)?purchased\s+([0-9][\d,]*(?:\.\d+)?)\s+bitcoin", re.I)

SOLD_TO_AT_CLOSING_RE = re.compile(
    r"(?:will|shall)\s+(?:be\s+)?(?:sold|transferred|contributed|delivered)\s+by\s+[^.]{1,100}?\s+to\s+[^.]{1,80}?\s+(?:at|upon)\s+(?:the\s+|such\s+)?closing",
    re.I,
)

PURCHASE_FROM_AT_CLOSING_RE = re.compile(
    r"(?:at|upon)\s+(?:the\s+|such\s+)?closing[^.]{0,200}?(?:will|shall)\s+(?:purchase|buy|acquire)\s+(?:from\s+[^.]{1,80})?\s+(?:the\s+)?(?:bitcoin|btc)\b",
    re.I,
)

RECEIVE_AT_CLOSING_RE = re.compile(
    r"(?:at|upon)\s+(?:the\s+|such\s+)?closing[^.]{0,160}?(?:will|shall)\s+(?:receive|take\s+delivery\s+of)\s+(?:approximately\s+)?[\d,]+(?:\.\d+)?\s*(?:bitcoin|btc)\b",
    re.I,
)

TO_RECIPIENT_RE = re.compile(
    "|".join([
        SOLD_TO_AT_CLOSING_RE.pattern,
        PURCHASE_FROM_AT_CLOSING_RE.pattern,
        RECEIVE_AT_CLOSING_RE.pattern,
    ]),
    re.I,
)

USD_PHRASE_RE = re.compile(
    r"(aggregate\s+purchase\s+price|aggregate\s+price(?:\s+equal\s+to|of))\s*(?:approximately|about)?\s*\$[0-9\.,]+(?:\s*(?:million|billion))?",
    re.I,
)
AVG_PHRASE_RE = re.compile(
    r"average\s+price\s+per\s+Bitcoin\s+of\s*\$[0-9\.,]+",
    re.I,
)

def extract_completed_purchase_events(text: str) -> List[Dict[str, Any]]:
    """STRICT: requires 'Tether has purchased ... Bitcoin' + closing context; captures wallet URLs."""
    t = re.sub(r"\s+", " ", text)

    events: List[Dict[str, Any]] = []
    for m in PURCHASE_RE.finditer(t):
        start = max(0, m.start() - 300)
        end   = min(len(t), m.end() + 700)
        window = t[start:end]

        # Must indicate tranche goes to the recipient at Closing
        if not TO_RECIPIENT_RE.search(window):
            continue

        btc = _safe_float(m.group(1))
        if btc is None:
            continue

        usd = None
        mu = USD_PHRASE_RE.search(window)
        if mu:
            usd = _money_to_float_with_unit(mu.group(0))

        avg = None
        ma = AVG_PHRASE_RE.search(window)
        if ma:
            avg = _money_to_float_with_unit(ma.group(0))

        if avg is None and usd is not None and btc:
            avg = usd / btc
        if usd is None and avg is not None and btc:
            usd = avg * btc

        wallets = _extract_wallet_urls(window)

        events.append({
            "btc": btc,
            "usd": usd,
            "avg_usd_per_btc": avg,
            "wallet_urls": wallets,
            "snippet": window.strip(),
        })

    return events

# =========================
# (2) BTC HOLDINGS SNAPSHOTS (from sec_btc_events_and_holdings_from_8k.py)
# =========================
MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_TOKEN = rf"{MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}}"

AS_OF_HELD_RE = re.compile(
    rf"As of\s+(?P<date>{DATE_TOKEN})[, ]+(?:[^.]{{0,120}}?)?\b(?:holds?|held)\s+(?P<btc>[0-9][\d,]*(?:\.\d+)?)\s+(?:Bitcoin|BTC)\b",
    re.I,
)

BITCOIN_HOLDINGS_OF_RE = re.compile(
    rf"\bBitcoin holdings of\s+(?P<btc>[0-9][\d,]*(?:\.\d+)?)\s+(?:Bitcoin|BTC)\b",
    re.I,
)

NOW_HOLDS_RE = re.compile(
    rf"\b(?:now\s+holds|holds)\s+(?P<btc>[0-9][\d,]*(?:\.\d+)?)\s+(?:Bitcoin|BTC)\b",
    re.I,
)

MARKET_VALUE_RE = re.compile(
    r"market\s+value\s+of\s*\$([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?",
    re.I,
)
AVG_PURCHASE_RE = re.compile(
    r"average\s+purchase\s+price\s+of\s*\$([0-9][\d,]*(?:\.\d+)?)(?:\s*per\s*Bitcoin)?",
    re.I,
)
AGG_COST_RE = re.compile(
    r"(?:acquired\s+for\s+an\s+aggregate(?:\s+purchase\s+price)?\s+of|aggregate\s+purchase\s+price\s+of|aggregate\s+of)\s*\$([0-9][\d,]*(?:\.\d+)?)(?:\s*(million|billion))?",
    re.I,
)
ANY_DATE_NEARBY_RE = re.compile(DATE_TOKEN, re.I)

def _parse_mdy_to_iso(s: str) -> Optional[str]:
    if not s: return None
    s = s.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    return None

def _scan_enrichers(window: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Returns (usd_cost_total, avg_usd_per_btc, market_value_usd) if found in the window.
    """
    usd_cost_total = None
    avg_usd = None
    market_value = None

    mc = MARKET_VALUE_RE.search(window)
    if mc:
        market_value = _money_to_float_with_unit(mc.group(0))

    ac = AGG_COST_RE.search(window)
    if ac:
        market_value_candidate = _money_to_float_with_unit(ac.group(0))
        if market_value_candidate is not None:
            usd_cost_total = market_value_candidate

    ap = AVG_PURCHASE_RE.search(window)
    if ap:
        try:
            avg_usd = float(ap.group(1).replace(",", ""))
        except Exception:
            avg_usd = None

    return usd_cost_total, avg_usd, market_value

def _find_nearby_date(t: str, center: int, fallback_iso: str) -> str:
    start = max(0, center - 160)
    end   = min(len(t), center + 160)
    win = t[start:end]
    m = ANY_DATE_NEARBY_RE.search(win)
    if m:
        iso = _parse_mdy_to_iso(m.group(0))
        if iso: return iso
    return fallback_iso

def extract_holdings_snapshots(text: str, filing_date_iso: str) -> List[Dict[str, Any]]:
    t = re.sub(r"\s+", " ", text)
    hits: List[Dict[str, Any]] = []
    used_spans = []

    def add_hit(btc_s: str, pos: int, date_iso: Optional[str]):
        btc = _safe_float(btc_s)
        if btc is None: return
        start = max(0, pos - 280)
        end   = min(len(t), pos + 420)
        window = t[start:end]
        usd_cost_total, avg_usd, market_value = _scan_enrichers(window)
        hits.append({
            "as_of": date_iso or _find_nearby_date(t, pos, filing_date_iso),
            "btc": btc,
            "usd_cost_total": usd_cost_total,
            "avg_usd_per_btc": avg_usd,
            "market_value_usd": market_value,
            "snippet": window.strip(),
        })

    for m in AS_OF_HELD_RE.finditer(t):
        date_iso = _parse_mdy_to_iso(m.group("date"))
        add_hit(m.group("btc"), m.start(), date_iso)
        used_spans.append((m.start(), m.end()))

    for m in BITCOIN_HOLDINGS_OF_RE.finditer(t):
        span = (m.start(), m.end())
        if any(not (span[1] <= s or span[0] >= e) for s, e in used_spans):
            continue
        add_hit(m.group("btc"), m.start(), None)
        used_spans.append(span)

    for m in NOW_HOLDS_RE.finditer(t):
        span = (m.start(), m.end())
        if any(not (span[1] <= s or span[0] >= e) for s, e in used_spans):
            continue
        add_hit(m.group("btc"), m.start(), None)
        used_spans.append(span)

    return hits

# =========================
# Driver
# =========================
def main():
    ap = argparse.ArgumentParser(description="Extract STRICT BTC completed purchases (with wallet URLs) and BTC holdings snapshots from SEC filings (8-K, 10-Q, 10-K, amendments).")
    ap.add_argument("--ticker", default="CEP", help="Ticker symbol (default: CEP)")
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

    # Forms filter + year filter
    subset = filings[
        filings["form"].str.upper().isin(forms_upper) & (filings["accessionYear"] == args.year)
    ].copy()

    subset = subset.sort_values(["filingDate","acceptanceDateTime"], na_position="last").tail(args.limit)
    if subset.empty:
        raise SystemExit(f"No {', '.join(sorted(forms_upper))} filings found for {args.ticker} in {args.year}.")

    print(f"Scanning {len(subset)} filings ({', '.join(sorted(forms_upper))}) for STRICT BTC purchases AND holdings snapshots…")

    rows_purchases: List[Dict[str, Any]] = []
    rows_holdings:  List[Dict[str, Any]] = []

    for _, row in subset.iterrows():
        accession = row.get("accessionNumber")
        primary   = (row.get("primaryDocument") or "")
        filing_dt = row.get("filingDate")
        filing_date_iso = filing_dt.date().isoformat() if pd.notna(filing_dt) else ""
        form_val  = str(row.get("form") or "").upper()

        # enumerate candidate HTMLs
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

            # (1) STRICT Completed purchases (with wallet URLs)
            try:
                events = extract_completed_purchase_events(text)
            except Exception as e:
                print(f"    Error parsing purchases in {u}: {e}")
                events = []

            for k, ev in enumerate(events):
                rows_purchases.append({
                    "ticker": args.ticker,
                    "cik": cik,
                    "year": args.year,
                    "form": form_val,
                    "accessionNumber": accession,
                    "filingDate": filing_dt,
                    "source_url": u,
                    "btc": ev.get("btc"),
                    "usd": ev.get("usd"),
                    "avg_usd_per_btc": ev.get("avg_usd_per_btc"),
                    "wallet_urls": " | ".join(ev.get("wallet_urls") or []),
                    "snippet": ev.get("snippet"),
                    "doc_index": doc_idx,
                    "event_index": k,
                })

            # (2) Holdings snapshots
            try:
                holds = extract_holdings_snapshots(text, filing_date_iso=filing_date_iso)
            except Exception as e:
                print(f"    Error parsing holdings in {u}: {e}")
                holds = []

            for j, hv in enumerate(holds):
                rows_holdings.append({
                    "ticker": args.ticker,
                    "cik": cik,
                    "year": args.year,
                    "form": form_val,
                    "accessionNumber": accession,
                    "filingDate": filing_dt,
                    "as_of": hv.get("as_of"),
                    "btc": hv.get("btc"),
                    "usd_cost_total": hv.get("usd_cost_total"),
                    "avg_usd_per_btc": hv.get("avg_usd_per_btc"),
                    "market_value_usd": hv.get("market_value_usd"),
                    "source_url": u,
                    "snippet": hv.get("snippet"),
                    "doc_index": doc_idx,
                    "event_index": j,
                })

    # =========================
    # Save + print: STRICT Completed Purchases
    # =========================
    if rows_purchases:
        dfp = pd.DataFrame(rows_purchases)
        # Derive any missing avg if possible
        m = dfp["avg_usd_per_btc"].isna() & dfp["usd"].notna() & dfp["btc"].notna() & (dfp["btc"] != 0)
        dfp.loc[m, "avg_usd_per_btc"] = dfp.loc[m, "usd"] / dfp.loc[m, "btc"]

        cols = ["filingDate","ticker","year","form","accessionNumber","source_url","btc","usd","avg_usd_per_btc","wallet_urls","snippet","doc_index","event_index","cik"]
        dfp = dfp[cols].sort_values(["filingDate","accessionNumber","doc_index","event_index"]).reset_index(drop=True)

        # Generic filenames (not "8k_...")
        csv_path = outdir / f"{args.ticker.lower()}_btc_completed_purchases_{args.year}.csv"
        dfp.to_csv(csv_path, index=False)
        ndjson_path = outdir / f"{args.ticker.lower()}_btc_completed_purchases_{args.year}.ndjson"
        with ndjson_path.open("w", encoding="utf-8") as f:
            for _, r in dfp.iterrows():
                rec = {k: (r[k].isoformat() if hasattr(r[k], "isoformat") else r[k]) for k in dfp.columns}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nSaved purchases CSV -> {csv_path.resolve()}")
        print(f"Saved purchases NDJSON -> {ndjson_path.resolve()}")
        print("\nCompleted BTC purchases (STRICT):")
        view_cols = ["filingDate","form","btc","usd","avg_usd_per_btc","source_url"]
        def fmt_small_big(x):
            return f"{x:,.8f}" if isinstance(x, (int, float)) and abs(x) < 1000 else (f"{x:,.2f}" if isinstance(x, (int, float)) else x)
        print(dfp[view_cols].to_string(index=False, float_format=lambda x: f"{x:,.8f}" if abs(x) < 1000 else f"{x:,.2f}"))
        tot_btc = dfp["btc"].dropna().sum()
        tot_usd = dfp["usd"].dropna().sum()
        wavg = (tot_usd / tot_btc) if tot_btc else None
        print("\nTotals (completed purchases):")
        print(f"  BTC: {tot_btc:,.8f}")
        print(f"  USD: ${tot_usd:,.2f}")
        if wavg:
            print(f"  Weighted avg $/BTC: ${wavg:,.2f}")
    else:
        print("\nNo completed BTC purchase events found.")

    # =========================
    # Save + print: Holdings Snapshots
    # =========================
    if rows_holdings:
        dfh = pd.DataFrame(rows_holdings)

        # Harmonize "as_of" to ISO date; if missing, fall back to filingDate.date()
        if "as_of" in dfh.columns:
            dfh["as_of"] = dfh.apply(
                lambda r: (r.get("as_of") or (r["filingDate"].date().isoformat() if pd.notna(r["filingDate"]) else None)),
                axis=1
            )

        # Keep only table fields (no snippet text)
        keep_cols = [
            "as_of","filingDate","ticker","year","form","accessionNumber","source_url",
            "btc","usd_cost_total","avg_usd_per_btc","market_value_usd",
            "doc_index","event_index","cik"
        ]
        dfh = dfh[[c for c in keep_cols if c in dfh.columns]] \
                .sort_values(["as_of","filingDate","accessionNumber","doc_index","event_index"]) \
                .reset_index(drop=True)

        # Save CSV/NDJSON (generic names)
        csv_path_h = outdir / f"{args.ticker.lower()}_btc_holdings_{args.year}.csv"
        dfh.to_csv(csv_path_h, index=False)

        ndjson_path_h = outdir / f"{args.ticker.lower()}_btc_holdings_{args.year}.ndjson"
        with ndjson_path_h.open("w", encoding="utf-8") as f:
            for _, r in dfh.iterrows():
                rec = {}
                for k, v in r.items():
                    rec[k] = v.isoformat() if hasattr(v, "isoformat") else v
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"\nSaved holdings CSV -> {csv_path_h.resolve()}")
        print(f"Saved holdings NDJSON -> {ndjson_path_h.resolve()}")

        # Console preview (clean table only)
        print("\nBTC Holdings snapshots (as_of asc):")
        show = dfh[["as_of","filingDate","form","btc","usd_cost_total","avg_usd_per_btc","market_value_usd","source_url"]].copy()

        def fmtnum(x):
            if pd.isna(x): return ""
            return f"{x:,.8f}" if abs(x) < 1000 else f"{x:,.2f}"

        for c in ["btc","usd_cost_total","avg_usd_per_btc","market_value_usd"]:
            if c in show.columns:
                show[c] = show[c].apply(fmtnum)

        print(show.to_string(index=False))
    else:
        print("\nNo BTC holdings snapshots found.")

if __name__ == "__main__":
    main()
