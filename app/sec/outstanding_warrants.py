#!/usr/bin/env python3
# app/sec/warrants_daily.py
# Extract outstanding warrant data from SEC HTML/iXBRL docs (10-Q / 10-K, optionally 20-F / 6-K).
# - Crawls EDGAR "submissions" + per-filing index.json to find candidate HTML docs
# - Heuristics to identify "Warrants" tables/notes near headings
# - Parses units (in thousands/millions), numbers outstanding, exercise price, term/expiry
# - Daily mode probes only accessions whose files changed in the past N hours
#
# pip install requests lxml pandas python-dateutil

import argparse
import datetime as dt
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
import requests
from lxml import etree
from urllib.parse import urljoin
from dateutil import parser as dateparse

# =========================
# SEC-polite config
# =========================
USER_AGENT = "Ally Zach <ally@panteracapital.com>"  # set a real contact per SEC guidance
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# =========================
# Logging
# =========================
def setup_logger(level: str = "INFO") -> logging.Logger:
    lg = logging.getLogger("sec_warrants")
    lg.setLevel(getattr(logging, level.upper(), logging.INFO))
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    lg.addHandler(sh)
    return lg

log = logging.getLogger("sec_warrants")  # configured in main()

# =========================
# Liability table filter
# =========================
# Anything that looks like warrant *liability* / derivative liability / fair value rollforward, etc.
LIABILITY_HINT_RX = re.compile(
    r"""
    \b(
        liabilit(?:y|ies)|
        warrant\s+liabilit(?:y|ies)|
        derivative\s+liabilit(?:y|ies)|
        liability[-\s]*classified\s+warrants?|
        classified\s+as\s+a?\s*liabilit(?:y|ies)|
        asc\s*815|
        fair\s*value|
        change\s+in\s+fair\s+value|
        remeasurement|
        mark[-\s]*to[-\s]*market|
        level\s*[123]\b
    )\b
    """,
    re.I | re.X,
)
def percent_numeric(vals: List[str]) -> float:
    """
    Return the fraction of cells that look like plain numeric values.
    Treats '1,234', '2.5', '(200,000)' as numeric; ignores '$' and commas.
    """
    if not vals:
        return 0.0
    hits = 0
    for v in vals:
        s = str(v or "").replace(",", "").replace("$", "").replace("\u00A0", " ")
        s = s.replace("(", "-").replace(")", "").strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
            hits += 1
    return hits / len(vals)

def _context_around_table(table_el: etree._Element, heading: str, rows: List[List[str]]) -> str:
    # reuse existing join/clean utilities you already have
    top = " ".join(" ".join(r) for r in rows[:2])
    parent = _join_text(table_el.getparent() or table_el)
    cap = ""
    cap_el = table_el.find(".//caption")
    if cap_el is not None:
        cap = _join_text(cap_el)
    return _clean(" ".join([heading or "", cap, top, parent]))

def is_liabilityish_table(rows: List[List[str]], heading: str, table_el: etree._Element) -> bool:
    ctx = _context_around_table(table_el, heading, rows).lower()
    # only trigger if we see liability-ish words, ideally alongside warrant references
    has_liab = bool(LIABILITY_HINT_RX.search(ctx))
    has_warrant = ("warrant" in ctx)
    # If it's clearly about warrant liabilities / fair value, skip.
    if has_warrant and has_liab:
        return True
    # Also skip generic liability rollforwards with "liabilities" in header+rows and money columns dominating
    if has_liab and percent_numeric(rows[0]) < 0.25:
        return True
    return False


# =========================
# HTTP helper (polite)
# =========================
def get(url: str) -> requests.Response:
    log.debug(f"GET {url}")
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code == 429:
        log.warning("HTTP 429 (rate-limited). Sleeping 1.5s and retrying…")
        time.sleep(1.5)
        resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(0.25)
    return resp

# =========================
# CIK + Filings + Files
# =========================
def resolve_cik(ticker: str) -> str:
    data = get(TICKER_MAP_URL).json()
    for item in data.values():
        if str(item.get("ticker", "")).upper() == ticker.upper():
            cik = f"{int(item['cik_str']):010d}"
            log.info(f"Resolved {ticker} -> CIK {cik}")
            return cik
    raise RuntimeError(f"Ticker not found in SEC map: {ticker}")

def fetch_filings(cik: str) -> pd.DataFrame:
    url = f"{BASE_SUBMISSIONS}/CIK{cik}.json"
    log.info(f"Fetching submissions: {url}")
    j = get(url).json()
    recent = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df
    for col in ("reportDate", "filingDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "acceptanceDateTime" in df.columns:
        s = pd.to_datetime(df["acceptanceDateTime"], utc=True, errors="coerce")
        df["acceptanceDateTime"] = s.dt.tz_convert(None)  # naive UTC
    df["form"] = df["form"].astype(str)
    return df

def cik_no_zeros(cik_10: str) -> str:
    return str(int(cik_10))

def accession_nodash(acc: str) -> str:
    return acc.replace("-", "")

def filing_base_url(cik_10: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros(cik_10)}/{accession_nodash(accession)}/"

def list_filing_files(cik_10: str, accession: str) -> pd.DataFrame:
    base = filing_base_url(cik_10, accession)
    url = urljoin(base, "index.json")
    j = get(url).json()
    df = pd.DataFrame(j.get("directory", {}).get("item", []))
    if df.empty:
        return df
    df["url"] = df["name"].apply(lambda n: urljoin(base, n))
    df["base_url"] = base
    if "last-modified" in df.columns:
        df["last_modified"] = pd.to_datetime(df["last-modified"], utc=True, errors="coerce").dt.tz_convert(None)
    else:
        df["last_modified"] = pd.NaT
    if "size" in df.columns:
        df["size"] = pd.to_numeric(df["size"], errors="coerce")
    return df

def is_html_like(name: str) -> bool:
    n = str(name).lower()
    return n.endswith((".htm", ".html", ".xhtml"))

# =========================
# HTML parsing utils
# =========================
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _join_text(node: etree._Element) -> str:
    return _clean("".join(node.itertext()))

def html_root_from_bytes(html_bytes: bytes) -> Optional[etree._Element]:
    parser = etree.HTMLParser(recover=True, huge_tree=True)
    try:
        return etree.HTML(html_bytes, parser=parser)
    except Exception as e:
        log.debug(f"HTML parse error: {e}")
        return None

def collect_tables(root: etree._Element) -> List[etree._Element]:
    return list(root.xpath(".//table")) if root is not None else []

def table_rows(table: etree._Element, max_rows: int = 80) -> List[List[str]]:
    rows: List[List[str]] = []
    for tr in table.xpath(".//tr"):
        cells = tr.xpath("./th|./td")
        if not cells:
            continue
        row = [_join_text(c) for c in cells]
        rows.append(row)
        if len(rows) >= max_rows:
            break
    w = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (w - len(r)) for r in rows]
    return rows

def heading_near_table(table: etree._Element, hop_limit: int = 12) -> str:
    cap = table.find(".//caption")
    if cap is not None:
        return _join_text(cap)
    prev = table.getprevious()
    hops = 0
    while prev is not None and hops < hop_limit:
        if hasattr(prev, "tag") and isinstance(prev.tag, str):
            txt = _join_text(prev)
            if txt:
                if prev.tag.lower() in {"h1","h2","h3","h4","h5","h6"} or len(txt) < 160:
                    return txt
        prev = prev.getprevious()
        hops += 1
    return ""

# =========================
# Units + number parsing
# =========================
_UNIT_RX = re.compile(r"in\s+(thousand|thousands|million|millions|billion|billions)\b", re.I)

def detect_unit_multiplier(context_text: str) -> float:
    t = _clean(context_text).lower()
    if "in thousands" in t or "(thousands)" in t: return 1_000.0
    if "in million"   in t or "(millions)"  in t: return 1_000_000.0
    if "in billion"   in t or "(billions)"  in t: return 1_000_000_000.0
    m = _UNIT_RX.search(t)
    if not m: return 1.0
    word = m.group(1).lower()
    return 1_000.0 if "thousand" in word else (1_000_000.0 if "million" in word else 1_000_000_000.0)

NUM_RX = re.compile(r"[-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?")

def parse_float(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).replace("\u00A0", " ")
    neg = s.strip().startswith("(") and s.strip().endswith(")")
    m = NUM_RX.search(s.replace("$",""))
    if not m: return None
    v = float(m.group(0).replace(",","").replace(" ",""))
    return -v if neg else v

def parse_int_units(s: str, mult: float) -> Optional[float]:
    v = parse_float(s)
    if v is None: return None
    return float(v) * float(mult)

def parse_money(s: str) -> Optional[float]:
    if s is None: return None
    s = str(s).replace("\u00A0", " ")
    if "$" not in s and NUM_RX.search(s) is None: return None
    v = parse_float(s)
    return v

def parse_date_soft(s: str) -> Optional[str]:
    try:
        d = dateparse.parse(s, fuzzy=True, default=dt.datetime(2000,1,1))
        # Ignore silly 2000 default if nothing parsed
        if d.year < 1900: return None
        return d.date().isoformat()
    except Exception:
        return None

# =========================
# Warrant detectors
# =========================
WARRANT_HDR_RX = re.compile(
    r"\bwarrant(s)?\b|warrant\s+liability|prefunded\s+warrants?|public\s+warrants?|private\s+placement\s+warrants?",
    re.I
)

WARRANT_TABLE_NAME_HINTS = (
    "warrants outstanding",
    "summary of warrants",
    "warrant activity",
    "warrant rollforward",
    "warrant liabilities",
)

EXERCISE_RX = re.compile(r"(exercise\s+price|strike\s+price)\b", re.I)
EXPIRE_RX   = re.compile(r"(expiration|expiry|maturity|term)\b", re.I)
OUTST_RX    = re.compile(r"(warrants?\s+outstanding|outstanding\s+warrants?)\b", re.I)
ISSUED_RX   = re.compile(r"(warrants?\s+issued)\b", re.I)
EXER_RX     = re.compile(r"(warrants?\s+exercised)\b", re.I)
EXPIRE_ROW_RX= re.compile(r"(warrants?\s+expired)\b", re.I)

def looks_like_warrant_table(rows: List[List[str]], heading: str) -> bool:
    h = (heading or "").lower()
    if any(k in h for k in WARRANT_TABLE_NAME_HINTS): return True
    # quick scan first row(s)
    top = " ".join(" ".join(r) for r in rows[:2]).lower()
    return ("warrant" in top)

def extract_warrant_values(
    rows: List[List[str]],
    heading: str,
    table_el: Optional[etree._Element]
) -> Dict[str, Any]:
    """
    Robust extractor for 'warrants' rollup tables like:

        Class | Amount Outstanding | Exercise Price | Expiration Date

    Works even if the word 'outstanding' isn't in row labels and relies on headers.
    Falls back to label-based scanning when headers are sparse.
    """
    # ---- context + units ----
    header_blob = (heading or "") + " " + " ".join(" ".join(r) for r in rows[:3])
    if table_el is not None:
        # pull nearby text/captions/parent for unit hints
        parent_txt = _join_text(table_el.getparent() or table_el)
        header_blob = _clean(header_blob + " " + parent_txt)
    mult = detect_unit_multiplier(header_blob)

    out: Dict[str, Any] = {
        "warrants_outstanding": None,
        "exercise_price_usd": None,
        "warrants_issued": None,
        "warrants_exercised": None,
        "warrants_expired": None,
        "warrant_expiration_date": None,
        "warrant_term_years": None,
        "units_multiplier": mult,
    }

    if not rows:
        return out

    # ---- header mapping ----
    headers = rows[0]
    hlower = [(h or "").strip().lower() for h in headers]
    w = len(headers)

    # canonical header indices (very tolerant)
    def find_col(keys: List[str]) -> Optional[int]:
        for j, h in enumerate(hlower):
            if all(k in h for k in keys):
                return j
        return None

    amt_col = (
        find_col(["amount", "outstanding"])
        or find_col(["outstanding"])
        or find_col(["quantity"])
        or find_col(["number"])
        or None
    )
    ex_col  = find_col(["exercise", "price"]) or find_col(["strike", "price"])
    exp_col = find_col(["expiration"]) or find_col(["expiry"]) or find_col(["maturity"])

    # helper: first non-empty cell text
    def lab(row: List[str]) -> str:
        if not row: return ""
        if row[0].strip(): return row[0]
        if len(row) > 1 and row[1].strip(): return row[1]
        return row[0]

    # ---- pass A: header-driven extraction (covers your screenshot case) ----
    totals_seen: List[float] = []
    per_row_counts: List[float] = []
    exercise_prices: List[float] = []

    for r in rows[1:]:
        # counts
        if amt_col is not None and amt_col < len(r):
            v = parse_int_units(r[amt_col], mult)
            if v is not None and v >= 1:
                per_row_counts.append(v)
                if lab(r).strip().lower().startswith("total"):
                    totals_seen.append(v)

        # exercise price
        if ex_col is not None and ex_col < len(r):
            mny = parse_money(r[ex_col])
            if mny is not None and mny > 0:
                exercise_prices.append(mny)

        # expiration date(s)
        if exp_col is not None and exp_col < len(r):
            d = parse_date_soft(r[exp_col])
            if d and out["warrant_expiration_date"] is None:
                out["warrant_expiration_date"] = d

        # term in years if present in row text
        m = re.search(r"(\d+(?:\.\d+)?)\s+year", " ".join(r), re.I)
        if m and out["warrant_term_years"] is None:
            try:
                out["warrant_term_years"] = float(m.group(1))
            except Exception:
                pass

    # choose outstanding:
    # 1) prefer explicit "Total" row, else 2) sum rows if header implies counts, else 3) max row
    if totals_seen:
        out["warrants_outstanding"] = max(totals_seen)
    elif per_row_counts:
        # If header clearly denotes counts, sum; otherwise take max (safer for mixed tables).
        if amt_col is not None and ("outstanding" in (headers[amt_col] or "").lower()):
            out["warrants_outstanding"] = float(sum(per_row_counts))
        else:
            out["warrants_outstanding"] = max(per_row_counts)

    if exercise_prices:
        # pick the max (common when multiple classes have different strikes)
        out["exercise_price_usd"] = max(exercise_prices)

    # ---- pass B: label-based fallback (when headers are weak) ----
    # Uses existing regexes EXERCISE_RX / EXPIRE_RX / OUTST_RX if present globally.
    if out["warrants_outstanding"] is None or out["exercise_price_usd"] is None or out["warrant_expiration_date"] is None:
        # detect candidate count/money columns by content
        def is_count_col(j: int) -> bool:
            vals = [rows[i][j] for i in range(1, min(20, len(rows))) if j < len(rows[i])]
            if not vals: return False
            nums = sum(1 for v in vals if NUM_RX.search(v or ""))
            monies = sum(1 for v in vals if "$" in (v or ""))
            return nums >= max(3, int(0.4 * len(vals))) and monies <= int(0.15 * len(vals))

        def is_money_col(j: int) -> bool:
            vals = [rows[i][j] for i in range(1, min(20, len(rows))) if j < len(rows[i])]
            if not vals: return False
            monies = sum(1 for v in vals if "$" in (v or "") or EXERCISE_RX.search(v or ""))
            return monies >= int(0.25 * len(vals))

        count_candidates = [j for j in range(1, w) if is_count_col(j)]
        money_candidates = [j for j in range(1, w) if is_money_col(j)]

        for r in rows:
            label = lab(r).lower()

            # counts from rows with 'warrant' in the label (e.g., "Class C-1 Warrants")
            if "warrant" in label and (out["warrants_outstanding"] is None):
                for j in (count_candidates or range(1, w)):
                    if j < len(r):
                        v = parse_int_units(r[j], mult)
                        if v is not None and v >= 1:
                            out["warrants_outstanding"] = max(out["warrants_outstanding"] or 0.0, v)

            # exercise price from money-ish cols
            if (out["exercise_price_usd"] is None) and ("exercise" in label or "strike" in label or "warrant" in label):
                for j in (money_candidates or range(1, w)):
                    if j < len(r):
                        v = parse_money(r[j])
                        if v is not None and v > 0:
                            out["exercise_price_usd"] = max(out["exercise_price_usd"] or 0.0, v)

            # expiration date
            if (out["warrant_expiration_date"] is None) and ("expir" in label or "maturity" in label or "warrant" in label):
                for cell in r:
                    d = parse_date_soft(cell)
                    if d:
                        out["warrant_expiration_date"] = d
                        break

            # term years
            if out["warrant_term_years"] is None:
                m = re.search(r"(\d+(?:\.\d+)?)\s+year", " ".join(r), re.I)
                if m:
                    try:
                        out["warrant_term_years"] = float(m.group(1))
                    except Exception:
                        pass

    # clean tiny artifacts
    for k in ("warrants_outstanding","warrants_issued","warrants_exercised","warrants_expired"):
        if out[k] is not None and out[k] < 1:
            out[k] = None

    return out
def extract_warrant_records(
    rows: List[List[str]],
    heading: str,
    table_el: Optional[etree._Element]
) -> List[Dict[str, Any]]:
    """
    Convert an entire 'Warrants' table into normalized records (one per row).
    Expected headers (very tolerant): Amount Outstanding | Exercise Price | Expiration Date
    Keeps 'Total' rows with a flag so you can filter later.
    """
    if not rows:
        return []

    # Units context
    header_blob = (heading or "") + " " + " ".join(" ".join(r) for r in rows[:3])
    if table_el is not None:
        parent_txt = _join_text(table_el.getparent() or table_el)
        header_blob = _clean(header_blob + " " + parent_txt)
    mult = detect_unit_multiplier(header_blob)

    headers = rows[0]
    hlower = [(h or "").strip().lower() for h in headers]
    w = len(headers)

    # Flexible header index finders
    def find_col(keys: List[str]) -> Optional[int]:
        for j, h in enumerate(hlower):
            if all(k in h for k in keys):
                return j
        return None

    label_col = 0  # first non-empty cell as the row label/class
    amt_col = (
        find_col(["amount", "outstanding"])
        or find_col(["outstanding"])
        or find_col(["quantity"])
        or find_col(["number"])
    )
    ex_col  = find_col(["exercise", "price"]) or find_col(["strike", "price"])
    exp_col = find_col(["expiration"]) or find_col(["expiry"]) or find_col(["maturity"])

    # Fallback: if we don't see the trio but the table still looks warrant-y, we still emit raw cells
    def row_label(r: List[str]) -> str:
        if not r: return ""
        if r[0].strip(): return r[0]
        if len(r) > 1 and r[1].strip(): return r[1]
        return r[0]

    records: List[Dict[str, Any]] = []
    for r in rows[1:]:
        lab = row_label(r)
        if not any(cell.strip() for cell in r):
            continue

        rec: Dict[str, Any] = {
            "row_label": lab,
            "warrants_outstanding": None,
            "exercise_price_usd": None,
            "warrant_expiration_date": None,
            "warrant_term_years": None,
            "is_total_row": lab.strip().lower().startswith("total"),
            "units_multiplier": mult,
            "table_heading": (heading or "")[:200],
        }

        # Parse columns when present
        if amt_col is not None and amt_col < len(r):
            v = parse_int_units(r[amt_col], mult)
            if v is not None and v >= 1:
                rec["warrants_outstanding"] = v

        if ex_col is not None and ex_col < len(r):
            mny = parse_money(r[ex_col])
            if mny is not None and mny > 0:
                rec["exercise_price_usd"] = mny

        if exp_col is not None and exp_col < len(r):
            d = parse_date_soft(r[exp_col])
            if d:
                rec["warrant_expiration_date"] = d

        # Term years (if present anywhere in row)
        m = re.search(r"(\d+(?:\.\d+)?)\s+year", " ".join(r), re.I)
        if m:
            try:
                rec["warrant_term_years"] = float(m.group(1))
            except Exception:
                pass

        # If headers were weak, try soft fallbacks:
        if rec["warrants_outstanding"] is None:
            # choose first numeric-ish cell as count (excluding money cells)
            for j in range(1, w):
                if j < len(r) and "$" not in (r[j] or ""):
                    v = parse_int_units(r[j], mult)
                    if v is not None and v >= 1:
                        rec["warrants_outstanding"] = v
                        break

        if rec["exercise_price_usd"] is None:
            # Only look for money values (with $ or in typical price ranges)
            for j in range(1, w):
                if j < len(r):
                    cell = r[j] or ""
                    # Only consider cells that look like prices (have $ or are in typical price ranges)
                    if "$" in cell or (parse_money(cell) is not None and parse_money(cell) > 0 and parse_money(cell) < 1000):
                        v = parse_money(cell)
                        if v is not None and v > 0:
                            rec["exercise_price_usd"] = v
                            break

        if rec["warrant_expiration_date"] is None:
            for cell in r:
                d = parse_date_soft(cell)
                if d:
                    rec["warrant_expiration_date"] = d
                    break

        records.append(rec)

    return records


def scan_doc_for_warrants(html_bytes: bytes, doc_url: str = "", diagnostics: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    root = html_root_from_bytes(html_bytes)
    if root is None:
        return {"ok": False, "reason": "parse_error"}, []

    diags: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_score = -1.0

    for idx, t in enumerate(collect_tables(root), start=1):
        rows = table_rows(t, max_rows=80)
        if not rows: continue
        heading = heading_near_table(t)
        if not (looks_like_warrant_table(rows, heading) or WARRANT_HDR_RX.search(heading or "")):
            # quick text sniff inside table
            txt = " ".join(" ".join(r) for r in rows[:3]).lower()
            if "warrant" not in txt:
                continue
        # in scan_doc_for_warrants(...) right after warrant-ish check:
        if is_liabilityish_table(rows, heading, t):
            continue

        vals = extract_warrant_values(rows, heading, t)
        got = any(vals.get(k) is not None for k in ("warrants_outstanding","exercise_price_usd","warrant_expiration_date","warrants_issued","warrants_exercised"))
        score = 0.0
        if "warrant" in (heading or "").lower(): score += 1.2
        score += 1.5 if vals.get("warrants_outstanding") else 0.8 if got else 0.0
        if vals.get("exercise_price_usd") is not None: score += 0.6
        if vals.get("warrant_expiration_date") is not None or vals.get("warrant_term_years") is not None: score += 0.4

        cand = {
            "ok": got,
            "table_index": idx,
            "heading": (heading or "")[:200],
            "score": round(score, 3),
            **vals,
        }
        if diagnostics:
            first_labels = "; ".join([rows[i][0] for i in range(min(6, len(rows))) if rows[i]])
            diags.append({"doc_url": doc_url, "table_index": idx, "heading": cand["heading"], "score": score, "first_labels": first_labels[:220]})

        if score > best_score:
            best_score = score
            best = cand

    return best or {"ok": False, "reason": "no_warrant_table"}, diags

def scan_doc_for_warrant_row_records(html_bytes: bytes, doc_url: str = "", diagnostics: bool = False) -> List[Dict[str, Any]]:
    root = html_root_from_bytes(html_bytes)
    if root is None:
        return []
    out: List[Dict[str, Any]] = []
    for t in collect_tables(root):
        rows = table_rows(t, max_rows=120)
        if not rows:
            continue
        heading = heading_near_table(t)
        # only consider tables that mention "warrant" in heading/top text, to avoid false positives
        top = " ".join(" ".join(r) for r in rows[:2]).lower()
        if ("warrant" not in (heading or "").lower()) and ("warrant" not in top):
            continue
        # in scan_doc_for_warrant_row_records(...) right after the warrant-ish check:
        if is_liabilityish_table(rows, heading, t):
            continue

        out.extend(extract_warrant_records(rows, heading, t))
    return out

# =========================
# URL candidate ranking
# =========================
def candidate_html_urls(files_df: pd.DataFrame, primary_doc: str, prelimit: Optional[int]) -> List[str]:
    if files_df.empty:
        return []
    df = files_df[files_df["name"].apply(is_html_like)].copy()
    if df.empty:
        return []
    prim = (primary_doc or "").lower()

    def pre_score(name: str) -> float:
        n = name.lower()
        score = 0.0
        if n == prim: score += 0.9
        for k in ("10-k","10q","10-q","20-f","6-k","warrant","equity","notes","footnote","financial"):
            if k in n: score += 0.45
        if re.search(r"\bex[-_]?99", n):
            score -= 0.3
        return score

    df["__pre"] = df["name"].apply(pre_score)
    df = df.sort_values(["__pre", "name"], ascending=[False, True])
    urls = df["url"].tolist()
    if prelimit is not None and prelimit > 0:
        urls = urls[:prelimit]
    return urls

# =========================
# Collection
# =========================
KEEP_COLS = [
    "ticker","form","accessionNumber","primaryDocument","filingDate","reportDate","acceptanceDateTime",
    "source_url","warrants_outstanding","exercise_price_usd","warrants_issued","warrants_exercised",
    "warrants_expired","warrant_expiration_date","warrant_term_years","units_multiplier","heading","score"
]

def collect_warrants_for_ticker(
    ticker: str,
    forms: str = "10-K,10-Q,20-F,6-K",
    from_date: str = "2024-01-01",
    limit_filings: int = 300,
    max_docs_per_filing: int = 10,
    log_every: int = 10,
    diagnostics: bool = False,
    year: int = 0,
    year_by: str = "accession",
    accession_whitelist: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cik = resolve_cik(ticker)
    filings = fetch_filings(cik)
    if filings.empty:
        return pd.DataFrame(columns=KEEP_COLS), pd.DataFrame()

    allowed = {f.strip().upper() for f in forms.split(",") if f.strip()}
    subset = filings[
        filings["form"].str.upper().apply(lambda fu: any(fu == a or fu.startswith(a + "/") for a in allowed))
    ].copy()

    subset["acc_year"] = (
        subset["accessionNumber"].astype(str).str.extract(r"^\d{10}-(\d{2})-")[0]
        .astype("float").apply(lambda y: 2000 + int(y) if pd.notna(y) else pd.NA)
    )

    if year and int(year) > 0:
        if year_by == "accession":
            subset = subset[subset["acc_year"].eq(int(year))]
        else:
            subset = subset[pd.to_datetime(subset["filingDate"], errors="coerce").dt.year.eq(int(year))]
    else:
        floor_year = pd.Timestamp(from_date).year
        subset = subset[subset["acc_year"].ge(floor_year)]

    if accession_whitelist:
        subset = subset[subset["accessionNumber"].isin(accession_whitelist)]

    subset = subset.sort_values("filingDate").tail(limit_filings)
    if subset.empty:
        return pd.DataFrame(columns=KEEP_COLS), pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    diag_rows_all: List[Dict[str, Any]] = []

    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        if i == 1 or (i % log_every == 0):
            stamp = row.get("filingDate")
            stamp = stamp.date() if pd.notna(stamp) else ""
            log.info(f"[{i}/{len(subset)}] {row.get('form')} {row.get('accessionNumber')} {stamp}")

        accession = row.get("accessionNumber")
        primary = (row.get("primaryDocument") or "")
        try:
            files = list_filing_files(cik, accession)
        except Exception as e:
            log.warning(f"  index.json fetch failed for {accession}: {e}")
            continue

        urls = candidate_html_urls(files, primary_doc=primary, prelimit=max_docs_per_filing if max_docs_per_filing > 0 else None)
        if not urls:
            continue

        best_rec = None
        best_score = -1.0

        for uidx, u in enumerate(urls, start=1):
            log.debug(f"  Doc {uidx}/{len(urls)} -> {u}")
            try:
                resp = get(u)
                rec, diags = scan_doc_for_warrants(resp.content, doc_url=u, diagnostics=diagnostics)
                diag_rows_all.extend(diags)
                if not rec.get("ok"):
                    continue
                if rec["score"] > best_score:
                    best_score = rec["score"]
                    best_rec = rec
            except Exception as e:
                log.warning(f"  Error parsing doc {u}: {e}")
                continue

        if best_rec:
            rows.append({
                "ticker": ticker.upper(),
                "form": row.get("form"),
                "accessionNumber": accession,
                "primaryDocument": primary,
                "filingDate": row.get("filingDate"),
                "reportDate": row.get("reportDate"),
                "acceptanceDateTime": row.get("acceptanceDateTime"),
                "source_url": urls[0] if urls else "",
                **{k: best_rec.get(k) for k in ["warrants_outstanding","exercise_price_usd","warrants_issued","warrants_exercised","warrants_expired","warrant_expiration_date","warrant_term_years","units_multiplier","heading","score"]},
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["filingDate","score"], ascending=[True, False]).reset_index(drop=True)
        df = df[[c for c in KEEP_COLS if c in df.columns]]

    diag_df = pd.DataFrame(diag_rows_all) if diagnostics else pd.DataFrame()
    return df, diag_df
ROW_KEEP_COLS = [
    "ticker","form","accessionNumber","primaryDocument","filingDate","reportDate","acceptanceDateTime",
    "source_url","row_label","warrants_outstanding","exercise_price_usd","warrant_expiration_date",
    "warrant_term_years","is_total_row","units_multiplier","table_heading"
]

def collect_warrant_rows_for_ticker(
    ticker: str,
    forms: str = "10-K,10-Q",
    from_date: str = "2024-01-01",
    limit_filings: int = 300,
    max_docs_per_filing: int = 10,
    log_every: int = 10,
    year: int = 0,
    year_by: str = "accession",
    accession_whitelist: Optional[Set[str]] = None,
) -> pd.DataFrame:
    cik = resolve_cik(ticker)
    filings = fetch_filings(cik)
    if filings.empty:
        return pd.DataFrame(columns=ROW_KEEP_COLS)

    allowed = {f.strip().upper() for f in forms.split(",") if f.strip()}
    subset = filings[
        filings["form"].str.upper().apply(lambda fu: any(fu == a or fu.startswith(a + "/") for a in allowed))
    ].copy()

    subset["acc_year"] = (
        subset["accessionNumber"].astype(str).str.extract(r"^\d{10}-(\d{2})-")[0]
        .astype("float").apply(lambda y: 2000 + int(y) if pd.notna(y) else pd.NA)
    )

    if year and int(year) > 0:
        if year_by == "accession":
            subset = subset[subset["acc_year"].eq(int(year))]
        else:
            subset = subset[pd.to_datetime(subset["filingDate"], errors="coerce").dt.year.eq(int(year))]
    else:
        floor_year = pd.Timestamp(from_date).year
        subset = subset[subset["acc_year"].ge(floor_year)]

    if accession_whitelist:
        subset = subset[subset["accessionNumber"].isin(accession_whitelist)]

    subset = subset.sort_values("filingDate").tail(limit_filings)

    recs: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(subset.iterrows(), start=1):
        if i == 1 or (i % log_every == 0):
            stamp = row.get("filingDate")
            stamp = stamp.date() if pd.notna(stamp) else ""
            log.info(f"[{i}/{len(subset)}] {row.get('form')} {row.get('accessionNumber')} {stamp}")

        accession = row.get("accessionNumber")
        primary = (row.get("primaryDocument") or "")
        try:
            files = list_filing_files(cik, accession)
        except Exception as e:
            log.warning(f"  index.json fetch failed for {accession}: {e}")
            continue

        urls = candidate_html_urls(files, primary_doc=primary, prelimit=max_docs_per_filing if max_docs_per_filing > 0 else None)
        if not urls:
            continue

        for u in urls:
            try:
                resp = get(u)
                rows_here = scan_doc_for_warrant_row_records(resp.content, doc_url=u, diagnostics=False)
                for rrec in rows_here:
                    recs.append({
                        "ticker": ticker.upper(),
                        "form": row.get("form"),
                        "accessionNumber": accession,
                        "primaryDocument": primary,
                        "filingDate": row.get("filingDate"),
                        "reportDate": row.get("reportDate"),
                        "acceptanceDateTime": row.get("acceptanceDateTime"),
                        "source_url": u,
                        **rrec,
                    })
            except Exception as e:
                log.warning(f"  Error parsing doc {u}: {e}")
                continue

    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=ROW_KEEP_COLS)
    df = df[[c for c in ROW_KEEP_COLS if c in df.columns]]
    return df

# =========================
# DAILY PROBE: collect accessions touched in last N hours
# =========================
def collect_recent_accessions(
    ticker: str,
    forms: str,
    from_date: str,
    limit_filings: int,
    recent_hours: int,
    year: int,
    year_by: str,
) -> Set[str]:
    """Return accessions ACCEPTED within the last `recent_hours` (defaults to 24h)."""
    cutoff_dt = dt.datetime.utcnow() - dt.timedelta(hours=recent_hours)
    cutoff_d  = cutoff_dt.date()

    cik = resolve_cik(ticker)
    filings = fetch_filings(cik)
    if filings.empty:
        return set()

    # form filter
    allowed = {f.strip().upper() for f in forms.split(",") if f.strip()}
    subset = filings[
        filings["form"].str.upper().apply(lambda fu: any(fu == a or fu.startswith(a + "/") for a in allowed))
    ].copy()

    # optional year filter (same as before)
    subset["acc_year"] = (
        subset["accessionNumber"].astype(str).str.extract(r"^\d{10}-(\d{2})-")[0]
        .astype("float").apply(lambda y: 2000 + int(y) if pd.notna(y) else pd.NA)
    )
    if year and int(year) > 0:
        if year_by == "accession":
            subset = subset[subset["acc_year"].eq(int(year))]
        else:
            subset = subset[pd.to_datetime(subset["filingDate"], errors="coerce").dt.year.eq(int(year))]
    else:
        floor_year = pd.Timestamp(from_date).year
        subset = subset[subset["acc_year"].ge(floor_year)]

    # sort & trim (keeps perf sane)
    subset = subset.sort_values("filingDate").tail(limit_filings)
    if subset.empty:
        return set()

    # --- NEW: 24h acceptance-time filter (fallback to filingDate if acceptance is missing) ---
    acc = pd.to_datetime(subset.get("acceptanceDateTime"), errors="coerce")
    filing = pd.to_datetime(subset.get("filingDate"), errors="coerce")

    mask_accept = acc.notna() & (acc >= cutoff_dt)
    # filingDate is date-only; compare to cutoff date
    mask_filing = filing.notna() & (filing.dt.date >= cutoff_d)

    recent = subset.loc[mask_accept | mask_filing, "accessionNumber"]
    return set(recent.astype(str).tolist())


# =========================
# PUBLIC API
# =========================
def get_sec_warrants_daily_df(
    tickers: List[str],
    forms: str = "10-K,10-Q,20-F,6-K",
    from_date: str = "2024-01-01",
    limit_filings: int = 300,
    max_docs_per_filing: int = 10,
    diagnostics: bool = False,
    year: int = 0,
    year_by: str = "accession",
    recent_hours: int = 24,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for t in (tickers or []):
        log.info(f"=== DAILY TICKER {t} (probe last {recent_hours}h) ===")
        acc_wh = collect_recent_accessions(
            ticker=t, forms=forms, from_date=from_date, limit_filings=limit_filings,
            recent_hours=recent_hours, year=year, year_by=year_by,
        )
        if not acc_wh:
            log.info(f"[{t}] No accessions touched in last {recent_hours}h.")
            continue
        df_t, _ = collect_warrants_for_ticker(
            ticker=t, forms=forms, from_date=from_date, limit_filings=limit_filings,
            max_docs_per_filing=max_docs_per_filing, diagnostics=diagnostics,
            year=year, year_by=year_by, accession_whitelist=acc_wh,
        )
        if not df_t.empty:
            frames.append(df_t)
    if not frames:
        return pd.DataFrame(columns=KEEP_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in KEEP_COLS if c in out.columns]]

def get_sec_warrants_full_df(
    tickers: List[str],
    forms: str = "10-K,10-Q,20-F,6-K",
    from_date: str = "2024-01-01",
    limit_filings: int = 300,
    max_docs_per_filing: int = 10,
    diagnostics: bool = False,
    year: int = 0,
    year_by: str = "accession",
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for t in (tickers or []):
        df_t, _ = collect_warrants_for_ticker(
            ticker=t, forms=forms, from_date=from_date, limit_filings=limit_filings,
            max_docs_per_filing=max_docs_per_filing, diagnostics=diagnostics,
            year=year, year_by=year_by,
        )
        if not df_t.empty:
            frames.append(df_t)
    if not frames:
        return pd.DataFrame(columns=KEEP_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in KEEP_COLS if c in out.columns]]

# =========================
# Minimal CLI (no file writes)
# =========================
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SEC warrants extractor (tables/notes).")
    ap.add_argument("--tickers", default="ETHZ,BMNR,BTCS,NAKA,SBET")
    ap.add_argument("--forms", default="10-K,10-Q")
    ap.add_argument("--from-date", default="2024-06-01")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--year", type=int, default=0)
    ap.add_argument("--year-by", choices=["accession","filingdate"], default="accession")
    ap.add_argument("--max-docs", type=int, default=10)
    ap.add_argument("--recent-hours", type=int, default=24)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--daily", action="store_true", default=True, help="Use recent-changes probe (default)")
    ap.add_argument("--full", action="store_false", dest="daily", help="Scan all filings (override)")

    ap.add_argument("--out", default="", help="CSV output path (optional). If empty, a dated filename is used.")
    ap.add_argument(
        "--summary", 
        action="store_true",
        help="Output one row per filing (summary mode). Default is all rows per table."
    )


    return ap.parse_args()


def main():
    args = _parse_args()
    global log
    log = setup_logger(args.log_level)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.daily:
        if args.summary:
            # daily summary (one row per filing)
            df = get_sec_warrants_daily_df(
                tickers=tickers,
                forms=args.forms,
                from_date=args.from_date,
                limit_filings=args.limit,
                max_docs_per_filing=args.max_docs,
                diagnostics=args.diagnostics,
                year=args.year,
                year_by=args.year_by,
                recent_hours=args.recent_hours,
            )
            mode = "daily_summary"
        else:
            # daily row mode (default)
            frames = []
            for t in tickers:
                acc_wh = collect_recent_accessions(
                    ticker=t,
                    forms=args.forms,
                    from_date=args.from_date,
                    limit_filings=args.limit,
                    recent_hours=args.recent_hours,
                    year=args.year,
                    year_by=args.year_by,
                )
                if not acc_wh:
                    continue
                frames.append(
                    collect_warrant_rows_for_ticker(
                        ticker=t,
                        forms=args.forms,
                        from_date=args.from_date,
                        limit_filings=args.limit,
                        max_docs_per_filing=args.max_docs,
                        year=args.year,
                        year_by=args.year_by,
                        accession_whitelist=acc_wh,
                    )
                )
            if any(not f.empty for f in frames):
                df = pd.concat(frames, ignore_index=True)
            else:
                df = pd.DataFrame(columns=ROW_KEEP_COLS)
            mode = "daily_rows"
    else:
        if args.summary:
            # full summary (one row per filing)
            df = get_sec_warrants_full_df(
                tickers=tickers,
                forms=args.forms,
                from_date=args.from_date,
                limit_filings=args.limit,
                max_docs_per_filing=args.max_docs,
                diagnostics=args.diagnostics,
                year=args.year,
                year_by=args.year_by,
            )
            mode = "full_summary"
        else:
            # full row mode (default)
            frames = []
            for t in tickers:
                frames.append(
                    collect_warrant_rows_for_ticker(
                        ticker=t,
                        forms=args.forms,
                        from_date=args.from_date,
                        limit_filings=args.limit,
                        max_docs_per_filing=args.max_docs,
                        year=args.year,
                        year_by=args.year_by,
                    )
                )
            if any(not f.empty for f in frames):
                df = pd.concat(frames, ignore_index=True)
            else:
                df = pd.DataFrame(columns=ROW_KEEP_COLS)
            mode = "full_rows"
    # --- drop rows with empty/null 'warrants_outstanding' (avoids header-ish rows) ---
    if not df.empty and "warrants_outstanding" in df.columns:
        df["warrants_outstanding"] = pd.to_numeric(df["warrants_outstanding"], errors="coerce")
        df = df[df["warrants_outstanding"].notna() & (df["warrants_outstanding"] >= 1)]

    # --- No file writes: just emit the DataFrame ---
    if not df.empty:
        # Pretty for CLI usage
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))
    else:
        log.info("No rows.")

    # (Optional) if someone imports this module and calls main(), let them grab df
    return df  # harmless when run as a script; useful when imported



if __name__ == "__main__":
    main()
