#!/usr/bin/env python3
# app/sec/cash_daily.py
# Extract cash / cash-like holdings from BALANCE SHEET tables in SEC HTML/iXBRL docs.
# Daily default probes accessions touched in the past N hours via index.json "last-modified"
# and extracts cash ONLY for those accessions.
#
# pip install requests lxml pandas

import argparse
import datetime
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

# =========================
# SEC-polite config
# =========================
USER_AGENT = "Ally Zach <ally@panteracapital.com>"  # <-- use real contact per SEC guidance
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

NS = {
    "x": "http://www.w3.org/1999/xhtml",
    "ix": "http://www.xbrl.org/2013/inlineXBRL",
    "dei": "http://xbrl.sec.gov/dei/2024-01-31",
    "xbrli": "http://www.xbrl.org/2003/instance",
}

MONTHS = (
    "(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    "Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DATE_RX = re.compile(rf"{MONTHS}\s+\d{{1,2}},\s+\d{{4}}", re.I)

IX_CASH_CONCEPTS = [
    "us-gaap:CashAndCashEquivalentsAtCarryingValue",
    "us-gaap:CashAndCashEquivalents",
    "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "us-gaap:MarketableSecuritiesCurrent",
    "us-gaap:ShortTermInvestments",
    "us-gaap:AvailableForSaleSecuritiesCurrent",
]

# =========================
# Logging
# =========================
def setup_logger(level: str = "INFO") -> logging.Logger:
    lg = logging.getLogger("sec_cash")
    lg.setLevel(getattr(logging, level.upper(), logging.INFO))
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(getattr(logging, level.upper(), logging.INFO))
    lg.addHandler(sh)
    return lg

log = logging.getLogger("sec_cash")  # configured in main()

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
    log.info(f"Recent filings: {len(df)}")
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
# HTML/iXBRL parsing utils
# =========================
def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _lower(s: str) -> str:
    return _clean(s).lower()

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

def table_rows(table: etree._Element, max_rows: int = 60) -> List[List[str]]:
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

def heading_near_table(table):
    cap = table.find(".//caption")
    if cap is not None:
        return _join_text(cap)

    prev = table.getprevious()
    hops = 0
    while prev is not None and hops < 12:
        if not (hasattr(prev, "tag") and isinstance(prev.tag, str)):
            prev = prev.getprevious()
            hops += 1
            continue

        txt = _join_text(prev)
        if txt:
            if prev.tag.lower() in {"h1","h2","h3","h4","h5","h6"} or len(txt) < 160:
                return txt
        prev = prev.getprevious()
        hops += 1
    return ""

def percent_numeric(vals: List[str]) -> float:
    if not vals:
        return 0.0
    n = 0
    for v in vals:
        s = str(v).replace(",", "").replace("$", "").replace("(", "-").replace(")", "").strip()
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
            n += 1
    return n / len(vals)

# =========================
# Money parsing + helpers
# =========================
def parse_money(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).replace("\u00A0", " ").replace("\u2009", " ").replace("\u202F", " ")
    s = s.replace("$", "").strip()
    if s in {"", "-", "—"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace("(", "").replace(")", "")
    m = re.search(r"[-+]?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?|[-+]?\d+(?:\.\d+)?", s)
    if not m:
        return None
    raw = m.group(0).replace(",", "").replace(" ", "")
    try:
        v = float(raw)
        return -v if neg else v
    except Exception:
        return None

def _is_currency_only(val: str) -> bool:
    if val is None:
        return False
    t = str(val).strip()
    return t in {"$", "$.", "$-", "US$", "USD"} or (t.startswith("$") and not re.search(r"\d", t))

def _label_cell(row: List[str]) -> str:
    if not row:
        return ""
    if row[0] and row[0].strip():
        return row[0]
    if len(row) > 1 and row[1] and row[1].strip():
        return row[1]
    return row[0] or ""

def find_header_dates(rows: List[List[str]]) -> Dict[int, str]:
    out = {}
    for i in range(min(8, len(rows))):
        for j, cell in enumerate(rows[i]):
            m = DATE_RX.search(cell)
            if m:
                out[j] = m.group(0)
    return out

def pick_value_column(rows: List[List[str]], prefer: str = "latest") -> int:
    if not rows or not rows[0]:
        return 1

    w = len(rows[0])

    def is_num_col(j: int) -> bool:
        vals = [rows[i][j] for i in range(min(20, len(rows))) if j < len(rows[i])]
        cur_ratio = sum(_is_currency_only(v) for v in vals) / max(1, len(vals))
        if cur_ratio >= 0.5:
            return False
        num_ratio = sum(parse_money(v) is not None for v in vals) / max(1, len(vals))
        return num_ratio >= 0.45

    candidates = [j for j in range(1, w) if is_num_col(j)]
    if not candidates:
        for j in range(1, w):
            vals = [rows[i][j] for i in range(min(12, len(rows))) if j < len(rows[i])]
            if any(parse_money(v) is not None for v in vals):
                return j
        return min(w - 1, 1)

    if prefer == "leftmost":
        return min(candidates)
    if prefer == "rightmost":
        return max(candidates)

    hdr_dates = find_header_dates(rows)
    parsed_dates: Dict[int, Optional[pd.Timestamp]] = {}
    for j in candidates:
        s = hdr_dates.get(j)
        parsed_dates[j] = pd.to_datetime(s, errors="coerce") if s else pd.NaT

    dated = [(j, dt) for j, dt in parsed_dates.items() if pd.notna(dt)]
    if dated:
        dated.sort(key=lambda x: x[1])
        return dated[-1][0]

    return min(candidates)

def detect_unit_multiplier(context_text: str) -> float:
    t = _lower(context_text)
    if "in thousands" in t or "(thousands)" in t:
        return 1_000.0
    if "in millions" in t or "(millions)" in t:
        return 1_000_000.0
    if "in billions" in t or "(billions)" in t:
        return 1_000_000_000.0
    m = re.search(r"\(.*in\s+(thousands|millions|billions).*?\)", t)
    if m:
        return {"thousands": 1_000.0, "millions": 1_000_000.0, "billions": 1_000_000_000.0}[m.group(1)]
    return 1.0

# =========================
# Balance sheet detection & extraction
# =========================
HEAD_TERMS = (
    "balance sheet",
    "balance sheets",
    "consolidated balance sheet",
    "consolidated balance sheets",
    "condensed consolidated balance sheet",
    "condensed consolidated balance sheets",
    "statement of financial position",
    "statements of financial position",
    "statement of financial condition",
    "statements of financial condition",
)

CASH_LABEL_RX = re.compile(r"\bcash(?:\s+and\s+cash\s+equivalents)?\b", re.I)
RESTRICTED_CASH_RX = re.compile(r"\brestricted\s+cash\b", re.I)
CASH_EQ_RESTRICTED_RX = re.compile(
    r"\bcash,\s*cash\s*equivalents(?:\s*and\s*restricted\s*cash)?\b|\bcash\s*and\s*cash\s*equivalents\s*and\s*restricted\s*cash\b",
    re.I,
)
MARKETABLE_RX = re.compile(r"\bmarketable\s+securities\b", re.I)
SHORT_TERM_INV_RX = re.compile(r"\bshort[-\s]*term\s+investments\b", re.I)
TREASURIES_RX = re.compile(r"\b(u\.?s\.?\s+treasur(?:y|ies)|treasur(?:y|ies))\b", re.I)
GOVT_SEC_RX = re.compile(r"\b(government\s+securities)\b", re.I)
STABLECOIN_RX = re.compile(r"\b(stablecoin(s)?|USDC|USDT|DAI|BUSD|GUSD|USDP|TUSD)\b", re.I)

ASSETS_HDR_RX = re.compile(r"^\s*(assets|current assets[: ]?)\b", re.I)
CURRENT_ASSETS_RX = re.compile(r"\bcurrent\s+assets\b", re.I)
TOTAL_CURRENT_ASSETS_RX = re.compile(r"^\s*total\s+current\s+assets\b", re.I)
LIAB_EQUITY_HDR_RX = re.compile(r"^\s*liabilities\b|\bliabilities\s+and\s+(?:stockholders|shareholders).+equity\b", re.I)

NEG_TABLE_HEADING_RX = re.compile(r"cash\s+flows?|statement\s+of\s+cash\s+flows?|operations|income\s+statement", re.I)
NEG_CASH_ROW_RX = re.compile(
    r"cash\s+flows?|net\s+cash|provided\s+by|used\s+in|cash\s+paid|cash\s+dividends|"
    r"beginning\s+of\s+period|end\s+of\s+period|supplemental|non[-\s]?cash",
    re.I
)

def looks_like_balance_sheet(rows: List[List[str]], heading_text: str) -> Tuple[bool, Dict[str, Any]]:
    h = _lower(heading_text)
    reason = {"why": "", "num_cols": 0, "core_hits": 0}

    if NEG_TABLE_HEADING_RX.search(h):
        return False, {"why": "neg_heading"}

    if any(term in h for term in HEAD_TERMS):
        w = len(rows[0]) if rows else 0
        for j in range(1, w):
            col_vals = [rows[i][j] for i in range(min(20, len(rows))) if j < len(rows[i])]
            if percent_numeric(col_vals) >= 0.45:
                reason["num_cols"] += 1
        reason["why"] = "heading"
        return (reason["num_cols"] >= 1), reason

    first_labels = [(_lower(_label_cell(r))) for r in rows[:30] if r]
    has_assets = any(ASSETS_HDR_RX.search(lab) or "total assets" in lab for lab in first_labels)
    has_liab_eq = any(
        LIAB_EQUITY_HDR_RX.search(lab)
        or "stockholders' equity" in lab
        or "stockholders’ equity" in lab
        or "shareholders' equity" in lab
        or "shareholders’ equity" in lab
        for lab in first_labels
    )
    if has_assets and has_liab_eq:
        w = len(rows[0]) if rows else 0
        num_cols = 0
        for j in range(1, w):
            col_vals = [rows[i][j] for i in range(min(20, len(rows))) if j < len(rows[i])]
            if percent_numeric(col_vals) >= 0.45:
                num_cols += 1
        if num_cols >= 1:
            return True, {"why": "labels", "num_cols": num_cols, "core_hits": 2}

    return False, {"why": "no"}

def extract_cash_like_values(rows: List[List[str]], heading: str) -> Dict[str, Any]:
    header_blob = heading + " " + " ".join(" ".join(r) for r in rows[:3])
    mult = detect_unit_multiplier(header_blob)

    col = pick_value_column(rows, prefer="latest")
    hdr_dates = find_header_dates(rows)
    col_date = hdr_dates.get(col)

    vals = {
        "cash_and_cash_equivalents_usd": None,
        "marketable_securities_usd": None,
        "short_term_investments_usd": None,
        "us_treasuries_usd": None,
        "government_securities_usd": None,
        "restricted_cash_usd": None,
        "combined_cash_cash_eq_restricted_usd": None,
        "stablecoins_usd": None,
        "units_multiplier": mult,
        "value_column_index": col,
        "value_column_date": col_date,
    }

    in_assets = False
    in_current_assets = False
    saw_current_header = False
    passed_liabilities = False

    for r in rows:
        if not r:
            continue
        label_raw = _label_cell(r)
        label = _lower(label_raw)

        if LIAB_EQUITY_HDR_RX.search(label):
            passed_liabilities = True
        if ASSETS_HDR_RX.search(label) or "total assets" in label:
            in_assets = True
        if CURRENT_ASSETS_RX.search(label):
            in_current_assets = True
            saw_current_header = True
        if TOTAL_CURRENT_ASSETS_RX.search(label):
            in_current_assets = False

        if passed_liabilities or not in_assets:
            continue
        if NEG_CASH_ROW_RX.search(label):
            continue

        num = parse_money(r[col]) if col < len(r) else None
        if num is None:
            continue
        v = num * mult

        allow_here = in_current_assets or (not saw_current_header)

        if allow_here:
            if CASH_EQ_RESTRICTED_RX.search(label):
                vals["combined_cash_cash_eq_restricted_usd"] = v
            if RESTRICTED_CASH_RX.search(label):
                vals["restricted_cash_usd"] = v
            elif CASH_LABEL_RX.search(label) and "restricted" not in label:
                vals["cash_and_cash_equivalents_usd"] = v
            elif MARKETABLE_RX.search(label):
                vals["marketable_securities_usd"] = v
            elif SHORT_TERM_INV_RX.search(label):
                vals["short_term_investments_usd"] = v
            elif TREASURIES_RX.search(label):
                vals["us_treasuries_usd"] = v
            elif GOVT_SEC_RX.search(label):
                vals["government_securities_usd"] = v
            elif STABLECOIN_RX.search(label):
                vals["stablecoins_usd"] = v

    agg = 0.0
    any_ = False
    for k in (
        "cash_and_cash_equivalents_usd",
        "marketable_securities_usd",
        "short_term_investments_usd",
        "us_treasuries_usd",
        "government_securities_usd",
        "stablecoins_usd",
    ):
        if vals[k] is not None:
            agg += vals[k]
            any_ = True
    vals["cash_like_total_excl_restricted_usd"] = agg if any_ else None
    return vals

def extract_cash_from_ixbrl_root(root: etree._Element) -> Dict[str, Any]:
    if root is None:
        return {}

    ctx_dates: Dict[str, Optional[pd.Timestamp]] = {}
    for ctx in root.xpath("//xbrli:context", namespaces=NS):
        cid = ctx.get("id")
        inst = ctx.find(".//xbrli:instant", namespaces=NS)
        dt = pd.to_datetime(inst.text, errors="coerce") if inst is not None else pd.NaT
        ctx_dates[cid] = dt

    latest: Dict[str, Tuple[pd.Timestamp, float]] = {}
    for concept in IX_CASH_CONCEPTS:
        for el in root.xpath(f"//ix:nonFraction[@name='{concept}']", namespaces=NS):
            val = parse_money("".join(el.itertext()))
            if val is None:
                continue
            dt = ctx_dates.get(el.get("contextRef"), pd.NaT)
            if pd.isna(dt):
                continue
            prev = latest.get(concept)
            if prev is None or dt > prev[0]:
                latest[concept] = (dt, val)

    if not latest:
        return {}

    out: Dict[str, Any] = {"ixbrl_used": True}
    def pick(*concepts):
        for c in concepts:
            if c in latest:
                return latest[c][1]
        return None

    out["cash_and_cash_equivalents_usd"] = pick(
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashAndCashEquivalents",
    )
    out["combined_cash_cash_eq_restricted_usd"] = pick(
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    )
    out["marketable_securities_usd"] = pick(
        "us-gaap:MarketableSecuritiesCurrent",
        "us-gaap:AvailableForSaleSecuritiesCurrent",
    )
    out["short_term_investments_usd"] = pick("us-gaap:ShortTermInvestments")

    agg = 0.0; any_ = False
    for k in ("cash_and_cash_equivalents_usd","marketable_securities_usd","short_term_investments_usd"):
        if out.get(k) is not None:
            agg += out[k]; any_ = True
    if any_:
        out["cash_like_total_excl_restricted_usd"] = agg

    if "us-gaap:CashAndCashEquivalentsAtCarryingValue" in latest:
        out["bs_value_column_date"] = latest["us-gaap:CashAndCashEquivalentsAtCarryingValue"][0].date().isoformat()
    elif "us-gaap:CashAndCashEquivalents" in latest:
        out["bs_value_column_date"] = latest["us-gaap:CashAndCashEquivalents"][0].date().isoformat()

    return out

# =========================
# Scanner
# =========================
def scan_doc_for_balance_sheet_and_extract(
    html_bytes: bytes, diag: bool = False, doc_url: str = ""
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    root = html_root_from_bytes(html_bytes)
    if root is None:
        return {"ok": False, "reason": "parse_error"}, []

    diag_rows: List[Dict[str, Any]] = []
    best_overall: Optional[Dict[str, Any]] = None
    best_overall_score = -1.0
    best_with_values: Optional[Dict[str, Any]] = None
    best_with_values_score = -1.0

    tables = collect_tables(root)
    for t_idx, t in enumerate(tables, start=1):
        rows = table_rows(t, max_rows=60)
        if not rows:
            continue

        heading = heading_near_table(t)
        ok, why = looks_like_balance_sheet(rows, heading)

        if diag:
            first_labels = "; ".join([_label_cell(rows[i]) for i in range(min(8, len(rows))) if rows[i]])
            diag_rows.append(
                {
                    "doc_url": doc_url,
                    "table_index": t_idx,
                    "ok": ok,
                    "why": why.get("why"),
                    "num_cols": why.get("num_cols"),
                    "core_hits": why.get("core_hits"),
                    "heading": heading[:160],
                    "first_labels": first_labels[:220],
                }
            )

        score = 0.0
        vals = {}
        if ok:
            if any(term in _lower(heading) for term in HEAD_TERMS):
                score += 1.2
            w = len(rows[0])
            num_cols = 0
            for j in range(1, w):
                col_vals = [rows[i][j] for i in range(min(20, len(rows))) if j < len(rows[i])]
                if percent_numeric(col_vals) >= 0.55:
                    num_cols += 1
            score += min(1.2, 0.45 * max(0, num_cols - 1))

            vals = extract_cash_like_values(rows, heading)
            got_any = any(
                vals.get(k) is not None
                for k in (
                    "cash_and_cash_equivalents_usd",
                    "marketable_securities_usd",
                    "short_term_investments_usd",
                    "us_treasuries_usd",
                    "government_securities_usd",
                    "restricted_cash_usd",
                    "combined_cash_cash_eq_restricted_usd",
                    "stablecoins_usd",
                )
            )
            if got_any:
                score += 3.0

            cand = {
                "ok": True,
                "heading": heading[:160],
                "rows_used": min(len(rows), 60),
                "cols": w,
                "score": round(score, 3),
                **vals,
            }

            if got_any and score > best_with_values_score:
                best_with_values_score = score
                best_with_values = cand

            if score > best_overall_score:
                best_overall_score = score
                best_overall = cand

    result = best_with_values or best_overall or {"ok": False, "reason": "no_balance_sheet_detected"}

    needs_ix = (not result.get("ok")) or not any(
        result.get(k) is not None
        for k in (
            "cash_and_cash_equivalents_usd",
            "marketable_securities_usd",
            "short_term_investments_usd",
            "us_treasuries_usd",
            "government_securities_usd",
            "restricted_cash_usd",
            "combined_cash_cash_eq_restricted_usd",
        )
    )
    if needs_ix and root is not None:
        ix_vals = extract_cash_from_ixbrl_root(root)
        if ix_vals:
            result = {
                "ok": True,
                "heading": result.get("heading") or "iXBRL facts",
                "rows_used": result.get("rows_used", 0),
                "cols": result.get("cols", 0),
                "score": max(1.0, float(result.get("score", 0.0))),
                **ix_vals,
            }

    return result, diag_rows

def pull_ix_meta(root: etree._Element) -> Dict[str, Any]:
    try:
        get_one = lambda name: root.xpath(f"string(//ix:nonNumeric[@name='{name}'])", namespaces=NS)
        return {
            "doc_type": _clean(get_one("dei:DocumentType")),
            "doc_period_end_date": _clean(get_one("dei:DocumentPeriodEndDate")),
            "registrant": _clean(get_one("dei:EntityRegistrantName")),
            "cik_from_ix": _clean(get_one("dei:EntityCentralIndexKey")),
        }
    except Exception:
        return {}

# =========================
# URL candidates
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
        if n == prim:
            score += 0.8
        for k in ("10-k", "10q", "10-q", "20-f", "6-k", "balance", "financialposition", "financialcondition", "statement"):
            if k in n:
                score += 0.5
        if re.search(r"\bex[-_]?99", n):
            score -= 0.4
        return score

    df["__pre"] = df["name"].apply(pre_score)
    df = df.sort_values(["__pre", "name"], ascending=[False, True])
    urls = df["url"].tolist()
    if prelimit is not None and prelimit > 0:
        urls = urls[:prelimit]
    return urls

# =========================
# Cash collection (with optional accession whitelist)
# =========================
def collect_cash_from_balance_sheets(
    ticker: str,
    forms: str = "10-K,10-Q,20-F,6-K",
    from_date: str = "2024-01-01",
    limit_filings: int = 300,
    max_docs_per_filing: int = 12,
    log_every: int = 10,
    diagnostics: bool = False,
    year: int = 0,
    year_by: str = "accession",
    accession_whitelist: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cik = resolve_cik(ticker)
    filings = fetch_filings(cik)
    if filings.empty:
        return pd.DataFrame(), pd.DataFrame()

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
        return pd.DataFrame(), pd.DataFrame()

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

        got = False
        for uidx, u in enumerate(urls, start=1):
            log.debug(f"  Doc {uidx}/{len(urls)} -> {u}")
            try:
                resp = get(u)
                root = html_root_from_bytes(resp.content)
                bs, diag_rows = scan_doc_for_balance_sheet_and_extract(resp.content, diag=diagnostics, doc_url=u)
                diag_rows_all.extend(diag_rows)
                if not bs.get("ok"):
                    continue
                got = True
                meta = pull_ix_meta(root) if root is not None else {}
                rec = {
                    "ticker": ticker.upper(),
                    "cik": cik,
                    "form": row.get("form"),
                    "accessionNumber": accession,
                    "primaryDocument": primary,
                    "filingDate": row.get("filingDate"),
                    "reportDate": row.get("reportDate"),
                    "acceptanceDateTime": row.get("acceptanceDateTime"),
                    "source_url": u,
                    "bs_heading": bs.get("heading"),
                    "bs_score": bs.get("score"),
                    "bs_value_column_date": bs.get("value_column_date"),
                    "cash_and_cash_equivalents_usd": bs.get("cash_and_cash_equivalents_usd"),
                    "marketable_securities_usd": bs.get("marketable_securities_usd"),
                    "short_term_investments_usd": bs.get("short_term_investments_usd"),
                    "us_treasuries_usd": bs.get("us_treasuries_usd"),
                    "government_securities_usd": bs.get("government_securities_usd"),
                    "restricted_cash_usd": bs.get("restricted_cash_usd"),
                    "stablecoins_usd": bs.get("stablecoins_usd"),
                    "combined_cash_cash_eq_restricted_usd": bs.get("combined_cash_cash_eq_restricted_usd"),
                    "cash_like_total_excl_restricted_usd": bs.get("cash_like_total_excl_restricted_usd"),
                    "units_multiplier": bs.get("units_multiplier"),
                    **meta,
                }
                rows.append(rec)
            except Exception as e:
                log.warning(f"  Error parsing doc {u}: {e}")
                continue

        if not got:
            log.debug("  No balance-sheet table with cash-like rows found in sampled docs.")

    if not rows:
        return pd.DataFrame(), (pd.DataFrame(diag_rows_all) if diagnostics else pd.DataFrame())

    df = (
        pd.DataFrame(rows)
        .sort_values(["filingDate", "bs_score"], ascending=[True, False])
        .reset_index(drop=True)
    )
    diag_df = pd.DataFrame(diag_rows_all) if diagnostics else pd.DataFrame()
    return df, diag_df

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
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=recent_hours)
    cik = resolve_cik(ticker)
    filings = fetch_filings(cik)
    if filings.empty:
        return set()

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

    subset = subset.sort_values("filingDate").tail(limit_filings)
    if subset.empty:
        return set()

    whitelist: Set[str] = set()
    for _, row in subset.iterrows():
        accession = row.get("accessionNumber")
        try:
            files = list_filing_files(cik, accession)
        except Exception as e:
            log.warning(f"index.json fetch failed for {accession}: {e}")
            continue
        if files.empty or "last_modified" not in files.columns:
            continue
        if (pd.to_datetime(files["last_modified"], errors="coerce") >= cutoff).any():
            whitelist.add(accession)
    return whitelist

# =========================
# PUBLIC API (for Prefect flow)
# =========================
KEEP_COLS = [
    "ticker","form","accessionNumber","filingDate","reportDate","source_url",
    "cash_and_cash_equivalents_usd","marketable_securities_usd","short_term_investments_usd",
    "us_treasuries_usd","government_securities_usd","restricted_cash_usd","stablecoins_usd",
    "combined_cash_cash_eq_restricted_usd","cash_like_total_excl_restricted_usd","units_multiplier"
]

def get_sec_cash_daily_df(
    tickers: List[str],
    forms: str = "10-K,10-Q,20-F,6-K",
    from_date: str = "2024-01-01",
    limit_filings: int = 300,
    max_docs_per_filing: int = 12,
    diagnostics: bool = False,
    year: int = 2025,
    year_by: str = "accession",
    recent_hours: int = 24,
) -> pd.DataFrame:
    """Daily default: probe recent accessions for each ticker, then extract cash just for those."""
    frames: List[pd.DataFrame] = []
    for t in (tickers or []):
        log.info(f"=== DAILY TICKER {t} (probe last {recent_hours}h) ===")
        acc_wh = collect_recent_accessions(
            ticker=t,
            forms=forms,
            from_date=from_date,
            limit_filings=limit_filings,
            recent_hours=recent_hours,
            year=year,
            year_by=year_by,
        )
        if not acc_wh:
            log.info(f"[{t}] No accessions touched in last {recent_hours}h.")
            continue

        df_t, _ = collect_cash_from_balance_sheets(
            ticker=t,
            forms=forms,
            from_date=from_date,
            limit_filings=limit_filings,
            max_docs_per_filing=max_docs_per_filing,
            log_every=10,
            diagnostics=diagnostics,
            year=year,
            year_by=year_by,
            accession_whitelist=acc_wh,
        )
        if not df_t.empty:
            frames.append(df_t)

    if not frames:
        return pd.DataFrame(columns=KEEP_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in KEEP_COLS if c in out.columns]]

def get_sec_cash_full_df(
    tickers: List[str],
    forms: str = "10-K,10-Q,20-F,6-K",
    from_date: str = "2024-01-01",
    limit_filings: int = 300,
    max_docs_per_filing: int = 12,
    diagnostics: bool = False,
    year: int = 2025,
    year_by: str = "accession",
) -> pd.DataFrame:
    """Full scan (no probe)."""
    frames: List[pd.DataFrame] = []
    for t in (tickers or []):
        df_t, _ = collect_cash_from_balance_sheets(
            ticker=t,
            forms=forms,
            from_date=from_date,
            limit_filings=limit_filings,
            max_docs_per_filing=max_docs_per_filing,
            log_every=10,
            diagnostics=diagnostics,
            year=year,
            year_by=year_by,
        )
        if not df_t.empty:
            frames.append(df_t)
    if not frames:
        return pd.DataFrame(columns=KEEP_COLS)
    out = pd.concat(frames, ignore_index=True)
    return out[[c for c in KEEP_COLS if c in out.columns]]

# =========================
# Minimal CLI (no files)
# =========================
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Daily SEC cash extractor (no file writes).")
    ap.add_argument("--tickers", default="MSTR,CEP,SMLR,NAKA,BMNR,SBET,ETHZ,BTCS,SQNS,BTBT,DFDV,UPXI")
    ap.add_argument("--forms", default="10-K,10-Q,20-F,6-K")
    ap.add_argument("--from-date", default="2024-01-01")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--year-by", choices=["accession","filingdate"], default="accession")
    ap.add_argument("--max-docs", type=int, default=12)
    ap.add_argument("--recent-hours", type=int, default=24)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--diagnostics", action="store_true")
    return ap.parse_args()

def main():
    args = _parse_args()
    global log
    log = setup_logger(args.log_level)

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if args.recent_hours and args.recent_hours > 0:
        df = get_sec_cash_daily_df(
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
        log.info(f"[DAILY] produced rows={len(df)}")
    else:
        df = get_sec_cash_full_df(
            tickers=tickers,
            forms=args.forms,
            from_date=args.from_date,
            limit_filings=args.limit,
            max_docs_per_filing=args.max_docs,
            diagnostics=args.diagnostics,
            year=args.year,
            year_by=args.year_by,
        )
        log.info(f"[FULL] produced rows={len(df)}")

    if not df.empty:
        log.info("\n" + df.head(min(5, len(df))).to_string(index=False))

if __name__ == "__main__":
    main()
