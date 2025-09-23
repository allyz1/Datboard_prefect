#!/usr/bin/env python3
# sec_atm_timeline_multi.py
# Build a single, combined ATM timeline across many tickers.
# - Text values saved "as-written" (e.g., "$18 million"); numeric fields parsed safely from those phrases.
# - Removed derived calculations (no remaining capacity math, no running cap).
# pip install requests pandas lxml

import argparse, json, re, time, datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterable
from unicodedata import name

import requests
import pandas as pd
from lxml import etree
from urllib.parse import urljoin
from types import SimpleNamespace

# =========================
# SEC-polite config
# =========================
USER_AGENT = "Ally Zach <ally@panteracapital.com>"
HEADERS = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}

BASE_SUBMISSIONS = "https://data.sec.gov/submissions"
TICKER_MAP_URL   = "https://www.sec.gov/files/company_tickers.json"

# =========================
# HTTP helper
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
# Ticker -> CIK
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
    j = get(f"{BASE_SUBMISSIONS}/CIK{cik}.json").json()
    recent = j.get("filings", {}).get("recent", {})
    df = pd.DataFrame(recent)
    if df.empty:
        return df
    # Normalize datetimes (UTC -> naive)
    for col in ("reportDate","filingDate","acceptanceDateTime"):
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce", utc=True)
            df[col] = s.dt.tz_localize(None)
    df["form"] = df["form"].astype(str)
    # Accession-year from "0000950170-25-047219" -> 2025
    if "accessionNumber" in df.columns:
        def year_from_acc(s: str):
            m = re.search(r"^\d{10}-(\d{2})-\d{6}$", str(s))
            return 2000 + int(m.group(1)) if m else None
        df["acc_year"] = df["accessionNumber"].apply(year_from_acc)
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
    df["base_url"] = base
    return df

def is_html_name(name: str) -> bool:
    n = str(name).lower()
    return n.endswith((".htm",".html",".xhtml"))

def _form_tokens_for_ranking(form: str) -> Iterable[str]:
    f = (form or "").upper()
    if f.startswith("8-K"):
        return ("8-k","8k","ex99","press","item")
    if f.startswith("S-1"):
        return ("s-1","s1","registration","prospectus","use of proceeds","plan of distribution")
    if f.startswith("S-3ASR") or f.startswith("S-3"):
        return ("s-3asr","s-3","s3","registration","prospectus","supplement","plan of distribution","424b5")
    if f.startswith("424B5"):
        return ("424b5","prospectus","supplement","plan of distribution")
    return ()

def candidate_html_urls(files_df: pd.DataFrame, primary_doc: str, filing_form: str, max_docs: int = 6) -> List[Tuple[str,str]]:
    if files_df.empty:
        return []
    df = files_df[files_df["name"].apply(is_html_name)].copy()
    if df.empty:
        return []
    prim = (primary_doc or "").lower()
    prefer = tuple(t.lower() for t in _form_tokens_for_ranking(filing_form))

    def ex_hint(name: str) -> str:
        n = name.lower()
        if n == prim: return "primary"
        m = re.search(r"(ex[-_]?(\d{1,2}|99|4|10)[\.\-_]?\d*)", n)
        return m.group(1) if m else ""

    def rank(name: str) -> Tuple[int,int,int,int,int]:
        n = name.lower()
        r0 = 0 if n == prim else 1
        r1 = 0 if any(tok in n for tok in prefer) else 1
        r2 = 0 if re.search(r"ex[-_]?99", n) else (1 if re.search(r"ex[-_]?1", n) else (2 if re.search(r"ex[-_]?10", n) else (3 if re.search(r"ex[-_]?4", n) else 4)))
        r3 = 0 if n.endswith((".xhtml",".htm",".html")) else 1
        # Deprioritize press releases
        r4 = 1 if "press" in n else 0
        return (r0,r1,r2,r3,r4)

    df["rank_tuple"] = df["name"].apply(rank)
    df = df.sort_values(["rank_tuple","name"])
    out: List[Tuple[str,str]] = []
    for _, r in df.head(max_docs).iterrows():
        out.append((r["url"], ex_hint(str(r["name"]))))

    return out

# =========================
# HTML -> paragraph blocks
# =========================
def clean_text(s: str) -> str:
    return " ".join((s or "").replace("\xa0"," ").split())

def extract_blocks(html_bytes: bytes) -> List[str]:
    parser = etree.HTMLParser(recover=True, huge_tree=True)
    root = etree.HTML(html_bytes, parser=parser)
    if root is None:
        return []
    blocks = []
    for node in root.xpath(".//p | .//li | .//td | .//th | .//div"):
        txt = clean_text("".join(node.itertext()))
        if not txt:
            continue
        if 40 <= len(txt) <= 1500:
            blocks.append(txt)
    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b); out.append(b)
    return out

# =========================
# Numeric gating & helpers
# (keep phrases as-written, parse safely)
# =========================
SCALE_WORDS = r"(?:thousand|thousands|million|millions|billion|billions|trillion|trillions|k|m|mm|bn|b|t|tn)"

MONEY_SCALED_RX = re.compile(rf"\$?\s*[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?", re.I)
SHARES_SCALED_RX = re.compile(rf"[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?\s+shares\b", re.I)
MONEY_RX   = re.compile(rf"\$\s*[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?", re.I)
SHARES_RX  = re.compile(rf"\b[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?\s+shares\b", re.I)
PCT_RX     = re.compile(r"\b\d{1,2}(?:\.\d+)?\s*%\b")
NUMBER_RX  = re.compile(r"\b\d{1,3}(?:,\d{3})+|\b\d+(?:\.\d+)?\b")

def has_numeric_signal(t: str) -> bool:
    return bool(MONEY_RX.search(t) or SHARES_RX.search(t) or PCT_RX.search(t) or NUMBER_RX.search(t))

def money_phrase(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = MONEY_SCALED_RX.search(s);  return re.sub(r"\s+", " ", m.group(0)).strip() if m else None

def shares_phrase(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = SHARES_SCALED_RX.search(s); return re.sub(r"\s+", " ", m.group(0)).strip() if m else None

def pct_phrase(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = PCT_RX.search(s);           return m.group(0) if m else None

def parse_scaled_number_from_phrase(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    s = token.strip().lower()
    mnum = re.search(r"[0-9][\d,]*(?:\.\d+)?", s)
    if not mnum:
        return None
    num = float(mnum.group(0).replace(",", ""))
    scale = 1.0
    if re.search(r"\b(thousand|thousands|k)\b", s):
        scale = 1e3
    elif re.search(r"\b(million|millions|m|mm)\b", s):
        scale = 1e6
    elif re.search(r"\b(billion|billions|bn|b)\b", s):
        scale = 1e9
    elif re.search(r"\b(trillion|trillions|tn|t)\b", s):
        scale = 1e12
    return num * scale

def parse_pct_from_phrase(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", token)
    return float(m.group(1)) if m else None

def parse_shares_from_phrase(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    core = token.replace("shares", "").strip()
    return parse_scaled_number_from_phrase(core)

AGENT_ALIASES = {
    "cantor fitzgerald": "Cantor Fitzgerald",
    "cantor": "Cantor Fitzgerald",
    "thinkequity llc": "ThinkEquity",
    "thinkequity": "ThinkEquity",
    "h.c. wainwright": "H.C. Wainwright",
    "hc wainwright": "H.C. Wainwright",
    "b. riley securities": "B. Riley Securities",
    "b. riley": "B. Riley Securities",
    "jefferies": "Jefferies",
    "roth capital": "Roth Capital",
    "roth": "Roth Capital",
    "piper sandler": "Piper Sandler",
    "oppenheimer": "Oppenheimer",
    "canaccord": "Canaccord",
    "stifel": "Stifel",
    "baird": "Baird",
    "cowen": "Cowen",
    "wells fargo": "Wells Fargo",
    "citigroup": "Citigroup",
    "goldman sachs": "Goldman Sachs",
    "morgan stanley": "Morgan Stanley",
    "j.p. morgan": "J.P. Morgan",
    "jp morgan": "J.P. Morgan",
    "bank of america": "Bank of America",
}

def find_agents(text: str) -> List[str]:
    found = set()
    low = text.lower()
    for alias, canon in AGENT_ALIASES.items():
        if alias in low:
            found.add(canon)
    m = re.search(r"(?i)(?:sales|distribution|placement)\s+agent[s]?(?:\s*(?:include|are|is|:))?\s*(.+?)(?:\.|;|,?\s+and\s+| as | pursuant | under )", text)
    if m:
        cand = re.split(r",| and ", m.group(1))
        for c in cand:
            c = c.strip()
            if c and len(c.split())<=6 and any(ch.isupper() for ch in c):
                found.add(c)
    return sorted(found)[:8]

# =========================
# ATM / sales regexes
# =========================
RX_ATM        = re.compile(r"\bat[-\s]?the[-\s]?market\b|\bATM\b", re.I)
RX_SALES_AGMT = re.compile(r"\b(equity\s+)?distribution\s+agreement\b|\bsales\s+agreement\b", re.I)
RX_RULE_415   = re.compile(r"\bRule\s*415\b", re.I)
RX_SUPPL      = re.compile(r"\bprospectus\s+supplement\b|\b424B5\b", re.I)
RX_SHELF      = re.compile(r"\b(Form|on)\s*S-3(ASR)?\b|\bautomatic\s+shelf\b|\bshelf\s+registration\b", re.I)

SCALED_MONEY_TOKEN = rf"\$?\s*[0-9][\d,]*(?:\.\d+)?(?:\s*(?:thousand|thousands|million|millions|billion|billions|trillion|trillions|k|m|mm|bn|b|t|tn))?"

RX_ATM_CAP      = re.compile(rf"(?:aggregate\s+(?:offering|program)\s+(?:price|amount)\s+(?:not\s+to\s+exceed|of)\s*|\bup\s+to\s*){SCALED_MONEY_TOKEN}", re.I)
RX_CAP_INCREASE = re.compile(rf"\b(increase|increasing|increased)\b[^$]{{0,80}}\bto\b[^$]{{0,10}}{SCALED_MONEY_TOKEN}", re.I)
RX_COMMISSION   = re.compile(r"(?:commission|compensation|fee)s?\s*(?:rate\s*)?(?:of|equal\s+to|at)?\s*(?:up\s+to\s*)?(\d{1,2}(?:\.\d+)?)\s*%", re.I)

RX_SOLD_DT_SH    = re.compile(rf"\bhave\s+sold\s+([0-9][\d,]*(?:\.\d+)?(?:\s*(?:{SCALE_WORDS}))?)\s+shares", re.I)
RX_SOLD_DT_GROSS = re.compile(rf"\bgross\s+(?:proceeds|sales\s+price)\s+of\s*{SCALED_MONEY_TOKEN}", re.I)
RX_NET_PROCEEDS  = re.compile(rf"\bnet\s+proceeds?\b[^$]{{0,20}}{SCALED_MONEY_TOKEN}", re.I)
RX_AVG_PRICE     = re.compile(rf"\baverage\s+(?:sale\s+)?price\b[^$]{{0,40}}{SCALED_MONEY_TOKEN}", re.I)

RX_PUBLIC_OFFERING = re.compile(r"\b(underwritten\s+)?public\s+offering\b", re.I)
RX_CARVEOUT = re.compile(rf"\bpermitted\s+to\s+enter\s+into\s+a\s+sales\s+agreement\b.*?\bup\s+to\b\s*{SCALED_MONEY_TOKEN}", re.I)

MONTHS_RX     = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
RX_AS_OF_DATE = re.compile(rf"As of\s+({MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}})", re.I)

# -------- Hard negatives & context gates --------
EXEC_COMP_NEG_RX = re.compile(
    r"\b(compensation committee|base salary|salary|cash bonus|annual bonus|performance bonus|severance|offer letter|employment agreement|"
    r"grant of (?:options|rsus)|restricted stock units?|rsus?|equity award|long[- ]term incentive|ltip|director compensation|non-employee director)\b",
    re.I
)

CRYPTO_RX = re.compile(r"\b(bitcoin|btc|ethereum|ether|eth)\b", re.I)

EQUITY_TERMS_RX = re.compile(
    r"\b(common stock|ordinary shares?|american depositary shares?|ads|class\s+[a-c]\b|preferred stock|ordinary share)\b",
    re.I
)

ATM_CONTEXT_RX = re.compile(
    r"\b(at[-\s]?the[-\s]?market|sales\s+agreement|equity\s+distribution\s+agreement|pursuant\s+to\s+the\s+sales\s+agreement)\b",
    re.I
)

# Toggle capturing non-ATM underwritten offerings (often noisy)
INCLUDE_PUBLIC_OFFERINGS = False

# =========================
# Filters & scoring
# =========================
def is_resale_only(t: str) -> bool:
    tL = t.lower()
    if ("selling stockholder" in tL or "selling shareholder" in tL or "resell from time to time" in tL) and \
       ("we will not receive any proceeds" in tL or "will not receive any proceeds" in tL):
        return True
    return False

def _score_block(t: str, tag: str) -> int:
    score = 0
    if MONEY_RX.search(t): score += 2
    if SHARES_RX.search(t): score += 2
    if PCT_RX.search(t): score += 1
    if tag in ("init","upsize","sales_update","pre_atm_lockup_carveout"): score += 1
    return score

def _first(rx: re.Pattern, s: str) -> Optional[str]:
    m = rx.search(s); return m.group(0) if m else None

# =========================
# Classifier
# =========================
def classify_block_numeric_only(t: str) -> List[Dict[str, Any]]:
    # Basic numeric gate + resale-only skip
    if not has_numeric_signal(t) or is_resale_only(t):
        return []

    # Hard negatives: executive comp
    if EXEC_COMP_NEG_RX.search(t):
        return []

    # Ignore crypto paras unless clearly in equity/ATM context
    if CRYPTO_RX.search(t) and not (EQUITY_TERMS_RX.search(t) or ATM_CONTEXT_RX.search(t) or SHARES_RX.search(t) or RX_ATM.search(t)):
        return []

    hits: List[Dict[str, Any]] = []

    # "As of" date hint
    as_of = None
    mdate = RX_AS_OF_DATE.search(t)
    if mdate:
        as_of = mdate.group(1)

    # ATM setup / cap / commission
    if RX_ATM.search(t) and (RX_SALES_AGMT.search(t) or RX_RULE_415.search(t) or RX_SUPPL.search(t) or RX_SHELF.search(t)):
        h = {
            "atm_event_type": "init",
            "program_cap_text":      money_phrase(_first(RX_ATM_CAP, t)),
            "commission_rate_text":  pct_phrase(_first(RX_COMMISSION, t)),
            "agents": " | ".join(find_agents(t)) or None,
            "as_of_doc_date_hint": as_of,
        }
        h["_score"] = _score_block(t, h["atm_event_type"])
        h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
        if h["program_cap_text"] or h["commission_rate_text"]:
            hits.append(h)

    # ATM cap upsizes
    if RX_CAP_INCREASE.search(t):
        h = {
            "atm_event_type": "upsize",
            "program_cap_text": money_phrase(_first(RX_CAP_INCREASE, t)),
            "agents": " | ".join(find_agents(t)) or None,
            "as_of_doc_date_hint": as_of,
        }
        h["_score"] = _score_block(t, h["atm_event_type"])
        h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
        if h["program_cap_text"]:
            hits.append(h)

    # ATM sold-to-date metrics (sales_update) — require ATM/equity context
    sold_sh_phrase = None
    m_sh = RX_SOLD_DT_SH.search(t)
    if m_sh:
        sold_sh_phrase = f"{m_sh.group(1)} shares"

    has_atm_ctx   = bool(RX_ATM.search(t) or ATM_CONTEXT_RX.search(t))
    has_equity_ctx= bool(SHARES_RX.search(t) or EQUITY_TERMS_RX.search(t))

    if has_atm_ctx and (m_sh or RX_SOLD_DT_GROSS.search(t) or RX_NET_PROCEEDS.search(t) or RX_AVG_PRICE.search(t)) and has_equity_ctx:
        h = {
            "atm_event_type": "sales_update",
            "sold_to_date_shares_text": sold_sh_phrase or shares_phrase(t),
            "sold_to_date_gross_text":  money_phrase(_first(RX_SOLD_DT_GROSS, t)),
            "net_proceeds_text":        money_phrase(_first(RX_NET_PROCEEDS, t)),
            "avg_price_text":           money_phrase(_first(RX_AVG_PRICE, t)),
            "as_of_doc_date_hint":      as_of,
        }
        if any(h.get(k) for k in ("sold_to_date_shares_text","sold_to_date_gross_text","net_proceeds_text","avg_price_text")):
            h["_score"] = _score_block(t, h["atm_event_type"])
            h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
            hits.append(h)

    # Underwritten public offering (optional)
    if INCLUDE_PUBLIC_OFFERINGS:
        if (RX_PUBLIC_OFFERING.search(t)) and (RX_SOLD_DT_GROSS.search(t) or PCT_RX.search(t) or SHARES_RX.search(t)) and has_equity_ctx:
            h = {
                "atm_event_type": "sales_update",
                "sold_to_date_shares_text": shares_phrase(t),
                "sold_to_date_gross_text":  money_phrase(_first(RX_SOLD_DT_GROSS, t)),
                "net_proceeds_text":        money_phrase(_first(RX_NET_PROCEEDS, t)),
                "avg_price_text":           money_phrase(_first(RX_AVG_PRICE, t)),
                "as_of_doc_date_hint":      as_of,
            }
            if any(h.get(k) for k in ("sold_to_date_shares_text","sold_to_date_gross_text","net_proceeds_text","avg_price_text")):
                h["_score"] = _score_block(t, h["atm_event_type"])
                h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
                hits.append(h)

    # Pre-ATM lockup carveout
    if RX_CARVEOUT.search(t):
        h = {
            "atm_event_type": "pre_atm_lockup_carveout",
            "program_cap_text": money_phrase(_first(RX_CARVEOUT, t)),
            "as_of_doc_date_hint": as_of,
        }
        h["_score"] = _score_block(t, h["atm_event_type"])
        h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
        if h["program_cap_text"]:
            hits.append(h)

    return hits

# =========================
# Utilities
# =========================
def form_to_doc_kind(form: str) -> str:
    f = (form or "").upper()
    if f.startswith("8-K"): return "current_report"
    if f.startswith("424B5"): return "prospectus_supplement"
    if f.startswith("S-3ASR"): return "shelf"
    if f.startswith("S-3"): return "shelf"
    if f.startswith("S-1"): return "registration"
    return "other"

def build_timeline_for_ticker(ticker: str, args) -> pd.DataFrame:
    """
    Fetch and parse SEC filings for a single ticker and return the ATM timeline
    rows as a DataFrame.  If args.since_hours > 0 we ignore the year filter and
    only keep filings whose effective datetime is within that rolling window.
    """
    try:
        cik = resolve_cik(ticker)
    except Exception as e:
        print(f"[{ticker}] CIK lookup failed: {e}")
        return pd.DataFrame()

    filings = fetch_filings(cik)
    if filings.empty:
        print(f"[{ticker}] No filings returned.")
        return pd.DataFrame()

    # -------------------------------
    # Pick a single datetime to filter on: prefer acceptanceDateTime
    # -------------------------------
    filings["effective_dt"] = filings["acceptanceDateTime"].where(
        filings["acceptanceDateTime"].notna(),
        filings["filingDate"]
    )

    # -------------------------------
    # Rolling-window vs. year filter
    # -------------------------------
    since = int(getattr(args, "since_hours", 0) or 0)
    if since > 0:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=since)
        time_mask = filings["effective_dt"].notna() & (filings["effective_dt"] >= cutoff)
        window_desc = f"ROLLING WINDOW: last {since}h since {cutoff.isoformat(timespec='seconds')}Z (UTC)"
    else:
        if args.year_by == "accession":
            time_mask = (filings["acc_year"] == args.year)
        else:
            time_mask = filings["filingDate"].dt.year.eq(args.year)
        window_desc = f"YEAR FILTER: {args.year} (year-by={args.year_by})"

    # -------------------------------
    # Filter by form types
    # -------------------------------
    bases = [f.strip().upper() for f in args.forms.split(",") if f.strip()]
    def form_matches(f: str) -> bool:
        fu = (f or "").upper()
        return any(fu == base or fu.startswith(base + "/") for base in bases)

    subset = filings[time_mask & filings["form"].apply(form_matches)].copy()

    # Sort so newest are last, limit to args.limit
    subset = subset.sort_values(
        ["effective_dt", "acceptanceDateTime", "filingDate"],
        na_position="last"
    ).tail(args.limit)

    print(f"[{ticker}] {window_desc}; candidates after filter: {len(subset)}")
    if subset.empty:
        print(f"[{ticker}] No {bases} filings in window.")
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for _, row in subset.iterrows():
        accession   = row.get("accessionNumber")
        primary     = (row.get("primaryDocument") or "")
        filing_dt   = row.get("filingDate")
        filing_form = str(row.get("form") or "").upper()
        doc_kind    = form_to_doc_kind(filing_form)

        try:
            files = list_filing_files(cik, accession)
            doc_list = candidate_html_urls(files, primary_doc=primary,
                                           filing_form=filing_form,
                                           max_docs=args.max_docs)
        except Exception as e:
            print(f"[{ticker}] Skipping {accession}: index.json error: {e}")
            continue

        per_filing_hits: List[Dict[str, Any]] = []

        for u, ex_hint in doc_list:
            try:
                html_bytes = get(u).content
                blocks = extract_blocks(html_bytes)
            except Exception as e:
                print(f"[{ticker}] Error fetching/parsing {u}: {e}")
                continue

            for b in blocks:
                for h in classify_block_numeric_only(b):
                    cap_usd  = parse_scaled_number_from_phrase(h.get("program_cap_text"))
                    comm_pct = parse_pct_from_phrase(h.get("commission_rate_text"))
                    sold_sh  = parse_shares_from_phrase(h.get("sold_to_date_shares_text"))
                    sold_g   = parse_scaled_number_from_phrase(h.get("sold_to_date_gross_text"))
                    net_usd  = parse_scaled_number_from_phrase(h.get("net_proceeds_text"))
                    avg_usd  = parse_scaled_number_from_phrase(h.get("avg_price_text"))

                    per_filing_hits.append({
                        "ticker": ticker,
                        "filingDate": filing_dt,
                        "form": filing_form,
                        "doc_kind": doc_kind,
                        "accessionNumber": accession,
                        "source_url": u,
                        "exhibit_hint": (ex_hint or "").lower(),
                        "atm_event_type": h.get("atm_event_type"),
                        "score": h.get("_score", 0),
                        "snippet": h.get("snippet"),
                        "agents": h.get("agents"),
                        "program_cap_text":         h.get("program_cap_text"),
                        "commission_rate_text":     h.get("commission_rate_text"),
                        "sold_to_date_shares_text": h.get("sold_to_date_shares_text"),
                        "sold_to_date_gross_text":  h.get("sold_to_date_gross_text"),
                        "net_proceeds_text":        h.get("net_proceeds_text"),
                        "avg_price_text":           h.get("avg_price_text"),
                        "as_of_doc_date_hint":      h.get("as_of_doc_date_hint"),
                        "program_cap_usd":          cap_usd,
                        "commission_pct":           comm_pct,
                        "sold_to_date_shares":      sold_sh,
                        "sold_to_date_gross_usd":   sold_g,
                        "net_proceeds_usd":         net_usd,
                        "avg_price_usd":            avg_usd,
                    })

        if per_filing_hits:
            df_f = pd.DataFrame(per_filing_hits).sort_values(
                ["score"], ascending=[False]
            ).head(args.max_snippets_per_filing)
            rows.extend(df_f.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame()

    raw_df = pd.DataFrame(rows).sort_values(
        ["filingDate", "accessionNumber", "score"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    tl_cols = [
        "ticker","filingDate","form","accessionNumber","doc_kind",
        "event_type_final",
        "program_cap_text","program_cap_usd",
        "sold_to_date_shares_text","sold_to_date_shares",
        "sold_to_date_gross_text","sold_to_date_gross_usd",
        "commission_pct","agents","source_url","snippet",
        "as_of_doc_date_hint",
    ]

    timeline: List[Dict[str, Any]] = []
    for _, r in raw_df.iterrows():
        et = str(r.get("atm_event_type") or "")
        event_final = (
            "atm_init" if et == "init" else
            "atm_upsize" if et == "upsize" else
            "sales_update" if et == "sales_update" else
            "pre_atm_lockup_carveout" if et == "pre_atm_lockup_carveout" else
            et
        )
        timeline.append({
            "ticker": r.get("ticker"),
            "filingDate": r.get("filingDate"),
            "form": r.get("form"),
            "accessionNumber": r.get("accessionNumber"),
            "doc_kind": r.get("doc_kind"),
            "event_type_final": event_final,
            "program_cap_text": r.get("program_cap_text"),
            "program_cap_usd": r.get("program_cap_usd"),
            "sold_to_date_shares_text": r.get("sold_to_date_shares_text"),
            "sold_to_date_shares": r.get("sold_to_date_shares"),
            "sold_to_date_gross_text": r.get("sold_to_date_gross_text"),
            "sold_to_date_gross_usd": r.get("sold_to_date_gross_usd"),
            "commission_pct": r.get("commission_pct"),
            "agents": r.get("agents"),
            "source_url": r.get("source_url"),
            "snippet": r.get("snippet"),
            "as_of_doc_date_hint": r.get("as_of_doc_date_hint"),
        })

    return pd.DataFrame(timeline, columns=tl_cols).sort_values(
        ["ticker","filingDate"]
    ).reset_index(drop=True)


    # ============== FINAL TIMELINE (no derived calcs) ==============
    # Added form and accessionNumber to the output.
    tl_cols = [
        "ticker","filingDate","form","accessionNumber","doc_kind",
        "event_type_final",
        "program_cap_text","program_cap_usd",
        "sold_to_date_shares_text","sold_to_date_shares",
        "sold_to_date_gross_text","sold_to_date_gross_usd",
        "commission_pct","agents","source_url","snippet",
        "as_of_doc_date_hint",
    ]

    timeline: List[Dict[str, Any]] = []

    for _, r in raw_df.iterrows():
        et = str(r.get("atm_event_type") or "")
        event_final = (
            "atm_init" if et == "init" else
            "atm_upsize" if et == "upsize" else
            "sales_update" if et == "sales_update" else
            "pre_atm_lockup_carveout" if et == "pre_atm_lockup_carveout" else
            et
        )

        timeline.append({
            "ticker": r.get("ticker"),
            "filingDate": r.get("filingDate"),
            "form": r.get("form"),
            "accessionNumber": r.get("accessionNumber"),
            "doc_kind": r.get("doc_kind"),
            "event_type_final": event_final,

            # Text + parsed numeric (no further math)
            "program_cap_text": r.get("program_cap_text"),
            "program_cap_usd": r.get("program_cap_usd"),
            "sold_to_date_shares_text": r.get("sold_to_date_shares_text"),
            "sold_to_date_shares": r.get("sold_to_date_shares"),
            "sold_to_date_gross_text": r.get("sold_to_date_gross_text"),
            "sold_to_date_gross_usd": r.get("sold_to_date_gross_usd"),
            "commission_pct": r.get("commission_pct"),
            "agents": r.get("agents"),

            "source_url": r.get("source_url"),
            "snippet": r.get("snippet"),
            "as_of_doc_date_hint": r.get("as_of_doc_date_hint"),
        })

    tl_df = pd.DataFrame(timeline, columns=tl_cols).sort_values(["ticker","filingDate"]).reset_index(drop=True)
    return tl_df

# =========================
# Library-friendly entrypoint (returns DataFrame)
# =========================
def collect_timelines(tickers: list[str],
                      since_hours: int = 0,
                      forms: str = "8-K,S-1,S-3,S-3ASR,424B5",
                      limit: int = 600,
                      max_docs: int = 6,
                      max_snippets_per_filing: int = 4,
                      year: int = 2025,
                      year_by: str = "accession") -> pd.DataFrame:
    """
    Build the combined ATM timeline for the given tickers and return a DataFrame.
    No files are written.
    """
    args = SimpleNamespace(
        tickers=",".join(tickers),
        tickers_file="",
        year=year,
        year_by=year_by,
        since_hours=since_hours,
        limit=limit,
        max_docs=max_docs,
        max_snippets_per_filing=max_snippets_per_filing,
        forms=forms,
        outdir="data",
        outfile=""
    )

    all_rows = []
    for t in sorted(set([t.strip().upper() for t in tickers if t.strip()])):
        print(f"\n=== {t} ===")
        tdf = build_timeline_for_ticker(t, args)
        if not tdf.empty:
            print(f"[{t}] timeline rows: {len(tdf)}")
            all_rows.append(tdf)
        else:
            print(f"[{t}] no timeline rows found.")

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

# =========================
# Driver (CLI)
# =========================
def main():
    ap = argparse.ArgumentParser(description="Combined ATM timeline for multiple tickers (text-as-written + safe numeric parsing, no derived calcs).")
    ap.add_argument("--tickers", default="MSTR,CEP,SMLR,NAKA,BMNR,SBET,ETHZ,BTCS,SQNS,BTBT,DFDV,UPXI", help="Comma-separated list of tickers (e.g., BMNR,MARA,RIOT)")
    ap.add_argument("--tickers-file", default="", help="Optional path to a newline-delimited list of tickers")

    # Year filters (kept for backwards compatibility)
    ap.add_argument("--year", type=int, default=2025, help="Year filter (default: 2025)")
    ap.add_argument("--year-by", choices=["accession","filingdate"], default="accession",
                    help="Filter by accession year or filingDate year (default: accession)")

    # Rolling window; if >0, overrides year filters
    ap.add_argument("--since-hours", type=int, default=24,
                help="If > 0, scan only filings from the past N hours (overrides --year/--year-by). Example: --since-hours 24")


    ap.add_argument("--limit", type=int, default=600, help="Max filings to scan per ticker (default: 600)")
    ap.add_argument("--max-docs", type=int, default=6, help="Max HTML docs per filing to fetch (default: 6)")
    ap.add_argument("--max-snippets-per-filing", type=int, default=4, help="Keep top-N snippets per filing (default: 4)")
    ap.add_argument("--forms", default="8-K,S-1,S-3,S-3ASR,424B5",
                    help="Base forms to include (amendments auto-included). Default: 8-K,S-1,S-3,S-3ASR,424B5")

    # Saving is optional; if --outfile omitted, nothing is written.
    ap.add_argument("--outdir", default="data", help="Output dir if you choose to save")
    ap.add_argument("--outfile", default="", help="If provided, save CSV here. If omitted, nothing is written.")
    args = ap.parse_args()

    # Build ticker set
    tickers = []
    if args.tickers:
        tickers.extend([t.strip().upper() for t in args.tickers.split(",") if t.strip()])
    if args.tickers_file:
        p = Path(args.tickers_file)
        if p.exists():
            tickers.extend([ln.strip().upper() for ln in p.read_text().splitlines() if ln.strip()])
    tickers = sorted(set(tickers))
    if not tickers:
        raise SystemExit("No tickers provided.")

    df = collect_timelines(
        tickers=tickers,
        since_hours=args.since_hours,
        forms=args.forms,
        limit=args.limit,
        max_docs=args.max_docs,
        max_snippets_per_filing=args.max_snippets_per_filing,
        year=args.year,
        year_by=args.year_by,
    )

    if df.empty:
        print("No timeline data for any ticker.")
        return

    # Only save if the user explicitly provided --outfile
    if args.outfile:
        outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.outfile, index=False)
        print(f"Saved CSV -> {Path(args.outfile).resolve()} (rows={len(df)}, columns={len(df.columns)})")
    else:
        # Just show a preview when running as a script
        print(df.head(10))
        print(f"\n(rows={len(df)}, columns={list(df.columns)})")

if __name__ == "__main__":
    main()
