# sec_deals/modes/outstanding.py
from __future__ import annotations
import re
import html as _html
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

# Setup logger
logger = logging.getLogger(__name__)

from ..core.parse_common import (
    SHARES_SCALED_RX,
    parse_shares_from_phrase,
    parse_scaled_number_from_phrase,
)

DealHit = Dict[str, Any]

# --- Normalization helper for aggressive iXBRL + odd spaces ---
_WS_NO_BREAK = re.compile(r"[\u00A0\u2000-\u200B\u202F]")  # nbsp, thin, hair, narrow, etc.

def _norm_html(s: str) -> str:
    """
    Normalize aggressive iXBRL markup and Unicode spaces.
    1) decode entities (&nbsp; &rsquo; etc)
    2) strip ALL tags (ix:*, span, etc.)
    3) normalize odd Unicode spaces to regular
    4) unify apostrophes (' -> ')
    5) collapse spaces
    """
    # 1) decode entities (&nbsp; &#160; &rsquo; → characters)
    t = _html.unescape(s or "")
    # 2) strip ALL tags (ix:*, div, span, etc.)
    t = re.sub(r"<[^>]+>", " ", t, flags=re.I)
    # 3) normalize odd Unicode spaces to regular
    t = _WS_NO_BREAK.sub(" ", t)
    # 4) unify apostrophes (' → ')
    t = t.replace("\u2019", "'")
    # 5) collapse spaces
    return re.sub(r"\s+", " ", t).strip()

# --- Dates ---
MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RX   = re.compile(rf"{MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}}", re.I)

# --- Fallback harvester helpers ---
_RX_INT = re.compile(r"\d{1,3}(?:,\d{3})+|\d{6,}")  # integer >= 100000 or comma-formatted
_RX_CLASS_A = re.compile(r"\bclass\s*a\b.*?\b(?:common\s+stock|ordinary\s+shares?)\b", re.I | re.S)
_RX_CLASS_B = re.compile(r"\bclass\s*b\b.*?\b(?:common\s+stock|ordinary\s+shares?)\b", re.I | re.S)
_RX_OUTSTANDING_ANY = re.compile(r"\b(?:issued\s+and\s+outstanding|outstanding)\b", re.I)

def _harvest_dual_after_date(t: str, max_window: int = 20000):
    """
    Robust fallback harvester for dual-class outstanding:
      1) locate 'As of <DATE>'
      2) look forward up to max_window chars
      3) ensure 'outstanding' appears
      4) independently find the nearest big integer to each of 'Class A' and 'Class B'
    Return {'date', 'num1', 'num2'} or None.
    """
    m_date = re.search(rf"\bas\s+of\s+({DATE_RX.pattern})\b", t, re.I)
    if not m_date:
        return None
    date_text = m_date.group(1)

    start = m_date.start()
    end = min(len(t), start + max_window)
    w = t[start:end]

    if not _RX_OUTSTANDING_ANY.search(w):
        return None

    def nearest_bigint_around(match) -> Optional[str]:
        i0, i1 = match.span()
        # Search forward from the class phrase for a big integer
        mnum_fwd = _RX_INT.search(w[i1:min(len(w), i1 + 2000)])
        if mnum_fwd:
            return mnum_fwd.group(0)
        # Fallback: search backward if forward fails
        mnum_back = _RX_INT.search(w[max(0, i0 - 1500):i0])
        if mnum_back:
            return mnum_back.group(0)
        return None

    ma = _RX_CLASS_A.search(w)
    mb = _RX_CLASS_B.search(w)
    if not (ma and mb):
        return None

    n1_txt = nearest_bigint_around(ma)
    n2_txt = nearest_bigint_around(mb)
    if not (n1_txt and n2_txt):
        return None

    return {"date": date_text, "num1": n1_txt, "num2": n2_txt}


# Cover page outstanding shares detection (first page only)
RX_OUTSTANDING_BLOCK = re.compile(
    rf"\bOutstanding\s+(?:on|as\s+of)\s+(?P<date>{DATE_RX.pattern})"
    rf"(?:[^$0-9A-Za-z]|&nbsp;|&#160;|</?\w+>|\s){{0,220}}"
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}}){{1,5}}|\d{{6,}})\b",         # 6+ digits or comma-formatted
    re.I | re.S,
)

# Handle "As of DATE, there were NUM ... outstanding" on the COVER, with HTML in between.
RX_COVER_ASOF_THEREWERE = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    # allow lots of HTML/tags/nbsp between date and the verb/number
    rf"(?:[^$0-9A-Za-z]|&nbsp;|&#160;|&rsquo;|</?\w+>|\s){{0,500}}?"
    rf"(?:there\s+were|we\s+had)\s*"
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+|\d{{6,}})"                 # comma-formatted or 6+ digits
    rf"(?:[^A-Za-z0-9]{{0,200}})?(?:shares)?"
    rf"(?:[^A-Za-z0-9]{{0,200}})?"
    rf"(?:of\s+(?:the\s+registrant['']s\s+)?(?:class\s+[a-z]\s+)?common\s+stock)?"
    rf"(?:[^A-Za-z0-9]{{0,200}})?"
    rf"(?:issued\s+and\s+outstanding|outstanding)\b",
    re.I | re.S,
)

RX_NEARBY_MONEY_OR_WARRANTS = re.compile(
    r"(?:\$|USD|CAD|€|£)\s*\d|exercise\s+price|strike\s+price|warrant|options?",
    re.I,
)

RX_NEARBY_COMMON_STOCK = re.compile(r"\b(?:class\s+[a-z]\s+)?common\s+stock\b", re.I)

def find_cover_outstanding_on_first_page(first_page_html: str) -> Optional[Dict[str, str]]:
    """
    Look only at the first page/cover chunk. Return dict with 'num' and 'date' if found.
    Handles dual-class cover sentences by capturing and summing both Class A and B amounts.
    Robust to iXBRL markup.
    """
    # Normalize: decode entities, collapse nbsp, strip ALL HTML/iXBRL tags, collapse spaces
    t = _norm_html(first_page_html)
    
    logger.debug(f"[Cover] Normalized text length: {len(t)}")

    # Try table style, then simple "as of… there were…"
    m = RX_OUTSTANDING_BLOCK.search(t)
    if m:
        logger.debug(f"[Cover] Matched RX_OUTSTANDING_BLOCK")
    else:
        m = RX_COVER_ASOF_THEREWERE.search(t)
        if m:
            logger.debug(f"[Cover] Matched RX_COVER_ASOF_THEREWERE")

    # NEW: Try the "was ... and ... was ..." dual-class pattern
    if not m:
        m_dual_was = RX_DUAL_NUMBER_WAS.search(t)
        if m_dual_was:
            logger.debug(f"[Cover] Matched RX_DUAL_NUMBER_WAS")
            gd = m_dual_was.groupdict()
            date_text = gd.get("date")
            n1_txt, n2_txt = (gd.get("num1") or "").strip(), (gd.get("num2") or "").strip()
            n1 = parse_shares_from_phrase(n1_txt) or parse_scaled_number_from_phrase(n1_txt)
            n2 = parse_shares_from_phrase(n2_txt) or parse_scaled_number_from_phrase(n2_txt)
            if n1 and n2 and float(n1).is_integer() and float(n2).is_integer():
                num_text = f"{n1_txt} (Class A) + {n2_txt} (Class B)"
                return {"date": date_text, "num": num_text}

    # NEW: Try the "NUM shares of Class A ... and NUM shares of Class B ... were issued and outstanding" pattern
    if not m:
        m_dual_were = RX_ASOF_DUAL_NUM_WERE.search(t)
        if m_dual_were:
            logger.debug(f"[Cover] Matched RX_ASOF_DUAL_NUM_WERE")
            gd = m_dual_were.groupdict()
            date_text = gd.get("date")
            n1_txt, n2_txt = (gd.get("num1") or "").strip(), (gd.get("num2") or "").strip()
            n1 = parse_shares_from_phrase(n1_txt) or parse_scaled_number_from_phrase(n1_txt)
            n2 = parse_shares_from_phrase(n2_txt) or parse_scaled_number_from_phrase(n2_txt)
            if n1 and n2 and float(n1).is_integer() and float(n2).is_integer():
                num_text = f"{n1_txt} (Class A) + {n2_txt} (Class B)"
                return {"date": date_text, "num": num_text}

    # Try other dual-class patterns
    if not m:
        m_dc = RX_DUAL_CLASS.search(t)
        if m_dc:
            logger.debug(f"[Cover] Matched RX_DUAL_CLASS")
            m = m_dc
        else:
            m_dc_simple = RX_DUAL_CLASS_SIMPLE.search(t)
            if m_dc_simple:
                logger.debug(f"[Cover] Matched RX_DUAL_CLASS_SIMPLE")
                m = m_dc_simple
            else:
                m_dc_reversed = RX_DUAL_CLASS_REVERSED.search(t)
                if m_dc_reversed:
                    logger.debug(f"[Cover] Matched RX_DUAL_CLASS_REVERSED")
                    m = m_dc_reversed
                else:
                    logger.debug(f"[Cover] No dual-class patterns matched")
        
        if m:
            gd = m.groupdict()
            date_text = gd.get("date")
            n1_txt, n2_txt = (gd.get("num1") or "").strip(), (gd.get("num2") or "").strip()
            logger.debug(f"[Cover] Dual-class match: date={date_text}, num1={n1_txt}, num2={n2_txt}")
            n1 = parse_shares_from_phrase(n1_txt) or parse_scaled_number_from_phrase(n1_txt)
            n2 = parse_shares_from_phrase(n2_txt) or parse_scaled_number_from_phrase(n2_txt)
            if n1 and n2 and float(n1).is_integer() and float(n2).is_integer():
                num_text = f"{n1_txt} (Class A) + {n2_txt} (Class B)"
                logger.debug(f"[Cover] Dual-class parsed successfully: {num_text}")
                return {"date": date_text, "num": num_text}
            else:
                logger.debug(f"[Cover] Dual-class parsing failed: n1={n1}, n2={n2}")

    # Last-resort harvest for very long cover sentences (par value clauses, etc.)
    if not m:
        h = _harvest_dual_after_date(t, max_window=20000)
        if h:
            logger.debug("[Cover] Harvest fallback matched")
            n1 = parse_shares_from_phrase(h["num1"]) or parse_scaled_number_from_phrase(h["num1"])
            n2 = parse_shares_from_phrase(h["num2"]) or parse_scaled_number_from_phrase(h["num2"])
            if n1 and n2 and float(n1).is_integer() and float(n2).is_integer():
                num_text = f'{h["num1"]} (Class A) + {h["num2"]} (Class B)'
                logger.debug(f"[Cover] Harvest success: {num_text}")
                return {"date": h["date"], "num": num_text}
            else:
                logger.debug(f"[Cover] Harvest parse failed: n1={n1}, n2={n2}")

    if not m:
        logger.debug(f"[Cover] No patterns matched")
        return None

    # --- Tight, local noise check around the captured number (±60 chars) ---
    num_start, num_end = m.start("num"), m.end("num")
    local = t[max(0, num_start - 60): min(len(t), num_end + 60)]
    has_money_noise = RX_NEARBY_MONEY_OR_WARRANTS.search(local) is not None
    is_par_value    = RX_PAR_VALUE.search(local) is not None
    
    logger.debug(f"[Cover] Noise check: has_money={has_money_noise}, is_par_value={is_par_value}")
    
    if has_money_noise and not is_par_value:
        logger.debug(f"[Cover] Rejected due to money noise without par value exception")
        return None

    date_text = m.group("date")
    num_text  = m.group("num")
    logger.debug(f"[Cover] Found: date={date_text}, num={num_text}")

    # Class B tail rescue (if the cover match was a single number)
    tail = t[m.end(): m.end() + 1800]
    m2 = RX_SECOND_CLASS_TAIL.search(tail)
    if m2:
        n2_txt = (m2.group("num2") or "").strip()
        n1 = parse_shares_from_phrase(num_text) or parse_scaled_number_from_phrase(num_text)
        n2 = parse_shares_from_phrase(n2_txt) or parse_scaled_number_from_phrase(n2_txt)
        logger.debug(f"[Cover] Class B tail found: n2_txt={n2_txt}, n1={n1}, n2={n2}")
        if n1 and n2 and float(n1).is_integer() and float(n2).is_integer():
            num_text = f"{num_text} (Class A) + {n2_txt} (Class B)"
            logger.debug(f"[Cover] Class B tail success: {num_text}")

    return {"date": date_text, "num": num_text}
SUBJ_VERB = r"(?:there\s+were|we\s+had|(?:the\s+)?(?:company|registrant|issuer)\s+had)"

def _to_iso_date(date_text: Optional[str]) -> Optional[str]:
    if not date_text:
        return None
    s = re.sub(r"\s+", " ", date_text.strip())
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            pass
    return None

# ---- Cover-only hints (first page language) ----
# Must mention the security somewhere in the same block
RX_SECURITY = re.compile(
    r"\b(?:class\s+[a-z]\s+)?(?:common\s+stock|ordinary\s+shares?)\b",
    re.I
)

# Keep legacy hints, but some filings won't include these boilerplate phrases
COVER_HINTS = re.compile(
    r"(indicate\s+the\s+number\s+of\s+shares\s+outstanding|"
    r"latest\s+practicable\s+date|"
    r"outstanding\s+on\s+" + DATE_RX.pattern + r"|"
    r"title\s+of\s+each\s+class)",
    re.I
)

# NEW: simple gate that matches the typical cover sentence structure
BASIC_GATE = re.compile(
    rf"\bas\s+of\s+{DATE_RX.pattern}\b.{{0,1200}}?\b(?:issued\s+and\s+outstanding|outstanding)\b",
    re.I | re.S,
)

# NEW: reverse ordering gate: "... outstanding ... as of <DATE>"
BASIC_GATE_REV = re.compile(
    rf"\b(?:issued\s+and\s+outstanding|outstanding)\b.{{0,1200}}?\bas\s+of\s+{DATE_RX.pattern}\b",
    re.I | re.S,
)

# NEW: "The number of shares ... as of DATE" gate (for colon-separated structures)
BASIC_GATE_NUMBER = re.compile(
    rf"\bthe\s+number\s+of\s+(?:outstanding\s+)?shares\b.*?\bas\s+of\s+{DATE_RX.pattern}\b",
    re.I | re.S,
)

# Exclude contexts that also say "outstanding" (warrants/options etc.)
RX_EXCLUDE = re.compile(
    r"\b(warrant|warrants|option|options|rsu|restricted\s+stock\s+unit|debenture|note[s]?)\b"
    r"|weighted\s+average|exercise\s+price|strike\s+price",
    re.I,
)

# Money near the captured number → likely a price, not shares
RX_MONEY_NEAR = re.compile(r"(?:USD|CAD|\$|€|£)\s*\d", re.I)

# Allow "par value $... per share" near the match (not pricing noise)
RX_PAR_VALUE = re.compile(
    r"\bpar\s+value(?:\s+of)?\s*(?:USD|CAD|\$|€|£)\s*\d+(?:\.\d+)?(?:\s*per\s+share)?",
    re.I,
)

# Capture trailing Class B shares from a dual-class cover sentence
RX_SECOND_CLASS_TAIL = re.compile(
    rf"\band\b.{{0,600}}?"
    rf"(?P<num2>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf".{{0,600}}?"
    rf"(?:of\s+(?:the\s+registrant['']s\s+)?class\s+b\b.*?(?:common\s+stock|ordinary\s+shares?)|class\s+b\b)",
    re.I | re.S,
)

# ---- Cover table style: "Outstanding on <DATE>" ... NUMBER (no word 'shares') ----
RX_OUTSTANDING_ON_TABLE = re.compile(
    rf"\bOutstanding\s+(?:on|as\s+of)\s+(?P<date>{DATE_RX.pattern})\b"
    rf"(?:[^0-9A-Za-z]|&nbsp;|&#160;|</?\w+>){0,160}?"
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+|\d+)",
    re.I | re.S,
)

# Rare reversed layout: number first then "Outstanding on <DATE>"
RX_TABLE_NUM_THEN_OUTSTANDING = re.compile(
    rf"(?P<num>\d{{1,3}}(?:,\d{{3}})+|\d+)\s*"
    rf"(?:[^A-Za-z0-9]{{0,120}})?"
    rf"\bOutstanding\s+(?:on|as\s+of)\s+(?P<date>{DATE_RX.pattern})\b",
    re.I | re.S,
)

# --- Common cover-page phrasings on 10-Q/10-K ---

# "The number of shares outstanding ... as of March 7, 2025 was 182,435,019 shares."
RX_COVER_WAS = re.compile(
    rf"number\s+of\s+shares\s+outstanding.*?\bas\s+of\b\s*(?P<date>{DATE_RX.pattern}).{{0,120}}?\b(?:was|were)\b\s*(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)",
    re.I | re.S,
)

# NEW: "As of DATE, the number of outstanding shares ... was NUM"
RX_ASOF_NUMBER_WAS = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,200}}?\bthe\s+number\s+of\s+(?:outstanding\s+)?shares\b"
    rf".{{0,200}}?\b(?:was|were)\s+(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\b",
    re.I | re.S,
)

# NEW: "The number of shares ... as of DATE: NUM shares"
RX_NUMBER_ASOF_COLON = re.compile(
    rf"\bthe\s+number\s+of\s+(?:outstanding\s+)?shares\b"
    rf".{{0,240}}?\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,40}}?[:;]?\s*"
    rf"(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?\b",
    re.I | re.S,
)

# Add near top with the other patterns:
SEC_VARIANTS = r"(?:our\s+|the\s+)?(?:registrant['']s\s+)?(?:class\s+[a-z]\s+)?(?:common\s+stock|ordinary\s+shares?)"


# “Indicate the number of shares outstanding ... : 182,435,019 as of March 7, 2025.”
RX_INDICATE = re.compile(
    rf"indicate\s+the\s+number\s+of\s+shares\s+outstanding.*?\bcommon\s+stock\b.*?\bdate\b[:\s]*"
    rf"(?P<num>\d[\d,\.]*(?:\s*(?:thousand|thousands|million|millions|billion|billions|k|m|mm|bn|b))?)"
    rf"(?:\s*(?:shares)?)?(?:\s*\b(?:as\s+of)\b\s*(?P<date>{DATE_RX.pattern}))?",
    re.I | re.S,
)
# “As of DATE, there were/ we had NUM shares ... issued and outstanding/outstanding”
RX_ASOF_THEREWERE = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b.{0,200}?\b(?:there\s+were|we\s+had)\s*"
    rf"(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?"
    rf"(?:.{0,200}?)(?:issued\s+and\s+outstanding|outstanding)\b",
    re.I | re.S,
)


# “There were/ We had NUM shares ... outstanding ... as of DATE” (reversed order)
RX_THEREWERE_ASOF = re.compile(
    rf"\b(?:there\s+were|we\s+had)\s*(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?"
    rf"(?:.{0,200}?)(?:issued\s+and\s+outstanding|outstanding)\b"
    rf".{0,200}?\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b",
    re.I | re.S,
)

# As of DATE, <subject> had NUM shares ... issued and outstanding/outstanding
RX_ASOF_HAD_NUM = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b.{0,200}?\b{SUBJ_VERB}\s*"
    rf"(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?"
    rf"(?:.{0,200}?)(?:issued\s+and\s+outstanding|outstanding)\b",
    re.I | re.S,
)

# As of DATE, <subject> had outstanding NUM shares ...   (word "outstanding" comes before the number)
RX_ASOF_HAD_OUTSTANDING_NUM = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b.{0,200}?\b{SUBJ_VERB}\s+"
    rf"(?:issued\s+and\s+outstanding|outstanding)\b.{0,120}?"
    rf"(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?",
    re.I | re.S,
)

# <subject> had NUM shares ... issued and outstanding ... as of DATE  (reversed order)
RX_HAD_NUM_ASOF = re.compile(
    rf"\b{SUBJ_VERB}\s*"
    rf"(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?"
    rf"(?:.{{0,200}})(?:issued\s+and\s+outstanding|outstanding)\b"
    rf".{{0,200}}\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b",
    re.I | re.S,
)

# NEW: "As of DATE, NUM shares ... were outstanding"
RX_ASOF_NUM_WERE = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,240}}?(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?"
    rf".{{0,240}}?\bwere\b.{{0,40}}?\boutstanding\b",
    re.I | re.S,
)

# NEW (reversed): "NUM shares ... were outstanding ... as of DATE"
RX_NUM_WERE_ASOF = re.compile(
    rf"(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?"
    rf".{{0,240}}?\bwere\b.{{0,40}}?\boutstanding\b"
    rf".{{0,240}}?\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b",
    re.I | re.S,
)

RX_ASOF_GENERIC = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,200}}?\b{SUBJ_VERB}\s*"
    rf"(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf"(?:\s+of\s+{SEC_VARIANTS})?"
    rf"(?:.{{0,240}}?)(?:issued\s+and\s+outstanding|outstanding)\b",
    re.I | re.S,
)
# Fallback: "There were NUM shares ... outstanding ... as of DATE" with arbitrary text between
RX_FALLBACK_THEREWERE = re.compile(
    rf"\b(?:there\s+were|we\s+had)\s*(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)\b"
    rf".{{0,240}}?\boutstanding\b"
    rf".{{0,240}}?\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b",
    re.I | re.S,
)

# Require we're in the right neighborhood
RX_CONTEXT_REQUIRED = re.compile(r"\boutstanding\b", re.I)

# As of DATE, <subject> had NUM shares of Class A ... and NUM shares of Class B (dual-class format)
RX_DUAL_CLASS = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,800}}?\b(?:the\s+registrant|the\s+company|company|registrant|issuer|we|our)\s+had\b"
    rf"(?:\s+(?:issued\s+and\s+outstanding|outstanding))?\s+"           # allow 'had outstanding' / 'had issued and outstanding'
    rf"(?P<num1>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf".{{0,600}}?\bof\s+(?:the\s+registrant['']s\s+|its\s+)?"
    rf"(?:class\s+[a-z].*?(?:common\s+stock|ordinary\s+shares?))"
    rf"(?:.{{0,200}}?\b(?:issued\s+and\s+outstanding|outstanding)\b)?"  # allow 'outstanding' after class A clause
    rf".{{0,600}}?\band\b.{{0,600}}?"
    rf"(?P<num2>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf".{{0,600}}?\bof\s+(?:the\s+registrant['']s\s+|its\s+)?"
    rf"(?:class\s+[a-z].*?(?:common\s+stock|ordinary\s+shares?))"
    rf"(?:.{{0,200}}?\b(?:issued\s+and\s+outstanding|outstanding)\b)?"  # allow 'outstanding' after class B clause
    rf"(?:.{{0,400}}?\b(?:issued\s+and\s+outstanding|outstanding)\b)?", # or a single trailing 'outstanding'
    re.I | re.S,
)

# Optional: Simpler backup for dual-class without repeating "common stock"
RX_DUAL_CLASS_SIMPLE = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,600}}?\b(?:the\s+registrant|company|issuer)?\s+had\s+"
    rf"(?P<num1>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\b.*?\band\b.*?"
    rf"(?P<num2>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\b.*?"
    rf"(?:issued\s+and\s+outstanding|outstanding)\b",
    re.I | re.S,
)

# Dual-class, reversed order:
# "<subject> had outstanding 6,535,014 shares of Class A Common Stock and 17,154,119 shares of Class B Common Stock outstanding as of August 14 2025"
RX_DUAL_CLASS_REVERSED = re.compile(
    rf"\b(?:the\s+registrant|the\s+company|company|registrant|issuer|we|our)\s+had\b"
    rf"(?:\s+(?:issued\s+and\s+outstanding|outstanding))?\s+"
    rf"(?P<num1>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf".{{0,600}}?\bof\s+(?:the\s+registrant['']s\s+|its\s+)?"
    rf"(?:class\s+[a-z].*?(?:common\s+stock|ordinary\s+shares?))"
    rf"(?:.{{0,200}}?\b(?:issued\s+and\s+outstanding|outstanding)\b)?"
    rf".{{0,600}}?\band\b.{{0,600}}?"
    rf"(?P<num2>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?"
    rf".{{0,600}}?\bof\s+(?:the\s+registrant['']s\s+|its\s+)?"
    rf"(?:class\s+[a-z].*?(?:common\s+stock|ordinary\s+shares?))"
    rf"(?:.{{0,200}}?\b(?:issued\s+and\s+outstanding|outstanding)\b)?"
    rf".{{0,400}}?\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b",
    re.I | re.S,
)

# Dual-class: "As of DATE, the number of shares ... Class A ... outstanding was NUM
# and the number of shares ... Class B ... outstanding was NUM"
RX_DUAL_NUMBER_WAS = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,400}}?\bthe\s+number\s+of\s+shares\b"
    rf".{{0,200}}?\bclass\s+a\b.*?\b(?:common\s+stock|ordinary\s+shares?)\b.*?\boutstanding\b.*?\bwas\b\s+"
    rf"(?P<num1>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)"
    rf".{{0,200}}?\band\b.{{0,200}}?\bthe\s+number\s+of\s+shares\b"
    rf".{{0,200}}?\bclass\s+b\b.*?\b(?:common\s+stock|ordinary\s+shares?)\b.*?\boutstanding\b.*?\bwas\b\s+"
    rf"(?P<num2>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\b",
    re.I | re.S,
)

# Dual-class: "As of DATE, NUM shares of Class A ... and NUM shares of Class B ... were issued and outstanding"
RX_ASOF_DUAL_NUM_WERE = re.compile(
    rf"\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b"
    rf".{{0,800}}?"
    rf"(?P<num1>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?\s+of\s+"
    rf"(?:our\s+|the\s+registrant['']s\s+|its\s+)?class\s+a\b.*?(?:common\s+stock|ordinary\s+shares?)"
    rf".{{0,800}}?\band\b.{{0,800}}?"
    rf"(?P<num2>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)\s*(?:shares)?\s+of\s+"
    rf"(?:our\s+|the\s+registrant['']s\s+|its\s+)?class\s+b\b.*?(?:common\s+stock|ordinary\s+shares?)"
    rf".{{0,800}}?\b(?:were|was)\b.{{0,80}}?\b(?:issued\s+and\s+outstanding|outstanding)\b",
    re.I | re.S,
)

def _first_match(text: str):
    # Put the table-style detectors FIRST so they win on cover pages like the screenshot
    for rx in (
        RX_OUTSTANDING_ON_TABLE,          # table header layout
        RX_TABLE_NUM_THEN_OUTSTANDING,    # reversed table layout
        RX_ASOF_DUAL_NUM_WERE,            # dual-class with "were issued and outstanding"
        RX_DUAL_CLASS,                    # dual-class (Class A + Class B) - check before single-number patterns
        RX_DUAL_CLASS_SIMPLE,             # simpler dual-class backup
        RX_DUAL_CLASS_REVERSED,           # dual-class, reversed order
        RX_DUAL_NUMBER_WAS,               # dual-class with explicit "was"
        # NEW simple "were outstanding" variants:
        RX_ASOF_NUM_WERE,
        RX_NUM_WERE_ASOF,
        RX_COVER_WAS,
        RX_ASOF_NUMBER_WAS,
        RX_NUMBER_ASOF_COLON,
        RX_INDICATE, RX_ASOF_THEREWERE, RX_THEREWERE_ASOF,
        RX_ASOF_HAD_NUM, RX_ASOF_HAD_OUTSTANDING_NUM, RX_HAD_NUM_ASOF,
        RX_ASOF_GENERIC, RX_FALLBACK_THEREWERE
    ):
        m = rx.search(text)
        if m:
            return m, rx
    return None, None




def _parse_number(num_text: str) -> Optional[float]:
    n = parse_shares_from_phrase(num_text)
    if n is None:
        n = parse_scaled_number_from_phrase(num_text)
    return n

def _local_noise_ok(text: str, span: tuple) -> bool:
    """Check if a number span is free of money/warrant noise (except par value context)."""
    local = text[max(0, span[0] - 60): min(len(text), span[1] + 60)]
    has_money = RX_NEARBY_MONEY_OR_WARRANTS.search(local) is not None
    
    if has_money:
        # allow any "par value …" phrasing, with or without "per share"
        if re.search(r"\bpar\s+value\b", local, re.I):
            return True
        if RX_PAR_VALUE.search(local):
            return True
        return False
    return True

def classify_block(block: str, filing_form: str) -> List[DealHit]:
    """
    Emit one DealHit if we find a cover-page style 'shares outstanding' statement.
    Fields:
      - event_type_final='shares_outstanding'
      - outstanding_shares_text / outstanding_shares
      - as_of_date_text / as_of_date_iso
      - snippet (trimmed)
    """
    # Normalize: decode entities, strip ALL HTML/iXBRL tags, collapse spaces
    t = _norm_html(block)
    
    logger.debug(f"[Block] Processing block, length={len(t)}, form={filing_form}")
    logger.debug(f"[Block] norm[:300]={t[:300]}")
    
    # Try the first-page table-style detector first
    cover_hit = find_cover_outstanding_on_first_page(t)
    if cover_hit:
        num_text = cover_hit["num"]
        date_text = cover_hit["date"]
        logger.debug(f"[Block] Cover hit found: date={date_text}, num={num_text}")
        
        # Handle dual-class: if num_text contains " + ", sum both parts
        if " + " in num_text:
            parts = re.findall(r"\d[\d,\.]*", num_text)
            nums = [parse_shares_from_phrase(p) or parse_scaled_number_from_phrase(p) for p in parts]
            num_val = sum(float(n) for n in nums if n)
            logger.debug(f"[Block] Dual-class sum: parts={parts}, total={num_val}")
        else:
            num_val = parse_shares_from_phrase(num_text) or parse_scaled_number_from_phrase(num_text)
            logger.debug(f"[Block] Single-class parse: num_val={num_val}")
        
        if num_val and num_val >= 10000 and float(num_val).is_integer():
            logger.debug(f"[Block] Cover hit valid, emitting with score=100")
            return [{
                "event_type_final": "shares_outstanding",
                "outstanding_shares_text": num_text,
                "outstanding_shares": num_val,
                "as_of_date_text": date_text,
                "as_of_date_iso": _to_iso_date(date_text),
                "snippet": f"Outstanding on {date_text}: {num_text}",
                "score": 100,   # high confidence cover match
            }]
        else:
            logger.debug(f"[Block] Cover hit rejected: num_val={num_val}")
    
    # Fall back to existing per-block patterns for non-cover-page matches

    # NEW gate: allow either legacy hints OR the common "as of ... outstanding" construct
    gate_cover = COVER_HINTS.search(t) is not None
    gate_basic = BASIC_GATE.search(t) is not None
    gate_basic_rev = BASIC_GATE_REV.search(t) is not None
    gate_basic_number = BASIC_GATE_NUMBER.search(t) is not None
    if not (gate_cover or gate_basic or gate_basic_rev or gate_basic_number):
        logger.debug(f"[Block] Gate FAIL: COVER_HINTS={gate_cover}, BASIC_GATE={gate_basic}, BASIC_GATE_REV={gate_basic_rev}, BASIC_GATE_NUMBER={gate_basic_number}")
        return []
    else:
        logger.debug(f"[Block] Gate PASS: COVER_HINTS={gate_cover}, BASIC_GATE={gate_basic}, BASIC_GATE_REV={gate_basic_rev}, BASIC_GATE_NUMBER={gate_basic_number}")

    # Security mention in the neighborhood (now supports ordinary shares too)
    if not RX_SECURITY.search(t):
        logger.debug("[Block] Security FAIL: no 'common stock'/'ordinary shares' nearby")
        return []
    else:
        logger.debug("[Block] Security PASS")

    logger.debug(f"[Block] Passed gate checks, searching for patterns")

    m, rx = _first_match(t)
    if not m:
        # Fallback: robust dual harvester scoped to the block
        h = _harvest_dual_after_date(t, max_window=20000)
        if h:
            logger.debug("[Block] Harvest fallback matched")
            n1 = _parse_number(h["num1"])
            n2 = _parse_number(h["num2"])
            if n1 and n2:
                num_val = n1 + n2
                if num_val >= 10000 and float(num_val).is_integer():
                    logger.debug(f"[Block] Harvest success: n1={n1}, n2={n2}, sum={num_val}")
                    return [{
                        "event_type_final": "shares_outstanding",
                        "outstanding_shares_text": f'{h["num1"]} (Class A) + {h["num2"]} (Class B)',
                        "outstanding_shares": num_val,
                        "as_of_date_text": h["date"],
                        "as_of_date_iso": _to_iso_date(h["date"]),
                        "snippet": t if len(t) <= 900 else (t[:880] + " …"),
                        "score": 100,
                    }]
            logger.debug(f"[Block] Harvest parse failed or too small: n1={n1}, n2={n2}")
        logger.debug(f"[Block] No pattern matched and harvest failed")
        return []

    logger.debug(f"[Block] Matched pattern: {rx.pattern[:100]}...")

    gd = m.groupdict()
    
    # Log the window around the matched number
    span = None
    try:
        # prefer 'num' span if exists
        if 'num' in m.groupdict() and m.group('num') is not None:
            span = m.span('num')
        elif 'num2' in m.groupdict() and m.group('num2') is not None:
            span = m.span('num2')
        elif 'num1' in m.groupdict() and m.group('num1') is not None:
            span = m.span('num1')
    except Exception:
        pass

    if span:
        left = max(0, span[0]-80); right = min(len(t), span[1]+80)
        logger.debug(f"[Block] Around-num window: {t[left:right]}")
    else:
        logger.debug("[Block] No numeric span available to window")
    
    # Handle dual-class case where num1 and num2 are present
    if rx == RX_DUAL_CLASS and gd.get("num1") and gd.get("num2"):
        num1_text = (gd.get("num1") or "").strip()
        num2_text = (gd.get("num2") or "").strip()
        num1_val = _parse_number(num1_text) if num1_text else None
        num2_val = _parse_number(num2_text) if num2_text else None
        
        if num1_val and num2_val:
            # Check noise around both numbers
            if not (_local_noise_ok(t, m.span("num1")) and _local_noise_ok(t, m.span("num2"))):
                logger.debug(f"[Block] Dual-class rejected: noise check failed")
                return []
            # Sum the two class shares
            num_val = num1_val + num2_val
            num_text = f"{num1_text} (Class A) + {num2_text} (Class B)"
            logger.debug(f"[Block] Dual-class (KIDZ): num1_text='{num1_text}' -> {num1_val}, num2_text='{num2_text}' -> {num2_val}, sum={num_val}")
        else:
            logger.debug(f"[Block] Dual-class parse failed: num1_val={num1_val}, num2_val={num2_val}")
            return []
    elif rx == RX_DUAL_CLASS_SIMPLE and gd.get("num1") and gd.get("num2"):
        num1_text = (gd.get("num1") or "").strip()
        num2_text = (gd.get("num2") or "").strip()
        num1_val = _parse_number(num1_text) if num1_text else None
        num2_val = _parse_number(num2_text) if num2_text else None
        
        if num1_val and num2_val:
            # Check noise around both numbers
            if not (_local_noise_ok(t, m.span("num1")) and _local_noise_ok(t, m.span("num2"))):
                logger.debug(f"[Block] Dual-class-simple rejected: noise check failed")
                return []
            # Sum the two class shares
            num_val = num1_val + num2_val
            num_text = f"{num1_text} (Class A) + {num2_text} (Class B)"
            logger.debug(f"[Block] Dual-class-simple: num1_text='{num1_text}' -> {num1_val}, num2_text='{num2_text}' -> {num2_val}, sum={num_val}")
        else:
            logger.debug(f"[Block] Dual-class-simple parse failed: num1_val={num1_val}, num2_val={num2_val}")
            return []
    elif rx == RX_DUAL_CLASS_REVERSED and gd.get("num1") and gd.get("num2"):
        num1_text = (gd.get("num1") or "").strip()
        num2_text = (gd.get("num2") or "").strip()
        num1_val = _parse_number(num1_text) if num1_text else None
        num2_val = _parse_number(num2_text) if num2_text else None
        
        if num1_val and num2_val:
            # Check noise around both numbers
            if not (_local_noise_ok(t, m.span("num1")) and _local_noise_ok(t, m.span("num2"))):
                logger.debug(f"[Block] Dual-class-reversed rejected: noise check failed")
                return []
            # Sum the two class shares
            num_val = num1_val + num2_val
            num_text = f"{num1_text} (Class A) + {num2_text} (Class B)"
            logger.debug(f"[Block] Dual-class-reversed: num1_text='{num1_text}' -> {num1_val}, num2_text='{num2_text}' -> {num2_val}, sum={num_val}")
        else:
            logger.debug(f"[Block] Dual-class-reversed parse failed: num1_val={num1_val}, num2_val={num2_val}")
            return []
    elif rx == RX_DUAL_NUMBER_WAS and gd.get("num1") and gd.get("num2"):
        num1_text = (gd.get("num1") or "").strip()
        num2_text = (gd.get("num2") or "").strip()
        num1_val = _parse_number(num1_text) if num1_text else None
        num2_val = _parse_number(num2_text) if num2_text else None
        
        if num1_val and num2_val:
            # Check noise around both numbers
            if not (_local_noise_ok(t, m.span("num1")) and _local_noise_ok(t, m.span("num2"))):
                logger.debug(f"[Block] Dual-number-was rejected: noise check failed")
                return []
            # Sum the two class shares
            num_val = num1_val + num2_val
            num_text = f"{num1_text} (Class A) + {num2_text} (Class B)"
            logger.debug(f"[Block] Dual-number-was: num1_text='{num1_text}' -> {num1_val}, num2_text='{num2_text}' -> {num2_val}, sum={num_val}")
        else:
            logger.debug(f"[Block] Dual-number-was parse failed: num1_val={num1_val}, num2_val={num2_val}")
            return []
    elif rx == RX_ASOF_DUAL_NUM_WERE and gd.get("num1") and gd.get("num2"):
        num1_text = (gd.get("num1") or "").strip()
        num2_text = (gd.get("num2") or "").strip()
        num1_val = _parse_number(num1_text) if num1_text else None
        num2_val = _parse_number(num2_text) if num2_text else None
        
        if num1_val and num2_val:
            # Check noise around both numbers
            if not (_local_noise_ok(t, m.span("num1")) and _local_noise_ok(t, m.span("num2"))):
                logger.debug(f"[Block] Dual-number-were rejected: noise check failed")
                return []
            # Sum the two class shares
            num_val = num1_val + num2_val
            num_text = f"{num1_text} (Class A) + {num2_text} (Class B)"
            logger.debug(f"[Block] Dual-number-were: num1_text='{num1_text}' -> {num1_val}, num2_text='{num2_text}' -> {num2_val}, sum={num_val}")
        else:
            logger.debug(f"[Block] Dual-number-were parse failed: num1_val={num1_val}, num2_val={num2_val}")
            return []
    else:
        num_text  = (gd.get("num") or "").strip()
        num_val = _parse_number(num_text) if num_text else None
    
    date_text = (gd.get("date") or "").strip()

    # Fallback: if "Indicate …" didn't include the explicit "as of" after num, grab first date in block
    if not date_text:
        mdt = DATE_RX.search(t)
        if mdt:
            date_text = mdt.group(0)
    
    # Sanity check: ensure num_val is valid
    if num_val is None:
        logger.debug("[Block] Reject: num_val=None (parse failed)")
        return []
    
    # reject tiny values (e.g., 63 from "$9.63") and decimals
    if num_val < 10000:
        logger.debug(f"[Block] Reject: num_val too small ({num_val})")
        return []

    if isinstance(num_val, float) and not float(num_val).is_integer():
        logger.debug(f"[Block] Reject: num_val not integer ({num_val})")
        return []

    # LOCAL noise check near the captured number (NOT global to the block)
    # Figure out where the number came from
    num_span = None
    if rx in (RX_DUAL_CLASS, RX_DUAL_CLASS_SIMPLE, RX_DUAL_CLASS_REVERSED, RX_DUAL_NUMBER_WAS, RX_ASOF_DUAL_NUM_WERE):
        # for dual-class, check the second number's window (more likely to be near "and … Class B …")
        num_span = m.span("num2") if m.span("num2") else m.span("num1")
    else:
        num_span = m.span("num")

    local = t[max(0, num_span[0] - 60): min(len(t), num_span[1] + 60)]
    has_exclude = RX_EXCLUDE.search(local) is not None
    has_par = RX_PAR_VALUE.search(local) is not None
    
    logger.debug(f"[Block] Local exclusion check: has_exclude={has_exclude}, has_par={has_par}")
    
    if has_exclude and not has_par:
        logger.debug(f"[Block] Rejected: local exclusion triggered. Window: {local}")
        return []

    as_of_iso = _to_iso_date(date_text) if date_text else None
    snippet = t if len(t) <= 900 else (t[:880] + " …")
    
    logger.debug(f"[Block] Final validation passed, emitting result: shares={num_val}")

    return [{
        "event_type_final": "shares_outstanding",
        "outstanding_shares_text": num_text,
        "outstanding_shares": num_val,
        "as_of_date_text": date_text or None,
        "as_of_date_iso": as_of_iso,
        "snippet": snippet,
        "score": 10,
    }]
