# sec_deals/modes/outstanding.py
from __future__ import annotations
import re
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..core.parse_common import (
    SHARES_SCALED_RX,
    parse_shares_from_phrase,
    parse_scaled_number_from_phrase,
)

DealHit = Dict[str, Any]

# --- Dates ---
MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RX   = re.compile(rf"{MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}}", re.I)
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

# --- Common cover-page phrasings on 10-Q/10-K ---

# “The number of shares outstanding ... as of March 7, 2025 was 182,435,019 shares.”
RX_COVER_WAS = re.compile(
    rf"number\s+of\s+shares\s+outstanding.*?\bas\s+of\b\s*(?P<date>{DATE_RX.pattern}).{{0,120}}?\b(?:was|were)\b\s*(?P<num>{SHARES_SCALED_RX.pattern}|\d[\d,\.]*)",
    re.I | re.S,
)

# Add near top with the other patterns:
SEC_VARIANTS = r"(?:our\s+|the\s+)?(?:registrant[’']s\s+)?(?:class\s+[a-z]\s+)?(?:common\s+stock|ordinary\s+shares?)"


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
    rf"(?:.{0,200}?)(?:issued\s+and\s+outstanding|outstanding)\b"
    rf".{0,200}?\bas\s+of\s+(?P<date>{DATE_RX.pattern})\b",
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


# Require we’re in the right neighborhood
RX_CONTEXT_REQUIRED = re.compile(r"\boutstanding\b", re.I)
RX_SECURITY = re.compile(r"\b(common\s+stock|ordinary\s+shares?)\b", re.I)

def _first_match(text: str):
    for rx in (RX_COVER_WAS, RX_INDICATE, RX_ASOF_THEREWERE, RX_THEREWERE_ASOF,
               RX_ASOF_HAD_NUM, RX_ASOF_HAD_OUTSTANDING_NUM, RX_HAD_NUM_ASOF,
               RX_ASOF_GENERIC, RX_FALLBACK_THEREWERE):
        m = rx.search(text)
        if m:
            return m, rx
    return None, None




def _parse_number(num_text: str) -> Optional[float]:
    n = parse_shares_from_phrase(num_text)
    if n is None:
        n = parse_scaled_number_from_phrase(num_text)
    return n

def classify_block(block: str, filing_form: str) -> List[DealHit]:
    """
    Emit one DealHit if we find a cover-page style 'shares outstanding' statement.
    Fields:
      - event_type_final='shares_outstanding'
      - outstanding_shares_text / outstanding_shares
      - as_of_date_text / as_of_date_iso
      - snippet (trimmed)
    """
    t = block
    if not RX_CONTEXT_REQUIRED.search(t):
        return []

    m, _ = _first_match(t)
    if not m:
        return []

    gd = m.groupdict()
    num_text  = (gd.get("num") or "").strip()
    date_text = (gd.get("date") or "").strip()

    # Fallback: if “Indicate …” didn’t include the explicit “as of” after num, grab first date in block
    if not date_text:
        mdt = DATE_RX.search(t)
        if mdt:
            date_text = mdt.group(0)

    num_val = _parse_number(num_text) if num_text else None
    if num_val is None:
        return []

    as_of_iso = _to_iso_date(date_text) if date_text else None
    snippet = t if len(t) <= 900 else (t[:880] + " …")

    return [{
        "event_type_final": "shares_outstanding",
        "outstanding_shares_text": num_text,
        "outstanding_shares": num_val,
        "as_of_date_text": date_text or None,
        "as_of_date_iso": as_of_iso,
        "snippet": snippet,
        "score": 10,
    }]
