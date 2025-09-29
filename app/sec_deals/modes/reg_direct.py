# sec_deals/modes/reg_direct.py
from __future__ import annotations
import re
from typing import List, Optional
from ..core.types import DealHit
from ..core.parse_common import (
    MONEY_RX, SHARES_RX, PCT_RX, MONEY_SCALED_RX, SHARES_SCALED_RX,
    money_phrase, shares_phrase, pct_phrase,
    parse_scaled_number_from_phrase, parse_shares_from_phrase,
    parse_pct_from_phrase, parse_days_from_phrase,
)

# ----------------------------
# Cues
# ----------------------------
RX_REGISTERED_DIRECT = re.compile(r"\bregistered\s+direct\b", re.I)
RX_ATM = re.compile(r"\bat[-\s]?the[-\s]?market\b|\bATM\b", re.I)

PRICE_PER_SH_RX = re.compile(
    rf"(?:purchase|offering|sale|subscription)?\s*price(?:\s+per\s+share)?\s*(?:of|:)?\s*\$\s*[0-9][\d,]*(?:\.\d+)?"
    rf"|\b{MONEY_SCALED_RX.pattern}\s+per\s+share\b",
    re.I,
)
GROSS_PROCEEDS_RX = re.compile(
    rf"(?:aggregate\s+)?gross\s+proceeds(?:\s+(?:of|from))?\s*{MONEY_SCALED_RX.pattern}",
    re.I,
)
EXERCISE_PRICE_RX = re.compile(
    rf"(?:exercise\s+price(?:\s+per\s+share)?\s*(?:of|:)?\s*{MONEY_SCALED_RX.pattern}"
    rf"|\b{MONEY_SCALED_RX.pattern}\s+exercise\s+price\b)",
    re.I,
)
CONVERT_PRICE_RX = re.compile(
    rf"(?:(?:initial\s+)?conversion\s+price(?:\s+per\s+share)?\s*(?:of|:)?\s*{MONEY_SCALED_RX.pattern}"
    rf"|\b{MONEY_SCALED_RX.pattern}\s+conversion\s+price\b)",
    re.I,
)
WARRANT_COVERAGE_RX = re.compile(
    r"(?:warrants?.{0,40}?(\d{1,3}(?:\.\d+)?)\s*%|warrant\s+coverage.{0,20}?(\d{1,3}(?:\.\d+)?)\s*%)",
    re.I,
)
BLOCKER_RX = re.compile(r"beneficial\s+ownership\s+(?:limitation|cap|blocker)\s+of\s+(\d{1,2}(?:\.\d+)?)\s*%", re.I)

SECURITY_TYPES_RX = re.compile(
    r"\b(?:common\s+stock|ordinary\s+shares?|american\s+depositary\s+shares?|ads|"
    r"pre[- ]?funded\s+warrants?|warrants?|convertible\s+preferred|"
    r"convertible\s+(?:notes?|debentures?))\b",
    re.I,
)

# Negatives (exec comp noise)
EXEC_COMP_NEG_RX = re.compile(
    r"\b(compensation committee|base salary|salary|cash bonus|annual bonus|performance bonus|severance|offer letter|employment agreement|"
    r"grant of (?:options|rsus)|restricted stock units?|rsus?|equity award|long[- ]term incentive|ltip|director compensation|non-employee director)\b",
    re.I
)

_BAD_MONEY_CONTEXT_RX = re.compile(
    r"\b(par\s+value|stated\s+value|public\s+float|aggregate\s+market\s+value|"
    r"highest\s+closing\s+sale\s+price|General\s+Instruction\s+I\.B\.6)\b",
    re.I
)
_BAD_SHARES_CONTEXT_RX = re.compile(
    r"\b(outstanding|issued\s+and\s+outstanding|held\s+by\s+non[- ]affiliates?|public\s+float|as\s+of\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})\b",
    re.I
)
_GOOD_OFFERING_CUES_RX = re.compile(
    r"\b(we\s+(are\s+)?offering|this\s+offering|we\s+sold|we\s+have\s+sold|we\s+propose\s+to\s+sell|"
    r"registered\s+direct\s+offering|securities\s+purchase\s+agreement)\b",
    re.I
)

def _money_phrase_rd(text: str) -> Optional[str]:
    m = PRICE_PER_SH_RX.search(text)
    if not m:
        return None
    i, j = m.span()
    window = text[max(0, i-80): min(len(text), j+80)]
    # drop money matches in bad contexts
    if _BAD_MONEY_CONTEXT_RX.search(window):
        return None
    # drop micro prices unless they’re pre-funded warrant prices
    if re.search(r"\$0\.0{2,}\d+", m.group(0)) and not re.search(r"pre[-\s]?funded\s+warrant", window, re.I):
        return None
    return re.sub(r"\s+", " ", m.group(0)).strip()

def _shares_phrase_rd(text: str) -> Optional[str]:
    m = SHARES_SCALED_RX.search(text)
    if not m:
        return None
    i, j = m.span()
    window = text[max(0, i-100): min(len(text), j+100)]
    # refuse administrative/population counts unless also an offering sentence
    if _BAD_SHARES_CONTEXT_RX.search(window) and not _GOOD_OFFERING_CUES_RX.search(window):
        return None
    return re.sub(r"\s+", " ", m.group(0)).strip()

def _score_block(t: str) -> int:
    s = 0
    if MONEY_RX.search(t): s += 2
    if SHARES_RX.search(t): s += 2
    if PCT_RX.search(t): s += 1
    return s

def _first(rx: re.Pattern, s: str) -> Optional[str]:
    m = rx.search(s); return m.group(0) if m else None

def classify_block(block: str, filing_form: str) -> List[DealHit]:
    """
    Registered Direct classifier:
      - 424B5: require deal numerics; prefer anything with price/share, proceeds, shares, warrants.
      - 8-K: only if it explicitly mentions 'registered direct' and has numerics.
      - Exclude pure ATMs unless the text also says 'registered direct'.
    """
    t = block
    f = (filing_form or "").upper()
    hits: List[DealHit] = []

    if EXEC_COMP_NEG_RX.search(t):
        return []

    # Filter ATMs unless explicitly registered direct
    if RX_ATM.search(t) and not RX_REGISTERED_DIRECT.search(t):
        return []

        # --- form gate ---
    is_424b5 = f.startswith("424B5")
    is_8k    = f.startswith("8-K")

    if is_424b5:
        # Must have deal-like numerics somewhere in the paragraph
        has_num = any(rx.search(t) for rx in (
            PRICE_PER_SH_RX, GROSS_PROCEEDS_RX, SHARES_SCALED_RX, EXERCISE_PRICE_RX, CONVERT_PRICE_RX
        ))
        if not has_num:
            return []
    elif is_8k:
        # Only consider 8-K paragraphs that explicitly say "registered direct" and have numerics
        if not RX_REGISTERED_DIRECT.search(t):
            return []
        has_num = any(rx.search(t) for rx in (
            PRICE_PER_SH_RX, GROSS_PROCEEDS_RX, SHARES_SCALED_RX
        ))
        if not has_num:
            return []
    else:
        # Other forms are out-of-scope for reg-direct mode
        return []

    # --- RD/offering flavor gate (applies to both 424B5 and 8-K paths that survived above) ---
    # 424B5 docs often don't literally say "registered direct" — allow offering cue + agent as a proxy.
    if not RX_REGISTERED_DIRECT.search(t):
        if not (_GOOD_OFFERING_CUES_RX.search(t) and
                re.search(r"(placement\s+agent|H\.C\. Wainwright|Roth|Jefferies|Cantor|Maxim|Aegis|AGP)", t, re.I)):
            return []


    # Extract fields
    price_ps_text  = _money_phrase_rd(t)
    gross_text     = money_phrase(_first(GROSS_PROCEEDS_RX, t))   # keep generic
    exercise_text  = money_phrase(_first(EXERCISE_PRICE_RX, t))
    convert_text   = money_phrase(_first(CONVERT_PRICE_RX, t))
    shares_text    = _shares_phrase_rd(t)

    m_cov = WARRANT_COVERAGE_RX.search(t)
    coverage_pct_text = m_cov.group(0) if m_cov else None

    m_blk = BLOCKER_RX.search(t)
    blocker_text = m_blk.group(0) if m_blk else None

    sec_types = sorted(set(re.findall(SECURITY_TYPES_RX, t)))
    sec_types_text = " | ".join(sec_types) if sec_types else None


    # Require at least one real economic signal
    if not any([price_ps_text, gross_text, shares_text, exercise_text, convert_text, coverage_pct_text]):
        return []

    h: DealHit = {
        "event_type_final": "registered_direct_priced",
        "price_per_share_text": price_ps_text,
        "gross_proceeds_text": gross_text,
        "shares_issued_text": shares_text,
        "warrant_coverage_text": coverage_pct_text,
        "exercise_price_text": exercise_text,
        "convert_price_text": convert_text,
        "ownership_blocker_text": blocker_text,
        "security_types_text": sec_types_text,
        "snippet": t if len(t) <= 900 else (t[:880] + " …"),
    }

    # Parsed numerics
    h["price_per_share_usd"]  = parse_scaled_number_from_phrase(price_ps_text)
    h["gross_proceeds_usd"]   = parse_scaled_number_from_phrase(gross_text)
    h["exercise_price_usd"]   = parse_scaled_number_from_phrase(exercise_text)
    h["convert_price_usd"]    = parse_scaled_number_from_phrase(convert_text)
    h["shares_issued"]        = parse_shares_from_phrase(shares_text)
    h["warrant_coverage_pct"] = parse_pct_from_phrase(coverage_pct_text)

    h["score"] = _score_block(t)
    hits.append(h)
    return hits
