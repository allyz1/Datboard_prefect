# sec_deals/modes/warrants.py
from __future__ import annotations
import re
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta  # pip install python-dateutil

from ..core.types import DealHit
from ..core.parse_common import (
    MONEY_RX, SHARES_RX, PCT_RX, MONEY_SCALED_RX, SHARES_SCALED_RX,
    money_phrase, shares_phrase, pct_phrase,
    parse_scaled_number_from_phrase, parse_shares_from_phrase,
    parse_pct_from_phrase, parse_days_from_phrase,
)

# ----------------------------
# Regex: cues, terms, economics
# ----------------------------
RX_WARRANT = re.compile(r"\b(pre[-\s]?funded|prefunded)?\s*warrants?\b", re.I)

MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
DATE_RX   = re.compile(rf"{MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}}", re.I)

EXPIRE_SENT_RX = re.compile(
    rf"(?:expire|expires|expiration(?:\s+date)?|term)\s*(?:on|is|:)?\s*(?P<date>{DATE_RX.pattern})"
    r"|expire[s]?\s*(?:in|after)\s*(?P<num>\d{1,2})\s*(?P<unit>year|years|month|months)",
    re.I,
)
TERM_RX = re.compile(
    r"(?:for\s+a\s+term\s+of\s+)?(?:(?P<words>\w+)\s*\((?P<num>\d+)\)\s*|(?P<num_only>\d+))\s*(?P<u>years?|months?)",
    re.I,
)
ANNIV_RX = re.compile(
    r"(?:on|until)\s+the\s+(?:\d+(?:st|nd|rd|th)\s+)?anniversary\s+of\s+the\s+(?:original\s+issue|issuance)\s+date",
    re.I,
)
NY_TIME_RX = re.compile(r"\b\d{1,2}:\d{2}\s*p\.?m\.?\s+New\s+York\s+City\s+time\b", re.I)

ISSUANCE_DATE_RX = re.compile(
    rf"(?:dated|on|as of)\s*(?P<date>{DATE_RX.pattern}).{{0,80}}?(?:issuance\s+date|original\s+issue\s+date)",
    re.I,
)
ISSUANCE_DATE_RX2 = re.compile(
    rf"(?:original\s+issue\s+date|issuance\s+date)\s+(?:is|was|shall\s+be)\s*(?P<date>{DATE_RX.pattern})",
    re.I,
)
RIGHTS_NOISE_RX = re.compile(
    r"\b(registration\s+rights?|preemptive\s+rights?|rights\s+agreement)\b", re.I
)
POS_ISSUANCE_RX = re.compile(
    r"\b(we\s+(?:are\s+)?issuing|we\s+issued|we\s+agree(?:d)?\s+to\s+issue|we\s+will\s+issue|"
    r"investors?\s+(?:were\s+)?issued|securities\s+purchase\s+agreement|placement\s+agency\s+agreement|"
    r"underwriting\s+agreement|warrant\s+agreement)\b",
    re.I
)
# Instrument proximity (equity types the warrant relates to)
_INSTR_RX = re.compile(
    r"\b(common\s+stock|class\s+[a-z]\s+common\s+stock|ordinary\s+shares?|class\s+[a-z]\s+ordinary\s+shares?|"
    r"american\s+depositary\s+shares?|ads?|adr|units?)\b", re.I)

# Explicitly exclude preferred-only warrant language
RX_PREF_ONLY_WARRANT = re.compile(
    r"warrants?.{0,80}?(?:to\s+purchase|exercisable\s+for)?\s*(?:series\s+[a-z0-9\-]+\s+)?preferred\s+stock",
    re.I,
)

# Accept $x, $x.x, $x.xx, and tiny decimals like $0.0001 (3–6 dp), and optional thousands/commas
_MONEY_FLEX_RX = r"\$(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d{1,6})?"
PRICE_PER_SH_RX = re.compile(
    rf"(?:exercise\s+price(?:\s+per\s+share)?\s*(?:of|:)?\s*{_MONEY_FLEX_RX}"
    rf"|\b{_MONEY_FLEX_RX}\s+exercise\s+price\b)",
    re.I,
)

# Exercise price (formula / non-$)
EX_PRICE_FORMULA_RX = re.compile(
    r"(initial\s+)?exercise\s+price\s+(?:equal\s+to|of)\s+.+?(?:\.|;|,|\n)", re.I,
)
EXCLUDES_RX = re.compile(
    r"\b(excludes?|does\s+not\s+include|does\s+not\s+reflect|not\s+include(?:d)?)\b.{0,40}?\b(warrants?)\b",
    re.I,
)

# Gross proceeds (sometimes for PFW tranches)
GROSS_PROCEEDS_RX = re.compile(
    rf"(?:aggregate\s+)?gross\s+proceeds(?:\s+(?:of|from))?\s*{MONEY_SCALED_RX.pattern}",
    re.I,
)

# Coverage %
WARRANT_COVERAGE_RX = re.compile(
    r"(?:warrants?.{0,40}?(\d{1,3}(?:\.\d+)?)\s*%|warrant\s+coverage.{0,20}?(\d{1,3}(?:\.\d+)?)\s*%)",
    re.I,
)
# Ratio coverage (e.g. "0.5 warrant per share", "two warrants for each share")
WARRANT_COVERAGE_RX2 = re.compile(
    r"(?:(?P<num>\d+(?:\.\d+)?)\s*warrants?\s+per\s+share)|"
    r"(?:(?:one[-\s]?half|1/2)\s+warrant\s+per\s+share)|"
    r"(?:(?:two)\s+warrants?\s+for\s+each\s+share)",
    re.I,
)

# Shares tied to warrants (underlying / labels / prefunded)
WARRANT_SHARES_CTX_RX = re.compile(
    rf"(?:pre[-\s]?funded|prefunded)?\s*warrants?\s+(?:to\s+purchase|exercisable\s+for)\s+(?P<shares>{SHARES_SCALED_RX.pattern})",
    re.I,
)
RX_PFW_COUNT_LEADING = re.compile(
    rf"(?P<count>{SHARES_SCALED_RX.pattern})\s+(?:pre[-\s]?funded|prefunded)\s+warrants?\b",
    re.I,
)
RX_WARRANT_SHARES_UNDERLYING = re.compile(
    rf"(?:underlying|issuable\s+upon\s+exercise\s+of|upon\s+exercise\s+of\s+the\s+warrants?)\s+(?P<shares>{SHARES_SCALED_RX.pattern})",
    re.I,
)
RX_WARRANT_SHARES_LABEL = re.compile(
    rf"(?P<shares>{SHARES_SCALED_RX.pattern}).{{0,30}}?\b(warrant\s+shares|pre[-\s]?funded\s+warrant\s+shares)\b",
    re.I,
)

# Expiration variants
RX_EXPIRE_UNTIL = re.compile(rf"exercis\w*.*?\buntil\b\s*(?P<date>{DATE_RX.pattern})", re.I)
RX_EXPIRE_ON_OR_PRIOR = re.compile(rf"\bon\s+or\s+prior\s+to\b.*?(?P<date>{DATE_RX.pattern})", re.I)

# Negatives / noise
EXEC_COMP_NEG_RX = re.compile(
    r"\b(compensation committee|base salary|salary|bonus|severance|offer letter|employment agreement|"
    r"grant of (?:options|rsus)|restricted stock units?|rsus?|equity award|ltip|director compensation)\b",
    re.I,
)
_BAD_MONEY_CONTEXT_RX = re.compile(
    r"\b(par\s+value|stated\s+value|public\s+float|aggregate\s+market\s+value|"
    r"highest\s+closing\s+sale\s+price|General\s+Instruction\s+I\.B\.6)\b",
    re.I,
)
_BAD_SHARES_CONTEXT_RX = re.compile(
    r"\b(outstanding|issued\s+and\s+outstanding|held\s+by\s+non[- ]affiliates?|public\s+float|as\s+of\s+[A-Za-z]+\s+\d{1,2},\s+\d{4})\b",
    re.I,
)

# Extra noise filters (tables/fees/tax narratives)
EXCLUDE_TABLE_RX = re.compile(
    r"\b(excludes?|does\s+not\s+include|doesn[’']t\s+include|not\s+included?)\b.{0,80}?\bshares?\b",
    re.I,
)
FEE_TABLE_RX = re.compile(r"\bRule\s+45[67]\([bg]\)|Rule\s+456\(b\)\b", re.I)  # 457(g), 456(b)
TAX_SECTION_RX = re.compile(r"\bSection\s+305\b|constructive\s+distribution|non[-\s]?U\.?S\.?\s+holder", re.I)

_GOOD_WARRANT_CUES_RX = re.compile(
    r"\b(warrant\s+(?:agreement|instrument|certificate)|securities\s+purchase\s+agreement|purchase\s+agreement|"
    r"we\s+issued\s+warrants|we\s+are\s+issuing\s+warrants|pre[-\s]?funded\s+warrants?)\b",
    re.I,
)

# Blocker % (capture all mentions, pick max)
BLOCKER_ALL_RX = re.compile(
    r"beneficial\s+ownership\s+(?:limitation|limitations|cap|caps|blocker|blocking)\b.{0,80}?(\d{1,2}(?:\.\d+)?)\s*%",
    re.I
)


# Security types
SECURITY_TYPES_RX = re.compile(
    r"\b(?:common\s+stock|ordinary\s+shares?|american\s+depositary\s+shares?|ads?|adr|"
    r"pre[- ]?funded\s+warrants?|warrants?|convertible\s+preferred|"
    r"convertible\s+(?:notes?|debentures?))\b",
    re.I,
)

# Token for distance scoring
RX_WARRANT_TOKEN = re.compile(r"\b(pre[-\s]?funded|prefunded)?\s*warrants?\b", re.I)

# Warrant share counts (explicit)
WARRANT_TO_PURCHASE_RX = re.compile(
    rf"(?:pre[-\s]?funded|prefunded)?\s*warrants?\s+(?:to\s+purchase|exercisable\s+for)\s+"
    rf"(?P<shares>{SHARES_SCALED_RX.pattern})\s+(?:shares?\s+of\s+)?(?:common\s+stock|ordinary\s+shares?)",
    re.I,
)

# Common stock + pre-funded combos
AND_PREFUNDED_RX = re.compile(
    rf"{SHARES_SCALED_RX.pattern}\s+shares?\s+of\s+(?:common\s+stock|ordinary\s+shares?).{{0,300}}?"
    rf"(?:and|,)\s+(?:pre[-\s]?funded|prefunded)\s+warrants?\s+(?:to\s+purchase|exercisable\s+for)?\s*(?:up\s+to\s+|up\s+to\s+an\s+aggregate\s+of\s+)?"
    rf"(?P<shares>{SHARES_SCALED_RX.pattern})",
    re.I | re.S,
)
EXCHANGE_PAIR_RX = re.compile(
    rf"exchang(?:e|ed|ing).{{0,400}}?\bfor\b.{{0,300}}?"
    rf"(?P<common>{SHARES_SCALED_RX.pattern})\s+shares?\s+of\s+(?:common\s+stock|ordinary\s+shares?).{{0,400}}?"
    rf"(?:and|,)\s+(?P<prefw>{SHARES_SCALED_RX.pattern})\s+(?:pre[-\s]?funded|prefunded)\s+warrants?",
    re.I | re.S,
)

# Warrant instrument counts (not underlying)
WARRANT_COUNT_RX = re.compile(
    rf"(?P<count>{SHARES_SCALED_RX.pattern})\s+(?:[a-z\-]+\s+)*warrants?\b(?!\s+to\s+purchase)",
    re.I,
)


# Common stock / pre-funded paired captures
RX_CS_SHARES = re.compile(
    rf"(?P<shares>{SHARES_SCALED_RX.pattern})\s+shares\s+of\s+(?:our\s+)?common\s+stock\b",
    re.I,
)
RX_CS_PFW_BOTH = re.compile(
    rf"(?P<cs>{SHARES_SCALED_RX.pattern})\s+shares\s+of\s+(?:our\s+)?common\s+stock\b.*?"
    rf"(?:and|,)\s*(?P<pfw>{SHARES_SCALED_RX.pattern})\s+(?:pre[-\s]?funded|prefunded)\s+warrants?\b",
    re.I | re.S,
)
RX_PFW_CS_BOTH = re.compile(
    rf"(?P<pfw>{SHARES_SCALED_RX.pattern})\s+(?:pre[-\s]?funded|prefunded)\s+warrants?\b.*?"
    rf"(?:and|,)\s*(?P<cs>{SHARES_SCALED_RX.pattern})\s+shares\s+of\s+(?:our\s+)?common\s+stock\b",
    re.I | re.S,
)

EXCLUDE_NOISE_RX = re.compile(
    r"\b(?:but\s+)?excludes?\b.*?\b(?:outstanding|issued)\b.*?\bwarrants?\b", re.I | re.S
)

# Prefunded cue (single source of truth)
RX_PREFUNDED = re.compile(r"\b(pre[-\s]?funded|prefunded)\b", re.I)

# Agent vs investor roles
WARRANT_ROLE_RX = re.compile(
    r"\b(placement\s+agent|representative(?:’s|\'s)?|underwriter|strategic\s+advisor)\b",
    re.I,
)
OFFERING_CTX_RX = re.compile(
    r"\b(registered\s+direct|private\s+placement|pipe|takedown|public\s+offering|best\s+efforts|"
    r"securities\s+purchase\s+agreement|spa)\b",
    re.I,
)
AGENT_PCT_RX = re.compile(r"\b(?:placement\s+agent|representative|underwriter)\b.*?(\d{1,2}(?:\.\d+)?)\s*%", re.I)
# --- NEW/REVISED warrant share phrasings ---

# A. Classic, but allow "up to" / "an aggregate of (up to)" between verb and number
WARRANT_SHARES_CTX_RX = re.compile(
    rf"(?:pre[-\s]?funded|prefunded)?\s*warrants?\s+"
    rf"(?:to\s+purchase|exercisable\s+for)\s+"
    rf"(?:up\s+to\s+(?:an\s+aggregate\s+of\s+)?|an\s+aggregate\s+of\s+)?"
    rf"(?P<shares>{SHARES_SCALED_RX.pattern})",
    re.I,
)

# B. “Warrants relating to 10,435,430 shares of Common Stock …”
RX_WARRANT_RELATING_TO_SHARES = re.compile(
    rf"warrants?\s+(?:relating\s+to|for)\s+"
    rf"(?P<shares>{SHARES_SCALED_RX.pattern})\s+shares?\s+of\s+(?:common\s+stock|ordinary\s+shares?)",
    re.I,
)

# C. Number-first: “an aggregate of 10,435,430 shares … issuable upon exercise of the Warrants”
RX_SHARES_THEN_ISSUABLE = re.compile(
    rf"(?P<shares>{SHARES_SCALED_RX.pattern})\s+shares?\s+of\s+(?:common\s+stock|ordinary\s+shares?).{{0,80}}?"
    rf"(?:issuable\s+upon\s+exercise\s+of|upon\s+exercise\s+of)\s+(?:the\s+)?warrants?",
    re.I | re.S,
)

# D. General fallback near “warrants”: number-of-shares appears within ~120 chars after “warrants”
RX_WARRANT_NEAR_SHARES = re.compile(
    rf"warrants?.{{0,120}}?(?P<shares>{SHARES_SCALED_RX.pattern})\s+shares?\s+of\s+(?:common\s+stock|ordinary\s+shares?)",
    re.I | re.S,
)


# ----------------------------
# Helpers
# ----------------------------
def _window(text: str, i: int, j: int, pad: int = 80) -> str:
    return text[max(0, i - pad): min(len(text), j + pad)]

def _parse_cs_and_pfw_counts(text: str) -> tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    m = RX_CS_PFW_BOTH.search(text) or RX_PFW_CS_BOTH.search(text)
    if not m:
        return None, None, None, None
    cs_txt = m.group("cs")
    pfw_txt = m.group("pfw")
    from_phrase = lambda s: parse_shares_from_phrase(s) if s else None
    return cs_txt, from_phrase(cs_txt), pfw_txt, from_phrase(pfw_txt)

def _nearest_distance(anchor_spans: List[Tuple[int,int]], target_span: Tuple[int,int]) -> int:
    if not anchor_spans:
        return 10**9
    t_mid = (target_span[0] + target_span[1]) // 2
    best = 10**9
    for a0, a1 in anchor_spans:
        a_mid = (a0 + a1) // 2
        d = abs(a_mid - t_mid)
        if d < best:
            best = d
    return best

def _money_phrase_wr(text: str) -> Optional[str]:
    m = PRICE_PER_SH_RX.search(text)
    if not m:
        return None
    i, j = m.span()
    window = text[max(0, i-80): min(len(text), j+80)]
    if _BAD_MONEY_CONTEXT_RX.search(window):
        return None
    # allow $0.0001 if explicitly “pre-funded warrant”
    if re.search(r"\$0\.0{2,}\d+", m.group(0)) and not re.search(r"pre[-\s]?funded\s+warrant", window, re.I):
        return None
    return re.sub(r"\s+", " ", m.group(0)).strip()

def _shares_phrase_wr(text: str) -> Optional[str]:
    for rx in (WARRANT_SHARES_CTX_RX, RX_WARRANT_SHARES_UNDERLYING, RX_WARRANT_SHARES_LABEL):
        m = rx.search(text)
        if m:
            i, j = m.span()
            window = text[max(0, i-120): min(len(text), j+120)]
            if _BAD_SHARES_CONTEXT_RX.search(window):
                continue
            grp = m.groupdict().get("shares") or m.group(0)
            return re.sub(r"\s+", " ", grp).strip()
    return None

def _to_num_shares(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    n = parse_shares_from_phrase(s)
    if n is None:
        n = parse_scaled_number_from_phrase(s)  # fallback for “535,805 pre-funded warrants”
    return n

def _ratio_to_pct(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    m = WARRANT_COVERAGE_RX2.search(text)
    if not m:
        return None
    if m.group("num"):
        try:
            return float(m.group("num")) * 100.0
        except Exception:
            return None
    if re.search(r"\bone[-\s]?half\b|1/2", text, re.I):
        return 50.0
    if re.search(r"\btwo\b\s+warrants?\s+for\s+each\s+share", text, re.I):
        return 200.0
    return None

def _extract_warrant_share_counts(text: str) -> dict:
    hits = []

    # Structured patterns first (prefunded combos, exchanges, leading counts)
    m_and = AND_PREFUNDED_RX.search(text)
    if m_and and m_and.groupdict().get("shares"):
        i, j = m_and.span()
        shares_str = m_and.group("shares")
        hits.append((i, j, re.sub(r"\s+", " ", shares_str).strip(), True))

    m_ex = EXCHANGE_PAIR_RX.search(text)
    if m_ex:
        i, j = m_ex.span()
        pfw_str = m_ex.group("pfw")
        if pfw_str:
            hits.append((i, j, re.sub(r"\s+", " ", pfw_str).strip(), True))

    for m in RX_PFW_COUNT_LEADING.finditer(text):
        i, j = m.span()
        pfw_str = m.group("count")
        hits.append((i, j, re.sub(r"\s+", " ", pfw_str).strip(), True))

    # Core warrant-linked share phrases (now with “up to / aggregate of”)
    for rx in (
        WARRANT_SHARES_CTX_RX,                # A
        RX_WARRANT_SHARES_UNDERLYING,         # original “underlying/upon exercise of … <num>”
        RX_WARRANT_SHARES_LABEL,              # “… <num> warrant shares”
        RX_WARRANT_RELATING_TO_SHARES,        # B
        RX_SHARES_THEN_ISSUABLE,              # C
        RX_WARRANT_NEAR_SHARES,               # D (fallback)
    ):
        for m in rx.finditer(text):
            shares_str = m.groupdict().get("shares") or m.group(0)
            i, j = m.span()
            win = _window(text, i, j, pad=120)
            is_pfw = RX_PREFUNDED.search(win) is not None
            shares_clean = re.sub(r"\s+", " ", shares_str).strip()
            # Skip obvious cap table/outstanding contexts
            if _BAD_SHARES_CONTEXT_RX.search(win):
                continue
            hits.append((i, j, shares_clean, is_pfw))

    pfw_text = pfw_num = std_text = std_num = None
    for _, _, phrase, is_pfw in sorted(hits, key=lambda x: x[0]):
        if is_pfw and pfw_text is None:
            pfw_text = phrase
            pfw_num  = _to_num_shares(pfw_text)
        if (not is_pfw) and std_text is None:
            std_text = phrase
            std_num  = _to_num_shares(std_text)
        if pfw_text is not None and std_text is not None:
            break

    warrant_type = None
    if pfw_text and std_text:
        warrant_type = "mixed"
    elif pfw_text:
        warrant_type = "prefunded"
    elif std_text:
        warrant_type = "standard"

    return {
        "pfw_text": pfw_text,
        "pfw_num": pfw_num,
        "std_text": std_text,
        "std_num": std_num,
        "warrant_type": warrant_type,
    }


def _expiry_from_text_enhanced(text: str) -> tuple[Optional[str], Optional[float], Optional[str]]:
    exp_date_text: Optional[str] = None
    term_years: Optional[float] = None
    issuance_text: Optional[str] = None

    for rx in (RX_EXPIRE_UNTIL, RX_EXPIRE_ON_OR_PRIOR, EXPIRE_SENT_RX):
        m = rx.search(text)
        if m and m.groupdict().get("date"):
            exp_date_text = m.group("date")
            break

    if exp_date_text is None:
        m_term = TERM_RX.search(text)
        if m_term:
            n = m_term.group("num") or m_term.group("num_only")
            u = (m_term.group("u") or "").lower()
            if n:
                try:
                    n_f = float(n)
                    term_years = (n_f / 12.0) if u.startswith("month") else n_f
                except Exception:
                    term_years = None

    mi = ISSUANCE_DATE_RX.search(text) or ISSUANCE_DATE_RX2.search(text)
    if mi:
        issuance_text = mi.group("date")

    return exp_date_text, term_years, issuance_text

def _compute_expiration_date(issuance_text: Optional[str], term_years: Optional[float]) -> Optional[str]:
    if not issuance_text or not term_years:
        return None
    try:
        dt = datetime.strptime(issuance_text, "%B %d, %Y")
        months = int(round(term_years * 12))
        dt_exp = dt + relativedelta(months=months)
        return dt_exp.strftime("%Y-%m-%d")
    except Exception:
        return None

def _score_block(t: str) -> int:
    s = 0
    if RX_WARRANT.search(t): s += 2
    if MONEY_RX.search(t):    s += 2
    if SHARES_RX.search(t):   s += 2
    if PCT_RX.search(t):      s += 1
    if DATE_RX.search(t):     s += 1
    return s

_SENT_SPLIT_RX = re.compile(r"(?:[\.;\:]\s*|\n+)")

def _instrument_near_warrant(text: str, window_chars: int = 300) -> bool:
    for s in _SENT_SPLIT_RX.split(text):
        if RX_WARRANT.search(s) and _INSTR_RX.search(s):
            return True
    for m in RX_WARRANT.finditer(text):
        i, j = m.span()
        window = text[max(0, i - window_chars): min(len(text), j + window_chars)]
        if _INSTR_RX.search(window):
            return True
    return False

def _parse_blockers(text: str) -> tuple[Optional[str], Optional[float]]:
    vals = []
    for m in BLOCKER_ALL_RX.finditer(text):
        try:
            vals.append(float(m.group(1)))
        except Exception:
            pass
    if not vals:
        return None, None
    vals.sort()
    if len(vals) == 1:
        return f"{vals[0]}%", vals[-1]
    return f"{vals[0]}% (to {vals[-1]}%)", vals[-1]

def _first(rx: re.Pattern, s: str) -> Optional[str]:
    m = rx.search(s); return m.group(0) if m else None

# ----------------------------
# Main classifier
# ----------------------------
def classify_block(block: str, filing_form: str) -> List[DealHit]:
    """
    Warrants classifier:
      - Requires a warrant cue + (exercise price OR warrant-underlying shares OR an expiration term/date).
      - Ensures warrant is tied to equity instruments (common/ordinary/ADS/units) via proximity gate.
      - Parses exercise price (cash + formula), underlying shares (prefunded vs standard),
        coverage %, blocker %, expiration (date/term with computed ISO), optional gross proceeds,
        warrant instrument counts, warrant role (agent vs investor), and agent fee % if present.
      - Works across 8-K, S-1, S-3, 424B5, 424B3, 424B7.
    """
    t = block
    f = (filing_form or "").upper()
    hits: List[DealHit] = []

    # Obvious negatives
    if EXEC_COMP_NEG_RX.search(t) or EXCLUDE_NOISE_RX.search(t):
        return []
    if EXCLUDES_RX.search(t):
        return []
    if RIGHTS_NOISE_RX.search(t) and not re.search(r"\bwarrants?\s+(?:to\s+purchase|exercisable|exercise\s+price)", t, re.I):
        return []


    # Must mention a warrant and be near an equity instrument
    if not RX_WARRANT.search(t):
        return []
    if not _instrument_near_warrant(t):
        return []

    # ----------------------------
    # Extract economics / terms
    # ----------------------------
    exercise_text  = _money_phrase_wr(t)
    exercise_formula_text = None
    if not exercise_text:
        m_exf = EX_PRICE_FORMULA_RX.search(t)
        if m_exf:
            exercise_formula_text = re.sub(r"\s+", " ", m_exf.group(0)).strip()

    # Prefunded vs standard underlying share counts
    shares_bucket  = _extract_warrant_share_counts(t)
    pfw_text       = shares_bucket["pfw_text"]
    pfw_num        = shares_bucket["pfw_num"]
    std_text       = shares_bucket["std_text"]
    std_num        = shares_bucket["std_num"]
    warrant_type   = shares_bucket["warrant_type"]

    # Expiration / term / issuance (compute BEFORE admission gate)
    exp_date_text, term_years, issuance_date_text = _expiry_from_text_enhanced(t)

    # ----------------------------
    # Admission gate
    # ----------------------------
    has_concrete_shares = bool(pfw_text or std_text)
    has_concrete_price  = bool(exercise_text or exercise_formula_text)
    has_term_or_date    = bool(exp_date_text or term_years)

    # Either explicit share count OR (price/term) + positive issuance cue
    if not (has_concrete_shares or ((has_concrete_price or has_term_or_date) and POS_ISSUANCE_RX.search(t))):
        return []

    # Back-compat single-field (legacy)
    if pfw_text and not std_text:
        shares_text = pfw_text
    elif std_text and not pfw_text:
        shares_text = std_text
    else:
        shares_text = None


    # Coverage (pct or ratio)
    coverage_text = pct_phrase(t) if WARRANT_COVERAGE_RX.search(t) else None
    if not coverage_text:
        m_ratio = WARRANT_COVERAGE_RX2.search(t)
        if m_ratio:
            coverage_text = re.sub(r"\s+", " ", m_ratio.group(0)).strip()

    # Blockers (min/max phrasing)
    blocker_text, blocker_pct_max = _parse_blockers(t)

    gross_text = money_phrase(_first(GROSS_PROCEEDS_RX, t))

    # Expiration / term / issuance
    exp_date_text, term_years, issuance_date_text = _expiry_from_text_enhanced(t)
    computed_exp_date_iso = None
    if not exp_date_text and issuance_date_text and term_years:
        computed_exp_date_iso = _compute_expiration_date(issuance_date_text, term_years)

    # Warrant instrument counts (not underlying)
    warrant_instruments_text = None
    warrant_instruments = None
    m_wcnt = WARRANT_COUNT_RX.search(t)
    if m_wcnt:
        warrant_instruments_text = re.sub(r"\s+", " ", m_wcnt.group("count")).strip()
        warrant_instruments = parse_scaled_number_from_phrase(warrant_instruments_text)

    # Agent vs investor role
    warrant_role = None
    if WARRANT_ROLE_RX.search(t):
        warrant_role = "agent"
    elif OFFERING_CTX_RX.search(t) or RX_PREFUNDED.search(t):
        warrant_role = "investor"

    # Agent fee %
    agent_fee_text = None
    agent_fee_pct = None
    m_fee = AGENT_PCT_RX.search(t)
    if m_fee:
        agent_fee_text = re.sub(r"\s+", " ", m_fee.group(0)).strip()
        try:
            agent_fee_pct = float(m_fee.group(1))
        except Exception:
            agent_fee_pct = None

    # ----------------------------
    # Noise/boilerplate filters
    # ----------------------------
    # Drop “the table above excludes ... shares ...” unless we also have real economics
    if EXCLUDE_TABLE_RX.search(t) and not (exercise_text or pfw_text or std_text or exp_date_text or term_years):
        return []
    # Drop fee table rows unless we got a real exercise price
    if FEE_TABLE_RX.search(t) and not exercise_text:
        return []
    # Drop tax sections unless they also carry concrete warrant economics
    if TAX_SECTION_RX.search(t) and not (RX_WARRANT_TOKEN.search(t) and (pfw_text or std_text or exercise_text)):
        return []

    # Require at least one concrete warrant signal
    if not any([exercise_text, exercise_formula_text, pfw_text, std_text, exp_date_text, term_years, coverage_text]):
        return []

    # Security types mentioned (normalized)
    sec_types = sorted(set(re.findall(SECURITY_TYPES_RX, t)))
    sec_types_text = " | ".join(sorted({s.lower() for s in sec_types})) if sec_types else None

    # Prefunded vs standard flags (explicit boolean hints for downstream)
    warrant_prefunded_flag = True if pfw_text or (pfw_num is not None) else (True if RX_PREFUNDED.search(t) else None)
    warrant_standard_flag  = True if std_text or (std_num is not None) else None

    # ----------------------------
    # Build hit
    # ----------------------------
    h: DealHit = {
        "event_type_final": "warrant_terms",

        # Exercise
        "exercise_price_text":          exercise_text,
        "exercise_price_text_formula":  exercise_formula_text,

        # Legacy single-field
        "shares_issued_text":           shares_text,

        # Explicit buckets
        "warrant_shares_prefunded_text":   pfw_text,
        "warrant_shares_outstanding_text": std_text,

        # Instrument counts
        "warrant_instruments_text":     warrant_instruments_text,

        # Other terms
        "warrant_coverage_text":        coverage_text,
        "ownership_blocker_text":       blocker_text,
        "gross_proceeds_text":          gross_text,
        "expiration_date_text":         exp_date_text,
        "expiration_date_iso":          computed_exp_date_iso,
        "warrant_term_years":           term_years,
        "issuance_date_text":           issuance_date_text,
        "security_types_text":          sec_types_text,

        # Warrant typing / flags
        "warrant_type":                 warrant_type,          # "prefunded" | "standard" | "mixed" | None
        "warrant_prefunded_flag":       warrant_prefunded_flag, # True | False(None if unknown)
        "warrant_standard_flag":        warrant_standard_flag,  # True | None

        # Role & fees
        "h_warrant_role":               warrant_role,          # "agent" | "investor" | None
        "agent_fee_text":               agent_fee_text,
        "agent_fee_pct":                agent_fee_pct,

        "snippet": t if len(t) <= 900 else (t[:880] + " …"),
    }

    # Parsed numerics
    h["exercise_price_usd"]             = parse_scaled_number_from_phrase(exercise_text)
    h["shares_issued"]                  = parse_shares_from_phrase(shares_text) if shares_text else None
    h["warrant_shares_prefunded"]       = pfw_num
    h["warrant_shares_outstanding"]     = std_num
    h["warrant_instruments"]            = warrant_instruments
    h["warrant_prefunded_flag"] = (h.get("warrant_type") == "prefunded") or (h.get("warrant_type") == "mixed")
    h["warrant_standard_flag"]  = (h.get("warrant_type") == "standard")  or (h.get("warrant_type") == "mixed")


    h["warrant_coverage_pct"]           = parse_pct_from_phrase(coverage_text)
    if h["warrant_coverage_pct"] is None:
        h["warrant_coverage_pct"]       = _ratio_to_pct(coverage_text)

    h["ownership_blocker_pct"]          = blocker_pct_max

    # Score & return (tune for noise/economics)
    base_score = _score_block(t)
    if _GOOD_WARRANT_CUES_RX.search(t):
        base_score += 1
    if EXCLUDE_TABLE_RX.search(t):
        base_score -= 2
    if FEE_TABLE_RX.search(t) or TAX_SECTION_RX.search(t):
        base_score -= 1
    if exercise_text or pfw_text or std_text or exp_date_text or term_years:
        base_score += 1

    h["score"] = max(0, base_score)

    hits.append(h)
    return hits
