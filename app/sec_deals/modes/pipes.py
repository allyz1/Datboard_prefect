# sec_deals/modes/pipes.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
from ..core.types import DealHit
from ..core.parse_common import (
    MONEY_RX, SHARES_RX, PCT_RX, MONEY_SCALED_RX, SHARES_SCALED_RX,
    money_phrase, shares_phrase, pct_phrase,
    parse_scaled_number_from_phrase, parse_shares_from_phrase,
    parse_pct_from_phrase, parse_days_from_phrase,
)

# ------------------------------------------------------------
# PIPE cues & negatives
# ------------------------------------------------------------
RX_ITEM_101 = re.compile(r"\bItem\s+1\.01\b", re.I)           # Entry into a Material Definitive Agreement
RX_ITEM_302 = re.compile(r"\bItem\s+3\.02\b", re.I)           # Unregistered Sales of Equity Securities

RX_PRIVATE_PLACEMENT = re.compile(r"\bprivate\s+placement\b|\bPIPE\b|\bPIPE\s+financing\b", re.I)
RX_SPA_SUB = re.compile(r"securities\s+purchase\s+agreement|subscription\s+agreement|purchase\s+agreement", re.I)
RX_RRA = re.compile(r"registration\s+rights\s+agreement", re.I)
RX_EXEMPTIONS = re.compile(r"Section\s*4\(a\)\(2\)|Rule\s*506\(b\)|Reg(?:ulation)?\s*S\b", re.I)
RX_WARRANT = re.compile(r"pre[- ]?funded\s+warrants?|warrants?\b", re.I)
RX_CONVERT = re.compile(r"convertible\s+(?:preferred|note|debenture)", re.I)

RX_REGISTERED_DIRECT = re.compile(r"registered\s+direct\b", re.I)
RX_ATM = re.compile(r"at[-\s]?the[-\s]?market|\bATM\b", re.I)

# Fields (phrases as-written)
PRICE_PER_SH_RX = re.compile(
    rf"(?:purchase|offering|sale|subscription)?\s*price(?:\s+per\s+share)?\s*(?:of|:)?\s*{MONEY_SCALED_RX.pattern}"
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
FILE_DEADLINE_RX = re.compile(r"file\s+a\s+registration\s+statement\s+within\s+(\d{1,3})\s+days", re.I)
EFFECT_DEADLINE_RX = re.compile(r"(declared\s+effective|effectiveness)\s+(?:within|by)\s+(\d{1,3})\s+days", re.I)

MONTHS_RX = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
CLOSING_DATE_RX = re.compile(rf"(closed\s+on|closing\s+on|expected\s+to\s+close\s+on)\s+({MONTHS_RX}\s+\d{{1,2}},\s+\d{{4}})", re.I)

SECURITY_TYPES_RX = re.compile(
    r"\b(?:common\s+stock|ordinary\s+shares?|american\s+depositary\s+shares?|ads|"
    r"pre[- ]?funded\s+warrants?|warrants?|convertible\s+preferred|"
    r"convertible\s+(?:notes?|debentures?))\b",
    re.I,
)

# Negatives / scope guards
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

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def is_resale_only_language(t: str) -> bool:
    tl = t.lower()
    return ("selling stockholder" in tl or "selling shareholder" in tl) and ("resell" in tl or "resale" in tl)

def has_deal_numeric_signal(t: str) -> bool:
    if MONEY_RX.search(t): return True
    if SHARES_RX.search(t): return True
    if GROSS_PROCEEDS_RX.search(t): return True
    if EXERCISE_PRICE_RX.search(t): return True
    if CONVERT_PRICE_RX.search(t): return True
    # "$X per share" pattern even without the explicit keyword
    if re.search(rf"{MONEY_SCALED_RX.pattern}\s+per\s+share\b", t, re.I): return True
    # Allow percentages only when clearly deal-contextual
    if PCT_RX.search(t) and re.search(r"(discount|premium|warrant|ownership|coverage|beneficial|conversion|coupon|interest)", t, re.I):
        return True
    return False


def _score_block(t: str, tag: str) -> int:
    score = 0
    if MONEY_RX.search(t): score += 2
    if SHARES_RX.search(t): score += 2
    if PCT_RX.search(t): score += 1
    if tag in ("pipe_announce","pipe_close","resale_filed","resale_effective"): score += 1
    return score

def _shares_phrase_filtered(t: str) -> Optional[str]:
    m = SHARES_SCALED_RX.search(t)
    if not m:
        return None

    i, j = m.span()
    # widen window a bit; resale sentences can be long
    window = t[max(0, i-80): min(len(t), j+80)]

    # 1) Drop administrative/authorized counts
    if re.search(r"\b(authorized|reserve[d]?|available|authorized\s+but\s+unissued)\b", window, re.I):
        return None

    # 2) Accept if clearly transactional OR resale/offering context
    transactional = re.search(
        r"\b(issued|sell|sold|purchase[d]?|subscrib(?:ed|e)|to\s+be\s+issued|in\s+the\s+private\s+placement)\b",
        window, re.I
    )
    resale_context = re.search(
        r"\b(offered?|offering|registered|resale|resell|selling\s+stockholder[s]?|to\s+be\s+sold|to\s+be\s+resold)\b",
        window, re.I
    )
    warrant_or_convert_context = re.search(
        r"\b(warrant|exercise\s+price|conversion|convertible)\b",
        window, re.I
    )

    if not (transactional or resale_context or warrant_or_convert_context):
        return None

    return re.sub(r"\s+", " ", m.group(0)).strip()



# ------------------------------------------------------------
# Public API: classify a single paragraph block for PIPE mode
# ------------------------------------------------------------
def classify_block(block: str, filing_form: str) -> List[DealHit]:
    """
    PIPE mode:
      - 8-K: must have economic numerics and PIPE/private-placement cues; emits pipe_announce/pipe_close.
      - S-1/S-3: 'selling stockholder/resale' language -> resale_filed (with field extraction).
      - 424B3/424B7: 'selling stockholder/resale' language -> resale_effective (with field extraction).
      - Registered directs / ATMs excluded unless explicitly concurrent with a private placement.
    """
    t = block
    f = (filing_form or "").upper()
    hits: List[DealHit] = []

    # Hard negatives
    if EXEC_COMP_NEG_RX.search(t):
        return []
    if CRYPTO_RX.search(t) and not (EQUITY_TERMS_RX.search(t) or RX_PRIVATE_PLACEMENT.search(t)):
        return []

    # Registered directs / ATMs are out-of-scope unless clearly concurrent private placement
    if RX_REGISTERED_DIRECT.search(t) and not re.search(r"concurrent\s+private\s+placement", t, re.I):
        return []
    if RX_ATM.search(t) and not RX_PRIVATE_PLACEMENT.search(t):
        return []

    resale_lang = is_resale_only_language(t)

    # -------- 8-K: PIPE announce/close --------
    if f.startswith("8-K"):
        if not has_deal_numeric_signal(t):
            return []
        is_pipeish = bool(
            RX_PRIVATE_PLACEMENT.search(t) or RX_SPA_SUB.search(t) or RX_RRA.search(t) or
            RX_EXEMPTIONS.search(t) or RX_ITEM_302.search(t) or RX_ITEM_101.search(t)
        )
        if not is_pipeish:
            return []

        evt = "pipe_close" if re.search(r"has\s+closed|closed\s+on|closing\s+occurred", t, re.I) else "pipe_announce"

        price_ps_text  = money_phrase(_first(PRICE_PER_SH_RX, t))
        gross_text     = money_phrase(_first(GROSS_PROCEEDS_RX, t))
        exercise_text  = money_phrase(_first(EXERCISE_PRICE_RX, t))
        convert_text   = money_phrase(_first(CONVERT_PRICE_RX, t))
        shares_text    = _shares_phrase_filtered(t)
        if not shares_text:
            # fallback for benign resale language like "up to X shares"
            m_sh = SHARES_SCALED_RX.search(t)
            shares_text = re.sub(r"\s+", " ", m_sh.group(0)).strip() if m_sh else None
        m_cov = WARRANT_COVERAGE_RX.search(t)
        coverage_pct_text = m_cov.group(0) if m_cov else None

        m_blk = BLOCKER_RX.search(t)
        blocker_text = m_blk.group(0) if m_blk else None

        m_fd = FILE_DEADLINE_RX.search(t)
        file_deadline_text = m_fd.group(0) if m_fd else None

        m_ed = EFFECT_DEADLINE_RX.search(t)
        effect_deadline_text = m_ed.group(0) if m_ed else None

        m_cd = CLOSING_DATE_RX.search(t)
        closing_text = m_cd.group(2) if m_cd else None

        sec_types = sorted(set(re.findall(SECURITY_TYPES_RX, t)))
        sec_types_text = " | ".join(sec_types) if sec_types else None

        if any([price_ps_text, gross_text, shares_text, exercise_text, convert_text, coverage_pct_text]):
            h: DealHit = {
                "event_type_final": evt,
                "price_per_share_text": price_ps_text,
                "gross_proceeds_text": gross_text,
                "shares_issued_text": shares_text,
                "warrant_coverage_text": coverage_pct_text,
                "exercise_price_text": exercise_text,
                "convert_price_text": convert_text,
                "ownership_blocker_text": blocker_text,
                "reg_file_deadline_text": file_deadline_text,
                "reg_effect_deadline_text": effect_deadline_text,
                "closing_date_text": closing_text,
                "security_types_text": sec_types_text,
            }
            # Parsed numerics
            h["price_per_share_usd"]   = parse_scaled_number_from_phrase(price_ps_text)
            h["gross_proceeds_usd"]    = parse_scaled_number_from_phrase(gross_text)
            h["exercise_price_usd"]    = parse_scaled_number_from_phrase(exercise_text)
            h["convert_price_usd"]     = parse_scaled_number_from_phrase(convert_text)
            h["shares_issued"]         = parse_shares_from_phrase(shares_text)
            h["warrant_coverage_pct"]  = parse_pct_from_phrase(coverage_pct_text)
            h["ownership_blocker_pct"] = parse_pct_from_phrase(blocker_text)
            h["reg_file_deadline_days"]= parse_days_from_phrase(file_deadline_text)
            h["reg_effect_deadline_days"]= parse_days_from_phrase(effect_deadline_text)

            h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
            h["score"] = _score_block(t, evt)
            hits.append(h)

    # -------- S-1 / S-3: resale_filed --------
    if (f.startswith("S-1") or f.startswith("S-3")) and resale_lang:
        price_ps_text  = money_phrase(_first(PRICE_PER_SH_RX, t))
        gross_text     = money_phrase(_first(GROSS_PROCEEDS_RX, t))
        exercise_text  = money_phrase(_first(EXERCISE_PRICE_RX, t))
        convert_text   = money_phrase(_first(CONVERT_PRICE_RX, t))
        shares_text    = _shares_phrase_filtered(t)

        sec_types = sorted(set(re.findall(SECURITY_TYPES_RX, t)))
        sec_types_text = " | ".join(sec_types) if sec_types else None

        h: DealHit = {
            "event_type_final": "resale_filed",
            "price_per_share_text": price_ps_text,
            "gross_proceeds_text": gross_text,
            "shares_issued_text": shares_text,
            "exercise_price_text": exercise_text,
            "convert_price_text": convert_text,
            "security_types_text": sec_types_text,
        }
        # Parsed numerics (light-touch; resales often don’t include economics)
        h["price_per_share_usd"]  = parse_scaled_number_from_phrase(price_ps_text)
        h["gross_proceeds_usd"]   = parse_scaled_number_from_phrase(gross_text)
        h["exercise_price_usd"]   = parse_scaled_number_from_phrase(exercise_text)
        h["convert_price_usd"]    = parse_scaled_number_from_phrase(convert_text)
        h["shares_issued"]        = parse_shares_from_phrase(shares_text)

        h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
        h["score"] = _score_block(t, "resale_filed")
        hits.append(h)

    # -------- 424B3 / 424B7: resale_effective --------
    if (f.startswith("424B3") or f.startswith("424B7")) and resale_lang:
        price_ps_text  = money_phrase(_first(PRICE_PER_SH_RX, t))
        gross_text     = money_phrase(_first(GROSS_PROCEEDS_RX, t))
        exercise_text  = money_phrase(_first(EXERCISE_PRICE_RX, t))
        convert_text   = money_phrase(_first(CONVERT_PRICE_RX, t))
        shares_text    = _shares_phrase_filtered(t)

        sec_types = sorted(set(re.findall(SECURITY_TYPES_RX, t)))
        sec_types_text = " | ".join(sec_types) if sec_types else None

        h: DealHit = {
            "event_type_final": "resale_effective",
            "price_per_share_text": price_ps_text,
            "gross_proceeds_text": gross_text,
            "shares_issued_text": shares_text,
            "exercise_price_text": exercise_text,
            "convert_price_text": convert_text,
            "security_types_text": sec_types_text,
        }
        # Parsed numerics (light-touch)
        h["price_per_share_usd"]  = parse_scaled_number_from_phrase(price_ps_text)
        h["gross_proceeds_usd"]   = parse_scaled_number_from_phrase(gross_text)
        h["exercise_price_usd"]   = parse_scaled_number_from_phrase(exercise_text)
        h["convert_price_usd"]    = parse_scaled_number_from_phrase(convert_text)
        h["shares_issued"]        = parse_shares_from_phrase(shares_text)

        h["snippet"] = t if len(t) <= 900 else (t[:880] + " …")
        h["score"] = _score_block(t, "resale_effective")
        hits.append(h)

    return hits


# small helper
def _first(rx: re.Pattern, s: str) -> Optional[str]:
    m = rx.search(s)
    return m.group(0) if m else None
