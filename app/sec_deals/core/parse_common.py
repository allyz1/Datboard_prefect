# sec_deals/core/parse_common.py
"""
Common regexes & safe parsers for money / shares / % phrases
shared across PIPE, registered-direct, and warrants classifiers.
"""

from __future__ import annotations
import re
from typing import Optional

# -------------------------------------------------------------------
# Regex primitives
# -------------------------------------------------------------------
SCALE_WORDS = r"(?:thousand|thousands|million|millions|billion|billions|trillion|trillions|k|m|mm|bn|b|t|tn)"

MONEY_SCALED_RX = re.compile(rf"\$?\s*[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?", re.I)
SHARES_SCALED_RX = re.compile(rf"[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?\s+shares\b", re.I)
MONEY_RX        = re.compile(rf"\$\s*[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?", re.I)
SHARES_RX       = re.compile(rf"\b[0-9][\d,]*(?:\.\d+)?(?:\s*{SCALE_WORDS})?\s+shares\b", re.I)
PCT_RX          = re.compile(r"\b\d{1,2}(?:\.\d+)?\s*%\b")

# -------------------------------------------------------------------
# Extract the first phrase as-written
# -------------------------------------------------------------------
def money_phrase(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = MONEY_SCALED_RX.search(s)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else None

def shares_phrase(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = SHARES_SCALED_RX.search(s)
    return re.sub(r"\s+", " ", m.group(0)).strip() if m else None

def pct_phrase(s: Optional[str]) -> Optional[str]:
    if not s: return None
    m = PCT_RX.search(s)
    return m.group(0) if m else None

# -------------------------------------------------------------------
# Numeric parsers
# -------------------------------------------------------------------
def parse_scaled_number_from_phrase(token: Optional[str]) -> Optional[float]:
    """
    '$10 million' -> 10_000_000.0
    '5k' -> 5000.0
    Returns None if no number found.
    """
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

def parse_shares_from_phrase(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    core = token.replace("shares", "").strip()
    return parse_scaled_number_from_phrase(core)

def parse_pct_from_phrase(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    m = re.search(r"(\d{1,2}(?:\.\d+)?)\s*%", token)
    return float(m.group(1)) if m else None

def parse_days_from_phrase(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    m = re.search(r"(\d{1,3})\s+days?", token, re.I)
    return int(m.group(1)) if m else None
