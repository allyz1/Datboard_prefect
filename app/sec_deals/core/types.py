# sec_deals/core/types.py
from __future__ import annotations
from typing import TypedDict, Optional

class DealHit(TypedDict, total=False):
    # Provenance
    ticker: str
    filingDate: str
    form: str
    accessionNumber: str
    source_url: str
    exhibit_hint: str
    event_type_final: str
    snippet: str
    score: float

    # As-written text fields
    price_per_share_text: str
    gross_proceeds_text: str
    shares_issued_text: str
    warrant_coverage_text: str
    exercise_price_text: str
    convert_price_text: str
    ownership_blocker_text: str
    reg_file_deadline_text: str
    reg_effect_deadline_text: str
    closing_date_text: str
    security_types_text: str
    agents: str

    # Parsed numerics
    price_per_share_usd: float
    gross_proceeds_usd: float
    shares_issued: float
    warrant_coverage_pct: float
    exercise_price_usd: float
    convert_price_usd: float
    ownership_blocker_pct: float
    reg_file_deadline_days: int
    reg_effect_deadline_days: int
