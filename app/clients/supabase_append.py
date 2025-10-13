# app/clients/supabase_client.py
import os, math
from functools import lru_cache
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence, List, Dict, Optional

import numpy as np
import pandas as pd
from supabase import create_client, Client

# NOTE: load_dotenv is *not* called here.
# In local dev, call `from app import config  # loads .env` once at program start.
# In Prefect/production, set real env vars on the agent machine.

# ---------- bootstrap ----------
@lru_cache(maxsize=1)
def get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    # Prefer service role for writes; fall back to anon if you only read
    key = (os.environ.get("SUPABASE_SERVICE_KEY")
           or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")   # keep backward compat
           or os.environ["SUPABASE_ANON_KEY"])
    return create_client(url, key)

# ---------- helpers ----------
def _json_safe(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v);  return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp, datetime, date)):
        return pd.to_datetime(v, errors="coerce").isoformat()
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    return str(v)

def _normalize_row(row: dict, *, drop_key: bool = True) -> dict:
    if drop_key and "key" in row:
        row = {k: v for k, v in row.items() if k != "key"}
    return {k: _json_safe(v) for k, v in row.items()}


def _require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

def _prep_table_df(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["date", "ticker", "asset", "total_holdings"])
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype("string")
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["asset"] = out["asset"].astype(str).str.upper()
    out["total_holdings"] = pd.to_numeric(out["total_holdings"], errors="coerce")
    out = out.dropna(subset=["date", "ticker"])
    out = out.sort_values(["ticker", "date"])
    return out[["date", "ticker", "asset", "total_holdings"]]

# ---- precheck helpers ----
def _compute_keys_local(df: pd.DataFrame) -> pd.Series:
    d8 = df["date"].str.replace("-", "", regex=False)
    return d8 + "-" + df["ticker"]  # must match your DB's GENERATED key

def _fetch_existing_keys(sb: Client, table: str, keys: List[str], chunk: int = 300) -> set:
    existing = set()
    for i in range(0, len(keys), chunk):
        batch = keys[i:i + chunk]
        resp = sb.table(table).select("key").in_("key", batch).execute()
        for row in (resp.data or []):
            k = row.get("key")
            if k is not None:
                existing.add(k)
    return existing

# ---------- row ops ----------
def insert_if_absent_by_key(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    sb = get_supabase()
    payload = _normalize_row(row)
    resp = (
        sb.table(table)
          .upsert(payload, on_conflict="key", ignore_duplicates=True, returning="representation")
          .execute()
    )
    return {"inserted": bool(resp.data), "data": resp.data or []}

def upsert_by_key(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    sb = get_supabase()
    payload = _normalize_row(row)
    resp = sb.table(table).upsert(payload, on_conflict="key").execute()
    return {"data": resp.data, "count": len(resp.data) if resp.data else 0}

def append_row(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    sb = get_supabase()
    payload = _normalize_row(row)
    resp = sb.table(table).insert(payload).execute()
    return {"data": resp.data, "count": len(resp.data) if resp.data else 0}

# ---------- batch / DataFrame ops ----------
def upload_df_by_key(
    table: str,
    df: pd.DataFrame,
    chunk_size: int = 1000,
    do_update: bool = False,
    precheck: bool = False,
) -> Dict[str, int]:
    if df is None or df.empty:
        return {"attempted": 0, "skipped_existing": 0, "sent": 0}

    sb = get_supabase()
    df2 = _prep_table_df(df)
    df2 = df2.drop_duplicates(subset=["date", "ticker"], keep="last")
    attempted = len(df2)
    skipped_existing = 0

    if precheck:
        keys_local = _compute_keys_local(df2).tolist()
        existing = _fetch_existing_keys(sb, table, keys_local)
        mask_missing = ~pd.Series(keys_local).isin(existing).values
        df2 = df2.loc[mask_missing]
        skipped_existing = attempted - len(df2)

    if df2.empty:
        return {"attempted": attempted, "skipped_existing": skipped_existing, "sent": 0}

    payload = df2.where(pd.notnull(df2), None).to_dict(orient="records")

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).upsert(
            batch,
            on_conflict="key",
            ignore_duplicates=(not do_update),
            returning="minimal",
        ).execute()
        sent += len(batch)

    return {"attempted": attempted, "skipped_existing": skipped_existing, "sent": sent}

def concat_and_upload(
    table: str,
    frames: List[pd.DataFrame],
    chunk_size: int = 1000,
    do_update: bool = False
) -> Dict[str, int]:
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
    if not frames:
        return {"attempted": 0, "skipped_existing": 0, "sent": 0}

    df = pd.concat(frames, ignore_index=True, sort=False, copy=False)

    wanted = ["date", "ticker", "asset", "total_holdings"]
    for col in wanted:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[wanted]
    df = df.dropna(how="all")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["asset"] = df["asset"].astype(str).str.upper()
    df["total_holdings"] = pd.to_numeric(df["total_holdings"], errors="coerce")
    df = df.dropna(subset=["date", "ticker"])
    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")

    return upload_df_by_key(
        table, df, chunk_size=chunk_size, do_update=do_update, precheck=True
    )
# ===== ATM_raw upload helpers (ADD THIS SECTION) =====

# Columns coming from your ATM timeline df:
ATM_RAW_COLS = [
    "ticker","filingDate","form","accessionNumber","doc_kind","event_type_final",
    "program_cap_text","program_cap_usd",
    "sold_to_date_shares_text","sold_to_date_shares",
    "sold_to_date_gross_text","sold_to_date_gross_usd",
    "commission_pct","agents","source_url","snippet","as_of_doc_date_hint",
]

def prep_atm_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ATM timeline DataFrame for insertion into ATM_raw.
    Does NOT require a 'key' column (primary key is DB-generated).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=ATM_RAW_COLS)

    out = df.copy()

    # Ensure all expected columns exist (fill missing with NA)
    for c in ATM_RAW_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[ATM_RAW_COLS]

    # Types / cleaning
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["form"] = out["form"].astype(str).str.upper()
    out["doc_kind"] = out["doc_kind"].astype(str)

    # Dates -> ISO date (string)
    out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce").dt.date.astype("string")
    out["as_of_doc_date_hint"] = out["as_of_doc_date_hint"].astype("string")

    # Numeric coercions
    for c in ("program_cap_usd","sold_to_date_shares","sold_to_date_gross_usd","commission_pct"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Text-like fields
    for c in ("program_cap_text","sold_to_date_shares_text","sold_to_date_gross_text",
              "agents","source_url","snippet","event_type_final","accessionNumber"):
        out[c] = out[c].astype("string")

    # Drop rows missing critical identifiers
    out = out.dropna(subset=["ticker", "accessionNumber", "source_url"], how="any")

    # De-dupe exact duplicates
    out = out.drop_duplicates(keep="last").reset_index(drop=True)
    return out


def insert_atm_raw_df(
    table: str,
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Append-only insert into ATM_raw. Relies on DB auto PK.
    Use when you don't have a unique index/constraint for upserts.
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_atm_raw_df(df)

    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [ _normalize_row(r) for r in df2.to_dict(orient="records") ]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).insert(batch).execute()  # append
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}


def upsert_atm_raw_df(
    table: str,
    df: pd.DataFrame,
    on_conflict: str,
    do_update: bool = False,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Idempotent upload using Postgres ON CONFLICT.
    Requires a UNIQUE index/constraint backing `on_conflict`.

    Example `on_conflict`:
      - "uniq_atm_raw" (constraint/index name), or
      - "accessionNumber,source_url,event_type_final" (column list)
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_atm_raw_df(df)

    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [ _normalize_row(r) for r in df2.to_dict(orient="records") ]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).upsert(
            batch,
            on_conflict=on_conflict,
            ignore_duplicates=(not do_update),
            returning="minimal",
        ).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}

# ===== Noncrypto_holdings_raw (from SEC cash extractor) =====

NONCRYPTO_TABLE = "Noncrypto_holdings_raw"
NONCRYPTO_ASSET_LABEL = "CASH_LIKE"

def _nc_choose_asof_date(row: pd.Series) -> Optional[pd.Timestamp]:
    """
    Prefer the balance-sheet value column date, else reportDate, else filingDate.
    Returns a *date* (no time).
    """
    for col in ("bs_value_column_date", "reportDate", "filingDate"):
        val = pd.to_datetime(row.get(col), errors="coerce", utc=False)
        if pd.notna(val):
            return val.date()
    return None

def _nc_pick_cash_value(row: pd.Series, prefer_combined_if_missing: bool = False) -> Optional[float]:
    """
    Primary: cash_like_total_excl_restricted_usd
    Fallback A: cash_and_cash_equivalents_usd
    Optional Fallback B: combined_cash_cash_eq_restricted_usd (if prefer_combined_if_missing=True)
    """
    candidates = [
        "cash_like_total_excl_restricted_usd",
        "cash_and_cash_equivalents_usd",
    ]
    if prefer_combined_if_missing:
        candidates.append("combined_cash_cash_eq_restricted_usd")

    for col in candidates:
        v = row.get(col)
        try:
            if v is not None and pd.notna(v):
                return float(v)
        except Exception:
            continue
    return None

def prep_noncrypto_from_cash_df(
    cash_df: pd.DataFrame,
    asset_label: str = NONCRYPTO_ASSET_LABEL,
    prefer_combined_if_missing: bool = False,
) -> pd.DataFrame:
    """
    Transform the SEC cash extractor output into the schema required by
    Noncrypto_holdings_raw: (date, ticker, asset, total_holdings).

    - Chooses an 'as of' date per row (bs_value_column_date > reportDate > filingDate).
    - Picks the best cash metric (see _nc_pick_cash_value).
    - Dedupes to one row per (ticker, date) by taking the highest bs_score.
    """
    if cash_df is None or cash_df.empty:
        return pd.DataFrame(columns=["date", "ticker", "asset", "total_holdings"])

    df = cash_df.copy()

    # Normalize numeric fields
    for col in (
        "cash_like_total_excl_restricted_usd",
        "cash_and_cash_equivalents_usd",
        "combined_cash_cash_eq_restricted_usd",
        "bs_score",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            if col == "bs_score":
                df[col] = 0.0  # default when absent

    # Derive as-of date + value
    df["__asof_date"] = df.apply(_nc_choose_asof_date, axis=1)
    df["__value"] = df.apply(
        lambda r: _nc_pick_cash_value(r, prefer_combined_if_missing=prefer_combined_if_missing),
        axis=1,
    )

    # Keep valid rows
    df = df.dropna(subset=["ticker", "__asof_date", "__value"]).copy()

    # Choose best row per (ticker, date) by highest bs_score
    df["__rank"] = (
        df.groupby(["ticker", "__asof_date"])["bs_score"]
          .rank(method="first", ascending=False)
    )
    best = df[df["__rank"] == 1.0].copy()

    out = pd.DataFrame({
        "date": pd.to_datetime(best["__asof_date"], errors="coerce").dt.date.astype("string"),
        "ticker": best["ticker"].astype(str).str.upper(),
        "asset": asset_label,
        "total_holdings": best["__value"].astype(float),
    })

    out = out.dropna(subset=["date", "ticker"])
    out = out.drop_duplicates(subset=["date", "ticker"], keep="last").reset_index(drop=True)
    return out[["date", "ticker", "asset", "total_holdings"]]

def upload_noncrypto_from_cash(
    cash_df: pd.DataFrame,
    table: str = NONCRYPTO_TABLE,
    chunk_size: int = 1000,
    do_update: bool = False,
    prefer_combined_if_missing: bool = False,
    precheck: bool = True,
) -> Dict[str, int]:
    """
    End-to-end helper:
      - transform cash extractor rows -> Noncrypto_holdings_raw schema
      - upsert by key (date,ticker) via upload_df_by_key()

    Returns: {"attempted": int, "skipped_existing": int, "sent": int}
    """
    prepared = prep_noncrypto_from_cash_df(
        cash_df,
        asset_label=NONCRYPTO_ASSET_LABEL,
        prefer_combined_if_missing=prefer_combined_if_missing,
    )
    return upload_df_by_key(
        table=table,
        df=prepared,
        chunk_size=chunk_size,
        do_update=do_update,
        precheck=precheck,
    )
    
# ===== Reg_direct_raw upload helpers (ADD THIS SECTION) =====
REG_DIRECT_TABLE = "Reg_direct_raw"

REG_DIRECT_RAW_COLS = [
    "ticker","filingDate","form","accessionNumber","event_type_final",
    "price_per_share_text","price_per_share_usd",
    "shares_issued_text","shares_issued",
    "gross_proceeds_text","gross_proceeds_usd",
    "warrant_coverage_text","warrant_coverage_pct",
    "exercise_price_text","exercise_price_usd",
    "convert_price_text","convert_price_usd",
    "ownership_blocker_text","security_types_text",
    "source_url","exhibit_hint","snippet","score",
]

def prep_reg_direct_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Registered Direct rows for insertion into Reg_direct_raw.
    Does NOT send a 'key' column (assume DB generates PK / has unique constraint).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REG_DIRECT_RAW_COLS)

    out = df.copy()

    # Ensure schema columns exist
    for c in REG_DIRECT_RAW_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[REG_DIRECT_RAW_COLS]

    # Basic types / casing
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["form"] = out["form"].astype(str).str.upper()
    out["accessionNumber"] = out["accessionNumber"].astype("string")
    out["event_type_final"] = out["event_type_final"].astype("string")
    out["exhibit_hint"] = out["exhibit_hint"].astype("string")

    # Dates to ISO date strings
    out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce").dt.date.astype("string")

    # Numeric coercions
    for c in (
        "price_per_share_usd","gross_proceeds_usd","shares_issued",
        "warrant_coverage_pct","exercise_price_usd","convert_price_usd","score",
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Text-like fields
    for c in (
        "price_per_share_text","gross_proceeds_text","shares_issued_text",
        "warrant_coverage_text","exercise_price_text","convert_price_text",
        "ownership_blocker_text","security_types_text","source_url","snippet",
    ):
        out[c] = out[c].astype("string")

    # Drop rows missing critical identifiers
    out = out.dropna(subset=["ticker", "accessionNumber", "source_url"], how="any")

    # De-dupe conservative: (accessionNumber, source_url, event_type_final)
    out = out.drop_duplicates(subset=["accessionNumber","source_url","event_type_final"], keep="last").reset_index(drop=True)
    return out


def insert_reg_direct_raw_df(
    table: str,
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Append-only insert into Reg_direct_raw (no upsert).
    Use when the table's PK/unique constraint is DB-managed and duplicates are impossible/unlikely.
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_reg_direct_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).insert(batch).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}


def upsert_reg_direct_raw_df(
    table: str,
    df: pd.DataFrame,
    on_conflict: str = "accessionNumber,source_url,event_type_final",
    do_update: bool = False,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Idempotent upload into Reg_direct_raw using PostgREST upsert.
    Provide a UNIQUE constraint backing `on_conflict`, e.g.:

      CREATE UNIQUE INDEX IF NOT EXISTS uniq_reg_direct_raw
      ON public."Reg_direct_raw"(accessionNumber, source_url, event_type_final);

    Then call with on_conflict="accessionNumber,source_url,event_type_final" or the index name.
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_reg_direct_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).upsert(
            batch,
            on_conflict=on_conflict,
            ignore_duplicates=(not do_update),
            returning="minimal",
        ).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}


def run_reg_direct_daily_and_upload(
    tickers: List[str],
    *,
    table: str = REG_DIRECT_TABLE,
    since_hours: int = 24,
    use_acceptance: bool = True,
    limit: int = 600,
    max_docs: int = 6,
    max_snips: int = 4,
    upsert: bool = True,
    on_conflict: str = "accessionNumber,source_url,event_type_final",
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    End-to-end: run the Registered Direct extractor (past N hours) and upload to Supabase.

    Returns a dict with counts, e.g. {"attempted": X, "sent": Y} for insert,
    or {"attempted": X, "sent": Y} for upsert (Y == rows sent to API).
    """
    if not tickers:
        return {"attempted": 0, "sent": 0}

    # Lazy import to avoid heavy deps at module import time
    try:
        from app.sec_deals.drivers.run_reg_direct import build_for_tickers
    except Exception:
        from sec_deals.drivers.run_reg_direct import build_for_tickers  # fallback if installed as top-level

    df = build_for_tickers(
        tickers,
        year_by="accession",
        limit=limit,
        max_docs=max_docs,
        max_snips=max_snips,
        since_hours=since_hours,
        use_acceptance=use_acceptance,
    )

    if upsert:
        return upsert_reg_direct_raw_df(
            table=table,
            df=df,
            on_conflict=on_conflict,
            do_update=False,
            chunk_size=chunk_size,
        )
    else:
        return insert_reg_direct_raw_df(
            table=table,
            df=df,
            chunk_size=chunk_size,
        )
# ===== Outstanding_shares_raw upload helpers =====
OUTSTANDING_TABLE = "Outstanding_shares_raw"

OUTSTANDING_RAW_COLS = [
    "ticker","filingDate","form","accessionNumber","event_type_final",
    "outstanding_shares_text","outstanding_shares",
    "as_of_date_text","as_of_date_iso",
    "report_date_text","report_date_iso",
    "source_url","exhibit_hint","snippet","score",
]

def prep_outstanding_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Outstanding Shares rows for insertion into Outstanding_shares_raw.
    Append-only: assumes table has an identity PK; we do NOT send 'key'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTSTANDING_RAW_COLS)

    out = df.copy()

    # Ensure expected columns exist
    for c in OUTSTANDING_RAW_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUTSTANDING_RAW_COLS]

    # Types / casing
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["form"] = out["form"].astype(str).str.upper()
    out["accessionNumber"] = out["accessionNumber"].astype("string")
    out["event_type_final"] = out["event_type_final"].astype("string")
    out["exhibit_hint"] = out["exhibit_hint"].astype("string")

    # Dates -> ISO date (string)
    out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce").dt.date.astype("string")
    out["as_of_date_iso"] = pd.to_datetime(out["as_of_date_iso"], errors="coerce").dt.date.astype("string")
    out["report_date_iso"] = pd.to_datetime(out["report_date_iso"], errors="coerce").dt.date.astype("string")

    # Numerics
    out["outstanding_shares"] = pd.to_numeric(out["outstanding_shares"], errors="coerce")
    out["score"] = pd.to_numeric(out["score"], errors="coerce")

    # Text-like fields
    for c in (
        "outstanding_shares_text","as_of_date_text",
        "report_date_text","source_url","snippet"
    ):
        out[c] = out[c].astype("string")

    # Drop rows missing critical identifiers
    out = out.dropna(subset=["ticker","accessionNumber","source_url"], how="any")

    # Light de-dupe across exact triples (optional; keeps latest)
    out = out.drop_duplicates(subset=["accessionNumber","source_url","event_type_final"], keep="last").reset_index(drop=True)
    return out


def insert_outstanding_raw_df(
    table: str,
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Append-only insert into Outstanding_shares_raw.
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_outstanding_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).insert(batch).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}

# ===== Pipes_raw upload helpers =====
PIPES_TABLE = "Pipes_raw"

PIPES_RAW_COLS = [
    "ticker","filingDate","form","accessionNumber","event_type_final",
    "price_per_share_text","price_per_share_usd",
    "shares_issued_text","shares_issued",
    "gross_proceeds_text","gross_proceeds_usd",
    "warrant_coverage_text","warrant_coverage_pct",
    "exercise_price_text","exercise_price_usd",
    "convert_price_text","convert_price_usd",
    "ownership_blocker_text","ownership_blocker_pct",
    "reg_file_deadline_text","reg_file_deadline_days",
    "reg_effect_deadline_text","reg_effect_deadline_days",
    "closing_date_text","security_types_text",
    "source_url","exhibit_hint","snippet","score",
]

def prep_pipes_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize PIPE rows for insertion into Pipes_raw.
    Append-only: assumes table has an identity PK; we do NOT send 'key'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=PIPES_RAW_COLS)

    out = df.copy()

    # Ensure expected columns exist
    for c in PIPES_RAW_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[PIPES_RAW_COLS]

    # Types / casing
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["form"] = out["form"].astype(str).str.upper()
    out["accessionNumber"] = out["accessionNumber"].astype("string")
    out["event_type_final"] = out["event_type_final"].astype("string")
    out["exhibit_hint"] = out["exhibit_hint"].astype("string")
    out["security_types_text"] = out["security_types_text"].astype("string")
    out["closing_date_text"] = out["closing_date_text"].astype("string")

    # Dates -> ISO date (string)
    out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce").dt.date.astype("string")

    # Numerics
    for c in (
        "price_per_share_usd","gross_proceeds_usd","shares_issued",
        "warrant_coverage_pct","exercise_price_usd","convert_price_usd",
        "ownership_blocker_pct","reg_file_deadline_days","reg_effect_deadline_days","score",
    ):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Text-like fields
    for c in (
        "price_per_share_text","gross_proceeds_text","shares_issued_text",
        "warrant_coverage_text","exercise_price_text","convert_price_text",
        "ownership_blocker_text","reg_file_deadline_text","reg_effect_deadline_text",
        "source_url","snippet",
    ):
        out[c] = out[c].astype("string")

    # Drop rows missing critical identifiers
    out = out.dropna(subset=["ticker","accessionNumber","source_url"], how="any")

    # Light de-dupe across exact triples (optional; keeps latest)
    out = out.drop_duplicates(subset=["accessionNumber","source_url","event_type_final"], keep="last").reset_index(drop=True)
    return out


def insert_pipes_raw_df(
    table: str,
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Append-only insert into Pipes_raw.
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_pipes_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).insert(batch).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}


# ===== Warrants_new_iss_raw upload helpers =====
WARRANTS_NEW_TABLE = "Warrants_new_iss_raw"

WARRANTS_NEW_COLS = [
    "ticker","filingDate","form","accessionNumber","event_type_final",
    # exercise
    "exercise_price_text","exercise_price_text_formula","exercise_price_usd",
    # shares (legacy + explicit buckets)
    "shares_issued_text","shares_issued",
    "warrant_shares_prefunded_text","warrant_shares_prefunded",
    "warrant_shares_outstanding_text","warrant_shares_outstanding",
    # typing / flags
    "warrant_type","warrant_prefunded_flag","warrant_standard_flag",
    # instruments (count of warrants issued)
    "warrant_instruments_text","warrant_instruments",
    # coverage / blockers
    "warrant_coverage_text","warrant_coverage_pct",
    "ownership_blocker_text","ownership_blocker_pct",
    # term / dates
    "expiration_date_text","expiration_date_iso","warrant_term_years","issuance_date_text",
    # money / security types / roles
    "gross_proceeds_text","security_types_text",
    "h_warrant_role","agent_fee_text","agent_fee_pct",
    # provenance
    "source_url","exhibit_hint","snippet","score",
]

def prep_warrants_new_iss_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize warrant NEW ISSUANCE rows for insertion into Warrants_new_iss_raw.
    Append-only or upsert depending on caller. No 'key' column is sent.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=WARRANTS_NEW_COLS)

    out = df.copy()

    # Ensure schema columns exist
    for c in WARRANTS_NEW_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[WARRANTS_NEW_COLS]

    # Basic types / casing
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["form"] = out["form"].astype(str).str.upper()
    out["accessionNumber"] = out["accessionNumber"].astype("string")
    out["event_type_final"] = out["event_type_final"].astype("string")
    out["exhibit_hint"] = out["exhibit_hint"].astype("string")
    out["h_warrant_role"] = out["h_warrant_role"].astype("string")
    out["security_types_text"] = out["security_types_text"].astype("string")

    # Dates -> ISO date (string)
    out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce").dt.date.astype("string")
    out["expiration_date_iso"] = pd.to_datetime(out["expiration_date_iso"], errors="coerce").dt.date.astype("string")
    out["issuance_date_text"] = out["issuance_date_text"].astype("string")

    # Numerics
    num_cols = [
        "exercise_price_usd","shares_issued",
        "warrant_shares_prefunded","warrant_shares_outstanding",
        "warrant_instruments","warrant_coverage_pct","ownership_blocker_pct",
        "warrant_term_years","agent_fee_pct","score",
    ]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Text-like fields
    text_cols = [
        "exercise_price_text","exercise_price_text_formula",
        "shares_issued_text",
        "warrant_shares_prefunded_text","warrant_shares_outstanding_text",
        "warrant_type","warrant_instruments_text",
        "warrant_coverage_text","ownership_blocker_text",
        "expiration_date_text","gross_proceeds_text","source_url","snippet",
    ]
    for c in text_cols:
        out[c] = out[c].astype("string")

    # Booleans where present
    for c in ("warrant_prefunded_flag","warrant_standard_flag"):
        if c in out.columns:
            out[c] = out[c].astype("boolean")

    # Drop rows missing critical identifiers
    out = out.dropna(subset=["ticker","accessionNumber","source_url"], how="any")

    # De-dupe conservatively by (accessionNumber, source_url, event_type_final)
    out = out.drop_duplicates(
        subset=["accessionNumber","source_url","event_type_final"], keep="last"
    ).reset_index(drop=True)

    return out


def insert_warrants_new_iss_raw_df(
    table: str,
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Append-only insert into Warrants_new_iss_raw (no upsert).
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_warrants_new_iss_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).insert(batch).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}


def upsert_warrants_new_iss_raw_df(
    table: str,
    df: pd.DataFrame,
    on_conflict: str = "accessionNumber,source_url,event_type_final",
    do_update: bool = False,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Idempotent upload into Warrants_new_iss_raw using PostgREST upsert.

    Recommended UNIQUE index:
      CREATE UNIQUE INDEX IF NOT EXISTS uniq_warrants_new_iss_raw
      ON public."Warrants_new_iss_raw"(accessionNumber, source_url, event_type_final);
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_warrants_new_iss_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).upsert(
            batch,
            on_conflict=on_conflict,
            ignore_duplicates=(not do_update),
            returning="minimal",
        ).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}


def run_warrants_daily_and_upload(
    tickers: List[str],
    *,
    table: str = WARRANTS_NEW_TABLE,
    recent_hours: int = 24,
    limit: int = 600,
    max_docs: int = 6,
    max_snips: int = 4,
    upsert: bool = True,
    on_conflict: str = "accessionNumber,source_url,event_type_final",
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    End-to-end: run the Warrants NEW ISSUANCE extractor (past N hours) and upload to Supabase.
    """
    if not tickers:
        return {"attempted": 0, "sent": 0}

    # Lazy import to avoid heavy deps at module import time
    try:
        from app.sec_deals.drivers.run_warrants import build_for_ticker
    except Exception:
        from sec_deals.drivers.run_warrants import build_for_ticker

    frames: List[pd.DataFrame] = []
    for t in sorted(set(x.strip().upper() for x in tickers if x.strip())):
        df_t = build_for_ticker(
            t,
            year=pd.Timestamp.utcnow().year,   # ignored when recent_hours > 0
            year_by="accession",
            limit=limit,
            max_docs=max_docs,
            max_snips=max_snips,
            recent_hours=recent_hours,
        )
        if df_t is not None and not df_t.empty:
            frames.append(df_t)

    if not frames:
        return {"attempted": 0, "sent": 0}

    df = pd.concat(frames, ignore_index=True)

    if upsert:
        return upsert_warrants_new_iss_raw_df(
            table=table,
            df=df,
            on_conflict=on_conflict,
            do_update=False,
            chunk_size=chunk_size,
        )
    else:
        return insert_warrants_new_iss_raw_df(
            table=table,
            df=df,
            chunk_size=chunk_size,
        )

# ===== Outstanding_warrants_raw upload helpers =====
OUT_WARRANTS_TABLE = "Outstanding_warrants_raw"

OUT_WARRANTS_RAW_COLS = [
    "ticker","filingDate","form","accessionNumber","primaryDocument",
    "reportDate","acceptanceDateTime",
    "source_url","table_heading","row_label","is_total_row",
    "warrants_outstanding","exercise_price_usd",
    "warrant_expiration_date","warrant_term_years",
    "units_multiplier","score",
]

def prep_outstanding_warrants_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OUT_WARRANTS_RAW_COLS)

    out = df.copy()
    for c in OUT_WARRANTS_RAW_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[OUT_WARRANTS_RAW_COLS]

    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["form"] = out["form"].astype(str).str.upper()
    for c in ("accessionNumber","primaryDocument","source_url","table_heading","row_label"):
        out[c] = out[c].astype("string")

    out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce").dt.date.astype("string")
    out["reportDate"] = pd.to_datetime(out["reportDate"], errors="coerce").dt.date.astype("string")
    out["acceptanceDateTime"] = pd.to_datetime(out["acceptanceDateTime"], errors="coerce").astype("string")
    out["warrant_expiration_date"] = pd.to_datetime(out["warrant_expiration_date"], errors="coerce").dt.date.astype("string")

    for c in ("warrants_outstanding","exercise_price_usd","warrant_term_years","units_multiplier","score"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "is_total_row" in out.columns:
        out["is_total_row"] = out["is_total_row"].astype("boolean")

    # keep only identifiable rows
    out = out.dropna(subset=["ticker","accessionNumber","source_url"], how="any")

    # optional: drop header-ish rows with no count
    out = out[(out["warrants_outstanding"].notna()) & (out["warrants_outstanding"] >= 1)]

    # de-dupe by (filing row)
    out = out.drop_duplicates(
        subset=["accessionNumber","source_url","row_label"], keep="last"
    ).reset_index(drop=True)

    return out

def insert_outstanding_warrants_raw_df(table: str, df: pd.DataFrame, chunk_size: int = 500) -> dict:
    sb = get_supabase()
    df2 = prep_outstanding_warrants_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}
    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]
    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i+chunk_size]
        sb.table(table).insert(batch).execute()
        sent += len(batch)
    return {"attempted": len(df2), "sent": sent}

def upsert_outstanding_warrants_raw_df(
    table: str,
    df: pd.DataFrame,
    on_conflict: str = "accessionNumber,source_url,row_label",
    do_update: bool = False,
    chunk_size: int = 500,
) -> dict:
    """
    Recommended unique index:
      CREATE UNIQUE INDEX IF NOT EXISTS uniq_outstanding_warrants_raw
      ON public."Outstanding_warrants_raw"(accessionNumber, source_url, row_label);
    """
    sb = get_supabase()
    df2 = prep_outstanding_warrants_raw_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}
    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]
    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i+chunk_size]
        sb.table(table).upsert(
            batch,
            on_conflict=on_conflict,
            ignore_duplicates=(not do_update),
            returning="minimal",
        ).execute()
        sent += len(batch)
    return {"attempted": len(df2), "sent": sent}

# ===== Recent_filings upload helpers =====
RECENT_FILINGS_TABLE = "Recent_filings"

RECENT_FILINGS_COLS = [
    "filingDate", "ticker", "form", "accessionNumber", "cik", "source_url"
]

def prep_recent_filings_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize recent filings DataFrame for insertion into Recent_filings.
    Append-only: assumes table has an identity PK; we do NOT send 'key'.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=RECENT_FILINGS_COLS)

    out = df.copy()

    # Ensure expected columns exist
    for c in RECENT_FILINGS_COLS:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[RECENT_FILINGS_COLS]

    # Types / casing
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["form"] = out["form"].astype(str).str.upper()
    out["accessionNumber"] = out["accessionNumber"].astype("string")
    out["cik"] = out["cik"].astype("string")
    out["source_url"] = out["source_url"].astype("string")

    # Dates -> ISO date (string)
    out["filingDate"] = pd.to_datetime(out["filingDate"], errors="coerce").dt.date.astype("string")

    # Drop rows missing critical identifiers
    out = out.dropna(subset=["ticker", "accessionNumber", "source_url"], how="any")

    # De-dupe by (accessionNumber, source_url) to avoid exact duplicates
    out = out.drop_duplicates(subset=["accessionNumber", "source_url"], keep="last").reset_index(drop=True)
    return out


def insert_recent_filings_df(
    table: str,
    df: pd.DataFrame,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    Append-only insert into Recent_filings.
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    sb = get_supabase()
    df2 = prep_recent_filings_df(df)
    if df2.empty:
        return {"attempted": 0, "sent": 0}

    payload = [_normalize_row(r) for r in df2.to_dict(orient="records")]

    sent = 0
    for i in range(0, len(payload), chunk_size):
        batch = payload[i:i + chunk_size]
        sb.table(table).insert(batch).execute()
        sent += len(batch)

    return {"attempted": len(df2), "sent": sent}


def run_pull_files_daily_and_upload(
    tickers: List[str],
    *,
    table: str = RECENT_FILINGS_TABLE,
    top_n: int = 5,
    chunk_size: int = 500,
) -> Dict[str, int]:
    """
    End-to-end: run the pull_files script and upload to Supabase.
    
    Returns a dict with counts, e.g. {"attempted": X, "sent": Y}.
    """
    if not tickers:
        return {"attempted": 0, "sent": 0}

    # Lazy import to avoid heavy deps at module import time
    try:
        from app.sec_deals.drivers.pull_files import gather_recent_filings
    except Exception:
        from sec_deals.drivers.pull_files import gather_recent_filings  # fallback if installed as top-level

    df = gather_recent_filings(tickers=tickers, top_n=top_n)
    
    return insert_recent_filings_df(
        table=table,
        df=df,
        chunk_size=chunk_size,
    )