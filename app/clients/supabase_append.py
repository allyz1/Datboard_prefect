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

def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
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
