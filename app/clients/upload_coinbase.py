# app/clients/upload_coinbase.py
from datetime import date
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from supabase import create_client

EXPECTED = ["date", "product_id", "open", "close"]

def df_to_records_clean(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    # ensure required columns exist
    for c in EXPECTED:
        if c not in df.columns:
            df[c] = None

    # keep only expected cols and order
    df = df.loc[:, EXPECTED].copy()

    # coerce numerics; drop rows missing open/close
    df[["open","close"]] = df[["open","close"]].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["open","close"], inplace=True)

    # normalize types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["product_id"] = df["product_id"].astype(str)

    # stable key (unique per day+product)
    df["key"] = df["date"].astype(str) + "-" + df["product_id"]

    # convert NaN -> None for PostgREST
    df = df.where(df.notnull(), None)

    # convert to plain python scalars
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            if isinstance(v, pd.Timestamp):
                v = v.date().isoformat()
            if isinstance(v, date):
                v = v.isoformat()
            rec[k] = v
        records.append(rec)

    # final guard against non-finite floats
    def bad(x): return isinstance(x, float) and (np.isnan(x) or np.isinf(x))
    cleaned = []
    for r in records:
        if bad(r.get("open")) or bad(r.get("close")):
            continue
        cleaned.append(r)

    return cleaned

def upload_coinbase_df(supabase_url: str, supabase_key: str, df: pd.DataFrame):
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    records = df_to_records_clean(df)
    if not records:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    client = create_client(supabase_url, supabase_key)
    res = client.table("coinbase").upsert(records, on_conflict="key").execute()
    return {"attempted": len(records), "sent": len(records), "skipped_existing": 0}
