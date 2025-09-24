# app/clients/upload_polygon.py
from datetime import date
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from supabase import create_client

# Only the columns produced by your polygon script
EXPECTED_COLS = [
    "date","ticker",
    "open","high","low","close",
    "volume","transactions","vwap"
]

def df_to_records_clean_polygon(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    # Keep only expected columns (in case extras are present)
    df = df.loc[:, [c for c in EXPECTED_COLS if c in df.columns]].copy()

    # Coerce numerics
    numeric_float = ["open","high","low","close","volume","vwap"]
    df[numeric_float] = df[numeric_float].apply(pd.to_numeric, errors="coerce")

    # transactions frequently comes as an int; keep robust
    df["transactions"] = pd.to_numeric(df.get("transactions"), errors="coerce")

    # Drop rows missing core OHLCV
    df.dropna(subset=["open","high","low","close","volume"], inplace=True)

    # Normalize types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str).str.upper()

    # Replace inf with NaN, then None for JSON
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.where(df.notnull(), None)

    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            # Normalize date to ISO for safety
            if isinstance(v, pd.Timestamp):
                v = v.date().isoformat()
            elif isinstance(v, date):
                v = v.isoformat()

            # Cast numpy to native
            if isinstance(v, (np.floating,)):
                v = float(v)
            if isinstance(v, (np.integer,)):
                v = int(v)

            # Ensure transactions is int if present and finite
            if k == "transactions":
                if v is None:
                    pass
                else:
                    try:
                        v = int(v)
                    except Exception:
                        v = None

            clean[k] = v

        # Drop if any NaN/inf snuck through
        bad = any(isinstance(val, float) and (val != val or val in (float("inf"), float("-inf")))
                  for val in clean.values())
        if not bad:
            records.append(clean)

    return records

def upload_polygon_df(
    supabase_url: str,
    supabase_key: str,
    df: pd.DataFrame,
    *,
    table: str = "polygon",
    on_conflict: str = "key",   # switch to 'date,ticker' if you use a composite PK
    chunk_size: int = 1000
):
    recs = df_to_records_clean_polygon(df)
    if not recs:
        return {"attempted": 0, "sent": 0}

    client = create_client(supabase_url, supabase_key)

    sent = 0
    for i in range(0, len(recs), chunk_size):
        batch = recs[i:i+chunk_size]
        client.table(table).upsert(batch, on_conflict=on_conflict).execute()
        sent += len(batch)

    return {"attempted": len(recs), "sent": sent}
