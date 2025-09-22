# app/clients/upload_yfinance.py
from datetime import date
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from supabase import create_client

EXPECTED_COLS = ["date","ticker","open","high","low","close","adj_close","volume"]

def df_to_records_clean(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    df = df.loc[:, [c for c in EXPECTED_COLS if c in df.columns]].copy()

    numeric = ["open","high","low","close","adj_close","volume"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["open","high","low","close","volume"], inplace=True)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["ticker"] = df["ticker"].astype(str)
    df["key"] = df["date"].astype(str) + "-" + df["ticker"]

    df = df.where(df.notnull(), None)

    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            if isinstance(v, pd.Timestamp):
                v = v.date().isoformat()
            if isinstance(v, date):
                v = v.isoformat()
            clean[k] = v
        # drop records with any bad float
        bad = any(isinstance(val, float) and (val != val or val in (float("inf"), float("-inf"))) for val in clean.values())
        if not bad:
            records.append(clean)
    return records

def upload_yfinance_df(supabase_url: str, supabase_key: str, df: pd.DataFrame):
    recs = df_to_records_clean(df)
    if not recs:
        return {"attempted": 0, "sent": 0}
    client = create_client(supabase_url, supabase_key)
    res = client.table("yfinance").upsert(recs, on_conflict="key").execute()
    return {"attempted": len(recs), "sent": len(recs)}
