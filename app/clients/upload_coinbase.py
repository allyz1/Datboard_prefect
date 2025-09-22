# app/clients/upload_coinbase.py
from datetime import date
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from supabase import create_client  # you already use this in your helper

def df_to_records_clean(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # keep only expected cols and order
    cols = ["date", "product_id", "open", "high", "low", "close", "volume"]
    df = df.loc[:, cols].copy()

    # coerce numerics; remove inf/nan rows
    num = ["open","high","low","close","volume"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=num, inplace=True)

    # key & canonical types
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["product_id"] = df["product_id"].astype(str)
    df["key"] = df["date"].astype(str) + "-" + df["product_id"]

    # convert any remaining NaN to None
    df = df.where(df.notnull(), None)

    # convert to plain python (no numpy scalars)
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        rec: Dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            if isinstance(v, pd.Timestamp):
                v = v.date().isoformat()
            if isinstance(v, date):
                v = v.isoformat()  # 'YYYY-MM-DD' for PostgREST date
            rec[k] = v
        records.append(rec)

    # belt & suspenders: remove any keys whose value is not JSON-safe
    def is_bad(x):
        return isinstance(x, float) and (x != x or x in (float("inf"), float("-inf")))
    for r in records:
        for k, v in list(r.items()):
            if is_bad(v):
                # drop the whole record rather than send a bad value
                r["__DROP__"] = True
                break
    records = [r for r in records if not r.get("__DROP__")]

    return records

def upload_coinbase_df(supabase_url: str, supabase_key: str, df: pd.DataFrame):
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    records = df_to_records_clean(df)
    if not records:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    client = create_client(supabase_url, supabase_key)
    res = client.table("coinbase").upsert(records, on_conflict="key").execute()
    # You can inspect res.data / res.count depending on your client version
    return {"attempted": len(records), "sent": len(records), "skipped_existing": 0}
