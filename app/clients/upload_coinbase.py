# app/clients/upload_coinbase.py
from datetime import date
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from supabase import create_client

# app/clients/upload_coinbase.py

EXPECTED = ["date", "product_id", "open", "close"]

def df_to_records_clean(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []

    for c in EXPECTED:
        if c not in df.columns:
            df[c] = None

    df = df.loc[:, EXPECTED].copy()

    df[["open","close"]] = df[["open","close"]].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["open","close"], inplace=True)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["product_id"] = df["product_id"].astype(str)

    # IMPORTANT: do NOT create/send 'key' if it's generated in DB
    df = df.where(df.notnull(), None)

    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        r = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            if isinstance(v, pd.Timestamp):
                v = v.date().isoformat()
            if isinstance(v, date):
                v = v.isoformat()
            r[k] = v
        records.append(r)

    return records

def upload_coinbase_df(supabase_url: str, supabase_key: str, df: pd.DataFrame):
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    records = df_to_records_clean(df)
    if not records:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    client = create_client(supabase_url, supabase_key)
    # If 'key' is the PK/unique generated column:
    res = client.table("coinbase").upsert(records, on_conflict="key").execute()
    # Alternatively (also fine), upsert on the natural key:
    # res = client.table("coinbase").upsert(records, on_conflict="date,product_id").execute()

    return {"attempted": len(records), "sent": len(records), "skipped_existing": 0}

def df_to_crypto_supply_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert CoinGecko supply DataFrame to clean records for Supabase upload.
    Expected columns: asset, circulating_supply, total_supply, date
    """
    if df is None or df.empty:
        return []

    # Ensure expected columns exist
    expected_cols = ["asset", "circulating_supply", "total_supply", "date"]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None

    df = df.loc[:, expected_cols].copy()

    # Convert numeric columns
    df[["circulating_supply", "total_supply"]] = df[["circulating_supply", "total_supply"]].apply(pd.to_numeric, errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Convert asset to uppercase string
    df["asset"] = df["asset"].astype(str).str.upper()
    
    # Ensure date is in ISO format
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")

    # Drop rows missing critical data
    df.dropna(subset=["asset", "date"], inplace=True)

    # Convert to records
    df = df.where(df.notnull(), None)
    records: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        r = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                v = float(v)
            if isinstance(v, pd.Timestamp):
                v = v.date().isoformat()
            if isinstance(v, date):
                v = v.isoformat()
            r[k] = v
        records.append(r)

    return records

def upload_crypto_supply_df(supabase_url: str, supabase_key: str, df: pd.DataFrame):
    """
    Upload CoinGecko supply data to Crypto_supply table in Supabase.
    
    Args:
        supabase_url: Supabase project URL
        supabase_key: Supabase service key
        df: DataFrame with columns: asset, circulating_supply, total_supply, date
    
    Returns:
        Dict with upload statistics
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    records = df_to_crypto_supply_records(df)
    if not records:
        return {"attempted": 0, "sent": 0, "skipped_existing": 0}

    client = create_client(supabase_url, supabase_key)
    
    # Upsert on natural key (asset, date) to avoid duplicates
    res = client.table("Crypto_supply").upsert(records, on_conflict="asset,date").execute()

    return {"attempted": len(records), "sent": len(records), "skipped_existing": 0}