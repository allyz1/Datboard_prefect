#!/usr/bin/env python3
"""
Polygon.io Daily Market Summary scraper.

This module fetches daily OHLC (open, high, low, close), volume, and VWAP data
for all U.S. stocks on a specified trading date using Polygon.io's API.

API Reference: https://polygon.io/docs/rest/stocks/aggregates/daily-market-summary
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone, date, timedelta
import os

# Import Supabase client
try:
    from app.clients.supabase_append import get_supabase, _normalize_row
except ImportError:
    print("Warning: Supabase client not available. CSV export will be used instead.")
    get_supabase = None
    _normalize_row = None

# Polygon.io configuration
USER_AGENT = "Ally Zach <ally@panteracapital.com>"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

# Polygon.io API endpoint
POLYGON_MARKET_SUMMARY_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks"

def get_with_retry(url: str, params: Dict, max_retries: int = 3, backoff: float = 1.5) -> Optional[requests.Response]:
    """
    Make HTTP request with retry logic and politeness delays.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            time.sleep(0.1)  # Be polite to Polygon.io
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                # Rate limited - wait longer
                sleep_time = backoff * (attempt + 1) * 2
                print(f"Rate limited, waiting {sleep_time}s...")
                time.sleep(sleep_time)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff * (attempt + 1))
    
    if last_err:
        print(f"Failed to fetch {url} after {max_retries} attempts: {last_err}")
    return None

def get_polygon_market_summary_yesterday(
    api_key: str,
    adjusted: bool = True,
    include_otc: bool = False,
) -> pd.DataFrame:
    """
    Fetch yesterday's market summary data from Polygon.io for all U.S. stocks.
    
    Args:
        api_key: Polygon.io API key
        adjusted: Whether to adjust for splits (default: True)
        include_otc: Whether to include OTC securities (default: False)
    
    Returns:
        DataFrame with columns: ticker, date, open, high, low, close, volume, vwap, transactions
    """
    # Get yesterday's date
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    target_date = yesterday.isoformat()
    if not api_key:
        print("Error: Polygon.io API key is required")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
    
    # API parameters
    params = {
        "adjusted": str(adjusted).lower(),
        "include_otc": str(include_otc).lower(),
    }
    
    # Add API key to headers
    headers_with_key = HEADERS.copy()
    headers_with_key["Authorization"] = f"Bearer {api_key}"
    
    # Construct URL with date
    url = f"{POLYGON_MARKET_SUMMARY_URL}/{target_date}"
    
    print(f"Fetching yesterday's market summary ({target_date}) from Polygon.io...")
    
    try:
        response = requests.get(url, params=params, headers=headers_with_key, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK":
            print(f"API returned error status: {data.get('status')}")
            print(f"Response: {data}")
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
        
        results = data.get("results", [])
        if not results:
            print(f"No market data found for {target_date}")
            print(f"API response keys: {list(data.keys())}")
            print(f"Query count: {data.get('queryCount', 'N/A')}")
            print(f"Results count: {data.get('resultsCount', 'N/A')}")
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
        
        # Convert to DataFrame
        rows = []
        for item in results:
            # Convert Unix timestamp to date
            timestamp_ms = item.get("t", 0)
            if timestamp_ms:
                trade_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date()
            else:
                trade_date = target_date
            
            row = {
                "ticker": item.get("T", "").upper(),
                "date": trade_date.isoformat(),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v"),
                "vwap": item.get("vw"),
                "transactions": item.get("n"),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Filter out rows with missing critical data
        df = df.dropna(subset=["ticker", "date", "close"])
        
        # Sort by ticker
        df = df.sort_values("ticker").reset_index(drop=True)
        
        print(f"Retrieved {len(df)} stock records for {target_date}")
        return df
        
    except Exception as e:
        print(f"Error fetching market summary: {e}")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])

def get_polygon_market_summary_simple(api_key: str) -> pd.DataFrame:
    """
    Simplified version that fetches yesterday's market data.
    Convenience function for integration with other scripts.
    """
    return get_polygon_market_summary_yesterday(api_key)

def get_polygon_market_summary_for_date(target_date: str, api_key: str) -> pd.DataFrame:
    """
    Fetch market summary data for a specific date.
    Useful for testing with known trading days.
    """
    if not api_key:
        print("Error: Polygon.io API key is required")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
    
    # API parameters
    params = {
        "adjusted": "true",
        "include_otc": "false",
    }
    
    # Add API key to headers
    headers_with_key = HEADERS.copy()
    headers_with_key["Authorization"] = f"Bearer {api_key}"
    
    # Construct URL with date
    url = f"{POLYGON_MARKET_SUMMARY_URL}/{target_date}"
    
    print(f"Fetching market summary for {target_date} from Polygon.io...")
    
    try:
        response = requests.get(url, params=params, headers=headers_with_key, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") != "OK":
            print(f"API returned error status: {data.get('status')}")
            print(f"Response: {data}")
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
        
        results = data.get("results", [])
        if not results:
            print(f"No market data found for {target_date}")
            print(f"API response keys: {list(data.keys())}")
            print(f"Query count: {data.get('queryCount', 'N/A')}")
            print(f"Results count: {data.get('resultsCount', 'N/A')}")
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
        
        # Convert to DataFrame
        rows = []
        for item in results:
            # Convert Unix timestamp to date
            timestamp_ms = item.get("t", 0)
            if timestamp_ms:
                trade_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date()
            else:
                trade_date = target_date
            
            row = {
                "ticker": item.get("T", "").upper(),
                "date": trade_date.isoformat(),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v"),
                "vwap": item.get("vw"),
                "transactions": item.get("n"),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Filter out rows with missing critical data
        df = df.dropna(subset=["ticker", "date", "close"])
        
        # Sort by ticker
        df = df.sort_values("ticker").reset_index(drop=True)
        
        print(f"Retrieved {len(df)} stock records for {target_date}")
        return df
        
    except Exception as e:
        print(f"Error fetching market summary: {e}")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])

def get_polygon_ticker_details(api_key: str, ticker: str) -> dict:
    """
    Get detailed ticker information including market cap from Polygon.io.
    """
    if not api_key:
        return {}
    
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
    headers_with_key = HEADERS.copy()
    headers_with_key["Authorization"] = f"Bearer {api_key}"
    
    try:
        response = requests.get(url, headers=headers_with_key, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "OK" and "results" in data:
            return data["results"]
        return {}
        
    except Exception as e:
        print(f"Error fetching ticker details for {ticker}: {e}")
        return {}

def get_polygon_market_summary_with_marketcap(
    target_date: str,
    api_key: str,
    adjusted: bool = True,
    include_otc: bool = False,
    max_tickers: int = 100,  # Limit to avoid too many API calls
) -> pd.DataFrame:
    """
    Fetch market summary data and enrich with market cap information.
    Note: This makes additional API calls for each ticker, so it's slower.
    """
    # First get the basic market data
    df = get_polygon_market_summary_for_date(target_date, api_key)
    
    if df.empty:
        return df
    
    # Limit the number of tickers to avoid too many API calls
    if len(df) > max_tickers:
        print(f"Limiting to top {max_tickers} tickers by volume to avoid excessive API calls")
        df = df.nlargest(max_tickers, 'volume')
    
    # Add market cap column
    df['market_cap'] = None
    
    print(f"Fetching market cap data for {len(df)} tickers...")
    
    for i, row in df.iterrows():
        ticker = row['ticker']
        details = get_polygon_ticker_details(api_key, ticker)
        
        if details:
            # Market cap is usually in the 'market_cap' field
            market_cap = details.get('market_cap')
            if market_cap:
                df.at[i, 'market_cap'] = market_cap
        
        # Be polite to the API
        if i % 10 == 0:
            print(f"Processed {i+1}/{len(df)} tickers...")
            time.sleep(0.1)
    
    return df

# ---------------------- DAT Ticker Configuration ----------------------

# Default DAT ticker list - modify this to include all your DAT holdings
DEFAULT_DAT_TICKERS = [
    "MSTR","CEP","SMLR","NAKA","BMNR","SBET","ETHZ","BTCS","SQNS","BTBT","DFDV","UPXI","HSDT","FORD"
]

# ---------------------- Ranking Functions ----------------------

def add_volume_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based ranking columns to the market data DataFrame.
    
    Args:
        df: DataFrame with market data including volume and close price
        
    Returns:
        DataFrame with additional ranking columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ensure numeric types
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0)
    
    # Calculate dollar volume
    df["dollar_volume"] = df["close"] * df["volume"]
    
    # Calculate ranks (1 = highest)
    df["rank_dollar_volume"] = df["dollar_volume"].rank(method="min", ascending=False).astype(int)
    df["rank_transactions"] = df["transactions"].rank(method="min", ascending=False).astype(int)
    
    # Add universe size
    df["universe_size"] = len(df)
    
    return df

def get_dat_ticker_ranks(df_ranks: pd.DataFrame, dat_tickers: List[str]) -> pd.DataFrame:
    """
    Extract ranking data for DAT tickers plus their ranking neighbors.
    
    Args:
        df_ranks: DataFrame with ranking data
        dat_tickers: List of ticker symbols to extract
        
    Returns:
        DataFrame with ranking data for DAT tickers and their ranking neighbors
    """
    if not dat_tickers or df_ranks.empty:
        return pd.DataFrame()
    
    # Clean and normalize ticker list
    dat_tickers = [t.strip().upper() for t in dat_tickers if t.strip()]
    
    # Filter for DAT tickers
    dat_df = df_ranks[df_ranks["ticker"].isin(dat_tickers)].copy()
    
    if dat_df.empty:
        return pd.DataFrame()
    
    # Add DAT flag
    dat_df["is_dat"] = True
    
    # Find DAT leaders in each category
    dat_dollar_leader_rank = dat_df["rank_dollar_volume"].min()
    dat_transaction_leader_rank = dat_df["rank_transactions"].min()
    
    # Get neighbors (above and below) for each category
    neighbors = []
    
    # Dollar volume ranking neighbors
    dollar_neighbors = df_ranks[
        (df_ranks["rank_dollar_volume"] >= dat_dollar_leader_rank - 1) & 
        (df_ranks["rank_dollar_volume"] <= dat_dollar_leader_rank + 1)
    ].copy()
    dollar_neighbors["is_dat"] = dollar_neighbors["ticker"].isin(dat_tickers)
    neighbors.append(dollar_neighbors)
    
    # Transaction ranking neighbors
    transaction_neighbors = df_ranks[
        (df_ranks["rank_transactions"] >= dat_transaction_leader_rank - 1) & 
        (df_ranks["rank_transactions"] <= dat_transaction_leader_rank + 1)
    ].copy()
    transaction_neighbors["is_dat"] = transaction_neighbors["ticker"].isin(dat_tickers)
    neighbors.append(transaction_neighbors)
    
    # Combine all neighbors and remove duplicates
    all_neighbors = pd.concat(neighbors, ignore_index=True).drop_duplicates(subset=["ticker"])
    
    # Sort by ticker for consistent output
    all_neighbors = all_neighbors.sort_values("ticker").reset_index(drop=True)
    
    return all_neighbors

def upload_rankings_to_supabase(
    df_ranks: pd.DataFrame,
    table_name: str = "Stock_rankings_daily_pull",
    replace_table: bool = True
) -> Dict[str, int]:
    """
    Upload ranking data to Supabase table.
    
    Args:
        df_ranks: DataFrame with ranking data
        table_name: Supabase table name
        replace_table: If True, delete all existing data before inserting
        
    Returns:
        Dictionary with upload statistics
    """
    if get_supabase is None:
        raise RuntimeError("Supabase client not available")
    
    if df_ranks.empty:
        return {"attempted": 0, "sent": 0, "errors": []}
    
    sb = get_supabase()
    
    try:
        # Replace table if requested
        if replace_table:
            print(f"Clearing existing data from {table_name}...")
            sb.table(table_name).delete().neq("ticker", "").execute()
        
        # Prepare data for upload
        df_upload = df_ranks.copy()
        
        # Ensure proper data types
        df_upload["date"] = pd.to_datetime(df_upload["date"], errors="coerce").dt.date.astype("string")
        df_upload["ticker"] = df_upload["ticker"].astype(str).str.upper()
        
        # Convert numeric columns (excluding OHLC, VWAP, and volume)
        numeric_cols = ["transactions", "dollar_volume", "rank_dollar_volume", "rank_transactions", "universe_size"]
        for col in numeric_cols:
            if col in df_upload.columns:
                df_upload[col] = pd.to_numeric(df_upload[col], errors="coerce")
        
        # Convert boolean columns
        if "is_dat" in df_upload.columns:
            df_upload["is_dat"] = df_upload["is_dat"].astype(bool)
        
        # Drop OHLC, VWAP, and volume columns
        columns_to_drop = ["open", "high", "low", "close", "vwap", "volume"]
        df_upload = df_upload.drop(columns=[col for col in columns_to_drop if col in df_upload.columns])
        
        # Convert to records and normalize
        records = df_upload.to_dict("records")
        clean_records = [_normalize_row(record) for record in records]
        
        # Upload in chunks
        chunk_size = 1000
        total_sent = 0
        errors = []
        
        print(f"Uploading {len(clean_records)} records to {table_name}...")
        
        for i in range(0, len(clean_records), chunk_size):
            chunk = clean_records[i:i + chunk_size]
            try:
                result = sb.table(table_name).insert(chunk).execute()
                total_sent += len(chunk)
                print(f"Uploaded chunk {i//chunk_size + 1}/{(len(clean_records)-1)//chunk_size + 1}")
            except Exception as e:
                error_msg = f"Chunk {i//chunk_size + 1}: {str(e)}"
                errors.append(error_msg)
                print(f"Error uploading chunk {i//chunk_size + 1}: {e}")
        
        return {
            "attempted": len(clean_records),
            "sent": total_sent,
            "errors": errors
        }
        
    except Exception as e:
        return {
            "attempted": 0,
            "sent": 0,
            "errors": [f"Upload failed: {str(e)}"]
        }

def get_market_volume_ranks(
    target_date: str,
    api_key: str,
    dat_tickers: Optional[List[str]] = None,
    include_otc: bool = False,
    adjusted: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get market-wide volume ranks and DAT ticker subset.
    
    Args:
        target_date: Date in YYYY-MM-DD format
        api_key: Polygon.io API key
        dat_tickers: List of DAT tickers to extract (optional)
        include_otc: Whether to include OTC securities
        adjusted: Whether to adjust for splits
        
    Returns:
        Tuple of (full_ranks_df, dat_ranks_df)
    """
    print(f"Fetching market data for {target_date}...")
    
    # Get market data
    df = get_polygon_market_summary_for_date(target_date, api_key)
    
    if df.empty:
        print("No market data retrieved.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"Retrieved {len(df)} stocks, computing ranks...")
    
    # Add ranking columns
    df_ranks = add_volume_ranks(df)
    
    # Get DAT ticker subset if requested
    dat_ranks = pd.DataFrame()
    if dat_tickers:
        dat_ranks = get_dat_ticker_ranks(df_ranks, dat_tickers)
        print(f"Found {len(dat_ranks)} DAT tickers in market data")
    
    return df_ranks, dat_ranks

if __name__ == "__main__":
    """
    Test the scraper by fetching yesterday's market summary data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute market-wide volume ranks and upload to Supabase.")
    parser.add_argument("--date", help="YYYY-MM-DD; default=yesterday (UTC)")
    parser.add_argument("--include-otc", default="false", choices=["true", "false"], help="Include OTC? default=false")
    parser.add_argument("--dat", help="Comma-separated tickers to report (default: uses built-in DAT list)")
    parser.add_argument("--table", default="Stock_rankings_daily_pull", help="Supabase table name")
    parser.add_argument("--no-replace", action="store_true", help="Don't replace table contents (append instead)")
    
    args = parser.parse_args()
    
    # Get API key from environment variable with fallback
    api_key = os.getenv("POLYGON_API_KEY", "BZAKLfIESg1ZfdJ0ldE49tplxBcj3jcW")
    if not api_key:
        print("Error: Set POLYGON_API_KEY environment variable")
        exit(1)
    
    # Determine target date
    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    
    include_otc = args.include_otc.lower() == "true"
    dat_list = args.dat.split(",") if args.dat else DEFAULT_DAT_TICKERS
    replace_table = not args.no_replace
    
    print(f"Computing market volume ranks for {target_date} (include_otc={include_otc})")
    
    # Get ranks
    df_ranks, dat_ranks = get_market_volume_ranks(
        target_date=target_date,
        api_key=api_key,
        dat_tickers=dat_list,
        include_otc=include_otc
    )
    
    if df_ranks.empty:
        print("No market data available.")
        exit(1)
    
    print(f"Universe size: {len(df_ranks)}")
    print("Computed ranks by dollar volume and transactions.")
    
    # Upload to Supabase
    if get_supabase is not None:
        try:
            upload_result = upload_rankings_to_supabase(
                df_ranks=dat_ranks,
                table_name=args.table,
                replace_table=replace_table
            )
            
            print(f"\nUpload Results:")
            print(f"  Attempted: {upload_result['attempted']} records")
            print(f"  Sent: {upload_result['sent']} records")
            if upload_result['errors']:
                print(f"  Errors: {len(upload_result['errors'])}")
                for error in upload_result['errors']:
                    print(f"    - {error}")
            else:
                print(f"  Success: All records uploaded successfully!")
                
        except Exception as e:
            print(f"Error uploading to Supabase: {e}")
            print("Falling back to CSV export...")
            dat_ranks.to_csv(f"dat_ranks_{target_date}.csv", index=False)
            print(f"Saved to dat_ranks_{target_date}.csv")
    else:
        print("Supabase not available, saving to CSV...")
        dat_ranks.to_csv(f"dat_ranks_{target_date}.csv", index=False)
        print(f"Saved to dat_ranks_{target_date}.csv")
    
    # Display results
    if not dat_ranks.empty:
        print(f"\nDAT ticker ranks with neighbors:")
        print(dat_ranks[["ticker", "is_dat", "rank_dollar_volume", "rank_transactions", "volume", "dollar_volume", "transactions"]].to_string(index=False))
    else:
        print("No DAT tickers found in market data.")
