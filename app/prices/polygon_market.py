#!/usr/bin/env python3
"""
Polygon.io Daily Market Summary scraper.

Adds: Full Market Snapshot (only ticker, todaysChangePerc) -> ranks & neighbors
Uses Yahoo Finance screener for most active stocks instead of Polygon volume data.
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone, date, timedelta
import os

# Import Supabase client
try:
    from app.clients.supabase_append import get_supabase, _normalize_row, replace_stock_rankings_table
except ImportError:
    print("Warning: Supabase client not available. CSV export will be used instead.")
    get_supabase = None
    _normalize_row = None
    replace_stock_rankings_table = None

# Polygon.io configuration
USER_AGENT = "Ally Zach <ally@panteracapital.com>"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

# Polygon.io API endpoints
POLYGON_MARKET_SUMMARY_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks"
POLYGON_SNAPSHOT_URL = "https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers"  # NEW

# Yahoo Finance screener for most active stocks
def get_yahoo_most_active_stocks(min_volume=0, count=250, offset=0):
    """
    Fetch Yahoo Finance screener results (most active stocks).
    Returns DataFrame with columns: Ticker, Name, Price, Volume, MarketCap
    """
    url = "https://query2.finance.yahoo.com/v1/finance/screener/predefined/saved"
    params = {
        "count": count,
        "offset": offset,
        "scrIds": "most_actives"
    }

    r = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    data = r.json()

    # Navigate to the quotes list
    try:
        quotes = data["finance"]["result"][0]["quotes"]
    except (KeyError, IndexError):
        raise ValueError("Unexpected JSON structure from Yahoo. Check endpoint response.")

    # Build DataFrame
    df = pd.DataFrame(quotes)
    df = df[["symbol", "shortName", "regularMarketPrice", "regularMarketVolume", "marketCap"]]
    df.rename(columns={
        "symbol": "Ticker",
        "shortName": "Name",
        "regularMarketPrice": "Price",
        "regularMarketVolume": "Volume",
        "marketCap": "MarketCap"
    }, inplace=True)

    # Filter and sort
    df = df[df["Volume"] > min_volume]
    df = df.sort_values("Volume", ascending=False).reset_index(drop=True)

    return df

def get_with_retry(url: str, params: Dict, max_retries: int = 3, backoff: float = 1.5, headers=None) -> Optional[requests.Response]:
    """
    Make HTTP request with retry logic and politeness delays.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            time.sleep(0.1)  # Be polite to Polygon.io
            resp = requests.get(url, params=params, headers=headers or HEADERS, timeout=30)
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
    Fetch yesterday's grouped-daily market data (OHLCV, vwap, transactions).
    """
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    target_date = yesterday.isoformat()
    if not api_key:
        print("Error: Polygon.io API key is required")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
    
    params = {
        "adjusted": str(adjusted).lower(),
        "include_otc": str(include_otc).lower(),
    }
    headers_with_key = HEADERS.copy()
    headers_with_key["Authorization"] = f"Bearer {api_key}"
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
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
        
        rows = []
        for item in results:
            timestamp_ms = item.get("t", 0)
            trade_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date() if timestamp_ms else yesterday
            rows.append({
                "ticker": (item.get("T") or "").upper(),
                "date": trade_date.isoformat(),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v"),
                "vwap": item.get("vw"),
                "transactions": item.get("n"),
            })
        
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["ticker", "date", "close"])
        df = df.sort_values("ticker").reset_index(drop=True)
        print(f"Retrieved {len(df)} stock records for {target_date}")
        return df
    except Exception as e:
        print(f"Error fetching market summary: {e}")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])

def get_polygon_market_summary_for_date(target_date: str, api_key: str) -> pd.DataFrame:
    """
    Fetch grouped-daily market data for a specific date.
    """
    if not api_key:
        print("Error: Polygon.io API key is required")
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
    
    params = {"adjusted": "true", "include_otc": "false"}
    headers_with_key = HEADERS.copy()
    headers_with_key["Authorization"] = f"Bearer {api_key}"
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
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "volume", "vwap", "transactions"])
        
        rows = []
        for item in results:
            timestamp_ms = item.get("t", 0)
            trade_date = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).date() if timestamp_ms else target_date
            rows.append({
                "ticker": (item.get("T") or "").upper(),
                "date": str(trade_date),
                "open": item.get("o"),
                "high": item.get("h"),
                "low": item.get("l"),
                "close": item.get("c"),
                "volume": item.get("v"),
                "vwap": item.get("vw"),
                "transactions": item.get("n"),
            })
        
        df = pd.DataFrame(rows)
        df = df.dropna(subset=["ticker", "date", "close"])
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

# ---------------------- DAT Ticker Configuration ----------------------
DEFAULT_DAT_TICKERS = [
    "MSTR","CEP","SMLR","NAKA","SQNS","BMNR","SBET","ETHZ","BTCS","BTBT","GAME","DFDV","UPXI","HSDT","FORD",
    "ETHM","STSS","FGNX","STKE","MARA","DJT","GLXY","CLSK","BRR","GME","EMPD","CORZ","FLD","USBC","LMFA",
    "DEFT","GNS","BTCM","ICG","COSM","KIDZ"
]

# ---------------------- Ranking Functions ----------------------
def add_volume_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks by dollar volume and volume rank (1 = highest).
    Works with Yahoo Finance data (Ticker, Price, Volume, MarketCap) or Polygon data (ticker, close, volume).
    """
    if df.empty:
        return df
    df = df.copy()
    
    # Handle both Yahoo Finance (uppercase) and Polygon (lowercase) column names
    if "Ticker" in df.columns:
        df["ticker"] = df["Ticker"].astype(str).str.upper()
    if "Price" in df.columns:
        df["close"] = pd.to_numeric(df["Price"], errors="coerce")
    if "Volume" in df.columns:
        df["volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    if "MarketCap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["MarketCap"], errors="coerce")
    
    # Ensure required columns exist
    if "volume" not in df.columns:
        df["volume"] = 0
    if "close" not in df.columns:
        df["close"] = 0.0
    if "transactions" not in df.columns:
        df["transactions"] = 0  # Yahoo Finance doesn't provide transactions
    
    # Normalize numeric columns
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0)
    
    # Calculate dollar volume
    df["dollar_volume"] = df["close"] * df["volume"]
    
    # Add date if missing (use today for Yahoo Finance data)
    if "date" not in df.columns:
        df["date"] = datetime.now(timezone.utc).date().isoformat()
    
    # Rank by dollar volume and volume (1 = highest)
    df["rank_dollar_volume"] = df["dollar_volume"].rank(method="min", ascending=False).astype(int)
    df["rank_trading_volume"] = df["volume"].rank(method="min", ascending=False).astype(int)
    
    # Rank transactions if available (from Polygon)
    if "transactions" in df.columns and df["transactions"].notna().any() and df["transactions"].sum() > 0:
        df["rank_transactions"] = df["transactions"].rank(method="min", ascending=False).astype(int)
    else:
        df["rank_transactions"] = None
    
    df["universe_size"] = len(df)
    return df

### NEW: snapshot change% fetch + ranks
def fetch_snapshot_change_perc(api_key: str, include_otc: bool = False, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Pull ONLY ticker and todaysChangePerc from the Full Market Snapshot endpoint (paged).
    """
    headers_with_key = HEADERS.copy()
    headers_with_key["Authorization"] = f"Bearer {api_key}"
    params = {"include_otc": str(include_otc).lower()}
    if tickers:
        params["tickers"] = ",".join([t.strip().upper() for t in tickers if t.strip()])

    url = POLYGON_SNAPSHOT_URL
    rows = []
    while True:
        resp = get_with_retry(url, params, headers=headers_with_key, max_retries=6, backoff=1.6)
        if not resp:
            break
        data = resp.json() or {}
        if data.get("status") != "OK":
            break
        for it in data.get("tickers", []) or []:
            rows.append({
                "ticker": (it.get("ticker") or "").upper(),
                "todays_change_perc": it.get("todaysChangePerc"),
            })
        next_url = data.get("next_url") or data.get("nextPagePath")
        if not next_url:
            break
        # next_url can be absolute or relative; send auth header again
        url = next_url if next_url.startswith("http") else f"https://api.polygon.io{next_url}"
        params = {}  # cursor is embedded in next_url; auth remains in headers
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["todays_change_perc"] = pd.to_numeric(df["todays_change_perc"], errors="coerce")
    df = df.dropna(subset=["ticker"]).reset_index(drop=True)
    return df

def add_change_ranks(df_change: pd.DataFrame) -> pd.DataFrame:
    """
    Rank by today's % change (1 = biggest gainer).
    """
    if df_change.empty:
        return df_change
    out = df_change.copy()
    n = len(out)
    tmp = out["todays_change_perc"].fillna(float("-inf"))
    out["rank_change_perc"] = tmp.rank(method="min", ascending=False).astype(int)
    out["pct_change_perc"] = 1.0 - (out["rank_change_perc"] - 1) / max(n - 1, 1)
    out["universe_size_change"] = n
    return out

def get_dat_ticker_ranks(df_ranks: pd.DataFrame, dat_tickers: List[str]) -> pd.DataFrame:
    """
    Extract ranking data for DAT tickers plus ranking neighbors (±1) around the DAT leader
    for each metric: dollar_volume, trading_volume, transactions, and change%.
    For change%, when all DATs are negative, finds neighbors based on similar change% values.
    """
    if not dat_tickers or df_ranks.empty:
        return pd.DataFrame()

    dat_tickers = [t.strip().upper() for t in dat_tickers if t.strip()]
    dat_df = df_ranks[df_ranks["ticker"].isin(dat_tickers)].copy()
    if dat_df.empty:
        return pd.DataFrame()

    dat_df["is_dat"] = True

    # Leaders (best ranks) in each metric
    dat_dollar_leader_rank = dat_df["rank_dollar_volume"].min() if "rank_dollar_volume" in dat_df.columns and not dat_df["rank_dollar_volume"].isna().all() else None
    dat_vol_leader_rank = dat_df["rank_trading_volume"].min() if "rank_trading_volume" in dat_df.columns and not dat_df["rank_trading_volume"].isna().all() else None
    dat_txn_leader_rank = dat_df["rank_transactions"].min() if "rank_transactions" in dat_df.columns and not dat_df["rank_transactions"].isna().all() else None
    dat_change_leader_rank = dat_df["rank_change_perc"].min() if "rank_change_perc" in dat_df.columns and not dat_df["rank_change_perc"].isna().all() else None

    neighbors = []

    # Dollar volume neighbors ±1
    if dat_dollar_leader_rank is not None:
        dollar_neighbors = df_ranks[
            (df_ranks["rank_dollar_volume"] >= dat_dollar_leader_rank - 1) &
            (df_ranks["rank_dollar_volume"] <= dat_dollar_leader_rank + 1)
        ].copy()
        dollar_neighbors["is_dat"] = dollar_neighbors["ticker"].isin(dat_tickers)
        neighbors.append(dollar_neighbors)

    # Trading volume neighbors ±1 (NEW - from Yahoo Finance)
    if dat_vol_leader_rank is not None:
        vol_neighbors = df_ranks[
            (df_ranks["rank_trading_volume"] >= dat_vol_leader_rank - 1) &
            (df_ranks["rank_trading_volume"] <= dat_vol_leader_rank + 1)
        ].copy()
        vol_neighbors["is_dat"] = vol_neighbors["ticker"].isin(dat_tickers)
        neighbors.append(vol_neighbors)

    # Transactions neighbors ±1
    if dat_txn_leader_rank is not None:
        txn_neighbors = df_ranks[
            (df_ranks["rank_transactions"] >= dat_txn_leader_rank - 1) &
            (df_ranks["rank_transactions"] <= dat_txn_leader_rank + 1)
        ].copy()
        txn_neighbors["is_dat"] = txn_neighbors["ticker"].isin(dat_tickers)
        neighbors.append(txn_neighbors)

    # Change% neighbors ±1
    # Special handling: if all DATs are negative, find neighbors by similar change% values rather than just rank
    if dat_change_leader_rank is not None:
        # Get the DAT leader's change% value
        dat_change_leader = dat_df.loc[dat_df["rank_change_perc"].idxmin()] if "rank_change_perc" in dat_df.columns else None
        
        if dat_change_leader is not None and "todays_change_perc" in dat_change_leader:
            leader_change_pct = dat_change_leader["todays_change_perc"]
            
            # Check if all DATs have negative change%
            all_dat_changes = dat_df["todays_change_perc"].dropna()
            all_negative = len(all_dat_changes) > 0 and (all_dat_changes < 0).all()
            
            if all_negative and pd.notna(leader_change_pct):
                # When all DATs are negative, find neighbors within ±0.5% change (similar performance)
                # This ensures we get meaningful neighbors even when ranks are very high
                change_threshold = 0.5  # ±0.5 percentage points
                chg_neighbors = df_ranks[
                    (df_ranks["todays_change_perc"].notna()) &
                    (df_ranks["todays_change_perc"] >= leader_change_pct - change_threshold) &
                    (df_ranks["todays_change_perc"] <= leader_change_pct + change_threshold)
                ].copy()
                
                # Also include rank neighbors as fallback/backup
                rank_neighbors = df_ranks[
                    (df_ranks["rank_change_perc"] >= dat_change_leader_rank - 1) &
                    (df_ranks["rank_change_perc"] <= dat_change_leader_rank + 1)
                ].copy()
                
                # Combine both approaches and deduplicate
                chg_neighbors = pd.concat([chg_neighbors, rank_neighbors]).drop_duplicates(subset=["ticker"])
            else:
                # Normal case: use rank neighbors
                chg_neighbors = df_ranks[
                    (df_ranks["rank_change_perc"] >= dat_change_leader_rank - 1) &
                    (df_ranks["rank_change_perc"] <= dat_change_leader_rank + 1)
                ].copy()
            
            chg_neighbors["is_dat"] = chg_neighbors["ticker"].isin(dat_tickers)
            neighbors.append(chg_neighbors)
        else:
            # Fallback to rank neighbors if change% not available
            chg_neighbors = df_ranks[
                (df_ranks["rank_change_perc"] >= dat_change_leader_rank - 1) &
                (df_ranks["rank_change_perc"] <= dat_change_leader_rank + 1)
            ].copy()
            chg_neighbors["is_dat"] = chg_neighbors["ticker"].isin(dat_tickers)
            neighbors.append(chg_neighbors)

    if not neighbors:
        return dat_df.sort_values("ticker").reset_index(drop=True)

    all_neighbors = pd.concat(neighbors, ignore_index=True).drop_duplicates(subset=["ticker"])
    all_neighbors = all_neighbors.sort_values("ticker").reset_index(drop=True)
    return all_neighbors

def upload_rankings_to_supabase(
    df_ranks: pd.DataFrame,
    table_name: str = "Stock_rankings_daily_pull",
    replace_table: bool = True
) -> Dict[str, int]:
    """
    Upload ranking data to Supabase table using dedicated function.
    """
    if replace_stock_rankings_table is None:
        raise RuntimeError("Supabase client not available")

    if df_ranks.empty:
        return {"attempted": 0, "sent": 0, "errors": []}

    if replace_table:
        return replace_stock_rankings_table(table_name, df_ranks)
    else:
        from app.clients.supabase_append import insert_stock_rankings_df
        return insert_stock_rankings_df(table_name, df_ranks)

def get_market_volume_ranks(
    target_date: str,
    api_key: str,
    dat_tickers: Optional[List[str]] = None,
    include_otc: bool = False,
    adjusted: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get market-wide ranks using:
    - Yahoo Finance: volume data (most active stocks)
    - Polygon: transaction data and percent change data
    """
    print(f"Fetching most active stocks from Yahoo Finance...")
    df_yahoo = get_yahoo_most_active_stocks(min_volume=0)  # Get all stocks, no volume filter
    
    if df_yahoo.empty:
        print("No stocks retrieved from Yahoo Finance.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Retrieved {len(df_yahoo)} stocks from Yahoo Finance")
    
    # Get transaction data from Polygon for the same tickers
    print(f"Fetching transaction data from Polygon for {target_date}...")
    ticker_list = df_yahoo["Ticker"].tolist()
    df_polygon = get_polygon_market_summary_for_date(target_date, api_key)
    
    # Merge Polygon transaction data with Yahoo Finance volume data
    if not df_polygon.empty:
        # Keep only tickers that are in Yahoo Finance results
        df_polygon_filtered = df_polygon[df_polygon["ticker"].isin([t.upper() for t in ticker_list])].copy()
        
        # Merge transactions from Polygon
        df_yahoo["ticker"] = df_yahoo["Ticker"].str.upper()
        df_merged = df_yahoo.merge(
            df_polygon_filtered[["ticker", "transactions"]],
            on="ticker",
            how="left"
        )
        print(f"  Merged transaction data for {df_merged['transactions'].notna().sum()} tickers from Polygon")
    else:
        print("  Warning: No Polygon transaction data available")
        df_merged = df_yahoo.copy()
        df_merged["ticker"] = df_merged["Ticker"].str.upper()
        df_merged["transactions"] = None

    print("Computing ranks...")
    df_ranks = add_volume_ranks(df_merged)

    # Fetch change% (only ticker + todaysChangePerc) from Polygon snapshot and merge ranks
    print("Fetching snapshot change% (todaysChangePerc) from Polygon...")
    df_change = fetch_snapshot_change_perc(api_key, include_otc=include_otc, tickers=None)
    if not df_change.empty:
        df_change_ranked = add_change_ranks(df_change)[["ticker", "todays_change_perc", "rank_change_perc"]]
        df_ranks = df_ranks.merge(df_change_ranked, on="ticker", how="left")
    else:
        print("Snapshot change% unavailable; continuing without change% ranks.")

    dat_ranks = pd.DataFrame()
    if dat_tickers:
        dat_ranks = get_dat_ticker_ranks(df_ranks, dat_tickers)
        print(f"Found {len(dat_ranks)} DAT tickers (with neighbors) in market data")

    return df_ranks, dat_ranks

if __name__ == "__main__":
    """
    CLI: compute ranks for grouped-daily; augment with snapshot change%; output / upload.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Compute market-wide ranks (dollar volume, transactions, change%) and upload.")
    parser.add_argument("--date", help="YYYY-MM-DD; default=yesterday (UTC)")
    parser.add_argument("--include-otc", default="false", choices=["true", "false"], help="Include OTC? default=false")
    parser.add_argument("--dat", help="Comma-separated tickers to report (default: uses built-in DAT list)")
    parser.add_argument("--table", default="Stock_rankings_daily_pull", help="Supabase table name")
    parser.add_argument("--no-replace", action="store_true", help="Don't replace table contents (append instead)")
    args = parser.parse_args()

    api_key = os.getenv("POLYGON_API_KEY", "BZAKLfIESg1ZfdJ0ldE49tplxBcj3jcW")
    if not api_key:
        print("Error: Set POLYGON_API_KEY environment variable")
        exit(1)

    target_date = args.date or (datetime.now(timezone.utc) - timedelta(days=1)).date().isoformat()
    include_otc = args.include_otc.lower() == "true"
    dat_list = [t.strip().upper() for t in (args.dat.split(",") if args.dat else DEFAULT_DAT_TICKERS) if t.strip()]
    replace_table = not args.no_replace

    print(f"Computing market ranks for {target_date} (include_otc={include_otc})")
    df_ranks, dat_ranks = get_market_volume_ranks(
        target_date=target_date,
        api_key=api_key,
        dat_tickers=dat_list,
        include_otc=include_otc
    )

    if df_ranks.empty:
        print("No market data available.")
        exit(1)
    
    # Save CSVs (excluding OHLC, VWAP, volume, universe_size)
    columns_to_drop_csv = ["open", "high", "low", "close", "vwap", "volume", "universe_size"]
    
    df_ranks_clean = df_ranks.drop(columns=[col for col in columns_to_drop_csv if col in df_ranks.columns], errors="ignore")
    df_ranks_clean.sort_values(["rank_dollar_volume","rank_transactions"]).to_csv(f"all_ranks_{target_date}.csv", index=False)
    print(f"Saved full ranks → all_ranks_{target_date}.csv")

    if not dat_ranks.empty:
        dat_ranks_clean = dat_ranks.drop(columns=[col for col in columns_to_drop_csv if col in dat_ranks.columns], errors="ignore")
        dat_ranks_clean.to_csv(f"dat_ranks_{target_date}.csv", index=False)
        print(f"Saved DAT ranks → dat_ranks_{target_date}.csv")

    # Upload only DAT subset as before
    if get_supabase is not None and not dat_ranks.empty:
        try:
            upload_result = upload_rankings_to_supabase(
                df_ranks=dat_ranks,
                table_name=args.table,
                replace_table=replace_table
            )
            print(f"\nUpload Results:\n  Attempted: {upload_result['attempted']}\n  Sent: {upload_result['sent']}")
            if upload_result["errors"]:
                print("  Errors:")
                for e in upload_result["errors"]:
                    print("   -", e)
            else:
                print("  Success: All records uploaded successfully!")
        except Exception as e:
            print(f"Error uploading to Supabase: {e}")
            print("Falling back to CSV export...")
            dat_ranks_clean = dat_ranks.drop(columns=[col for col in columns_to_drop_csv if col in dat_ranks.columns], errors="ignore")
            dat_ranks_clean.to_csv(f"dat_ranks_{target_date}.csv", index=False)
            print(f"Saved to dat_ranks_{target_date}.csv")

    # Console preview
    if not dat_ranks.empty:
        cols = ["ticker","is_dat",
                "rank_dollar_volume","rank_transactions","rank_change_perc",
                "todays_change_perc","dollar_volume","transactions"]
        have = [c for c in cols if c in dat_ranks.columns]
        print("\nDAT ranks with neighbors (sample):")
        print(dat_ranks[have].head(20).to_string(index=False))
    else:
        print("No DAT tickers found in market data.")
