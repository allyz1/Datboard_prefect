#!/usr/bin/env python3
"""
Polygon.io Ticker Overview Daily Fetcher

Fetches ticker overview data from Polygon.io API for yesterday's date
and uploads to Polygon_outstanding_raw table.
"""

import os
import sys
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional
import requests
import pandas as pd
from prefect import task, get_run_logger

# Import supabase upload function
from ..clients.supabase_append import insert_polygon_outstanding_raw_df

# Configuration
API_KEY = "BZAKLfIESg1ZfdJ0ldE49tplxBcj3jcW"
BASE_URL = "https://api.polygon.io/v3/reference/tickers"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "User-Agent": "DATs-Polygon-Client/1.0"
}

# Rate limiting configuration (1 minute delay every 5 requests)
RATE_LIMIT_DELAY = 1.0   # Base delay between requests (1 second)
RATE_LIMIT_BURST = 5     # Number of requests before longer delay
BURST_DELAY = 60.0       # Longer delay after burst (1 minute)
MAX_RETRIES = 3
RETRY_DELAY = 2.0

# Default tickers from the main flow
DEFAULT_TICKERS = [
    "MSTR", "CEP", "SMLR", "NAKA", "SQNS", "BMNR", "SBET", "ETHZ", "BTCS", "BTBT",
    "GAME", "DFDV", "UPXI", "HSDT", "FORD", "ETHM", "STSS", "FGNX", "STKE", "MARA",
    "DJT", "GLXY", "CLSK", "BRR", "GME", "EMPD", "CORZ", "FLD", "USBC", "LMFA",
    "DEFT", "GNS", "BTCM", "ICG", "COSM", "KIDZ"
]

class DailyPolygonTickerClient:
    """Client for daily ticker overview fetching with rate limiting"""
    
    def __init__(self):
        self.request_count = 0
        self.last_request_time = 0
        self.logger = get_run_logger()
    
    def _handle_rate_limit(self):
        """Handle rate limiting with burst detection"""
        # Rate limiting disabled - process requests immediately
        self.request_count += 1
    
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make HTTP request with retry logic and rate limiting"""
        for attempt in range(MAX_RETRIES):
            try:
                self._handle_rate_limit()
                
                response = requests.get(url, headers=HEADERS, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = RETRY_DELAY * (attempt + 1)
                    self.logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(wait_time)
                    continue
                elif response.status_code in [500, 502, 503, 504]:
                    # Server error - retry
                    wait_time = RETRY_DELAY * (attempt + 1)
                    self.logger.warning(f"Server error ({response.status_code}). Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                    time.sleep(wait_time)
                    continue
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait_time = RETRY_DELAY * (attempt + 1)
                self.logger.warning(f"Request failed: {e}. Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}")
                time.sleep(wait_time)
        
        raise Exception(f"Failed to make request after {MAX_RETRIES} attempts")
    
    def get_ticker_overview(self, ticker: str, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Get ticker overview data for a specific ticker and date
        
        Args:
            ticker: Stock ticker symbol (e.g., 'MSTR', 'AAPL')
            date_str: Date in YYYY-MM-DD format (optional, defaults to most recent)
            
        Returns:
            Dictionary containing ticker overview data
        """
        url = f"{BASE_URL}/{ticker}"
        params = {}
        
        if date_str:
            params['date'] = date_str
        
        self.logger.info(f"Fetching {ticker} overview" + (f" for {date_str}" if date_str else " (latest)"))
        
        try:
            data = self._make_request(url, params)
            return data.get('results', {})
        except Exception as e:
            self.logger.error(f"Error fetching {ticker} overview: {e}")
            return {}
    
    def fetch_yesterday_for_tickers(self, tickers: List[str], target_date: Optional[date] = None) -> pd.DataFrame:
        """
        Fetch ticker overview data for yesterday (or specified date) for all tickers
        
        Args:
            tickers: List of ticker symbols
            target_date: Date to fetch (defaults to yesterday)
            
        Returns:
            DataFrame with ticker overview data
        """
        if target_date is None:
            target_date = date.today() - timedelta(days=1)
        
        date_str = target_date.isoformat()
        self.logger.info(f"Fetching ticker overview data for {len(tickers)} tickers on {date_str}")
        
        results = []
        for i, ticker in enumerate(tickers, 1):
            try:
                self.logger.info(f"[{i}/{len(tickers)}] Fetching {ticker} for {date_str}")
                
                overview = self.get_ticker_overview(ticker, date_str)
                
                if overview:
                    # Add our tracking fields
                    overview['query_date'] = date_str
                    overview['ticker'] = ticker
                    results.append(overview)
                    self.logger.info(f"✓ {ticker}: Market cap ${overview.get('market_cap', 'N/A'):,}")
                else:
                    self.logger.warning(f"✗ {ticker}: No data available for {date_str}")
                    
            except Exception as e:
                self.logger.error(f"Error fetching {ticker}: {e}")
                continue
        
        if not results:
            self.logger.warning("No data retrieved for any tickers")
            return pd.DataFrame()
        
        # Convert to DataFrame and keep only desired fields
        df = pd.DataFrame(results)
        
        # Define the specific fields we want to keep
        desired_fields = [
            'query_date', 'ticker', 'cik', 'market_cap', 
            'share_class_shares_outstanding', 'weighted_shares_outstanding'
        ]
        
        # Filter to only desired fields
        available_fields = [field for field in desired_fields if field in df.columns]
        df_filtered = df[available_fields].copy()
        
        # Fill missing fields with empty values
        for field in desired_fields:
            if field not in df_filtered.columns:
                df_filtered[field] = ''
        
        # Reorder columns
        df_filtered = df_filtered[desired_fields]
        
        self.logger.info(f"Successfully fetched {len(df_filtered)} records")
        return df_filtered

@task(retries=2, retry_delay_seconds=60)
def fetch_polygon_ticker_overview_daily(
    tickers: List[str] = None, 
    target_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Fetch ticker overview data for yesterday (or specified date) for all tickers
    
    Args:
        tickers: List of ticker symbols (defaults to DEFAULT_TICKERS)
        target_date: Date to fetch (defaults to yesterday)
        
    Returns:
        DataFrame with ticker overview data
    """
    logger = get_run_logger()
    
    if tickers is None:
        tickers = DEFAULT_TICKERS
    
    logger.info(f"Starting Polygon ticker overview fetch for {len(tickers)} tickers")
    
    client = DailyPolygonTickerClient()
    df = client.fetch_yesterday_for_tickers(tickers, target_date)
    
    if df.empty:
        logger.warning("No ticker overview data retrieved")
        return df
    
    logger.info(f"Retrieved {len(df)} ticker overview records")
    return df

@task(retries=2, retry_delay_seconds=60)
def upload_polygon_ticker_overview_df(df: pd.DataFrame, table: str = "Polygon_outstanding_raw") -> dict:
    """
    Upload ticker overview DataFrame to Supabase table
    
    Args:
        df: DataFrame with ticker overview data
        table: Target table name (defaults to "Polygon_outstanding_raw")
        
    Returns:
        Upload statistics dictionary
    """
    logger = get_run_logger()
    
    if df.empty:
        logger.warning("No data to upload")
        return {"attempted": 0, "sent": 0, "errors": []}
    
    logger.info(f"Uploading {len(df)} records to {table}")
    
    try:
        # Use the dedicated Polygon upload function
        stats = insert_polygon_outstanding_raw_df(
            table=table,
            df=df,
            chunk_size=100
        )
        
        logger.info(f"Upload completed: {stats}")
        
        # Log any errors that occurred
        if stats.get("errors"):
            for error in stats["errors"]:
                logger.error(f"Upload error: {error}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"attempted": len(df), "sent": 0, "errors": [str(e)]}

def daterange(start_date: date, end_date: date):
    """Generate dates from start_date to end_date (inclusive)"""
    d = start_date
    while d <= end_date:
        yield d
        d += timedelta(days=1)

def main():
    """Standalone execution for testing with single date OR date range"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Polygon ticker overview data")
    # existing single-date flag (still supported)
    parser.add_argument("--date", help="Date to fetch (YYYY-MM-DD), defaults to yesterday")
    
    # NEW: date range flags
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to fetch")
    parser.add_argument("--table", default="Polygon_outstanding_raw", help="Target table name")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but don't upload")
    
    # NEW: output CSV path
    parser.add_argument("--out", help="Path to output CSV (when using --start/--end or --date)")
    
    args = parser.parse_args()
    
    # Parse dates
    single_date = None
    start_date = None
    end_date = None
    
    if args.date and (args.start or args.end):
        print("Error: use either --date OR --start/--end, not both.")
        sys.exit(1)
    
    if args.date:
        try:
            single_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD")
            sys.exit(1)
    
    if args.start or args.end:
        if not (args.start and args.end):
            print("Error: --start and --end must be provided together.")
            sys.exit(1)
        try:
            start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        except ValueError:
            print("Error: Invalid date format for --start/--end. Use YYYY-MM-DD")
            sys.exit(1)
        if end_date < start_date:
            print("Error: --end must be on/after --start.")
            sys.exit(1)
    
    # Use provided tickers or defaults
    tickers = args.tickers or DEFAULT_TICKERS
    
    # Helper to safely call the Prefect task synchronously
    # Check if it's a task and get the underlying function
    fetch_fn = getattr(fetch_polygon_ticker_overview_daily, "fn", fetch_polygon_ticker_overview_daily)
    
    # Collect DataFrames
    dfs = []
    
    if single_date:
        print(f"Fetching {len(tickers)} tickers for {single_date}")
        df = fetch_fn(tickers, single_date)
        if not df.empty:
            dfs.append(df)
    else:
        # Range or fallback to yesterday
        if start_date and end_date:
            print(f"Fetching {len(tickers)} tickers for range {start_date} → {end_date}")
            for d in daterange(start_date, end_date):
                print(f"- {d}")
                df = fetch_fn(tickers, d)
                if not df.empty:
                    dfs.append(df)
        else:
            # original behavior: yesterday
            print(f"Fetching {len(tickers)} tickers (target date: yesterday)")
            df = fetch_fn(tickers, None)
            if not df.empty:
                dfs.append(df)
    
    if not dfs:
        print("No data retrieved.")
        return
    
    # Concatenate all results
    out_df = pd.concat(dfs, ignore_index=True)
    
    # Ensure numeric columns are typed properly (avoid empty-string issues)
    for col in ["market_cap", "share_class_shares_outstanding", "weighted_shares_outstanding"]:
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
    
    # Output CSV if requested (or if a range was used)
    if args.out:
        out_path = args.out
    elif start_date and end_date:
        out_path = f"polygon_overview_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"
    elif single_date:
        out_path = f"polygon_overview_{single_date.isoformat()}.csv"
    else:
        out_path = "polygon_overview_yesterday.csv"
    
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {len(out_df)} rows to {out_path}")
    
    # Upload only if NOT dry-run and only for single day (keep range runs offline by default)
    if not args.dry_run and not (start_date and end_date):
        stats = upload_polygon_ticker_overview_df(out_df, args.table)
        print(f"\nUpload stats: {stats}")
    else:
        print("\nDry run or range mode - data not uploaded")

if __name__ == "__main__":
    main()
