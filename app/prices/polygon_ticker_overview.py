#!/usr/bin/env python3
"""
Polygon.io Ticker Overview Daily Fetcher

Fetches ticker overview data from Polygon.io API for yesterday's date
and uploads to Polygon_outstanding_raw table.
"""

import os
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
    "MSTR", "CEP", "SMLR", "NAKA", "BMNR", "SBET", "ETHZ", "BTCS", 
    "SQNS", "BTBT", "DFDV", "UPXI", "HSDT", "FORD"
]

class DailyPolygonTickerClient:
    """Client for daily ticker overview fetching with rate limiting"""
    
    def __init__(self):
        self.request_count = 0
        self.last_request_time = 0
        self.logger = get_run_logger()
    
    def _handle_rate_limit(self):
        """Handle rate limiting with burst detection"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Always wait at least the base delay
        if time_since_last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last)
        
        # Check for burst limit
        if self.request_count > 0 and self.request_count % RATE_LIMIT_BURST == 0:
            self.logger.info(f"Rate limit burst detected after {self.request_count} requests. Waiting {BURST_DELAY}s...")
            time.sleep(BURST_DELAY)
        
        self.last_request_time = time.time()
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
        return stats
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"attempted": len(df), "sent": 0, "errors": [str(e)]}

def main():
    """Standalone execution for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch Polygon ticker overview data for yesterday")
    parser.add_argument("--date", help="Date to fetch (YYYY-MM-DD), defaults to yesterday")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to fetch")
    parser.add_argument("--table", default="Polygon_outstanding_raw", help="Target table name")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but don't upload")
    
    args = parser.parse_args()
    
    # Parse date
    target_date = None
    if args.date:
        try:
            target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            print(f"Error: Invalid date format '{args.date}'. Use YYYY-MM-DD")
            sys.exit(1)
    
    # Use provided tickers or defaults
    tickers = args.tickers or DEFAULT_TICKERS
    
    print(f"Fetching ticker overview data for {len(tickers)} tickers")
    if target_date:
        print(f"Target date: {target_date}")
    else:
        print("Target date: yesterday")
    
    # Fetch data
    df = fetch_polygon_ticker_overview_daily(tickers, target_date)
    
    if df.empty:
        print("No data retrieved")
        return
    
    print(f"\nRetrieved {len(df)} records:")
    print(df.to_string(index=False))
    
    if not args.dry_run:
        # Upload data
        stats = upload_polygon_ticker_overview_df(df, args.table)
        print(f"\nUpload stats: {stats}")
    else:
        print("\nDry run - data not uploaded")

if __name__ == "__main__":
    main()
