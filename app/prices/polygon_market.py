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

if __name__ == "__main__":
    """
    Test the scraper by fetching yesterday's market summary data.
    """
    print("Testing Polygon.io market summary scraper...")
    
    # Get API key from environment variable with fallback
    api_key = os.environ.get("POLYGON_API_KEY", "BZAKLfIESg1ZfdJ0ldE49tplxBcj3jcW")
    
    if not api_key:
        print("Error: No Polygon.io API key available")
        exit(1)
    
    # Test with October 10th, 2025 (should be a trading day)
    test_date = "2025-10-10"
    print(f"Testing with specific date: {test_date}")
    
    # Choose which function to test:
    # Option 1: Basic market data (fast)
    df = get_polygon_market_summary_for_date(test_date, api_key)
    
    # Option 2: Market data with market cap (slower, limited to top 100 by volume)
    # df = get_polygon_market_summary_with_marketcap(test_date, api_key, max_tickers=10)
    
    if df.empty:
        print("No market data retrieved for yesterday.")
    else:
        print(f"\nRetrieved {len(df)} stock records for yesterday:")
        print(f"Date: {df['date'].iloc[0]}")
        print(f"Sample data:")
        print(df.head(10).to_string(index=False))
