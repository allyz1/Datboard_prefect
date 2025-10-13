#!/usr/bin/env python3
"""
CoinGecko supply data scraper for ETH, SOL, and BTC.

This module scrapes current supply information from CoinGecko's website
for the three major cryptocurrencies and returns the data in a structured format.
"""

import requests
import pandas as pd
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone
import re

# CoinGecko configuration
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}

# CoinGecko URLs for the three cryptocurrencies
COINGECKO_URLS = {
    "BTC": "https://www.coingecko.com/en/coins/bitcoin",
    "ETH": "https://www.coingecko.com/en/coins/ethereum", 
    "SOL": "https://www.coingecko.com/en/coins/solana",
}

def get_with_retry(url: str, max_retries: int = 3, backoff: float = 1.5) -> Optional[requests.Response]:
    """
    Make HTTP request with retry logic and politeness delays.
    """
    session = requests.Session()
    session.headers.update(HEADERS)
    
    last_err = None
    for attempt in range(max_retries):
        try:
            time.sleep(1.0)  # Be polite to CoinGecko
            resp = session.get(url, timeout=30)
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

def get_coingecko_api_data(assets: List[str] = None) -> pd.DataFrame:
    """
    Alternative method using CoinGecko's free API to get supply data.
    This is more reliable than web scraping.
    """
    if assets is None:
        assets = ["BTC", "ETH", "SOL"]
    
    # CoinGecko API endpoint
    api_url = "https://api.coingecko.com/api/v3/coins/markets"
    
    # Map our asset symbols to CoinGecko IDs
    asset_to_id = {
        "BTC": "bitcoin",
        "ETH": "ethereum", 
        "SOL": "solana"
    }
    
    # Get the CoinGecko IDs for our assets
    coin_ids = [asset_to_id.get(asset, asset.lower()) for asset in assets if asset in asset_to_id]
    
    if not coin_ids:
        return pd.DataFrame(columns=["asset", "circulating_supply", "total_supply", "max_supply", "timestamp", "source_url"])
    
    # API parameters
    params = {
        "vs_currency": "usd",
        "ids": ",".join(coin_ids),
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    
    try:
        print(f"Fetching supply data from CoinGecko API for {assets}")
        response = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        rows = []
        timestamp = datetime.now(timezone.utc).isoformat()
        
        for item in data:
            # Find the corresponding asset symbol
            asset = None
            for sym, cg_id in asset_to_id.items():
                if item["id"] == cg_id:
                    asset = sym
                    break
            
            if not asset:
                continue
                
            row = {
                "asset": asset,
                "circulating_supply": item.get("circulating_supply"),
                "total_supply": item.get("total_supply"),
                "date": datetime.now(timezone.utc).date().isoformat(),
            }
            rows.append(row)
            
            print(f"{asset}: Circulating={item.get('circulating_supply')}, "
                  f"Total={item.get('total_supply')}")
        
        return pd.DataFrame(rows)
        
    except Exception as e:
        print(f"API request failed: {e}")
        return pd.DataFrame(columns=["asset", "circulating_supply", "total_supply", "date"])

def extract_supply_data(html_content: str, asset: str) -> Dict[str, Optional[float]]:
    """
    Extract supply data from CoinGecko HTML content.
    
    Returns dict with keys: circulating_supply, total_supply, max_supply
    """
    result = {
        "circulating_supply": None,
        "total_supply": None, 
        "max_supply": None
    }
    
    # Look for supply data in various formats
    # CoinGecko typically shows supply data in sections like "Market Data" or "Token Stats"
    
    # Pattern 1: Look for "Circulating Supply" text followed by numbers
    circulating_patterns = [
        r'Circulating Supply[^>]*>([^<]*)',
        r'circulating supply[^>]*>([^<]*)',
        r'"circulating_supply":\s*([0-9.]+)',
        r'Circulating Supply[^0-9]*([0-9,]+\.?[0-9]*)',
    ]
    
    # Pattern 2: Look for "Total Supply" 
    total_patterns = [
        r'Total Supply[^>]*>([^<]*)',
        r'total supply[^>]*>([^<]*)',
        r'"total_supply":\s*([0-9.]+)',
        r'Total Supply[^0-9]*([0-9,]+\.?[0-9]*)',
    ]
    
    # Pattern 3: Look for "Max Supply"
    max_patterns = [
        r'Max Supply[^>]*>([^<]*)',
        r'max supply[^>]*>([^<]*)',
        r'"max_supply":\s*([0-9.]+)',
        r'Max Supply[^0-9]*([0-9,]+\.?[0-9]*)',
    ]
    
    def extract_number(text: str) -> Optional[float]:
        """Extract numeric value from text, handling commas and units."""
        if not text:
            return None
        
        # Remove HTML tags and clean up
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        
        # Look for numbers with optional commas and decimals
        number_match = re.search(r'([0-9,]+\.?[0-9]*)', clean_text)
        if number_match:
            number_str = number_match.group(1).replace(',', '')
            try:
                return float(number_str)
            except ValueError:
                pass
        
        return None
    
    # Try to extract circulating supply
    for pattern in circulating_patterns:
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            value = extract_number(match.group(1))
            if value:
                result["circulating_supply"] = value
                break
    
    # Try to extract total supply
    for pattern in total_patterns:
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            value = extract_number(match.group(1))
            if value:
                result["total_supply"] = value
                break
    
    # Try to extract max supply
    for pattern in max_patterns:
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            value = extract_number(match.group(1))
            if value:
                result["max_supply"] = value
                break
    
    return result

def get_coingecko_supply_data(assets: List[str] = None) -> pd.DataFrame:
    """
    Get supply data from CoinGecko for specified assets.
    Tries API first, falls back to web scraping if needed.
    
    Args:
        assets: List of asset symbols to fetch (default: ["BTC", "ETH", "SOL"])
    
    Returns:
        DataFrame with columns: asset, circulating_supply, total_supply, date
    """
    if assets is None:
        assets = ["BTC", "ETH", "SOL"]
    
    # Try API first (more reliable)
    print("Attempting to fetch data from CoinGecko API...")
    api_df = get_coingecko_api_data(assets)
    
    if not api_df.empty:
        print("Successfully fetched data from API")
        return api_df
    
    print("API failed, falling back to web scraping...")
    
    # Fallback to web scraping
    rows = []
    timestamp = datetime.now(timezone.utc).isoformat()
    
    for asset in assets:
        if asset not in COINGECKO_URLS:
            print(f"Warning: No URL configured for asset {asset}")
            continue
            
        url = COINGECKO_URLS[asset]
        print(f"Fetching supply data for {asset} from {url}")
        
        response = get_with_retry(url)
        if not response:
            print(f"Failed to fetch data for {asset}")
            continue
            
        supply_data = extract_supply_data(response.text, asset)
        
        row = {
            "asset": asset,
            "circulating_supply": supply_data["circulating_supply"],
            "total_supply": supply_data["total_supply"],
            "date": datetime.now(timezone.utc).date().isoformat(),
        }
        
        rows.append(row)
        print(f"{asset}: Circulating={supply_data['circulating_supply']}, "
              f"Total={supply_data['total_supply']}")
    
    if not rows:
        return pd.DataFrame(columns=["asset", "circulating_supply", "total_supply", "date"])
    
    df = pd.DataFrame(rows)
    return df

def get_coingecko_supply_data_simple() -> pd.DataFrame:
    """
    Simplified version that fetches BTC, ETH, SOL supply data.
    Convenience function for integration with other scripts.
    """
    return get_coingecko_supply_data(["BTC", "ETH", "SOL"])

if __name__ == "__main__":
    """
    Test the scraper by fetching supply data for all three assets.
    """
    print("Testing CoinGecko supply data scraper...")
    
    df = get_coingecko_supply_data_simple()
    
    if df.empty:
        print("No data retrieved.")
    else:
        print(f"\nRetrieved {len(df)} records:")
        print(df.to_string(index=False))
