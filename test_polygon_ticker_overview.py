#!/usr/bin/env python3
"""
Test script for the new Polygon ticker overview daily functionality
"""

import sys
import os
from datetime import date, timedelta

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.prices.polygon_ticker_overview import fetch_polygon_ticker_overview_daily, upload_polygon_ticker_overview_df

def main():
    print("Testing Polygon Ticker Overview Daily Functionality")
    print("=" * 60)
    
    # Test with a small subset of tickers
    test_tickers = ["MSTR", "UPXI", "BTCS"]
    yesterday = date.today() - timedelta(days=1)
    
    print(f"Testing with tickers: {test_tickers}")
    print(f"Target date: {yesterday}")
    print("-" * 60)
    
    try:
        # Fetch data
        print("Fetching ticker overview data...")
        df = fetch_polygon_ticker_overview_daily(test_tickers, yesterday)
        
        if df.empty:
            print("No data retrieved")
            return
        
        print(f"\nRetrieved {len(df)} records:")
        print(df.to_string(index=False))
        
        # Test upload (dry run - comment out the actual upload for testing)
        print(f"\nWould upload to Polygon_outstanding_raw table:")
        print(f"  - {len(df)} records")
        print(f"  - Columns: {list(df.columns)}")
        
        # Uncomment the line below to actually test the upload
        # stats = upload_polygon_ticker_overview_df(df, "Polygon_outstanding_raw")
        # print(f"Upload stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
