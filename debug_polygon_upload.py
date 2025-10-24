#!/usr/bin/env python3
"""
Debug script for polygon ticker overview upload issues
"""

import os
import sys
import pandas as pd
from datetime import date, timedelta

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_supabase_connection():
    """Test if Supabase connection works"""
    try:
        from app.clients.supabase_append import get_supabase
        sb = get_supabase()
        print("Supabase connection successful")
        return True
    except KeyError as e:
        print(f"Missing environment variable: {e}")
        print("Required: SUPABASE_URL and SUPABASE_SERVICE_KEY")
        return False
    except Exception as e:
        print(f"Supabase connection failed: {e}")
        return False

def test_upload_function():
    """Test the upload function with sample data"""
    try:
        from app.clients.supabase_append import insert_polygon_outstanding_raw_df
        
        # Create sample data
        sample_data = {
            'query_date': [date.today() - timedelta(days=1)],
            'ticker': ['TEST'],
            'cik': ['1234567'],
            'market_cap': [1000000000],
            'share_class_shares_outstanding': [1000000],
            'weighted_shares_outstanding': [1000000]
        }
        df = pd.DataFrame(sample_data)
        
        print(f"Testing upload with sample data:")
        print(df.to_string(index=False))
        
        # Test upload
        result = insert_polygon_outstanding_raw_df(
            table="Polygon_outstanding_raw",
            df=df,
            chunk_size=100
        )
        
        print(f"Upload result: {result}")
        return result
        
    except Exception as e:
        print(f"Upload test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_environment():
    """Check environment variables"""
    print("Environment Variables:")
    print(f"SUPABASE_URL: {'Set' if os.getenv('SUPABASE_URL') else 'Missing'}")
    print(f"SUPABASE_SERVICE_KEY: {'Set' if os.getenv('SUPABASE_SERVICE_KEY') else 'Missing'}")
    print(f"SUPABASE_SERVICE_ROLE_KEY: {'Set' if os.getenv('SUPABASE_SERVICE_ROLE_KEY') else 'Missing'}")
    print(f"SUPABASE_ANON_KEY: {'Set' if os.getenv('SUPABASE_ANON_KEY') else 'Missing'}")

def main():
    print("Polygon Ticker Overview Upload Debug")
    print("=" * 50)
    
    # Check environment
    check_environment()
    print()
    
    # Test Supabase connection
    if not test_supabase_connection():
        print("Cannot proceed without Supabase connection")
        return
    print()
    
    # Test upload function
    result = test_upload_function()
    if result:
        print(f"\nUpload test completed: {result}")
    else:
        print("\nUpload test failed")

if __name__ == "__main__":
    main()
