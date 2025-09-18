#!/usr/bin/env python3
# ingest_main.py
# Pull DataFrames from sources, concat, and upload to Supabase (skip duplicates).

import pandas as pd
import argparse
import logging


from sequans_bitcoin_api_scraper import get_sequans_holdings_df
from app.crawlers.dfdv_daily import get_dfdv_holdings_df
from app.crawlers.naka_daily import get_nakamoto_holdings_df
from app.crawlers.btcs_daily import get_btcs_eth_holdings_df
from app.pipelines.ETH_SEC_holdings_main import get_sec_eth_holdings_df
from app.pipelines.BTC_SEC_holdings_main import get_sec_btc_holdings_df

from app.clients.supabase_append import concat_and_upload  # from the file above


DEFAULT_TABLE = "Holdings_test"

def main():
    ap = argparse.ArgumentParser(description="Combine holdings DFs and upload to Supabase.")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="Supabase table name")
    ap.add_argument("--update", action="store_true", help="Upsert (update existing) instead of insert-only-if-new")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # 1) Pull dataframes
    df_sqns = get_sequans_holdings_df("SQNS", "BTC")   # date,ticker,asset,total_holdings
    df_dfdv = get_dfdv_holdings_df("DFDV", "SOL")      # date,ticker,asset,total_holdings

    df_naka = get_nakamoto_holdings_df(hours=24, ticker="NAKA", asset="BTC")
    
    df_btcs = get_btcs_eth_holdings_df(year=2025)  # -> date, ticker, asset, total_holdings
    
    
    df_eth = get_sec_eth_holdings_df(
        tickers="BMNR,SBET,BTBT,ETHZ",
        hours_back=24,
        forms=("8-K","10-K","10-Q"),
        verbose=False) 
    
    df_btc = get_sec_btc_holdings_df(
        tickers="MSTR,CEP,NAKA,SMLR",
        hours_back=24,
        forms=("8-K","10-K","10-Q"),
        verbose=False)
    
    # 2) Concat (add more sources here as needed)
    frames = [df_sqns, df_dfdv, df_naka, df_btcs, df_eth]
    # You can preview combined: print(pd.concat(frames).head())

    # 3) Upload (skip duplicates by generated key)
    result = concat_and_upload(args.table, frames, do_update=args.update)
    print(f"Attempted to send {result['attempted']} rows (duplicates auto-skipped unless --update).")

if __name__ == "__main__":
    main()

