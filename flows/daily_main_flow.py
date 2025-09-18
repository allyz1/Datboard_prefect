# flows/daily_main_flow.py
from prefect import flow, task, get_run_logger
import pandas as pd
from app import config  # loads .env locally; in prod set real env vars
from app.clients.supabase_append import concat_and_upload

# Producers
from app.crawlers.sequans_bitcoin_api_scraper import get_sequans_holdings_df
from app.crawlers.dfdv_daily import get_dfdv_holdings_df
from app.crawlers.naka_daily import get_nakamoto_holdings_df
from app.crawlers.btcs_daily import get_btcs_eth_holdings_df
from app.pipelines.ETH_SEC_holdings_main import get_sec_eth_holdings_df
from app.pipelines.BTC_SEC_holdings_main import get_sec_btc_holdings_df

DEFAULT_TABLE = "Holdings_test"

@task(retries=2, retry_delay_seconds=60)
def t_sequans() -> pd.DataFrame:
    return get_sequans_holdings_df("SQNS", "BTC")

@task(retries=2, retry_delay_seconds=60)
def t_dfdv() -> pd.DataFrame:
    return get_dfdv_holdings_df("DFDV", "SOL")

@task(retries=2, retry_delay_seconds=60)
def t_naka() -> pd.DataFrame:
    return get_nakamoto_holdings_df(hours=24, ticker="NAKA", asset="BTC")

@task(retries=2, retry_delay_seconds=60)
def t_btcs() -> pd.DataFrame:
    return get_btcs_eth_holdings_df(year=2025)

@task(retries=2, retry_delay_seconds=60)
def t_sec_eth() -> pd.DataFrame:
    return get_sec_eth_holdings_df(
        tickers="BMNR,SBET,BTBT,ETHZ",
        hours_back=24,
        forms=("8-K","10-K","10-Q"),
        verbose=False,
    )

@task(retries=2, retry_delay_seconds=60)
def t_sec_btc() -> pd.DataFrame:
    return get_sec_btc_holdings_df(
        tickers="MSTR,CEP,NAKA,SMLR",
        hours_back=24,
        forms=("8-K","10-K","10-Q"),
        verbose=False,
    )

@flow(name="daily-main-pipeline", log_prints=True)
def daily_main_pipeline(table: str = DEFAULT_TABLE, do_update: bool = False):
    logger = get_run_logger()
    # Run producers in parallel
    f1 = t_sequans.submit()
    f2 = t_dfdv.submit()
    f3 = t_naka.submit()
    f4 = t_btcs.submit()
    f5 = t_sec_eth.submit()
    f6 = t_sec_btc.submit()

    # Collect results
    frames = [f.result() for f in (f1, f2, f3, f4, f5, f6)]

    # Upload
    stats = concat_and_upload(table, frames, do_update=do_update)
    logger.info(f"Attempted: {stats['attempted']}, skipped: {stats['skipped_existing']}, sent: {stats['sent']}")
    return stats

if __name__ == "__main__":
    daily_main_pipeline()
