# flows/daily_main_flow.py
from prefect import flow, task, get_run_logger
import pandas as pd
from app import config  # loads .env locally; in prod set real env vars
from app.clients.supabase_append import concat_and_upload
from app.clients.upload_coinbase import upload_coinbase_df
from app.prices.yahoo import get_last_n_days_excluding_today_yf
from app.clients.upload_yfinance import upload_yfinance_df
import os

# Producers
from app.crawlers.sequans_bitcoin_api_scraper import get_sequans_holdings_df
from app.crawlers.dfdv_daily import get_dfdv_holdings_df
from app.crawlers.naka_daily import get_nakamoto_holdings_df
from app.crawlers.btcs_daily import get_btcs_eth_holdings_df
from app.pipelines.ETH_SEC_holdings_main import get_sec_eth_holdings_df
from app.pipelines.BTC_SEC_holdings_main import get_sec_btc_holdings_df

# ⬇️ NEW: ATM timeline + Supabase ATM helpers
# If your sec_atm_timeline_multi.py lives elsewhere, adjust the import path accordingly.
from app.sec.ATM_daily import collect_timelines
from app.clients.supabase_append import insert_atm_raw_df, upsert_atm_raw_df

# ⬇️ Add this import (your helper saved under app/prices/)
from app.prices.coinbase import get_last_n_days_excluding_today  # returns df with date, product_id, open, high, low, close, volume

DEFAULT_TABLE = "Holdings_raw"

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

# ⬇️ New Coinbase task (runs after others)
@task(retries=2, retry_delay_seconds=60)
def t_coinbase_prices() -> pd.DataFrame:
    df = get_last_n_days_excluding_today(n=3)  # 3 fully closed days, excludes today
    cols = ["date","product_id","open","high","low","close","volume"]
    return df.reindex(columns=cols)

# New Prefect task
@task(retries=2, retry_delay_seconds=60)
def t_yfinance_prices() -> pd.DataFrame:
    # Choose your tickers (example includes both equities and crypto)
    tickers = ["MSTR", "CEP", "NAKA", "SMLR", "SQNS", "BMNR", "SBET", "BTBT", "ETHZ", "BTCS", "DFDV", "UPXI"]
    df = get_last_n_days_excluding_today_yf(tickers=tickers, n=7)
    # enforce expected order
    cols = ["date","ticker","open","high","low","close","adj_close","volume"]
    return df.reindex(columns=cols)

# ---------------- NEW: ATM tasks ----------------
@task(retries=2, retry_delay_seconds=60)
def t_atm_timeline(tickers: list[str], hours: int = 24) -> pd.DataFrame:
    """
    Build the ATM timeline DataFrame for the last `hours` hours.
    """
    df = collect_timelines(
        tickers=tickers,
        since_hours=hours,
        forms="8-K,S-1,S-3,S-3ASR,424B5",
        limit=600,
        max_docs=6,
        max_snippets_per_filing=4,
    )
    return df

@task(retries=2, retry_delay_seconds=60)
def t_upload_atm(df: pd.DataFrame, upsert: bool = False) -> dict:
    """
    Upload ATM df to Supabase ATM_raw.
    - append-only by default
    - if upsert=True, requires a unique index (e.g., uniq_atm_raw on accessionNumber, source_url, event_type_final)
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}

    if upsert:
        # If you created the index:
        #   create unique index if not exists uniq_atm_raw
        #   on "ATM_raw" ("accessionNumber","source_url","event_type_final");
        return upsert_atm_raw_df("ATM_raw", df, on_conflict="uniq_atm_raw", do_update=False)
    else:
        return insert_atm_raw_df("ATM_raw", df)


@flow(name="daily-main-pipeline", log_prints=True)
def daily_main_pipeline(
    table: str = DEFAULT_TABLE,
    do_update: bool = False,
    atm_tickers: list[str] | None = None,
    atm_hours: int = 24,
    atm_do_upsert: bool = False,
):
    logger = get_run_logger()

    # 1) Run holdings producers in parallel
    f1 = t_sequans.submit()
    f2 = t_dfdv.submit()
    f3 = t_naka.submit()
    f4 = t_btcs.submit()
    f5 = t_sec_eth.submit()
    f6 = t_sec_btc.submit()

    # 2) Collect & upload holdings
    frames = [f.result() for f in (f1, f2, f3, f4, f5, f6)]
    stats_holdings = concat_and_upload(table, frames, do_update=do_update)
    logger.info(
        f"[Holdings -> {table}] attempted={stats_holdings['attempted']} "
        f"skipped={stats_holdings['skipped_existing']} sent={stats_holdings['sent']}"
    )

    # 3) Run Coinbase daily prices and upsert into `coinbase`
    ctask = t_coinbase_prices.submit()
    coinbase_df = ctask.result()

    if not coinbase_df.empty:
        stats_cb = upload_coinbase_df(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"], coinbase_df
        )
        logger.info(
            f"[Prices -> coinbase] attempted={stats_cb['attempted']} sent={stats_cb['sent']}"
        )
    else:
        logger.warning("[Prices -> coinbase] No rows produced for last 3 days.")
        
    ytask = t_yfinance_prices.submit()
    yf_df = ytask.result()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        logger.error("[Prices -> yfinance] Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
        yf_stats = None
    else:
        if not yf_df.empty:
            yf_stats = upload_yfinance_df(supabase_url, supabase_key, yf_df)
            logger.info(
                f"[Prices -> yfinance] attempted={yf_stats['attempted']} sent={yf_stats['sent']}"
            )
        else:
            logger.warning("[Prices -> yfinance] No rows produced for last 3 days.")
            yf_stats = None
            
            # 5) ⬇️ NEW: ATM timeline → ATM_raw
        if atm_tickers is None:
            # sensible default set; adjust as you like
            atm_tickers = ["DFDV", "UPXI", "BMNR", "BTBT", "ETHZ", "MARA", "RIOT"]

        atm_df = t_atm_timeline.submit(atm_tickers, hours=atm_hours).result()
        if atm_df is not None and not atm_df.empty:
            atm_stats = t_upload_atm.submit(atm_df, upsert=atm_do_upsert).result()
            logger.info(f"[ATM_raw] attempted={atm_stats.get('attempted', 0)} sent={atm_stats.get('sent', 0)}")
        else:
            logger.info("[ATM_raw] No ATM rows in requested window.")
            atm_stats = {"attempted": 0, "sent": 0}

        return {
            "holdings": stats_holdings,
            "coinbase": stats_cb,
            "yfinance": yf_stats,
            "atm_raw": atm_stats,
        }


if __name__ == "__main__":
    daily_main_pipeline()
