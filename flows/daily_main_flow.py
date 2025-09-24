# flows/daily_main_flow.py
from prefect import flow, task, get_run_logger
import pandas as pd
from datetime import datetime, timezone, timedelta, date
import os

from app import config  # loads .env locally; in prod set real env vars
from app.clients.supabase_append import concat_and_upload
from app.clients.upload_coinbase import upload_coinbase_df
from app.prices.coinbase import get_last_n_days_excluding_today  # date, product_id, ohlc, volume

# Polygon helpers (fetch + upload)
from app.prices.polygon_daily import fetch_polygon_daily_range_one_ticker
from app.clients.upload_polygon import upload_polygon_df

# Producers (holdings + SEC)
from app.crawlers.sequans_bitcoin_api_scraper import get_sequans_holdings_df
from app.crawlers.dfdv_daily import get_dfdv_holdings_df
from app.crawlers.naka_daily import get_nakamoto_holdings_df
from app.crawlers.btcs_daily import get_btcs_eth_holdings_df
from app.pipelines.ETH_SEC_holdings_main import get_sec_eth_holdings_df
from app.pipelines.BTC_SEC_holdings_main import get_sec_btc_holdings_df

# ATM timeline + Supabase ATM helpers
from app.sec.ATM_daily import collect_timelines
from app.clients.supabase_append import insert_atm_raw_df, upsert_atm_raw_df

DEFAULT_TABLE = "Holdings_raw"
DEFAULT_TICKERS = ["MSTR","CEP","SMLR","NAKA","BMNR","SBET","ETHZ","BTCS","SQNS","BTBT","DFDV","UPXI"]

# ---------------- Holdings tasks ----------------
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

# ---------------- Coinbase prices task ----------------
@task(retries=2, retry_delay_seconds=60)
def t_coinbase_prices() -> pd.DataFrame:
    df = get_last_n_days_excluding_today(n=3)  # 3 fully closed days, excludes today
    cols = ["date","product_id","open","high","low","close","volume"]
    return df.reindex(columns=cols)

# ---------------- Polygon prices task (uses env var) ----------------
def _utc_today() -> date:
    return datetime.now(timezone.utc).date()

@task(retries=2, retry_delay_seconds=60)
def t_polygon_prices(tickers: list[str], adjusted: bool = True) -> pd.DataFrame:
    """Fetch last 3 days (UTC, excl. today) of daily bars for tickers from Polygon."""
    logger = get_run_logger()

    # Read from env (deploy injects POLYGON_API_KEY like the Supabase vars)
    polygon_api_key = os.environ.get("POLYGON_API_KEY", "").strip()
    if not polygon_api_key:
        logger.error("[Polygon] Missing POLYGON_API_KEY env var")
        return pd.DataFrame(columns=["ticker","date","open","high","low","close","volume","transactions","vwap"])

    today = _utc_today()
    end_date = today - timedelta(days=1)
    start_date = end_date - timedelta(days=2)
    if end_date < start_date:
        start_date = end_date

    logger.info(f"[Polygon] Range {start_date} → {end_date} adjusted={adjusted} tickers={len(tickers)}")

    frames = []
    for tk in tickers:
        df_t = fetch_polygon_daily_range_one_ticker(
            tk, start_date, end_date,
            api_key=polygon_api_key,
            adjusted=adjusted,
            max_retries=3,
            timeout=20,
        )
        if df_t.empty:
            logger.info(f"[Polygon] {tk}: 0 rows")
        else:
            dmin, dmax = df_t["date"].min(), df_t["date"].max()
            logger.info(f"[Polygon] {tk}: rows={len(df_t)} dates={dmin}..{dmax}")
            logger.info("\n" + df_t.tail(min(3, len(df_t))).to_string(index=False))
        frames.append(df_t)

    if any(not f.empty for f in frames):
        combined = pd.concat(frames, ignore_index=True)
        cols = ["ticker","date","open","high","low","close","volume","transactions","vwap"]
        return combined.reindex(columns=cols)

    return pd.DataFrame(columns=["ticker","date","open","high","low","close","volume","transactions","vwap"])

# ---------------- ATM tasks ----------------
@task(retries=2, retry_delay_seconds=60)
def t_atm_timeline(tickers: list[str], hours: int = 24) -> pd.DataFrame:
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
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}
    if upsert:
        # requires a unique index: accessionNumber, source_url, event_type_final
        return upsert_atm_raw_df("ATM_raw", df, on_conflict="uniq_atm_raw", do_update=False)
    else:
        return insert_atm_raw_df("ATM_raw", df)

# ---------------- Flow ----------------
@flow(name="daily-main-pipeline", log_prints=True)
def daily_main_pipeline(
    table: str = DEFAULT_TABLE,
    do_update: bool = False,
    tickers: list[str] | None = None,       # <- single list used for Polygon + ATM
    atm_hours: int = 24,
    atm_do_upsert: bool = False,
):
    logger = get_run_logger()
    tickers = tickers or DEFAULT_TICKERS

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

    # 3) Coinbase daily prices → `coinbase`
    coinbase_df = t_coinbase_prices.submit().result()
    if not coinbase_df.empty:
        stats_cb = upload_coinbase_df(
            os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"], coinbase_df
        )
        logger.info(f"[Prices -> coinbase] attempted={stats_cb['attempted']} sent={stats_cb['sent']}")
    else:
        logger.warning("[Prices -> coinbase] No rows produced for last 3 days.")
        stats_cb = {"attempted": 0, "sent": 0}

    # 4) Polygon daily prices (last 3 days) → `polygon`
    polygon_df = t_polygon_prices.submit(tickers, adjusted=True).result()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        logger.error("[Prices -> polygon] Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
        polygon_stats = None
    else:
        if not polygon_df.empty:
            polygon_stats = upload_polygon_df(
                supabase_url, supabase_key, polygon_df,
                table="polygon", on_conflict="key"  # use "date,ticker" if you use composite PK
            )
            logger.info(f"[Prices -> polygon] attempted={polygon_stats['attempted']} sent={polygon_stats['sent']}")
        else:
            logger.warning("[Prices -> polygon] No rows produced for last 3 days.")
            polygon_stats = {"attempted": 0, "sent": 0}

    # 5) ATM timeline → ATM_raw  (same tickers)
    atm_df = t_atm_timeline.submit(tickers, hours=atm_hours).result()
    if atm_df is not None and not atm_df.empty:
        atm_stats = t_upload_atm.submit(atm_df, upsert=atm_do_upsert).result()
        logger.info(f"[ATM_raw] attempted={atm_stats.get('attempted', 0)} sent={atm_stats.get('sent', 0)}")
    else:
        logger.info("[ATM_raw] No ATM rows in requested window.")
        atm_stats = {"attempted": 0, "sent": 0}

    return {
        "holdings": stats_holdings,
        "coinbase": stats_cb,
        "polygon": polygon_stats,
        "atm_raw": atm_stats,
    }

if __name__ == "__main__":
    daily_main_pipeline()
