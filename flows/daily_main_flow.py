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

# Reg Direct extractor (builds a DataFrame)
from app.sec_deals.drivers import run_reg_direct as RD

# Reg Direct upload helpers (already in your supabase_append module)
from app.clients.supabase_append import (
    prep_reg_direct_raw_df,      # if you want to pre-check/normalize (optional)
    upsert_reg_direct_raw_df,
    insert_reg_direct_raw_df,
)

# Outstanding shares extractor
from app.sec_deals.drivers import run_outstanding as OUT

# Supabase append-only insert
from app.clients.supabase_append import insert_outstanding_raw_df

# === NEW: SEC cash daily + Noncrypto upload ===
from app.sec.cash_daily import (
    collect_recent_accessions,
    collect_cash_from_balance_sheets,
)
from app.clients.supabase_append import upload_noncrypto_from_cash

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
    df = get_last_n_days_excluding_today(n=7)  # 3 fully closed days, excludes today
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

@task(retries=2, retry_delay_seconds=60)
def t_reg_direct_df(tickers: list[str], since_hours: int = 24) -> pd.DataFrame:
    """
    Build Registered Direct rows for the last N hours.
    Tries RD.build_for_tickers if present; falls back to per-ticker loop.
    """
    if not tickers:
        return pd.DataFrame()

    params = dict(
        year=2025,
        year_by="accession",
        limit=600,
        max_docs=6,
        max_snips=4,
        since_hours=since_hours,
        use_acceptance=True,
    )

    # Preferred: single call aggregator
    if hasattr(RD, "build_for_tickers"):
        return RD.build_for_tickers(tickers, **params)

    # Fallback: loop per ticker
    if not hasattr(RD, "build_for_ticker"):
        raise ImportError("run_reg_direct module lacks build_for_tickers and build_for_ticker")

    frames: list[pd.DataFrame] = []
    uniq = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    for tk in uniq:
        df_t = RD.build_for_ticker(tk, **params)
        if isinstance(df_t, pd.DataFrame) and not df_t.empty:
            frames.append(df_t)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@task(retries=2, retry_delay_seconds=60)
def t_upload_reg_direct(df: pd.DataFrame) -> dict:
    """
    Append Registered Direct rows to Reg_direct_raw.
    No upsert/unique constraint; relies on auto-increment PK.
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}
    return insert_reg_direct_raw_df("Reg_direct_raw", df, chunk_size=500)

@task(retries=2, retry_delay_seconds=60)
def t_outstanding_df(tickers: list[str], since_hours: int = 24) -> pd.DataFrame:
    """
    Build Outstanding Shares rows for the last N hours.
    Uses OUT.build_for_ticker per symbol.
    """
    if not tickers:
        return pd.DataFrame()

    params = dict(
        year=2025,
        year_by="filingdate",     # your script default; acceptance window is controlled by since_hours/use_acceptance
        limit=400,
        max_docs=4,
        max_snips=2,
        since_hours=since_hours,
        use_acceptance=True,
    )

    frames: list[pd.DataFrame] = []
    uniq = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    for tk in uniq:
        df_t = OUT.build_for_ticker(tk, **params)
        if isinstance(df_t, pd.DataFrame) and not df_t.empty:
            frames.append(df_t)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@task(retries=2, retry_delay_seconds=60)
def t_upload_outstanding(df: pd.DataFrame) -> dict:
    """
    Append Outstanding Shares rows to Outstanding_shares_raw (no upsert).
    """
    if df is None or df.empty:
        return {"attempted": 0, "sent": 0}
    return insert_outstanding_raw_df("Outstanding_shares_raw", df, chunk_size=500)



# ---------------- NEW: SEC cash → Noncrypto_holdings_raw ----------------
def _cash_recent_for_ticker(
    ticker: str,
    recent_hours: int,
    forms: str,
    from_date: str,
    limit_filings: int,
    year: int,
    year_by: str,
    max_docs_per_filing: int,
    log_every: int,
    diagnostics: bool,
) -> pd.DataFrame:
    # Probe which accessions had any file touched in the window
    accs = collect_recent_accessions(
        ticker=ticker,
        forms=forms,
        from_date=from_date,
        limit_filings=limit_filings,
        recent_hours=recent_hours,
        year=year,
        year_by=year_by,
    )
    if not accs:
        return pd.DataFrame()

    # Run extractor and filter to those accessions
    df_all, _ = collect_cash_from_balance_sheets(
        ticker=ticker,
        forms=forms,
        from_date=from_date,
        limit_filings=limit_filings,
        max_docs_per_filing=max_docs_per_filing,
        log_every=log_every,
        diagnostics=diagnostics,
        year=year,
        year_by=year_by,
    )
    if df_all is None or df_all.empty:
        return pd.DataFrame()

    return df_all[df_all["accessionNumber"].astype(str).isin(accs)].copy()

    
@task(retries=2, retry_delay_seconds=60)
def t_sec_cash_noncrypto(
    tickers: list[str],
    recent_hours: int = 24,
    forms: str = "10-K,10-Q,20-F,6-K",
    from_date: str = "2024-01-01",
    year: int = 2025,
    year_by: str = "accession",
    limit_filings: int = 300,
    max_docs_per_filing: int = 12,
    prefer_combined_if_missing: bool = False,
    do_update: bool = False,
) -> dict:
    """
    Produce cash-like holdings from SEC docs that changed in the last `recent_hours`,
    then upload them to Noncrypto_holdings_raw.
    """
    logger = get_run_logger()
    frames: list[pd.DataFrame] = []

    for tk in tickers:
        logger.info(f"[SEC cash] probe+extract {tk} (last {recent_hours}h)")
        df_cash = _cash_recent_for_ticker(
            ticker=tk,
            recent_hours=recent_hours,
            forms=forms,
            from_date=from_date,
            limit_filings=limit_filings,
            year=year,
            year_by=year_by,
            max_docs_per_filing=max_docs_per_filing,
            log_every=10,
            diagnostics=False,
        )
        if df_cash is None or df_cash.empty:
            logger.info(f"[SEC cash] {tk}: 0 rows in window")
            continue
        frames.append(df_cash)

    if not frames:
        logger.warning("[SEC cash] No cash rows produced in requested window.")
        return {"attempted": 0, "skipped_existing": 0, "sent": 0}

    cash_df = pd.concat(frames, ignore_index=True)
    stats = upload_noncrypto_from_cash(
        cash_df,
        table="Noncrypto_holdings_raw",
        do_update=do_update,
        prefer_combined_if_missing=prefer_combined_if_missing,
        precheck=True,
    )
    logger.info(f"[Noncrypto_holdings_raw] attempted={stats['attempted']} "
                f"skipped={stats['skipped_existing']} sent={stats['sent']}")
    return stats

# ---------------- Flow ----------------
@flow(name="daily-main-pipeline", log_prints=True)
def daily_main_pipeline(
    table: str = DEFAULT_TABLE,
    do_update: bool = False,
    tickers: list[str] | None = None,       # <- single list used for Polygon + ATM
    atm_hours: int = 24,
    atm_do_upsert: bool = False,
    sec_cash_hours: int = 24,
    reg_direct_hours: int = 24,
    reg_direct_do_upsert: bool = True,
    outstanding_hours: int = 24,
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

    # 6) NEW: SEC cash → Noncrypto_holdings_raw
    noncrypto_stats = t_sec_cash_noncrypto.submit(
        tickers=tickers,
        recent_hours=sec_cash_hours,
        forms="10-K,10-Q,20-F,6-K",
        from_date="2024-01-01",
        year=2025,
        year_by="accession",
        limit_filings=300,
        max_docs_per_filing=12,
        prefer_combined_if_missing=False,
        do_update=False,
    ).result()
    
    # 6) Registered Directs → Reg_direct_raw  (same tickers)
    rd_df = t_reg_direct_df.submit(tickers, since_hours=reg_direct_hours).result()
    if rd_df is not None and not rd_df.empty:
        rd_stats = t_upload_reg_direct.submit(rd_df).result()
        logger.info(f"[Reg_direct_raw] attempted={rd_stats.get('attempted', 0)} sent={rd_stats.get('sent', 0)}")
    else:
        logger.info("[Reg_direct_raw] No Registered Direct rows in requested window.")
        rd_stats = {"attempted": 0, "sent": 0}

    # Outstanding Shares → Outstanding_shares_raw
    out_df = t_outstanding_df.submit(tickers, since_hours=outstanding_hours).result()
    if out_df is not None and not out_df.empty:
        out_stats = t_upload_outstanding.submit(out_df).result()
        logger.info(f"[Outstanding_shares_raw] attempted={out_stats.get('attempted', 0)} sent={out_stats.get('sent', 0)}")
    else:
        logger.info("[Outstanding_shares_raw] No outstanding-share rows in requested window.")
        out_stats = {"attempted": 0, "sent": 0}

    return {
        "holdings": stats_holdings,
        "coinbase": stats_cb,
        "polygon": polygon_stats,
        "atm_raw": atm_stats,
        "noncrypto_holdings": noncrypto_stats,
        "reg_direct_raw": rd_stats,
        "outstanding_shares_raw": out_stats,
    }

if __name__ == "__main__":
    daily_main_pipeline()
