import os, psycopg
from prefect import flow, task, get_run_logger

MVS = [
    "companies_asset_map",
    "daily_growth_agg_mv",
    "ts_agg_mv",
    "daily_time_series_per_ticker_mv",
    "daily_time_series_per_ticker_proforma_mv",
    "daily_time_series_growth_drivers_per_ticker_mv",
    "mv_series_latest",
    "mv_polygon_latest",
    "mv_sec_outstanding_latest",
    "mv_crypto_latest",
]

@task
def _lock(dsn: str) -> bool:
    with psycopg.connect(dsn, autocommit=True) as c, c.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(424242);")
        return cur.fetchone()[0]

@task
def _unlock(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c, c.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(424242);")

@task
def _refresh_one(dsn: str, mv: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c, c.cursor() as cur:
        cur.execute("SET statement_timeout='10min'; SET lock_timeout='30s';")
        try:
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv};")
        except Exception:
            cur.execute(f"REFRESH MATERIALIZED VIEW {mv};")

@flow(name="refresh-supabase-materialized-views", retries=0, timeout_seconds=60*30)
def refresh_snapshots_flow():
    logger = get_run_logger()
    dsn = os.getenv("SUPABASE_DB_DSN")
    if not dsn: raise RuntimeError("SUPABASE_DB_DSN not set")
    if not _lock.submit(dsn).result():
        logger.info("Another refresh is running; exiting.")
        return
    try:
        for mv in MVS:
            logger.info(f"Refreshing {mv} …")
            _refresh_one.submit(dsn, mv).result()
            logger.info(f"✓ {mv}")
    finally:
        _unlock.submit(dsn)

if __name__ == "__main__":
    refresh_snapshots_flow()
