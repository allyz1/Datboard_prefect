import os
import psycopg
from prefect import flow, task, get_run_logger

# --------- CONFIG ---------
# Put your Supabase Postgres DSN in an env var on the worker/runner:
#   SUPABASE_DB_DSN=postgres://postgres:<DB_PASSWORD>@db.<project>.supabase.co:5432/postgres?sslmode=require
DB_DSN = os.getenv("SUPABASE_DB_DSN")

# Refresh order: parents first, then downstream
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

# Optional: bump/override at runtime via comma-separated env var
override = os.getenv("REFRESH_ONLY")
if override:
    MVS = [m.strip() for m in override.split(",") if m.strip()]
# --------------------------

@task
def advisory_lock(conn_str: str, key: int = 424242) -> bool:
    with psycopg.connect(conn_str, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s);", (key,))
        return bool(cur.fetchone()[0])

@task
def advisory_unlock(conn_str: str, key: int = 424242) -> None:
    with psycopg.connect(conn_str, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_unlock(%s);", (key,))

@task
def refresh_one(conn_str: str, mv_name: str) -> None:
    # Run outside a transaction so CONCURRENTLY is allowed
    with psycopg.connect(conn_str, autocommit=True) as conn, conn.cursor() as cur:
        # sane timeouts
        cur.execute("SET statement_timeout = '10min';")
        cur.execute("SET lock_timeout = '30s';")
        try:
            cur.execute(f"REFRESH MATERIALIZED VIEW CONCURRENTLY {mv_name};")
        except Exception as e:
            # Fallback if MV lacks a unique index or concurrent refresh isn’t allowed
            cur.execute(f"REFRESH MATERIALIZED VIEW {mv_name};")

@flow(name="refresh-supabase-materialized-views", retries=0, timeout_seconds=60*30)
def refresh_snapshots_flow():
    """
    Refresh all materialized views in dependency order.
    Use schedule timezone America/Los_Angeles for 1:00 PM PT.
    """
    logger = get_run_logger()

    if not DB_DSN:
        raise RuntimeError("SUPABASE_DB_DSN is not set")

    if not MVS:
        logger.info("No MVs configured to refresh.")
        return

    got_lock = advisory_lock.submit(DB_DSN).result()
    if not got_lock:
        logger.info("Another refresh is in progress (advisory lock). Exiting.")
        return

    try:
        for mv in MVS:
            logger.info(f"Refreshing {mv} ...")
            refresh_one.submit(DB_DSN, mv).result()
            logger.info(f"✓ {mv} refreshed")
    finally:
        advisory_unlock.submit(DB_DSN)

if __name__ == "__main__":
    refresh_snapshots_flow()
