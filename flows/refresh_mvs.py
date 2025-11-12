import os
import socket
from urllib.parse import urlparse

import psycopg
from psycopg import sql
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
            cur.execute(
                sql.SQL("REFRESH MATERIALIZED VIEW CONCURRENTLY {}").format(
                    sql.Identifier(mv)
                )
            )
        except Exception:
            cur.execute(
                sql.SQL("REFRESH MATERIALIZED VIEW {}").format(sql.Identifier(mv))
            )


def _resolve_dsn() -> str:
    explicit = os.getenv("SUPABASE_DB_DSN")
    if explicit:
        return explicit

    supabase_url = os.getenv("SUPABASE_URL")
    password = os.getenv("SUPABASE_DB_PASSWORD")
    if not supabase_url or not password:
        raise RuntimeError(
            "SUPABASE_DB_DSN not set and missing SUPABASE_URL/SUPABASE_DB_PASSWORD"
        )

    parsed = urlparse(supabase_url)
    if not parsed.hostname or not parsed.hostname.endswith(".supabase.co"):
        raise RuntimeError("SUPABASE_URL must be https://<project>.supabase.co")

    project = parsed.hostname.split(".")[0]
    host = os.getenv("SUPABASE_DB_HOST", f"db.{project}.supabase.co")
    user = os.getenv("SUPABASE_DB_USER", "postgres")
    port = os.getenv("SUPABASE_DB_PORT", "5432")
    db_name = os.getenv("SUPABASE_DB_NAME", "postgres")

    hostaddr = os.getenv("SUPABASE_DB_HOSTADDR")
    if not hostaddr:
        try:
            hostaddr = socket.gethostbyname(host)
        except Exception:
            hostaddr = None

    parts = [
        f"host={host}",
        f"port={port}",
        f"dbname={db_name}",
        f"user={user}",
        f"password={password}",
        "sslmode=require",
        "connect_timeout=15",
    ]
    if hostaddr:
        parts.insert(1, f"hostaddr={hostaddr}")

    return " ".join(parts)

@flow(name="refresh-supabase-materialized-views", retries=0, timeout_seconds=60*30)
def refresh_snapshots_flow():
    logger = get_run_logger()
    dsn = _resolve_dsn()
    password = os.getenv("SUPABASE_DB_PASSWORD")
    masked = dsn if not password else dsn.replace(password, "****")
    logger.info(f"Conninfo preview: {masked}")
    try:
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT inet_server_addr(), inet_server_port();")
            addr, port = cur.fetchone()
            logger.info(f"Connectivity probe successful: {addr}:{port}")
    except Exception as exc:
        logger.error(f"Connectivity probe failed: {exc}")
        raise
    if not _lock.submit(dsn).result():
        logger.info("Another refresh is running; exiting.")
        return
    try:
        for mv in MVS:
            logger.info(f"Refreshing {mv} …")
            _refresh_one.submit(dsn, mv).result()
            logger.info(f"✓ {mv}")
    finally:
        _unlock.submit(dsn).result()

if __name__ == "__main__":
    refresh_snapshots_flow()
