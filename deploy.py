# deploy.py  (repo root)
from prefect.client.schemas.schedules import CronSchedule
from prefect import flow
from prefect.runner.storage import GitRepository
from prefect.blocks.system import Secret  # ok to keep even if not referenced directly

LONG_TICKERS = [
    "MSTR","XXI","SMLR","NAKA","SQNS","BMNR","SBET","ETHZ","BTCS","BTBT","GAME","DFDV",
    "UPXI","HSDT","FWDI","ETHM","STSS","FGNX","STKE","MARA","DJT","GLXY","CLSK","BRR",
    "GME","EMPD","CORZ","FLD","USBC","LMFA","DEFT","GNS","ICG","COSM","KIDZ"
]

if __name__ == "__main__":
    # ------------------- Deployment 1: your existing daily ingest -------------------
    daily_flow = flow.from_source(
        source=GitRepository(url="https://github.com/allyz1/Datboard_prefect.git"),
        entrypoint="flows/daily_main_flow.py:daily_main_pipeline",
    )

    daily_flow.deploy(
        name="daily",
        work_pool_name="managed-pool",
        schedule=CronSchedule(cron="0 6 * * *", timezone="America/Los_Angeles"),
        parameters={
            "table": "Holdings_raw",
            "do_update": False,
            "tickers": LONG_TICKERS,
            "atm_hours": 24,
            "atm_do_upsert": False,
            "polygon_extra_tickers": ["ORBS"],
            "sec_cash_hours": 24,
            "sec_btc_hours": 336,
            "reg_direct_hours": 24,
            "reg_direct_do_upsert": True,
            "outstanding_hours": 24,
            "pipes_hours": 24,
            "warrants_hours": 24,
            "warrants_do_upsert": False,
        },
        job_variables={
            "pip_packages": [
                "pandas","numpy","requests","beautifulsoup4","lxml",
                "python-dateutil","python-dotenv","supabase",
            ],
            "env": {
                "SUPABASE_URL": "{{ prefect.blocks.secret.superbaseurl }}",
                "SUPABASE_SERVICE_KEY": "{{ prefect.blocks.secret.superbaseanonkey }}",
                "POLYGON_API_KEY": "{{ prefect.blocks.secret.polygonapikey }}",
            },
        },
    )

    # ------------------- Deployment 2: MV refresh at 1:00 PM PT -------------------
    # Flow lives in the same repo; it builds the Postgres DSN from SUPABASE_URL + SUPABASE_DB_PASSWORD
    mv_flow = flow.from_source(
        source=GitRepository(url="https://github.com/allyz1/Datboard_prefect.git"),
        entrypoint="flows/refresh_mvs.py:refresh_snapshots_flow",
    )

    mv_flow.deploy(
        name="mv-refresh-1pm-pt",
        work_pool_name="managed-pool",
        schedule=CronSchedule(cron="0 13 * * *", timezone="America/Los_Angeles"),
        job_variables={
            "pip_packages": [
                "prefect>=2.16",
                "psycopg[binary,pool]>=3.1",
            ],
            "env": {
                # already in your project; used to derive the DB host
                "SUPABASE_URL": "{{ prefect.blocks.secret.superbaseurl }}",
                # NEW: database password (create a Secret block named supabasedbpassword)
                "SUPABASE_DB_PASSWORD": "{{ prefect.blocks.secret.supabasedpassword }}",
                # Optional: if you prefer to pass a full DSN instead of building it, create a secret and uncomment:
                # "SUPABASE_DB_DSN": "{{ prefect.blocks.secret.supabasedsn }}",
            },
        },
    )

    print("Registered deployments:")
    print("   - daily_main_pipeline/daily")
    print("   - refresh-supabase-materialized-views/mv-refresh-1pm-pt")
