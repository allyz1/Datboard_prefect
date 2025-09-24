# deploy.py  (put this at your repo root)
from prefect.client.schemas.schedules import CronSchedule
from prefect import flow
from prefect.runner.storage import GitRepository
from prefect.blocks.system import Secret

LONG_TICKERS = [
    "MSTR","CEP","SMLR","NAKA","BMNR","SBET","ETHZ","BTCS","SQNS","BTBT","DFDV","UPXI"
]

if __name__ == "__main__":
    remote_flow = flow.from_source(
        source=GitRepository(url="https://github.com/allyz1/Datboard_prefect.git"),
        entrypoint="flows/daily_main_flow.py:daily_main_pipeline",
    )

    remote_flow.deploy(
        name="daily",
        work_pool_name="managed-pool",
        schedule=CronSchedule(cron="0 6 * * *", timezone="America/Los_Angeles"),
        parameters={
            "table": "Holdings_raw",
            "do_update": False,
            "tickers": LONG_TICKERS,        # 👈 one list for Polygon + ATM
            "atm_hours": 24,
            "atm_do_upsert": False,
        },
        job_variables={
            "pip_packages": [
                "pandas","numpy","requests","beautifulsoup4","lxml",
                "python-dateutil","python-dotenv","supabase",
            ],
            "env": {
                "SUPABASE_URL": "{{ prefect.blocks.secret.superbaseurl }}",
                "SUPABASE_SERVICE_KEY": "{{ prefect.blocks.secret.superbaseanonkey }}",
                "POLYGON_API_KEY": "{{ prefect.blocks.secret.polygonapikey }}",  # 👈 same style as Supabase
            },
        },
    )

    print("✅ Registered deployment: daily-main-pipeline/daily")