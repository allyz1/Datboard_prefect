# deploy.py  (put this at your repo root)
from prefect.client.schemas.schedules import CronSchedule
from prefect import flow
from prefect.runner.storage import GitRepository
from prefect.blocks.system import Secret

if __name__ == "__main__":
    # Load your flow from the private GitHub repo
    remote_flow = flow.from_source(
        source=GitRepository(
            url="https://github.com/allyz1/Datboard_prefect.git"
        ),
        # path in the repo to your flow function
        entrypoint="flows/daily_main_flow.py:daily_main_pipeline",
    )

    # Register the deployment to the managed pool, install deps at runtime, set env
    remote_flow.deploy(
        name="daily",
        work_pool_name="managed-pool",  # your prefect:managed pool
        schedule=CronSchedule(cron="0 6 * * *", timezone="America/Los_Angeles"),
        parameters={"table": "Holdings_raw", "do_update": False},
        job_variables={
            # Prefect Managed will pip install these packages at run time
            "pip_packages": [
                "pandas", "numpy", "requests", "beautifulsoup4", "lxml",
                "python-dateutil", "python-dotenv", "supabase"
            ],
            # Env that your code reads via os.environ[…]
            "env": {
                "SUPABASE_URL": "{{ prefect.blocks.secret.superbaseurl }}",
                "SUPABASE_SERVICE_KEY": "{{ prefect.blocks.secret.superbaseanonkey }}",
            },
        },
    )

    print("✅ Registered deployment: daily-main-pipeline/daily")
