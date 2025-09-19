# deploy.py  (repo root)
from prefect.client.schemas.schedules import CronSchedule
from flows.daily_main_flow import daily_main_pipeline  # adjust if your path differs

if __name__ == "__main__":
    daily_main_pipeline.deploy(
        name="daily",
        work_pool_name="managed-pool",   # <-- your prefect:managed work pool
        parameters={"table": "Holdings_test", "do_update": False},
        schedule=CronSchedule(cron="0 6 * * *", timezone="America/Los_Angeles"),
        pull_steps=[
            {
                "prefect.deployments.steps.git_clone": {
                    "id": "clone",
                    "repository": "https://github.com/allyz1/Datboard_prefect.git",
                    "branch": "main",
                    # GitHub PAT stored in a Secret block named 'github-pat' (Password type)
                    "access_token": "{{ prefect.blocks.secret.github-pat }}",
                }
            },
            # DEBUG: show what the runner sees after clone so we can locate requirements.txt
            {
                "prefect.deployments.steps.run_shell_script": {
                    "directory": "{{ clone.directory }}",
                    "stream_output": True,
                    "script": (
                        "echo 'Clone directory:' $(pwd)\n"
                        "echo '== ls -la ==' && ls -la\n"
                        "echo '== find requirements.txt (depth 3) ==' && "
                        "find . -maxdepth 3 -name 'requirements.txt' -print\n"
                    ),
                }
            },
            # Install deps from requirements.txt at repo root
            {
                "prefect.deployments.steps.pip_install_requirements": {
                    "directory": "{{ clone.directory }}",
                    "requirements_file": "requirements.txt",
                    "stream_output": True,
                }
                # If this still can't find it, comment the block above and use absolute path:
                # {"prefect.deployments.steps.pip_install_requirements": {
                #     "requirements_file": "{{ clone.directory }}/requirements.txt",
                #     "stream_output": True,
                # }}
            },
        ],
        # Env vars your code reads via os.environ[...]
        job_variables={
            "env": {
                "SUPABASE_URL": "{{ prefect.blocks.secret.supabase-url }}",
                "SUPABASE_SERVICE_KEY": "{{ prefect.blocks.secret.supabase-service-key }}",
            }
        },
    )
    print("✅ Registered deployment: daily-main-pipeline/daily")
