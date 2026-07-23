from datetime import datetime
from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

with DAG(
    dag_id="batch_scoring_dag",
    schedule ="0 0 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["batch", "scoring","tsd"],
) as dag:
    batch_scoring_task = KubernetesPodOperator(
        task_id="batch_scoring_task",
        name="batch-scoring-task",
        namespace="tsd",
        image="tsd-batch:latest",
        image_pull_policy="IfNotPresent",
        arguments =["--fleet_path", "s3a://tsd/fleet", "--out_path", "s3a://tsd/reports"],
        env_vars={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "S3_ENDPOINT": "http://minio:9000",
            "S3_ACCESS_KEY": "minioadmin",
            "S3_SECRET_KEY": "minioadmin",
        },
        get_logs=True,
        on_finish_action="delete_pod",
    )