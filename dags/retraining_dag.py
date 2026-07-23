from datetime import datetime
from airflow.decorators import dag
from retraining_tasks import (
    simulate_fleet_task,
    feature_engineering_task,
    train_eval_split_task,
    train_and_evaluate_model_task,
    set_challenger_task,
)

@dag(
    dag_id="retraining_dag",
    schedule=None,                 # on-demand for now
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["tsd", "retraining"],
)
def retraining():
    sim   = simulate_fleet_task()
    feat  = feature_engineering_task()
    split_and_save = train_eval_split_task()
    train = train_and_evaluate_model_task()
    update_challenger = set_challenger_task()

    sim >> feat >> split_and_save >> train >> update_challenger      # explicit order — no data deps to infer it

retraining()