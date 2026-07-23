from airflow.decorators import task

MODEL_NAME = "xgboost_tsd_model_fixed"
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MINIO_ENDPOINT = "http://minio:9000"
STORAGE = {"key": "minioadmin", 
           "secret": "minioadmin",
           "client_kwargs": {"endpoint_url": MINIO_ENDPOINT}}
FEATURE_ORDER = [
    "userDataReadIops", "userDataWriteIops", "readLatencyMs", "writeLatencyMs", "cpuPercent", "memoryPercent",
    "iops_data_read_rolling_mean_5m", "iops_data_read_rolling_std_5m", "iops_data_read_pct_change_1h",
    "iops_data_write_rolling_mean_5m", "iops_data_write_rolling_std_5m", "iops_data_write_pct_change_1h",
    "read_latency_rolling_mean_5m", "read_latency_rolling_std_5m", "read_latency_pct_change_1h",
    "write_latency_rolling_mean_5m", "write_latency_rolling_std_5m", "write_latency_pct_change_1h",
]
@task
def simulate_fleet_task(ti):
    from src.simulate_fleet import build_fleet_pandas
    run_id = ti.run_id
    n_devices = 10
    n_rows = 10000
    fleet_df = build_fleet_pandas(n_devices, n_rows)
    fleet_df.to_parquet(f"s3://tsd/training/{run_id}/raw.parquet", storage_options=STORAGE)
    


@task
def feature_engineering_task(ti):
    from src.feature_engineering import extract_features
    import pandas as pd
    run_id = ti.run_id
    fleet_df = pd.read_parquet(f"s3://tsd/training/{run_id}/raw.parquet", storage_options=STORAGE)
    features_df = fleet_df.groupby('device_id', group_keys=False).apply(extract_features).dropna()
    features_df.to_parquet(f"s3://tsd/training/{run_id}/features_extracted.parquet", storage_options=STORAGE)

@task
def train_eval_split_task(ti):
    import pandas as pd
    run_id = ti.run_id
    features_df = pd.read_parquet(f"s3://tsd/training/{run_id}/features_extracted.parquet", storage_options=STORAGE)
    split_ratio = 0.8
    split_index = int(len(features_df) * split_ratio)
    train_data = features_df.iloc[:split_index]
    test_data = features_df.iloc[split_index:]
   
    train_data.to_parquet(f"s3://tsd/training/{run_id}/train_data.parquet", storage_options=STORAGE)
    test_data.to_parquet(f"s3://tsd/training/{run_id}/test_data.parquet", storage_options=STORAGE)
@task
def train_and_evaluate_model_task(ti):
    from src.train import train_model,evaluate_model
    import mlflow
    import pandas as pd
    from sklearn.utils.class_weight import compute_sample_weight
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.create_experiment("retraining_experiment") if client.get_experiment_by_name("retraining_experiment") is None else client.get_experiment_by_name("retraining_experiment").experiment_id
    run_id = ti.run_id
    train_data = pd.read_parquet(f"s3://tsd/training/{run_id}/train_data.parquet", storage_options=STORAGE)
    test_data = pd.read_parquet(f"s3://tsd/training/{run_id}/test_data.parquet", storage_options=STORAGE)
    train_x = train_data[FEATURE_ORDER]
    train_y = train_data['label']
    test_x = test_data[FEATURE_ORDER]
    test_y = test_data['label']
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_param("model_type", "xgboost")
        params={
            'objective': 'multi:softprob',
            'num_class': 5,
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'eval_metric': 'mlogloss'
        }
        mlflow.log_params(params)
        sample_weights = compute_sample_weight(class_weight='balanced', y=train_y)
        model =train_model(model_type="xgboost", X_train=train_x, y_train=train_y, params=params, sample_weights=sample_weights)
        metrics = evaluate_model(model, test_x, test_y)
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        with open("classification_report.txt", "w") as f:
            f.write(str(metrics['classification_report']))
        mlflow.log_artifact("classification_report.txt")
        mlflow.xgboost.log_model(model, name="model", registered_model_name=MODEL_NAME)

@task
def set_challenger_task(ti):
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    run_id = ti.run_id
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    version = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)