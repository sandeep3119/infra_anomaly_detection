import os

from pyspark.sql import SparkSession,functions as F
from pyspark.sql.types import StructField, StructType, LongType
from datetime import date
import argparse
from functools import partial
import mlflow
import onnxruntime as ort
import numpy as np

from src.batch_feature import add_features

FEATURE_ORDER = [
    "userDataReadIops", "userDataWriteIops", "readLatencyMs", "writeLatencyMs", "cpuPercent", "memoryPercent",
    "iops_data_read_rolling_mean_5m", "iops_data_read_rolling_std_5m", "iops_data_read_pct_change_1h",
    "iops_data_write_rolling_mean_5m", "iops_data_write_rolling_std_5m", "iops_data_write_pct_change_1h",
    "read_latency_rolling_mean_5m", "read_latency_rolling_std_5m", "read_latency_pct_change_1h",
    "write_latency_rolling_mean_5m", "write_latency_rolling_std_5m", "write_latency_pct_change_1h",
]
_session = None

def load_model_bytes(tracking_uri, name, alias):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_client = mlflow.tracking.MlflowClient()
    model_path = mlflow.artifacts.download_artifacts(
            f"models:/{name}@{alias}"
    )
    model_version = mlflow_client.get_model_version_by_alias(name, alias).version
    model_file = [f for f in os.listdir(model_path) if f.endswith(".onnx")][0]
    try:
        with open(os.path.join(model_path, model_file), "rb") as f:
            return f.read(), model_version
    except Exception as e:
        raise RuntimeError(f"Failed to load model bytes: {e}")
    

def score_partition(broadcast_model, batches):
    global _session
    if _session is None:                                  
        _session = ort.InferenceSession(broadcast_model.value)
    input_name = _session.get_inputs()[0].name
    for pdf in batches:
        X = pdf[FEATURE_ORDER].to_numpy(dtype=np.float32) 
        preds = _session.run(None, {input_name: X})[0]
        pdf = pdf.copy()
        pdf["predicted_class"] = preds.reshape(-1)
        yield pdf
def main(fleet_path, out_path):
    spark = (SparkSession.builder.appName("tsd-batch-score").config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.5.0")
    .config("spark.hadoop.fs.s3a.endpoint", os.getenv("S3_ENDPOINT", "http://minio:9000"))
    .config("spark.hadoop.fs.s3a.access.key", os.getenv("S3_ACCESS_KEY", "minioadmin"))
    .config("spark.hadoop.fs.s3a.secret.key", os.getenv("S3_SECRET_KEY", "minioadmin"))
    .config("spark.hadoop.fs.s3a.path.style.access", "true")       
    .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
    .getOrCreate()) 
    fleet = spark.read.parquet(fleet_path)
    feat = add_features(fleet)                      
    scorable = feat.dropna(subset=FEATURE_ORDER)
    out_schema = StructType(scorable.schema.fields + [StructField("predicted_class", LongType())])

    model_bytes, model_version = load_model_bytes(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"),
        name=os.getenv("MLFLOW_MODEL_NAME", "xgboost_tsd_model_fixed"),
        alias=os.getenv("MLFLOW_MODEL_ALIAS", "champion")
    )
    broadcast_model = spark.sparkContext.broadcast(model_bytes)
    scored = scorable.mapInPandas(partial(score_partition, broadcast_model), schema=out_schema).cache()

    scored_out = scored.withColumn("scoring_date", F.lit(str(date.today()))).withColumn("model_version", F.lit(model_version))
    scored_out.write.mode("overwrite").partitionBy("scoring_date").parquet(out_path)

    summary = scored.groupBy("tier", "predicted_class").count().orderBy("tier", "predicted_class")
    summary = summary.withColumn("model_version", F.lit(model_version))
    summary.show()
    summary.write.mode("overwrite").parquet(f"{out_path}/summary/")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scoring for TSD anomaly detection.")
    parser.add_argument("--fleet_path", type=str, required=True, help="Path to the fleet data in Parquet format.")
    parser.add_argument("--out_path", type=str, required=True, help="Output path for the scored data in Parquet format.")
    
    args = parser.parse_args()
    
    main(args.fleet_path, args.out_path)