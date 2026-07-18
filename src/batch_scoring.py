from pyspark.sql import SparkSession, Window , functions as F
from pyspark.sql.types import StructField, StructType, LongType
from datetime import date
import argparse
from functools import partial

from src.batch_feature import add_features

FEATURE_ORDER = [
    "userDataReadIops", "userDataWriteIops", "readLatencyMs", "writeLatencyMs", "cpuPercent", "memoryPercent",
    "iops_data_read_rolling_mean_5m", "iops_data_read_rolling_std_5m", "iops_data_read_pct_change_1h",
    "iops_data_write_rolling_mean_5m", "iops_data_write_rolling_std_5m", "iops_data_write_pct_change_1h",
    "read_latency_rolling_mean_5m", "read_latency_rolling_std_5m", "read_latency_pct_change_1h",
    "write_latency_rolling_mean_5m", "write_latency_rolling_std_5m", "write_latency_pct_change_1h",
]
_session = None

def score_partition(model,batches):
    global _session
    import onnxruntime as ort, numpy as np
    if _session is None:                                  
        _session = ort.InferenceSession(model.value)
    input_name = _session.get_inputs()[0].name
    for pdf in batches:
        X = pdf[FEATURE_ORDER].to_numpy(dtype=np.float32) 
        preds = _session.run(None, {input_name: X})[0]
        pdf = pdf.copy()
        pdf["predicted_class"] = preds.reshape(-1)
        yield pdf
def main(fleet_path, model_path, out_path):
    spark = SparkSession.builder.appName("tsd-batch-score").getOrCreate()   
    fleet = spark.read.parquet(fleet_path)
    feat = add_features(fleet)                      
    scorable = feat.dropna(subset=FEATURE_ORDER)
    out_schema = StructType(scorable.schema.fields + [StructField("predicted_class", LongType())])
    with open(model_path, "rb") as f:
        broadcast_model = spark.sparkContext.broadcast(f.read())
    scored = scorable.mapInPandas(partial(score_partition, broadcast_model), schema=out_schema).cache()

    scored_out = scored.withColumn("scoring_date", F.lit(str(date.today())))
    scored_out.write.mode("overwrite").partitionBy("scoring_date").parquet(out_path)

    summary = scored.groupBy("tier", "predicted_class").count().orderBy("tier", "predicted_class")
    summary.show()
    summary.write.mode("overwrite").parquet(f"{out_path}/summary/")
    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch scoring for TSD anomaly detection.")
    parser.add_argument("--fleet_path", type=str, required=True, help="Path to the fleet data in Parquet format.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the serialized model file.")
    parser.add_argument("--out_path", type=str, required=True, help="Output path for the scored data in Parquet format.")
    
    args = parser.parse_args()
    
    main(args.fleet_path, args.model_path, args.out_path)