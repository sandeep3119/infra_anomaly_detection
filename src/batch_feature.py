from pyspark.sql import Window, functions as F
import numpy as np
FEATURE_MAP = {
    "userDataReadIops":  "iops_data_read",
    "userDataWriteIops": "iops_data_write",
    "readLatencyMs":     "read_latency",
    "writeLatencyMs":    "write_latency",
}



def add_features(df):
    w_roll  = Window.partitionBy("device_id").orderBy("timestamp").rowsBetween(-4, 0)
    w_order = Window.partitionBy("device_id").orderBy("timestamp")
    cnt = F.count("*").over(w_roll)          # rows in the 5-window, per device
    for col, prefix in FEATURE_MAP.items():
        df = (df
            .withColumn(f"{prefix}_rolling_mean_5m",
                        F.when(cnt < 5, None).otherwise(F.avg(col).over(w_roll)))
            .withColumn(f"{prefix}_rolling_std_5m",
                        F.when(cnt < 5, None).otherwise(F.stddev(col).over(w_roll)))
            .withColumn(f"{prefix}_pct_change_1h",
                        (F.col(col) - F.lag(col, 60).over(w_order)) / F.lag(col, 60).over(w_order))
        )
    return df

def compare_spark_pandas_features(spark_df, raw_pd):
    """
    Compare features computed by Spark and Pandas.
    """
    for col, prefix in FEATURE_MAP.items():
        raw_pd[f"{prefix}_rolling_mean_5m"] = raw_pd[col].rolling(5).mean()
        raw_pd[f"{prefix}_rolling_std_5m"]  = raw_pd[col].rolling(5).std()
        raw_pd[f"{prefix}_pct_change_1h"]   = raw_pd[col].pct_change(60)
    feat_cols = [f"{p}_{s}" for p in FEATURE_MAP.values()
             for s in ("rolling_mean_5m", "rolling_std_5m", "pct_change_1h")]
    for c in feat_cols:
        match = np.allclose(spark_df[c].values, raw_pd[c].values, rtol=1e-6, atol=1e-6, equal_nan=True)
        print(f"{'Match' if match else 'Discrepency'}  {c}")