
import pandas as pd
from .config import settings


RAW_FIELDS = ["userDataReadIops","userDataWriteIops","readLatencyMs","writeLatencyMs","cpuPercent","memoryPercent"]
DRIFT_KEY = "drift:samples"          # module-level, one source of truth (like RAW_FIELDS)
DRIFT_MAX = 5000

FEATURE_ORDER = [
    "userDataReadIops", "userDataWriteIops", "readLatencyMs", "writeLatencyMs", "cpuPercent", "memoryPercent",
    "iops_data_read_rolling_mean_5m", "iops_data_read_rolling_std_5m", "iops_data_read_pct_change_1h",
    "iops_data_write_rolling_mean_5m", "iops_data_write_rolling_std_5m", "iops_data_write_pct_change_1h",
    "read_latency_rolling_mean_5m", "read_latency_rolling_std_5m", "read_latency_pct_change_1h",
    "write_latency_rolling_mean_5m", "write_latency_rolling_std_5m", "write_latency_pct_change_1h",
]

def _key(device_id):
    return f"device:{device_id}"

def push_and_fetch(r,device_id,reading):
    reading = ",".join(str(x) for x in reading)
    pipe = r.pipeline()          
    pipe.rpush(_key(device_id), reading)
    pipe.ltrim(_key(device_id), -61, -1)
    pipe.expire(_key(device_id), settings.redis_buffer_ttl_secs)
    pipe.llen(_key(device_id))
    pipe.lrange(_key(device_id), 0, -1)
    results = pipe.execute()
    window = results[4]
    rows =[]
    for row in window:
        rows.append( [float(x) for x in row.split(',')])
    return results[3],rows


def compute_features(rows):
    df = pd.DataFrame(rows, columns= RAW_FIELDS)

    df["iops_data_read_rolling_mean_5m"]=df['userDataReadIops'].rolling(window=5).mean()
    df["iops_data_read_rolling_std_5m"]=df['userDataReadIops'].rolling(window=5).std()
    df["iops_data_read_pct_change_1h"]=df['userDataReadIops'].pct_change(periods=60)


    df["iops_data_write_rolling_mean_5m"]=df['userDataWriteIops'].rolling(window=5).mean()
    df["iops_data_write_rolling_std_5m"]=df['userDataWriteIops'].rolling(window=5).std()
    df["iops_data_write_pct_change_1h"]=df['userDataWriteIops'].pct_change(periods=60)


    df["read_latency_rolling_mean_5m"]=df['readLatencyMs'].rolling(window=5).mean()
    df["read_latency_rolling_std_5m"]=df['readLatencyMs'].rolling(window=5).std()
    df["read_latency_pct_change_1h"]=df['readLatencyMs'].pct_change(periods=60)

    df["write_latency_rolling_mean_5m"]=df['writeLatencyMs'].rolling(window=5).mean()
    df["write_latency_rolling_std_5m"]=df['writeLatencyMs'].rolling(window=5).std()
    df["write_latency_pct_change_1h"]=df['writeLatencyMs'].pct_change(periods=60)
    return df[FEATURE_ORDER].tail(1)


def add_drift_samples(r, reading):
    reading = ",".join(str(x) for x in reading)         
    r.rpush(DRIFT_KEY, reading)
    r.ltrim(DRIFT_KEY, -DRIFT_MAX, -1)