import redis
import pandas as pd
from .drift import drift_evidently
from prometheus_client import Gauge, start_http_server
import time
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("drift_runner")
RAW_FIELDS = ["userDataReadIops","userDataWriteIops","readLatencyMs","writeLatencyMs","cpuPercent","memoryPercent"]
drift_gauge = Gauge("feature_drift_score", "Per-feature drift score", ["feature"])
drift_alert_gauge = Gauge("drift_alert", "1 if any feature breached threshold")
THRESHOLD = 1.0  
DRIFT_INTERVAL = 300
MIN_SAMPLES =100


def connect_redis_get_reference():
    r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
    df = pd.read_csv("data/simulated_data.csv")
    reference_df = df[df["label"] == 0][RAW_FIELDS]
    return r,reference_df
def calculate_live_drift(r, reference_df):
    raw = r.lrange("drift:samples", 0, -1)
    if len(raw) < MIN_SAMPLES:
        logger.info("Not Enough samples yet skipping the cycle",extra={"count":len(raw)})
        return None
    rows = [[float(x) for x in s.split(",")] for s in raw]
    current_df = pd.DataFrame(rows, columns=RAW_FIELDS)
    drift_scores = drift_evidently( reference_df,current_df,RAW_FIELDS)
    return drift_scores
def raise_drift_alert(drifted):
    logger.warning(
        "DRIFT ALERT — retraining review required",
        extra={"drifted_features": drifted}
    )
    drift_alert_gauge.set(1)  

def main():
    start_http_server(8001)
    r,reference_df = connect_redis_get_reference()
    while True:
        scores = calculate_live_drift(r, reference_df)
        if scores is None:
            time.sleep(DRIFT_INTERVAL)
            continue
        for feature,score in scores.items():
            drift_gauge.labels(feature=feature).set(score)
        drifted = {f: s for f, s in scores.items() if s > THRESHOLD}
        if drifted:
            raise_drift_alert(drifted)
        else:
            logger.info("no drift", extra={"max_score": max(scores.values())})
            drift_alert_gauge.set(0)
        time.sleep(DRIFT_INTERVAL)


if __name__ == "__main__":
    main()
 