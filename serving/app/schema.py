from pydantic import BaseModel



class InferenceRequest(BaseModel):
    userDataReadIops: float
    userDataWriteIops: float
    readLatencyMs: float
    writeLatencyMs: float
    cpuPercent: float
    memoryPercent: float
    iops_data_read_rolling_mean_5m: float
    iops_data_read_rolling_std_5m: float
    iops_data_read_pct_change_1h : float
    iops_data_write_rolling_mean_5m: float
    iops_data_write_rolling_std_5m: float
    iops_data_write_pct_change_1h: float
    read_latency_rolling_mean_5m: float
    read_latency_rolling_std_5m: float
    read_latency_pct_change_1h: float
    write_latency_rolling_mean_5m: float
    write_latency_rolling_std_5m: float
    write_latency_pct_change_1h: float

class BatchInferenceRequest(BaseModel):
    requests: list[InferenceRequest]


class InferenceResponse(BaseModel):
    request_id : str
    identified_anomaly : int
    anomaly_type : str
    confidence : float
    latency_ms : float
