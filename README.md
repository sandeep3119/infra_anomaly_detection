# Infastructure Telemetry Anomaly Detection

A production-grade anomaly detection pipeline for infrastructure telemetry, inspired by storage array monitoring patterns.

## Motivation

A typical storage arrays emit continuous telemetry — disk IOPS, read/write latency, CPU, memory — at high frequency. Detecting anomalies in this stream early (disk degradation, I/O error bursts, latency spikes, node failures) reduces mean time to detection and prevents cascading failures.

This project implements the full MLOps workflow: data simulation → feature engineering → model training → ONNX export → FastAPI inference service → drift detection → canary deployment.

## Architecture

```
Simulated Telemetry
       │
       ▼
Feature Engineering (rolling statistics, pct change)
       │
       ▼
Model Training (XGBoost + IsolationForest) ──► MLflow Registry
       │
       ▼
ONNX Export + Quantization 
       │
       ▼
FastAPI Inference Service (ONNX Runtime, Singleton loader)
       │
       ▼
Prometheus + Grafana (request latency, prediction distribution, drift metrics)
       │
       ▼
Drift Detection (PSI/KL divergence) ──► Retraining Trigger
       │
       ▼
Canary Deployment (K8s traffic split, automated rollback)
```

## Dataset

Synthetic Storage Array telemetry — 10,000 rows at 1-minute intervals (~7 days). Features include:

| Feature | Description |
|---|---|
| `userDataReadIops` | Read IOPS with time-of-day pattern (sinusoidal, peaks at noon) |
| `userDataWriteIops` | Write IOPS with correlated time-of-day pattern |
| `readLatencyMs` | Read latency in milliseconds |
| `writeLatencyMs` | Write latency in milliseconds |
| `cpuPercent` | Node CPU utilisation |
| `memoryPercent` | Node memory utilisation |

Four anomaly types injected into non-overlapping windows:

| Label | Type | Signal |
|---|---|---|
| 1 | Disk Degradation | Gradual IOPS decline over 500 rows (hardest to detect — slow drift) |
| 2 | I/O Error Burst | Sudden 3x write IOPS spike + 5x write latency increase |
| 3 | Latency Spike | Read/write latency jumps by 10-15ms, IOPS unaffected |
| 4 | Node Failure | All metrics drop to near-zero |

Class distribution: ~91% normal, ~9% anomalous (realistic imbalance for infrastructure telemetry).

## Feature Engineering

Rolling features computed over 5-minute windows and 1-hour pct change for all IOPS and latency columns:

- `*_rolling_mean_5m` — smoothed signal, filters noise
- `*_rolling_std_5m` — captures volatility, spikes in variance precede failures
- `*_pct_change_1h` — key signal for disk degradation (gradual decline invisible in raw values)

XGBoost operates on single rows — temporal context must be engineered explicitly.

## Project Structure

```
TSD/
├── data/
│   └── simulated_data.csv           # Generated telemetry dataset
├── models/
│   ├── xgboost_tsd_model.onnx       # Exported ONNX model
│   └── xgboost_tsd_model_quantized.onnx  # Quantized ONNX model
├── notebooks/
│   └── TSD_001_data_simulation.ipynb     # Simulation, feature engineering, EDA
├── serving/
│   └── app/
│       ├── main.py          # FastAPI app, lifespan, /predict, /predict/batch
│       ├── inference.py     # ONNXInferenceEngine Singleton
│       ├── schema.py        # InferenceRequest (18 fields), InferenceResponse
│       ├── health_check.py  # /health/live, /health/ready
│       ├── config.py        # Pydantic BaseSettings, reads .env
│       └── logger.py        # JSON structured logging
├── src/
│   └── simulate.py          # Reusable simulation module
└── requirements.txt
```

## Stack

- **Data:** pandas, numpy
- **ML:** scikit-learn, XGBoost, ONNX Runtime
- **Experiment tracking:** MLflow
- **Serving:** FastAPI, ONNX Runtime, uvicorn
- **Monitoring:** Prometheus, Grafana
- **Infra:** Kubernetes, Docker

## Status

| Ticket | Status | Description |
|---|---|---|
| TSD-001 | ✅ Done | Data simulation, feature engineering, EDA |
| TSD-002 | ✅ Done | XGBoost (multi:softprob, 5-class, sample_weight balanced) + IsolationForest. Both tracked in MLflow with params, metrics, classification report. Registered with `champion` alias in MLflow model registry. |
| TSD-003 | ✅ Done | XGBoost exported to ONNX via onnxmltools. Benchmark: native XGBoost 2x faster than ONNX Runtime (expected for tree ensembles — no operator fusion). Dynamic int8 quantization applied — no size reduction (tree models have no weight matrices). ONNX value: portability + single runtime dependency in serving container. |
| TSD-004 | ✅ Done | FastAPI inference service — ONNX Runtime serving, Singleton model loader (double-checked locking), `/predict` (single) + `/predict/batch`, `/health/live` + `/health/ready`, structured JSON logging. Client sends all 18 features (6 raw + 12 rolling). Inference latency ~18ms. Known limitation: rolling features computed client-side — per-device server-side buffer deferred to TSD-004b (requires Redis, absorbed into TSD-006). |
| TSD-005 | ✅ Done | MLflow registry integration + hot-reload — model loaded from registry at startup via `champion` alias (no model baked into image). Background daemon thread polls MLflow every 60s, atomically swaps `app.state.model` with `threading.Lock` on new champion version. Previous model retained in `app.state.previous_model` for rollback. ONNX artifact logged via `mlflow.onnx.log_model`. Startup time ~5s (MLflow download) vs ~80ms (local file). |
| TSD-006 | ⏳ Pending | Prometheus + Grafana monitoring |
| TSD-007 | ⏳ Pending | Drift detection + automated retraining trigger |
| TSD-008 | ⏳ Pending | Canary deployment with automated rollback |
