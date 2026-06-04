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
│   └── simulated_data.csv       # Generated telemetry dataset
├── notebooks/
│   └── TSD_001_data_simulation.ipynb   # Simulation, feature engineering, EDA
├── src/
│   └── simulate.py              # Reusable simulation module
└── requirements.txt
```

## Stack

- **Data:** pandas, numpy
- **ML:** scikit-learn, XGBoost, ONNX Runtime
- **Experiment tracking:** MLflow
- **Serving:** FastAPI
- **Monitoring:** Prometheus, Grafana
- **Infra:** Kubernetes, Docker

## Status

| Ticket | Status | Description |
|---|---|---|
| TSD-001 | ✅ Done | Data simulation, feature engineering, EDA |
| TSD-002 | ✅ Done | XGBoost (multi:softprob, 5-class, sample_weight balanced) + IsolationForest. Both tracked in MLflow with params, metrics, classification report. Registered with `champion` alias in MLflow model registry. |
| TSD-003 | ⏳ Pending | ONNX export, quantization, inference benchmarking |
| TSD-004 | ⏳ Pending | FastAPI inference service with ONNX Runtime |
| TSD-005 | ⏳ Pending | MLflow model registry, champion/challenger, hot-reload |
| TSD-006 | ⏳ Pending | Prometheus + Grafana monitoring |
| TSD-007 | ⏳ Pending | Drift detection + automated retraining trigger |
| TSD-008 | ⏳ Pending | Canary deployment with automated rollback |
