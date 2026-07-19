# Infastructure Telemetry Anomaly Detection

A production-grade anomaly detection pipeline for infrastructure telemetry, inspired by storage array monitoring patterns.

## Motivation

A typical storage arrays emit continuous telemetry — disk IOPS, read/write latency, CPU, memory — at high frequency. Detecting anomalies in this stream early (disk degradation, I/O error bursts, latency spikes, node failures) reduces mean time to detection and prevents cascading failures.

This project implements the full MLOps workflow: data simulation → feature engineering → model training → ONNX export → FastAPI inference service → server-side feature store → drift detection → Kubernetes progressive delivery → offline batch scoring.

## Architecture

The same champion model serves **two consumers**: a latency-optimised online path and a
throughput-optimised offline batch path.

```
Simulated Fleet Telemetry (N devices, tiered)
       │
       ▼
Feature Engineering (rolling statistics, pct change)
       │
       ▼
Model Training (XGBoost + IsolationForest) ──► MLflow Registry (champion alias)
       │
       ▼
ONNX Export + Quantization
       │
       ├─────────────────────────────────┬──────────────────────────────────┐
       ▼                                 ▼                                  │
  ONLINE PATH (TSD-004..009)       OFFLINE BATCH PATH (TSD-010)             │
  latency-bound          throughput-bound, scheduled              │
       │                                 │                                  │
  FastAPI + ONNX Runtime           Read fleet parquet (Spark)               │
  ◄── registry hot-reload               │                                   │
       │                           Distributed feature engineering          │
  Redis feature store              Window.partitionBy(device_id)            │
  (per-device rolling buffer)      → rolling mean/std, pct_change           │
       │                                 │                                  │
  real-time alerting               Batch score (broadcast ONNX bytes        │
       │                            + mapInPandas)                          │
       │                                 │                                  │
       │                           Fleet anomaly report (parquet)           │
       │                                                                    │
       ├──► Prometheus + Grafana (golden signals, prediction distribution, drift)
       │                                                                    │
       ▼                                                                    │
Drift Detection (Evidently, scheduled runner) ──► alert / human-gated retrain
       │                                                                    │
       ▼                                                                    │
Kubernetes (minikube): blue-green (Service selector flip)  ◄────────────────┘
                       + canary (ingress-nginx weighted routing)
```

## Online vs Offline Batch

Two independent axes run through this pipeline, both driven by the **same champion model
resolved from the MLflow registry by alias** — so both consumers provably score with the
same model version:

| Axis | Online | Offline / batch |
|---|---|---|
| **Serving** | live, per-request, latency-bound (FastAPI + Redis, ~9.5 ms) | scheduled, bulk, throughput-bound (Spark) |
| **Features** | computed per request from the Redis buffer | computed in bulk (Spark window functions) |

**Why both?** The online path answers *"is this device anomalous right now?"* and is
optimised for latency. It is the wrong tool for three jobs operations still needs:

- **Different SLAs.** Lower-priority telemetry doesn't need sub-10ms alerts; scoring it on
  a schedule is sufficient and far cheaper than keeping it in the hot path.
- **Backfill.** The online model only scored data that arrived *after* it went live. When a
  new champion is promoted, re-scoring history is a bulk job the one-row-at-a-time API
  cannot do.
- **Fleet-wide, long-horizon questions.** The Redis buffer is a *sliding window* (depth 61,
  sliding TTL) — by design it has no long memory. *"Which devices anomalied most this
  quarter?"* requires a durable historical store and cross-device aggregation.

These are high-volume, throughput-bound, no-one-is-waiting jobs — the opposite profile from
online serving, and exactly what batch compute is built for.

**Why not just log online predictions and query them?** For data that flowed through the
online path, you can. But a prediction log only contains what the API saw: post-launch,
high-priority telemetry. It cannot cover history predating the service, or telemetry that
deliberately never hits the online path. Prediction logging and batch scoring are
**complementary** — logging captures *"what did we predict live"*, batch covers
*"score everything else, in bulk"*.

**The shared constraint — training/serving skew.** The batch path introduces a *second*
feature-computation implementation (Spark), which is a fresh opportunity to reintroduce the
skew TSD-007 eliminated. The batch features must be numerically identical to the training
pandas math: same window size and boundaries, same `pct_change` lookback (`periods=60` →
`lag(60)`), same sample-vs-population `std`, same cold-start nulling (Spark computes
*partial* windows where pandas emits `NaN` — partial rows must be explicitly nulled), same
column order. A pandas-vs-Spark **equivalence check** over an overlapping slice is the guard
that proves skew hasn't crept back in.

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
│   └── simulated_data.csv               # Generated telemetry dataset
├── models/
│   ├── xgboost_tsd_model.onnx           # Exported ONNX model
│   └── xgboost_tsd_model_quantized.onnx # Quantized ONNX model
├── notebooks/
│   ├── TSD_001_data_simulation+EDA+Feature_Engineering.ipynb
│   ├── TSD_001_CLass_Imbalance.ipynb
│   ├── TSD_002_model_training.ipynb
│   ├── TSD_003_onnx_export.ipynb
│   └── TSD_009_drift_detection.ipynb    # (drift exploration; misnamed — predates TSD-008)
├── serving/
│   ├── app/
│   │   ├── main.py            # FastAPI app, lifespan, /predict, /predict/batch, metrics, registry hot-reload
│   │   ├── inference.py       # ONNXInferenceEngine (ONNX Runtime session)
│   │   ├── feature_store.py   # Redis per-device rolling buffer + server-side feature computation (TSD-007)
│   │   ├── schema.py          # InferenceRequest (deviceID + 6 raw features), InferenceResponse
│   │   ├── health_check.py    # /health/live, /health/ready
│   │   ├── config.py          # Pydantic BaseSettings (mlflow, redis, ttl)
│   │   └── logger.py          # JSON structured logging
│   ├── Dockerfile             # ONNX-Runtime serving image
│   ├── Dockerfile.drift       # Drift-runner image
│   └── requirements.txt
├── monitoring/
│   ├── prometheus.yml         # Scrape config (inference + drift_runner)
│   └── grafana/dashboards/    # Version-controlled dashboard JSON
├── src/
│   ├── simulate.py            # Single-device telemetry simulation
│   ├── simulate_fleet.py      # Multi-device fleet simulator → partitioned parquet (TSD-010)
│   ├── train.py              # Training + evaluation functions
│   ├── drift.py              # PSI (from-scratch) + Evidently drift computation (TSD-008)
│   ├── drift_runner.py       # Scheduled drift detector — Prometheus gauges + alert
│   ├── batch_feature.py      # Spark window-function features + pandas equivalence check (TSD-010)
│   └── batch_scoring.py      # Spark batch scoring — MLflow champion, ONNX, s3a reports (TSD-010)
├── scripts/
│   └── simulate_traffic.py    # Telemetry traffic generator (normal / --drift)
├── k8s/                       # Kubernetes manifests (TSD-009, TSD-010)
│   ├── namespace.yaml
│   ├── redis.yaml
│   ├── mlflow.Dockerfile / mlflow.yaml   # MLflow with state baked into image
│   ├── inference.yaml / inference-green.yaml  # blue + green deployments
│   ├── drift-runner.yaml
│   ├── prometheus.yaml / grafana.yaml
│   ├── ingress.yaml           # blue-green selector flip + canary weighted routing
│   ├── minio.yaml             # S3 object store (data lake)
│   └── batch-job.yaml / batch-cronjob.yaml   # offline batch scoring, one-shot + scheduled
├── serving/Dockerfile.batch   # PySpark batch scoring image
├── docker-compose.yml         # mlflow + inference + prometheus + grafana + redis + drift_runner
└── requirements.txt
```

## Stack

- **Data:** pandas, numpy
- **ML:** scikit-learn, XGBoost, ONNX Runtime
- **Experiment tracking:** MLflow (registry, champion alias, hot-reload)
- **Serving:** FastAPI, ONNX Runtime, uvicorn
- **Feature store:** Redis (per-device rolling buffer)
- **Batch:** PySpark, MinIO (S3)
- **Monitoring:** Prometheus, Grafana
- **Drift detection:** Evidently
- **Infra:** Docker, Kubernetes (minikube), ingress-nginx
- **Progressive delivery:** blue-green (Service selector) + canary (ingress weighted routing)

## Getting Started

### Prerequisites
- Docker + Docker Compose
- Python 3.10 (for the traffic generator / notebooks)
- For the Kubernetes path: `minikube` + `kubectl`

### Option A — Docker Compose (local dev)

Brings up the full stack: MLflow, inference, Redis, drift-runner, Prometheus, Grafana.

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Inference API (Swagger) | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 |

Generate telemetry traffic (warms the per-device buffers and feeds drift detection):

```bash
pip install -r requirements.txt
python scripts/simulate_traffic.py            # normal traffic
python scripts/simulate_traffic.py --drift    # drifted traffic (spikes readLatencyMs)
```

The drift runner evaluates every `DRIFT_INTERVAL` (default 300s) and exposes
`feature_drift_score` / `drift_alert` on http://localhost:8001/metrics.

### Option B — Kubernetes (minikube)

```bash
minikube start --cpus=4 --memory=6144
eval $(minikube docker-env)                    # build images into minikube's daemon

docker build -f k8s/mlflow.Dockerfile     -t tsd-mlflow:latest .
docker build -f serving/Dockerfile        -t tsd-inference:latest .
docker build -f serving/Dockerfile.drift  -t tsd-drift:latest .

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/redis.yaml -f k8s/mlflow.yaml
kubectl apply -f k8s/inference.yaml -f k8s/drift-runner.yaml
kubectl apply -f k8s/prometheus.yaml -f k8s/grafana.yaml
kubectl get pods -n tsd                         # wait for all Running/Ready
```

**Blue-green** (deploy green, flip the Service selector, roll back):

```bash
kubectl apply -f k8s/inference-green.yaml
kubectl patch svc inference -n tsd -p '{"spec":{"selector":{"app":"tsd-inference","version":"green"}}}'
kubectl patch svc inference -n tsd -p '{"spec":{"selector":{"app":"tsd-inference","version":"blue"}}}'   # rollback
```

**Canary** (weighted split via ingress-nginx; adjust the weight to progress/roll back):

```bash
minikube addons enable ingress
kubectl apply -f k8s/ingress.yaml
kubectl patch ingress inference-canary -n tsd --type=merge \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/canary-weight":"50"}}}'
```

## Monitoring Dashboard

Single Grafana dashboard covering the four golden signals (TSD-006) and data-drift
detection (TSD-008). Drift panels: per-feature drift score with a threshold line,
a 0/1 drift-alert status, and the max drift score across features.

![TSD monitoring dashboard](misc/Dashboard.png)

## Status

| Ticket | Status | Description |
|---|---|---|
| TSD-001 | ✅ Done | Data simulation, feature engineering, EDA |
| TSD-002 | ✅ Done | XGBoost (multi:softprob, 5-class, sample_weight balanced) + IsolationForest. Both tracked in MLflow with params, metrics, classification report. Registered with `champion` alias in MLflow model registry. |
| TSD-003 | ✅ Done | XGBoost exported to ONNX via onnxmltools. Benchmark: native XGBoost 2x faster than ONNX Runtime (expected for tree ensembles — no operator fusion). Dynamic int8 quantization applied — no size reduction (tree models have no weight matrices). ONNX value: portability + single runtime dependency in serving container. |
| TSD-004 | ✅ Done | FastAPI inference service — ONNX Runtime serving, Singleton model loader (double-checked locking), `/predict` (single) + `/predict/batch`, `/health/live` + `/health/ready`, structured JSON logging. Client sends all 18 features (6 raw + 12 rolling). Inference latency ~18ms. Known limitation: rolling features computed client-side — per-device server-side buffer deferred to TSD-004b (requires Redis, absorbed into TSD-006). |
| TSD-005 | ✅ Done | MLflow registry integration + hot-reload — model loaded from registry at startup via `champion` alias (no model baked into image). Background daemon thread polls MLflow every 60s, atomically swaps `app.state.model` with `threading.Lock` on new champion version. Previous model retained in `app.state.previous_model` for rollback. ONNX artifact logged via `mlflow.onnx.log_model`. Startup time ~5s (MLflow download) vs ~80ms (local file). |
| TSD-006 | ✅ Done | Prometheus + Grafana observability — full stack via Docker Compose (mlflow, inference, prometheus, grafana). Custom metrics: `prediction_class_total` (Counter, labelled by class), `model_version` (Gauge). Auto metrics via `prometheus-fastapi-instrumentator`: request count, latency histogram. Grafana dashboard covers all 4 golden signals — Latency (P95), Traffic (request rate), Errors (4xx/5xx rate), Saturation (resident memory) — plus prediction distribution + live model version. Dashboard JSON version-controlled in `monitoring/grafana/dashboards/`. |
| TSD-007 | ✅ Done | Redis per-device rolling buffer (promoted from TSD-004b) — feature computation moved server-side. Client now sends `deviceID` + 6 raw features only; server maintains a per-device history list in Redis (`device:{id}` key) and computes the 12 rolling features, eliminating client-side training/serving skew. Buffer depth = 61 (max feature lookback: `pct_change(periods=60)` needs 60 prior readings + the current one). Rolling features reuse the exact training-time pandas math, with an explicit `FEATURE_ORDER` select to enforce column order into the ONNX model. Cold-start gate returns HTTP 425 until the buffer holds 61 readings (reject over impute — fabricated features are indistinguishable from real ones to the model). Append + trim + expire + count + read run in a single Redis pipeline (MULTI/EXEC) so concurrent requests for the same device cannot interleave and corrupt the window — the distributed analogue of the `threading.Lock` used for the model swap in TSD-005. Sliding TTL (2h, > the ~61-min buffer span) auto-reclaims dead devices; AOF persistence intentionally off (buffer is reconstructible — a restarted device re-warms in ~61 readings). Redis added as a 5th Docker Compose service. Inference latency ~9.5ms steady-state (Redis round-trip sub-millisecond). |
| TSD-008 | ✅ Done | Data drift detection — monitors covariate shift on the 6 raw features (data drift is detectable label-free; concept drift is not, since live anomaly labels are unavailable). Serving service appends each incoming reading to a separate global Redis list (`drift:samples`, capped ~5000, no TTL — deliberately different config from the TSD-007 per-device buffer: a *population* sample for distribution estimation vs a *point* query for current features), under a best-effort `try/except` so a monitoring failure never breaks `/predict`. A standalone scheduled `drift_runner` process (6th Compose service, decoupled failure domain from serving) compares a live window against a **normal-only reference** (`label==0` rows — anomalies excluded so the baseline means "normal," not "normal+anomalies") using **Evidently 0.7.x** (`DataDriftPreset`, Wasserstein distance). Threshold (1.0) calibrated empirically against the observed noise floor (~0.18) rather than a library default — clean separation from the ~20 drift signal. Per-feature scores exposed as a labelled Prometheus gauge (`feature_drift_score`) plus a 0/1 `drift_alert` gauge for Grafana alerting; runs an own `start_http_server` (long-lived loop, since Prometheus cannot scrape a batch job that exits — Pushgateway is the alternative). On breach: **alert only**, not auto-retrain — retraining stays human-gated because auto-retraining on unlabelled drift can teach the model to accept a degraded state as normal, and the drift sample window is sized for *detection*, not *training* (real retraining sources full-volume data from a warehouse). Also implemented PSI from scratch (quantile bins, frozen reference edges, epsilon smoothing for `ln(0)`) in `src/drift.py` as the "understand it before using the tool" version. Robustness: runner tolerates an empty/cold sample store (skips cycle below `MIN_SAMPLES`). |
| TSD-009 | ✅ Done | Kubernetes deployment + progressive delivery (blue-green & canary). Full 6-service stack ported from Docker Compose to minikube (`k8s/` manifests): namespace, Redis, MLflow (state baked into a custom image — `mlflow.db` + `mlartifacts` copied in so the champion is present, vs Compose's host bind-mount), inference (blue), drift-runner, Prometheus (config via ConfigMap, FQDN scrape targets), Grafana (Prometheus datasource auto-provisioned via ConfigMap — the GitOps fix for TSD-006's manual setup). **Blue-green:** blue + green Deployments share `app: tsd-inference`, differ in a `version` label; the Service selector pins one version, and a one-line `kubectl patch` flips all traffic blue↔green (verified live by watching Service endpoints move between pod IPs) — instant switch, instant rollback, zero downtime (K8s only adds Ready pods to endpoints, so green must be Ready before the flip). **Canary:** ingress-nginx weighted routing — two Ingresses on the same host, one marked `canary: "true"` with `canary-weight` sending an exact % to a green-backed canary Service (true weight-based split, independent of pod count, unlike the native replica-ratio approach); progression/rollback by patching the weight annotation.|
| TSD-010 | ✅ Done | Offline batch scoring on Spark, complementing (not replacing) the online path. Multi-device fleet simulator writes partitioned parquet to MinIO (S3). Spark computes the 12 rolling features with window functions, validated numerically against the pandas training features (equivalence check, 1e-6 tolerance). Scores with the champion resolved from MLflow by alias — model bytes broadcast to executors, ONNX session built per-executor — and writes date-partitioned reports back to MinIO. Runs as a Kubernetes CronJob. |
