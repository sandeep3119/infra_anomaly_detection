from fastapi import FastAPI
from .config import settings
from .logger import setup_logger
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .schema import InferenceRequest,InferenceResponse,BatchInferenceRequest
import time
import numpy as np
import uuid
from .inference import ONNXInferenceEngine
from .health_check import router as HealthCheckRouter
import os
import threading
import mlflow



logger = setup_logger("infa_anomaly_detection", settings.log_level)

label_map = {
            0 :"normal",
            1 :"disk_degradation",
            2 : "io_error_burst",
            3 : "latency_spike",
            4 : "node_failure"
        }

def load_model(app):
    logger.info(f"Download Model initiated ....")
    model_path = mlflow.artifacts.download_artifacts(
            f"models:/{settings.model_registry_name}@{settings.model_alias}"
    )
    version_info = app.state.mlflow_client.get_model_version_by_alias(settings.model_registry_name, settings.model_alias)
    model_file = [f for f in os.listdir(model_path) if f.endswith(".onnx")][0]
    model = ONNXInferenceEngine(model_path=os.path.join(model_path,model_file))
    app.state.model = model
    app.state.model_version = int(version_info.version)
    logger.info(f"Model Loaded Successfully.", extra={"model_version":app.state.model_version})
    return model

def poll_model(app,interval): 
    while True:
        try:

            time.sleep(interval)
            version_info = app.state.mlflow_client.get_model_version_by_alias(settings.model_registry_name, settings.model_alias)
            new_version = int(version_info.version)
            if new_version > app.state.model_version:
                logger.info(f"New champion model found, updating the current model")
                new_model_path = mlflow.artifacts.download_artifacts(
                    f"models:/{settings.model_registry_name}@{settings.model_alias}"
                )
                model_file = [f for f in os.listdir(new_model_path) if f.endswith(".onnx")][0]
                new_model = ONNXInferenceEngine(os.path.join(new_model_path,model_file))
                with app.state.model_lock:
                    app.state.previous_model = app.state.model  # keep old one
                    app.state.previous_version = app.state.model_version
                    app.state.model = new_model
                    app.state.model_version = new_version
            else:
                logger.info(f"No new champion model found.. Will poll again in next {interval} secs")
        except Exception as e:
            logger.error(f"Error occured while polling the model : {e}")
   


@asynccontextmanager
async def lifespan(app:FastAPI):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        app.state.mlflow_client = mlflow.tracking.MlflowClient()
        model = await loop.run_in_executor(pool, load_model,app)
        app.state.model_lock = threading.Lock()

    poll_thread = threading.Thread(
        target=poll_model,
        args = (app,settings.poll_interval_secs),
        daemon=True
    )
    poll_thread.start()
    yield
    app.state.model = None

# Create FastAPI app
app = FastAPI(
    title="TSD_anomaly_detection",
    description="Infra Time series data anomaly detection",
    version="1.0.0",
    lifespan=lifespan,
    redirect_slashes=False
)



app.include_router(HealthCheckRouter)

@app.post("/predict")
async def predict(request:InferenceRequest) -> InferenceResponse:
    logger.info("Request Recieved ...")
    start_time= time.perf_counter()
    request_id = str(uuid.uuid4())
    input = np.array([[
        request.userDataReadIops,
        request.userDataWriteIops,
        request.readLatencyMs,
        request.writeLatencyMs,
        request.cpuPercent,
        request.memoryPercent,
        request.iops_data_read_rolling_mean_5m,
        request.iops_data_read_rolling_std_5m,
        request.iops_data_read_pct_change_1h,
        request.iops_data_write_rolling_mean_5m,
        request.iops_data_write_rolling_std_5m,
        request.iops_data_write_pct_change_1h,
        request.read_latency_rolling_mean_5m,
        request.read_latency_rolling_std_5m,
        request.read_latency_pct_change_1h,
        request.write_latency_rolling_mean_5m,
        request.write_latency_rolling_std_5m,
        request.write_latency_pct_change_1h
        ]],dtype=np.float32)
    with app.state.model_lock:
        predicted_class,confidence,all_probs = app.state.model.predict(input)
    end_time = time.perf_counter()
    latency_ms= round(((end_time - start_time) * 1000),4)
    inference_obj = InferenceResponse(
        request_id = request_id,
        identified_anomaly=predicted_class,
        anomaly_type=label_map[predicted_class],
        confidence= confidence,
        latency_ms= latency_ms
    )
    logger.info("Prediction complete",extra={
        "request_id":request_id,
        "predicted_class":predicted_class,
        "confidence": confidence,
        "all_prob": all_probs,
        "latency_ms": latency_ms

    })
    return inference_obj

@app.post("/predict/batch")
async def predict_batch(request:BatchInferenceRequest) -> list[InferenceResponse]:
    inference_batch_response = []
    for req in request.requests:
        prediction = await predict(req)
        inference_batch_response.append(prediction)
    
    return inference_batch_response

    