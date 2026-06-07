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


logger = setup_logger("infa_anomaly_detection", settings.log_level)

label_map = {
            0 :"normal",
            1 :"disk_degradation",
            2 : "io_error_burst",
            3 : "latency_spike",
            4 : "node_failure"
        }



def load_model(model_path):
    full_path = os.path.join(model_path,settings.model_name)
    logger.info(f"Load Model initiated ....",extra={"model_dir":full_path})
    inference_engine_obj = ONNXInferenceEngine.get_instance(model_path=full_path)
    logger.info(f"Model Loaded Successfully.")
    return inference_engine_obj

@asynccontextmanager
async def lifespan(app:FastAPI):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(pool, load_model,settings.local_model_dir)
        app.state.model = result
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

    