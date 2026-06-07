import onnxruntime as rt
import numpy as np
import threading
import logging

logger = logging.getLogger("infa_anomaly_detection")

class ONNXInferenceEngine:
    
    def __init__(self, model_path: str):
        self.runtime = rt.InferenceSession(model_path)
        logger.info("Inference Session Created ...")
        
    
    def predict(self, features: np.ndarray) -> tuple[int, float, list[float]]:
            onnx_inputs = {"input": features}
            onnx_outputs = self.runtime.run(None, onnx_inputs)
            predicted_class = int(onnx_outputs[0][0])
            all_probs = onnx_outputs[1][0].tolist()
            confidence = max(all_probs)
            return predicted_class,confidence,all_probs