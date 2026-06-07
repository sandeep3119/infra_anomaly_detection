from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mlflow_tracking_uri: str = "http://localhost:5000"
    local_model_dir: str = "models/"
    model_name: str = "xgboost_tsd_model_fixed.onnx"
    model_alias: str = "champion"
    model_registry_name: str = "xgboost_tsd_model_fixed"
    poll_interval_secs: int = 60
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"


settings = Settings()