# MLflow tracking server with state baked in (db + artifacts).
# Solves two problems vs the Compose python:3.10-slim + pip-install approach:
#   1. Deps are pre-installed -> fast pod startup (no pip install at runtime).
#   2. mlflow.db + mlartifacts are copied in -> the registered champion model
#      exists, so the inference pod can load it at startup.
# Build from the project root:  docker build -f k8s/mlflow.Dockerfile -t tsd-mlflow:latest .
FROM python:3.10-slim
WORKDIR /mlflow
RUN pip install --no-cache-dir mlflow flask boto3
COPY mlflow.db ./mlflow.db
COPY mlartifacts ./mlartifacts
EXPOSE 5000
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--default-artifact-root", "s3://mlflow/"]
