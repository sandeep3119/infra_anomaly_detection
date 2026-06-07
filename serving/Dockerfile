FROM python:3.10-slim

WORKDIR /inference_service

COPY serving/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/app ./app
COPY  models ./models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
