FROM python:3.10-slim

WORKDIR /inference_service

COPY serving/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
