# Backend image: FastAPI + transformers (CPU inference).
# Runs locally (docker compose) and on Hugging Face Spaces (Docker SDK) as-is.
FROM python:3.12-slim

WORKDIR /app

# Install CPU-only PyTorch first to avoid pulling CUDA wheels.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# The API needs the app plus the demo corpus, dataset stats and real metrics.
COPY app ./app
COPY data ./data
COPY models ./models

# HF Spaces runs the container with a read-only home; keep the model cache in a
# writable location. /tmp works both there and under docker compose locally.
ENV MODEL_NAME=l3cube-pune/marathi-ner \
    MAX_LENGTH=256 \
    HF_HOME=/tmp/huggingface \
    CORS_ORIGINS=http://localhost:5173,http://localhost:8080

# Port 8000 for docker compose; on HF Spaces set `app_port: 8000` in the Space
# README so the platform proxies to it.
EXPOSE 8000

# Model weights are downloaded on first startup and cached in HF_HOME.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
