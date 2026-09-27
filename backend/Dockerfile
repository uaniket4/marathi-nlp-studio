# Backend image: FastAPI + transformers (CPU inference).
# Runs locally (docker compose) and on Cloud Run / any Docker host.
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

ARG MODEL_NAME=l3cube-pune/marathi-ner
ENV MODEL_NAME=${MODEL_NAME} \
    MAX_LENGTH=256 \
    HF_HOME=/app/hf_cache \
    CORS_ORIGINS=http://localhost:5173,http://localhost:8080

# Bake the model into the image at build time. Cloud Run scales to zero, so this
# avoids re-downloading ~500 MB on every cold start and keeps startup fast and
# network-independent (the RUN below is the only time weights are fetched).
RUN python -c "from transformers import AutoTokenizer, AutoModelForTokenClassification; \
    AutoTokenizer.from_pretrained('${MODEL_NAME}'); \
    AutoModelForTokenClassification.from_pretrained('${MODEL_NAME}')"

# From here on, never touch the network for model files — use the baked cache.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Port 8000 for docker compose; hosts that inject $PORT (Cloud Run sets 8080)
# are honored automatically.
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
