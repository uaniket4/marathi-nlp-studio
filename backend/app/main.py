"""FastAPI application for Marathi NER.

Loads the model once at startup (lifespan) and exposes:
  * GET  /health      liveness + model status
  * GET  /model-info  model + dataset metadata and real metrics (if evaluated)
  * POST /predict      run NER on submitted text
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import assistant as assistant_mod
from . import search as search_mod
from .config import get_settings
from .corpus import build_corpus, get_corpus
from .inference import entity_statistics, group_by_type, predict
from .model import get_load_error, get_model, load_metrics, load_model
from .schemas import (
    AssistantRequest,
    AssistantResponse,
    DatasetInfoResponse,
    ExtractResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger("marathi_ner")
settings = get_settings()

_DATASET_STATS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_stats.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the transformer exactly once, at startup, then build the search index.
    load_model()
    err = get_load_error()
    if err:
        logger.error("Model failed to load at startup: %s", err)
    else:
        logger.info("Model loaded: %s", settings.MODEL_NAME)
    try:
        build_corpus(get_model())
        corpus = get_corpus()
        logger.info("Corpus indexed: %d documents", len(corpus.documents) if corpus else 0)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to build corpus index")
    yield


app = FastAPI(
    title="Marathi NER API",
    description="Named Entity Recognition for Marathi text (L3Cube-MahaNER).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Catch-all so clients always receive a clean JSON error, never a bare 500
    # with an HTML/plain body or a raw stack trace. The real error is logged.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )


@app.get("/", include_in_schema=False)
def root() -> dict:
    # The UI lives on the frontend dev server / static build, not here. This
    # root gives a friendly pointer instead of a bare 404 when the API origin
    # is opened directly in a browser.
    return {
        "name": "Marathi NLP Studio API",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/predict",
            "/extract",
            "/search",
            "/assistant",
            "/model-info",
            "/dataset-info",
        ],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    model = get_model()
    if model is not None:
        return HealthResponse(status="ok", model_loaded=True)
    return HealthResponse(
        status="degraded",
        model_loaded=False,
        detail=get_load_error() or "Model not loaded.",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = get_model()
    metrics = load_metrics()
    if model is None:
        # Model unavailable: still report configuration honestly.
        return ModelInfoResponse(
            model_name=settings.MODEL_NAME,
            architecture="Unknown (model not loaded)",
            base_model=settings.BASE_MODEL,
            dataset="L3Cube-MahaNER",
            entity_types=[],
            num_entity_classes=0,
            max_length=settings.MAX_LENGTH,
            trained=False,
            metrics=None,
        )
    arch = model.model.config.architectures
    return ModelInfoResponse(
        model_name=model.model_name,
        architecture=arch[0] if arch else model.model.config.model_type,
        base_model=settings.BASE_MODEL,
        dataset="L3Cube-MahaNER",
        entity_types=model.entity_types,
        num_entity_classes=len(model.entity_types),
        max_length=model.max_length,
        trained=bool(metrics and metrics.get("trained")),
        metrics=metrics,
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest) -> PredictResponse:
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs or /health.",
        )
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    try:
        entities = predict(model, text)
        statistics = entity_statistics(entities)
    except Exception:  # noqa: BLE001 - never leak stack traces to clients
        # Log the real traceback server-side; return a clean message to clients.
        logger.exception("Prediction failed for %d-char input", len(text))
        raise HTTPException(status_code=500, detail="Prediction failed. Please retry.")
    return PredictResponse(text=req.text, entities=entities, statistics=statistics)


def _require_model():
    model = get_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs or /health.",
        )
    return model


@app.post("/extract", response_model=ExtractResponse)
def extract_endpoint(req: PredictRequest) -> ExtractResponse:
    """Information extraction: entities grouped into structured, typed lists."""
    model = _require_model()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    try:
        entities = predict(model, text)
        grouped = group_by_type(entities)
        statistics = entity_statistics(entities)
    except Exception:  # noqa: BLE001
        logger.exception("Extraction failed for %d-char input", len(text))
        raise HTTPException(status_code=500, detail="Extraction failed. Please retry.")
    return ExtractResponse(
        text=req.text, entities=entities, grouped=grouped, statistics=statistics
    )


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest) -> SearchResponse:
    """Entity-aware search over the local demo corpus."""
    model = _require_model()
    corpus = get_corpus()
    if corpus is None or not corpus.documents:
        raise HTTPException(status_code=503, detail="Search index is not available.")
    try:
        result = search_mod.search(
            model, corpus, req.query, req.entity_types, req.limit
        )
    except Exception:  # noqa: BLE001
        logger.exception("Search failed for query")
        raise HTTPException(status_code=500, detail="Search failed. Please retry.")
    return SearchResponse(**result)


@app.post("/assistant", response_model=AssistantResponse)
def assistant_endpoint(req: AssistantRequest) -> AssistantResponse:
    """Entity-aware Marathi assistant answering questions about the context."""
    model = _require_model()
    try:
        result = assistant_mod.answer(model, req.context, req.question)
    except Exception:  # noqa: BLE001
        logger.exception("Assistant failed")
        raise HTTPException(status_code=500, detail="Assistant failed. Please retry.")
    return AssistantResponse(**result)


@app.get("/dataset-info", response_model=DatasetInfoResponse)
def dataset_info() -> DatasetInfoResponse:
    """Dataset statistics (computed from the real data) + corpus size."""
    if not os.path.isfile(_DATASET_STATS_PATH):
        raise HTTPException(status_code=404, detail="Dataset stats not available.")
    with open(_DATASET_STATS_PATH, "r", encoding="utf-8") as fh:
        stats = json.load(fh)
    corpus = get_corpus()
    return DatasetInfoResponse(
        **stats,
        corpus_size=len(corpus.documents) if corpus else 0,
        corpus_source=corpus.source if corpus else "",
    )
