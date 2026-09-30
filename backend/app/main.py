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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import assistant as assistant_mod
from . import pipeline as pipeline_mod
from . import search as search_mod
from .config import get_settings
from .corpus import build_corpus, get_corpus
from .documents import UnsupportedDocument, extract_text
from .inference import (
    build_steps,
    entity_statistics,
    group_by_type,
    predict_with_trace,
)
from .model import get_load_error, get_model, load_metrics, load_model
from .schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AssistantRequest,
    AssistantResponse,
    DatasetInfoResponse,
    DocumentSearchRequest,
    ExtractResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
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
            "/analyze",
            "/search",
            "/upload-document",
            "/search-document",
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
        entities, trace = predict_with_trace(model, text)
        statistics = entity_statistics(entities)
        tokens = trace if req.explain else None
        steps = build_steps(trace, entities) if req.explain else None
    except Exception:  # noqa: BLE001 - never leak stack traces to clients
        # Log the real traceback server-side; return a clean message to clients.
        logger.exception("Prediction failed for %d-char input", len(text))
        raise HTTPException(status_code=500, detail="Prediction failed. Please retry.")
    return PredictResponse(
        text=req.text, entities=entities, statistics=statistics, tokens=tokens, steps=steps
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_endpoint(req: AnalyzeRequest) -> AnalyzeResponse:
    """Run the syllabus-aligned classical NLP pipeline over the text.

    Returns every stage (tokenization, morphology, stemming, n-gram LM, HMM POS,
    NP chunking, NER, sentiment) with its real computed output. NER runs
    internally, so the NER Playground gets entities + full pipeline in one call.
    """
    model = _require_model()
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    try:
        result = pipeline_mod.analyze(model, text)
    except Exception:  # noqa: BLE001 - never leak stack traces to clients
        logger.exception("Pipeline analysis failed for %d-char input", len(text))
        raise HTTPException(status_code=500, detail="Analysis failed. Please retry.")
    return AnalyzeResponse(**result)


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
        entities, trace = predict_with_trace(model, text)
        grouped = group_by_type(entities)
        statistics = entity_statistics(entities)
        tokens = trace if req.explain else None
        steps = build_steps(trace, entities, grouped) if req.explain else None
    except Exception:  # noqa: BLE001
        logger.exception("Extraction failed for %d-char input", len(text))
        raise HTTPException(status_code=500, detail="Extraction failed. Please retry.")
    return ExtractResponse(
        text=req.text,
        entities=entities,
        grouped=grouped,
        statistics=statistics,
        tokens=tokens,
        steps=steps,
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


@app.post("/upload-document", response_model=UploadResponse)
async def upload_document_endpoint(file: UploadFile = File(...)) -> UploadResponse:
    """Extract plain text from an uploaded .txt / .pdf / .docx file."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    try:
        text, truncated = extract_text(file.filename or "", raw, settings.MAX_TEXT_CHARS)
    except UnsupportedDocument as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:  # noqa: BLE001 - never leak stack traces to clients
        logger.exception("Text extraction failed for %s", file.filename)
        raise HTTPException(
            status_code=422,
            detail="Could not read that file. It may be corrupted or password-protected.",
        )
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No readable text found in that file (it may be scanned images).",
        )
    return UploadResponse(
        filename=file.filename or "document",
        text=text,
        char_count=len(text),
        truncated=truncated,
    )


@app.post("/search-document", response_model=SearchResponse)
def search_document_endpoint(req: DocumentSearchRequest) -> SearchResponse:
    """Entity-aware search within a user-supplied document, with pipeline steps."""
    model = _require_model()
    try:
        result = search_mod.search_in_text(
            model, req.document, req.query, req.entity_types, req.limit
        )
    except Exception:  # noqa: BLE001
        logger.exception("Document search failed")
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
