"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .config import get_settings

_MAX = get_settings().MAX_TEXT_CHARS


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX,
        description="Marathi text to analyze.",
        examples=["सचिन तेंडुलकर मुंबईमध्ये राहतात."],
    )


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class PredictResponse(BaseModel):
    text: str
    entities: List[Entity]
    statistics: Dict[str, int]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    architecture: str
    base_model: str
    dataset: str
    entity_types: List[str]
    num_entity_classes: int
    max_length: int
    trained: bool
    metrics: Optional[dict] = None


# --- Information extraction ---
class ExtractResponse(BaseModel):
    text: str
    entities: List[Entity]
    grouped: Dict[str, List[str]]
    statistics: Dict[str, int]


# --- Search ---
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX)
    entity_types: List[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    text: str
    entities: List[Entity]
    matched_entities: List[Entity]
    matched_keywords: List[str]
    entity_match_count: int
    keyword_match_count: int
    relevance: float


class SearchResponse(BaseModel):
    query: str
    query_entities: List[Entity]
    query_keywords: List[str]
    total: int
    results: List[SearchResult]


# --- Assistant ---
class AssistantRequest(BaseModel):
    context: str = Field(default="", max_length=_MAX)
    question: str = Field(..., min_length=1, max_length=1000)


class Intent(BaseModel):
    types: List[str]
    is_count: bool
    is_all: bool


class AssistantResponse(BaseModel):
    answer: str
    intent: Intent
    entities: List[Entity]


# --- Dataset ---
class DatasetInfoResponse(BaseModel):
    name: str
    language: str
    splits: Dict[str, int]
    token_counts: Dict[str, int]
    entity_types: List[str]
    entity_distribution_train: Dict[str, int]
    num_entity_classes: int
    source: str
    corpus_size: int
    corpus_source: str
