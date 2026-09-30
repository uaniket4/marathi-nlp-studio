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
    explain: bool = Field(
        default=False,
        description="Include the per-token trace and pipeline steps in the response.",
    )


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float


class TopK(BaseModel):
    label: str
    prob: float


class TokenInfo(BaseModel):
    index: int
    token: str
    start: int
    end: int
    is_special: bool
    predicted_label: str
    entity_type: Optional[str] = None
    boundary: Optional[str] = None
    confidence: float
    top_k: List[TopK]


class PipelineStep(BaseModel):
    id: str
    title: str
    description: str
    count: Optional[int] = None
    module: Optional[str] = None
    experiment: Optional[str] = None


class PredictResponse(BaseModel):
    text: str
    entities: List[Entity]
    statistics: Dict[str, int]
    tokens: Optional[List[TokenInfo]] = None
    steps: Optional[List[PipelineStep]] = None


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
    tokens: Optional[List[TokenInfo]] = None
    steps: Optional[List[PipelineStep]] = None


# --- Search ---
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=_MAX)
    entity_types: List[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=50)


class DocumentSearchRequest(BaseModel):
    document: str = Field(..., min_length=1, max_length=_MAX)
    query: str = Field(..., min_length=1, max_length=_MAX)
    entity_types: List[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=50)


class UploadResponse(BaseModel):
    filename: str
    text: str
    char_count: int
    truncated: bool


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
    steps: Optional[List[PipelineStep]] = None
    passage_count: Optional[int] = None


# --- Assistant ---
class AssistantRequest(BaseModel):
    context: str = Field(default="", max_length=_MAX)
    question: str = Field(..., min_length=1, max_length=1000)


class Intent(BaseModel):
    types: List[str]
    is_count: bool
    is_all: bool


class Source(BaseModel):
    title: str
    url: str


class CorefLink(BaseModel):
    mention: str
    antecedent: str
    type: str


class AssistantResponse(BaseModel):
    answer: str
    intent: Intent
    entities: List[Entity]
    steps: Optional[List[PipelineStep]] = None
    sources: List[Source] = Field(default_factory=list)
    # ML intent classifier + rule-based coreference (present on context turns).
    intent_label: Optional[str] = None
    intent_confidence: Optional[float] = None
    context_used: Optional[str] = None
    coref_links: Optional[List[CorefLink]] = None


# --- Syllabus pipeline (classical NLP stages) ---
class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=_MAX,
        description="Marathi text to run through the classical NLP pipeline.",
        examples=["सचिन तेंडुलकर मुंबईत राहतो."],
    )


class PipelineStage(BaseModel):
    id: str
    module: str
    experiment: str
    title: str
    description: str
    # Each stage carries its own shape of computed output.
    data: Dict = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    text: str
    tokens: List[str]
    stages: List[PipelineStage]
    entities: List[Entity]
    statistics: Dict[str, int]


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
