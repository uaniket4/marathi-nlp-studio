"""API tests using FastAPI's TestClient.

The model is loaded via the app lifespan. Endpoints that need the model are
skipped if it could not be loaded (e.g. offline CI without the weights cached),
so the suite still validates request/response contracts everywhere possible.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ""))

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan -> load_model()
        yield c


def _model_loaded(client) -> bool:
    return client.get("/health").json().get("model_loaded", False)


def test_health_ok_shape(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("ok", "degraded")
    assert "model_loaded" in body


def test_model_info_shape(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["dataset"] == "L3Cube-MahaNER"
    assert "entity_types" in body
    assert "trained" in body


def test_predict_empty_text_rejected(client):
    # min_length validation -> 422 before touching the model.
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_missing_field_rejected(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_predict_too_long_rejected(client):
    resp = client.post("/predict", json={"text": "अ" * 100000})
    assert resp.status_code == 422


def test_predict_marathi_text(client):
    if not _model_loaded(client):
        pytest.skip("Model not loaded; skipping live inference test.")
    resp = client.post(
        "/predict",
        json={"text": "सचिन तेंडुलकर मुंबईमध्ये राहतात."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"].startswith("सचिन")
    assert isinstance(body["entities"], list)
    assert len(body["entities"]) >= 1
    for ent in body["entities"]:
        assert set(ent) == {"text", "label", "start", "end", "confidence"}
        assert 0 <= ent["confidence"] <= 1
        assert ent["start"] < ent["end"]
        assert ent["text"]
    assert isinstance(body["statistics"], dict)


def test_predict_whitespace_only(client):
    # Passes min_length but is empty after strip -> 422 from handler.
    if not _model_loaded(client):
        pytest.skip("Model not loaded.")
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422


# --- /extract ---
def test_extract_empty_rejected(client):
    resp = client.post("/extract", json={"text": ""})
    assert resp.status_code == 422


def test_extract_marathi_text(client):
    if not _model_loaded(client):
        pytest.skip("Model not loaded.")
    resp = client.post(
        "/extract",
        json={"text": "रतन टाटा यांनी टाटा मोटर्स ही कंपनी मुंबई येथे सुरू केली."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["entities"], list)
    assert isinstance(body["grouped"], dict)
    # grouped values are lists of de-duplicated surface strings.
    for label, values in body["grouped"].items():
        assert label.isupper()
        assert isinstance(values, list)
        assert len(values) == len(set(values))
    assert isinstance(body["statistics"], dict)


# --- /search ---
def test_search_empty_query_rejected(client):
    resp = client.post("/search", json={"query": "", "entity_types": [], "limit": 5})
    assert resp.status_code == 422


def test_search_limit_out_of_range_rejected(client):
    resp = client.post("/search", json={"query": "मुंबई", "limit": 999})
    assert resp.status_code == 422


def test_search_returns_ranked_results(client):
    if not _model_loaded(client):
        pytest.skip("Model not loaded.")
    resp = client.post("/search", json={"query": "मुंबई", "entity_types": [], "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "मुंबई"
    assert set(body) >= {"query", "query_entities", "query_keywords", "total", "results"}
    assert len(body["results"]) <= 5
    # results are sorted by descending relevance.
    rels = [r["relevance"] for r in body["results"]]
    assert rels == sorted(rels, reverse=True)
    for r in body["results"]:
        assert 0.0 <= r["relevance"] <= 1.0


# --- /assistant ---
def test_assistant_missing_question_rejected(client):
    resp = client.post("/assistant", json={"context": "काहीतरी"})
    assert resp.status_code == 422


def test_assistant_empty_context_prompts_for_text(client):
    if not _model_loaded(client):
        pytest.skip("Model not loaded.")
    resp = client.post(
        "/assistant", json={"context": "", "question": "कोणत्या व्यक्ती आहेत?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"]  # non-empty prompt to add context
    assert body["entities"] == []


def test_assistant_answers_from_entities(client):
    if not _model_loaded(client):
        pytest.skip("Model not loaded.")
    resp = client.post(
        "/assistant",
        json={
            "context": "पंतप्रधान नरेंद्र मोदी यांनी दिल्लीत भाषण केले.",
            "question": "कोणत्या व्यक्ती आहेत?",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "PERSON" in body["intent"]["types"]
    assert any(e["label"] == "PERSON" for e in body["entities"])


# --- /dataset-info ---
def test_dataset_info_shape(client):
    resp = client.get("/dataset-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["name"] == "L3Cube-MahaNER"
    assert body["num_entity_classes"] == 7
    assert set(body["splits"]) == {"train", "validation", "test"}
    assert "corpus_size" in body
