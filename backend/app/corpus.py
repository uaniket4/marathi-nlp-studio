"""Local document corpus and search index.

Loads a small demo corpus of real Marathi sentences (sampled from the
L3Cube-MahaNER training split — see ``data/documents.json``) and runs the NER
model over each document exactly once at startup to build an in-memory index of
entities per document. This index powers the Search workspace.

This is a clearly-labelled demonstration corpus, not a production search index.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from .config import get_settings
from .inference import predict

_DOCS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "documents.json")


class IndexedDocument:
    """A corpus document plus its extracted entities and lookup sets."""

    def __init__(self, doc: dict, entities: List[Dict]) -> None:
        self.id: str = doc["id"]
        self.title: str = doc["title"]
        self.text: str = doc["text"]
        self.entities: List[Dict] = entities
        # Lowercased entity surface texts by type, and a token set for keywords.
        self.entity_texts = {e["text"] for e in entities}
        self.entity_types = {e["label"] for e in entities}
        self.tokens = set(self.text.split())

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "entities": self.entities,
        }


class Corpus:
    """In-memory corpus with a startup-built entity index."""

    def __init__(self) -> None:
        self.documents: List[IndexedDocument] = []
        self.source: str = ""
        self.built: bool = False

    def build(self, model) -> None:
        """Load documents and index their entities using the shared model."""
        if not os.path.isfile(_DOCS_PATH):
            self.built = False
            return
        with open(_DOCS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.source = data.get("source", "")
        self.documents = []
        for doc in data.get("documents", []):
            entities = predict(model, doc["text"]) if model is not None else []
            self.documents.append(IndexedDocument(doc, entities))
        self.built = True

    def all_entity_types(self) -> List[str]:
        types = set()
        for d in self.documents:
            types.update(d.entity_types)
        return sorted(types)


_corpus: Optional[Corpus] = None


def build_corpus(model) -> None:
    global _corpus
    _corpus = Corpus()
    _corpus.build(model)


def get_corpus() -> Optional[Corpus]:
    return _corpus
