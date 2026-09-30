"""Runtime intent classifier — pure numpy, no scikit-learn.

Loads the offline-trained TF-IDF + LogisticRegression parameters from
``data/intent_clf.json`` (produced by ``training/train_intent_classifier.py``)
and reproduces scikit-learn's ``TfidfVectorizer.transform`` +
``LogisticRegression.predict_proba`` exactly, using only numpy. This keeps the
deployed backend free of a scikit-learn dependency (same offline-train /
JSON-runtime pattern as the HMM POS tagger).

The classifier maps a Marathi assistant question to one of 9 intents:
PERSON_QUERY, LOCATION_QUERY, ORGANIZATION_QUERY, DATE_QUERY, TIME_QUERY,
MEASURE_QUERY, DESIGNATION_QUERY, ALL_ENTITIES_QUERY, ENTITY_COUNT_QUERY.

Nothing is fabricated: every weight comes from the real fitted model.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

_TABLE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intent_clf.json")


class _Vectorizer:
    """One TF-IDF block (word or char analyzer), reproduced in numpy."""

    def __init__(self, spec: Dict) -> None:
        self.analyzer: str = spec["analyzer"]
        self.lowercase: bool = spec.get("lowercase", True)
        self.ngram_min, self.ngram_max = spec["ngram_range"]
        self.token_re = (
            re.compile(spec["token_pattern"]) if spec.get("token_pattern") else None
        )
        self.vocabulary: Dict[str, int] = spec["vocabulary"]
        self.idf = np.asarray(spec["idf"], dtype=np.float64)
        self.n_features = self.idf.shape[0]

    def _ngrams(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        if self.analyzer == "word":
            units = self.token_re.findall(text)
            joiner = " "
        else:  # 'char': slide over the raw (lowercased) string, spaces included
            units = list(text)
            joiner = ""
        grams: List[str] = []
        n_units = len(units)
        for n in range(self.ngram_min, self.ngram_max + 1):
            if n == 1:
                grams.extend(units)
            else:
                for i in range(n_units - n + 1):
                    grams.append(joiner.join(units[i : i + n]))
        return grams

    def transform(self, text: str) -> np.ndarray:
        """CountVectorizer counts -> * idf -> L2 normalize (sublinear_tf=False)."""
        vec = np.zeros(self.n_features, dtype=np.float64)
        for term, c in Counter(self._ngrams(text)).items():
            col = self.vocabulary.get(term)
            if col is not None:
                vec[col] = c
        vec *= self.idf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class _IntentModel:
    """Holds the exported parameters and the numpy inference path."""

    def __init__(self, table: Dict) -> None:
        self.vectorizers = [_Vectorizer(s) for s in table["vectorizers"]]
        self.classes: List[str] = list(table["classes"])
        self.coef = np.asarray(table["coef"], dtype=np.float64)          # (C, F)
        self.intercept = np.asarray(table["intercept"], dtype=np.float64)  # (C,)

    def _features(self, text: str) -> np.ndarray:
        # FeatureUnion concatenates each block in export order (word, then char).
        return np.concatenate([v.transform(text) for v in self.vectorizers])

    def classify(self, question: str) -> Dict:
        """Return ``{label, confidence, proba}`` for a question.

        ``proba`` is a ``{class: probability}`` dict. Confidence is the max
        softmax probability. Empty/whitespace input yields a neutral result.
        """
        question = question or ""
        if not question.strip():
            return {"label": None, "confidence": 0.0, "proba": {}}
        x = self._features(question)
        logits = self.coef @ x + self.intercept          # (C,)
        logits -= logits.max()                            # softmax stability
        exp = np.exp(logits)
        probs = exp / exp.sum()
        proba = {cls: round(float(p), 6) for cls, p in zip(self.classes, probs)}
        top = int(np.argmax(probs))
        return {
            "label": self.classes[top],
            "confidence": round(float(probs[top]), 4),
            "proba": proba,
        }


_model: Optional[_IntentModel] = None


def _load() -> _IntentModel:
    with open(_TABLE_PATH, "r", encoding="utf-8") as fh:
        return _IntentModel(json.load(fh))


def get_model() -> _IntentModel:
    """Load (once) and return the shared intent model."""
    global _model
    if _model is None:
        _model = _load()
    return _model


def reload() -> _IntentModel:
    """Force a re-read of the JSON table (used after retraining)."""
    global _model
    _model = _load()
    return _model


def classify(question: str) -> Dict:
    """Convenience wrapper over the shared model."""
    return get_model().classify(question)
