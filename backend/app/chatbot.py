"""Runtime retrieval chatbot — pure numpy, no scikit-learn.

Loads the offline-trained TF-IDF index (``data/chatbot.json`` +
``data/chatbot_matrix.npz``, produced by ``training/train_chatbot.py``) and
answers a Marathi question by returning the most **cosine-similar sentence(s)**
from a real L3Cube-Pune corpus. There is no LLM and no text generation — every
reply is a verbatim corpus sentence with its similarity score, or an honest
"I don't know" when nothing is similar enough.

Same offline-train / JSON-runtime pattern as the intent classifier and HMM POS
tagger, so the deployed backend needs no scikit-learn dependency.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chatbot.json")
_NPZ_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chatbot_matrix.npz")


class _RetrievalIndex:
    """The corpus TF-IDF matrix + query scoring, all in numpy."""

    def __init__(self, meta: Dict, npz) -> None:
        cfg = meta["config"]
        self.token_pattern = cfg["token_pattern"]
        self.token_re = re.compile(self.token_pattern)
        self.ngram_min, self.ngram_max = cfg["ngram_range"]
        self.sim_floor = float(cfg.get("sim_floor", 0.12))
        self.stopwords = set(meta.get("stopwords", []))
        self.suffixes = list(meta.get("query_suffixes", []))
        self.sentences: List[str] = meta["sentences"]
        self.source = meta.get("source", "")
        self.url = meta.get("url", "")

        self.vocab: Dict[str, int] = {t: i for i, t in enumerate(meta["vocab"])}
        self.idf = np.asarray(meta["idf"], dtype=np.float64)

        # CSR matrix (rows already L2-normalized at train time).
        self._data = npz["data"].astype(np.float64)
        self._indices = npz["indices"].astype(np.int64)
        indptr = npz["indptr"].astype(np.int64)
        self.n_docs = int(npz["shape"][0])
        # Row id of every stored value, so scores sum per document via add.at.
        self._row_ids = np.repeat(np.arange(self.n_docs), np.diff(indptr))

    # --- query preprocessing (query side only; corpus is untouched) ---------
    def clean_query(self, question: str) -> List[str]:
        """Tokenize, strip agglutinated postpositions, drop stop/instruction words."""
        out: List[str] = []
        for tok in self.token_re.findall(question.lower()):
            for suf in self.suffixes:
                if tok.endswith(suf) and len(tok) > len(suf) + 1:
                    tok = tok[: -len(suf)]
                    break
            if tok.endswith("ां") and len(tok) > 3:  # oblique/honorific ending
                tok = tok[:-2]
            if tok in self.stopwords or len(tok) < 2:
                continue
            out.append(tok)
        return out

    def _query_vector(self, tokens: List[str]) -> np.ndarray:
        grams: List[str] = list(tokens)
        for i in range(len(tokens) - 1):
            grams.append(tokens[i] + " " + tokens[i + 1])
        vec = np.zeros(self.idf.shape[0], dtype=np.float64)
        for term, c in Counter(grams).items():
            col = self.vocab.get(term)
            if col is not None:
                vec[col] = (1.0 + math.log(c)) * self.idf[col]  # sublinear tf
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(self, question: str, top: int = 3) -> Dict:
        """Return the most similar corpus sentences to `question`.

        ``{sentences, scores, best, cleaned}`` — cosine scores in [0, 1]. When
        the best score is below the similarity floor the caller should treat it
        as "no confident match".
        """
        tokens = self.clean_query(question)
        if not tokens:
            return {"sentences": [], "scores": [], "best": 0.0, "cleaned": ""}
        qvec = self._query_vector(tokens)
        if not qvec.any():
            return {"sentences": [], "scores": [], "best": 0.0, "cleaned": " ".join(tokens)}
        # cosine = dot(row, qvec) since corpus rows are already L2-normalized.
        scores = np.zeros(self.n_docs, dtype=np.float64)
        np.add.at(scores, self._row_ids, self._data * qvec[self._indices])
        order = scores.argsort()[::-1][:top]
        return {
            "sentences": [self.sentences[i] for i in order],
            "scores": [round(float(scores[i]), 4) for i in order],
            "best": round(float(scores[order[0]]), 4),
            "cleaned": " ".join(tokens),
            "n_docs": self.n_docs,
        }


_index: Optional[_RetrievalIndex] = None


def _load() -> _RetrievalIndex:
    with open(_JSON_PATH, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    npz = np.load(_NPZ_PATH)
    return _RetrievalIndex(meta, npz)


def get_index() -> _RetrievalIndex:
    """Load (once) and return the shared retrieval index."""
    global _index
    if _index is None:
        _index = _load()
    return _index


def reload() -> _RetrievalIndex:
    """Force a re-read of the index files (used after retraining)."""
    global _index
    _index = _load()
    return _index


def search(question: str, top: int = 3) -> Dict:
    """Convenience wrapper over the shared index."""
    return get_index().search(question, top)
