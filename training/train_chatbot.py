"""Train the retrieval chatbot (TF-IDF nearest-neighbour), offline.

The chatbot answers a Marathi question by returning the most similar sentence(s)
from a **real L3Cube-Pune corpus** — there is no LLM and no text generation. This
is a classic information-retrieval / vector-space model: fit a TF-IDF vectorizer
on the corpus, then at query time rank corpus sentences by cosine similarity.

Like the intent classifier and the HMM POS tagger, this is a *development-time*
script: scikit-learn fits the model here, and the fitted parameters are exported
to a small table (JSON + a compressed numpy matrix) that the runtime loads and
evaluates in **pure numpy** — so the deployed backend needs no scikit-learn.

Corpus: the curated Marathi health sentences plus the Apache-2.0 Kaggle dataset
``training/marathi_rural_health_crisis_reasoning_v1.json``. The assistant turns
from that dataset are added as retrieval documents. To train on another domain,
pass a plain-text file with one sentence per line using --corpus <path>.

Usage:
    pip install -r training/requirements.txt
    python training/train_chatbot.py

Fixed, deterministic — no randomness, so the index is reproducible byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from collections import Counter, OrderedDict
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_HERE = os.path.dirname(__file__)
_DEFAULT_CORPUS = os.path.join(_HERE, "health_corpus.txt")
_KAGGLE_CORPUS = os.path.join(_HERE, "marathi_rural_health_crisis_reasoning_v1.json")
_JSON = os.path.normpath(os.path.join(_HERE, "..", "backend", "data", "chatbot.json"))
_NPZ = os.path.normpath(os.path.join(_HERE, "..", "backend", "data", "chatbot_matrix.npz"))

# Devanagari-block (keeps whole words incl. vowel-sign matras) OR latin runs.
# `\b\w+\b` splits Marathi words at combining matras, so we use an explicit
# script-range pattern that the numpy runtime mirrors exactly.
TOKEN_PATTERN = r"[ऀ-ॿ]+|[A-Za-z]+"
WORD_NGRAMS = (1, 2)
MIN_TOKENS, MAX_TOKENS = 4, 40  # keep natural sentences, drop fragments/outliers
MIN_DF = 1                      # keep rare medical terms from the small QA set
SIM_FLOOR = 0.12                # below this cosine → honest "don't know"

_SOURCE = (
    "Curated Marathi health corpus (training/health_corpus.txt) + "
    "Kaggle marathi-rural-health-crisis-reasoning-v1"
)
_URL = "https://www.kaggle.com/datasets/jayarajmaitreyaa/marathi-rural-health-crisis-reasoning-v1"
_LICENSE = "Project dataset + Apache 2.0 (Kaggle source)"

# Question/instruction words carrying no topical signal. Stripped from the *query*
# only (never from the corpus) so "मुंबईबद्दल सांगा" ranks by "मुंबई", not "सांगा".
STOPWORDS = [
    "आणि", "व", "आहे", "आहेत", "होते", "होता", "होती", "मध्ये", "या", "हे", "ही",
    "तो", "ती", "ते", "का", "काय", "कोण", "कोणता", "कोणते", "कोणती", "कोणत्या",
    "किती", "कुठे", "कुठली", "कधी", "एक", "साठी", "ला", "चा", "ची", "चे", "बद्दल",
    "सांगा", "सांग", "विषयी", "आपण", "मला", "म्हणजे", "माहिती", "द्या", "नाव",
    "बाबत", "मी", "तू", "आम्ही", "असा", "असे", "कसा", "कसे", "कशी", "कोणी",
]
# Postpositions that agglutinate onto a noun ("महाराजांबद्दल" = महाराज + बद्दल).
QUERY_SUFFIXES = ["बद्दलची", "बद्दलचे", "बद्दलचा", "बद्दल", "विषयी", "बाबत",
                  "संबंधी", "संबंधीची", "बाबतची"]


def load_corpus(corpus_path: str = "") -> List[str]:
    """Load, clean and de-duplicate the training corpus sentences."""
    path = corpus_path or _DEFAULT_CORPUS
    with io.open(path, encoding="utf-8") as fh:
        raw = [ln.strip() for ln in fh]
    if not corpus_path and os.path.exists(_KAGGLE_CORPUS):
        with io.open(_KAGGLE_CORPUS, encoding="utf-8") as fh:
            qa_records = json.load(fh)
        raw.extend(
            message["content"].strip()
            for record in qa_records
            for message in record.get("messages", [])
            if message.get("role") == "assistant" and message.get("content")
        )
    seen, corpus = set(), []
    for s in raw:
        s = s.strip()
        if not (MIN_TOKENS <= len(s.split()) <= MAX_TOKENS):
            continue
        if s in seen:
            continue
        seen.add(s)
        corpus.append(s)
    return corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="", help="Optional plain-text corpus (one sentence per line).")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} unique sentences for the retrieval index.")

    vec = TfidfVectorizer(
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        ngram_range=WORD_NGRAMS,
        sublinear_tf=True,
        min_df=MIN_DF,
        norm="l2",
        smooth_idf=True,
        use_idf=True,
    )
    matrix = vec.fit_transform(corpus).astype(np.float32).tocsr()
    print(f"Fitted TF-IDF: {matrix.shape[0]} docs x {matrix.shape[1]} terms, nnz={matrix.nnz}")

    _export(vec, matrix, corpus)
    _verify_numpy_matches_sklearn(vec, corpus)
    print("Sanity check passed: numpy transform matches scikit-learn.")
    return 0


# --- export (JSON metadata + compressed numpy CSR matrix) -------------------
def _export(vec: TfidfVectorizer, matrix, corpus: List[str]) -> None:
    # vocabulary_ maps term -> column; store terms ordered by column index.
    vocab_terms = [None] * len(vec.vocabulary_)
    for term, idx in vec.vocabulary_.items():
        vocab_terms[idx] = term
    payload = {
        "_comment": (
            "Offline-trained TF-IDF retrieval index (word 1-2, sublinear tf, L2). "
            "The chatbot returns the most cosine-similar corpus sentence(s) — no "
            "LLM, no generation. Loaded and scored in pure numpy at runtime. Built "
            "by training/train_chatbot.py. See docs/datasets.md."
        ),
        "source": _SOURCE,
        "url": _URL,
        "license": _LICENSE,
        "config": {
            "token_pattern": TOKEN_PATTERN,
            "ngram_range": list(WORD_NGRAMS),
            "sublinear_tf": True,
            "sim_floor": SIM_FLOOR,
        },
        "stopwords": STOPWORDS,
        "query_suffixes": QUERY_SUFFIXES,
        "vocab": vocab_terms,
        "idf": [round(float(v), 7) for v in vec.idf_.tolist()],
        "sentences": corpus,
    }
    os.makedirs(os.path.dirname(_JSON), exist_ok=True)
    with io.open(_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    np.savez_compressed(
        _NPZ,
        data=matrix.data.astype(np.float32),
        indices=matrix.indices.astype(np.int32),
        indptr=matrix.indptr.astype(np.int32),
        shape=np.asarray(matrix.shape, dtype=np.int64),
    )
    print(f"Exported index → {_JSON} + {_NPZ}")


def _numpy_transform(query: str, vocab: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    """Reproduce sklearn TfidfVectorizer.transform (sublinear tf, l2) in numpy."""
    import re
    tokens = re.findall(TOKEN_PATTERN, query.lower())
    grams: List[str] = list(tokens)
    for i in range(len(tokens) - 1):
        grams.append(tokens[i] + " " + tokens[i + 1])
    vec = np.zeros(idf.shape[0], dtype=np.float64)
    for term, c in Counter(grams).items():
        col = vocab.get(term)
        if col is not None:
            vec[col] = (1.0 + math.log(c)) * idf[col]  # sublinear_tf
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _verify_numpy_matches_sklearn(vec: TfidfVectorizer, corpus: List[str]) -> None:
    vocab = vec.vocabulary_
    idf = vec.idf_
    samples = corpus[:3] + ["सचिन तेंडुलकर कोण आहेत", "करोना विषाणू"]
    for s in samples:
        mine = _numpy_transform(s, vocab, idf)
        sk = np.asarray(vec.transform([s]).todense()).ravel()
        if not np.allclose(mine, sk, atol=1e-6):
            raise AssertionError(
                f"numpy != sklearn for {s!r}: max diff {np.abs(mine - sk).max():.2e}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
