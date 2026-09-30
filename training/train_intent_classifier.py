"""Train the intent classifier (TF-IDF + Logistic Regression), offline.

This is a *development-time* script: scikit-learn is used only here to fit the
model. The fitted parameters are exported to a small JSON table
(``backend/data/intent_clf.json``) that the runtime loads and evaluates in pure
numpy — so the deployed backend needs **no scikit-learn dependency**. This is
the same offline-train / JSON-runtime pattern used by the HMM POS tagger.

Dataset: ``training/intent_dataset.csv`` — a **manually curated** set of Marathi
assistant queries labelled with one of 9 intents. It is NOT a public dataset;
no public corpus matches these assistant-specific intents (see docs/datasets.md).

Usage:

    pip install -r training/requirements.txt
    python training/train_intent_classifier.py

Fixed seed (42) so results are reproducible.
"""

from __future__ import annotations

import csv
import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, make_pipeline

SEED = 42

_HERE = os.path.dirname(__file__)
_DATASET = os.path.join(_HERE, "intent_dataset.csv")
_EXPORT = os.path.normpath(
    os.path.join(_HERE, "..", "backend", "data", "intent_clf.json")
)

# Word features capture whole keywords (व्यक्ती, ठिकाणे, किती, सर्व); character
# n-grams capture the shared stem across Marathi inflections (व्यक्ती in
# व्यक्तींचा / व्यक्तींची), which is the main discriminative signal here. Both
# analyzers are exactly reproducible in pure numpy at runtime.
TOKEN_PATTERN = r"(?u)\b\w+\b"
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (2, 5)


def load_dataset(path: str) -> Tuple[List[str], List[str]]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            text = (row["text"] or "").strip()
            label = (row["intent"] or "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)
    return texts, labels


def build_features() -> FeatureUnion:
    """Word (1-2) + char (2-5) TF-IDF, concatenated. Reproducible in numpy."""
    word = TfidfVectorizer(
        analyzer="word",
        lowercase=True,
        token_pattern=TOKEN_PATTERN,
        ngram_range=WORD_NGRAMS,
        sublinear_tf=False,
        norm="l2",
        smooth_idf=True,
        use_idf=True,
    )
    char = TfidfVectorizer(
        analyzer="char",
        lowercase=True,
        ngram_range=CHAR_NGRAMS,
        sublinear_tf=False,
        norm="l2",
        smooth_idf=True,
        use_idf=True,
    )
    # Order matters: numpy runtime concatenates blocks in this same order.
    return FeatureUnion([("word", word), ("char", char)])


def build_clf() -> LogisticRegression:
    return LogisticRegression(max_iter=2000, C=10.0, random_state=SEED)


def build_pipeline():
    return make_pipeline(build_features(), build_clf())


def _export_vectorizer(vec: TfidfVectorizer) -> Dict:
    return {
        "analyzer": vec.analyzer,
        "lowercase": True,
        "token_pattern": TOKEN_PATTERN if vec.analyzer == "word" else None,
        "ngram_range": list(vec.ngram_range),
        "vocabulary": {term: int(idx) for term, idx in vec.vocabulary_.items()},
        "idf": [round(float(v), 8) for v in vec.idf_.tolist()],
    }


def export_model(features: FeatureUnion, clf: LogisticRegression, path: str) -> None:
    """Serialize everything numpy inference needs to reproduce sklearn exactly."""
    vectorizers = [_export_vectorizer(v) for _, v in features.transformer_list]
    payload = {
        "_comment": (
            "Offline-trained TF-IDF (word 1-2 + char 2-5) + LogisticRegression "
            "intent classifier. Loaded and evaluated in pure numpy at runtime (no "
            "sklearn dependency). Trained by training/train_intent_classifier.py on "
            "the manually-curated training/intent_dataset.csv. See docs/datasets.md."
        ),
        "vectorizers": vectorizers,
        "classes": list(clf.classes_),
        "coef": [[round(float(c), 8) for c in row] for row in clf.coef_.tolist()],
        "intercept": [round(float(b), 8) for b in clf.intercept_.tolist()],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def main() -> int:
    texts, labels = load_dataset(_DATASET)
    n_classes = len(set(labels))
    print(f"Loaded {len(texts)} examples across {n_classes} intents from {_DATASET}")

    # --- held-out split (for a single honest test number) ---
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    split_pipe = build_pipeline()
    split_pipe.fit(X_train, y_train)
    test_acc = split_pipe.score(X_test, y_test)
    print(f"Held-out (80/20) accuracy: {test_acc:.4f}")

    # --- stratified k-fold CV (small data -> a more stable estimate) ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(
        build_pipeline(), texts, labels, cv=cv, scoring="f1_macro"
    )
    print(f"5-fold CV macro-F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # --- refit on ALL data, then export the deployable table ---
    features = build_features()
    clf = build_clf()
    Xall = features.fit_transform(texts)
    clf.fit(Xall, labels)
    export_model(features, clf, _EXPORT)
    print(f"Exported classifier to {_EXPORT}")

    # --- sanity check: numpy runtime must match sklearn's predict_proba ---
    _verify_numpy_matches_sklearn(features, clf)
    print("Sanity check passed: numpy runtime matches sklearn predict_proba.")
    return 0


def _verify_numpy_matches_sklearn(features, clf) -> None:
    """Load the exported JSON, run the pure-numpy path, and assert equality."""
    import sys

    sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "backend")))
    from app import intent_clf  # noqa: E402  (uses the freshly-exported JSON)

    intent_clf.reload()  # ensure it reads the file we just wrote
    samples = [
        "या मजकुरात कोणत्या व्यक्ती आहेत?",
        "मजकुरातील सर्व नावे दाखवा",
        "येथे एकूण किती नावे आहेत?",
        "कोणती ठिकाणे आली आहेत?",
    ]
    sk_proba = clf.predict_proba(features.transform(samples))
    classes = list(clf.classes_)
    for i, s in enumerate(samples):
        res = intent_clf.classify(s)
        np_vec = np.array([res["proba"][c] for c in classes])
        if not np.allclose(np_vec, sk_proba[i], atol=1e-6):
            raise AssertionError(
                f"numpy != sklearn for {s!r}: max diff "
                f"{np.abs(np_vec - sk_proba[i]).max():.2e}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
