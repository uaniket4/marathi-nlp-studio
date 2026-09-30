"""Evaluate the intent classifier and compare it to the keyword baseline.

Two classifiers on the SAME manually-curated dataset, scored fairly:

  * ML  — TF-IDF (word+char) + LogisticRegression. Out-of-fold predictions from
          5-fold stratified CV, so no sample is predicted by a model that saw it.
  * Baseline — the existing keyword matcher (``app.assistant.detect_intent``)
          mapped into the same 9-intent label space. Deterministic, no training,
          so it predicts every sample directly.

Reports accuracy, per-class precision/recall/F1 and macro-F1 for both, prints a
comparison table and writes ``training/intent_metrics.json``. Fixed seed (42).

    python training/evaluate_intent.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "backend")))

from app import keyword_intent  # noqa: E402  (torch-free keyword baseline)
from train_intent_classifier import SEED, build_pipeline, load_dataset  # noqa: E402

_DATASET = os.path.join(_HERE, "intent_dataset.csv")
_OUT = os.path.join(_HERE, "intent_metrics.json")

# First entity type -> intent label (baseline mapping into the 9-intent space).
_TYPE_TO_INTENT = {
    "PERSON": "PERSON_QUERY",
    "LOCATION": "LOCATION_QUERY",
    "ORGANIZATION": "ORGANIZATION_QUERY",
    "DATE": "DATE_QUERY",
    "TIME": "TIME_QUERY",
    "MEASURE": "MEASURE_QUERY",
    "DESIGNATION": "DESIGNATION_QUERY",
}


def baseline_predict(question: str) -> str:
    """Map the keyword detector's output into the 9-intent label space."""
    intent = keyword_intent.detect_intent(question)
    if intent["is_all"]:
        return "ALL_ENTITIES_QUERY"
    if intent["types"]:
        # A specific entity type dominates (even if 'किती' is also present).
        return _TYPE_TO_INTENT.get(intent["types"][0], "ALL_ENTITIES_QUERY")
    if intent["is_count"]:
        return "ENTITY_COUNT_QUERY"
    # No keyword matched -> the assistant's default is "summarise everything".
    return "ALL_ENTITIES_QUERY"


def _metrics(y_true, y_pred, labels) -> dict:
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    per_class = {
        lab: {
            "precision": round(report[lab]["precision"], 4),
            "recall": round(report[lab]["recall"], 4),
            "f1": round(report[lab]["f1-score"], 4),
            "support": int(report[lab]["support"]),
        }
        for lab in labels
    }
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro",
                                   zero_division=0), 4),
        "per_class": per_class,
    }


def main() -> int:
    texts, labels = load_dataset(_DATASET)
    classes = sorted(set(labels))
    y = np.array(labels)

    # ML: out-of-fold predictions (fair — a sample is never in its train fold).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    ml_pred = cross_val_predict(build_pipeline(), texts, labels, cv=cv)
    ml = _metrics(y, ml_pred, classes)

    # Baseline: deterministic keyword matcher over every sample.
    base_pred = np.array([baseline_predict(t) for t in texts])
    base = _metrics(y, base_pred, classes)

    out = {
        "dataset": "training/intent_dataset.csv (manually curated, not public)",
        "num_examples": len(texts),
        "num_intents": len(classes),
        "intents": classes,
        "protocol": "5-fold stratified CV out-of-fold predictions (ML); "
                    "deterministic keyword mapping (baseline). seed=42",
        "ml_tfidf_logreg": ml,
        "keyword_baseline": base,
    }
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"Intent classification — {len(texts)} examples, {len(classes)} intents\n")
    print(f"{'Model':<26}{'Accuracy':>10}{'Macro-F1':>10}")
    print("-" * 46)
    print(f"{'Keyword baseline':<26}{base['accuracy']:>10.4f}{base['macro_f1']:>10.4f}")
    print(f"{'TF-IDF + LogReg (ML)':<26}{ml['accuracy']:>10.4f}{ml['macro_f1']:>10.4f}")
    print(f"\nPer-class F1 (ML):")
    for lab in classes:
        print(f"  {lab:<22}{ml['per_class'][lab]['f1']:>8.4f}"
              f"   (support {ml['per_class'][lab]['support']})")
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
