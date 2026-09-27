"""Evaluation utilities and a standalone evaluator.

Provides seqeval-based, entity-level metrics. All tag sequences are normalized
to IOB2 with full entity names (e.g. ``B-Person``/``I-Person``) so that models
using different internal label schemes (the IOB codes of a locally trained
model, or the flat labels of the official ``l3cube-pune/marathi-ner`` model)
are scored on a common footing.

Standalone usage:

    python evaluate.py                         # evaluate MODEL_NAME on test
    MODEL_NAME=./backend/models/marathi-ner python evaluate.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from seqeval.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import ENTITY_FULL_NAMES  # noqa: E402

# Reverse map: full name -> IOB code (for models that emit full names).
_FULL_TO_CODE = {v: k for k, v in ENTITY_FULL_NAMES.items()}


def iob_code_to_iob2(tag: str) -> str:
    """Convert a MahaNER IOB code (``BNEP``/``INEP``/``BED``/``O``) to IOB2.

    ``BNEP`` -> ``B-Person``, ``INETI`` -> ``I-Time``, ``O`` -> ``O``.
    """
    if tag == "O" or not tag:
        return "O"
    prefix, code = tag[0], tag[1:]
    full = ENTITY_FULL_NAMES.get(code, code)
    return f"{prefix}-{full}"


def flat_to_iob2(labels: List[str]) -> List[str]:
    """Convert a flat per-token label sequence (``Person``/``Other``) to IOB2.

    Consecutive tokens sharing a non-``Other`` type become ``B-`` then ``I-``.
    """
    out: List[str] = []
    prev = None
    for lab in labels:
        if lab in ("Other", "O", ""):
            out.append("O")
            prev = None
            continue
        prefix = "B" if lab != prev else "I"
        out.append(f"{prefix}-{lab}")
        prev = lab
    return out


def seqeval_report(pred_tags: List[List[str]], gold_tags: List[List[str]]) -> Dict:
    """Compute entity-level metrics from IOB2 tag sequences."""
    report = classification_report(
        gold_tags, pred_tags, output_dict=True, zero_division=0
    )
    per_entity = {
        k: {
            "precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1": round(v["f1-score"], 4),
            "support": int(v["support"]),
        }
        for k, v in report.items()
        if k not in ("micro avg", "macro avg", "weighted avg")
    }
    return {
        "summary": {
            "precision": round(precision_score(gold_tags, pred_tags, zero_division=0), 4),
            "recall": round(recall_score(gold_tags, pred_tags, zero_division=0), 4),
            "f1": round(f1_score(gold_tags, pred_tags, zero_division=0), 4),
            "accuracy": round(accuracy_score(gold_tags, pred_tags), 4),
        },
        "per_entity": per_entity,
    }


def compute_seqeval_metrics(preds, labels, id2label: Dict[int, str]) -> Dict:
    """For HF Trainer: convert id arrays (ignoring -100) to IOB2 and score."""
    pred_tags: List[List[str]] = []
    gold_tags: List[List[str]] = []
    for pred_row, label_row in zip(preds, labels):
        p_seq, g_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            p_seq.append(iob_code_to_iob2(id2label[int(p)]))
            g_seq.append(iob_code_to_iob2(id2label[int(l)]))
        pred_tags.append(p_seq)
        gold_tags.append(g_seq)
    return seqeval_report(pred_tags, gold_tags)


def _standalone() -> int:
    """Evaluate the configured MODEL_NAME on the test split via live inference."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from app.config import get_settings
    from dataset import load_dataset

    s = get_settings()
    print(f"Loading model: {s.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(s.MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(s.MODEL_NAME)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    data = load_dataset(s.DATA_DIR)
    test = data["test"]
    print(f"Test sentences: {len(test)}")

    pred_tags: List[List[str]] = []
    gold_tags: List[List[str]] = []

    with torch.no_grad():
        for tokens, tags in zip(test.tokens, test.tags):
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=s.MAX_LENGTH,
                return_tensors="pt",
            )
            logits = model(**enc).logits[0]
            pred_ids = logits.argmax(-1).tolist()
            word_ids = enc.word_ids(batch_index=0)

            word_pred: List[str] = []
            seen = set()
            for wid, pid in zip(word_ids, pred_ids):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                word_pred.append(id2label[pid])
            # Truncation guard: pad any missing words as O.
            while len(word_pred) < len(tokens):
                word_pred.append("O")

            pred_tags.append(flat_to_iob2(word_pred))
            gold_tags.append([iob_code_to_iob2(t) for t in tags])

    metrics = seqeval_report(pred_tags, gold_tags)
    metrics.update(
        {
            "model": s.MODEL_NAME,
            "trained": True,
            "dataset": "L3Cube-MahaNER",
            "counts": {
                "train": len(data["train"]),
                "validation": len(data["validation"]),
                "test": len(data["test"]),
                "num_labels": len(id2label),
            },
        }
    )

    os.makedirs(os.path.dirname(s.METRICS_PATH), exist_ok=True)
    with open(s.METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, ensure_ascii=False, indent=2)
    print(f"Wrote metrics to {s.METRICS_PATH}")
    print(json.dumps(metrics["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_standalone())
