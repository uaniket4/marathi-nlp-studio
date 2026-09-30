"""Evaluate the rule-based coreference linker on a small annotated set.

This is a **rule-based demonstration**, not a learned model, evaluated on a
small **manually-annotated** set of Marathi follow-up questions
(``training/coref_test.csv``). Each row gives a context's gold entities (so the
coref rules are tested independently of NER quality), a follow-up question with
a pronoun, and the expected antecedent. We report resolution accuracy.

    python training/evaluate_coref.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "backend")))

from app import coref  # noqa: E402  (torch-free)

_DATASET = os.path.join(_HERE, "coref_test.csv")
_OUT = os.path.join(_HERE, "coref_metrics.json")


def _parse_entities(field: str):
    """Parse ``text:label:start;text:label:start`` into entity dicts."""
    ents = []
    for part in (field or "").split(";"):
        part = part.strip()
        if not part:
            continue
        text, label, start = part.rsplit(":", 2)
        ents.append({"text": text, "label": label, "start": int(start)})
    return ents


def main() -> int:
    rows = []
    with open(_DATASET, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    correct = 0
    details = []
    for row in rows:
        entities = _parse_entities(row["entities"])
        expected = row["expected"].strip()
        res = coref.resolve(row["question"], entities)
        antecedents = [l["antecedent"] for l in res["links"]]
        ok = expected in antecedents
        correct += int(ok)
        details.append({
            "question": row["question"],
            "expected": expected,
            "resolved": antecedents,
            "correct": ok,
        })

    n = len(rows)
    acc = round(correct / n, 4) if n else 0.0
    out = {
        "dataset": "training/coref_test.csv (manually annotated, rule-based demo)",
        "method": "rule-based (type + recency + light number hints); NOT ML",
        "num_examples": n,
        "accuracy": acc,
        "details": details,
    }
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"Rule-based coreference — {n} manually-annotated cases\n")
    print(f"Accuracy: {acc:.4f}  ({correct}/{n})\n")
    for d in details:
        mark = "OK " if d["correct"] else "XX "
        print(f"  {mark}{d['question']}  ->  {d['resolved'] or '(none)'}  "
              f"(expected {d['expected']})")
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
