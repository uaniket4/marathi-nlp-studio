"""Report NER evaluation results (per-entity Precision table).

The heavy lifting — real inference over the 1999-sentence L3Cube-MahaNER test
split and entity-level seqeval scoring — lives in ``training/evaluate.py`` and
its results are stored in ``backend/models/metrics.json``. This script is a thin
front-end over that: it prints the per-entity **Precision / Recall / F1** table
in a readable form. Nothing here re-computes or fabricates numbers.

    python training/evaluate_ner.py           # print the stored (real) metrics
    python training/evaluate_ner.py --run     # re-run inference, then print

``--run`` requires torch + transformers (the training environment) and rewrites
metrics.json via the existing evaluate._standalone().
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "backend")))

from app.config import get_settings  # noqa: E402

_METRICS = get_settings().METRICS_PATH


def _print_table(m: dict) -> None:
    summary = m.get("summary", {})
    counts = m.get("counts", {})
    print(f"NER evaluation — {m.get('dataset', 'dataset')} "
          f"(test: {counts.get('test', '?')} sentences)\n")
    print(f"{'Entity':<16}{'Precision':>11}{'Recall':>9}{'F1':>9}{'Support':>9}")
    print("-" * 54)
    for ent in sorted(m.get("per_entity", {})):
        v = m["per_entity"][ent]
        print(f"{ent:<16}{v['precision']:>11.4f}{v['recall']:>9.4f}"
              f"{v['f1']:>9.4f}{v['support']:>9d}")
    print("-" * 54)
    print(f"{'OVERALL':<16}{summary.get('precision', 0):>11.4f}"
          f"{summary.get('recall', 0):>9.4f}{summary.get('f1', 0):>9.4f}")
    print(f"\nToken-level accuracy: {summary.get('accuracy', 0):.4f}")
    print("Metrics are entity-level (seqeval), computed by training/evaluate.py.")


def main() -> int:
    if "--run" in sys.argv:
        from evaluate import _standalone  # noqa: E402  (needs torch)
        print("Re-running NER inference over the test split...\n")
        _standalone()
        print()
    if not os.path.isfile(_METRICS):
        print(f"No metrics file at {_METRICS}. Run with --run to generate it.")
        return 1
    with open(_METRICS, "r", encoding="utf-8") as fh:
        _print_table(json.load(fh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
