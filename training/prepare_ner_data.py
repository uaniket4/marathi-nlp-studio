"""Verify and describe the vendored L3Cube-MahaNER data (no re-download).

The IOB files under ``data/IOB/`` are already vendored (public L3Cube-MahaNER,
CC BY-NC-SA 4.0). This script does NOT download anything — it verifies the files
parse, reports split sizes / token counts / entity distribution, and prints the
MahaNER-code → 7-type label mapping the rest of the pipeline relies on.

    python training/prepare_ner_data.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "backend")))

from app.config import get_settings  # noqa: E402
from dataset import ENTITY_FULL_NAMES, load_dataset  # noqa: E402


def _entity_counts(split) -> Counter:
    """Count entities (B- tags) per type in a split."""
    c: Counter = Counter()
    for seq in split.tags:
        for tag in seq:
            if tag.startswith("B") and tag[1:] in ENTITY_FULL_NAMES:
                c[ENTITY_FULL_NAMES[tag[1:]]] += 1
    return c


def main() -> int:
    data_dir = get_settings().DATA_DIR
    print(f"Verifying L3Cube-MahaNER IOB data under {os.path.join(data_dir, 'IOB')}\n")
    data = load_dataset(data_dir)

    print(f"{'Split':<12}{'Sentences':>12}{'Tokens':>12}")
    print("-" * 36)
    for name in ("train", "validation", "test"):
        sp = data[name]
        print(f"{name:<12}{len(sp):>12d}{sp.num_tokens:>12d}")

    print("\nLabel mapping (MahaNER IOB code -> entity type):")
    for code, full in ENTITY_FULL_NAMES.items():
        print(f"  B{code}/I{code:<6} -> {full}")

    print("\nEntity distribution (train, by B- count):")
    dist = _entity_counts(data["train"])
    for ent, n in dist.most_common():
        print(f"  {ent:<14}{n:>8d}")

    print("\nProvenance: L3Cube-MahaNER "
          "(github.com/l3cube-pune/MarathiNLP, arXiv 2204.06029).")
    print("License: CC BY-NC-SA 4.0 (research use). Vendored, not re-downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
