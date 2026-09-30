"""Offline builder for the bigram HMM POS tables (dev-only, not run at runtime).

Downloads the UD_Marathi-UFAL treebank (Universal Dependencies, CC BY-SA 4.0),
counts initial-tag / tag-transition / tag-emission frequencies from the gold
(word, UPOS) sequences, and writes a compact JSON of *raw counts* to
``backend/data/hmm_pos_mr.json``. The runtime tagger (``app/hmm_pos.py``) loads
those counts and applies add-one smoothing inside Viterbi — so every number the
UI shows is a real corpus count, nothing is fabricated.

The resulting file is committed to the repo; runtime does no downloading or
training. Re-run this only to refresh the tables:

    python backend/scripts/train_hmm_pos.py

Data source: https://github.com/UniversalDependencies/UD_Marathi-UFAL
Licence: CC BY-SA 4.0. This trains on a *small* treebank; the tagger is a
teaching demo and no accuracy is claimed.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections import defaultdict
from typing import Dict, List, Tuple

_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_Marathi-UFAL/master/"
_FILES = [
    "mr_ufal-ud-train.conllu",
    "mr_ufal-ud-dev.conllu",
    "mr_ufal-ud-test.conllu",
]

_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "hmm_pos_mr.json")


def _fetch(name: str) -> str:
    url = _BASE + name
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read().decode("utf-8")


def _sentences(conllu_text: str) -> List[List[Tuple[str, str]]]:
    """Yield [(form, upos), ...] per sentence, skipping multiword/empty tokens."""
    sents: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    for line in conllu_text.splitlines():
        line = line.rstrip("\n")
        if not line:
            if cur:
                sents.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        idx = cols[0]
        if "-" in idx or "." in idx:  # multiword range / empty node
            continue
        form, upos = cols[1], cols[3]
        if upos == "_":
            continue
        cur.append((form, upos))
    if cur:
        sents.append(cur)
    return sents


def main() -> None:
    sentences: List[List[Tuple[str, str]]] = []
    for name in _FILES:
        try:
            sentences.extend(_sentences(_fetch(name)))
        except Exception as exc:  # noqa: BLE001 - dev script, report and continue
            print(f"  skipped {name}: {exc}")

    initial_counts: Dict[str, int] = defaultdict(int)
    transition_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    emission_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tag_totals: Dict[str, int] = defaultdict(int)

    for sent in sentences:
        if not sent:
            continue
        initial_counts[sent[0][1]] += 1
        prev_tag = None
        for form, tag in sent:
            tag_totals[tag] += 1
            emission_counts[tag][form] += 1
            if prev_tag is not None:
                transition_counts[prev_tag][tag] += 1
            prev_tag = tag

    tables = {
        "source": "UD_Marathi-UFAL (Universal Dependencies), CC BY-SA 4.0",
        "note": "Bigram HMM POS counts from a small treebank; Viterbi applies "
        "add-one smoothing at runtime. Teaching demo — no accuracy claimed.",
        "num_sentences": len(sentences),
        "num_tokens": sum(tag_totals.values()),
        "tagset": sorted(tag_totals.keys()),
        "initial_counts": dict(initial_counts),
        "transition_counts": {k: dict(v) for k, v in transition_counts.items()},
        "emission_counts": {k: dict(v) for k, v in emission_counts.items()},
        "tag_totals": dict(tag_totals),
    }

    os.makedirs(os.path.dirname(_OUT), exist_ok=True)
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump(tables, fh, ensure_ascii=False, indent=1)
    print(
        f"wrote {_OUT}: {tables['num_sentences']} sentences, "
        f"{tables['num_tokens']} tokens, {len(tables['tagset'])} tags"
    )


if __name__ == "__main__":
    main()
