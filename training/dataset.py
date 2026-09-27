"""Loader for the L3Cube-MahaNER dataset (IOB form).

The dataset files are tab-separated with a header row:

    words \t labels \t sentence_id

One token per line; rows are grouped into sentences by ``sentence_id``.
See the project README for the full data description.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Canonical IOB tag -> human-readable entity type for the MahaNER dataset.
# These are the ONLY entity types present in the data (verified by inspection).
ENTITY_FULL_NAMES: Dict[str, str] = {
    "NEP": "Person",
    "NEL": "Location",
    "NEO": "Organization",
    "NED": "Date",
    "NEM": "Measure",
    "NETI": "Time",
    "ED": "Designation",
}


@dataclass
class Split:
    """A single dataset split as parallel token/label sequences per sentence."""

    tokens: List[List[str]] = field(default_factory=list)
    tags: List[List[str]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def num_tokens(self) -> int:
        return sum(len(s) for s in self.tokens)


def read_iob_file(path: str) -> Split:
    """Read one ``*_iob.txt`` file into a :class:`Split`.

    Sentences are delimited both by a change in ``sentence_id`` and by blank
    lines, so we rely on ``sentence_id`` which is always present.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Run scripts/download_data.py first."
        )

    split = Split()
    cur_tokens: List[str] = []
    cur_tags: List[str] = []
    cur_sid: str | None = None

    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n")
        if "words" not in header:
            # No header (already stripped) -> rewind and treat as data.
            fh.seek(0)
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word, tag = parts[0], parts[1]
            sid = parts[2] if len(parts) >= 3 else cur_sid
            if cur_sid is not None and sid != cur_sid and cur_tokens:
                split.tokens.append(cur_tokens)
                split.tags.append(cur_tags)
                cur_tokens, cur_tags = [], []
            cur_sid = sid
            cur_tokens.append(word)
            cur_tags.append(tag)

    if cur_tokens:
        split.tokens.append(cur_tokens)
        split.tags.append(cur_tags)
    return split


def build_label_maps(splits: List[Split]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build ``label2id`` / ``id2label`` from the tags actually present.

    ``O`` is forced to id 0 for convention; the rest are sorted for stability.
    """
    labels = set()
    for sp in splits:
        for seq in sp.tags:
            labels.update(seq)
    labels.discard("O")
    ordered = ["O"] + sorted(labels)
    label2id = {lab: i for i, lab in enumerate(ordered)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def load_dataset(data_dir: str) -> Dict[str, Split]:
    """Load train/valid/test splits from ``data_dir/IOB``."""
    iob = os.path.join(data_dir, "IOB")
    return {
        "train": read_iob_file(os.path.join(iob, "train_iob.txt")),
        "validation": read_iob_file(os.path.join(iob, "valid_iob.txt")),
        "test": read_iob_file(os.path.join(iob, "test_iob.txt")),
    }
