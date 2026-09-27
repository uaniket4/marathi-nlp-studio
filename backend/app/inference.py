"""Inference: turn raw Marathi text into structured entity spans.

Uses the fast tokenizer's offset mapping to map subword predictions back to
character spans in the original text, then groups contiguous tokens of the same
entity into single spans with an averaged confidence.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

# IOB entity codes -> full names, kept in sync with the training package.
_CODE_TO_FULL = {
    "NEP": "Person",
    "NEL": "Location",
    "NEO": "Organization",
    "NED": "Date",
    "NEM": "Measure",
    "NETI": "Time",
    "ED": "Designation",
}


def normalize_label(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Normalize a model label to ``(entity_full_name, boundary)``.

    Returns ``(None, None)`` for the outside/other class. ``boundary`` is one of
    ``"B"``, ``"I"`` or ``"FLAT"`` (models without B/I distinction).

    Handles three schemes:
      * flat full names: ``Person`` / ``Other``
      * IOB2 full names: ``B-Person`` / ``I-Person``
      * MahaNER IOB codes: ``BNEP`` / ``INEP`` / ``BED``
    """
    if not label or label in ("O", "Other"):
        return None, None

    # IOB2 with an explicit separator, e.g. "B-Person".
    if len(label) > 2 and label[1] == "-" and label[0] in ("B", "I"):
        return label[2:], label[0]

    # MahaNER IOB code, e.g. "BNEP" / "INETI".
    if label[0] in ("B", "I") and label[1:] in _CODE_TO_FULL:
        return _CODE_TO_FULL[label[1:]], label[0]

    # Flat full name.
    if label in _CODE_TO_FULL.values():
        return label, "FLAT"

    # Unknown scheme -> treat the whole label as a flat type.
    return label, "FLAT"


def to_display_type(entity_full_name: str) -> str:
    """Uppercase display label, e.g. ``Person`` -> ``PERSON``."""
    return entity_full_name.upper()


def predict(model, text: str) -> List[Dict]:
    """Run NER on ``text`` and return a list of entity dicts.

    Each entity: ``{text, label, start, end, confidence}`` with char offsets
    into the original ``text``.
    """
    text = text or ""
    if not text.strip():
        return []

    enc = model.tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=model.max_length,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        logits = model.model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    conf_vals, pred_ids = probs.max(dim=-1)
    pred_ids = pred_ids.tolist()
    conf_vals = conf_vals.tolist()

    entities: List[Dict] = []
    cur: Optional[Dict] = None

    def flush() -> None:
        nonlocal cur
        if cur is not None:
            span_text = text[cur["start"]:cur["end"]].strip()
            if span_text:
                cur["text"] = span_text
                cur["confidence"] = round(cur["_conf_sum"] / cur["_n"], 4)
                del cur["_conf_sum"], cur["_n"]
                entities.append(cur)
        cur = None

    for idx, (start, end) in enumerate(offsets):
        # Special tokens have (0, 0) offsets.
        if start == 0 and end == 0:
            continue
        label = model.id2label[pred_ids[idx]]
        etype, boundary = normalize_label(label)
        if etype is None:
            flush()
            continue

        display = to_display_type(etype)
        # Start a new span when there is no open span, an explicit B- boundary,
        # or the entity type changes. Consecutive same-type tokens (including the
        # FLAT scheme, which has no B/I) extend the current span.
        starts_new = cur is None or boundary == "B" or cur["label"] != display
        if starts_new:
            flush()
            cur = {
                "label": display,
                "start": int(start),
                "end": int(end),
                "_conf_sum": conf_vals[idx],
                "_n": 1,
            }
        else:
            cur["end"] = int(end)
            cur["_conf_sum"] += conf_vals[idx]
            cur["_n"] += 1

    flush()
    return entities


def entity_statistics(entities: List[Dict]) -> Dict[str, int]:
    """Count entities per display type plus a total."""
    stats: Dict[str, int] = {}
    for ent in entities:
        stats[ent["label"]] = stats.get(ent["label"], 0) + 1
    return stats


def group_by_type(entities: List[Dict]) -> Dict[str, List[str]]:
    """Group entity surface texts by display type, de-duplicated, order-preserving.

    Produces the structured view used by information extraction, e.g.
    ``{"PERSON": ["रतन टाटा"], "ORGANIZATION": ["टाटा मोटर्स"]}``.
    """
    grouped: Dict[str, List[str]] = {}
    for ent in entities:
        bucket = grouped.setdefault(ent["label"], [])
        if ent["text"] not in bucket:
            bucket.append(ent["text"])
    return grouped

