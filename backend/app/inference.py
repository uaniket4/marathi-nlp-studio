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
    into the original ``text``. This is the lightweight path used by search,
    the assistant and corpus indexing; it discards the per-token detail.
    """
    entities, _ = predict_with_trace(model, text)
    return entities


def predict_with_trace(model, text: str, top_k: int = 3) -> Tuple[List[Dict], List[Dict]]:
    """Run NER and return ``(entities, token_trace)``.

    ``token_trace`` exposes the per-token detail the UI uses to show *how the
    model sees the text*: the subword piece, its char offset, the raw predicted
    label, the normalized entity type + BIO boundary, the confidence (softmax
    max) and the top-k label probabilities. All values come straight from the
    model — nothing is fabricated.
    """
    text = text or ""
    if not text.strip():
        return [], []

    enc = model.tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=model.max_length,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"][0].tolist()
    token_pieces = model.tokenizer.convert_ids_to_tokens(input_ids)

    with torch.no_grad():
        logits = model.model(**enc).logits[0]
    probs = torch.softmax(logits, dim=-1)
    conf_vals, pred_ids = probs.max(dim=-1)
    pred_ids = pred_ids.tolist()
    conf_vals = conf_vals.tolist()

    # Per-token top-k probabilities for transparency.
    k = min(top_k, probs.shape[-1])
    topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1)
    topk_probs = topk_probs.tolist()
    topk_ids = topk_ids.tolist()

    trace: List[Dict] = []
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
        raw_label = model.id2label[pred_ids[idx]]
        etype, boundary = normalize_label(raw_label)
        is_special = start == 0 and end == 0

        trace.append(
            {
                "index": idx,
                "token": token_pieces[idx] if idx < len(token_pieces) else "",
                "start": int(start),
                "end": int(end),
                "is_special": is_special,
                "predicted_label": raw_label,
                "entity_type": to_display_type(etype) if etype else None,
                "boundary": boundary,
                "confidence": round(conf_vals[idx], 4),
                "top_k": [
                    {"label": model.id2label[tid], "prob": round(p, 4)}
                    for tid, p in zip(topk_ids[idx], topk_probs[idx])
                ],
            }
        )

        # Special tokens have (0, 0) offsets and never form entity spans.
        if is_special:
            continue
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
    return entities, trace


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


def build_steps(
    trace: List[Dict],
    entities: List[Dict],
    grouped: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """Describe the NER pipeline as ordered, human-readable stages.

    Every count here is derived from real model output (``trace``/``entities``),
    never fabricated. Descriptions explain what each stage does; the UI renders
    them as a step-by-step view of how the model processes the text.
    """
    content_tokens = [t for t in trace if not t["is_special"]]
    special_tokens = [t for t in trace if t["is_special"]]
    labeled_tokens = [t for t in content_tokens if t["entity_type"]]

    steps = [
        {
            "id": "tokenize",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Tokenization",
            "description": (
                f"The text is split into {len(content_tokens)} subword tokens "
                f"(plus {len(special_tokens)} special tokens like [CLS]/[SEP]) "
                "using the model's WordPiece tokenizer. Each token keeps its "
                "character offset in the original text."
            ),
            "count": len(content_tokens),
        },
        {
            "id": "classify",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Token classification",
            "description": (
                "The transformer produces a score for every entity label at "
                "each token. A softmax turns those scores into probabilities; "
                "the highest one is the token's predicted label and confidence."
            ),
            "count": len(trace),
        },
        {
            "id": "decode",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "BIO decoding",
            "description": (
                f"{len(labeled_tokens)} tokens were tagged with an entity type "
                "(the rest are 'O' / outside). B- marks the beginning of an "
                "entity and I- marks a continuation."
            ),
            "count": len(labeled_tokens),
        },
        {
            "id": "aggregate",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Span aggregation",
            "description": (
                f"Contiguous tokens of the same type are merged into "
                f"{len(entities)} entity span(s), mapped back to character "
                "positions, with confidence averaged over the merged tokens."
            ),
            "count": len(entities),
        },
    ]

    if grouped is not None:
        steps.append(
            {
                "id": "group",
                "module": "Module 6 — Applications",
                "experiment": "Exp 10",
                "title": "Grouping by type",
                "description": (
                    f"Entities are de-duplicated and grouped into "
                    f"{len(grouped)} type bucket(s) for structured extraction."
                ),
                "count": len(grouped),
            }
        )

    return steps

