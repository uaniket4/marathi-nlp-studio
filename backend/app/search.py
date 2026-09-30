"""Entity-aware search over the local corpus.

Pipeline:

    query -> NER on query -> extract query entities + keywords
          -> match against indexed documents (entity + keyword overlap)
          -> relevance score -> ranked results

Ranking is a transparent, deterministic combination of entity-match and
keyword-match counts. This is lexical + entity matching, NOT semantic search.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from .corpus import Corpus
from .inference import predict

# Minimal Marathi/function-word stoplist so keyword matching focuses on content.
_STOPWORDS = {
    "आणि", "व", "आहे", "आहेत", "होते", "होता", "मध्ये", "यांनी", "यांचा",
    "या", "हे", "ही", "तो", "ती", "ते", "का", "काय", "कोण", "कोणत्या",
    "कोणती", "किती", "आज", "एक", "दोन", "साठी", "ला", "चा", "ची", "चे",
    "उदा", "मजकुरात", "दाखवा",
}

# Weights: an entity match is worth more than a bare keyword match.
_ENTITY_WEIGHT = 2.0
_KEYWORD_WEIGHT = 1.0

# Sentence boundaries for splitting an uploaded document into passages.
_SENTENCE_SPLIT = re.compile(r"[।.!?\n]+")


def _keywords(text: str) -> List[str]:
    out = []
    for tok in text.split():
        t = tok.strip(".,!?।:;\"'()")
        if len(t) >= 2 and t not in _STOPWORDS:
            out.append(t)
    return out


def _snippet(text: str, max_chars: int = 160) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _score_doc(
    doc_id: str,
    title: str,
    text: str,
    entities: List[Dict],
    entity_type_set: set,
    query_entity_texts: set,
    query_keywords: set,
    selected: set,
    max_score: float,
) -> Optional[Dict]:
    """Score one document/passage against the query. Returns None if it does not match."""
    # Entity-type filter: document must contain at least one selected type.
    if selected and not (entity_type_set & selected):
        return None

    matched_entities = [
        e
        for e in entities
        if e["text"] in query_entity_texts
        and (not selected or e["label"] in selected)
    ]
    entity_match_count = len({e["text"] for e in matched_entities})

    keyword_matches = sorted(kw for kw in query_keywords if kw in text)
    keyword_match_count = len(keyword_matches)

    raw = _ENTITY_WEIGHT * entity_match_count + _KEYWORD_WEIGHT * keyword_match_count
    if raw <= 0:
        return None
    relevance = round(min(1.0, raw / max_score), 4)

    return {
        "id": doc_id,
        "title": title,
        "snippet": _snippet(text),
        "text": text,
        "entities": entities,
        "matched_entities": matched_entities,
        "matched_keywords": keyword_matches,
        "entity_match_count": entity_match_count,
        "keyword_match_count": keyword_match_count,
        "relevance": relevance,
    }


def _max_score(query_entity_texts: set, query_keywords: set) -> float:
    return _ENTITY_WEIGHT * max(1, len(query_entity_texts)) + _KEYWORD_WEIGHT * max(
        1, len(query_keywords)
    )


def search(
    model,
    corpus: Corpus,
    query: str,
    entity_types: Optional[List[str]] = None,
    limit: int = 10,
) -> Dict:
    """Run the search pipeline and return ranked results + query analysis."""
    query = (query or "").strip()
    selected = set(entity_types or [])

    query_entities = predict(model, query) if query else []
    query_entity_texts = {e["text"] for e in query_entities}
    query_keywords = set(_keywords(query))
    max_score = _max_score(query_entity_texts, query_keywords)

    results = []
    for doc in corpus.documents:
        scored = _score_doc(
            doc.id,
            doc.title,
            doc.text,
            doc.entities,
            doc.entity_types,
            query_entity_texts,
            query_keywords,
            selected,
            max_score,
        )
        if scored is not None:
            results.append(scored)

    results.sort(key=lambda r: (r["relevance"], r["entity_match_count"]), reverse=True)

    steps = [
        {
            "id": "query_ner",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Query analysis",
            "description": (
                f"NER runs on the query, finding {len(query_entity_texts)} "
                f"entity term(s) and {len(query_keywords)} content keyword(s)."
            ),
            "count": len(query_entity_texts) + len(query_keywords),
        },
        {
            "id": "match",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Corpus matching & scoring",
            "description": (
                f"Each of the {len(corpus.documents)} corpus document(s) is scored "
                "by how many query entities and keywords it contains (entity match "
                f"×{int(_ENTITY_WEIGHT)}, keyword ×{int(_KEYWORD_WEIGHT)}). This is "
                "lexical + entity matching, not semantic search."
            ),
            "count": len(corpus.documents),
        },
        {
            "id": "rank",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Ranking",
            "description": (
                f"{len(results)} document(s) matched and are ranked by relevance."
            ),
            "count": len(results),
        },
    ]

    return {
        "query": query,
        "query_entities": query_entities,
        "query_keywords": sorted(query_keywords),
        "total": len(results),
        "results": results[:limit],
        "steps": steps,
    }


def _segment(document: str) -> List[Dict]:
    """Split a document into passages with char offsets, dropping empties."""
    passages = []
    pos = 0
    for i, raw in enumerate(_SENTENCE_SPLIT.split(document)):
        # Recover the passage's start offset in the original document.
        start = document.find(raw, pos) if raw else pos
        pos = start + len(raw)
        text = raw.strip()
        if text:
            passages.append({"id": f"p{i}", "start": start, "text": text})
    return passages


def search_in_text(
    model,
    document: str,
    query: str,
    entity_types: Optional[List[str]] = None,
    limit: int = 10,
) -> Dict:
    """Search within a user-supplied document, showing the pipeline steps.

    The document is segmented into passages; NER runs on each passage and on the
    query; passages are ranked with the same entity + keyword scoring used for
    corpus search. Returns ranked passages plus a description of each step.
    """
    document = document or ""
    query = (query or "").strip()
    selected = set(entity_types or [])

    query_entities = predict(model, query) if query else []
    query_entity_texts = {e["text"] for e in query_entities}
    query_keywords = set(_keywords(query))
    max_score = _max_score(query_entity_texts, query_keywords)

    passages = _segment(document)

    results = []
    for p in passages:
        p_entities = predict(model, p["text"])
        p_type_set = {e["label"] for e in p_entities}
        scored = _score_doc(
            p["id"],
            f"Passage {p['id'][1:]}",
            p["text"],
            p_entities,
            p_type_set,
            query_entity_texts,
            query_keywords,
            selected,
            max_score,
        )
        if scored is not None:
            results.append(scored)

    results.sort(key=lambda r: (r["relevance"], r["entity_match_count"]), reverse=True)

    steps = [
        {
            "id": "segment",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Document segmentation",
            "description": (
                f"The uploaded document is split into {len(passages)} passage(s) "
                "on sentence boundaries, so matches can be located precisely."
            ),
            "count": len(passages),
        },
        {
            "id": "query_ner",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Query analysis",
            "description": (
                f"NER runs on the query, finding {len(query_entity_texts)} "
                f"entity term(s) and {len(query_keywords)} content keyword(s)."
            ),
            "count": len(query_entity_texts) + len(query_keywords),
        },
        {
            "id": "match",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Passage matching & scoring",
            "description": (
                "Each passage is scored by how many query entities and keywords "
                f"it contains (entity match ×{int(_ENTITY_WEIGHT)}, keyword ×"
                f"{int(_KEYWORD_WEIGHT)}). This is lexical + entity matching, not "
                "semantic search."
            ),
            "count": len(passages),
        },
        {
            "id": "rank",
            "module": "Module 6 — Applications",
            "experiment": "Exp 10",
            "title": "Ranking",
            "description": (
                f"{len(results)} passage(s) matched and are ranked by relevance."
            ),
            "count": len(results),
        },
    ]

    return {
        "query": query,
        "query_entities": query_entities,
        "query_keywords": sorted(query_keywords),
        "total": len(results),
        "results": results[:limit],
        "steps": steps,
        "passage_count": len(passages),
    }
