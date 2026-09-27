"""Entity-aware search over the local corpus.

Pipeline:

    query -> NER on query -> extract query entities + keywords
          -> match against indexed documents (entity + keyword overlap)
          -> relevance score -> ranked results

Ranking is a transparent, deterministic combination of entity-match and
keyword-match counts. This is lexical + entity matching, NOT semantic search.
"""

from __future__ import annotations

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

    max_score = _ENTITY_WEIGHT * max(1, len(query_entity_texts)) + _KEYWORD_WEIGHT * max(
        1, len(query_keywords)
    )

    results = []
    for doc in corpus.documents:
        # Entity-type filter: document must contain at least one selected type.
        if selected and not (doc.entity_types & selected):
            continue

        # Entity matches: query entity surface texts that appear in the document.
        matched_entities = [
            e
            for e in doc.entities
            if e["text"] in query_entity_texts
            and (not selected or e["label"] in selected)
        ]
        entity_match_count = len({e["text"] for e in matched_entities})

        # Keyword matches: query content words present in the document text.
        keyword_matches = sorted(
            kw for kw in query_keywords if kw in doc.text
        )
        keyword_match_count = len(keyword_matches)

        raw = _ENTITY_WEIGHT * entity_match_count + _KEYWORD_WEIGHT * keyword_match_count
        if raw <= 0:
            continue
        relevance = round(min(1.0, raw / max_score), 4)

        results.append(
            {
                "id": doc.id,
                "title": doc.title,
                "snippet": _snippet(doc.text),
                "text": doc.text,
                "entities": doc.entities,
                "matched_entities": matched_entities,
                "matched_keywords": keyword_matches,
                "entity_match_count": entity_match_count,
                "keyword_match_count": keyword_match_count,
                "relevance": relevance,
            }
        )

    results.sort(key=lambda r: (r["relevance"], r["entity_match_count"]), reverse=True)

    return {
        "query": query,
        "query_entities": query_entities,
        "query_keywords": sorted(query_keywords),
        "total": len(results),
        "results": results[:limit],
    }
