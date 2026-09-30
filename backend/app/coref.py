"""Rule-based coreference resolution (NOT machine learning).

A small, transparent, deterministic linker: it maps pronouns in a follow-up
question to the most likely antecedent **entity** in the working context, using

  * **type** — personal pronouns (तो/ती/ते/त्याने/त्यांनी…) link to PERSON
    entities; place adverbs (तिथे/तेथे/तिकडे…) link to LOCATION entities;
  * **recency** — among candidates of the right type, the one that appears
    latest in the context (largest start offset) wins;
  * light **number** hints — plural pronouns (ते/त्यांनी/त्यांना/they) prefer
    the most recent PERSON but fall back to any PERSON if none is plural.

This is a rule-based *demonstration*, not a learned model. It takes the entity
list already produced by the NER stage, so it stays torch-free and can be
evaluated on its own (see training/evaluate_coref.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional

# Pronoun surface form -> entity type it refers to.
_PERSON_PRONOUNS = [
    "त्यांनी", "त्यांना", "त्यांचा", "त्यांचे", "त्यांची", "त्यांच्या",
    "त्याने", "त्याला", "त्याचा", "त्याचे", "त्याची",
    "तिने", "तिला", "तिचा", "तिचे", "तिची",
    "तो", "ती", "ते",
    "he", "she", "they", "him", "her", "them", "his", "their",
]
_LOCATION_PRONOUNS = [
    "तिथे", "तेथे", "तिकडे", "तिथून", "तेथून", "येथे", "इथे",
    "there", "here",
]

# Plural person pronouns (used only as a light preference hint).
_PLURAL_PERSON = {"त्यांनी", "त्यांना", "त्यांचा", "त्यांचे", "त्यांची", "त्यांच्या",
                  "ते", "they", "them", "their"}


def _most_recent(entities: List[Dict], label: str) -> Optional[Dict]:
    """Return the entity of ``label`` that appears latest in the context."""
    cands = [e for e in entities if e.get("label") == label]
    if not cands:
        return None
    return max(cands, key=lambda e: e.get("start", 0))


def _find_pronoun(question: str, pronouns: List[str]) -> Optional[str]:
    """Return the first listed pronoun that occurs as a token in ``question``.

    Longer forms are checked first (the list is ordered longest-first per type)
    so that त्यांनी is matched before ते. Matching is on whitespace-delimited
    tokens with light punctuation stripped, to avoid substring false hits.
    """
    toks = [t.strip("?.!,।'\"") for t in (question or "").lower().split()]
    tokset = set(toks)
    for p in pronouns:
        if p in tokset:
            return p
    return None


def resolve(question: str, entities: List[Dict]) -> Dict:
    """Resolve pronouns in ``question`` against context ``entities``.

    Returns ``{resolved_question, links}`` where ``links`` is a list of
    ``{mention, antecedent, type}``. The resolved question substitutes the first
    resolved pronoun with its antecedent surface form, so downstream stages can
    treat the follow-up as if it named the entity directly. If nothing resolves,
    ``resolved_question`` equals ``question`` and ``links`` is empty.
    """
    entities = entities or []
    links: List[Dict] = []
    resolved = question or ""

    person_pron = _find_pronoun(question, _PERSON_PRONOUNS)
    if person_pron:
        ant = _most_recent(entities, "PERSON")
        if ant:
            links.append({"mention": person_pron, "antecedent": ant["text"],
                          "type": "PERSON"})

    loc_pron = _find_pronoun(question, _LOCATION_PRONOUNS)
    if loc_pron:
        ant = _most_recent(entities, "LOCATION")
        if ant:
            links.append({"mention": loc_pron, "antecedent": ant["text"],
                          "type": "LOCATION"})

    # Substitute the first resolved mention into the question (surface rewrite).
    for link in links:
        resolved = resolved.replace(link["mention"], link["antecedent"], 1)

    return {"resolved_question": resolved, "links": links}
