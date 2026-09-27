"""Deterministic, NER-powered Marathi assistant.

Given a context text and a Marathi question, it detects the question's intent
(which entity type is asked about, and whether it is a count question), runs the
shared NER model over the context, and produces a readable Marathi answer from
the extracted entities. No hardcoded answers and no external LLM — every answer
is derived from real model output.
"""

from __future__ import annotations

from typing import Dict, List

from .inference import predict

# Question keyword -> entity display type. Marathi + a few English fallbacks.
_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "PERSON": ["व्यक्ती", "व्यक्तींचा", "व्यक्तीं", "कोण", "माणस", "person", "people"],
    "LOCATION": ["ठिकाण", "ठिकाणे", "ठिकाणां", "कुठे", "स्थळ", "शहर", "गाव", "location", "place"],
    "ORGANIZATION": ["संस्था", "संस्थां", "संघटना", "कंपनी", "संघ", "पक्ष", "organization", "company"],
    "DATE": ["तारीख", "तारखा", "तारखां", "दिनांक", "कधी", "date"],
    "TIME": ["वेळ", "वेळा", "समय", "time"],
    "MEASURE": ["मोजमाप", "परिमाण", "मापे", "measure", "quantity"],
    "DESIGNATION": ["पद", "पदनाम", "हुद्दा", "designation", "title"],
}

# "Show all entities" style questions.
_ALL_KEYWORDS = ["सर्व", "सगळ्या", "सगळे", "entities", "एंटिटी", "सारे"]
# Count questions.
_COUNT_KEYWORDS = ["किती", "संख्या", "count", "how many"]

# Marathi phrasing per type: (singular subject, "mention" phrase, "none" phrase).
_LABEL_MR = {
    "PERSON": ("व्यक्ती", "व्यक्तींचा उल्लेख आहे", "कोणत्याही व्यक्तीचा उल्लेख आढळला नाही"),
    "LOCATION": ("ठिकाणे", "ठिकाणांचा उल्लेख आहे", "कोणतेही ठिकाण आढळले नाही"),
    "ORGANIZATION": ("संस्था", "संस्थांचा उल्लेख आहे", "कोणत्याही संस्थेचा उल्लेख आढळला नाही"),
    "DATE": ("तारखा", "तारखांचा उल्लेख आहे", "कोणतीही तारीख आढळली नाही"),
    "TIME": ("वेळा", "वेळांचा उल्लेख आहे", "कोणतीही वेळ आढळली नाही"),
    "MEASURE": ("परिमाणे", "परिमाणांचा उल्लेख आहे", "कोणतेही परिमाण आढळले नाही"),
    "DESIGNATION": ("पदनामे", "पदनामांचा उल्लेख आहे", "कोणतेही पदनाम आढळले नाही"),
}


def detect_intent(question: str) -> Dict:
    """Return ``{types: [...], is_count: bool, is_all: bool}`` for a question."""
    q = (question or "").lower()
    types: List[str] = []
    for label, keywords in _INTENT_KEYWORDS.items():
        if any(kw.lower() in q for kw in keywords):
            types.append(label)
    is_all = any(kw.lower() in q for kw in _ALL_KEYWORDS)
    is_count = any(kw.lower() in q for kw in _COUNT_KEYWORDS)
    return {"types": types, "is_count": is_count, "is_all": is_all}


def _unique_texts(entities: List[Dict]) -> List[str]:
    out: List[str] = []
    for e in entities:
        if e["text"] not in out:
            out.append(e["text"])
    return out


def _answer_for_type(label: str, texts: List[str], is_count: bool) -> str:
    singular, mention, none_phrase = _LABEL_MR.get(
        label, (label, f"{label} उल्लेख आहे", f"{label} आढळले नाही")
    )
    if not texts:
        return f"या मजकुरात {none_phrase}."
    n = len(texts)
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(texts, 1))
    if is_count:
        return f"या मजकुरात {n} {singular} आहेत:\n{listing}"
    return f"या मजकुरात {n} {mention}:\n{listing}"


def answer(model, context: str, question: str) -> Dict:
    """Produce an entity-aware Marathi answer from the context text."""
    context = (context or "").strip()
    if not context:
        return {
            "answer": "कृपया आधी वर मराठी मजकूर जोडा, नंतर प्रश्न विचारा.",
            "intent": {"types": [], "is_count": False, "is_all": False},
            "entities": [],
        }

    entities = predict(model, context)
    intent = detect_intent(question)
    types = intent["types"]

    # "All entities" or no recognised intent -> summarise everything found.
    if intent["is_all"] or not types:
        if not entities:
            return {
                "answer": "या मजकुरात कोणतीही नावे (entities) आढळली नाहीत.",
                "intent": intent,
                "entities": [],
            }
        lines = ["या मजकुरातील ओळखलेली नावे (entities):", ""]
        used: List[Dict] = []
        for label in sorted({e["label"] for e in entities}):
            texts = _unique_texts([e for e in entities if e["label"] == label])
            used.extend(e for e in entities if e["label"] == label)
            singular = _LABEL_MR.get(label, (label,))[0]
            lines.append(f"{singular} ({label}): " + ", ".join(texts))
        return {"answer": "\n".join(lines), "intent": intent, "entities": used}

    # Specific type(s) asked about.
    parts: List[str] = []
    used: List[Dict] = []
    for label in types:
        subset = [e for e in entities if e["label"] == label]
        used.extend(subset)
        parts.append(_answer_for_type(label, _unique_texts(subset), intent["is_count"]))
    return {"answer": "\n\n".join(parts), "intent": intent, "entities": used}
