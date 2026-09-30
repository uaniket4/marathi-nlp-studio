"""Keyword-based intent detection (the baseline, torch-free).

This is the original rule-based intent matcher: it maps a Marathi question to
entity display types plus show-all / count flags using curated keyword lists.
It is kept as (a) a fast, dependency-free fallback inside the assistant and
(b) the documented **baseline** the ML intent classifier is compared against
(see training/evaluate_intent.py). It imports nothing heavy, so evaluation
scripts can use it without loading torch.
"""

from __future__ import annotations

from typing import Dict, List

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
