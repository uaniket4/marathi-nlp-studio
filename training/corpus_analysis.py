"""Small corpus analysis over the public MahaNER train text (no new download).

Reports simple, classical corpus statistics over the Marathi words in the
already-vendored L3Cube-MahaNER training split: token/type counts, type-token
ratio, top content words, and coverage by a small Marathi stopword list. This is
a light descriptive analysis to characterise the corpus — no model, no LLM.

    python training/corpus_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "backend")))

from app.config import get_settings  # noqa: E402
from dataset import load_dataset  # noqa: E402

_OUT = os.path.join(_HERE, "corpus_stats.json")

# A small, high-frequency Marathi function-word list (for stopword coverage).
_STOPWORDS = {
    "आणि", "व", "आहे", "आहेत", "होता", "होते", "होती", "या", "हे", "ही",
    "तो", "ती", "ते", "का", "काय", "एक", "मध्ये", "ला", "चा", "ची", "चे",
    "पण", "किंवा", "म्हणून", "म्हणजे", "साठी", "पर्यंत", "नाही", "असे",
    "यांनी", "यांचा", "त्या", "त्याने", "आपण", "मी", "तू", "आम्ही",
}


def _is_devanagari_word(tok: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in tok)


def main() -> int:
    data = load_dataset(get_settings().DATA_DIR)
    train = data["train"]

    words = [w for seq in train.tokens for w in seq if _is_devanagari_word(w)]
    total = len(words)
    freq = Counter(words)
    types = len(freq)
    ttr = round(types / total, 4) if total else 0.0

    stop_hits = sum(c for w, c in freq.items() if w in _STOPWORDS)
    stop_coverage = round(stop_hits / total, 4) if total else 0.0

    content = [(w, c) for w, c in freq.most_common() if w not in _STOPWORDS]
    top_content = content[:20]

    stats = {
        "corpus": "L3Cube-MahaNER train split (public Marathi text)",
        "total_tokens": total,
        "unique_types": types,
        "type_token_ratio": ttr,
        "stopword_coverage": stop_coverage,
        "top_stopwords": [
            {"word": w, "count": c}
            for w, c in freq.most_common() if w in _STOPWORDS
        ][:10],
        "top_content_words": [{"word": w, "count": c} for w, c in top_content],
    }
    with open(_OUT, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print("Corpus analysis — L3Cube-MahaNER train split\n")
    print(f"Total word tokens (Devanagari): {total}")
    print(f"Unique word types:              {types}")
    print(f"Type-token ratio:               {ttr}")
    print(f"Stopword coverage:              {stop_coverage:.2%} "
          f"(small {len(_STOPWORDS)}-word list)\n")
    print("Top 20 content words (stopwords removed):")
    for w, c in top_content:
        print(f"  {w:<16}{c:>7d}")
    print(f"\nWrote {_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
