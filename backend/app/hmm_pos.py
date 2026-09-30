"""Bigram HMM part-of-speech tagger with Viterbi decoding.

Loads the precomputed count tables (``data/hmm_pos_mr.json``, built offline by
``scripts/train_hmm_pos.py`` from the small UD_Marathi-UFAL treebank) exactly
once, then decodes a token sequence to a UPOS tag path with the Viterbi
algorithm. Add-one (Laplace) smoothing is applied to transition and emission
probabilities so unseen words/transitions still get a non-zero score.

This is a teaching demo trained on a *small* treebank — no accuracy is claimed.
Every probability derives from real corpus counts; nothing is fabricated.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

_TABLES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hmm_pos_mr.json")

# Universal POS tag -> short human description, for the UI.
UPOS_DESCRIPTIONS: Dict[str, str] = {
    "ADJ": "विशेषण (adjective)",
    "ADP": "शब्दयोगी अव्यय (adposition)",
    "ADV": "क्रियाविशेषण (adverb)",
    "AUX": "सहायक क्रियापद (auxiliary)",
    "CCONJ": "समुच्चयबोधक (coordinating conjunction)",
    "DET": "निश्चायक (determiner)",
    "INTJ": "उद्गारवाचक (interjection)",
    "NOUN": "नाम (noun)",
    "NUM": "संख्या (numeral)",
    "PART": "अव्यय (particle)",
    "PRON": "सर्वनाम (pronoun)",
    "PROPN": "विशेषनाम (proper noun)",
    "PUNCT": "विरामचिन्ह (punctuation)",
    "SCONJ": "उपपद (subordinating conjunction)",
    "VERB": "क्रियापद (verb)",
    "X": "इतर (other)",
}


class HMMPosTagger:
    """A bigram HMM tagger loaded from precomputed treebank counts."""

    def __init__(self, tables: Dict) -> None:
        self.source: str = tables.get("source", "")
        self.num_sentences: int = tables.get("num_sentences", 0)
        self.num_tokens: int = tables.get("num_tokens", 0)
        self.tagset: List[str] = tables.get("tagset", [])
        self._initial: Dict[str, int] = tables.get("initial_counts", {})
        self._transition: Dict[str, Dict[str, int]] = tables.get("transition_counts", {})
        self._emission: Dict[str, Dict[str, int]] = tables.get("emission_counts", {})
        self._tag_totals: Dict[str, int] = tables.get("tag_totals", {})
        # Vocabulary size across all tags, for emission add-one smoothing.
        vocab = set()
        for words in self._emission.values():
            vocab.update(words.keys())
        self._vocab_size = max(1, len(vocab))
        self._num_sent = max(1, self.num_sentences)
        self._num_tags = max(1, len(self.tagset))

    # --- smoothed log-probabilities (add-one) ---
    def _log_initial(self, tag: str) -> float:
        return math.log(
            (self._initial.get(tag, 0) + 1) / (self._num_sent + self._num_tags)
        )

    def _log_transition(self, prev: str, cur: str) -> float:
        row = self._transition.get(prev, {})
        total = sum(row.values())
        return math.log((row.get(cur, 0) + 1) / (total + self._num_tags))

    def _log_emission(self, tag: str, word: str) -> float:
        row = self._emission.get(tag, {})
        total = self._tag_totals.get(tag, 0)
        return math.log((row.get(word, 0) + 1) / (total + self._vocab_size + 1))

    def is_known(self, word: str) -> bool:
        return any(word in row for row in self._emission.values())

    def viterbi(self, tokens: List[str]) -> List[Tuple[str, str, float]]:
        """Decode ``tokens`` to ``[(token, tag, logprob), ...]``.

        ``logprob`` is the cumulative Viterbi log-probability of the best path up
        to and including that token. Returns [] for empty input.
        """
        if not tokens or not self.tagset:
            return []
        tags = self.tagset

        # V[i][tag] = best log-prob of a path ending in `tag` at position i.
        V: List[Dict[str, float]] = [{}]
        back: List[Dict[str, Optional[str]]] = [{}]
        first = tokens[0]
        for tag in tags:
            V[0][tag] = self._log_initial(tag) + self._log_emission(tag, first)
            back[0][tag] = None

        for i in range(1, len(tokens)):
            V.append({})
            back.append({})
            word = tokens[i]
            for cur in tags:
                emit = self._log_emission(cur, word)
                best_prev, best_score = None, -math.inf
                for prev in tags:
                    score = V[i - 1][prev] + self._log_transition(prev, cur) + emit
                    if score > best_score:
                        best_score, best_prev = score, prev
                V[i][cur] = best_score
                back[i][cur] = best_prev

        # Backtrack from the best final state.
        last = len(tokens) - 1
        best_tag = max(tags, key=lambda t: V[last][t])
        path: List[str] = [best_tag]
        for i in range(last, 0, -1):
            best_tag = back[i][best_tag]
            path.insert(0, best_tag)

        return [(tokens[i], path[i], round(V[i][path[i]], 4)) for i in range(len(tokens))]

    def sample_transitions(self, path_tags: List[str], limit: int = 6) -> List[Dict]:
        """A few smoothed transition probabilities used along ``path_tags``.

        Returns the consecutive (prev -> cur) tag pairs actually taken in the
        decoded path, with their smoothed probability — a transparent peek at the
        transition table the decoder used.
        """
        seen = set()
        out: List[Dict] = []
        for prev, cur in zip(path_tags, path_tags[1:]):
            key = (prev, cur)
            if key in seen:
                continue
            seen.add(key)
            row = self._transition.get(prev, {})
            total = sum(row.values())
            prob = (row.get(cur, 0) + 1) / (total + self._num_tags)
            out.append(
                {
                    "from": prev,
                    "to": cur,
                    "prob": round(prob, 4),
                    "count": row.get(cur, 0),
                }
            )
            if len(out) >= limit:
                break
        return out


_tagger: Optional[HMMPosTagger] = None


def get_tagger() -> HMMPosTagger:
    """Load (once) and return the shared HMM tagger."""
    global _tagger
    if _tagger is None:
        with open(_TABLES_PATH, "r", encoding="utf-8") as fh:
            _tagger = HMMPosTagger(json.load(fh))
    return _tagger
