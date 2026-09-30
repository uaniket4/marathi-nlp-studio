"""Classical NLP pipeline orchestrator (syllabus-aligned).

Runs the text through the classical stages taught in the Mumbai University NLP
lab syllabus, each producing *real computed* output:

  * word tokenization                       (Module 6 / Exp 10)
  * morphological analysis: akshara
    syllabification + suffix detection        (Module 1-2 / Exp 1, 8)
  * rule-based stemming                       (Module 2 / Exp 2)
  * n-gram language modeling with
    Unsmoothed / Add-One / Add-Delta probs    (Module 3 / Exp 3, 4, 5)
  * bigram HMM POS tagging + Viterbi          (Module 4 / Exp 6, 7)
  * rule-based NP chunking / shallow parsing  (Module 5 / Exp 9)
  * NER (the transformer) + lexicon sentiment (Module 6 / Exp 10)

Everything here is pure Python over small precomputed JSON tables plus the
existing NER model — no new neural models, no fabricated numbers. Each stage is
labelled truthfully about what it is (heuristic stemmer, small-treebank HMM,
illustrative sentiment lexicon).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

from . import hmm_pos
from .inference import entity_statistics, predict_with_trace

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Add-delta smoothing constant used for the language-modeling stage.
_DELTA = 0.5

# --- lazy-loaded data assets ---
_suffixes: List[str] = []
_sentiment: Dict[str, set] = {}


def _load_assets() -> None:
    global _suffixes, _sentiment
    if not _suffixes:
        with open(os.path.join(_DATA_DIR, "mr_suffixes.json"), "r", encoding="utf-8") as fh:
            sfx = json.load(fh)["suffixes"]
        # Longest-first so the most specific inflection is stripped.
        _suffixes = sorted({s for s in sfx if s}, key=len, reverse=True)
    if not _sentiment:
        with open(
            os.path.join(_DATA_DIR, "mr_sentiment_lexicon.json"), "r", encoding="utf-8"
        ) as fh:
            lex = json.load(fh)
        _sentiment = {"positive": set(lex["positive"]), "negative": set(lex["negative"])}


# --- tokenization -----------------------------------------------------------
# A word is a run of Devanagari letters/matras or ASCII letters; digits (ASCII
# or Devanagari) form number tokens; punctuation (incl. the Marathi danda ।) is
# emitted as its own token. Devanagari combining marks (matras, anusvara,
# virama) live in the U+0900–U+097F block, so a plain block range keeps whole
# aksharas together — unlike ``\w``, which drops nonspacing marks.
_WORD = r"[A-Za-zऀ-॥॰-ॿ]+"
_NUM = r"[0-9०-९]+"
_TOKEN_RE = re.compile(_NUM + r"|" + _WORD + r"|[।॥.,!?;:\"'()\-]", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Split text into word, number and punctuation tokens (order-preserving)."""
    return _TOKEN_RE.findall(text or "")


def _is_word(token: str) -> bool:
    return bool(re.match(r"[A-Za-zऀ-॥॰-ॿ]", token or "", re.UNICODE))


# --- akshara syllabification ------------------------------------------------
def syllabify(word: str) -> List[str]:
    """Split a Devanagari word into orthographic syllables (aksharas).

    Combining marks (matras, anusvara, visarga, nukta) attach to the preceding
    cluster; a virama (्) binds the following consonant into the same cluster.
    """
    clusters: List[str] = []
    cur = ""
    prev_virama = False
    for ch in word:
        o = ord(ch)
        is_combining = (
            0x0900 <= o <= 0x0903
            or 0x093A <= o <= 0x094F
            or 0x0951 <= o <= 0x0957
            or o == 0x093C
            or 0x0962 <= o <= 0x0963
        )
        if not cur:
            cur = ch
        elif is_combining or prev_virama:
            cur += ch
        else:
            clusters.append(cur)
            cur = ch
        prev_virama = o == 0x094D
    if cur:
        clusters.append(cur)
    return clusters


# --- rule-based stemmer -----------------------------------------------------
def stem(word: str) -> Tuple[str, str]:
    """Strip the longest matching inflectional suffix. Returns (stem, suffix).

    Heuristic only: a suffix is stripped only when at least two aksharas remain,
    so short roots are not destroyed. ``suffix`` is "" when nothing was stripped.
    """
    _load_assets()
    if not _is_word(word):
        return word, ""
    for sfx in _suffixes:
        if word.endswith(sfx) and word != sfx:
            root = word[: -len(sfx)]
            if len(syllabify(root)) >= 2:
                return root, sfx
    return word, ""


def analyze(model, text: str) -> Dict:
    """Run the full syllabus pipeline over ``text`` and return the result.

    Returns ``{text, tokens, stages:[...], entities, statistics}`` where each
    stage is ``{id, module, experiment, title, description, data}``.
    """
    text = (text or "").strip()
    tokens = tokenize(text)
    word_tokens = [t for t in tokens if _is_word(t)]

    stages: List[Dict] = []
    stages.append(_stage_tokenize(tokens, word_tokens))
    stages.append(_stage_morphology(word_tokens))
    stages.append(_stage_stemming(word_tokens))
    stages.append(_stage_ngram(word_tokens))

    pos_path = hmm_pos.get_tagger().viterbi(word_tokens) if word_tokens else []
    stages.append(_stage_pos(pos_path))
    stages.append(_stage_chunk(pos_path))

    entities, _trace = predict_with_trace(model, text)
    stages.append(_stage_ner(entities))
    stages.append(_stage_sentiment(word_tokens))

    return {
        "text": text,
        "tokens": word_tokens,
        "stages": stages,
        "entities": entities,
        "statistics": entity_statistics(entities),
    }


def _stage(id_, module, experiment, title, description, data) -> Dict:
    return {
        "id": id_,
        "module": module,
        "experiment": experiment,
        "title": title,
        "description": description,
        "data": data,
    }


def _stage_tokenize(tokens: List[str], word_tokens: List[str]) -> Dict:
    punct = [t for t in tokens if not _is_word(t)]
    return _stage(
        "tokenize",
        "Module 6 — Applications",
        "Exp 10",
        "Word tokenization",
        (
            f"The text is split into {len(tokens)} token(s): {len(word_tokens)} "
            f"word(s) and {len(punct)} punctuation mark(s). This is a whitespace + "
            "punctuation tokenizer over Devanagari, independent of the model's "
            "subword tokenizer."
        ),
        {"tokens": tokens, "word_count": len(word_tokens), "punct_count": len(punct)},
    )


def _stage_morphology(word_tokens: List[str]) -> Dict:
    items = []
    for w in word_tokens:
        aksharas = syllabify(w)
        root, sfx = stem(w)
        items.append(
            {
                "word": w,
                "aksharas": aksharas,
                "syllable_count": len(aksharas),
                "suffix": sfx,
                "root": root,
            }
        )
    with_suffix = sum(1 for it in items if it["suffix"])
    return _stage(
        "morphology",
        "Module 1-2 — Word & morphology analysis",
        "Exp 1, 8",
        "Morphological analysis",
        (
            f"Each word is broken into orthographic syllables (aksharas) using "
            f"Devanagari Unicode rules, and a probable inflectional suffix is "
            f"detected ({with_suffix} of {len(items)} word(s) carry one). The "
            "guessed root is heuristic, not a full morphological analysis."
        ),
        {"items": items, "with_suffix": with_suffix},
    )


def _stage_stemming(word_tokens: List[str]) -> Dict:
    pairs = []
    for w in word_tokens:
        root, sfx = stem(w)
        pairs.append({"word": w, "stem": root, "suffix": sfx, "changed": bool(sfx)})
    changed = sum(1 for p in pairs if p["changed"])
    return _stage(
        "stemming",
        "Module 2 — Stemming / lemmatization",
        "Exp 2",
        "Rule-based stemming",
        (
            f"A rule-based stemmer strips known Marathi inflectional suffixes; "
            f"{changed} of {len(pairs)} word(s) were reduced to a shorter stem. "
            "This is heuristic suffix stripping, not a trained lemmatizer, so some "
            "stems are approximate."
        ),
        {"pairs": pairs, "changed": changed},
    )


def _stage_ngram(word_tokens: List[str]) -> Dict:
    unigrams = Counter(word_tokens)
    bigrams = Counter(zip(word_tokens, word_tokens[1:]))
    vocab = len(unigrams)

    rows = []
    for (w1, w2), c in bigrams.most_common():
        c1 = unigrams[w1]
        unsmoothed = c / c1 if c1 else 0.0
        add_one = (c + 1) / (c1 + vocab) if vocab else 0.0
        add_delta = (c + _DELTA) / (c1 + _DELTA * vocab) if vocab else 0.0
        rows.append(
            {
                "w1": w1,
                "w2": w2,
                "count": c,
                "w1_count": c1,
                "p_unsmoothed": round(unsmoothed, 4),
                "p_add_one": round(add_one, 4),
                "p_add_delta": round(add_delta, 4),
            }
        )

    top_unigrams = [{"word": w, "count": c} for w, c in unigrams.most_common(10)]
    return _stage(
        "ngram",
        "Module 3 — Language modeling",
        "Exp 3, 4, 5",
        "N-gram language model",
        (
            f"Real counts over the text: {vocab} unique word(s), "
            f"{sum(bigrams.values())} bigram token(s). For each bigram P(w2|w1) is "
            "shown Unsmoothed, with Add-One (Laplace) and with Add-Delta "
            f"(δ={_DELTA}) smoothing over a vocabulary of {vocab}."
        ),
        {
            "vocab_size": vocab,
            "bigram_count": sum(bigrams.values()),
            "delta": _DELTA,
            "top_unigrams": top_unigrams,
            "bigrams": rows,
        },
    )


def _stage_pos(pos_path: List[Tuple[str, str, float]]) -> Dict:
    tagger = hmm_pos.get_tagger()
    tagged = [
        {
            "token": tok,
            "tag": tag,
            "description": hmm_pos.UPOS_DESCRIPTIONS.get(tag, tag),
            "logprob": lp,
            "known": tagger.is_known(tok),
        }
        for tok, tag, lp in pos_path
    ]
    path_tags = [t for _, t, _ in pos_path]
    known = sum(1 for t in tagged if t["known"])
    return _stage(
        "pos",
        "Module 4 — POS tagging",
        "Exp 6, 7",
        "HMM POS tagging (Viterbi)",
        (
            f"A bigram HMM decodes the most likely POS tag sequence with the "
            f"Viterbi algorithm ({known} of {len(tagged)} word(s) seen in "
            f"training). Tables are counts from the small UD_Marathi-UFAL "
            f"treebank ({tagger.num_tokens} tokens) with add-one smoothing — a "
            "teaching demo, no accuracy claimed."
        ),
        {
            "tagged": tagged,
            "transitions": tagger.sample_transitions(path_tags),
            "treebank_tokens": tagger.num_tokens,
            "treebank_sentences": tagger.num_sentences,
            "source": tagger.source,
        },
    )


# NP chunk: optional determiner/number, any adjectives, then one or more nominals.
_NOMINAL = {"NOUN", "PROPN", "PRON"}


def _stage_chunk(pos_path: List[Tuple[str, str, float]]) -> Dict:
    seq = [(tok, tag) for tok, tag, _ in pos_path]
    chunks: List[Dict] = []
    i = 0
    n = len(seq)
    while i < n:
        j = i
        # optional DET / NUM
        if seq[j][1] in ("DET", "NUM"):
            j += 1
        # zero or more ADJ
        while j < n and seq[j][1] == "ADJ":
            j += 1
        # one or more nominal heads
        head_start = j
        while j < n and seq[j][1] in _NOMINAL:
            j += 1
        if j > head_start:  # a valid NP needs at least one nominal head
            chunks.append(
                {
                    "text": " ".join(tok for tok, _ in seq[i:j]),
                    "tokens": [tok for tok, _ in seq[i:j]],
                    "tags": [tag for _, tag in seq[i:j]],
                }
            )
            i = j
        else:
            i += 1
    return _stage(
        "chunk",
        "Module 5 — Parsing",
        "Exp 9",
        "NP chunking (shallow parsing)",
        (
            f"A rule-based grammar over the POS tags groups noun phrases "
            f"(optional determiner/number + adjectives + noun/pronoun head), "
            f"finding {len(chunks)} noun-phrase chunk(s). This is shallow parsing, "
            "not a full syntactic parse."
        ),
        {"chunks": chunks},
    )


def _stage_ner(entities: List[Dict]) -> Dict:
    return _stage(
        "ner",
        "Module 6 — Applications",
        "Exp 10",
        "Named entity recognition",
        (
            f"The fine-tuned transformer tags named entities, finding "
            f"{len(entities)} entity span(s). This is the real model output — the "
            "same NER used across the studio."
        ),
        {"entities": entities},
    )


def _stage_sentiment(word_tokens: List[str]) -> Dict:
    _load_assets()
    pos_hits, neg_hits = [], []
    for w in word_tokens:
        root, _ = stem(w)
        if w in _sentiment["positive"] or root in _sentiment["positive"]:
            pos_hits.append(w)
        elif w in _sentiment["negative"] or root in _sentiment["negative"]:
            neg_hits.append(w)
    score = len(pos_hits) - len(neg_hits)
    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"
    return _stage(
        "sentiment",
        "Module 6 — Applications",
        "Exp 10",
        "Lexicon-based sentiment",
        (
            f"Word-level polarity from a small illustrative Marathi lexicon: "
            f"{len(pos_hits)} positive and {len(neg_hits)} negative match(es) → "
            f"'{label}'. This is a lexicon lookup, not a trained classifier, and "
            "does not handle negation or context."
        ),
        {
            "label": label,
            "score": score,
            "positive_words": pos_hits,
            "negative_words": neg_hits,
        },
    )
