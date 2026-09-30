"""Deterministic, NER- and pipeline-powered Marathi assistant.

The assistant is conversational but **deterministic** — there is no external
LLM. It handles two kinds of turns:

  * conversational (greetings, help, thanks, identity) — fixed, friendly Marathi
    phrasings, never fabricated facts about the user's text; and
  * analytical — questions about the context text, answered from *real* output of
    the classical NLP pipeline (NER entities, HMM POS tags, rule-based stems,
    lexicon sentiment). Every number/word shown is computed, not invented.

Each answer carries syllabus-tagged reasoning ``steps`` so the UI can show how
the answer was derived.
"""

from __future__ import annotations

from typing import Dict, List

from . import chatbot, coref, hmm_pos, intent_clf, pipeline
from .inference import predict
from .keyword_intent import (  # torch-free keyword baseline (shared source of truth)
    _ALL_KEYWORDS,
    _COUNT_KEYWORDS,
    _INTENT_KEYWORDS,
    detect_intent,
)

# Conversational intents (do not need context). Order matters: checked top-down,
# so "wellbeing" is listed before "greeting" — "नमस्कार, कसे आहात?" and
# "hi how are you" carry both cues, and the more specific one should win.
_CONV_KEYWORDS: Dict[str, List[str]] = {
    "wellbeing": [
        "कसे आहात", "कसा आहेस", "कशी आहेस", "कसं आहे", "कसं चाललं", "कसं चाललंय",
        "कशी तब्येत", "how are you", "how are u", "how are ou", "how r u", "how are",
    ],
    "greeting": ["नमस्कार", "नमस्ते", "हॅलो", "हाय", "hello", "hi", "hey"],
    "thanks": ["धन्यवाद", "आभार", "thanks", "thank you", "thankyou"],
    "goodbye": ["बाय", "निरोप", "पुन्हा भेटू", "bye", "goodbye"],
    "identity": ["तू कोण", "तुम्ही कोण", "तुझे नाव", "तुमचे नाव", "who are you", "your name", "कोण आहेस"],
    "help": ["मदत", "काय करू शकत", "कशी मदत", "help", "what can you do", "कसे वापर"],
}

# Analytical (pipeline-grounded) intents and their trigger keywords.
_SENTIMENT_KEYWORDS = ["भावना", "मूड", "sentiment", "सकारात्मक", "नकारात्मक", "positive", "negative"]
_ROOT_KEYWORDS = ["मूळ शब्द", "मूळ", "धातू", "root", "stem", "मूळरूप"]
# POS query -> the UPOS tags that satisfy it.
_POS_QUERIES: Dict[str, Dict] = {
    "verb": {"tags": {"VERB", "AUX"}, "kw": ["क्रियापद", "verb"], "mr": "क्रियापदे"},
    "noun": {"tags": {"NOUN", "PROPN"}, "kw": ["नाम", "noun"], "mr": "नामे"},
    "adjective": {"tags": {"ADJ"}, "kw": ["विशेषण", "adjective"], "mr": "विशेषणे"},
    "pronoun": {"tags": {"PRON"}, "kw": ["सर्वनाम", "pronoun"], "mr": "सर्वनामे"},
}

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


def _empty_intent() -> Dict:
    return {"types": [], "is_count": False, "is_all": False}


# 9-intent ML label -> entity operation ``{types, is_count, is_all}``.
_INTENT_LABEL_TO_OP: Dict[str, Dict] = {
    "PERSON_QUERY": {"types": ["PERSON"], "is_count": False, "is_all": False},
    "LOCATION_QUERY": {"types": ["LOCATION"], "is_count": False, "is_all": False},
    "ORGANIZATION_QUERY": {"types": ["ORGANIZATION"], "is_count": False, "is_all": False},
    "DATE_QUERY": {"types": ["DATE"], "is_count": False, "is_all": False},
    "TIME_QUERY": {"types": ["TIME"], "is_count": False, "is_all": False},
    "MEASURE_QUERY": {"types": ["MEASURE"], "is_count": False, "is_all": False},
    "DESIGNATION_QUERY": {"types": ["DESIGNATION"], "is_count": False, "is_all": False},
    "ALL_ENTITIES_QUERY": {"types": [], "is_count": False, "is_all": True},
    "ENTITY_COUNT_QUERY": {"types": [], "is_count": True, "is_all": True},
}

# Below this softmax confidence, fall back to the keyword baseline (safer on
# phrasings the small ML training set never saw).
_INTENT_CONF_FLOOR = 0.35


def _classify_intent(question: str) -> Dict:
    """Pick the entity op via the ML classifier, falling back to keywords.

    Returns ``{op, label, confidence, source}`` where ``op`` is a
    ``{types, is_count, is_all}`` dict. The ML label drives the answer when it is
    confident; otherwise the deterministic keyword matcher (the baseline) does,
    so unusual phrasings still degrade gracefully — no LLM either way.
    """
    ml = intent_clf.classify(question)
    label, conf = ml.get("label"), ml.get("confidence", 0.0)
    if label and conf >= _INTENT_CONF_FLOOR:
        op = dict(_INTENT_LABEL_TO_OP[label])
        return {"op": op, "label": label, "confidence": conf, "source": "ml"}
    kw = detect_intent(question)
    return {"op": kw, "label": label, "confidence": conf, "source": "keyword"}



import re as _re


def _kw_match(q: str, kw: str) -> bool:
    """True if keyword ``kw`` occurs in ``q`` as a whole word / phrase.

    Plain substring matching wrongly fires short keywords inside longer words —
    e.g. "hi" inside "Shivaji" or "something", "hey" inside "they" — which used
    to misroute real questions to the greeting reply. Single-token keywords are
    therefore matched on word boundaries; multi-word keywords keep substring
    matching. ``\\w`` covers Devanagari in Python 3, so this works for both
    scripts.
    """
    kw = (kw or "").lower().strip()
    if not kw:
        return False
    if " " in kw:
        return kw in q
    return _re.search(r"(?<!\w)" + _re.escape(kw) + r"(?!\w)", q) is not None


def detect_conversational(question: str) -> str:
    """Return a conversational-intent key, or "" if the question is analytical."""
    q = (question or "").lower().strip()
    for intent, keywords in _CONV_KEYWORDS.items():
        if any(_kw_match(q, kw) for kw in keywords):
            return intent
    return ""


def _detect_analytical(question: str) -> str:
    """Return an analytical-intent key (sentiment/root/verb/...) or ""."""
    q = (question or "").lower()
    if any(_kw_match(q, kw) for kw in _SENTIMENT_KEYWORDS):
        return "sentiment"
    for key, spec in _POS_QUERIES.items():
        if any(_kw_match(q, kw) for kw in spec["kw"]):
            return "pos:" + key
    # Root/stem last: "मूळ" is broad, so let entity/POS matches win first.
    if any(_kw_match(q, kw) for kw in _ROOT_KEYWORDS):
        return "root"
    return ""


def _step(id_, title, description, count=None, module=None, experiment=None) -> Dict:
    return {
        "id": id_,
        "title": title,
        "description": description,
        "count": count,
        "module": module,
        "experiment": experiment,
    }


# --- conversational replies (deterministic, context-free) -------------------
# Each intent has a few interchangeable phrasings; consecutive turns of the same
# intent rotate through them (via ``_reply_cycle``) so repeats don't look
# identical. Still fully deterministic — no LLM, no invented facts.
_CONV_REPLIES: Dict[str, List[str]] = {
    "greeting": [
        "नमस्कार! मी मराठीत बोलणारा सहायक आहे. तुम्ही कोणताही प्रश्न विचारा — "
        "उदा. 'सचिन तेंडुलकर कोण आहेत?', 'करोना विषाणू', किंवा 'शिवाजी महाराजांबद्दल "
        "सांगा'. मी L3Cube मराठी संग्रहातून (corpus) सर्वात जुळणारे वाक्य शोधून उत्तर देतो.",
        "नमस्कार! 🙏 काय जाणून घ्यायचे आहे? एखादी व्यक्ती, ठिकाण किंवा विषयाबद्दल "
        "विचारा — मी माझ्या मराठी संग्रहातून जुळणारी वाक्ये शोधतो.",
        "नमस्कार! मला मराठीत प्रश्न विचारा; मी L3Cube संग्रहातून TF-IDF समानतेने "
        "जुळणारे वाक्य शोधून, त्यातील नावे (NER) ओळखून उत्तर देतो.",
    ],
    "wellbeing": [
        "मी उत्तम आहे, विचारल्याबद्दल धन्यवाद! 😊 तुमचा प्रश्न सांगा — मी माझ्या मराठी "
        "संग्रहातून जुळणारी माहिती शोधून उत्तर देतो.",
        "मी एक प्रोग्राम आहे, त्यामुळे नेहमी तयार आहे! तुम्हाला काय जाणून घ्यायचे आहे?",
        "छान आहे, धन्यवाद! बोला, कशाबद्दल माहिती हवी आहे?",
    ],
    "thanks": [
        "आपले स्वागत आहे! आणखी काही विचारायचे असल्यास मोकळ्या मनाने विचारा.",
        "काही हरकत नाही! आणखी एखादा प्रश्न असेल तर सांगा. 🙏",
    ],
    "goodbye": [
        "धन्यवाद! पुन्हा भेटू. 🙏",
        "निरोप! गरज पडल्यास पुन्हा विचारा. 🙏",
    ],
    "identity": [
        "मी एक साधा, नियमाधारित (deterministic) मराठी सहायक आहे. मी कोणताही बाह्य "
        "LLM वापरत नाही — प्रश्नांची उत्तरे मी L3Cube मराठी संग्रहातून (corpus) "
        "TF-IDF समानतेने सर्वात जुळणारे खरे वाक्य शोधून देतो आणि त्यात ओळखलेली नावे "
        "(NER) दाखवतो, त्यामुळे प्रत्येक उत्तर खऱ्या मजकुरावर आधारित असते.",
    ],
    "help": [
        "मी हे करू शकतो:\n"
        "• प्रश्नांची उत्तरे — 'सचिन तेंडुलकर कोण आहेत?', 'करोना विषाणू' "
        "(L3Cube मराठी संग्रहातून जुळणारे वाक्य शोधून)\n"
        "• एखाद्या विषयाची माहिती — 'शिवाजी महाराजांबद्दल सांगा'\n"
        "• उत्तरातील नावे/entities ओळखणे (NER)\n"
        "थेट खाली प्रश्न टाइप करा — मजकूर आधी टाकण्याची गरज नाही.",
    ],
}

# Per-intent rotation counter (process-local). Advances each time an intent
# fires, so repeated greetings cycle through the phrasings above.
_reply_cycle: Dict[str, int] = {}


def _conversational_answer(intent_key: str) -> Dict:
    variants = _CONV_REPLIES[intent_key]
    idx = _reply_cycle.get(intent_key, 0)
    reply = variants[idx % len(variants)]
    _reply_cycle[intent_key] = idx + 1
    steps = [
        _step(
            "intent",
            "Intent detection",
            f"The question matched a conversational intent ('{intent_key}'), so a "
            "fixed, deterministic reply is returned — no text facts are invented.",
            module="Conversational",
            experiment="—",
        )
    ]
    return {"answer": reply, "intent": _empty_intent(), "entities": [], "steps": steps}


# --- analytical replies (grounded in the real pipeline) ---------------------
def _sentiment_answer(context: str) -> Dict:
    words = [t for t in pipeline.tokenize(context) if pipeline._is_word(t)]
    data = pipeline._stage_sentiment(words)["data"]
    label_mr = {"positive": "सकारात्मक", "negative": "नकारात्मक", "neutral": "तटस्थ"}[data["label"]]
    lines = [f"या मजकुराची भावना: {label_mr}."]
    if data["positive_words"]:
        lines.append("सकारात्मक शब्द: " + ", ".join(dict.fromkeys(data["positive_words"])))
    if data["negative_words"]:
        lines.append("नकारात्मक शब्द: " + ", ".join(dict.fromkeys(data["negative_words"])))
    if not data["positive_words"] and not data["negative_words"]:
        lines.append("(शब्दकोशातील कोणताही भावनादर्शक शब्द आढळला नाही — म्हणून तटस्थ.)")
    lines.append("टीप: ही एक लहान उदाहरणादाखल शब्दसूची आहे, प्रशिक्षित classifier नाही.")
    steps = [
        _step("tokenize", "Word tokenization",
              f"मजकूर {len(words)} शब्दांत विभागला.", len(words),
              "Module 6 — Applications", "Exp 10"),
        _step("sentiment", "Lexicon-based sentiment",
              f"{len(data['positive_words'])} सकारात्मक + {len(data['negative_words'])} "
              f"नकारात्मक जुळणी → '{data['label']}'.", data["score"],
              "Module 6 — Applications", "Exp 10"),
    ]
    return {"answer": "\n".join(lines), "intent": _empty_intent(), "entities": [], "steps": steps}


def _pos_answer(context: str, pos_key: str) -> Dict:
    spec = _POS_QUERIES[pos_key]
    words = [t for t in pipeline.tokenize(context) if pipeline._is_word(t)]
    path = hmm_pos.get_tagger().viterbi(words)
    hits = [tok for tok, tag, _ in path if tag in spec["tags"]]
    hits = list(dict.fromkeys(hits))
    if hits:
        listing = "\n".join(f"{i}. {w}" for i, w in enumerate(hits, 1))
        answer = f"या मजकुरात {len(hits)} {spec['mr']} आढळली:\n{listing}"
    else:
        answer = f"या मजकुरात कोणतीही {spec['mr']} आढळली नाहीत."
    answer += (
        "\n\nटीप: POS tagging लहान UD_Marathi-UFAL treebank वर आधारित HMM आहे "
        "(शिक्षणासाठी demo) — त्यामुळे काही tags चुकू शकतात."
    )
    steps = [
        _step("tokenize", "Word tokenization",
              f"मजकूर {len(words)} शब्दांत विभागला.", len(words),
              "Module 6 — Applications", "Exp 10"),
        _step("pos", "HMM POS tagging (Viterbi)",
              f"Viterbi ने POS tags काढले; {len(hits)} शब्द '{pos_key}' प्रकारात आले.",
              len(hits), "Module 4 — POS tagging", "Exp 6, 7"),
    ]
    return {"answer": answer, "intent": _empty_intent(), "entities": [], "steps": steps}


def _root_answer(context: str) -> Dict:
    words = [t for t in pipeline.tokenize(context) if pipeline._is_word(t)]
    pairs = [(w, pipeline.stem(w)) for w in words]
    changed = [(w, root) for w, (root, sfx) in pairs if sfx]
    if changed:
        listing = "\n".join(f"{i}. {w} → {root}" for i, (w, root) in enumerate(changed, 1))
        answer = f"मूळ शब्द (नियमाधारित stemming):\n{listing}"
    else:
        answer = "कोणत्याही शब्दाला ओळखता येणारा प्रत्यय आढळला नाही, म्हणून मूळ रूप तेच राहते."
    answer += "\n\nटीप: हे नियमाधारित प्रत्यय-छाटणी आहे, पूर्ण lemmatizer नाही — काही मूळे अंदाजे आहेत."
    steps = [
        _step("tokenize", "Word tokenization",
              f"मजकूर {len(words)} शब्दांत विभागला.", len(words),
              "Module 6 — Applications", "Exp 10"),
        _step("stemming", "Rule-based stemming",
              f"{len(changed)} शब्दांतून ओळखीचे प्रत्यय काढून मूळ काढले.", len(changed),
              "Module 2 — Stemming / lemmatization", "Exp 2"),
    ]
    return {"answer": answer, "intent": _empty_intent(), "entities": [], "steps": steps}


# --- offline health retrieval (deterministic) -------------------------------
def _knowledge_answer(model, question: str) -> Dict:
    """Answer health questions from the offline, trained health corpus."""
    result = chatbot.search(question, top=3)
    if not result["sentences"] or result["best"] < chatbot.get_index().sim_floor:
        return {
            "answer": (
                "माफ करा, माझ्या आरोग्यविषयक प्रशिक्षण-संग्रहात या प्रश्नाचे "
                "विश्वसनीय उत्तर सापडले नाही. कृपया आरोग्य, लक्षणे, प्रतिबंध, "
                "आहार किंवा प्राथमिक उपचार याबद्दल विचारा."
            ),
            "intent": _empty_intent(),
            "entities": [],
            "steps": [
                _step("retrieve", "Health corpus retrieval",
                      "प्रश्न ऑफलाइन आरोग्य प्रशिक्षण-संग्रहावर तपासला; जुळणीचा "
                      "विश्वास मर्यादेपेक्षा कमी आहे.", 0,
                      "Module 6 — Applications", "TF-IDF"),
            ],
            "sources": [],
        }

    body = result["sentences"][0]
    entities = predict(model, body)
    return {
        "answer": body,
        "intent": _empty_intent(),
        "entities": entities,
        "steps": [
            _step("retrieve", "Health corpus retrieval",
                  f"ऑफलाइन प्रशिक्षण-संग्रहातील {result['n_docs']} आरोग्य-वाक्यांमधून "
                  f"TF-IDF समानतेने जुळणारे वाक्य निवडले (score {result['best']}).",
                  1, "Module 6 — Applications", "TF-IDF"),
            _step("ner", "NER over the answer",
                  f"उत्तरावर NER चालवून {len(entities)} नावे ओळखली.",
                  len(entities), "Module 6 — Applications", "Exp 10"),
        ],
        "sources": [],
    }


# --- entity-based replies (NER over the context) ----------------------------
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


def _intent_summary(intent: Dict) -> str:
    if intent["is_all"]:
        return "The question asks for all entities."
    if intent["types"]:
        kind = "count" if intent["is_count"] else "list"
        return f"Detected a {kind} question about: {', '.join(intent['types'])}."
    return "No specific entity type detected — summarising everything found."


def _build_steps(intent: Dict, total_entities: int, used_entities: int,
                 clf: Dict = None, coref_links: List[Dict] = None) -> List[Dict]:
    """Describe the assistant's deterministic reasoning as ordered steps.

    Every number is derived from real model output; there is no external LLM and
    no fabricated reasoning — the assistant answers from the entities the NER
    model actually found in the context. When the ML intent classifier and the
    coreference linker drive the answer, their real outputs are shown too.
    """
    if clf and clf.get("source") == "ml":
        intent_desc = (
            f"The ML intent classifier (TF-IDF + Logistic Regression, pure-numpy "
            f"runtime) labelled the question '{clf['label']}' with "
            f"{clf['confidence'] * 100:.1f}% confidence. " + _intent_summary(intent)
        )
        intent_module = "Module 5 — Text classification"
    elif clf:
        intent_desc = (
            f"The ML classifier was unsure ('{clf['label']}', "
            f"{clf['confidence'] * 100:.1f}%), so the keyword baseline decided. "
            + _intent_summary(intent)
        )
        intent_module = "Module 5 — Text classification"
    else:
        intent_desc = ("The question is matched against Marathi keyword patterns. "
                       + _intent_summary(intent))
        intent_module = "Conversational"

    steps = [
        _step("intent", "Intent classification", intent_desc,
              len(intent["types"]), intent_module, "Exp (intent)"),
    ]
    if coref_links:
        links_txt = "; ".join(
            f"{l['mention']} → {l['antecedent']}" for l in coref_links
        )
        steps.append(
            _step("coref", "Rule-based coreference",
                  f"{len(coref_links)} pronoun(s) resolved against context entities "
                  f"by type + recency (not ML): {links_txt}.",
                  len(coref_links), "Module 6 — Applications", "Coref (rule-based)")
        )
    steps.extend([
        _step(
            "ner", "NER over context",
            f"The NER model runs on the context and finds {total_entities} "
            "entity mention(s).",
            total_entities, "Module 6 — Applications", "Exp 10",
        ),
        _step(
            "derive", "Answer derivation",
            f"{used_entities} entity mention(s) relevant to the question are "
            "selected and formatted into a Marathi answer.",
            used_entities, "Module 6 — Applications", "Exp 10",
        ),
    ])
    return steps


def answer(model, context: str, question: str) -> Dict:
    """Produce a conversational, deterministic Marathi answer.

    Dispatch order: conversational intents (fixed replies) → if the user supplied
    context, the pipeline-grounded analytical/entity intents run over it →
    otherwise the question is answered from the offline health corpus.
    All deterministic; no external LLM.
    """
    context = (context or "").strip()

    # 1) Conversational turns need neither context nor retrieval.
    conv = detect_conversational(question)
    if conv:
        return _conversational_answer(conv)

    # 2) No context supplied → answer from the offline health corpus.
    if not context:
        return _knowledge_answer(model, question)

    # 3) Context supplied → analytical, pipeline-grounded intents over it.
    analytical = _detect_analytical(question)
    if analytical == "sentiment":
        return _sentiment_answer(context)
    if analytical == "root":
        return _root_answer(context)
    if analytical.startswith("pos:"):
        return _pos_answer(context, analytical.split(":", 1)[1])

    # 4) Entity (NER) intents — ML intent classifier + rule-based coreference.
    entities = predict(model, context)
    clf = _classify_intent(question)
    intent = clf["op"]
    types = intent["types"]

    # Rule-based coreference: resolve pronouns in the follow-up against the
    # entities found in the working context (type + recency; NOT ML).
    cref = coref.resolve(question, entities)
    coref_links = cref["links"]

    extra = {
        "intent_label": clf["label"],
        "intent_confidence": clf["confidence"],
        "context_used": context,
        "coref_links": coref_links,
    }

    if intent["is_all"] or not types:
        if not entities:
            return {
                "answer": "या मजकुरात कोणतीही नावे (entities) आढळली नाहीत.",
                "intent": intent,
                "entities": [],
                "steps": _build_steps(intent, 0, 0, clf, coref_links),
                **extra,
            }
        # A pure count question ("एकूण किती नावे आहेत?") gets a total, not a list.
        if intent["is_count"] and not types:
            n = len(entities)
            by_type = ", ".join(
                f"{_LABEL_MR.get(l, (l,))[0]} {sum(1 for e in entities if e['label'] == l)}"
                for l in sorted({e["label"] for e in entities})
            )
            answer = f"या मजकुरात एकूण {n} नावे (entities) ओळखली गेली."
            if by_type:
                answer += f"\nप्रकारानुसार: {by_type}."
            return {
                "answer": answer,
                "intent": intent,
                "entities": entities,
                "steps": _build_steps(intent, len(entities), len(entities), clf, coref_links),
                **extra,
            }
        lines = ["या मजकुरातील ओळखलेली नावे (entities):", ""]
        used: List[Dict] = []
        for label in sorted({e["label"] for e in entities}):
            texts = _unique_texts([e for e in entities if e["label"] == label])
            used.extend(e for e in entities if e["label"] == label)
            singular = _LABEL_MR.get(label, (label,))[0]
            lines.append(f"{singular} ({label}): " + ", ".join(texts))
        return {
            "answer": "\n".join(lines),
            "intent": intent,
            "entities": used,
            "steps": _build_steps(intent, len(entities), len(used), clf, coref_links),
            **extra,
        }

    parts: List[str] = []
    used = []
    for label in types:
        subset = [e for e in entities if e["label"] == label]
        used.extend(subset)
        parts.append(_answer_for_type(label, _unique_texts(subset), intent["is_count"]))
    return {
        "answer": "\n\n".join(parts),
        "intent": intent,
        "entities": used,
        "steps": _build_steps(intent, len(entities), len(used), clf, coref_links),
        **extra,
    }
