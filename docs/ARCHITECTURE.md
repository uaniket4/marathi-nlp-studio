# Marathi NLP Studio — Architecture & Syllabus Guide

A single, deterministic **no-LLM** Marathi NLP pipeline exposed through four web
tools. This document explains **every stage**, **which NLP-syllabus topic each
stage demonstrates**, **how to run the app**, and **how to present it to a
teacher / examiner**.

> Design principle: everything is deterministic, reproducible, and **measured**.
> No large language model is used at any stage — not in training, evaluation, or
> at runtime. Every number in this project comes from a real evaluation run.

---

## 1. High-level architecture

```
                            ┌──────────────────────────────────────┐
   Marathi text / question  │            FRONTEND (React)          │
   ───────────────────────► │  Vite + Tailwind, persistent sidebar │
                            │  NER Playground · Search · Assistant  │
                            │  Model & Dataset                      │
                            └───────────────┬──────────────────────┘
                                            │  HTTPS  /predict /extract
                                            │         /search /assistant
                                            ▼
                            ┌──────────────────────────────────────┐
                            │           BACKEND (FastAPI)          │
                            │  model + tokenizer loaded ONCE        │
                            │                                       │
                            │  Preprocess ─► Transformer NER (BERT) │
                            │       │              │                │
                            │       │              ▼                │
                            │       │        entity spans           │
                            │       ▼              │                │
                            │  Intent (TF-IDF+LR)  │                │
                            │       │              │                │
                            │       ▼              ▼                │
                            │  Coreference ─► Filter ─► MR template │
                            │                                       │
                            │  Retrieval (offline health TF-IDF) ◄──│
                            └──────────────────────────────────────┘
```

Two deployment targets:
- **Frontend** → Vercel (static build).
- **Backend** → AWS EC2 (FastAPI + PyTorch, quantized BERT), HTTPS via Caddy.

---

## 2. The pipeline, stage by stage

Each stage lists **what it does**, the **file** that implements it, and the
**syllabus topic it justifies**.

### Stage 1 — Text preprocessing
- **What:** tokenization, rule-based stemming, syllabification, and small Marathi
  stopword handling on the input text.
- **File:** [`backend/app/pipeline.py`](../backend/app/pipeline.py)
- **Syllabus topic:** *Text preprocessing & morphology* — tokenization,
  stemming, stopword removal, syllable segmentation for an Indic (Devanagari)
  script.

### Stage 2 — Transformer NER (the core model)
- **What:** a `BertForTokenClassification` model (MahaBERT-based) labels each
  subword; subword predictions are re-aligned to words and grouped into IOB2
  entity spans with averaged confidence. 7 entity types: Person, Location,
  Organization, Date, Time, Measure, Designation.
- **Files:** [`backend/app/model.py`](../backend/app/model.py) (load once at
  startup), [`backend/app/inference.py`](../backend/app/inference.py) (predict +
  span reconstruction).
- **Syllabus topic:** *Sequence labeling / Named Entity Recognition*, *word
  embeddings & contextual representations (BERT)*, *subword tokenization*.

### Stage 3 — POS tagging (HMM + Viterbi)
- **What:** part-of-speech tags produced by an offline-trained Hidden Markov
  Model decoded with the Viterbi algorithm, stored as a JSON probability table.
- **File:** [`backend/app/hmm_pos.py`](../backend/app/hmm_pos.py) +
  `hmm_pos_mr.json`.
- **Syllabus topic:** *Probabilistic sequence models* — HMMs, transition/emission
  probabilities, Viterbi decoding.

### Stage 4 — Intent classification (TF-IDF + Logistic Regression)
- **What:** the assistant routes a Marathi question to one of **9 intents**
  (e.g. `PERSON_QUERY`, `ENTITY_COUNT_QUERY`). Features are **TF-IDF n-grams**
  (word 1–2 + char 2–5); the classifier is **Logistic Regression**, trained
  offline with scikit-learn and exported to JSON. At runtime the TF-IDF
  transform + softmax is replayed in **pure numpy** (no scikit-learn dependency
  in production).
- **Files:** train [`training/train_intent_classifier.py`](../training/train_intent_classifier.py)
  → `backend/data/intent_clf.json`; runtime
  [`backend/app/intent_clf.py`](../backend/app/intent_clf.py); torch-free keyword
  baseline [`backend/app/keyword_intent.py`](../backend/app/keyword_intent.py).
- **Syllabus topic:** *Vector space model & TF-IDF*, *text classification*,
  *supervised learning (logistic regression)*, *n-gram features*.

### Stage 5 — Coreference resolution (rule-based)
- **What:** resolves pronouns in a follow-up question to the correct antecedent
  entity using **entity type + recency** (e.g. *ते → रतन टाटा*). Explicitly
  rule-based, not ML.
- **File:** [`backend/app/coref.py`](../backend/app/coref.py)
- **Syllabus topic:** *Discourse & coreference resolution*, *anaphora
  resolution*, *rule-based NLP*.

### Stage 6 — Information retrieval (offline health corpus)
- **What:** for a question with no working context, ranks trained Marathi health
  sentences with TF-IDF. No network calls, embeddings, or LLM.
- **File:** [`backend/app/chatbot.py`](../backend/app/chatbot.py),
  [`backend/app/search.py`](../backend/app/search.py) (entity + keyword ranking
  for the Search tool).
- **Syllabus topic:** *Information retrieval* — lexical matching, term overlap,
  relevance ranking.

### Stage 7 — Deterministic answer templating
- **What:** the final Marathi answer is filled from a fixed template using only
  real pipeline output (entities, counts). No text is generated by a model.
- **File:** [`backend/app/assistant.py`](../backend/app/assistant.py)
- **Syllabus topic:** *Natural language generation (template / rule-based NLG)*.

### Stage 8 — Evaluation
- **What:** NER scored with **seqeval** (entity-level P/R/F1) on the 1,999-sentence
  MahaNER test split; intent scored with **5-fold cross-validation** (accuracy,
  per-class P/R/F1, **macro-F1**) against a keyword baseline; coreference on a
  small manually-annotated set.
- **Files:** [`training/evaluate.py`](../training/evaluate.py),
  [`training/evaluate_ner.py`](../training/evaluate_ner.py),
  [`training/evaluate_intent.py`](../training/evaluate_intent.py),
  [`training/evaluate_coref.py`](../training/evaluate_coref.py).
- **Syllabus topic:** *Evaluation of NLP systems* — precision/recall/F1,
  macro-averaging, cross-validation, baseline comparison.

---

## 3. Syllabus coverage at a glance

| Syllabus topic | Where it is demonstrated | Measured result |
|---|---|---|
| Preprocessing / morphology | `pipeline.py` (tokenize, stem, syllabify, stopwords) | — |
| Subword tokenization + BERT | `inference.py`, `model.py` | — |
| Sequence labeling / NER | Transformer NER, 7 types | F1 **79.3%** (seqeval) |
| Probabilistic models (HMM/Viterbi) | `hmm_pos.py` | — |
| TF-IDF + text classification | Intent classifier | macro-F1 **73.1%** |
| Baseline comparison | keyword vs ML intent | baseline **38.4%** → ML **73.1%** |
| Coreference / anaphora | `coref.py` | **18/18** on demo set |
| Information retrieval | `chatbot.py`, `search.py` | offline TF-IDF and lexical ranking |
| Template-based NLG | `assistant.py` | deterministic answers |
| Evaluation methodology | seqeval, 5-fold CV, macro-F1 | reproducible, seed 42 |

Full numbers and reproduction commands are in
[`README.md`](../README.md) §8 and [`docs/datasets.md`](datasets.md).

---

## 4. Data — public vs manually curated

| Data | Kind | License | Used for |
|---|---|---|---|
| L3Cube-MahaNER (`data/IOB/`) | **PUBLIC** | CC BY-NC-SA 4.0 | NER train/eval |
| `l3cube-pune/marathi-ner` model | **PUBLIC** | CC-BY-4.0 | runtime NER |
| `training/intent_dataset.csv` | **MANUALLY CURATED** (not public) | project-internal | intent classifier |
| `training/coref_test.csv` | **MANUALLY ANNOTATED** (18 cases) | project-internal | coref demo |

Honesty note: the intent dataset is **not** claimed to be a public corpus — it is
hand-written for this project's 9 assistant intents. Coreference is a **rule-based
demonstration** on a small curated set, not a benchmark claim.

---

## 5. How to use the application

The app has four tools, all driven by the **same** NER model.

1. **NER Playground** — paste any Marathi sentence and click *Analyze*. Entities
   are highlighted inline, tabulated, and counted with confidence scores.
   *Try:* `सचिन तेंडुलकर मुंबईमध्ये राहतात.`
2. **Search** — type a query; the app runs NER on it and ranks a labelled demo
   corpus by entity + keyword overlap, with entity-type filters and a relevance %.
3. **AI Assistant** — ask a Marathi question:
   - An **open question** (e.g. `सचिन तेंडुलकर कोण आहेत?`) retrieves a Marathi
    offline health-corpus passage — that passage becomes the working **context**.
   - A **follow-up** (e.g. `कोणत्या व्यक्ती आहेत?` or `ते कुठे राहतात?`) is
     answered against that context: intent is classified, pronouns are resolved
     by coreference, NER runs, entities are filtered, and a Marathi answer is
     templated. The UI shows the **detected intent + confidence %** and any
     **coreference link** (`ते → रतन टाटा`).
4. **Model & Dataset** — shows the real seqeval metrics and the dataset entity
   distribution.

### Running it locally
```bash
# Backend
cd backend
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload                       # http://localhost:8000

# Frontend
cd frontend
npm install
npm run dev                                         # http://localhost:5173
```

### Reproducing every metric (dev machine, needs scikit-learn — dev-only)
```bash
pip install -r training/requirements.txt
python training/evaluate_ner.py       # NER per-entity precision table (seqeval)
python training/train_intent_classifier.py   # → backend/data/intent_clf.json
python training/evaluate_intent.py    # ML vs keyword baseline table
python training/evaluate_coref.py     # coreference accuracy on the demo set
```

---

## 6. How to explain it to a teacher (viva script)

Use this as a 2-minute walkthrough:

1. **Problem.** "Marathi is a low-resource language. I built a deterministic,
   end-to-end Marathi text-understanding studio that extracts named entities and
   answers questions — with **no LLM** anywhere, so every result is explainable
   and reproducible."
2. **Pipeline.** "A question flows through seven classical stages:
   *preprocessing → transformer NER → intent classification → coreference →
   retrieval → filtering → template answer.* Each maps to a syllabus topic." →
   point at the table in §3.
3. **Where the ML is.** "NER is a fine-tuned BERT token classifier. Intent is a
   **TF-IDF + Logistic Regression** classifier I trained myself and run in pure
   numpy. Coreference is **rule-based**. POS tagging is an **HMM decoded with
   Viterbi**."
4. **Why it is credible.** "I did not fabricate numbers. NER scores **79.3% F1**
   via seqeval on the public MahaNER test split. My ML intent classifier reaches
   **73% macro-F1**, nearly double the **38%** keyword baseline — I show both.
   Coreference is 18/18 on a small set I annotated, and I label it a rule-based
   demonstration, not a benchmark."
5. **Honesty.** "One dataset is public (L3Cube-MahaNER, CC BY-NC-SA 4.0); the
   intent dataset is hand-curated and I say so explicitly. No license is claimed
   without checking the source."
6. **Demo.** Open the Assistant: ask an open question (retrieval), then a
   follow-up with a pronoun (coreference + intent + template). Point out the
   *Detected intent*, *confidence*, and *coreference* chips in the UI.

Likely questions and short answers:
- *"Why no LLM?"* → determinism, explainability, reproducibility, and it keeps
  every stage a nameable syllabus technique.
- *"How is intent better than keywords?"* → it generalizes across inflected
  phrasings the fixed keyword lists miss; the macro-F1 nearly doubles.
- *"Is coreference machine-learned?"* → no, it is rule-based (type + recency);
  presented as a demonstration.
- *"Are the metrics real?"* → yes; every command in §5 regenerates them from
  data, seed 42.

---

## 7. Key files reference

| Concern | File |
|---|---|
| Routes / API | `backend/app/main.py` |
| Model load | `backend/app/model.py` |
| NER inference | `backend/app/inference.py` |
| Preprocessing | `backend/app/pipeline.py` |
| Intent (runtime) | `backend/app/intent_clf.py` |
| Intent (baseline) | `backend/app/keyword_intent.py` |
| Coreference | `backend/app/coref.py` |
| Assistant orchestration | `backend/app/assistant.py` |
| Retrieval | `backend/app/chatbot.py`, `search.py` | 
| Frontend shell | `frontend/src/App.jsx` |
| Assistant UI | `frontend/src/pages/Assistant.jsx`, `components/ChatMessage.jsx` |
| Training / evaluation | `training/*.py` |

