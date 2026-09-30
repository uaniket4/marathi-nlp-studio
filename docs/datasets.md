# Datasets

This project uses **one public dataset** (for NER), the **pretrained model's**
own training corpus (referenced, not re-downloaded), and **one manually-curated
dataset** (for intent classification). Coreference uses a tiny manually-annotated
test set for demonstration only. Nothing here is fabricated; licenses are taken
from the original sources.

---

## 1. L3Cube-MahaNER (PUBLIC) — used for NER training/evaluation

| Field | Value |
|---|---|
| **Dataset** | L3Cube-MahaNER (Marathi Named Entity Recognition) |
| **Source** | L3Cube-Pune, MarathiNLP |
| **URL** | https://github.com/l3cube-pune/MarathiNLP/tree/main/L3Cube-MahaNER |
| **Paper** | arXiv:2204.06029 |
| **Language** | Marathi (mr) |
| **Task** | Token-level NER (IOB tagging) |
| **# examples** | train 21,494 / valid 1,500 / test 1,999 sentences (≈199k train tokens) |
| **Entity types** | 7 — Person, Location, Organization, Date, Measure, Time, Designation |
| **License** | **CC BY-NC-SA 4.0** (research / non-commercial) |
| **Usage here** | Held-out **test split (1,999 sentences)** for NER evaluation (`training/evaluate.py`, seqeval, entity-level). |
| **Preprocessing** | TSV `words / labels / sentence_id`; grouped into sentences by `sentence_id` (`training/dataset.py`). |
| **Label mapping** | MahaNER IOB codes → types: `NEP→Person, NEL→Location, NEO→Organization, NED→Date, NEM→Measure, NETI→Time, ED→Designation`. Codes normalized to IOB2 (`B-Person`…) for scoring. |

Vendored under `data/IOB/{train,test,valid}_iob.txt` — **not re-downloaded** by any
script. `training/prepare_ner_data.py` verifies and describes it.

---

## 2. `l3cube-pune/marathi-ner` (PUBLIC MODEL) — the NER model

| Field | Value |
|---|---|
| **Model** | `l3cube-pune/marathi-ner` (base: `l3cube-pune/marathi-bert-v2`) |
| **URL** | https://huggingface.co/l3cube-pune/marathi-ner |
| **Task** | Token classification, 7 entity types (same as above) |
| **License** | **CC-BY-4.0** |
| **Usage here** | Runtime NER (`backend/app/model.py`), quantized; evaluated on the MahaNER test split. |

---

## 3. Intent dataset (MANUALLY CURATED — **not** publicly sourced)

| Field | Value |
|---|---|
| **Dataset** | `training/intent_dataset.csv` |
| **Origin** | **Manually curated for this project.** No public corpus matches these assistant-specific intents. |
| **Language** | Marathi (mr) + a few English / transliterated variants |
| **Task** | Sentence classification into 9 assistant intents |
| **# examples** | 207 rows (≈23 per intent) |
| **Intents** | `PERSON_QUERY, LOCATION_QUERY, ORGANIZATION_QUERY, DATE_QUERY, TIME_QUERY, MEASURE_QUERY, DESIGNATION_QUERY, ALL_ENTITIES_QUERY, ENTITY_COUNT_QUERY` |
| **License** | Project-internal (authored here). |
| **Usage here** | Train TF-IDF + LogisticRegression offline (`training/train_intent_classifier.py`); export to `backend/data/intent_clf.json`; run in pure numpy at runtime (`backend/app/intent_clf.py`). Evaluated vs the keyword baseline (`training/evaluate_intent.py`). |

> This dataset is **explicitly not** presented as a public dataset. It is small,
> hand-written, and intended only to drive the assistant's intent routing.

---

## 4. Coreference test set (MANUALLY ANNOTATED — demonstration only)

| Field | Value |
|---|---|
| **Dataset** | `training/coref_test.csv` |
| **Origin** | **Manually annotated for this project** (18 cases). |
| **Task** | Pronoun → antecedent resolution over a short Marathi context |
| **Method evaluated** | **Rule-based** (type + recency + light number hints) — *not* ML |
| **Usage here** | `training/evaluate_coref.py` reports resolution accuracy. Presented as a small **rule-based demonstration**, not a benchmark result. |

---

### Honesty notes
- NER metrics come from a real seqeval run over the 1,999-sentence public test split.
- Intent metrics come from a real scikit-learn + numpy run (5-fold CV, seed 42).
- Coreference accuracy is on a small manually-annotated set and is a rule-based demo.
- No LLM is used anywhere in training, evaluation, or runtime.
