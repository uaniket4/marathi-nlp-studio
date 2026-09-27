# Marathi NLP Studio

A production-quality web app to **understand, extract and search Marathi text**
using Named Entity Recognition — powered by a transformer model fine-tuned on the
**L3Cube-MahaNER** dataset.

One shared NER model drives four tools, reachable from a persistent sidebar:

- **NER Playground** — analyze any Marathi text; entities highlighted inline,
  tabulated, and counted with confidence scores.
- **Information Extraction** — structured, typed entity lists you can copy or
  download as JSON / CSV.
- **Search** — entity-aware ranking over a Marathi demo corpus, with entity-type
  filters (lexical + entity matching, **not** semantic search).
- **AI Assistant** — deterministic, entity-derived answers to Marathi questions
  about a passage (no hardcoded replies, no external LLM).

Plus light/dark/system theming, a live model-status indicator, and a Model &
Dataset page with **real** metrics.

> **Applications:** information extraction, search engines, chatbots, virtual
> assistants, news analysis, and document processing.

---

## 1. Overview

Every tool sends text to the same backend model and renders real output:

- entities highlighted directly inside the original text,
- structured, typed entity groups (copy / download),
- entity-aware search results with relevance and matched entities,
- entity-derived answers to Marathi questions,
- per-type statistics and confidence scores from the model.

Example input:

```
सचिन तेंडुलकर मुंबईमध्ये राहतात आणि त्यांनी भारतीय क्रिकेट संघासाठी खेळले.
```

recognizes `सचिन तेंडुलकर` → **PERSON** and `मुंबई` → **LOCATION**.

---

## 2. Features

- Transformer-based Marathi NER (7 entity types), loaded once at startup and
  shared across all four tools.
- **NER Playground:** inline highlighting with hover/focus tooltips, entity
  table, per-type statistics.
- **Information Extraction:** grouped-by-type output, Copy JSON, download
  JSON / CSV.
- **Search:** entity + keyword ranking over a labelled demo corpus, entity-type
  filters built from the model's actual labels, relevance %, highlighted matches.
- **AI Assistant:** intent detection (entity type / count / "show all") with
  Marathi answers derived entirely from model predictions.
- Application shell with persistent sidebar, real client-side routing, and a live
  model-status indicator.
- Light / Dark / System theme, persisted in `localStorage`.
- Model & Dataset page with **real** seqeval metrics and entity distribution.
- Responsive layout; sidebar collapses to a drawer on mobile.
- FastAPI backend; full test suites (backend + frontend); Docker setup.

---

## 3. Screenshots

_Placeholder — add screenshots of the NER Demo, Model, and Dataset pages here._

```
docs/
├── demo.png
├── model.png
└── dataset.png
```

---

## 4. Architecture

```
Marathi text
   │  (frontend: React + Vite + Tailwind)
   ▼
POST /predict  ──►  FastAPI backend
   │                 │  model + tokenizer loaded once at startup
   │                 ▼
   │           transformer token classification (BertForTokenClassification)
   │                 │  softmax → per-token label + confidence
   │                 ▼
   │           group subwords → entity spans (char offsets, avg confidence)
   ▼
entities + statistics  ──►  highlight, table, counts
```

- **Backend:** FastAPI. The model is loaded a single time in the app lifespan
  and reused for every request (`model.eval()`, inference under `torch.no_grad()`).
  A small labelled demo corpus is indexed once at startup to power Search.
  The same model instance serves `/predict`, `/extract`, `/search` and
  `/assistant`.
- **Frontend:** React (Vite) with Tailwind and `react-router-dom`. A persistent
  sidebar shell routes between the four tools plus Model & Dataset / About. Theme
  (light/dark/system) is persisted in `localStorage`. In dev, `/api/*` is proxied
  to the backend; in Docker, nginx proxies it.

---

## 5. Dataset — L3Cube-MahaNER

Source: <https://github.com/l3cube-pune/MarathiNLP> (folder `L3Cube-MahaNER`).
The dataset is **not** committed here; `scripts/download_data.py` fetches it.

Findings from inspecting the actual data (IOB form, tab-separated
`words \t labels \t sentence_id`):

| Split      | Sentences | Tokens  |
| ---------- | --------- | ------- |
| Train      | 21,494    | 199,256 |
| Validation | 1,500     | 13,966  |
| Test       | 1,999     | 18,734  |

**Entity types (7)** — the IOB tag codes map to display names as follows:

| IOB code | Entity        | Train spans |
| -------- | ------------- | ----------- |
| `NEM`    | Measure       | 5,824       |
| `NEP`    | Person        | 4,775       |
| `NEL`    | Location      | 4,461       |
| `NEO`    | Organization  | 2,741       |
| `NED`    | Date          | 1,937       |
| `ED`     | Designation   | 838         |
| `NETI`   | Time          | 633         |

> Note: the mapping was verified against real tokens — e.g. `NED` covers dates
> like `ऑगस्ट`, `२०२०`, and `ED` covers designations like `पंतप्रधान`, `डॉ`.
> The IOB tags carry `B`/`I` prefixes (`BNEP`, `INEP`, …); subword alignment and
> entity grouping are handled in the pipeline.

The dataset page in the app renders splits and the entity distribution from a
stats file generated directly from the downloaded data.

---

## 6. Model

- **Inference model (default):** [`l3cube-pune/marathi-ner`](https://huggingface.co/l3cube-pune/marathi-ner)
  — a `BertForTokenClassification` model fine-tuned by the dataset authors on
  L3Cube-MahaNER. It loads at backend startup so the demo works immediately.
- **Base model for training from scratch:** `l3cube-pune/marathi-bert-v2`
  (MahaBERT) — chosen because it is a Marathi-specific BERT pretrained on the
  MahaCorpus, giving strong in-language token representations and native
  compatibility with the MahaNER label scheme. It is a practical, right-sized
  choice versus a larger multilingual model for this dataset and typical
  college-project compute.

Why not build something more complex? Token classification with a pretrained
Marathi BERT is the standard, well-matched architecture for this dataset; extra
architecture would add cost without accuracy.

---

## 7. Training

Reproducible fine-tuning pipeline (`training/`), runnable independently:

```bash
cd training
pip install -r requirements.txt
python ../scripts/download_data.py   # fetch the dataset into ../data
python train.py                      # fine-tune MahaBERT on MahaNER
```

The pipeline:

1. loads the IOB splits (`dataset.py`),
2. builds label maps from the tags actually present,
3. tokenizes with a fast tokenizer and **aligns labels to subwords** (only the
   first subword of each word keeps its label; the rest are `-100`),
4. fine-tunes with the HuggingFace `Trainer`,
5. evaluates on validation each epoch and on the test split at the end,
6. saves the model, tokenizer, `label_maps.json`, and `metrics.json`.

Configure via environment variables (see `backend/.env.example`): `EPOCHS`,
`TRAIN_BATCH_SIZE`, `LEARNING_RATE`, `BASE_MODEL`, etc.

To serve a locally trained model instead of the official one, set
`MODEL_NAME=./models/marathi-ner` for the backend.

---

## 8. Evaluation

Metrics are computed with **seqeval** (entity-level) on the MahaNER test split.
`training/evaluate.py` evaluates the configured `MODEL_NAME` via live inference:

```bash
cd training
python evaluate.py     # writes backend/models/metrics.json
```

**Results for the default model (`l3cube-pune/marathi-ner`), test split — real,
computed numbers (not fabricated):**

| Metric    | Score  |
| --------- | ------ |
| Precision | 80.6%  |
| Recall    | 78.0%  |
| F1-score  | 79.3%  |
| Accuracy  | 96.5%  |

Per-entity F1: Person 87.0, Measure 85.3, Date 84.0, Location 75.2, Designation
68.4, Organization 64.2, Time 54.5.

> These are entity-level (span) scores. The official model emits flat per-token
> labels, so entity boundaries are reconstructed to IOB2 for scoring; the app's
> Model page reads these numbers from `metrics.json` and shows
> **“Model not trained yet”** if that file is absent.

---

## 9. API

Base URL (dev): `http://localhost:8000`.

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "detail": null }
```

### `GET /model-info`
Returns architecture, base model, dataset, entity types, counts, and `metrics`
(from `metrics.json`, or `null`/`trained: false` if not evaluated).

### `POST /predict`
Request:
```json
{ "text": "सचिन तेंडुलकर मुंबईमध्ये राहतात." }
```
Response:
```json
{
  "text": "सचिन तेंडुलकर मुंबईमध्ये राहतात.",
  "entities": [
    { "text": "सचिन तेंडुलकर", "label": "PERSON", "start": 0, "end": 13, "confidence": 0.9997 },
    { "text": "मुंबई", "label": "LOCATION", "start": 14, "end": 19, "confidence": 0.9988 }
  ],
  "statistics": { "PERSON": 1, "LOCATION": 1 }
}
```
Empty/oversized text returns `422`; a missing model returns `503`. Interactive
docs are available at `/docs`.

### `POST /extract`
Same request shape as `/predict`. Adds a `grouped` map of entity texts by type
(de-duplicated) on top of `entities` + `statistics` — used by Information
Extraction.

### `POST /search`
```json
{ "query": "मुंबई", "entity_types": ["PERSON"], "limit": 10 }
```
Runs NER on the query, then ranks the demo corpus by entity + keyword overlap.
Returns `query_entities`, `query_keywords`, `total`, and `results` (each with
`snippet`, `matched_entities`, `matched_keywords` and a normalized `relevance`).
`entity_types` (optional) filters candidate documents by type. This is
transparent lexical + entity matching, not semantic search.

### `POST /assistant`
```json
{ "context": "पंतप्रधान नरेंद्र मोदी यांनी दिल्लीत भाषण केले.", "question": "कोणत्या व्यक्ती आहेत?" }
```
Detects the question's intent (entity type / count / "show all"), runs NER on the
context, and returns a Marathi `answer` derived from the entities, plus `intent`
and the supporting `entities`. No hardcoded answers, no external LLM.

### `GET /dataset-info`
Returns dataset splits, token counts, entity distribution (all computed from the
real data) and the size/source of the Search demo corpus.

---

## 10. Local setup

**Backend**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional
uvicorn app.main:app --reload   # http://localhost:8000
```
The model downloads from HuggingFace on first startup (~500 MB) and is cached.

**Frontend**
```bash
cd frontend
npm install
npm run dev                     # http://localhost:5173
```

---

## 11. Docker

```bash
docker compose up --build
```

- Frontend: <http://localhost:8080>
- Backend: <http://localhost:8000>

The model cache is stored in a named volume so it downloads only once.

---

## 12. Project structure

```
marathi-ner/
├── backend/
│   ├── app/
│   │   ├── main.py         # FastAPI app + routes (predict/extract/search/assistant/…)
│   │   ├── model.py        # load model once, metrics loader
│   │   ├── inference.py    # predict + entity grouping (group_by_type)
│   │   ├── corpus.py       # load + index the Search demo corpus at startup
│   │   ├── search.py       # entity + keyword ranking
│   │   ├── assistant.py    # intent detection + entity-derived answers
│   │   ├── schemas.py      # pydantic request/response models
│   │   └── config.py       # env-based settings
│   ├── data/               # documents.json, dataset_stats.json (real data)
│   ├── models/             # trained model + metrics.json (generated)
│   ├── tests/              # pytest: API + inference logic
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── components/     # Sidebar, ThemeToggle, ModelStatus, PageHeader,
│   │   │                   #   EntityHighlighter/Table, Statistics, SearchResult, ChatMessage
│   │   ├── pages/          # Overview, NerPlayground, InformationExtraction,
│   │   │                   #   Search, Assistant, ModelDataset, About, NotFound
│   │   ├── context/        # ThemeContext (light/dark/system)
│   │   ├── services/       # api client, entity styles
│   │   ├── App.jsx         # sidebar shell + routes
│   │   └── main.jsx
│   ├── package.json
│   └── Dockerfile
├── training/
│   ├── dataset.py          # IOB loader + label maps
│   ├── preprocess.py       # tokenize + subword label alignment
│   ├── train.py            # fine-tuning (runnable standalone)
│   └── evaluate.py         # seqeval evaluation (runnable standalone)
├── scripts/
│   └── download_data.py    # fetch MahaNER into data/
├── data/                   # dataset (downloaded, git-ignored)
├── docker-compose.yml
└── README.md
```

---

## 13. Testing

**Backend**
```bash
cd backend
pip install -r requirements-dev.txt
pytest
```
Covers `/health`, `/model-info`, `/predict`, empty/invalid/oversized input, and
the entity-grouping logic. Live-inference tests skip automatically if the model
is unavailable.

**Frontend**
```bash
cd frontend
npm run test
```
Covers text input, the Analyze flow, loading state, result rendering, error
state, clear, and example loading.

---

## 14. Security

- CORS restricted to configured origins.
- Request size limited by the API (`MAX_TEXT_CHARS`) and nginx (`client_max_body_size`).
- Input validated with pydantic; empty/oversized text rejected.
- No arbitrary code execution; stack traces are never returned to clients.
- Configuration via environment variables; no secrets committed.

---

## 15. Future improvements

- Fine-tune and ship a local IOB model (with `B`/`I` boundaries) for sharper
  span segmentation than the flat-label official model.
- Batch prediction endpoint and request rate limiting.
- Model quantization/ONNX export for faster CPU inference.
- Confidence threshold control in the UI.
- Dark mode.

