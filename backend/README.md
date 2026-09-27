---
title: Marathi NLP Studio API
emoji: 🪔
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Marathi NLP Studio — API

FastAPI backend for **Marathi NLP Studio**: Named Entity Recognition over Marathi
text using a transformer fine-tuned on **L3Cube-MahaNER**
(`l3cube-pune/marathi-ner`, `BertForTokenClassification`). One shared model
instance serves every endpoint.

This Space is the backend only. The UI is a separate static frontend (deployed on
Vercel) that calls this API.

## Endpoints

| Method | Path            | Purpose                                             |
| ------ | --------------- | --------------------------------------------------- |
| GET    | `/health`       | Liveness + model status                             |
| GET    | `/model-info`   | Architecture, entity types, real seqeval metrics    |
| GET    | `/dataset-info` | Dataset splits + entity distribution (real data)    |
| POST   | `/predict`      | NER on submitted text                               |
| POST   | `/extract`      | Entities grouped into typed lists                   |
| POST   | `/search`       | Entity-aware ranking over the demo corpus           |
| POST   | `/assistant`    | Entity-derived Marathi answers about a passage      |

Interactive docs at `/docs`.

## Configuration (Space → Settings → Variables)

| Variable      | Example                                  | Notes                                  |
| ------------- | ---------------------------------------- | -------------------------------------- |
| `CORS_ORIGINS`| `https://your-app.vercel.app`            | Comma-separated; `*` allows any origin |
| `MODEL_NAME`  | `l3cube-pune/marathi-ner`                | Default; the official MahaNER model    |
| `MAX_LENGTH`  | `256`                                    | Tokenizer max length                   |

The model (~500 MB) downloads on first startup and is cached in `HF_HOME`
(`/tmp/huggingface`); the first request after a cold start takes longer.
