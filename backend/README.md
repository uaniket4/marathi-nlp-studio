# Marathi NLP Studio — API

FastAPI backend for **Marathi NLP Studio**: Named Entity Recognition over Marathi
text using a transformer fine-tuned on **L3Cube-MahaNER**
(`l3cube-pune/marathi-ner`, `BertForTokenClassification`). One shared model
instance serves every endpoint.

This is the backend only. The UI is a separate static frontend (deployed on
Vercel) that calls this API. See [../DEPLOY.md](../DEPLOY.md) for hosting.

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

## Configuration (environment variables)

| Variable       | Default                     | Notes                                        |
| -------------- | --------------------------- | -------------------------------------------- |
| `CORS_ORIGINS` | `http://localhost:5173,…`   | Comma-separated; `*` allows any origin       |
| `MODEL_NAME`   | `l3cube-pune/marathi-ner`   | The official MahaNER model                   |
| `MAX_LENGTH`   | `256`                       | Tokenizer max length                         |
| `QUANTIZE`     | `0`                         | `1` = int8 dynamic quantization (small hosts)|
| `TORCH_THREADS`| `1`                         | CPU threads for inference                    |
| `HF_HOME`      | (torch default)             | Model cache dir; set to a writable path      |

The model (~500 MB) downloads on first startup and is cached in `HF_HOME`; the
first request after a cold start takes longer.

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --port 8000        # http://localhost:8000/docs
```
