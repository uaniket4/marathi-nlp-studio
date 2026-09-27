# Deployment

Marathi NLP Studio is split across two hosts, because the PyTorch model is far
too large for Vercel's serverless size limit:

- **Frontend (static Vite build) → Vercel** — already deployed.
- **Backend (FastAPI + PyTorch) → Render (Docker, free plan)** — connect the repo
  as a Blueprint (one dashboard step).

---

## Live URLs

| Piece    | URL                                                      |
| -------- | -------------------------------------------------------- |
| Frontend | https://marathi-nlp-studio.vercel.app                    |
| Backend  | `https://<your-service>.onrender.com` *(create it below)* |

---

## Backend → Render

`render.yaml` (repo root) is a Blueprint that builds `backend/Dockerfile` on
Render's **free** plan. To keep torch + BERT inside the free tier's 512 MB RAM it
sets `QUANTIZE=1` (int8 dynamic quantization) and `TORCH_THREADS=1`.

**1. Create the service**

- Render Dashboard → **New → Blueprint** → connect this GitHub repo.
- Render reads `render.yaml` and creates the `marathi-nlp-studio` web service.
- First build installs CPU torch + transformers; first boot downloads the
  ~500 MB model. The free plan spins down after 15 min idle, so the first request
  after a cold start is slow (it re-downloads the model — the free tier has no
  persistent disk).

**2. Note the URL** Render assigns `https://<service>.onrender.com`. Confirm
`CORS_ORIGINS` (set by the Blueprint to the Vercel URL) matches your frontend.

**3. Point the frontend at it**

```bash
cd frontend
vercel env rm VITE_API_BASE production
echo "https://<your-service>.onrender.com" | vercel env add VITE_API_BASE production
vercel deploy --prod
```

**4. Verify**

```bash
curl https://<your-service>.onrender.com/health
```

Then open the frontend — all four tools should return live results.

> **512 MB caveat:** even quantized, torch's peak memory while loading the model
> can brush against the free tier's limit. If Render's logs show the service
> killed/OOM on boot, bump it to the Starter plan, or use a host with more free
> RAM (Google Cloud Run: set memory 1–2 GB, scales to zero, generous free quota).

---

## Frontend → Vercel (already done, for reference)

```bash
cd frontend
vercel link --project marathi-nlp-studio
vercel deploy --prod
```

`frontend/vercel.json` pins the Vite framework and adds SPA rewrites so client
routes (`/search`, `/assistant`, …) don't 404 on refresh.

> **Auto-deploy note:** the Vercel project is linked to the GitHub repo. For
> pushes to build automatically, set **Project → Settings → Build & Deployment →
> Root Directory = `frontend`** in the Vercel dashboard (CLI deploys from
> `frontend/` already work without this).
