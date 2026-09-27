# Deployment

Marathi NLP Studio is split across two hosts, because the PyTorch model is far
too large for Vercel's serverless size limit:

- **Frontend (static Vite build) → Vercel** — already deployed.
- **Backend (FastAPI + PyTorch) → Hugging Face Spaces (Docker)** — one-time
  manual push (needs your Hugging Face login).

---

## Live URLs

| Piece    | URL                                                      |
| -------- | -------------------------------------------------------- |
| Frontend | https://marathi-nlp-studio.vercel.app                    |
| Backend  | https://uaniket4-marathi-nlp-studio.hf.space *(create it below)* |

The frontend is built with `VITE_API_BASE = https://uaniket4-marathi-nlp-studio.hf.space`,
so the Space **must** be owner `uaniket4`, name `marathi-nlp-studio` for the two
to connect without a rebuild.

---

## Backend → Hugging Face Spaces

The `backend/` folder is Space-ready: `backend/Dockerfile` (CPU torch, listens on
8000) and `backend/README.md` (Space card with `sdk: docker`, `app_port: 8000`).

**1. Log in and create the Space**

```bash
hf auth login                 # paste a token from https://huggingface.co/settings/tokens (write)
hf repo create marathi-nlp-studio --repo-type space --space_sdk docker
```

**2. Push the backend subtree to the Space**

From the repo root:

```bash
git remote add space https://huggingface.co/spaces/uaniket4/marathi-nlp-studio
git subtree push --prefix backend space main
```

(If the Space already has a commit and the push is rejected, run
`git push space "$(git subtree split --prefix backend):main" --force`.)

**3. Allow the frontend origin (CORS)**

In the Space UI → **Settings → Variables and secrets**, add:

| Variable       | Value                                   |
| -------------- | --------------------------------------- |
| `CORS_ORIGINS` | `https://marathi-nlp-studio.vercel.app` |

(Use `*` to allow any origin.) The Space rebuilds automatically. First startup
downloads the ~500 MB model, so the first request after a cold start is slow.

**4. Verify**

```bash
curl https://uaniket4-marathi-nlp-studio.hf.space/health
```

Then open the frontend — all four tools should return live results.

---

## Frontend → Vercel (already done, for reference)

```bash
cd frontend
vercel link --project marathi-nlp-studio
vercel env add VITE_API_BASE production   # value: the Space URL above
vercel deploy --prod
```

`frontend/vercel.json` pins the Vite framework and adds SPA rewrites so client
routes (`/search`, `/assistant`, …) don't 404 on refresh.

> **Auto-deploy note:** the Vercel project is linked to the GitHub repo. For
> pushes to build automatically, set **Project → Settings → Build & Deployment →
> Root Directory = `frontend`** in the Vercel dashboard (CLI deploys from
> `frontend/` already work without this).

---

## If you change the backend URL

If the Space ends up at a different URL, update it in one place and redeploy the
frontend:

```bash
cd frontend
vercel env rm VITE_API_BASE production
echo "https://<your-space-url>" | vercel env add VITE_API_BASE production
vercel deploy --prod
```
