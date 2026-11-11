# Deploy Chronos Studio for free

> One page. Copy the block for the platform you want. Everything below uses only
> free tiers (as of the project's Part 19 ship date). No credit card is needed
> for the options marked **free**; a card may be required only to *create* an
> account on some platforms. The whole app is two containers and needs no
> database, no Redis and no API key.
>
> **Publishing to other people?** Work through
> [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md) first. The checked-in
> `frontend/vercel.json` host is a placeholder, the container paths were only
> statically verified in the audit sandbox, and anonymous public access needs
> authentication, explicit CORS, traceback suppression, rate limits and a real
> Docker smoke test. For a private local install those steps are not required.

---

## 0. What you are deploying

```
browser ──► frontend (nginx: static SPA) ── /api/ ──► backend (FastAPI/uvicorn)
                 port 8080 in compose                       port 8000
```

The browser **always** calls a relative `/api/...` URL, so it never needs to
know a hostname. Sessions live in the backend process's memory only and expire
after `CHRONOS_SESSION_TTL_HOURS` (default 6). Data is not persisted: a restart
starts clean. This is a feature (§5.15 privacy), and also why a single free
container is enough.

Prerequisites: Docker + Docker Compose v2 locally, a free GitHub account.

```bash
git clone <your-fork> && cd chronos-studio
cp .env.example .env            # all values optional; GEMINI_API_KEY can stay blank
docker compose up --build       # → http://localhost:8080
curl http://localhost:8000/api/v1/health
# PDF / 300-dpi PNG exports (optional, larger image):
docker compose build --build-arg INSTALL_BROWSER=1 backend && docker compose up
```

Without a Gemini key the assistant runs in **offline mode** and the app is
fully functional. To enable live answers, get a free key from
<https://aistudio.google.com/apikey> and set `GEMINI_API_KEY` in `.env`.

---

## 1. Fastest: one container on Fly.io (free allowance)

Fly's free allowance covers one small always-on Machine. We serve the built SPA
and the API from one process by putting nginx in the same image.

```bash
# one-time
curl -L https://fly.io/install.sh | sh
fly auth login
fly launch --no-deploy --name chronos-studio --region sin   # Singapore ≈ Manila
```

Create `fly.toml`:

```toml
app = "chronos-studio"
primary_region = "sin"

[build]
  dockerfile = "Dockerfile.single"

[env]
  CHRONOS_DEBUG = "false"
  CHRONOS_ALLOW_ALL_ORIGINS = "true"
  CHRONOS_SESSION_TTL_HOURS = "6"
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = false      # free tier: keep the demo responsive
  min_machines_running = 1

[[vm]]
  memory = "512mb"
  cpu_kind = "shared"
  cpus = 1
```

`Dockerfile.single` (checked in) builds the SPA, installs nginx + Python, and
runs both under `supervisord`/a small entrypoint; see
[`scripts/single-container/`](scripts/single-container/). Then:

```bash
fly secrets set GEMINI_API_KEY=...     # optional
fly deploy
fly open
```

> If you prefer not to use a Dockerfile, deploy the **backend** Dockerfile and
> serve the prebuilt `frontend/dist` from any static host (§3) — but the
> single-container path is the least moving parts.

---

## 2. Two services: Render (free web service) + static site

1. **Backend** → *New → Web Service*, connect the repo, **Root Directory
   `backend`**, Runtime **Docker**, Instance **Free**, Health Check Path
   `/api/v1/health`. Env: `CHRONOS_DEBUG=false`,
   `CHRONOS_ALLOW_ALL_ORIGINS=true`, `CHRONOS_SESSION_TTL_HOURS=6`, and
   optionally `GEMINI_API_KEY`. Render supplies `$PORT`; the backend currently
   listens on 8000, so either set the container port to 8000 or add
   `CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}` in an
   override Dockerfile.
2. **Frontend** → *New → Static Site*, Root Directory `frontend`, Build Command
   `npm ci && npm run build`, Publish Directory `dist`. Add a **rewrite** rule
   `/* → /index.html` (status 200) for SPA routing.
3. Because the two services have different hostnames, the browser's relative
   `/api/` calls would hit the static host. Add a Render **rewrite/proxy** from
   `/api/*` on the static site to `https://<backend>.onrender.com/api/*`, or
   (simpler) use the single-container Fly option above.

Render's free web service sleeps after inactivity; the first request can take
~30 s. Mention this to demo audiences.

**Railway** is the same split: create two services from the same repo, set the
backend root to `backend/`, the frontend root to `frontend/`, add a `/api`
proxy on the frontend service (or use Railway's private networking with
`VITE_API_TARGET` at build time), and set the same env vars.

---

## 3. HF Spaces (Docker) + Vercel (static frontend)

**Hugging Face Spaces** — free CPU space with a Dockerfile:

1. *New Space → Docker → Blank*, CPU basic (free).
2. Copy `backend/Dockerfile` plus `app/`, `scripts/`, `requirements*.txt` into
   the Space. Spaces default to port **7860**; set `EXPOSE 7860` and
   `CMD uvicorn app.main:app --host 0.0.0.0 --port 7860` (or set `PORT`).
3. The Space URL is your API origin, e.g.
   `https://your-name-chronos.hf.space`.

**Vercel** — free static hosting for the SPA:

1. Import the repo, Root Directory `frontend`, Build `npm run build`, Output
   `dist`.
2. Add a rewrite in `frontend/vercel.json` (checked in) that proxies `/api/(.*)`
   to the HF Space URL, and a catch-all SPA rewrite:

```json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "https://your-name-chronos.hf.space/api/$1" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

3. `vercel --prod`. The browser still calls only relative `/api/` URLs, so no
   CORS change is needed.

> **Replace the placeholder before deploying.** The checked-in
> `frontend/vercel.json` destination `https://your-name-chronos.hf.space/api/$1`
> is an example, not a live service. Set it to your real HF Space or API origin.
> The static deploy test checks that `/api/` and `/index.html` rewrites exist;
> it does **not** validate the hostname. No Docker daemon was available in the
> audit sandbox, so none of these container paths has been started here.

> **Alternative:** put both frontend and backend in the HF Space using
> `Dockerfile.single` and skip Vercel entirely — one URL, simplest of all.

---

## 4. Checklist before you share the link

| Check | Command / where |
|---|---|
| Health | `curl https://<host>/api/v1/health` → `{"status":"ok",...}` |
| UI loads | open `/`, "New analysis" is enabled, no console errors |
| Demo works | click **Watch a demo → Seasonal demand**; 17/17 steps, score 100 |
| Seed data | **Example data** lists 19 datasets |
| Report | Finalize S15 → download one-file HTML; it opens offline |
| Privacy | **Delete session** clears the server-side session; no DB anywhere |
| No key needed | leave `GEMINI_API_KEY` blank and confirm offline assistant works |
| PDF (optional) | build with `INSTALL_BROWSER=1`, or expect `errors.txt` for PDF/PNG only |

Static verification of the whole deploy configuration (no Docker daemon
needed) is part of the test suite:

```bash
cd backend && PYTHONPATH=. pytest -q tests/test_part19_deploy.py
```

It parses `docker-compose.yml`, checks every Dockerfile `COPY` source exists,
checks `requirements.txt` covers `pyproject.toml`'s core dependencies, checks
`nginx.conf` proxies `/api/` to `backend:8000`, and checks that no
browser-facing source code hardcodes `localhost`.

---

## 5. Notes and limits

* **Memory:** the backend needs ≥512 MB. `scikit-learn`, `statsmodels` and
  Plotly figure JSON are the main consumers. The optimized 10k-row paste +
  profile + STL path takes 1.73–2.27 s on the dev sandbox; enable swap / use
  the 1 GB tier if a cold or heavily loaded free-tier instance is slower.
* **Cold starts:** free tiers sleep. A demo replay takes ~20 s on a warm
  container; add ~30 s on a cold Render service.
* **Scaling:** sessions are in-process. One replica is correct. With two
  replicas, pin sessions (sticky) or replace `SessionStore` with Redis — see
  `METHODS.md` §13.
* **Optional PDF:** Chromium adds ~150 MB. Default image omits it; all other
  nine export kinds still work.
* **HTTPS:** every host above supplies TLS. Never put a Gemini key in the
  frontend build; it is a server-side/per-request secret only.
