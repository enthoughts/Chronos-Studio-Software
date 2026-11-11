# Public-release checklist for Chronos Studio

> **For your own local use, you can stop here.** The private install is already
> validated: 1014 backend tests, 53 frontend tests, 91.61% `app/ts` coverage,
> clean typecheck/build and live smoke. The items below only matter if you later
> expose the app to other people on a public URL.
>
> This sandbox has **no Docker daemon**, **no Chromium binary**, and the optional
> `weasyprint` / `arch` / `pmdarima` / `prophet` / `statsforecast` / `sktime` /
> `tbats` / `lightgbm` / `xgboost` packages are absent. Those checks must run on
> your machine or a CI runner with Docker.

---

## 0. Pick the topology first

**Recommended: one container (`Dockerfile.single`).** It serves the SPA and API
from one origin, so the browser keeps using relative `/api/` URLs and you can
**ignore `frontend/vercel.json` entirely**. This removes the placeholder-host
problem with the fewest moving parts. See `DEPLOY.md` §1 for Fly.io, or use the
same image on Render / Railway / HF Spaces.

**Split topology:** backend on HF Spaces / Render / Railway, frontend on Vercel.
Then `frontend/vercel.json` must be edited to point at your real API origin:

```json
{
  "rewrites": [
    { "source": "/api/(.*)", "destination": "https://YOUR-REAL-API-ORIGIN/api/$1" },
    { "source": "/(.*)", "destination": "/index.html" }
  ]
}
```

Never deploy with `your-name-chronos.hf.space`. The static deploy test only
checks rewrite shapes, not the hostname.

---

## 1. Run a real container smoke test

On a machine with Docker:

```bash
cp .env.example .env
# Optional but recommended for report exports:
docker compose build --build-arg INSTALL_BROWSER=1 backend
docker compose up -d
curl -fsS http://localhost:8080/                       # SPA
curl -fsS http://localhost:8080/api/v1/health         # proxied API
curl -fsS http://localhost:8000/api/v1/health         # direct API
docker compose logs --tail=100
docker compose down
```

Then open `http://localhost:8080`, run **Watch a demo → Seasonal demand**, and
confirm 17/17 steps and a 100/100 verdict. For a single-container image:

```bash
docker build -f Dockerfile.single --build-arg INSTALL_BROWSER=1 -t chronos-public .
docker run --rm -p 8080:8080 -e PORT=8080 chronos-public
curl -fsS http://localhost:8080/api/v1/health
```

Add these commands to CI; see `.github/workflows/public-release.yml`.

---

## 2. Resolve optional dependencies and exports

`backend/requirements-deploy.txt` currently installs the core plus
`python-docx`, `playwright`, `scikit-learn` and `google-genai`. Missing optional
families are greyed out by `app/core/config.py::optional_features()`; that is
safe, but a public release should decide which ones to advertise.

- **PDF / 300-dpi PNG:** build with `INSTALL_BROWSER=1`, or set
  `CHRONOS_CHROME=/path/to/chromium`. Verify a finalized report produces PDF and
  high-resolution figure files, not `errors.txt`.
- **All model families / readers:** install from `pyproject.toml` extras, e.g.
  `pip install -e ".[report,ml,ai,dev]"`, or pin the ones you want in
  `requirements-deploy.txt` and rebuild. `weasyprint` needs system Pango/Cairo
  libraries; `kaleido`/Chromium are needed for static figures.
- **Either install them or keep them visibly disabled.** Do not advertise a
  feature that `/api/v1/health` reports as unavailable.
- Keep `GEMINI_API_KEY` blank if you want the fully offline assistant; set it
  through platform secrets, never in a committed `.env`.

---

## 3. Production configuration

Start from `.env.production.example` (created alongside this file). Important
differences from the dev defaults:

```dotenv
CHRONOS_DEBUG=false
CHRONOS_ALLOW_ALL_ORIGINS=false
CHRONOS_ALLOWED_ORIGINS=https://your-real-domain.example
CHRONOS_SESSION_TTL_HOURS=2
CHRONOS_UPLOAD_DIR=/tmp/chronos_uploads
INSTALL_BROWSER=1
```

Override the same values in `docker-compose.yml`, `Dockerfile.single`,
`backend/Dockerfile` and the platform env/secrets panel. The checked-in Docker
files currently set `CHRONOS_ALLOW_ALL_ORIGINS=true`; that is acceptable for a
same-origin single container but should be changed for a split deployment.

Also make these settings environment-driven before a public launch:

- `Settings.max_sessions` (currently hardcoded 200)
- `Settings.max_upload_mb` (currently hardcoded 64)
- `Settings.ai_max_requests_per_session`, `ai_max_requests_per_minute`,
  `ai_min_interval_seconds` (currently hardcoded)
- `docs_url` / `openapi_url` in `app/main.py` (currently always enabled)

---

## 4. Security code changes required for anonymous public access

CORS is **not authentication**. The current app is fine for one trusted user,
but a public URL needs at least the following.

1. **Protect session endpoints.** `GET/DELETE /sessions/{session_id}`,
   `/results`, `/explain` and `/reset` currently accept a session ID in the URL
   without requiring a matching `X-Session-Id` header or a login. Add one of:
   - a real auth layer (signed cookie / OAuth / reverse-proxy auth), or
   - at minimum require `X-Session-Id` on those routes and compare it with the
     path parameter.
   Session IDs are opaque UUIDs but are still bearer tokens if leaked.

2. **Do not return Python tracebacks in production.**
   `app/core/errors.py::as_error_payload()` puts a 4,000-character traceback in
   `error.technical` for unexpected 500s regardless of `CHRONOS_DEBUG`. Gate it:

   ```python
   technical = (
       "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
       if get_settings().debug else
       "Enable CHRONOS_DEBUG=true locally for diagnostic details."
   )
   ```

   Then update `tests/test_part0_contracts.py::test_unexpected_exception_is_not_leaked_as_the_primary_message`
   to assert both debug and production behaviour.

3. **Disable or protect `/docs` and `/openapi.json`** when `debug` is false.

4. **Tighten CORS** to `CHRONOS_ALLOWED_ORIGINS`, not `*`. Use HTTPS and set
   `CHRONOS_ALLOW_ALL_ORIGINS=false`.

5. **Add nginx hardening and rate limits** to `frontend/nginx.conf` and
   `scripts/single-container/nginx.conf.template`:
   - `X-Content-Type-Options nosniff`, `X-Frame-Options DENY`,
     `Referrer-Policy`, and a practical `Content-Security-Policy`;
   - `limit_req` for `/api/` and especially `/api/v1/demo/` and `/api/v1/ai/`;
   - align `client_max_body_size` with the backend 64 MB limit (currently 25m);
   - tune the 300 s proxy timeouts.

6. **Harden the single-container image.** `backend/Dockerfile` already uses a
   non-root `chronos` user, but `Dockerfile.single` runs nginx and uvicorn as
   root. Add an unprivileged user / unprivileged nginx port or use a rootless
   base image.

7. **Upload hygiene.** `read_any()` checks the 64 MB limit and allowed
   extensions, but a public service should also validate content type, reject
   path-traversal filenames, scan uploads if users can share files, and cap
   paste size. Keep `CHRONOS_UPLOAD_DIR` on a size-limited tmpfs as
   `docker-compose.yml` already does.

8. **Secrets and logs.** Never bake a real `GEMINI_API_KEY` into an image or
   `.env`; use platform secrets, rotate it, and confirm no key or uploaded data
   appears in logs. The config loader deliberately does not log `.env` values.

9. **Dependency and image scanning.** Pin versions, run `pip-audit` /
   `npm audit`, and rebuild the image regularly.

---

## 5. Persistence, scaling and load

`app/core/session.py::SessionStore` is thread-safe but **process-local and
in-memory**. A restart wipes every session; multiple uvicorn workers or multiple
instances do not share sessions.

For a small public deployment:

- run **one** uvicorn worker per container, or enable sticky sessions;
- keep `min_machines_running = 1` on Fly so the demo stays responsive;
- lower `CHRONOS_SESSION_TTL_HOURS` and cap `max_sessions`;
- add `/api/v1/health` liveness/readiness checks and log/memory monitoring;
- load-test the 10k-row paste + S1 + S4 path and the 17-step demo with your
  expected concurrent users.

For a larger deployment, replace only `SessionStore` with Redis or a database
as its docstring suggests; every caller already goes through `get_store()`.
Add authentication and per-user quotas at the same time.

---

## 6. Statistical credibility before calling it a public forecasting tool

These are not deploy blockers for a clearly labelled research prototype, but
they are blockers for scientific/operational claims:

- validate on real out-of-domain datasets and against independent R /
  statsmodels / published benchmarks;
- have a statistician or domain reviewer check `METHODS.md` and `AUDIT.md`;
- keep the disclosed caveats visible: overlapping folds, low power in short
  series, Gaussian VaR, and squared-return realised-variance proxy;
- do not present the app as validated for medical, financial, safety-critical
  or regulated decisions;
- if you enable live Gemini, keep the offline assistant as the default and
  publish the AI privacy/data-use note.

---

## 7. Frontend performance

The production JS chunk is 504.04 kB / 142.74 kB gzip, just above Vite's 500 kB
advisory. For a public mobile audience, split chart-heavy panels with
`React.lazy()` / dynamic `import()`, or raise `build.chunkSizeWarningLimit`
only after measuring. Plotly is already a local UMD script outside the Rollup
graph.

---

## 8. Release gate

Add `.github/workflows/public-release.yml` (created with this checklist) that
runs on every push / PR:

- backend `pytest -q` and `pytest -q --cov=app/ts --cov-fail-under=80`;
- frontend `npm run typecheck`, `npm test`, `npm run build`;
- static deploy tests `pytest -q tests/test_part19_deploy.py`;
- a Docker build + `curl` health smoke on a runner that has Docker.

Do not tag or deploy until all jobs pass. Keep `AUDIT.md`, `METHODS.md`,
`DEPLOY.md`, `CHANGELOG.md` and this checklist in the release notes.

---

## 9. Go / no-go summary

| If you only want… | Do |
|---|---|
| Private local install | Nothing more; already validated. |
| Private URL just for yourself | Steps 0–3, plus a password / auth proxy. |
| Public demo, clearly labelled prototype | Steps 0–5 and 8; keep statistical caveats visible. |
| Public production forecasting service | All steps 0–8, plus independent validation and a security review. |

**Bottom line:** for your own use, install it now. If you later publish it, the
shortest safe path is the single-container image, a real Docker smoke test, a
non-placeholder API origin (or no Vercel split at all), explicit CORS/debug
settings, authentication, traceback suppression, non-root container, rate
limits, and a CI gate. Tell me the host you choose and I can apply the exact
config, code and workflow changes.
