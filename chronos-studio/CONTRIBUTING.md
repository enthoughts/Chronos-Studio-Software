# Contributing to Chronos Studio

> Read [METHODS.md](METHODS.md) before changing a statistic. Read
> [GLOSSARY.md](GLOSSARY.md) before inventing a new term. The build contract is
> [`CRAFT_PROMPT.md`](CRAFT_PROMPT.md); Parts 0–19 are all shipped.

---

## 1. Repository map

```
chronos-studio/
├── backend/
│   ├── app/
│   │   ├── core/        schemas, SessionStore, settings, errors, formatting
│   │   ├── ts/          ALL statistics: io, eda, preprocess, decompose,
│   │   │               tests, correlate, advisor, models, select, diagnose,
│   │   │               forecast, metrics, report, charts
│   │   ├── advisor/     rule engines (prep / stat / next / decompose)
│   │   ├── ai/          digest, prompts, grounding validator, offline mode
│   │   ├── charts/      single-source theme + plot/table catalogue + exporters
│   │   └── api/         thin FastAPI routers (no statistics live here)
│   ├── tests/           pytest suite + tests/fixtures/ (golden files)
│   ├── scripts/         gen_examples.py, generate_golden.py
│   ├── app/ts/eda/      stl_cache.py (shared exact robust STL fit)
│   ├── pyproject.toml   canonical dependency + coverage config
│   ├── requirements.txt / requirements-deploy.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── copy/        ALL user-facing strings, generated glossary.ts,
│   │   │                beginner.ts (panel help), errors.ts (error library)
│   │   ├── components/  Ribbon, Stepper, TopBar, ResultCard, PlotlyFigure,
│   │   │                HelpDrawer, ErrorLibraryPanel, HistoryControls,
│   │   │                GlossaryText / GlossarySelection, DemoPanel…
│   │   ├── panels/      one panel per stage
│   │   ├── state/       store.tsx, api.ts, types.ts, ai.ts
│   │   ├── charts/      PlotlyFigure.tsx (the only Plotly mount point)
│   │   └── styles/      theme.css, app.css
│   ├── nginx.conf, vercel.json, Dockerfile
│   └── package.json
├── scripts/
│   ├── generate_glossary.py      → GLOSSARY.md + frontend/src/copy/glossary.ts
│   └── single-container/         nginx template + entrypoint for free tiers
├── docker-compose.yml, Dockerfile.single, .env.example
├── README.md, METHODS.md, GLOSSARY.md, CHANGELOG.md, DEPLOY.md
└── CRAFT_PROMPT.md
```

**Rule:** the API layer never computes a statistic, and the UI never computes a
statistic. `ts/` owns the numbers; `api/` adapts them; the frontend only reads
them. Everything user-facing lives in `frontend/src/copy/`.

---

## 2. Set up a dev environment

```bash
# backend
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev,report,ml,ai]"     # optional extras, all feature-flagged
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# frontend (second terminal)
cd frontend
npm ci
npm run dev                             # http://localhost:5173, proxies /api
```

Or use `scripts/dev.sh`, or the two containers:

```bash
cp .env.example .env
docker compose up --build               # http://localhost:8080
```

The app runs with **no** `.env` and **no** API key. Leave `GEMINI_API_KEY`
blank for the offline assistant.

---

## 3. Tests

```bash
# backend — full suite (990 tests)
cd backend && PYTHONPATH=. pytest -q

# backend — coverage (§5.15 requires ≥80% on ts/; pyproject sets fail_under=80)
PYTHONPATH=. pytest -q --cov=app/ts --cov-fail-under=80 \
                       --cov-report=term-missing --cov-report=html
# → open backend/htmlcov/index.html; current result: 91.71% on 82 ts/ files
# The 10k performance test relaxes its <3 s budget only when a coverage tracer
# is active; the normal `pytest -q` run enforces the strict interactive budget.

# backend — just the Part 19 quality gates
PYTHONPATH=. pytest -q tests/test_part19_golden.py tests/test_part19_demo.py \
                       tests/test_part19_deploy.py
PYTHONPATH=. pytest -q -m golden            # golden-file comparisons only

# frontend
cd frontend
npm run typecheck
npm test                                   # vitest (12 files / 52 tests)
npm run build                              # tsc -b && vite build
```

Live smoke scripts used during the build live in `/tmp` (they are deliberately
not checked in): they drive `/api/v1/...` end-to-end and take Playwright
screenshots. Recreate them from the endpoint list in the README.

### Test conventions

* Use `TestClient(app, raise_server_exceptions=False)` when asserting on an
  intentionally injected HTTP 500; the default re-raises.
* A full staged run must call `GET /analysis/correlate` before `POST
  /analysis/gate`, and `POST /analysis/suggest` before `GET /analysis/suggest`.
  `STAGE_DEPENDENCIES` in `app/core/session.py` is the source of truth.
* `session_dep` needs an `X-Session-Id` header; `/sessions/{id}/results`
  returns **stage → list**, not a flat list.
* Errors serialise as `{"error": {"code","plain","fix","technical","context"}}`.
* `ResultBase` forbids extra fields and requires `plain`, `why`, `method`;
  `reproducible_snippet` must `ast.parse`.
* Never restore broad exception swallowing in `report/bundle.py::_public_catalog`
  or `scorecard.py::_all_ids`: a genuine catalogue failure must be a visible
  500, not an empty "ready" report.

---

## 4. Regenerating generated files

| Generated file | Generator | When |
|---|---|---|
| `GLOSSARY.md`, `frontend/src/copy/glossary.ts` | `python scripts/generate_glossary.py` | after editing any term in that script |
| `backend/tests/fixtures/golden_models.json` | `cd backend && PYTHONPATH=. python scripts/generate_golden.py` | only after **reviewing** a statistical change |
| `backend/tests/fixtures/airpassengers_reference.json` | statsmodels reference script (see METHODS.md §10.1) | only when upgrading statsmodels, with reviewer sign-off |
| `backend/app/assets/examples/*` | `cd backend && python scripts/gen_examples.py` | when adding a seed dataset (also update `manifest.json`) |

```
⚠️  A changed AIC is a statistical regression, not a fixture refresh.
    Regenerating a golden file to make a test pass without explaining WHY the
    number moved is the one thing this project cannot accept.
```

The glossary generator asserts ≥120 terms (currently 242) and unique slugs, and
writes both outputs from one `TERMS` list so the docs and the in-app tooltips
cannot disagree. **Edit `scripts/generate_glossary.py`, never the outputs.**

---

## 5. Adding a model (one file + one registry row)

1. Add a `ModelWrapper` subclass in `app/ts/models/<family>/`.
2. Add one `ModelSpec` row in `app/ts/models/registry.py` with `data_class`,
   assumptions, minimum sample and the `fit` key. `fit_many()` picks it up.
3. `ModelWrapper.fit()` must **never raise** a modelling failure — put the
   reason in `FitOutcome.summary.failed` / `failure_reason`.
4. Baselines stay mandatory. Set `ic_comparable=False` unless differencing and
   the effective sample really match the comparison set.
5. Add a golden case in `backend/tests/golden_common.py::CASES` (one per family),
   regenerate, and add a targeted test in `tests/test_part<N>_*.py`.
6. If the model is ML, add a model card and benchmark it against ETS/ARIMA.

---

## 6. Adding an advisor rule

Rules live in `app/advisor/rules_*.py`. A rule has `when(ctx) -> bool`,
`because(ctx) -> str`, a `kind` (`recommend | blocker | protect | note`),
`confidence`, `provenance` (which computed statistic it uses), an `alternative`,
`risk_if_ignored` and `how_to`. Ranking is deterministic (blockers first, then
confidence, then declaration/tie-break id).

* A rule must be traceable to a **displayed statistic** — no magic constants
  presented as evidence; if a threshold is used, name it in `because`.
* Safeguards (`§6`) are `protect` / `blocker` rules, not notes. Add a regression
  test in `tests/test_part19_golden.py` for any new safeguard.
* Never difference a series two tests call stationary; never log/Box–Cox
  non-positive values; never compute a relative metric without the naive
  benchmark.

---

## 7. Code style and accessibility

```bash
cd backend && ruff check app tests         # line-length 100, target py311
cd frontend && npm run typecheck          # strict TS, noUnusedLocals
```

* **Colours:** every colour literal lives in `app/charts/theme.py` (backend) or
  `src/styles/theme.css` (frontend). `tests/test_part16_catalog.py` fails on
  hex literals elsewhere. Palettes must be colourblind-safe and never encode
  meaning by colour alone — add words/icons/shapes.
* **Figures/tables:** every figure needs `role="img"` + an `aria-label` that
  summarises the **values**; every table needs a `<caption>` and an `.sr-only`
  value summary. Reduced motion and `:focus-visible` are already global — do
  not remove them.
* **Keyboard:** the ribbon is a WAI-ARIA toolbar (one tab stop, Arrow/Home/End).
  New toolbar-like widgets must follow the same pattern.
* **Strings:** add user-facing text to `frontend/src/copy/strings.ts`, never
  inline in JSX. New jargon → glossary generator.
* **Browser URLs:** always relative `/api/...`. Never hardcode `localhost` or a
  host in browser code (`tests/test_part19_deploy.py` checks this).
* **Headers:** HTTP header values must be latin-1 safe. Exact em-dash verdicts
  belong in JSON/documents, not headers.

---

## 8. Pull-request checklist

- [ ] `PYTHONPATH=. pytest -q` passes (no skips introduced to hide a failure).
- [ ] Coverage on `app/ts` still ≥80%.
- [ ] `npm run typecheck && npm test && npm run build` pass.
- [ ] A new statistic has a golden or property test and a METHODS.md paragraph.
- [ ] A new term is in `scripts/generate_glossary.py` and both outputs regenerated.
- [ ] CHANGELOG.md updated.
- [ ] No generated fixture was refreshed without an explanation of the change.
- [ ] `docker compose config -q` and `pytest -q tests/test_part19_deploy.py` pass.
- [ ] Screenshots in the README still match the UI (or new ones added).

---

## 9. CRAFT keyword commands (honoured throughout the build)

`Generate Part X`, `Manifest`, `Status`, `Audit`, `Refactor <path>`,
`Explain <module>`, `Test <module>`, `UI <panel>`, `Doc`, `Deploy`, `Handoff`.
See `CRAFT_PROMPT.md` §4.5 for the exact response skeleton used for each part.
