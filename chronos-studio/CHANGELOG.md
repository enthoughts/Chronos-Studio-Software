# Changelog

All notable changes to Chronos Studio. The build was generated in twenty
ordered parts from the [CRAFT prompt](CRAFT_PROMPT.md); each line below is that
part's contribution. Versioning follows the prompt's part numbers rather than
semantic releases.

Format: **Added** / **Changed** / **Fixed** / **Verified** / **Docs**.

---

## [0.1.0] — standalone desktop package and workspace optimization

### Added

- Installable `chronos-studio-standalone_0.1.0_amd64.deb` for Linux Mint 22 / Ubuntu 24.04 x86_64, with a basename-based `.sha256` sidecar. It serves the built React UI and FastAPI API from one private local process and opens an app-mode browser window.
- `standalone/chronos_studio.py`, `standalone/launch.sh`, `standalone/requirements-runtime.txt`, `standalone/build-deb.py`, a desktop entry and an SVG icon.
- `STANDALONE.md` documents installation, first-run offline wheel setup, privacy, advanced launcher options and the deliberate size tradeoffs.
- Offline CPython 3.12 wheelhouse with FastAPI, Uvicorn, NumPy, pandas, SciPy, statsmodels, Plotly, OpenPyXL, python-docx, scikit-learn, `google-genai>=1.0,<2` and transitive dependencies.

### Changed

- The workspace was optimized to keep only the Chronos Studio source tree, the current v4 source archive, the dataset CSVs and `chronos_dataset_guides.html`. Removed the 188 verification screenshots, Playwright browser and pip/npm caches, stale v2/v3 archives, the duplicate root CRAFT prompt, a stray test image, generated `frontend/dist`, Python bytecode caches and the five reference PNGs in `uploads/`.
- The v4 source archive was rebuilt to include the standalone build inputs and refreshed docs; the current checksum is in `CHRONOS-local-install-v4.zip.sha256`.
- Snapshot-eligible usage fell from 44.895 MiB to 11.084 MiB before the standalone artifact was added; raw disk usage fell from 857.513 MiB to 91.325 MiB.
- The standalone package omits Playwright/bundled Chromium and a bundled Python interpreter to fit the workspace budget. PDF and 300-dpi PNG exports degrade with an explicit error; the system Python 3.12 creates a private venv from the bundled wheels on first launch.

### Fixed

- The standalone launcher no longer shuts down when Firefox is already running. A plain `firefox --new-window URL` exits immediately after forwarding the URL to the existing Firefox instance; the process watcher misread that as the app window closing. Firefox now uses a dedicated `-no-remote -profile` instance, the generic default-browser fallback is never watched, and short-lived launchers inside a five-second grace period no longer stop the server.
- The Assistant's first-use payload preview no longer stays on “Checking live AI availability…”. Capability config is now fetched when the drawer opens rather than waiting for an analysis session, config failures show a definitive offline-safe message, and **Offline only** mode never auto-opens the live-AI preview. Three frontend regression tests cover the gating, key-header-only config request and definitive status message.
- **S12 `Run diagnostics` no longer returns `internal.error` on the standalone package.** The offline wheelhouse pins SciPy 1.16.3, whose `stats.anderson()` does not accept the `method="interpolate"` keyword introduced in SciPy 1.17. `anderson_darling_test()` now tries the 1.17 call and falls back to the older signature on `TypeError`; both result shapes are still read correctly. Reinstall the rebuilt `.deb` or replace the v4 source tree to pick up the fix.
- **S11/S12 no longer leave the stepper on “Not started” after a successful run.** Both stage-marking POSTs now reload their stored report, refresh the pipeline state and then show the success toast. A pure frontend regression test pins that order.
- Reinstalling the standalone package now rebuilds the private venv when the bundled wheel archive or runtime requirements change, using a stamp in `~/.local/share/chronos-studio/venv`. The check stays completely offline and prevents a reused 0.1.0 venv from masking updated dependencies.
- Session-expired error help, the `.chronos` glossary entry and `STANDALONE.md` no longer promise session restore. This build exports `.chronos` as a report/reproducibility artifact only; replaying an analysis requires re-uploading the data.
- **Catalogue and report status no longer 500 when Theta has no residual battery.** `residual_acf()` and `cusum_paths()` now treat `battery: None` as an empty battery, matching `residual_qq()` and the diagnostics tables. A full-path regression includes Theta and checks `/analysis/catalog` plus `/export/status`.
- The golden statsmodels guard accepts the offline CPython 3.12 wheelhouse’s 0.14.6 engine in addition to the recorded 0.15.0, because 0.15.0 has no cp312 wheel and every frozen numeric value still matches. Optional publication-export tests now skip with a clear reason when Kaleido, Playwright or local Chromium is absent instead of failing the standalone suite.
- **The documented `ice_cream.csv` S7 seasonal-difference → S3 state-space-imputation path now reaches S14.** `protocols._check_records()` treated a whole-series difference as unreplayable, so `/analysis/evaluate` returned `evaluate.preprocessing_unsupported` even after S8 was `READY WITH CAVEATS`. S14 now scores the untouched original series and re-decides differencing and imputation inside each training fold; trimming, aggregation, smoothing and row deletion remain refused. Six regression tests pin both the accepted guide chain and the still-unsupported operations.
- **The one-file HTML report and vector SVG figures no longer disappear without Chromium.** A pure-Python, offline SVG renderer draws scatter, filled confidence bands, bars, boxes, histograms, heatmaps, grid subplots, shapes and annotations from the Plotly JSON. Every one of the 49 catalogue figures is tested as valid self-contained SVG, and the all-items HTML scope embeds all of them. PNG/PDF still degrade with explicit `catalog.kaleido_missing` / `export.chromium_missing` errors and `errors.txt`; the HTML can be printed to PDF from the browser.
- Browser discovery no longer hard-codes a developer-sandbox Playwright cache path. `app/core/browser.py` checks `CHRONOS_CHROME`, `CHROME_PATH`, the current user's Playwright cache, `PLAYWRIGHT_BROWSERS_PATH` and system Chromium paths.

### Verified

- Production frontend build, static/API single-process routing, `.deb` control metadata, archive contents, checksum, offline wheel set and HTML guide were validated during packaging. A simulated short-lived Firefox launcher now leaves the server alive, while a dedicated browser exiting after the grace period still stops the server. The `.deb` is approximately 109 MB; the exact size and SHA-256 are in its `.sha256` sidecar.
- A direct compatibility check exercised both the SciPy 1.17 result shape and a simulated SciPy 1.16 signature, then ran the full residual battery; `tests/test_part13_selection_diagnostics.py` passes 65/65, including the new regression test. The full backend suite was rerun in a clean CPython 3.12 venv built from the exact offline wheelhouse; the Theta catalogue path, all 49 offline SVG figures, the full `ice_cream.csv` S0–S15 guide replay, frontend typecheck/tests/build and standalone boot were exercised. The fixed `.deb` and v4 source archive were rebuilt with refreshed SHA-256 sidecars.
- No API key, `.env`, `.tsbuildinfo`, `node_modules`, `.venv`, cache or bytecode directory is shipped in either deliverable. The source archive omits generated `frontend/dist`; the standalone package ships only the prebuilt `frontend/dist` assets needed to serve the UI offline.
- Live Gemini remains optional and user-keyed; offline analysis and the offline Assistant need neither a key nor internet access.

---

## [0.1.0] — Live AI Assistant fix and v4 local-install refresh

### Fixed

- Live Gemini was reported unavailable because the optional `google-genai` SDK was absent from the running backend. The compatible 1.x SDK is now installed in the workspace and pinned as `>=1.0,<2` in `requirements-deploy.txt`, `requirements.txt` and the `ai` extra.
- `app/ai/client.py::sdk_available()` no longer caches a permanent `False`; installing the SDK at runtime is detected on the next capability request.
- The Assistant drawer now distinguishes a missing SDK from a missing key, shows the exact repair command, and exposes **Open settings** from the first-use payload preview. The confirm button stays disabled only for the real reason, instead of always saying “enter a key”.
- The running backend was restarted with `GEMINI_API_KEY` unset, so no previously inherited key is used implicitly. No key value was read, copied, stored or sent to an external service.

### Added

- Clean `CHRONOS-local-install-v4.zip` with a basename-based `.sha256` sidecar, including all earlier v3 fixes plus the Live AI source, documentation and regression-test changes.
- Four new regression tests: dynamic SDK detection, live-mode resolution, non-secret config response and frontend live-availability diagnosis.

### Verified

- `npm run typecheck`, `npm test -- --run` (15 files / 60 tests) and `npm run build` pass; current bundle is `index-u8iH4vMh.js` (508.20 kB / 144.07 kB gzip, unchanged chunk warning).
- `pytest -q tests/test_part17_ai.py` passes with the new backend tests.
- Restarted backend health reports `ai_configured: false`, `sdk_available: true`, `live_ready: false` with no key, and `live_ready: true` with a dummy header key; no key value is echoed.
- API preview and offline chat stream return 200 with no key in the payload or response. Playwright Chromium confirms the first-use preview, explicit Live AI status, working Settings link, enabled confirm button with a dummy key, offline answer rendering and zero external AI requests.
- `CHRONOS-local-install-v3.zip` is retained as a clean-but-stale historical snapshot; use v4 for a fresh install.

---

## [0.1.0] — v3 local-install refresh

### Added

- Clean `CHRONOS-local-install-v3.zip` with a basename-based `.sha256` sidecar. It includes the current frontend source, the standalone dataset guide, the five attached CSVs under `datasets/`, regression tests and updated install/audit documentation.
- `frontend/src/state/stageMemory.ts` persists the selected stage per session, dataset content hash and series version. A reload now returns to S2 or S6 rather than silently adopting the backend's last-completed `current_stage`.
- `frontend/src/panels/nextAction.ts` makes the S7 Apply policy explicit and testable; six new frontend tests cover stage memory and the resettable `worse` acknowledgement.

### Fixed

- The S7 override now resets whenever the recommendation or preview changes, so two different `worse` cards cannot share one ticked checkbox. The locked-Apply message is linked to the checkbox with `aria-describedby`.
- `refresh()` now calls `activeStageAfterRefresh()` as an explicit, unit-tested guard instead of relying only on a comment to prevent background stage theft.

### Verified

- `npm run typecheck`, `npm test -- --run` (15 files / 59 tests) and `npm run build` pass; current bundle is `index-D6qCImZg.js` (506.82 kB / 143.71 kB gzip, unchanged chunk warning). `tsconfig.tsbuildinfo` is not shipped.
- Live Chromium checks: first reload after log-apply opens S3; selecting S2 and reloading stays S2; after the backend reaches S7, selecting S6 and reloading stays S6; wheel scrolling does not change either stage. The S7 Apply button is disabled with `worse`, unlocks after the checkbox, and commits the lag-12 difference.
- `CHRONOS-local-install-v2.zip` is retained only as a historical clean-but-stale snapshot; v3 is likewise superseded by v4.

---

## [0.1.0] — S2/S6 stage-jump and locked-Apply fixes

### Fixed

- A background `refresh()` in `frontend/src/state/store.tsx` overwrote the user-selected stage with the backend's `pipeline.current_stage`. Opening S6 called `GET /analysis/correlate` and `GET /analysis/next`; the latter marks S7 complete, so the stepper jumped to S7 even when the user had only opened or scrolled S6. The same mechanism could move S2 to S3 after S3 completed. `refresh()` now leaves `activeStage` alone; bootstrap and `startSession` still restore `current_stage` on an intentional reload.
- S7's ranked actions arrive with server-side previews. A `worse` verdict therefore disabled Apply immediately and left no way to accept the recommendation after reading it. `frontend/src/panels/CorrelatePanel.tsx` now requires an explicit <em>I read the preview — apply anyway</em> acknowledgement before unlocking Apply; the acknowledgement resets whenever the verdict is no longer `worse`.

### Verified

- Playwright Chromium against the running local app: after the fix, S6 stays active on open and after a wheel scroll; S2 stays active after being selected from a reloaded S3 and after scrolling. A reload still intentionally opens the backend's `current_stage` (S3 in the log-apply test).
- Live-UI S7 override: Apply is disabled with the `worse` preview, becomes enabled after ticking the checkbox, and clicking it adds the lag-12 `difference` treatment to S3.
- `npm run typecheck`, `npm test -- --run` (13 files / 53 tests) and `npm run build` pass; current bundle is `index-Cr7tv0b1.js` (504.72 kB, unchanged chunk-size warning). A TestClient S0→S8 run on `ice_cream.csv` with seed `20260101` verified that log → seasonal difference → Kalman imputation reaches S8 `ready_with_caveats` with no blockers.

### Docs

- `chronos_dataset_guides.html` now labels raw version-1 versus after-log version-2 values for `ice_cream.csv`, reconciles the S5/S7 screenshot values, documents the new override and the S13 transformed-plus-differenced refusal, and replaces the obsolete “Apply before preview” workaround.
- `CHRONOS-local-install-v2.zip` predates both this entry and the S1 fix; the v3 local-install refresh supersedes it. Do not use v2 as a fresh install source.

---

## [0.1.0] — S1 jump-bar fix

### Fixed

- Step 1's <em>Jump to</em> buttons silently failed for most results: the collapsed view only mounted three inline figures, the first two tables and the first two stats, while the jump bar addressed every result ID. Clicking a hidden target called `getElementById` on a card that was not in the DOM, so nothing scrolled or highlighted.
- The jump bar now expands the full result list before scrolling to a hidden card, highlights the three inline figures through their `quick-` anchors, and also lists non-figure/table/stat results. It carries the hint <em>opens the full list when needed</em>.
- <em>Outliers and structural changes</em> (`fig.outliers`) and <em>Flagged observations</em> (`table.outliers`) are reachable from the jump bar again.

### Verified

- `npm run typecheck`, `npm test -- --run` (13 files / 53 tests) and `npm run build` all pass. The 504 kB JS chunk warning is unchanged.

### Note

- Frontend source fix only. The previously published `CHRONOS-local-install-v2.zip` predates it; v3 includes this fix. Do not use v2 as a fresh install source.

---

## [0.1.0] — Local-install packaging correction · 2026-11-10

### Fixed

- The first `CHRONOS-local-install.zip` accidentally bundled the gitignored
  `backend/.env` containing a Gemini key value. Removed that file from the
  workspace, excluded `.env` and `*.tsbuildinfo` from archives, and published
  `CHRONOS-local-install-v2.zip` with a basename-based `.sha256` checksum.
- Corrected the Linux Mint extraction path and checksum commands in
  `INSTALL_LOCAL.md`; added a nested-`CHRONOS` repair recipe.

### Added

- `INSTALL_LOCAL.md`, `scripts/install-local-linux-mint.sh` and
  `scripts/run-local.sh` for a private Python-venv + Node local install.

### Verified

- The v2 archive contains no `.env`, no API key value and no generated
  TypeScript build cache; zip integrity and executable script bits were checked.

---

## [0.1.0] — Part 15 critical audit · 2026-11-08 · *verified*

A post-release audit of the evaluation engine, report contract and Part 19 UI.
Full findings, residual limitations and the install recommendation are in
[`AUDIT.md`](AUDIT.md).

### Fixed

- **Part 15 statistical defects**: exact irregular-grid / out-of-support
  discrete CRPS; refusal of out-of-support mean log scores; finite-draw PIT/RQR
  with one seeded inferential draw instead of Fisher-combining dependent
  repetitions; per-row fold scaling for MASE/RMSSE; full-length NaN-padded
  interval scores; canonical one-sided Clark–West; Holm-adjusted DM p-values;
  tie exclusions; valid stationary-bootstrap MCS p; higher-is-better ranking;
  calibration scored by deviation from nominal.
- **Silent row drops**: `point_metrics` pads shorter vectors and counts every
  exclusion; the API verdict now reconciles finite + excluded rows with the
  declared fold count instead of asserting “no silent row drops”.
- **Leakage audit**: one pre-initialized row per `(model, fold)`, mandatory
  `expected_fits`, recorded factory/fit failures, recomputed future-driver
  frames, training-only transform parameters and complete decision digests.
- **Honest reporting**: `include_baselines=false`, honest primary/leader wording,
  no prose-derived primary model, Holm columns/footnotes/captions, canonical
  Clark–West note, one-draw RQR disclosure, `interval_construction`, and a
  calibration chart that refuses to invent a missing nominal level.
- **Part 19 top bar**: right-edge clipping replaced by a responsive overflow
  menu with compact-width copies of beginner controls and session information.
  `EvaluationPanel` now rejects non-integer simulation path counts.

### Verified

- Backend compile clean; **1014 backend tests pass**; scoped `app/ts` coverage
  **91.61%** (13,925 statements, 1,168 missing).
- Frontend typecheck 0, **53 tests pass**, production build 0; Vite reports a
  504.04 kB / 142.74 kB gzip chunk advisory.
- Live health/demo/proxy 200; P19 UI shots 198–213 regenerate with no
  application JS exceptions; fresh top-bar shots 214–215; no horizontal
  overflow at 1560, 1280 or 1024 px.
- Real API verdict probe: `ready`, 7/7 leakage checks, 18/18 finite rows per
  model, no blockers.

### Known limitations

- No Docker daemon in the sandbox: deploy configs are statically verified, not
  container-smoke-tested. `frontend/vercel.json` still has a placeholder API
  host. Optional `weasyprint`, `arch`, and several model libraries are absent;
  PDF/300-dpi PNG exports need Chromium. Short-series overlap, Gaussian VaR and
  squared-return variance proxies remain disclosed caveats. No production
  security/load/persistence audit or independent peer review has been done.

---

## [0.1.0] — Part 19 · Hardening, Docs & Deploy · *ship*

The final part. Everything below is new in this release.

### Added

- **Golden-file fixtures per model family.** `tests/fixtures/golden_models.json`
  (13 cases, one per implemented family) plus a reproducible generator
  `scripts/generate_golden.py` and shared case definitions
  `tests/golden_common.py`.
- **statsmodels / R-comparison reference** for AirPassengers:
  `tests/fixtures/airpassengers_reference.json` with ADF, KPSS (both `auto` and
  `legacy` bandwidths), `SARIMAX(1,1,1)(1,1,1,12)` and additive-trend /
  multiplicative-seasonal ETS, plus frozen app ADF/KPSS outputs.
- **Property tests**: MASE(perfect) = 0, MASE(naive) = 1, a 25-seed sweep, and
  training-only scale behaviour.
- **Advisor regression tests** for the enforced safeguards: no log/Box–Cox on
  non-positive data, λ-interval-contains-1 refusal, "do not difference an
  already stationary series", and the S7 white-noise → leave-alone /
  unit-root → first-difference regressions with deterministic ranking.
- **Demo mode and seed studies** (`app/api/demo.py`): three narrated scenarios,
  each a 17-step replay of the real S0→S15 endpoints with progress, per-step
  timings, cooperative **cancel**, job deletion and a one-hour TTL sweep.
  All three reach S15 complete with a 100/100 readiness verdict.
- **Undo/redo** for preprocessing: `POST /preprocess/redo`, `can_redo` in
  `/preprocess/state`, and branch-clearing on a fresh apply.
- **Accessibility**: keyboard-navigable ribbon (roving tabindex + Arrow/Home/End
  per WAI-ARIA toolbar), a 242-term glossary with hover/focus tooltips and a
  searchable glossary dialog, "Simplify view", plain-language mode, guided demo
  tour, and screen-reader figure/table value summaries.
- **Glossary everywhere**: `GlossaryText` auto-links arbitrary prose and
  `GlossarySelection` defines a phrase selected anywhere in the app, including
  table headers and backend-generated prose.
- **Per-panel "What is this?" drawer** (`copy/beginner.ts`,
  `components/HelpDrawer.tsx`): stage-specific reading, try-this, common
  misunderstanding and glossary chips, reachable from every results header
  and the top bar.
- **Plain-English error library** (`copy/errors.ts`,
  `components/ErrorLibraryPanel.tsx`): 43 searchable articles plus prefix
  families; an error toast links to its own code, meaning, fix and "avoid".
- **Undo/redo UI and stepper back navigation**: `HistoryControls` in the top
  bar / options pane / applied-step strip with Ctrl/Cmd+Z, Shift+Z and Y;
  explicit Back/Next controls and Arrow/Home/End stage navigation.
- **Shared robust STL fit** (`app/ts/eda/stl_cache.py`) and one merged Plotly
  `update_layout` per figure, with exactness / cache-hit regression tests.
- **Deploy configuration**: `backend/requirements-deploy.txt`,
  `backend/.dockerignore`, `frontend/.dockerignore`, a real
  `frontend/nginx.conf`, `Dockerfile.single` for one-container free tiers,
  `scripts/single-container/{entrypoint.sh,nginx.conf.template}` and
  `frontend/vercel.json`.
- **Docs**: `METHODS.md`, `GLOSSARY.md` (242 terms), `CONTRIBUTING.md`,
  `CHANGELOG.md`, `DEPLOY.md`.

### Changed

- **Performance.** Mann–Kendall S is now an O(n log n) Fenwick-tree computation
  above 2,000 points (mathematically identical); Theil–Sen uses a deterministic
  shape-preserving 1,000-point grid above 1,500 points; `_tail_outliers` uses
  one vectorised `norm.isf` instead of 10,000 scalar scipy calls; the EDA
  theme applies layout in one merged Plotly call; and the weekly robust STL fit
  is shared between seasonality and outlier detection through an exact
  eight-entry LRU cache. Small-n results are bit-for-bit unchanged. 10k-row
  paste + profile + STL decompose: **1.73–2.27 s including ingestion** on the
  dev sandbox, down from a >300 s timeout; the normal suite now asserts the
  strict §7 `<3 s` budget.
- `docker-compose.yml` gained healthchecks, `depends_on: service_healthy`,
  `restart: unless-stopped`, `init: true`, a bounded `tmpfs` for uploads and an
  `INSTALL_BROWSER` build arg. The frontend nginx config now sets a 25 MB
  upload limit, gzip and 300 s proxy timeouts.
- `requirements.txt` now documents the canonical `pyproject.toml` extras and
  points containers at `requirements-deploy.txt`, resolving the core/extras
  mismatch.
- `.env.example` gained `INSTALL_BROWSER`, `CHRONOS_CHROME` and a Docker section.
- `GLOSSARY_STARTER` (9 terms, previously unused) is superseded by the
  generated 242-term glossary.

### Fixed

- The 10k-row S4 hang caused by an O(n²) Theil–Sen list of ~50 million Python
  floats (~400 MB).
- Demo replay ordering: S6 → S7 → S8 (running S7 after the gate marked S8
  stale), and `n_paths` raised to the enforced `MIN_PATHS = 2000` floor.
- `frontend/Dockerfile` now uses a checked-in nginx config instead of a
  shell-`printf` heredoc.
- The 10k-row strict performance test is coverage-aware: a Python line tracer
  gets a 12 s CI ceiling, while the normal interactive suite enforces `<3 s`.
  This keeps both the performance gate and the coverage gate meaningful.

### Verified

- **990 / 990 backend tests** in 609 s; coverage-gated run also 990 / 990 in
  936 s at **91.71%** on `app/ts` (12,584 / 13,722 statements, 82 files),
  above the configured 80% gate. 41 golden / property / advisor / performance
  tests, 12 demo / seed / redo / cancel / 10k tests, 31 static deploy tests.
- **52 frontend tests** across 12 files, clean `npx tsc --noEmit`, clean
  production build (73 modules, 31.31 kB CSS, 502.40 kB JS).
- **Live API + Playwright smoke**: shots 198–213 cover glossary search (242
  rows), tooltips and selection lookup, Simplify view (24 → 8 commands),
  plain-language mode, keyboard ribbon, guided tour, 17/17 demo completion,
  per-panel help, error library and toast deep link, Back/Next navigation and
  real undo/redo. No uncaught application errors; no `NaN`, `Infinity` or
  `undefined` strings. The deliberately expired session produced expected HTTP
  400/404 network logs only.
- No Docker daemon is available in the build sandbox: `docker compose up` is
  verified by `tests/test_part19_deploy.py` and the commands in `DEPLOY.md`,
  not by a live container run. Stated in `METHODS.md` §13.

---

## [0–18] — the preceding build parts

| Part | Content |
|---|---|
| 0 | Scaffold & contracts: repo tree, `ResultBase`/`PipelineState`/`AdvisorReport`, `SessionStore`, `ChronosError`, FastAPI health + CORS, React/Vite ribbon/options/results shell, Dockerfiles, `.env.example`, README stub, pytest + vitest setup. |
| 1 | Ingestion: CSV/Excel/JSON/Parquet/Stata/SPSS readers, paste, column mapping, frequency inference, data-class detection, example datasets. |
| 2 | EDA: descriptives, normality battery, runs test, rolling-SD stability, seasonal profile, outlier detection, the full S1 figure set with alt text. |
| 3 | Preprocessing advisor & engine: impute/smooth/transform/outlier ops, preview-before-commit, S2/S3 panels with accept/reject/undo. |
| 4 | Decomposition: classical, STL, MSTL, UCM, HP/CF, X-13 adapter; `F_T`/`F_S`, Mann–Kendall, Sen, Cox–Stuart, periodogram/Fisher g, ARCH-LM on the remainder; component scorecard. |
| 5 | Stationarity & structural tests: ADF/PP/KPSS/DF-GLS/Zivot–Andrews/Canova–Hansen/OCSB/HEGY, Chow/Quandt–Andrews/Bai–Perron/CUSUM, 2×2 reconciliation, `AdvisorReport`, S5 verdict matrix. |
| 6 | ACF/PACF/CCF with Bartlett bands, pattern classifier, Ljung–Box, next-step advisor with preview re-runs, S6/S7 UI. |
| 7 | Readiness gate, checklist/blockers/caveats, caveat propagation. |
| 8 | Model registry + `ModelWrapper`, baselines, ARMA/ARIMA/SARIMA, ARFIMA, bounded auto-ARIMA fallback with root checks. |
| 9 | ETS/Holt–Winters grid, Theta, UCM/structural, SARIMAX/ARIMAX with intervention dummies and exogenous scenarios, ARDL. |
| 10 | Count & intermittent: Poisson with mandatory overdispersion pre-check, NB2, zero-inflated/hurdle, INAR/INGARCH, Croston/SBA/TSB/ADIDA, renewal/Hawkes, discrete predictive distributions. |
| 11 | Volatility: ARCH/GARCH/EGARCH/GJR/GARCH-t, ARIMA–GARCH, variance path, leverage reporting, non-convergence fallbacks. |
| 12 | VAR/VARX + Granger/IRF/FEVD, VECM/Johansen, SETAR, Markov-switching, NNAR, GP, RF/LightGBM/XGB/k-NN/SVR with leakage-safe features + model card, forecast combination. |
| 13 | Selection & diagnostics: IC comparability guard, rolling-origin CV, DM/Clark–West/MZ, residual battery, QQ with bands, PIT/RQR, adequacy verdict, S11/S12 UI. |
| 14 | Forecasting: analytic + simulation intervals, discrete prediction intervals, bias-adjusted back-transformation, horizon advisor, scenarios, fan charts, S13 UI. |
| 15 | Evaluation metrics: point/relative/probabilistic scores, PIT/RQR uniformity, DM matrix, metric-selection policy engine, S14 UI and plots. |
| 16 | Plots & tables completeness: audit against §5.10, per-family plots, captions, PNG/SVG/PDF/XLSX exports, single-source styling, report basket, 300-dpi figures. |
| 17 | AI assistant: Gemini client, `AnalysisDigest`, grounded prompts, streaming chat, number-grounding validator, offline narrative mode, rate limits, privacy controls + payload preview. |
| 18 | Reports & finalization: HTML/DOCX/PDF/LaTeX/XLSX builders, manuscript prose, `.chronos` archive, reproducible Python/R scripts, ten-check weighted scorecard, exact verdicts, S15 UI, one-file HTML. |

### Part 18 hardening (post-ship)

- Manuscript facts now use catalogue rows with panel-named fallbacks (no
  `[PLACEHOLDER`); finalization leaves travelling caveats unchanged.
- Readiness uses **stored results**, not completed-stage status alone;
  re-running an upstream stage marks S14/S15 stale.
- Catalogue cache is fingerprinted (sha256 over result ids, series version,
  timestamps, stage statuses); genuine catalogue errors are no longer swallowed
  into empty "ready" reports.
- Fixed a live `RuntimeError: dictionary changed size during iteration` during
  concurrent S15 finalization by snapshotting outer/inner result mappings in
  `charts/completeness.py::_result_rows` and `core/session.py`.

---

## Known deviations from the CRAFT prompt

Documented in [METHODS.md](METHODS.md) §13. In short: `arch` is absent
(statsmodels fallback used), PDF/300-dpi PNG need Chromium, several advanced
families (X-13 binary, Prophet, TBATS, LSTM/TFT, stochastic volatility, panel)
are **flagged** in the registry rather than implemented, and the large-n trend
statistics use documented bounded algorithms above 1,500/2,000 observations.
`docker compose up` is verified statically because the build sandbox has no
Docker daemon.
