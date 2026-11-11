# Chronos Studio — post-release critical audit

**Audit date:** 2026-11-08 (Asia/Manila)  
**Scope:** all critical errors and issues found in the active audit, with special attention to Part 15 / Step 14 evaluation. Earlier Part 19 completion is historical, not evidence for this audit.  
**Outcome:** critical defects found were fixed, regression-tested, and revalidated. Residual limitations are listed below; they are disclosed rather than hidden.

---

## 1. What was wrong and what changed

### 1.1 Part 15 statistical correctness

| Area | Defect found | Fix | Regression evidence |
|---|---|---|---|
| Discrete CRPS | Ignored irregular grid spacing and observations outside the stored PMF support; wrong tails. | Exact energy-identity CRPS `E|X−y| − ½E|X−X'|`, valid for irregular grids and out-of-support observations. | `test_discrete_crps_matches_energy_identity_for_irregular_and_outside_support` |
| Discrete log score | Out-of-support PMF probabilities were replaced by a finite edge penalty. | Refuse the mean log score and count out-of-support observations. | `test_out_of_support_pmf_refuses_the_mean_log_score` |
| Randomized PIT / RQR | Divided by total draws including NaNs; dependent randomization repetitions were Fisher-combined as if independent; discrete empty PMF produced a fake PIT. | Finite-draw denominators, one seeded inferential draw with extra draws reported only as stability, disclosed `[0,1]` PIT bounds for empty PMFs, asymptotic CvM survival. | `test_pit_and_rqr_use_one_inferential_draw_not_a_shrunk_mean`, `test_empirical_pit_divides_by_finite_draws_only`, `test_dunn_smyth_rqr_does_not_fisher_combine_dependent_repetitions` |
| Point MASE / RMSSE | Pooled errors were divided by a mean scale, which is wrong when folds have different scales. | Per-row fold scale alignment, explicit refusal for ambiguous scale maps, counted zero/undefined scales, MASE by horizon. | `test_pooled_mase_and_rmsse_use_each_folds_training_scale` |
| Short vectors | `_finite` truncated to the shortest actual/forecast/benchmark vector, silently losing rows. | Pad to the longest vector with NaN and count every excluded row. | `test_point_metrics_pads_short_vectors_and_counts_every_excluded_row` |
| Bands / coverage | Truncated interval arrays were scored as if complete; missing bands were counted as non-coverage. | Full-length NaN padding with `n_expected` / `n_excluded`; missing bands are excluded, not treated as violations. | `test_band_scores_pad_to_full_length_and_count_exclusions`, `test_score_fold_does_not_count_missing_bands_as_uncovered` |
| Clark–West / nested tests | Noncanonical variance and two-sided p; `benchmark_forecast` was ignored. | Canonical adjusted MSPE differential with the forecast-gap term and a one-sided p-value; explicit benchmark forecast path. Clark–West applies to genuinely nested comparisons. | `test_clark_west_is_canonical_one_sided_and_uses_forecast_gap`, `test_nested_tests_uses_benchmark_forecast_in_the_adjustment` |
| Diebold–Mariano / HAC | Zero or undefined HAC long-run variance could manufacture significance; no multiple-comparison adjustment. | Refuse an invalid LRV; report Holm-adjusted p-values for the family of unique pairs. Holm controls the family-wise error rate by a step-down rule. | `test_dm_refuses_a_non_positive_hac_variance_instead_of_manufacturing_significance`, `test_dm_matrix_reports_holm_adjusted_family_wise_p_values` |
| Directional accuracy / MCS | Ties were counted as “not up”; MCS bootstrap p was decorative and used a fixed seed. | Exclude and count ties; valid stationary-bootstrap MCS p with refusal below 30 rows. | `test_directional_ties_are_excluded_not_counted_as_down`, `test_mcs_bootstrap_p_is_a_valid_probability_and_not_decorative` |
| Metric ranking / calibration | Higher-is-better metrics were ranked as lower-is-better; calibration was scored by mean coverage rather than deviation from nominal. | `_HIGHER_BETTER` / `_ranked_models`; calibration score is mean absolute deviation from nominal. | `test_ranked_models_respects_higher_is_better`, `test_coverage_metric_is_deviation_from_nominal_not_mean_coverage` |

### 1.2 Protocol, leakage and honest reporting

- `leakage_audit` previously used a tautological poison check, a self-set exog boolean, and an `all(...)` over omitted failed fits. It now pre-initializes one row per `(model, fold)`, requires `expected_fits`, records failed factory/fit reasons, recomputes training-only decisions and transform parameters, and **recomputes** the future-driver frame instead of trusting `exog_held_flat`.
- Full-series Box–Cox λ reuse is refused; fold transform parameters are recomputed from each training window.
- Rolling-origin test blocks that overlap now produce an explicit warning: pooled rows repeat observations, HAC/Holm DM p-values are reported, and coverage/binomial/directional tests are descriptive under overlap.
- `include_baselines=false` is honored without losing the scaling benchmark; baseline IDs come from stored outcomes.
- The API report now derives `primary_model` from inputs/the winner rather than parsing the prose phrase “leads on”.
- Headline/advisor wording says the primary model “scores/ranks first”, not that it “leads on” a causal or inferential claim.
- Holm-adjusted DM p-values, canonical Clark–West notes, one-draw RQR disclosure, technical MASE/RMSSE/CRPS/log-score text, and `interval_construction` are propagated through API, charts and report prose.
- `_verdict` no longer unconditionally marks “out-of-sample” and “no silent row drops” as passed. A new `_integrity` helper reconciles each model’s finite + excluded rows with the declared fold count, verifies the training-only fit audit, checks the benchmark scale, and refuses `READY` when row accounting, leakage, benchmark, coverage construction or fold failures fail. Bad states produce `NOT_READY` / `INCONCLUSIVE` and explicit blockers.
- `calibration_curve` no longer assumes a non-null numeric `nominal`; missing or non-numeric values become unavailable points and are disclosed in alt text instead of raising or inventing coverage.

### 1.3 Frontend / Part 19 regression

- The top bar clipped secondary controls at the right edge in shot 207. Secondary actions now live in a `details.topbar-more` overflow menu. Compact widths also get reachable copies of Glossary / Simplify / Plain language and the session summary, not `display:none` dead ends.
- Verification: no horizontal overflow at 1560, 1280 or 1024 px; the menu stays inside the viewport; compact controls are visible at 1024 px; fresh shots `214_audit_topbar.png` and `215_audit_topbar_more.png` were captured.
- `EvaluationPanel` now rejects non-integer `nPaths`, with a frontend regression test.
- P19 screenshot scripts were rerun; the deliberate error-toast test still produces expected 400/404 network logs but no application JS exceptions.

---

## 2. Validation after the fixes

| Check | Result |
|---|---|
| Backend compile | `python -m compileall -q app` clean |
| Full backend suite | **1014 passed**, exit 0; final rerun after all code and documentation changes (`/tmp/audit_full_backend_final.log`) |
| Scoped statistical-layer coverage | `pytest -q --cov=app/ts --cov-fail-under=80` → **91.61%**, 13,925 statements, 1,168 missing (`/tmp/audit_full_backend_cov.log`) |
| Targeted P15 / P13 / P18 / P19 regressions | Exit 0 (`/tmp/p15_audit_targeted4.log`, `/tmp/p15_audit_targeted3.log`) |
| Frontend | `npm run typecheck` 0, `npm test` **53 passed**, `npm run build` 0; production JS 504.04 kB / gzip 142.74 kB with Vite’s >500 kB chunk warning |
| Live HTTP | `/api/v1/health`, `/api/v1/demo/scenarios` and the Vite proxy return 200 |
| UI smoke | P19 shots 198–213 regenerate cleanly; `DEMO_COMPLETE 17/17 · 100%`, `BODY_BAD 0 0 0`, `APP_ERRORS 0`; expected 400/404 console logs come from the deliberate expired-session test |
| Responsive top bar | 1560 / 1280 / 1024 px: `scrollWidth == clientWidth`, no child outside viewport, compact menu reachable |
| API verdict probe | Real evaluation returned `verdict=ready`, 7/7 leakage checks passed, 18/18 finite rows per model, no blockers (`/tmp/audit_verdict_probe.log`) |

> **Post-audit local-fix addendum (2026-11-10).** The v3 local-install refresh adds
> frontend stage-selection persistence, a resettable S7 Apply acknowledgement and
> six new frontend regression tests. Current frontend validation is
> `npm run typecheck` 0, `npm test -- --run` **15 files / 59 tests**, and
> `npm run build` 0; the production JS chunk is **506.82 kB / 143.71 kB gzip**
> with the same >500 kB advisory warning. No backend code changed, so the
> **1014 passed** backend result and the public-deployment caveats below remain
> unchanged.

> **Live AI addendum (2026-11-10).** v4 installs and pins
> `google-genai>=1.0,<2`, removes the cached SDK-capability flag, and gives the
> Assistant drawer an explicit SDK-versus-key diagnosis plus a Settings link
> from the first-use preview. Current frontend suite is **15 files / 60 tests**;
> `tests/test_part17_ai.py` passes with three new backend capability tests. The
> restarted local backend reports no inherited key, `sdk_available: true`, and
> `live_ready` only when a key header is supplied. No live Gemini call was made
> and no key value was stored. Public-deployment caveats are unchanged.

> **Standalone build addendum.** A Linux Mint 22 / Ubuntu 24.04 x86_64 `.deb`
> now bundles the built UI, backend, five CSV datasets, dataset guide and an
> offline CPython 3.12 wheelhouse. It uses one local FastAPI process and an
> app-mode browser window. Playwright/bundled Chromium and a bundled Python
> interpreter are intentionally excluded for size; PDF and 300-dpi PNG exports
> degrade explicitly, and the package depends on `python3.12-venv`. This is a
> private local installer, not a signed public release; the production caveats
> above still apply.

The new regression suite is `backend/tests/test_part15_audit_regressions.py` (24 tests). It covers both unit-level statistical defects and the API report contract.

---

## 3. Residual limitations — disclosed, not silently “fixed”

1. **No Docker runtime validation.** The sandbox has no Docker daemon; `docker compose up`, `Dockerfile.single`, Render/Railway/Fly/HF deployment paths are statically verified only. Do not read the deploy files as proof a container has booted.
2. **`frontend/vercel.json` still contains a placeholder API host** (`https://your-name-chronos.hf.space/api/$1`). It must be replaced with a real Space/API origin before Vercel deployment. Static tests check rewrite shapes, not the hostname.
3. **Optional dependencies are absent in this environment.** `weasyprint`, `arch`, `pmdarima`, `prophet`, `statsforecast`, `sktime`, `tbats`, `lightgbm` and `xgboost` are reported unavailable by `/api/v1/health`. PDF and 300-dpi PNG report exports require Chromium/`INSTALL_BROWSER=1`; the app reports the refusal rather than pretending the artifact exists.
4. **Statistical caveats remain by design.** Realised variance is a noisy squared-return proxy; VaR uses a Gaussian lower-tail quantile; short series and overlapping folds have low inferential power. These are disclosed in notes/warnings, not converted into passes.
5. **No independent peer review or external clinical/financial validation.** Golden fixtures, property tests and 91.61% coverage are strong internal evidence, not proof of real-world forecast validity.
6. **No production security, load, persistence or backup audit.** Sessions live in server memory; there is no database. The browser uses relative `/api` URLs; no hardcoded localhost host was found in browser source.
7. **Frontend bundle warning.** The built JS chunk is 504.04 kB (142.74 kB gzip), just over Vite’s 500 kB advisory threshold.
8. **Connected apps were enabled but unused.** No Gmail, Calendar, Docs, Drive, Linear or Notion operation was performed.
9. **Broad defensive `except Exception` blocks** in AI/report helpers were triaged as defensive; intentional `[PLACEHOLDER: …]` manuscript slots and abstract `NotImplementedError`s are not defects.

---

## 4. Credibility assessment

**Substantially more credible than before the audit, but not yet production-credible.**

- For a **local research / teaching / development install**, credibility is high: the statistical core now refuses invalid operations, reconciles every held-out row, uses canonical nested-model and multiple-testing procedures, discloses construction and exclusions, and is covered by 1014 backend tests, 53 frontend tests and 91.61% statistical-layer coverage.
- For **scientific or operational forecasting**, credibility is moderate: methods are documented and internally consistent, but overlap, low power, Gaussian VaR, noisy variance proxies, optional dependencies and lack of independent external validation remain real caveats.
- For a **public, unattended production service**, credibility is low until the deployment caveats above are closed. In particular, a real container smoke test, a non-placeholder Vercel/API origin, a security review, and optional-dependency decisions are still needed.

---

## 5. Is it safe and ready to install?

**Yes — safe to install locally for evaluation, development, teaching and research, in a clean Python/Node environment, following `README.md` / `DEPLOY.md`.** Reasons: full test suite passes, typecheck/build are clean, live health/demo/UI smoke works, no application JS exceptions were observed, no external-app writes were made, and the audit removed the critical statistical and reporting defects.

**No — not ready to call production-safe or ready for a public unattended deployment.** Before exposing it, replace the Vercel placeholder, run a real Docker/container smoke test on the target host, decide which optional dependencies to install, add security/load/persistence review, and verify PDF/PNG export on the deployment image. Treat the current build as a credible, well-tested prototype rather than a certified forecasting service. The exact public-release runbook is in [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md), with a CI gate in `.github/workflows/public-release.yml`.

---

## 6. Reproduce the audit

```bash
cd backend
python -m compileall -q app
PYTHONPATH=. pytest -q
PYTHONPATH=. pytest -q --cov=app/ts --cov-fail-under=80 --cov-report=term-missing
PYTHONPATH=. pytest -q tests/test_part15_audit_regressions.py tests/test_part15_metrics.py

cd ../frontend
npm run typecheck
npm test
npm run build
```

## 7. References used for the statistical corrections

- Clark, T. E. & West, K. D. (2007), *Approximately Normal Tests for Equal Predictive Accuracy in Nested Models*, Journal of Econometrics 138(1), 291–311. NBER working paper: https://www.nber.org/system/files/working_papers/w14601/w14601.pdf
- Philadelphia Fed WP 11-31 on one-sided adjusted MSPE statistics: https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/2011/wp11-31.pdf
- Holm step-down family-wise error control: https://www.clinicalbiostats.com/tutorials/undergraduate-statistics-introduction-to-statistics/holm-adjustments-for-multiple-tests/
- Dunn & Smyth (1996) randomized quantile residuals; Diebold & Mariano (1995); Gneiting & Raftery (2007); Kupiec (1995); Christoffersen (1998). These are also cited in `METHODS.md` and the API result cards.
