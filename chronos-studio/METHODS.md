# Chronos Studio — METHODS

> **Audience:** a reviewer, supervisor or thesis committee who must be able to
> defend every number the application prints. This document states exactly which
> test, which implementation, which assumption and which data window produced
> each reported quantity. Where the app deliberately differs from a textbook
> default, the difference and the reason are written down.
>
> Companion documents: [README.md](README.md) (usage),
> [GLOSSARY.md](GLOSSARY.md) (242 plain-English terms),
> [CONTRIBUTING.md](CONTRIBUTING.md) (how to change anything safely),
> [CHANGELOG.md](CHANGELOG.md), [DEPLOY.md](DEPLOY.md).

---

## 0. Provenance and the non-negotiable rules

Every result is a `ResultBase` carrying `id`, `stage`, `title`, `plain`, `why`,
`method`, `caveats`, a `reproducible_snippet` (valid Python source) and the data
it was computed from. No card is free text written by a language model: the AI
assistant may rephrase a stored number but may not invent one. A
`PipelineState` records stage status, `series_version`, the content hash of the
loaded data, the analysis seed and the installed library versions. Changing the
series bumps `series_version`; results from an older version are marked
**stale** and the UI says "Needs re-running" instead of reusing them.

The twelve safeguards in the CRAFT prompt (§6) are enforced in code, not in UI
copy. The most directly testable ones are pinned by
`backend/tests/test_part19_golden.py`:

| # | Rule | Where enforced | Regression test |
|---|---|---|---|
| 1 | Never difference a series two unit-root/KPSS tests say is already stationary | `app/advisor/rules_stat.py` `stat.already_stationary.no_diff` (a `protect` rule, not a hint) | `test_stat_advisor_protects_an_already_stationary_series`, `test_next_advisor_leaves_a_white_noise_stationary_series_alone` |
| 2 | Never difference more than twice without a recorded justification | `app/ts/preprocess/ops.py` difference op + `PreprocessingRecord.reason` | existing Parts 3/5 suite |
| 3 | Never log / Box–Cox non-positive values; block multiplicative decomposition too | `app/advisor/rules_prep.py::preferred_transform` returns `yeojohnson`; a **blocker** rule fires and no log rule is offered | `test_advisor_never_logs_or_boxcoxes_a_nonpositive_series` |
| 4 | Never compute MASE / relative metrics without fitting the naive benchmark | `app/ts/metrics/point.py::point_metrics` requires `benchmark` and a training-only `scales` array; `app/ts/metrics/policy.py` | `test_mase_of_a_perfect_forecast_is_zero`, `test_mase_of_the_naive_forecast_is_one`, `test_mase_properties_hold_over_many_random_series` |
| 5 | Never compare AIC/BIC across differing differencing or effective samples | `FittedSummary.ic_comparable` + `app/ts/select/` comparability guard; baselines have `ic_comparable=False` | `test_baselines_have_no_fabricated_information_criteria` |
| 6 | Never report in-sample fit as forecast accuracy | evaluation reads only out-of-sample fold blocks (`app/ts/metrics/`) | Parts 13/15 suite |
| 7 | Never let a test-period value influence a modelling choice | rolling-origin CV expands the training window only; MASE denominators use training windows only | Parts 13/15 suite |
| 8 | Never present an interval without its coverage and construction | every forecast card states nominal coverage and `analytic` / `simulation` / `bootstrap` | Parts 14/16 suite |
| 9 | Always show `n`, frequency and the exact evaluation window beside an accuracy number | result `method`/`plain` templates + report tables | Parts 15/18 suite |
| 10 | Always attach the residual-adequacy verdict to a forecast | `S12` adequacy card is a dependency of `S15`; forecast cards cite it | Part 18 smoke |
| 11 | Flag horizon > ~n/5 and n < 2 periods per season | `app/ts/forecast/` horizon advisor; reported as caveats | Parts 14/18 suite |
| 12 | Never silently drop rows; count and report every dropped row | `app/ts/io/ingest.py` issues + `n_silently_dropped` counters and the data-quality card | Parts 1/2 suite |

---

## 1. Ingestion and the data-class branch

`read_any` accepts CSV, Excel, JSON, Parquet, Stata and SPSS (the last two only
when their optional reader is installed). `ingest_table` then:

1. applies the **user-declared** `ColumnMapping` (time column, value column,
   exogenous columns, `dayfirst`, and the explicit `row_order_is_time` flag when
   there is no date column). Nothing about the mapping is guessed silently;
2. parses the index to a `DatetimeIndex`, infers frequency and seasonal period,
   and records `n_dropped` with a reason for every row that cannot be placed;
3. classifies the values into **continuous / count / intermittent /
   event_timestamp**. This `data_class` is the single switch that determines
   which model families and which error metrics are legal downstream.

`Dataset.content_hash` fingerprints the loaded values; reports and the
`.chronos` archive carry it so a reader can prove which file produced a number.

**Seed data.** Nineteen datasets ship inside the image
(`backend/app/assets/examples/manifest.json` and `scripts/gen_examples.py`),
covering all four data classes. `test_all_bundled_seed_datasets_load_with_sane_metadata`
loads every one through the real HTTP endpoint and checks `n ≥ 12` and that the
detected data class equals the manifest's declared class.

---

## 2. Exploratory analysis (S1)

`app/ts/eda/descriptives.py::describe` computes `n`, missing count/percentage,
start/end, mean, median, SD, variance, CV, min/max, quartiles, IQR, range,
skewness and excess kurtosis (bias-corrected, `scipy.stats`), first/last, total
and percent change, CAGR, counts of increases/decreases/ties/zeros/negatives,
median absolute deviation, and a normality battery
(Shapiro–Wilk, Jarque–Bera, D'Agostino–Pearson).

Additional evidence: a runs test for persistence, rolling-SD stability
(first vs last third of the rolling SD), a seasonal profile with a
Kruskal–Wallis test across season groups, `STL` seasonal strength, and a
peak-to-trough amplitude trend.

**Outlier detection.** Candidate points are z-scores of a robust remainder.
`_tail_outliers` asks whether each of the most extreme points is far out *for
its rank*: for rank `k` of `n`, the expected magnitude is
`norm.isf(0.5·α·k/n)` with α = 0.01, and a point is kept if
`|z| ≥ 1.15 × expected`. This stops the ordinary tail of a short series being
reported as anomalies. In Part 19 the ten-thousand scalar `norm.isf` calls were
replaced by **one vectorised call** over the same formula — numerically
identical (pinned by `test_tail_outlier_rule_is_vectorised_and_stays_interactive`),
~1.1 s → ~0.01 s on 10k rows. Spikes, level shifts and temporary changes are
labelled separately; level shifts are detected on the **deseasonalised** series
because STL absorbs a step into its trend.

---

## 3. Preprocessing advisor and engine (S2/S3)

`app/advisor/rules_prep.py` ranks rules from computed facts only: gap-length
advice, zero-run handling, variance-stabilisation advice, outlier flagging
(never deletion by default), and aggregation. `preferred_transform` offers
**exactly one** of log / sqrt / Box–Cox / Yeo–Johnson / none, chosen by the
Box–Cox λ confidence interval; an interval containing λ = 1 means "leave the
scale alone". A variance fix requires evidence (rolling-SD ratio > 1.25, a
level–spread correlation ≥ 0.5, or a genuinely widening seasonal swing) — λ
alone is a random number on noise.

`POST /preprocess/preview` runs the **same** `apply_op` as commit on a copy of
the series. `POST /preprocess/apply` records a `PreprocessingRecord` (version,
method, params, the advisor's own reason, reversibility) and bumps
`series_version`. **Undo** replays the remaining chain from the untouched
original; **redo** (added in Part 19) re-applies the last undone record, and any
fresh apply clears the redo history exactly as an editor does. Both are pinned
end-to-end by `test_preprocess_undo_then_redo_restores_the_same_series`.

---

## 4. Decomposition and components (S4)

Methods: classical (additive/multiplicative), STL, MSTL, UCM (Kalman), HP and
Christiano–Fitzgerald band-pass, plus an X-13 adapter used only when
`X13AS_PATH` points at a real binary. Multiplicative decomposition is blocked
on non-positive data by the same guard as the log transform.

Quantification (`app/ts/decompose/components.py`):

* **Trend** — direction from the component's own first/last movement, Mann–Kendall
  τ with continuity correction and tie-corrected variance, Sen's slope with a
  distribution-free 95% interval, Cox–Stuart, and the variance share. The
  strength band is *capped by the test*: a trend Mann–Kendall cannot distinguish
  from noise is never labelled better than "weak", whatever F_T says.
* **Tests run on the deseasonalised series**, not the loess-smoothed trend: a
  smoothed line is massively autocorrelated by construction, so its p-value is
  meaningless (it used to report p < 0.0001 for white noise).
* **Season** — period(s), `F_S`, peak-to-trough amplitude in units and % of the
  mean, the seasonal index table, Kruskal–Wallis, and whether the swing is
  growing.
* **Cycle** — periodogram peaks and Fisher's g-test. A statistically significant
  peak that is not on a calendar lag is reported as a **cycle**, never as
  seasonality.
* **Remainder** — variance share, Ljung–Box, normality, ARCH-LM and the explicit
  "is this actually noise?" verdict. Structure here is a finding for the
  modelling stage, not something to smooth away.

### Performance pass (Part 19, §5.15)

Mann–Kendall and Theil–Sen are O(n²). On 10,000 points the original
implementation built a Python list of ~50,000,000 pairwise slopes (~400 MB and
several minutes), which made S4 unusable. Part 19 keeps both statistics
**exact** for ordinary series and introduces documented bounded paths for very
long ones:

| Statistic | Exact path | Large-n path | Threshold |
|---|---|---|---|
| Mann–Kendall S | vectorised `np.sign` double loop | Fenwick tree over compressed ranks, O(n log n), **mathematically identical** | n > 2,000 |
| Theil–Sen slope | all pairwise slopes via `np.triu_indices` | deterministic, shape-preserving thinning to 1,000 evenly spaced points (endpoints kept); slopes still divided by the **original** index distance so units stay "per time step" | n > 1,500 |
| Tail-outlier expectation | one vectorised `norm.isf` | unchanged (O(n)) | — |
| Plotly theme application | one `update_layout` call per layout key | one merged `update_layout(layout)` call; identical nested-property merge, ~0.2 s saved on nine EDA figures | all figures |
| Robust STL fit (weekly period) | two independent `statsmodels` fits in S1 | an eight-entry LRU cache keyed by the exact float64 values, period and seasonal window; seasonality and outlier passes share one fit | n ≥ 2m+2 |

Nothing is random, so every result stays reproducible. Exactness is pinned by
`test_mann_kendall_fenwick_is_exact_vs_brute_force` (including ties),
`test_sens_slope_small_n_is_identical_to_all_pairs` (n = 10, 200, 1,400) and
`test_robust_stl_cache_is_exact_and_reuses_the_weekly_fit` (bit-identical
trend / seasonal / residual arrays plus one cache hit).

Measured on the development sandbox (slower than a laptop), after all five
optimizations:

```
paste 10k rows      0.67–0.91 s
profile (S1)        0.59–0.89 s
decompose STL (S4)  0.39–0.56 s
--------------------------------
total               1.73–2.27 s   (before Part 19: > 300 s, timed out)
```

`test_10k_rows_end_to_end_stay_interactive` re-runs these three HTTP calls and
asserts the strict §7 `<3 s` budget including ingestion. A coverage tracer is
not an interactive runtime: the same test uses the historical 12 s CI ceiling
only when `coverage.Coverage.current()` is active.
`test_10k_row_trend_statistics_are_no_longer_quadratic` asserts the two trend
statistics alone finish in < 5 s (measured 0.045 s). The full 990-test suite
passes in 609 s; the coverage-gated run passes in 936 s at **91.71%** on
`app/ts` (12,584 / 13,722 statements).

---

## 5. Stationarity and structural tests (S5)

| Test | Implementation | Null | Notes |
|---|---|---|---|
| ADF | `statsmodels.tsa.stattools.adfuller`, `autolag="AIC"`; three deterministic specs (`n`, `c`, `ct`) | unit root | MacKinnon response-surface p-values; lag count printed |
| KPSS | in-app `statsmodels` KPSS with an explicit short-lag Newey–West bandwidth | level/trend stationarity | p-value clipped to [0.01, 0.10] as statsmodels does |
| Phillips–Perron | non-parametric correction of the ADF statistic | unit root | no added lags |
| DF-GLS | GLS-detrended Dickey–Fuller | unit root | better small-sample power |
| Zivot–Andrews | one endogenous break | unit root with break | level/trend/both |
| Canova–Hansen, OCSB, HEGY | seasonal unit roots | seasonal stability | HEGY separates frequencies |
| Chow, Quandt–Andrews, Bai–Perron, CUSUM | structural breaks | no break | known vs unknown vs multiple dates vs drift |
| ARCH-LM | Engle LM on residuals | homoskedasticity | motivates GARCH |

**Reconciliation** is a 2×2 matrix combining a unit-root test and a
stationarity test. Both stationary → do not difference (a `protect` rule);
both non-stationary → difference; disagreement → "inconclusive" with the
contradiction named. This is the enforced version of safeguard #1.

### Why our KPSS number differs from statsmodels

The app deliberately uses a short-lag bandwidth giving 5 lags on
AirPassengers, whereas `statsmodels.kpss(..., nlags="auto")` uses 8 and
`nlags="legacy"` uses 14. All three reject level-stationarity (the verdict
agrees) but the statistics differ (2.336, 1.651, 1.052). The difference is
documented in `tests/fixtures/airpassengers_reference.json` and pinned by
`test_kpss_bandwidth_difference_is_documented_and_verdicts_agree`. A reviewer
should never have to discover this by accident.

---

## 6. ACF/PACF, the next-step advisor and the gate (S6–S8)

ACF/PACF/CCF are computed with Bartlett bands (±1.96/√n), a pattern
classifier, order hints and a Ljung–Box battery (including a seasonal variant).
`app/advisor/rules_next.py` ranks detrend / difference / seasonal-adjust /
leave-alone actions, and the **preview** endpoint re-runs the stationarity tests
on the candidate series so a recommendation is checked before it is committed.
Ranking is deterministic (ties broken by rule id) — pinned by
`test_next_advisor_recommends_a_first_difference_for_a_unit_root`.

`S8` produces the readiness gate: a checklist, blockers and a `caveats` object.
Caveats propagate into downstream payloads so a forecast or a report cannot
hide a known limitation.

---

## 7. Model registry, fitting and the comparability guard (S9–S10)

One `ModelSpec` table (`app/ts/models/registry.py`) describes every model, its
data class, assumptions and minimum sample. `ModelWrapper.fit()` never raises a
modelling failure: `fit_one()` always returns a `FitOutcome`, with non-convergence
or inapplicability recorded in `summary.failed` and `failure_reason`. This is
why a failed family can still appear honestly in a golden fixture.

Families: baselines (white noise, Gaussian WN, naive, seasonal naive, mean,
drift — **always fitted**), ARMA/ARIMA/SARIMA/ARFIMA, ETS/Holt–Winters/Theta,
UCM, SARIMAX/ARIMAX/ARDL, count (Poisson with a mandatory overdispersion
pre-check, NB2, zero-inflated/hurdle, INAR/INGARCH), intermittent (Croston,
SBA, TSB, ADIDA), point processes (renewal, Hawkes), volatility (ARCH, GARCH,
EGARCH, GJR-GARCH, GARCH-t/GED), multivariate (VAR/VARX, VECM), nonlinear
(SETAR, Markov-switching, NNAR, GP) and ML (RF, LightGBM/XGB, k-NN, SVR —
**opt-in and always benchmarked against ETS/ARIMA with a model card**).
Forecast combination requires component forecasts first and fails cleanly
otherwise.

`FittedSummary.ic_comparable` is false for baselines and whenever differencing
or effective sample size differs; §6.5 forbids cross-model AIC/BIC ranking in
those cases. The app refuses rather than printing a misleading table.

---

## 8. Selection, diagnostics, forecasting and evaluation (S11–S14)

**Selection** uses expanding-window rolling-origin cross-validation (no test
value ever enters a training statistic), information criteria with the
comparability guard, and Diebold–Mariano / Clark–West comparisons with HAC
corrections.

**Diagnostics** cover residual autocorrelation (Ljung–Box), QQ with bands,
normality, ARCH-LM, PIT and RQR/Dunn–Smyth uniformity (KS/CvM), and produce a
single residual-adequacy verdict that is attached to every forecast.

**Forecasts** are predictive distributions, not just points: analytic intervals
under Gaussian assumptions, seeded simulation intervals otherwise, discrete
predictive distributions for count/intermittent models, back-transformation
bias adjustment after log/Box–Cox, exogenous scenario paths, and fan charts.
Every interval states its nominal coverage and how it was built. `MIN_PATHS =
2000` simulation paths is a hard floor (HTTP 422 below it), which is why the
demo replay uses exactly 2,000.

**Metrics** (`app/ts/metrics/`): MAE, RMSE, ME, MPE, MAPE, sMAPE, MASE, RMSSE,
GMRAE, MdRAE, relative MAE/RMSE, MASE-by-horizon; CRPS (sorted-draw empirical
identity, Gneiting & Raftery 2007), pinball/quantile loss, log score with the
continuous caveat, Winkler interval score, PI coverage, average width,
calibration, PIT/RQR uniformity, and a DM matrix. The **metric-selection
policy** bans MAPE on intermittent/zero-heavy data and switches to CRPS when a
probabilistic forecast exists, always with a printed reason.

Properties pinned as tests: MASE of the perfect forecast = 0, MASE of the
naive forecast = 1, and both over a 25-seed sweep with random slope magnitude
and sign; `mase_scale` returns `None` for a constant series (so MASE is
refused, not reported as division by zero).

---

## 9. AI assistant, reporting and provenance (S15, Part 17)

The assistant consumes only a compact `AnalysisDigest` of **computed** results,
the question and the current panel id. Data values are OFF by default, the
payload is previewable before the first live request, the key lives in the
browser tab (`sessionStorage`), and a number-grounding validator checks the
answer against the digest. Offline narrative mode needs no key; removing the
whole AI layer leaves a fully functional app.

`app/ts/report/scorecard.py` computes ten weighted completeness checks —
quality (10), preprocessing (10), components (10), ≥2 agreeing stationarity
tests (12), ACF/PACF (8), ≥2 candidates + benchmarks (10), out-of-sample
evaluation (12), residual adequacy (10), forecast intervals (10),
assumptions/caveats (8) — from **stored results**, never from stage status
alone. Verdicts are exact: *"Analysis complete — ready for manuscript
writing"*, *"Nearly ready — 2 items outstanding"*, *"Not yet — see blockers"*.
Numbered next steps name the exact panel to visit. Finalization writes three
S15 cards whose `caveats` are empty, so a report card can never be
mis-read as evidence of its own readiness.

Exports: one-file HTML (self-contained, no external CSS/JS/fonts), DOCX, PDF,
LaTeX tables, XLSX tables, reproducible Python and R scripts, a `.chronos`
session archive, a figures ZIP (300-dpi PNG + SVG + PDF) and an everything ZIP.
`render_all` writes any builder failure into `errors.txt` — a bundle is not
proof every export succeeded, which is why the tests assert that file is absent.

---

## 10. Golden references (Part 19, §5.15 / §7)

Two independent fixture files live in `backend/tests/fixtures/`.

### 10.1 statsmodels / R-comparison reference — AirPassengers

`airpassengers_reference.json` was produced by calling statsmodels directly on
the **bundled** 1949-01-01 → 1960-12-01 monthly series (n = 144, mean
280.2986111111111, period 12). statsmodels is the engine behind the equivalent
R routines in `tseries::adf.test`, `urca::ur.kpss` and `forecast::ets`/`Arima`.

| Quantity | Reference value | How the app is checked |
|---|---|---|
| `adfuller(y, autolag="AIC", regression="c")` | statistic **0.8153688792060528**, p **0.9918802434376411**, usedlag **13** | `adf(y,"constant")` must match to 1e-10 (`test_app_adf_constant_matches_statsmodels_exactly`) |
| `kpss(y, regression="c", nlags="auto")` | statistic **1.6513122354165206**, p 0.01, lags **8** | documented difference; verdict must agree |
| `kpss(..., nlags="legacy")` | statistic **1.0521750110135095**, lags 14 | regenerated in-test |
| `SARIMAX(1,1,1)(1,1,1,12)` | AIC **922.2053774351551**, BIC **936.0162471091439** | regenerated in-test to 1e-5 |
| `ExponentialSmoothing(trend="add", seasonal="mul", m=12)` | AIC **709.8917192771407**, SSE **15952.880435010864**, 3-step forecast **445.2424 / 418.2254 / 465.3097** | regenerated in-test to 1e-5 |
| App ADF (none / constant / constant+trend) | 2.3906017343512316 / 0.8153688792060528 / −2.1007818138446708, all lags 13, all "unit root" | frozen per family |
| App KPSS (level / trend) | 2.335887287437073 (unit root) / 0.10279920639041347 (stationary), lags 5 | frozen per family |

`test_fixture_is_an_independent_statsmodels_reference` recomputes all four
estimators with the installed statsmodels and compares against the frozen
values, and checks that the major.minor version recorded in the fixture matches
the installed one. If statsmodels is upgraded, **this test fails on purpose**:
a human must review the statistical change before refreshing the fixture.

### 10.2 Per-family golden fits

`golden_models.json` holds one case per implemented family (13 cases), each
loaded through `read_any → ingest_table` and fitted through the public
`fit_one`. Stable fields (`failed`, `converged`, `n_obs_used`, `n_params`,
`ic_comparable`) and the information criteria are compared with 1e-6 relative
tolerance. Representative values: ETS/airpassengers AIC 696.2552607367107 /
AICc 699.5110746901992; ARIMA 1350.4334330698105; UCM 1178.946289433545;
Poisson/outbreak_counts 316.8651097728058; GARCH/volatile_returns
1055.0569185612785; VAR/traffic_spend 1884.1632294128722;
ARIMAX 1094.333288068537; SETAR/lynx 1800.8317538253812;
renewal/machine_service 390.25840636838404. The combination family is pinned
as `failed: true` with a non-empty reason, because it correctly requires
component forecasts.

Regenerate with `PYTHONPATH=. python scripts/generate_golden.py` from
`backend/`. **A changed AIC is a statistical regression, not a fixture
refresh** — review the diff before committing.

---

## 11. Demo mode, background jobs and seed studies (Part 19)

`app/api/demo.py` exposes three narrated studies
(`seasonal_demand` → AirPassengers, `marketing_mix` → traffic_spend,
`intermittent_parts` → spare_parts). Each is an ordered 17-step recipe that
replays the **real** S0→S15 endpoints in dependency order
(S6 → S7 → S8 → S9 → … → S14 → S15). `POST /demo/start` creates a fresh
session and returns immediately; a daemon worker executes the steps via an
in-process ASGI client, and `GET /demo/status` reports labelled progress,
`done/total/percent`, per-step timings and errors. `POST /demo/cancel` is
cooperative (the current step finishes, then the thread stops);
`DELETE /demo/jobs/{id}` removes the job record. Jobs are in-memory,
TTL-swept after one hour and contain no user values.

This is the same machinery §5.15 asks for heavy models: CV grids and ML fits are
the slow steps in the recipe, and they run off the request thread with progress
and cancel so the UI never blocks. Measured replay: **17/17 steps, ~20 s, zero
errors, S15 complete, score 100/100** for all three scenarios.

---

## 12. Accessibility, privacy and reproducibility

* **WCAG 2.1 AA.** The ribbon implements the WAI-ARIA toolbar pattern (one tab
  stop, Arrow/Home/End navigation). Every figure has `role="img"` plus an
  `aria-label` that **summarises the values**; every table has a `<caption>` and
  an `.sr-only` value summary. Status is never colour-only (words + icons +
  shapes). `:focus-visible` provides a keyboard focus ring and
  `prefers-reduced-motion: reduce` suppresses all transitions. "Simplify view"
  hides all but eight essential panels; the glossary supplies 242 tooltips that
  work on hover **and** keyboard focus and are dismissable with Escape.
* **Colour.** A single theme module (`app/charts/theme.py`) owns every colour
  literal; a test forbids hex literals elsewhere. Palettes are colourblind-safe
  and print/light/dark modes share one source.
* **Privacy.** No database, no server-side persistence beyond the session TTL
  (`CHRONOS_SESSION_TTL_HOURS`, default 6). "Delete session" hard-deletes in
  memory. AI payload preview, data-values-off-by-default, a visible banner when
  values are enabled, and a hard request cap. Reports never contain a Gemini
  key.
* **Reproducibility.** One seed fixes every random draw (simulation paths,
  bootstrap IRF bands, ML splits). Library versions, content hash, the exact
  evaluation window and the reproducible Python/R snippet travel with the
  results and the archive.

---

## 13. Known limitations (stated honestly)

1. `arch` is not installed in the development image, so GARCH-family fits use
   the statsmodels/fallback path; the registry still lists the full family.
2. PDF and 300-dpi PNG exports require a Chromium binary. Local development
   uses `CHRONOS_CHROME` or the cached Playwright Chromium; a default Docker
   image omits it (build with `INSTALL_BROWSER=1`) and those two export kinds
   degrade into the bundle's `errors.txt`. WeasyPrint is not used.
3. X-13ARIMA-SEATS, Prophet, TBATS, LightGBM/XGBoost, LSTM/TFT, stochastic
   volatility, panel models and LSTM-style deep models are **flagged** in the
   registry rather than implemented; the app says so instead of fabricating a
   result.
4. The Theil–Sen and Mann–Kendall large-n paths are approximations/different
   algorithms only above 1,500 / 2,000 observations respectively; they are
   deterministic and documented in §4. Below those limits the statistics are
   exact and unchanged.
5. A single-container free-tier deployment holds sessions in one process:
   horizontal scaling would require replacing `SessionStore` with a shared
   store. This does not affect a single-user or demo deployment.
6. There is no Docker daemon in the build sandbox, so `docker compose up` is
   verified by static configuration tests (`test_part19_deploy.py`) and by the
   documented commands in [DEPLOY.md](DEPLOY.md), not by a live container run
   here.
