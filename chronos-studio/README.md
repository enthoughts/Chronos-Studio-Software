# Chronos Studio

**A free, browser-based time-series workbench — "JASP for time series".**
Load a spreadsheet, get guided preprocessing advice, decomposition, stationarity
verdicts, ACF/PACF, model suggestions, diagnostics, forecasts, accuracy
evaluation and a manuscript-ready report. No code, no install for the user.

> Build status: **Parts 0–19 complete, plus a 2026-11-08 critical audit** — ingestion through
> forecasting, out-of-sample evaluation, the §5.10 / §8 plots-and-tables completeness pass,
> the grounded AI assistant, S15 reports / finalization, and the final hardening / docs /
> deploy pass. Historical closures stay distinct: Parts 0–13 ended at **787 backend / 3 frontend
> tests**; Part 14 at **818 / 3**; Part 15 at **850 / 7**; Part 16 at **865 / 13**; Part 17 at
> **883 / 20**; Part 18 at **906 / 23**; Part 19 closed at **990 / 52**. The post-release audit
> raised the current suites to **1014 backend / 53 frontend tests**.
> The full backend suite now passes in **~555 s**; `pytest --cov=app/ts --cov-fail-under=80`
> reports **91.61%** on the statistical layer (13,925 statements, 1,168 missing). The strict
> 10k-row paste + S1 profile + S4 STL path takes **1.73–2.27 s including ingestion** on the dev
> sandbox, below the §7 `<3 s` budget. Typecheck, 53 frontend tests and the production build are
> clean; live HTTP/UI smoke produced shots 198–215 with no application JS exceptions. The audit
> fixed Part 15 statistical defects, silent row drops, leakage-audit completeness, honest report
> wording and the clipped Part 19 top bar; see [`AUDIT.md`](AUDIT.md) and the changelog.
> `README` / `METHODS` / `GLOSSARY` / `CONTRIBUTING` / `CHANGELOG` / `DEPLOY`,
> `docker-compose.yml`, `Dockerfile.single`, demo mode and seed data ship with Part 19. No Docker
> daemon is available in this sandbox, so `docker compose up` is validated statically, not by a
> live container run. `frontend/vercel.json` contains a placeholder API host that must be
> replaced before deployment; optional `weasyprint` / `arch` and several model libraries are
> absent here, so PDF/300-dpi PNG exports and those optional families need a target-host check.
> For a private install none of the public-deploy hardening is needed; if you later publish the app,
> follow [`PUBLIC_RELEASE_CHECKLIST.md`](PUBLIC_RELEASE_CHECKLIST.md) and the CI gate in
> `.github/workflows/public-release.yml`.

---

## Quick start

For a clean Linux Mint local install from the current
`CHRONOS-local-install-v4.zip` workspace archive, see
[`INSTALL_LOCAL.md`](INSTALL_LOCAL.md) and run
`bash scripts/install-local-linux-mint.sh` once. Then:

```bash
bash scripts/run-local.sh
# or, after activating backend/.venv:
./scripts/dev.sh
```

For a JASP/VS-Code-style installable desktop application instead, see
[`STANDALONE.md`](STANDALONE.md) and install
`chronos-studio-standalone_0.1.0_amd64.deb`.

That installs the frontend once, then runs:

| | URL |
|---|---|
| App | http://localhost:5173 |
| API docs (Swagger) | http://localhost:8000/docs |

Manual equivalent:

```bash
# terminal 1
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# terminal 2
cd frontend && npm install && npm run dev
```

Docker:

```bash
docker compose up --build      # app on http://localhost:8080
```

---

## What works today

**Step 0 — Load data.** Upload `.csv/.tsv/.txt/.xlsx/.xls/.ods/.json/.parquet/.dta/.sav/.rds`,
paste a table straight from Excel, or open one of **19 bundled teaching datasets** (including two that are nothing but event dates
and one weekly series whose *variance* is the interesting part).

The loader then tells you, in plain language and with the evidence:

- which column is the **time index** (or that it built one from `year` + `month`),
- the **calendar** and seasonal period — including the cases that break most tools:
  `2020-01`, `1994Q3`, epoch seconds/millis, Excel serial dates, day-first
  `03/04/2020`, irregular spacing, and twice-monthly data that is silently
  rounded to "monthly" by `pd.infer_freq`,
- **gaps** inserted as blanks rather than closing them up (the classic bug that
  shifts every later value onto the wrong month),
- duplicates, missing runs and their dates, currency/thousands/parenthesis
  numbers, `1.234,56` European decimals,
- a **data class** — continuous / count / intermittent / binary / event-timestamp —
  which later decides which models get offered and which accuracy metrics are legal.
  A file with a single date column and no values is read as **event timestamps**: each
  row is one occurrence, events are counted into the coarsest calendar that keeps at
  least 24 periods, and the raw timestamps are kept so the point-process models can
  work on intervals rather than counts.

Every guess is overridable in *Change mapping*, and every load produces a
content hash so a result can be reproduced.

**Step 5 — Test stationarity (Part 5)** ADF, KPSS, Phillips–Perron, DF–GLS,
Zivot–Andrews, the OCSB seasonal unit-root test, Canova–Hansen, and four structural-break
tests (Chow, Quandt–Andrews sup-Wald, Bai–Perron, CUSUM) — then **one verdict**, in words,
with the numbers it used.

The design decisions here are all about refusing to be confidently wrong:

- **Two tests with opposite nulls.** ADF hopes to reject "unit root"; KPSS hopes not to
  reject "stable". The 2×2 of their answers is shown on screen, including the two cells that
  mean "the tests disagree" and "not enough evidence" — those are answers, not failures.
- **A break can overturn a unit-root verdict, but only with two witnesses.** A random walk
  will always let a sup-Wald test place a "break", so Quandt–Andrews is never sufficient on
  its own; Zivot–Andrews (which has a unit-root null and allows a break) must also reject,
  and the size of the level change must exceed 4.5× the series' ordinary move. When those
  disagree the verdict is `unit_root_or_break` and says so, rather than picking the tidier
  story.
- **Simulated critical values are labelled.** No published table covers n = 72 with a 15%
  trim, so the battery simulates (400 replications, seed 20260101, cached) and every result
  carries a caveat saying so. The null itself matches the series: white noise for stationary
  data, AR(φ = 0.5) for persistent data — an i.i.d. null called a random walk's drift a break,
  and a random-walk null could not see a textbook level shift at all.
- **Missing tests are missing.** A test that cannot run (too short, or numerically
  rank-deficient — which is how statsmodels' ADF responds to a quarterly series, returning a
  statistic of 3.5e13) is omitted and said to be omitted. It is never reported as a pass.
- **The guardrail.** A stationary verdict produces an explicit "do not difference this
  series", with the cost: a moving-average unit root, wider intervals, worse MASE.
- **Break dates go forward, not away.** Each detected date is offered to the model stage as
  a step dummy; the app never deletes the affected observations.

**Step 3 — Decompose (Part 4).** Six methods (STL, classical, MSTL, unobserved
components, Hodrick–Prescott, band-pass), additive or multiplicative, plus a
`compare` endpoint that runs several at once — then **one sentence** saying what the
series is mostly made of, and a card per component saying what it is and is not.

The decisions worth knowing about:

- **The decomposition is a description, not a model.** No likelihood, no standard
  errors, no forecast — and that caveat rides on every result, because the single most
  common misuse of an STL plot is to read the seasonal line as a prediction.
- **A small remainder is not a solved problem.** The scorecard reports the three shares
  by value ("86% trend, 14% seasonal, 0.3% remainder") and, when the remainder is the
  largest, says so and points at serial dependence *inside* it instead of inventing a
  main component. The remainder card then tests it: Ljung–Box, ARCH-LM and Jarque–Bera,
  so "not noise" is a finding rather than an assumption.
- **A strength band can never contradict its own test.** F_T divides by the remainder, so
  a tiny remainder makes any wiggle look strong: on white noise F_T = 0.99 while
  Mann–Kendall finds no drift (p = 0.15). The band is therefore capped by the drift test
  and the caveat explains the inflation, instead of printing "strong trend" over a
  p-value that says otherwise.
- **Multiplicative is refused, never silently swapped.** Dividing by the level is
  undefined at zero or below, so the app returns `decompose.multiplicative_undefined`
  naming the count ("93 of 180 values are zero or negative") and lets you choose — a
  silent fallback to additive would answer a different question than the one asked.
- **Fisher's g gets a 1% bar and a warning.** Calibrated against 300 simulated white-noise
  series: at a nominal 5% it rejects 7.7% of the time, so the operational threshold is 1%
  and a marginal p-value is reported as *no cycle*, with the calibration stated on the
  card.
- **The trend is a choice until proven otherwise.** `compare` overlays the trends and
  quantifies their disagreement (21.10 on airline passengers, comparable to the series'
  own variation). Where methods without a remainder (HP, band-pass) have no F_T, the
  cell says *not computed* rather than 0.000, which would read as a finding.
- **Tests run where their assumptions hold.** Mann–Kendall, Sen's slope and Cox–Stuart
  run on the seasonally adjusted series — never on the smoothed trend, which would
  produce a spuriously tiny p-value — and the periodogram runs on the detrended series,
  never the raw one.

**Step 6 — ACF/PACF and "what next" (Part 6).** Autocorrelation and partial
autocorrelation with **both** significance bands, an automatic reading of the shape, portmanteau
tests at several lags, a variance ratio, candidate model orders — and then a ranked list of actions
where every item can be previewed before it is applied.

The decisions that matter:

- **Two bands are drawn, and the correct one is labelled.** The ±1.96/√n band is the one everyone
  has seen; the Bartlett band (1 + 2·Σr²)/n is the one that stays valid once the series is
  autocorrelated, and on a persistent series they differ by a factor of three. Judging a trending
  series against the white-noise band declares almost every lag significant.
- **A shape is a hint, never a verdict.** A slowly decaying ACF is *consistent with* a unit root,
  a trend, long memory or a near-unit-root process — all four look identical. Every reading says
  so, and the correlogram is cross-checked against Step 5 rather than overriding it.
- **White noise is allowed to be white noise.** At 5% with 36 lags about two significant lags are
  expected by chance. The reader distinguishes those from a real cut-off by requiring a *contiguous
  run* from lag 1 — a count-based rule called pure noise "MA(2)-like", which is why it no longer
  does.
- **`d` and `D` come from Step 5, never from the plot.** When the tests have not confirmed a unit
  root, the order hints keep d = 0 and say that differencing would be the correlogram overruling
  the tests.
- **OCSB is read the right way round.** Its null *is* the seasonal unit root, so a non-rejection
  means a seasonal difference is needed. Airline passengers therefore gets "take a seasonal
  difference — not a first difference", which is what Step 5's own verdict text says; recommending
  `diff(1)` would have made two panels contradict each other.
- **Every action is ranked against doing nothing.** `next.none.already_adequate` is a real
  recommendation with a real reason, and on white noise it ranks first: "no transformation is
  needed, go to modelling".
- **The preview is mandatory and can veto.** Each candidate is applied to a copy and re-diagnosed.
  A first difference of white noise comes back **worse** (lag 1 falls to −0.41, the spread grows
  from 7.4 to 10.5) and the Apply button disables itself with the reason on screen. A log transform
  is judged on the level-to-spread link, not on the ACF — judging a variance-stabilising transform
  by its correlogram would mark a correct transform as a failure.
- **CCF is pre-whitened, and the direction is spelled out.** Two independent random walks correlate
  at every lag; pre-whitening collapses that, and the peak is reported as "input leads by 3
  observations" rather than "lag = +3". A sign error here is invisible without the sentence.

---

## Step 8 — the readiness gate (Part 7)

Before any model is fitted, one screen answers one question: **should a model be built on this
series, and what has to travel with it if one is?** The answer is a single card — `READY`,
`READY WITH CAVEATS`, `NOT READY` or `INCONCLUSIVE` — and under it every criterion that produced it,
**with its number**.

**A check cannot pass by not running.** This is the rule the stage exists to enforce, and it is why
`Criterion` has four states rather than two: `pass | fail | unknown | not_applicable`, with `passed`
returning `None` for the last two. Three checks were caught passing vacuously during the build and
fixed:

| check | the vacuous pass | the fix |
|---|---|---|
| structural breaks | "no break detected" when no break test had run | `breaks_tested` recorded; unrun → *not measured* |
| gaps, outliers | "no missing values / no outliers" before Step 1 profiled anything | `profile_ran` recorded; unprofiled → *not measured* |
| seasonality | "weak seasonality" when the seasonal strength had never been computed | unmeasured → *not measured*, pointing at Step 4 |

**Unknown outranks caveat.** Any blocker → `NOT READY`. Otherwise any unassessed check →
`INCONCLUSIVE`. Otherwise any caveat → `READY WITH CAVEATS`. Otherwise `READY`. The middle ordering
is the deliberate one: a caveat list is only complete *for the checks that ran*, so calling a series
"ready with caveats" while stationarity is untested is a false assurance. `INCONCLUSIVE` names the
stages still to run and still lists the caveats that are already true.

**A horizon is a question, not a gap.** An unset horizon cannot be too long, so it is
`not_applicable`, not `unknown` — otherwise every series would open as inconclusive for not having
planned a forecast yet. Set it here (Step 8) and the n/5 warning from §5.11 arrives *before* the
forecast, while the plan can still change: on AirPassengers, H = 12 passes (the sample is 12.0
horizons long) and H = 60 fails (2.4).

**Caveats travel.** Every failed check, plus the stationarity verdict, the differencing decision, any
modelled break, any imputation and any over-long horizon, is written into `session.state.caveats` —
the same list the report reads — as a sentence that stands alone with no checklist in sight. The
gate records which lines it contributed and withdraws them on the next run, so changing the horizon
from 60 to 12 removes the horizon-60 warning instead of accumulating both.

**The agreement count respects each test's null.** ADF rejecting is evidence *for* stationarity;
KPSS rejecting is evidence *against*. A raw count of rejections scores them backwards. A *seasonal*
verdict counts only the seasonal witnesses and reports the regular battery separately ("seasonal
unit root (2 seasonal witnesses)"), because adding them together produces one impressive number
describing two different claims. A verdict that no test supports at all is a **blocker**, not a
caveat.

**Staleness is visible.** A stored verdict records the series version it described. Apply anything in
Step 2 and the card says so, names both versions, and refuses to present the old verdict as current.

Verified in the running app (screenshots `60_gate_*`): AirPassengers after S1/S4/S5/S6 →
`READY WITH CAVEATS` (7 pass, outliers and variance fail, 4 travelling caveats); nothing run yet →
`INCONCLUSIVE`, 7 checks *not measured*; white noise → `READY`; `electricity_daily` → `NOT READY`
(the unit-root tests conflict, 3 against 1); `structural_break` → ready with the break caveat.

---

## Step 9–10 — models: the registry, the baselines and the ARMA family (Part 8)

**A model is offered because a measurement says so.** The registry is *data*, not code: 21 rows,
each declaring the data classes it accepts, the minimum length it needs, the checks it has to pass
and the trigger that puts it in the list. `suggest` filters those rows against what Steps 1–7
measured, and every offer carries the number behind it — "lag-1 autocorrelation is 0.96, so the last
value carries real information" is a reason; "ARIMA is popular" is not. Every candidate also carries
**what would make it win**, so the list can be argued with instead of accepted.

On AirPassengers (`n = 144`, period 12, `d = 0`, `D = 1`) the registry returns **9 candidates and 12
exclusions**: `mean, naive, seasonal_naive, drift, white_noise, ets, theta, sarima, arima`. The
exclusions are not silent — the count families because the data is continuous, the intermittent
families because **0 of 144 values are zero**, GARCH because it is planned for part 11 and there is
no fitting code behind the row today.

**The baselines are not optional and not ranked.** Mean, naive, seasonal naive, drift and white noise
are fitted on every run whether the user ticks them or not, and they carry **no information
criterion** — `aic = None`, `ic_comparable = False`. An AIC on a model with zero estimated parameters
is not a small number, it is a category error, and putting it in the same sorted column as ETS's 699.51
would invite exactly the wrong comparison. Their interval width is the `√h` random-walk width, which
is the honest one: the naive forecast's uncertainty grows like a random walk, not like its in-sample
residuals.

**A failed fit is a row, never a missing row.** `fit_one` cannot raise. A SARIMA search that finds
nothing, a count model on data with no zeros, a series too short for two seasonal cycles — each comes
back with `failed = True` and one plain sentence, and the comparison table shows it in place. The one
thing worse than a model that did not fit is a model that quietly vanished from a table of nine.

**Comparability is a property of the number, not of the table.** Models are ranked by AICc, and the
rank only ever contains fits that were estimated on the same differenced series. Theta, ARFIMA,
Croston and the hurdle model report **no IC** — Theta's "fit" is an algebraic identity
(`0.5·L + 0.5·(2Z − L) = Z`), so its in-sample residuals reproduce the series and its MASE would be
exactly 0.0; it is withheld on purpose and the caveat says so. Such models are listed after the IC
rank, ordered by parameter count, and never interleaved with a sorted AICc column.

### What the fits say on AirPassengers

| # | model | AICc | BIC | k | MASE | Ljung–Box p |
|---|---|---|---|---|---|---|
| 1 | **ETS(A,N,M)** | **699.51** | 737.83 | 14 | **0.237** | 3.8e-06 |
| 2 | SARIMA (1,0,2)(0,1,1)₁₂ | 925.92 | 943.12 | 6 | 0.346 | 0.150 |
| 3 | ARIMA (2,0,3) | 1351.50 | 1374.19 | 8 | 0.731 | 0.001 |
| 4 | Theta (θ = 2, season 12) | — | — | 2 | — | — |
| 5–9 | drift · naive · seasonal naive · mean · white noise | — | — | 1 | 0.796 · 0.802 · 0.944 · 3.136 · 3.136 | 0.000 |

The gap is the story: the multiplicative-error ETS is **226 AICc units** ahead of the best SARIMA, so
choosing between them is not a coin flip — and both clear the seasonal-naive benchmark (MASE 0.237 and
0.346 against 0.944), which is the only comparison that answers "did the model earn its complexity?".
Full fit of 9 models takes **3.8 s**; the table is stored, so reloading the page does not re-fit.

### Decisions worth knowing

- **`d` and `D` are inherited, not searched.** They come from the Step 5 verdict (here `D = 1` from
  the seasonal unit root) and are passed into every ARIMA/SARIMA/ARFIMA fit. A model that re-estimates
  differencing on its own would silently answer a different question from the one the gate approved.
- **SARIMA refuses fewer than two cycles.** With `n < 2m` a seasonal AR/MA term is not identified, and
  the search converges on nonsense rather than failing. It now says "SARIMA needs at least two full
  cycles at period 12 (24 observations) and there are 20", which is actionable.
- **The order grid is bounded by sample size**, not fixed at 3×3×2×2: `n < 200` gives p,q ≤ 2 and
  P,Q ≤ 1, and `p+q+P+Q ≤ 4`. The unbounded grid was 134 specs, took 88 s, and selected a parameter
  sitting on the unit-root boundary — slower *and* wrong.
- **Root checks report the smallest modulus**, for both AR and MA, against a `1e-4` boundary band:
  "every AR root lies outside the unit circle (minimum modulus 1.284)" or the failure, naming which
  polynomial. Boundary-adjacent estimates are marked in the parameter table rather than passed.
- **ARFIMA's short-memory part is fitted mean-zero**, or integration turns the constant into
  polynomial growth. `_frac_integrate(fractional_difference(x, 0.25), 0.25)` reproduces `x` to
  **0.0** on 200 held-back points, which is the check that proves the filter is causal and exact.
- **Counts get integer intervals that bracket their own mean**, via `np.maximum(draw, 1)` for a
  truncated-Poisson draw, not `draw + 1` (which shifts the whole distribution). MAPE is refused
  outright on count and intermittent data: "a percentage error is undefined at zero and misleading
  near it".
- **Zero-inflation is judged against the conditional rate**, not the pooled mean. A series whose rate
  dips to λ ≈ 0.5 produces 60% zeros legitimately; using the pooled mean as the reference would call
  every low-rate series a hurdle model.
- **Caveats travel into the fit and out again.** The 25 the gate recorded still sit in the sidebar;
  each fit adds its own on top — ETS carries its symmetric-interval approximation and the
  **16.1-AICc gap** that leaves the seasonal component choice ambiguous ("the second-best
  specification (add/mul) is within 16.1 AICc units, so the component choice is not decisive"). The
  ETS row's own caveat is the difference between a model and a model with a stated limit.

**Verified in the running app** (screenshots `61_models_*`–`65_models_*`): S9 builds the candidate
list and shows the reason plus the win condition for each; S10 fits all 9 models, ranks them, expands
ETS's parameter table, lists the 12 exclusions with their measurements, and shows the same numbers
after a page reload. **74 new tests** (`tests/test_part8_models.py`), suite total **494**.

---

## Step 9–10 (continued) — structural, exogenous and nonlinear models (Part 9)

Part 8 shipped nine fittable models. Part 9 adds four more, and the way they were added is the
interesting part: **no code outside the new files had to learn their names.** `ucm`, `arimax`, `ardl`
and `markov_switching` were pending registry rows whose `feature_flag` flipped, plus one `IMPLEMENTED_*`
dict per module. The suggestion engine, the comparison table, the ranking, the failure path and the
staleness logic all picked them up unchanged — which is the claim Part 8 made and this part tests.

| model | what it answers | why it is not just another ARIMA |
|---|---|---|
| **UCM** (local level / local linear trend / BSM) | "what is the underlying level, and how sure are you?" | It reports the *components* as output. Missing values are handled by the Kalman filter skipping the update — not by imputing first. |
| **SARIMAX / ARIMAX** | "how much does this driver move the target?" | Takes a real outside column, plus pulse/step dummies built from Step 2's flags, plus future scenario paths. |
| **ARDL** | "is there a long-run relationship at all?" | Pesaran–Shin–Smith bounds test, valid whether the regressor is I(0) or I(1). |
| **Markov-switching AR** | "were some periods behaving differently?" | Two regimes with their own mean, variance and persistence, and a labelled history you can check against what happened. |

### Results on the bundled promotions dataset (n = 108, driver = promo days)

The dataset is new in this part and exists because a model that uses an outside column cannot be
tested on a dataset with one column. A month's units are driven by its promotion days at **+88 units
per day**, which is larger than the trend and the noise.

| # | model | AICc | k | MASE |
|---|---|---|---|---|
| 1 | **ARDL(2,2) — y on PromoDays** | **1028.92** | 6 | **0.092** |
| 2 | SARIMAX (1,0,0) + 1 regressor | 1038.26 | 4 | 0.110 |
| 3 | ETS(A,N,M) | 1192.28 | 14 | 0.729 |
| 4 | SARIMA (0,0,2)(0,0,1)₁₂ | 1289.10 | 6 | 0.851 |
| 5 | ARIMA (0,0,3) | 1442.74 | 6 | 0.848 |
| 6 | UCM — local level | 1544.64 | 3 | 0.948 |
| 7 | Theta(θ=2, season 12) | — | 2 | — |
| 8–12 | drift · mean · naive · seasonal naive · white noise | — | 1 | 1.046 · 0.824 · 1.046 · 0.992 · 0.824 |

Both exogenous models recover the generating coefficient: ARDL estimates **+87.33 per promotion day
(SE 1.1)**, SARIMAX **+86.95 (SE 0.84)**, against a true 88. The best model that ignores the driver
(ETS) is **164 AICc units** behind and has a MASE nine times worse — which is what a driver model is
supposed to look like when the driver is real.

### Five defects this part found, and what they were

1. **The state-space model had eleven free parameters it could not identify.** Giving each of the
   m−1 seasonal states its own variance makes the likelihood flat in most directions: the fit still
   "converged", in 55 s, having estimated nothing. One variance per *component* (level, trend,
   season) is the Harvey specification, and the search went from 55 s to 8 s. Switching from
   Nelder–Mead to L-BFGS-B with three starts took it to 3.5 s for the *same* optimum (586.47).
2. **The scenario path was recorded and then ignored.** `POST /analysis/scenario` stored the future
   driver values, and `fit/forecast` read the wrapper's fit-time copy — so the forecast did not move
   when the scenario changed, and nothing said so. Now the forecast path is read at predict time:
   six promotion days lifts the SARIMAX forecast from **566.0 to 1087.7**, and the response carries
   either "conditional on the supplied paths" or the sentence saying how many drivers were held flat.
3. **The bounds test briefly had a p-value that contradicted its own band.** `stats.f.sf(F, 2, dof)`
   gave p = 0.049 — "significant at 5%" — on the same fit where F = 3.114 sits *below* the tabulated
   lower bound of 4.94. The bounds test exists precisely because the ordinary F distribution is
   invalid here, so there is now **no p-value at all**, and the caveat says why. The statistic is
   cross-checked in the tests against a hand-built error-correction Wald test (3.1138 vs 3.1137).
4. **The regime detector passed every flat series.** The naive statistics — "most variable window ÷
   median window" — sits at about **1.6 on white noise**, because a maximum over n overlapping windows
   is biased upward. Any threshold near 1.15 would have offered a Markov-switching model for a random
   walk. The statistic is now calibrated against a simulated AR(1) null of the same length and
   persistence, so 1.0 means "exactly as lumpy as noise this shape", and it is measured on the
   **detrended** series: real regimes score 2.16, white noise 0.95–1.17, and this build's promotion
   dataset 0.77 — excluded, with both numbers in the reason.
5. **`EdaOutput` had no `outliers` field, and `getattr(...) or []` hid it.** The intervention dummies
   were being built from an always-empty list, so the "pulse from Step 2's flags" path was dead code
   that looked alive. The field exists now, it is carried out of the pipeline, and AirPassengers
   yields `[{'time': '1960-03-01', 'kind': 'spike'}]`.

### Refusals that name their number

- A multiplicative ETS on data containing a zero is *not legal*, so it is never in the grid — the fit
  says "6 legal specifications" and picks an additive one.
- SARIMAX with no outside column: "it needs at least 1 outside (exogenous) column and this dataset has 0".
- ARDL on a promotions dataset is offered; on AirPassengers it is excluded, with that sentence.
- A two-regime model where the second regime owns 2.6% of the sample: "not enough to estimate a
  separate mean, variance and persistence" — and it points at Step 2's outlier panel instead.
- A Kalman structural model reports `n_obs_used = 86` for a series with 10 missing values, and its
  residuals are `NaN` at those points rather than fabricated zeros.

**Verified in the running app** (screenshots `70_part9_*`–`73_part9_*`): the promotions dataset loads
with its driver, the gate returns `READY`, S9 offers 12 candidates (including UCM, ARDL, SARIMAX and
a correctly-excluded Markov switch), S10 fits all 12 and ranks the two driver models first.
**49 new tests** (`tests/test_part9_models.py`), suite total **584** at the end of part 10.

---

## Step 9–10 (continued) — count dependence, intermittent demand and event timestamps (Part 10)

Parts 8 and 9 covered series measured on a continuous scale. Part 10 covers the three data shapes a
continuous branch cannot honestly reach, and the reason each one needs its own machinery:

| shape | what the ARIMA/ETS branch gets wrong | what this part adds |
|---|---|---|
| **counts with serial dependence** | Treats a rate as a function of the calendar. When an outbreak month is followed by more outbreak months, the residuals stay autocorrelated and the intervals are too narrow. | INGARCH(1,1) and INAR(1), fitted by conditional maximum likelihood. |
| **intermittent demand** | Forecasts a non-zero mean in every empty period, so the errors are dominated by phantom demand. | ADIDA (aggregate–disaggregate) and Croston with a conditional SY correction. |
| **event timestamps** | There is no value column to model — only *when* things happened. | Renewal (exponential vs Weibull by AIC) and Hawkes (self-exciting, exponential kernel with a branching ratio). |

### The discrete predictive distribution is now an object

`ts/models/predictive.py` is the part of this build that Part 15 depends on. A count model does
not produce "a number and a band"; it produces a **PMF per future period**, and every derived quantity
reads that same array:

| quantity | how it is computed | why not the obvious alternative |
|---|---|---|
| interval | exact CDF inversion on the integer support | Interpolating between integers invents outcomes a count cannot take. |
| **CRPS** | the ranked-probability sum, Σ(F_k − 1{y ≤ x_k})² | The continuous CRPS formula on a PMF reports a different quantity. |
| **log score** | −log P(Y = y), finite stand-in outside the grid | +∞ would poison an average; 0 would flatter a model that assigned no mass. |
| **PIT** | F(y−1) + U·P(Y = y), U seeded from (h, y) | Without randomisation a discrete PIT is a staircase and rejects a *correct* model. Recorded (h, y) → the plot is reproducible. |
| **P(≥ 1)** | tail mass directly | The stockout question, answered rather than approximated. |

`DISCRETE_PREDICTIVE` in `ts/models/fit.py` lists the seven models whose predictive is a PMF
(`poisson`, `neg_binomial`, `zero_inflated`, `ingarch`, `inar`, `renewal`, `hawkes`), and the forecast
endpoint now ships the array (`"discrete": true`, `predictive: {support, pmf, ...}`). The Poisson, NB2
and hurdle models were retrofitted to build and keep their PMF instead of taking a scipy quantile and
throwing the distribution away; the hurdle's simulated interval was replaced by the exact two-part
mixture, since the tail of a truncated Poisson is exactly where simulation is thinnest.

### Results on the bundled datasets

**`outbreak_counts.csv`** (60 monthly counts, mean 11.97, variance/mean 1.97, lag-1 autocorrelation
**0.38**):

| # | model | AICc | k | MASE | Ljung–Box p |
|---|---|---|---|---|---|
| 1 | **INGARCH(1,1) — NB2** | **346.19** | 4 | 0.842 | 0.442 |
| 2 | Negative binomial (NB2) | 434.12 | 7 | 0.600 | 0.438 |
| 3 | INAR(1) | 360.66 | 2 | 0.829 | 0.202 |
| 4–8 | drift · mean · naive · seasonal naive · white noise | — | 1 | 1.042 · 0.908 · 1.041 · 0.917 · 0.908 | |

INGARCH wins on AICc by **88 units** over the NB2 regression it generalises, and its Ljung–Box p rises
to 0.442 — the residual dependence the Poisson GLM leaves behind is gone. NB2 keeps the better MASE
(0.60), and the comparison table says so rather than hiding it: AICc ranks explanation, MASE ranks
forecast, and the two are allowed to disagree.

**`spare_parts.csv`** (96 months, 71 zeros):

| # | model | AICc | k | MASE | plain answer |
|---|---|---|---|---|---|
| 1 | ADIDA, bucket 4 | — | 3 | 0.956 | aggregates 4 periods until demand is ≈ once per block, then divides back to **1.872/period** |
| 2 | Croston (SBA) | — | 3 | **0.739** | 5.6 per event every 3.2 periods |
| 3 | Croston (SY-corrected) | — | 3 | 0.888 | mean interval **3.84 > 1.32**, so the correction applies: **1.670** vs 1.758 uncorrected |
| 4 | TSB | — | 3 | 1.031 | occurrence probability decays when demand stops |

**`aftershocks.csv`** (456 event timestamps, no value column — new in this part) and
**`machine_service.csv`** (95 timestamps, near-regular gaps — the control):

| dataset | renewal AICc | Hawkes AICc | Hawkes branching ratio | winner |
|---|---|---|---|---|
| aftershocks | 824.70 | **415.77** | **85.5% self-generated** | Hawkes |
| machine service | **390.53** | 609.27 | **0.0%** | renewal |

The control exists because "Hawkes finds self-excitation" is worthless if it finds it everywhere. On the
regular series the branching ratio collapses to zero and the extra kernel costs 219 AICc units.

Both models also agree with the data on the period scale: the aftershock file is **11.4 events per
month** on average and renewal forecasts 11.59, Hawkes 11.92; the service file averages 3.39 per month
and the two forecast 3.43 and 3.46. The renewal row's Weibull shape of **0.452** says the hazard is
*decreasing* — waiting longer makes the next event less likely — which is the signature of clustering,
and the reason Hawkes wins by 409 units.

### Recoveries, because convergence is not evidence

| model | simulated truth | recovered |
|---|---|---|
| INGARCH | α 0.35, β 0.55, ω 2.0, n = 240 | α **0.308**, β **0.590**, persistence **0.897** (true 0.90) |
| INGARCH | α 0.30, β 0.40, n = 2000 | α 0.306, β 0.430, persistence **0.736** (true 0.70) |
| INGARCH | α 0.15, β 0.20, n = 2000 | α 0.166, β 0.151, persistence **0.317** (true 0.35) |
| INAR | α 0.50, λ 8.0, n = 500 (thinning simulation) | α **0.489**, λ **8.204** |
| Renewal | Weibull k = 0.7 | shape **0.87**, hazard decreasing |
| Hawkes | μ 0.05, α 0.6, β 1.0 | branching **0.604** |
| Hawkes | μ 0.02, α 0.8, β 2.0 | branching **0.403** |

### Six defects this part found

1. **INGARCH reported a constant dressed as an estimate.** The persistence α+β was reparameterised as a
   *sum-to-3 softmax*, so α+β came back as exactly **0.970 for every series** — simulated 0.90, 0.70,
   0.50, all 0.970, with ω silently absorbing the difference. The reparameterisation is now a logistic
   for persistence and a logistic for the α share, and the recovery table above is the evidence it can
   be wrong.
2. **The renewal forecast was cumulative, so a flat rate climbed like a ramp.** Step h multiplied the
   per-period rate by h, producing 3.4 → 6.8 → 10.3 events per period when the fitted rate was 3.4.
3. **The renewal and Hawkes fitted values were in event counts while the series was in periods,** and
   MASE compared the two: the renewal row scored **48.08** against a benchmark of 1.0. Both now report
   expected events *per period*; the same two models now score 0.93 and 0.94.
4. **`_period_days()` inferred the calendar from the gap between events.** For a weekly series that is
   ~7 days per event, not 7 days per period, which happens to coincide only on purpose. The fit
   orchestrator now passes `days_per_period(frequency_code)` and the wrapper uses it first.
5. **Same-day events were dropped from the interval set.** `inter_arrivals` discarded zero gaps, which
   on day-resolution data means discarding every pair of events on the same day: 456 aftershocks give a
   mean gap of 7.2 days if those are dropped and 1.7 if kept — a 4× bias in the estimated rate. Zero
   gaps are now kept (a floor of one second keeps the kernel finite) and negative ones are not.
6. **Day-resolution event data made the point-process likelihood meaningless.** 289 of 455 intervals in
   the first aftershock example were exactly zero, and the Weibull log-likelihood moved by **9000**
   between a one-second and a one-hour floor under those zeros (AICc −8155 vs +1863). A continuous-time
   kernel cannot separate events recorded on the same date, so the wrapper now refuses when more than
   20% of intervals are tied, quoting the count, and points at timestamps with times of day. The bundled
   example was regenerated with times of day, which is why its numbers above differ from an earlier run.
7. **A file of nothing but dates was a mapping error.** `ingest_text("date\n2021-01-01\n…")` raised
   "The value column you chose is not in the file", so the entire event branch was unreachable from an
   upload. There is now an `event_mode` on `ColumnMapping`, set automatically for a single date column,
   surfaced as a "— each row is one event —" option in the value picker, counted into the coarsest
   calendar that keeps ≥ 24 periods, and stored as `event_times` so the point-process models get
   intervals rather than counts. The value-model and event-model branches also now exclude each other
   by name, with the reason quoted.

Smaller ones: a single-column date file no longer trips the "constant run" warning; `suggest_mapping`
no longer returns a mapping with no value and no time column; the API's `_series()` rebuilds the count
series from the raw timestamps at the calendar the ingest layer chose, so the two layers cannot disagree.

### Refusals that name their number

- INGARCH on a continuous series: "it models counts that depend on earlier counts, and this series is
  continuous — there are no count values to recurse on."
- INAR/INGARCH on a series with lag-1 autocorrelation 0.06: "almost no serial dependence for the
  recursion to capture — the plain Poisson regression is simpler" (listed with the warning, not hidden).
- ADIDA with fewer than 4 blocks: refused, with the block count in the sentence.
- Hawkes refuses a branching ratio at or above 0.95: near-critical is not identified.
- Every value model on event data, and every point process on a value series, with the missing thing
  named rather than "incompatible".

**Verified in the running app** (screenshots `80_part10_*`–`95_part10_*`): the count dataset loads and
S9 offers INGARCH and INAR as `advanced` rows with their measured reasons; the intermittent dataset
offers ADIDA and Croston-SY; the aftershock file loads as `event timestamp` and offers exactly two
candidates — renewal and Hawkes — with 27 rows excluded; the control dataset picks renewal and Hawkes
loses (renewal 390.53 vs Hawkes 609.27); a one-column date CSV uploaded by hand loads as 260 events over
70 monthly periods and shows the right preview. The shots are `80_part10_01_counts_loaded` through
`95_part10_16_event_upload`. **42 new tests** (`tests/test_part10_models.py`), suite total **585**.

---

## Step 9–10 (continued) — conditional variance (Part 11)

Parts 8–10 all model *where the series is going* and assume the noise has a constant spread. A great
many series break that second half while obeying the first: the level behaves like a random walk with
drift and the noise arrives in clusters — calm stretches, then violent ones. Fitting a constant-variance
model to that gives correctly placed point forecasts and **intervals that are too narrow when it
matters and too wide when it doesn't**, which is the worst combination for anybody using them.

Eight rows are fitted (`arch`, `garch`, `garch_t`, `gjr`, `gjr_t`, `egarch`, `egarch_t`,
`arima_garch`); two are registered and refused with a reason (`sv` needs a Bayesian layer, `realized`
needs an intraday panel the app does not ingest). `arch` is not a dependency — the family is hand-written
because `arch` is not installable here and `statsmodels.tsa.arch` was removed in 0.15.

### Two forecasts, not one

§8 asks for a mean path *and* a variance path. The S10 comparison table now grows a **variance path**
button on every fitted volatility row, and the card it opens (`98_part11_03_variance_path`) is the
deliverable:

```
GARCH(1,1)-t — t          persistence 0.973
Volatility now → at step 60:   0.538 → 0.340   long run 0.272
"The sample ends more turbulent than its own long run, so the path falls toward 0.272 as the
 shock fades. A constant-variance model would report one width for all 60 steps — too wide now,
 too narrow if the calm returns."
```

The path is the fitted recursion iterated forward, and it is **not** `σ̄²·h`: the term structure
converges at a rate set by persistence, and at h = 60 the difference is 0.34 vs 0.27. `/analysis/fit/forecast`
ships `variance_path: {variance, sd, unconditional_sd, persistence, leverage}` alongside the mean path, so
Part 16 draws the chart and Part 18 can cite the numbers without recomputing them.

### Two bugs in the forecast recursion, both found by the tests

| bug | what it did | fix |
|---|---|---|
| the h ≥ 2 expectation form applied at h = 1 | The GJR one-step forecast used ½γ instead of the *realised* sign of the last shock, so "tomorrow" disagreed with the filter's own last value. | h = 1 uses the realised shock and its sign; only h ≥ 2 integrates the sign out. `test_one_step_gjr_uses_the_realised_sign_not_the_expectation` asserts both. |
| EGARCH overflowed after a 1000σ shock | `exp(log-variance)` recursed past the float limit and the path became `inf` at step 14. | The log recursion is clipped to ±30 at every step; the path stays finite and positive (`test_huge_shock_never_produces_a_negative_variance`). |
| in-sample fitted values were `const + residual` | That is identically `y`, so every residual metric collapsed to zero and the S10 table printed **MASE = 2.9e-18** — a perfect fit claimed for a model whose whole point is that the *value* is unpredictable. | `_fitted` is the conditional mean (a constant, or the AR(1) one-step path). MASE now reads **0.724**, level with the mean baseline, which is the honest number (`test_in_sample_fitted_values_are_the_conditional_mean_not_the_observation`). |

### The ARCH-LM gate, and why the obvious object is the wrong one

The spec's trigger is "ARCH-LM rejects on the **residuals**". Two simpler readings are both wrong, and
both were tried before the third:

| object tested | result | why it fails |
|---|---|---|
| the raw **level** | rejects on every random walk | the *mean* is moving, not the variance |
| the **first difference**, always | rejects on **96% of 300 white-noise draws** | differencing a stationary series induces an MA(1) root, and the *squares* of an MA(1) are autocorrelated by construction |
| after the differencing **S5 chose** (`d_hint`) | correct | this is the series the mean model is actually fitted to |

A gate that passes 96% of pure noise is worse than no gate, because it looks like evidence. The facts
layer now records `arch_lm_resid_p` (the one the registry reads) and `arch_lm_level_p` (quoted in the
refusal so the reader can see what would have fooled them). On the flat white-noise dataset all nine
volatility rows are excluded with *"ARCH-LM on the mean-adjusted series returns p = 0.895, which is no
evidence of it"*.

### Recoveries, because convergence is not evidence

Simulated GARCH with a known truth, n = 2000:

| truth | recovered persistence | AICc verdict |
|---|---|---|
| α = 0.08, β = 0.85 (0.93) | 0.946 | — |
| α = 0.15, β = 0.70 (0.85) | 0.870 | — |
| α = 0.08, β = 0.80 with **t(5)** innovations | 0.731 Gaussian / 0.779 GARCH | **t wins by 161 AICc**; fitted ν = 5.15 |
| the same with **Gaussian** innovations | — | **Gaussian wins by 3 AICc** — the t version is not a free win |
| symmetric GARCH, GJR fitted | γ̂ = 0.006 | **no leverage story told** (threshold 0.02) |
| GJR with γ = 0.09, t(6) | γ̂ > 0 | leverage caveat printed, and the sign convention differs for EGARCH (γ < 0) |

### One defect this part found outside the family

Weekly data labelled on Fridays reindexed onto a `W` (Sunday-anchored) calendar produced a series of
**nothing but NaN** — and the only message upstream was *"the chosen value column is empty after
cleaning"*, which points at the file rather than at the calendar. `timeindex.regularise` now refuses when
a calendar matches **zero** observations, and says why: *"none of the 1,626 timestamps fall on the weekly
calendar anchored at 1995-09-10 (step 7 days); the observations are offset from the calendar grid."*

**Verified in the running app** (screenshots `97_part11_01_fit_table` · `98_part11_02_variance_path` ·
`99_part11_03_volatility_refused`): the bundled
`volatile_returns` series loads as 741 clean weekly values, S9 offers GARCH and GARCH-t with the measured
ARCH-LM reason, S10 ranks GARCH-t (1044.49) above GARCH (1055.11), the variance-path card renders the
decaying term structure, and the white-noise dataset refuses all nine variance rows with the p-value in
the sentence, with 28 rows excluded in total and every reason carrying its number. **49 new tests** (`tests/test_part11_volatility.py`), suite total **634**.

---

## Step 9–10 (continued) — joint series, regime changes and machine learning (Part 12)

Every model so far answered a question about **one** column. This part takes on the three rows that
cannot be written as `y` on its own lags: a *system* of series that move together (`var`, `vecm`), a
single series whose law changes partway through (`markov_switching`, `setar`), and the tabular learners
(`gradient_boost`, `random_forest`, `knn`, `svr`, `gaussian_process`, `nnar`) that make no statistical
statement at all and therefore need a different kind of card.

Six rows are fitted (`var`, `vecm`, `markov_switching`, `setar`, plus the six ML rows); three are
registered and refused by a **measured** reason rather than a schedule — `lightgbm`, `xgboost` and
`lstm` are missing their libraries (`import lightgbm` → `ModuleNotFoundError`, quoted verbatim in the
UI), and `combination` was Part 13's row. The registry is now **48 rows / 42 implemented**.

### A joint model needs a second dataset, not a second model

The bundled `traffic_spend` example (n = 180) has a target and a `MarketingSpend` driver that are
cointegrated. Fitting a univariate model to the target throws the long-run relationship away; fitting
the pair as a system keeps it. On the same session:

| row | AICc | BIC | k | MASE | Ljung–Box p |
|---|---|---|---|---|---|
| VAR (7 lags, levels) | **1897.26** | 1978.76 | 30 | **0.270** | 0.185 |
| VECM (rank 1) | 2073.91 | 2104.41 | 10 | 0.459 | 0.000 |
| Gaussian process | — | — | 0 | 0.286 | 0.154 |
| Gradient-boosted trees | — | — | 200 | 0.842 | 0.000 |
| random walk with drift | — | — | 1 | 0.488 | 0.000 |
| naive | — | — | 1 | 0.492 | 0.000 |
| seasonal naive | — | — | 1 | 1.002 | 0.000 |
| mean / white noise | — | — | 1 | 3.776 | 0.000 |

The interesting result is that **the VAR beats its own benchmarks and the ML rows do not**. A
gradient-boosted forest at MASE 0.842 is worse than a random walk with drift at 0.488 on a series this
noisy and this short, which is exactly what the model card is for: the numbers are printed, not hidden.

### The joint card, and the assumption it prints

`POST /analysis/fit/multivariate` returns the Granger table, the impulse responses, the forecast-error
variance decomposition and — for a VECM — the adjustment coefficients. The card
(`101_part12_02_multivariate`) leads with the thing that is *not* a finding:

> Fitted to levels, 7 lag(s) over 2 series. Ordering `the target → MarketingSpend`: the first series is
> allowed to affect the others within the same period, the last is not. That is an assumption, not a
> finding, and it changes the decomposition below.

Measured on `traffic_spend`:

| question | answer |
|---|---|
| Does MarketingSpend help predict the target? | F = 5.84, p = 0.000 (p on differences 0.001 — it survives differencing, so it is not the trend) |
| Does the target help predict MarketingSpend? | F = 14.50, p = 0.000 (differences 0.000) |
| Share of the target's 12-step error from itself / from spend | 82.5% / 17.5% |
| Share of spend's 12-step error from the target / from itself | 48.1% / 51.9% |
| Largest companion root | 0.998 — close to 1, so the response decays slowly and the bands stay wide |
| Johansen trace | 94.90 vs critical 12.53 → rank 1, which is what makes a levels VAR defensible |
| VECM β / α | β = [1.000, −1.101]; α_target = −0.140, α_spend = +0.257 |

The impulse responses carry a **90% band from 200 bootstrap draws** (seed 20260101), and the band is
drawn, not implied — the chart was silently empty for two rounds before the bug below was found.

### Four defects this part found, three of them in the UI

| bug | what it did | fix |
|---|---|---|
| the IRF chart drew nothing | `const n = paths[0].length` took the **series count** (2) where the **step count** (12) was meant, so every x-coordinate ran to 11,000 and the SVG clipped the path away. An empty panel, no error, no warning. | extent from `paths.length`; the chart now has a regression note in the comment |
| an SVG with no `viewBox` clips silently | even with the right coordinates, `width="100%"` on an SVG without a `viewBox` keeps the *attribute* coordinate system, so a 0–1000 drawing fell outside a 998px box. | `viewBox="0 0 1000 62"` + `preserveAspectRatio="none"` |
| the band was typed flat, the payload is 2-D | `bands_90[series].lower` is *step × series*; typing it `number[]` made `y(row[j])` `NaN`, so the polygon and the line both vanished. | `IrfBand.lower: number[][]`, and the band is indexed by its own response column |
| VECM reported MASE **—** | `VECMModel` never set `_fitted`, so the table said "not measured" for a row that was fitted. The offline check had it at 0.459; only the *running* server didn't, because the process predated the fix. | `self._fitted = _in_sample_fit(data, B)`; the table now reads **0.459** |
| a shipped part quoted as a future plan | `lightgbm` said *"planned for part 12"* **during part 12**, and `sv`/`realized` said *"planned for part 11"* after Part 11 shipped. | the measured import failure wins over the schedule; a new test fails if any pending row names a shipped part |

### Regimes: a threshold has to be supported, not merely searched for

A threshold model fitted by scanning a grid will always find *some* split that looks good, so the
question is whether the split is better than the best split of a series with no regimes. The naive
threshold F-test is **invalid** and this part refuses to quote it: on a linear AR(2) it returned
p = 0.0085 and would have announced a regime change that does not exist. SETAR now uses a
**199-replicate sup-F bootstrap over the full (delay, threshold) grid**, and refuses below p = 0.05:

| series | naive F-test | sup-F bootstrap | verdict |
|---|---|---|---|
| linear AR(2), seed 80 | p = 0.0085 | **p = 0.130** | refused — correct |
| three simulated SETARs | — | p = 0.005 – 0.030 | thresholds recovered within 0.4% of truth |
| 20 white-noise draws | — | 1 false positive | 5%, as advertised |
| `traffic_spend` | — | not supported | refused, with the p-value in the sentence |

Randomising the grid *inside* the replicates was tried first and biases sup-F downward — it compares a
searched maximum against unsearched nulls — so the bootstrap uses the same full grid as the fit.

### Machine learning, with the card §5.8 F demands

The six ML rows are ranked by **measured** MASE from a recursive out-of-sample scheme, carry
`ic_comparable = false` and print **no** AIC, because an in-sample information criterion on a
gradient-boosted forest is a number about the forest, not about the forecast. Each row opens a card:

> **Model card.** 14 features, 8 lags, recursive multi-step. Validated by time-ordered holdout (last 34
> of 168 rows) plus 3 expanding folds, one-step horizon; never a shuffled split.
> Holdout MASE 1.301 over 34 rows · folds 1.972 / 1.016 · 90% band covered **71%** out of sample

The coverage line is the honest part: a nominal 90% band that covers 71% is disclosed rather than
quietly drawn. Feature construction is a **single function** shared by training and forecasting (`build_design`
/ `feature_row`), because two copies drift and the drift is a leak.

### Refusals that name their number

Every excluded row still appears, greyed, with the measurement that excluded it:

- *"The library it needs is not installed in this build: `lightgbm` — `import lightgbm` fails with
  `ModuleNotFoundError`… the gradient-boosted row above runs on scikit-learn instead."*
- *"Once its trend is removed, its most variable stretch is only 0.64× as variable as noise of this
  length and persistence would be, which is too little contrast for a second regime to attach to
  (needs 1.30…)."* — Markov-switching and SETAR, on `traffic_spend`.
- *"This build has no fitting code for it yet — it is planned for part 13."* — forecast combination.

**Verified in the running app** (screenshots `100_part12_01_fit_table` · `101_part12_02_multivariate` ·
`102_part12_03_model_card` · `103_part12_04_refusals`): `traffic_spend` loads as 180 rows, S9 offers 22
candidates with 26 excluded and every reason carrying its number, S10 ranks VAR first at AICc 1897.26 /
MASE 0.270 with VECM at **0.459** (not a dash), the joint card renders Granger 5.84 / 14.50, the FEVD
82.5 / 17.5 and two impulse responses with their bootstrap envelopes, the ML row opens its model card,
and `lightgbm` / `xgboost` / `lstm` are refused by their measured import failures. **57 new tests**
(`tests/test_part12_multivariate_ml.py`), suite total **692**.

### The optional AI key

Part 17 is fully usable without a key. A live Gemini request may use `GEMINI_API_KEY` from the
environment or a **gitignored `backend/.env`** (`chmod 600`), or a key typed into **Assistant
Settings** and kept only in that browser tab's `sessionStorage`. The typed key travels in the
`X-Gemini-Key` header for one request at a time; it is never placed in the payload, a response, a log,
chat history, `repr`, or server-side session. `has_gemini_key()` answers a boolean; the `gemini`
feature flag is `false` until both a key *and* `google-genai` are present. Offline mode needs neither.
The v4 install pins `google-genai>=1.0,<2` and installs it through
`backend/requirements-deploy.txt`; Assistant → Settings now reports whether the
SDK or key is missing and links directly from the first-use payload preview.
`test_the_ai_key_never_appears_in_a_response_or_a_repr` and the Part 17 tests pin those boundaries.

---

## Step 11–12 — selection and diagnostics (Part 13)

Part 12 ended with seven fitted models and a table sorted by AICc. That table answers the wrong
question: an information criterion rewards *fit to the sample the model was fitted on*, and the user
is about to ask the model about data it has never seen. Part 13 replaces the ranking with one measured
on held-out forecasts, and adds the panel that says whether the winner's residuals are noise.

Two new modules and two new screens: `ts/select/` (IC guard, rolling-origin CV, DM / Clark–West /
Mincer–Zarnowitz, the report) and `ts/diagnose/` (the residual battery, PIT/RQR, the adequacy verdict),
plus the `combination` row that Part 12 registered and could not fit. Registry: **48 rows / 43
implemented**.

### Rolling origin is the criterion, and the UI says so in those words

§5.9 requires the app to state that CV is primary and ICs are secondary. The panel leads with the
paragraph, not the table:

> **ets** — CV MASE 0.310 · 4 folds · h=12 · expanding
>
> ets came first on rolling-origin MASE across 4 rolling-origin folds at horizon 12, with a MASE of
> 0.310 (MAE 3.81, RMSE 4.59, fold spread 0.093). The next model, drift, scored 0.804 — +159.3% worse,
> and wider than the fold-to-fold spread. Diebold–Mariano on the paired loss differential gives
> p = 0.000, a real improvement rather than noise. ets fails the joint test (p = 0.0075): the slope is
> 0.897, below 1, so the forecast varies more than the outcome does — it overshoots the movements.

That last sentence is the point of the whole panel. ETS wins the comparison and **fails the
Mincer–Zarnowitz regression**: its forecast is unbiased and efficient in slope terms only at 0.897, so
it over-reacts to movements. A comparison table that only ranks models can never say that.

### Four defects this part found, three of them in code I had just written

| bug | what it did | fix |
|---|---|---|
| `fold_origins` dropped 18 of 28 requested folds | The sliding-window expression went negative and was clipped to 0, so the `min_train` check silently discarded folds — and "sliding" was silently "expanding". A fold that is quietly not built is worse than one that refuses. | window length derived from the earliest fold's size; the `step` parameter added so the combination can request overlapping origins deliberately |
| folds were numbered backwards | Fold 0 was the *latest* origin. A fold table that runs backwards is a table somebody misreads. | walk backwards from the end, then reverse |
| the combination was built from in-sample fits | Its first version averaged the whole benchmark ladder plus one model and scored **MASE 2.10** against a component at 0.25 — arithmetically impossible, and the tell that the fitted path was misaligned with the series. | components are the fitted models only (one benchmark allowed when there is a single real model); weights from rolling origins; `_fitted` aligned to the series |
| DM and MZ were silently omitted | The endpoint built a paired error map and then passed `errors_by_model=None`; the panel then said the tests "need errors from the same origins" — true, and also the bug | `CVResult` keeps every step of every test block; DM now runs on 48 pooled pairs |

The first row is the one worth dwelling on. It shipped a CV score in the *first* screenshot of this
part — seven models ranked, all plausible, one of them off a 10-fold-only comparison that had asked
for 28.

### The information-criteria guard refuses rather than sorts

`comparability()` compares four facts that must match before two ICs may be ordered: observation
count, differencing, transform stack, and likelihood class. The fourth is the one most software misses
— a GARCH quasi-likelihood and an ARIMA likelihood are both called `loglik` and are not the same
quantity. When they differ the guard returns **no ranking at all** with the reason:

> These fits cannot be ranked by an information criterion because they were not scored on the same
> thing — observation counts 168 vs 180; differencing (d=1, D=0) vs (d=0, D=0). An IC differences
> likelihoods, so the likelihoods have to be for the same observations.

On `traffic_spend` the guard *passes* (all fits share n = 180, d = 0), so the AICc columns populate —
and ETS's 504.24 agrees with the CV verdict, which is the reassuring case. The refusal path is tested
with all four facts and through the API.

### The adequacy verdict corroborates before it condemns

Twelve tests at 5% reject a perfect model 46% of the time, so `judge()` counts **independent
mechanisms**, not p-values:

| series | verdict | why |
|---|---|---|
| white-noise residuals | **ADEQUATE** | 0 of 10 checks reject |
| the mean baseline on `traffic_spend` | **INADEQUATE — REVISE** | Ljung–Box, runs test and CUSUM fail in the same direction — three mechanisms, not three p-values |
| ETS on `traffic_spend` | **ADEQUATE** | 0 of 9 reject; the one model that survives |
| AR(1) residuals | **INADEQUATE** | Ljung–Box + runs, remedy "increase the AR or MA order by one" |
| t(3) residuals | **ADEQUATE WITH CAVEATS** | Jarque–Bera only, and its remedy is *"nothing to do for the point forecast"* |

The two rules that make this useful rather than decorative:

* **A lone rejection is a caveat.** One low p-value among ten, with no second mechanism behind it, is
  what chance produces — and the verdict says so rather than ordering a refit.
* **A CUSUM crossing needs corroboration.** CUSUM is a *cumulative* sum, so it wanders; a boundary
  crossing with no sign test or runs test behind it is recorded as a note, not a failure. Without that
  rule, clean white noise came back "adequate with caveats" on the first run.

Non-normality and ARCH are labelled **distributional**, with remedies that say what they do *not*
mean: *"the point forecast is unaffected; intervals that assume a constant spread are not"*, and
*"fit an ARIMA-GARCH row — this is not a reason to change the mean model's order."*

### Combination: four rules, one honest caveat

The `combination` row is fitted to the other models' **held-out** forecasts, never to their in-sample
fits, and it carries `ic_comparable = False` so it cannot enter an IC ranking. All four rules ship
(mean / median / IC-weighted / Granger–Ramanathan) and each one's own weakness is in its card:

* **median** — "not a weighted average: these weights are recorded as 1/k for reference"
* **ic_weighted** — "come from the components' in-sample fits; a measure of fit, not of forecast accuracy"
* **granger_ramanathan** — "a weight outside [−1, 2] is the regression using one component to correct
  another, which is unstable on collinear forecasts"

Its interval is the **envelope** of the component intervals, and the card says so: the correct
predictive distribution needs the joint distribution of the components' errors, which this row does not
estimate.

### Diagnostics that test the distribution, not the mean

PIT and RQR are implemented with the detail that decides whether they are interpretable:

* **A discrete model's PIT must be randomized.** Raw PIT lands on the PMF's atoms, so a *perfect*
  Poisson model looks violently non-uniform and every uniformity test rejects it. `pit_from_quantiles`
  refuses without `F(y−1)`, and says why in the refusal.
* **RQR is randomized by construction**, so it runs 5 reps and reports whether the verdict changed
  between them (`stability`) rather than picking a rep that suits.
* **Coverage is tested, not thresholded** — 36 of 40 inside a 90% band is 87.5%, and the binomial test
  says p = 0.53 with the 95% interval printed.

**Verified in the running app** (screenshots `104_part13_01_selection_why` ·
`105_part13_02_shortlist` · `107_part13_04_pairwise` · `106_part13_03_diagnostics_ets` ·
`106_part13_03_diagnostics_mean`): `traffic_spend` fits ETS/ARIMA/drift, S11 ranks ETS at CV MASE
0.310 over drift 0.804 with fold spreads beside every mean, Diebold–Mariano returns p < 0.0001 on 48
paired errors, Mincer–Zarnowitz catches the slope of 0.897, and S12 returns ADEQUATE for ETS beside
INADEQUATE — REVISE for the mean baseline with three named mechanisms and a remedy under each. **64 new
tests** (`tests/test_part13_selection_diagnostics.py`), suite total **756**.

---

## Step 13 — forecasting (Part 14)

Part 13 ended with a measured winner and a residual-adequacy verdict. Part 14 turns that **stored fit**
into an auditable 1…H forecast. Six computational modules in `ts/forecast/` — horizons, intervals,
back-transformation, combination, reconciliation and the bundle — feed `api/forecast.py`,
`charts/primitives.py::fan_chart()` and the S13 `ForecastPanel`. The frontend only parses user choices
and renders backend `ResultBase` cards; every point, band, weight, PMF and coherence error is computed
on the backend.

### The horizon is a decision, not a text box

`GET /analysis/forecast/horizon` reports `recommended_max = floor(n/5)`, warns when fewer than two full
seasons are available, and refuses H > n/5 until `acknowledge_long_horizon=true` is sent. On the 180-point
`traffic_spend` example, H = 60 is first refused with a recommended maximum of 36; the advisor card says
why and the explicit "I understand" checkbox is the only way through. A short seasonal history is a
separate warning, not a reason to silently change H.

### Five bands, two constructions, and no bare interval

The fan always computes 50 / 66 / 80 / 90 / 95% bands. Analytic bands come from the fitted wrapper's own
`predict(h, alpha)`; simulation uses at least **2,000 seeded paths** from in-sample residual innovations,
and is the default for non-Gaussian, intermittent and count forecasts. Count models ship the discrete PMF
and integer quantiles. Every band states its target coverage, source and exact construction; the fan
draws the levels widest-first, keeps the observed tail, labels the point path, carries alt text and ships
its numbers as a `TableResult` in `data_behind`. On `traffic_spend` at H = 12 the analytic mean widths run
12.075 → 17.083 → 22.944 → 29.448 → 35.099, so the nested ribbons are visibly a statement about
uncertainty, not decoration.

### Original units require an explicit choice

An active log, Box–Cox or Yeo–Johnson transform refuses a forecast until the user chooses **naive**,
**Duan smearing** or **lognormal** bias adjustment. Quantiles pass through the exact monotone inverse;
only the mean receives the bias correction. The result reports the formula, the smearing factor, any
clipping, and `silent_exponentiation = false`. A differencing operation adds its own replay warning. All
three methods were exercised in the running app; smearing moved the first traffic-series point to
598.909 while leaving its monotone quantiles intact.

### Scenarios, combinations and hierarchies are conditional, and say so

* **ARIMAX scenarios:** future driver paths come from the existing Step 10 scenario endpoint. The
  comparison table runs the same fit twice — drivers held flat versus the supplied paths — and names any
  drivers held at their last observed value. On `MarketingSpend` the largest absolute change was
  171.221; the table says this is the effect of an assumed driver, not a forecast of the driver.
* **Combinations:** mean, median and Akaike-weighted combinations pool only stored fits and, where
  simulation is used, pool seeded paths. IC weights need finite, comparable AICc values; an
  incomparable set is refused with "use mean or median" as the usable fix. The weights and components
  travel with the result.
* **Hierarchy:** optional bottom-up, top-down and diagonal-W OLS / MinT reconciliation use
  `S = [I_b; 1_b′]` and propagate a 95% total interval with `G_h W_h G_h′`. The UI supplies histories,
  the backend refits the Step 11 winner to each series, and the result reports coherence error and the
  independence / fixed-share assumptions. The OLS smoke test reconciled at 7.1e-15.

### Honesty travels with every card

The S12 verdict, every warning, the coverage/construction text on the relevant bands and tables,
the training window and `test_observations_used = 0` travel with the bundle and its `ResultBase`
cards. No holdout observation
informs a forecast. A changed series produces `forecast.stale_fits` rather than a forecast on two scales.
Context lives in the caveat callout; only decision-changing warnings (long horizon, inadequate residuals,
held-flat drivers, clipped back-transforms) are repeated in the urgent callout. The duplicate React keys,
twin callouts and `00:00:00` date tails found in the first live run are pinned by regressions.

**Verified in the running app** (screenshots `115_part14_01_forecast_controls` through
`134_part14_20_back_transform_method`): the 12-step ETS fan with five stated bands, the refusal and
acknowledgement at H = 60, the ARIMAX scenario comparison, OLS hierarchy, the Poisson count forecast
with its PMF table, and the explicit smearing back-transform — with no console errors in any of the
three browser sessions.

The catalogue-level audit of the remaining §5.10 families is the **Part 16 completeness pass** below,
deliberately scheduled by §8 after the model and metric parts exist. At that historical closure S15
finalization and Parts 18–19 were still deferred; Part 17 is the AI assistant pass, Part 18 is the
whole-report writer, and Part 19 is the hardening / deploy pass — all three are now complete below.
Part 16 remains a catalogue and per-item export pass.

---

## Step 14 — evaluation metrics (Part 15)

Part 15 turns a stored S13 forecast into an **out-of-sample** evaluation without ever calling the
full-history fit an accuracy score. Eight modules in `ts/metrics/`, the `api/metrics.py` router,
`charts/evaluation.py`, and the S14 `EvaluationPanel` implement the §§5.10/5.12 metric policy,
rolling-origin protocol, comparative tests, tables and ten plots. The panel collects the scheme; every
score, reason, caveat, chart and `data_behind` table is produced by the backend.

### Zero-leakage rolling origins

Each fold refits every model on its own training window. Differencing order `d`/`D`, ADF decisions,
Box–Cox / Yeo–Johnson λ, imputation summaries, outlier decisions, the naive / seasonal-naive scale and
future exogenous rows are derived from training rows only. Replacing every test-period value with a
sentinel leaves each decision digest unchanged; strict `train_end < test_start`, wrapper row counts and
held-flat future drivers are audited separately. A failed audit refuses S14. Missing test rows and
failed folds are counted, never silently dropped, and `test_observations_used_for_fitting = 0` travels
with the report. Active transforms require an explicit `naive`, `smearing` or `lognormal` choice.

### The metric set is chosen from facts and explains itself

`policy.py` selects MASE + MAE for short series; MASE + RMSSE + ensemble CRPS for intermittent /
zero-heavy data; discrete CRPS + PMF log score + randomized PIT for counts and event data;
QLIKE + variance MSE + Kupiec + Christoffersen for volatility-only targets; CRPS + pinball + PIT +
Winkler for probabilistic continuous forecasts; and the full point set otherwise. MAPE, sMAPE and MPE
are refused on count / intermittent data, and the reason sits beside the chosen metrics in the UI.
Perfect forecasts have **MASE = 0** and the naive forecast has **MASE = 1**, pinned as property tests.

The battery includes ME, MAE, RMSE, MPE/MAPE with near-zero refusal, sMAPE's 0–200% note, RMSSE,
GMRAE/MdRAE, relative ratios and MASE by horizon; CRPS via the sorted-draw identity, pinball, normal
log score with its density caveat, Winkler, coverage/sharpness and calibration; randomized PIT, KS and
Cramér–von Mises; Dunn–Smyth RQR repeated over seeded draws with a Fisher-combined test; DM with HAC
lag `H−1`, Clark–West / Mincer–Zarnowitz, a transparent MCS and directional hit rate. Count models use
the exact per-step PMF. Croston / TSB / ADIDA additionally expose a **disclosed empirical integer PMF**
from their own seeded occurrence/size simulation: stochastic integerisation preserves the simulated
mean, the table says it is not an exact parametric law, and a dotted ranked-probability CRPS curve is
added without displacing the intermittent MASE + RMSSE ranking.

### Results, API and UI

`POST /analysis/evaluate` walks the real S13 gate, stores `objects["evaluation"]`, marks S14 complete,
and returns 21–22 `ResultBase` cards including the verdict, policy and reasons, primary statistic,
metric table and heatmap, split chart, fold/protocol/leakage tables, MASE-by-horizon, fold boxplots,
loss curves, PIT/RQR/calibration/coverage, DM and confusion figures, comparative nested tests and the
branch-specific PMF or variance table. Every figure carries alt text and a `TableResult` in
`data_behind`; every result carries the S12 adequacy verdict, the exact `n` / frequency / evaluation
window, coverage/construction and p-values rendered through `p_phrase` / `p_number`. `GET
/analysis/evaluate` re-renders the stored report and never refits a fold. The S14 panel controls
horizon, folds, expanding/sliding origins, levels, models, paths, seed and RQR repetitions, and shows
the policy reasons, leakage status and backend result cards.

**Verified in the running app** (screenshots `135_part15_01_evaluation_controls` through
`167_part15_33_back_transform_summary`): the continuous probabilistic report, exact count PMF scores,
intermittent empirical-PMF disclosure and dotted CRPS curve, volatility QLIKE / VaR backtests, and the
transformed-series bias-adjustment path — with zero console errors across five browser sessions. Live
API checks found zero bad p strings, strict finite JSON, zero single-character caveats, ten figures,
`S14 = complete`, and a passing leakage audit. That historical closure was **850 backend tests
(787 + 31 + 32)** and **7 frontend tests (3 + 4)**, with a clean `tsc --noEmit` and production
bundle. Part 16 is the later, distinct completeness pass documented next; Part 17 follows it, and
Part 18 closes S15 below; Part 19 completes the build.

## Step 15 — plots & tables completeness catalogue (Part 16)

Part 16 is the §5.10 / §8 audit of every plot and table the computed session can honestly show. It adds a
read-only S15 Catalogue panel and three endpoints under `/api/v1/analysis/catalog`; it does **not**
add a pipeline stage or mark S15 complete. `health.stages` and `/sessions/{id}/results` still expose
exactly S0..S15. At the Part 16 closure `GET /analysis/catalog` returned `S15 STATUS not_started`
because the catalogue was a read-only view; Part 18 now mounts its own whole-report router and
finalizes S15 on a user-confirmed action, while leaving per-item downloads with Part 16.

### Scorecard, families and provenance

`app/charts/completeness.py` derives **49 plots / 49 tables** across the nine required groups from stored
results and objects, against the §5.10 floors of **35 plots / 30 tables**. Every item carries a caption,
alt text where applicable, a data-behind table, provenance, and report-insertable metadata; the
scorecard reports captions, alt text, data-behind, group counts, styling and exports. Its 25 derived
builders recover data-behind rows from the plotted traces rather than recomputing statistics. Object-gated
families that are genuinely unavailable are explicit status rows, never fabricated charts: a fitted
volatility example produces the real `models.garch_variance_path`; a count example produces real
`forecast.count_pmf_ribbons`; VAR sessions produce numeric `models.var_irf` / `models.var_fevd` with
their bootstrap and Cholesky-ordering captions. Static greps and tests pin that direct-`go.Figure`
modules import the shared theme and use `apply()` / `to_json()`; no ad-hoc palette, font or dash
literal survives outside `theme.py`.

### Single-source styling and real exports

`app/charts/theme.py` is the one source for the Okabe–Ito palette, fonts, dash patterns, light / dark /
print variants, title geometry and legend spacing. `restyled()` recolours an existing figure without
resetting its height, top margin or hidden-legend state; `apply()` reserves one extra top-margin row per
wrapped horizontal legend and grows the figure by the same amount, so two-line titles and five-row CUSUM
legends no longer collide or clip in the browser or the 800 px static page. `app/charts/exporters.py`
uses the local Chromium, local Plotly UMD bundle and Kaleido to produce **3750 × 2500 PNG at 300 dpi**,
vector SVG / PDF, and XLSX / CSV data downloads with captions and provenance. No cloud export or network
font is involved.

A dark/print restyle used to reapply `base_layout()` and silently turn hidden legends back on while
resetting the top margin to 18 px; that visual defect is now pinned by two backend geometry tests and a
browser sweep. The browser sweep checks both bounding-box collisions and whether the first bold title
line stays inside its SVG container in light, dark, print, volatility and count sessions.

### API and UI

The three catalogue endpoints are catalogue readback (`GET /analysis/catalog`), report-basket CRUD
(`GET|POST /analysis/catalog/report`) and per-item download
(`GET /analysis/catalog/{item_id}/download`). The S15 `CatalogPanel` renders the scorecard, group / kind
/ search / mode controls, report basket, table previews, explanations and all five per-item download
buttons. The basket's add / remove / set / clear state uses the authoritative API response, not a locally
guessed id list; a pure frontend regression test pins that transition. `data-testid` hooks cover the
scorecard, controls, basket, grid and every item.

### Verification and closure

The Part 16 suite is **15 backend tests** (`tests/test_part16_catalog.py`) and **6 frontend catalogue
tests** (`CatalogPanel.test.ts`), including scorecard thresholds, strict finite JSON, nine groups,
captions / alt / data-behind, all five real exports at 300 dpi, basket CRUD, unknown-format and
unknown-id errors, dark / print remapping, single-source styling, the unchanged 16-stage contract, the
Part 18 router boundary, and the title / legend geometry regressions. The fresh Part 16 closure was **865
backend / 13 frontend tests**, `TSC_OK`, and a clean production build
(`index-Rgonc9fg.js`, 357.62 kB / gzip 99.09 kB). Part 17 later adds 18 backend / 7 frontend
tests; its current closure is documented below. Three targeted smokes still return `EXIT 0` with
49 / 49 plots and tables in every mode, five real download formats, and the three OpenAPI catalogue paths.

**Verified in the running app** (screenshots `168_part16_01_catalogue_scorecard_light` through
`181_part16_14_count_pmf_ribbons`): 98 catalogue items / 49 Plotly figures, scorecard `49` / `complete`
/ `app.charts.theme`, API statuses `[200, 200]`, basket add / clear, a real UI PNG download, dark and
print modes, Evaluation group 14 items, Tables only 49, IRF / FEVD captions, GARCH status, the real
volatility variance path and count PMF ribbons. Zero console errors were recorded in the continuous,
variance and count browser sessions; zero title / legend overlaps and zero clipped titles were recorded
in all five catalogue renderings; and the downloaded 300 dpi PNG is readable with its full title and
subtitle. Part 17 is documented next, then Part 18 and the final Part 19 pass.

## AI assistant — grounded, offline-first (Part 17)

Part 17 adds `backend/app/ai/`, `backend/app/api/ai.py`, `frontend/src/state/ai.ts` and
`frontend/src/components/AssistantDrawer.tsx`. It unlocks the AI layer without changing the statistical
pipeline: `health.stages` stays **16**. At the Part 17 closure S15 was still `not_started` and the
Part 18 `app.api.export` router was absent; Part 18 now mounts that router and finalizes S15.

### Compact digest, strict prompts and post-hoc grounding

`ai/schemas.py` extends the pinned Part 0 `AnalysisDigest` rather than replacing it. `ai/digest.py`
builds the only statistical payload the assistant may see: stage verdicts, `facts_used`, result ids,
top result briefs and their computed numbers, warnings, caveats, ranked advisor rules, figure metadata
and series metadata. Raw series values are omitted by default; the explicit **Send data values** toggle
(default OFF) is bounded, previewable, and keeps a visible banner while it is on. The final live preview
for the staged AirPassengers session was about **3,996 prompt tokens / 15,203 digest characters**,
with six result briefs, 107 grounded numbers, 40 facts, 30 result ids and three ranked rules. No fitted
object, opaque DataFrame or key travels in that payload.

`SYSTEM_PROMPT` forbids invented numbers, requires `[result-id]` citations, plain language with the
technical term in parentheses, and the exact fallback `the app has not computed that yet — open the X
panel`. **What next** may quote only the digest's already-ranked advisor rules. `ai/grounding.py`
rechecks the completed answer: it extracts result-id citations, flags numerals absent from the digest,
and `sanitize_numbers()` removes unsupported numerals before they can reach the UI. The offline
generator uses complete computed sentences and labels every reported number.

### Optional Gemini, deterministic offline mode and quota handling

`ai/client.py` supports `google-genai` as an optional `[ai]` extra, `GEMINI_API_KEY`, default
`gemini-2.5-flash`, opt-in `gemini-2.5-pro`, temperature **0.2** and a seed where the SDK supports it.
The key may be entered in Settings, is stored only in that browser tab's `sessionStorage`, is sent per
request in `X-Gemini-Key`, and is never logged or persisted server-side. A payload preview is required
before the first live request; the preview shows the system prompt, contents, digest, token estimate,
privacy note and `api_key_in_payload: false`. **No API key? Use offline mode** uses `ai/narrative.py`,
a deterministic generator that needs neither a key nor an SDK and leaves the whole app usable. `google-genai`
is not installed in this sandbox, so the live path is pinned by mocked client tests while the offline path
was exercised end to end.

`/api/v1/ai/chat/stream` is server-sent events with clean cancellation and per-session history.
`ai/state.py` enforces a **60-request/session** hard cap, **15 requests/minute**, a 4 s minimum interval
and a free-tier queue, exposes token and estimated-cost counters, degrades to grounded offline answers on
rate-limit errors without storing partial assistant text, and supports **Clear chat** and **Delete my
session**. The assistant may only recommend: suggested panels and result explanations are clickable app
actions that the user confirms.

### Drawer, result-card actions and verification

The drawer provides Settings, Payload preview, streaming chat with cancel, history, usage, an image
attachment for plot interpretation, and Delete my session. Result cards add **Explain this** and
**Interpret plot**; the ribbon offers Explain this, Interpret plot, What next, Write it up, Simplify,
Rewrite for audience (beginner / peer / reviewer) and Explain like I'm 15. **Write it up** is labelled
`AI-drafted, verify before submission`; **Interpret plot** uses the stored figure-data payload or an
attached PNG / JPEG / WebP image (≤4 MiB), never invented pixels. Recommended actions open a panel only
after a user click.

`tests/test_part17_ai.py` adds **18 backend tests** covering the pinned digest subclass and prompt,
config and key privacy, values opt-in, grounding and number sanitization, offline streaming, citations
and history, ranked What next, the exact fallback, the labelled write-up, figure/image interpretation,
the explain-result endpoint, cap / delete / clear, preview-required live mode, rate-limit degradation,
clean cancel, temperature / seed, audience rewrites and image rejection. `src/state/ai.test.ts` adds
**7 frontend tests**. The fresh closure is **883 backend / 20 frontend tests**, `TSC_OK`, a clean
production build (`index-D9l9na41.css` 18.98 kB / gzip 4.63 kB; `index-Cgt50u1s.js` 381.71 kB / gzip
105.89 kB), and a full backend suite in 583.854 s.

**Verified in the running app** (screenshots `182_part17_01_payload_preview` through
`188_part17_07_attached_plot_image`): the preview with no key in the payload, Explain this on a result
card, a grounded stationarity answer with a result-id citation and no unsupported numerals, the Settings
privacy banner, the labelled write-up draft, three confirmable What-next actions and an attached plot
image. All five AI routes were present in OpenAPI and every observed live AI request returned 200;
`SHOTS_OK`, zero console errors, and zero `NaN`,
`Infinity` or `undefined` strings in the rendered body. `/ai/config` reports offline ready, SDK absent,
environment-key boolean present, live not ready, temperature 0.2, cap 60, 15/min, 4.0 s interval, max
40 data values and values OFF. No real Gemini API call was made. Part 18 is documented next; Part 19 completes the build.

## Reports & finalization — Part 18

Part 18 adds `backend/app/ts/report/`, `backend/app/api/export.py`,
`frontend/src/panels/ReportPanel.tsx`, typed `reportStatus` / `finalizeReport` /
`exportDownload` API calls, the S15 report UI and its CSS. It consumes the Part 16 catalogue and the
stored `ResultBase` cards; it never refits a model, recomputes a statistic or duplicates the
per-item PNG/SVG/PDF/XLSX/CSV endpoints. The fixed 16-stage contract, `session_dep` header,
privacy / AI contracts and statistical safeguards are unchanged.

### Weighted scorecard, exact verdict and finalization

`ts/report/scorecard.py` scores the ten §5.13 checks — data quality, written preprocessing,
components, ≥2 agreeing stationarity tests, ACF/PACF, ≥2 candidates plus benchmarks, out-of-sample
evaluation, residual adequacy, uncertainty-interval forecasts and recorded caveats — with weights
summing to 100. Every row carries its result ids and the exact panel to visit. The only verdict
strings are `Analysis complete — ready for manuscript writing`,
`Nearly ready — 2 items outstanding` and `Not yet — see blockers`; a numbered What-to-do-next list
names the exact panels. `GET /export/status` is read-only. `POST /export/finalize` requires S14 and
S12, stores `report.readiness`, `report.scorecard` and `report.next_steps` and marks S15 `complete`.

### Whole-report builders and grounded prose

`ts/report/prose.py` holds five manuscript templates — Methods, Results, Model Selection,
Diagnostics and Forecast Accuracy — whose `[PLACEHOLDER: statistic]` slots are filled from stored
descriptives, decomposition, stationarity, shortlist, evaluation, adequacy and forecast results. A
missing statistic becomes the pinned "the app has not computed that yet — open the X panel" fallback,
never a guessed number; tests assert no placeholder survives. `html.py` emits a one-file HTML report
with inline CSS generated from the single-source house theme and base64 SVG figures — no external
scripts, fonts or images. `docx.py` writes numbered figures / tables, captions and filled prose;
`pdf.py` prints that HTML through local Chromium print-CSS; `latex.py` writes booktabs `longtable`
snippets; `xlsx.py` writes one sheet per catalogue table plus README and Scorecard sheets.
`figures.py` exports every figure as 300 dpi PNG, vector SVG and vector PDF. `script.py` emits one
stdlib-only `.py` that replays the staged API calls and an optional `jsonlite` / `httr` `.R`
equivalent. `archive.py` writes a `.chronos` ZIP with the data hash, all replayable settings,
pipeline state, every result JSON, the catalogue, scorecard, manuscript, caveats and a reopenable
CSV. `all.py` bundles every deliverable into one ZIP. No Gemini key is stored in any export.

### API, UI and scope

The router exposes `GET /export/status`, `POST /export/finalize`, `GET /export/{kind}` and a small
discovery index. HTML is inline by default for sharing and becomes an attachment with `download=1`;
all other kinds are attachments with `Cache-Control: no-store` and `X-Report-*` metadata. Both
`scope=all` and `scope=selected` are supported — the latter consumes the Part 16 report basket — and
`mode` selects light / dark / print styling. `ReportPanel` shows the exact verdict, weighted score,
ten evidence rows, numbered next panels, blockers, scope / theme controls, all ten export buttons,
privacy copy, the confirmed Finalize action and a link back to the catalogue. `CatalogPanel` gained a
matching "Back to S15 report" action; the Report and Catalogue ribbon commands remain reachable even
when S15 dependencies are incomplete.

### Verification and closure

`tests/test_part18_report.py` adds **23 backend tests** and `src/panels/ReportPanel.test.ts` adds
**3 frontend tests**. They exercise all three exact verdict strings, including the pinned
`Nearly ready — 2 items outstanding` branch with two numbered panels; the empty-session stage-order
refusal; the 100/100 fully staged verdict; stored S15 result cards and parseable snippets;
self-contained HTML with real numbers and no placeholders; DOCX sections / numbering; a real `%PDF`;
LaTeX captions; XLSX sheets; `ast.parse` on the Python script; the R equivalent; `.chronos` hashes /
settings / results; 3750 px PNG plus vector SVG/PDF; and the everything bundle with no build errors.
Manuscript facts fall back to catalogue rows or panel-named prose, never guessed numbers. Report
cards keep their outstanding labels out of `ResultBase.caveats`, so finalization cannot manufacture
the evidence for its own caveats check; completed stages without stored results still yield the
Not-yet verdict, and the pinned state machine marks a finalized report stale when an upstream stage
is re-run. A catalogue-id cache is fingerprinted by the series version, stored result ids and stage
states, so a data change invalidates it; a genuinely broken Part 16 catalogue propagates a visible
500 rather than being silently replaced by an empty ready report. A live status / finalization race
exposed `dictionary changed size during iteration` when S15 first gained a result bucket; `Session`
and the Part 16 catalogue now snapshot their mappings, pinned by a deterministic concurrent-insertion
test. The fresh closure is **906 backend / 23 frontend tests**, `TSC_OK`, a clean production build
(`60 modules`, `index-rs-g1-k2.js` 393.47 kB / gzip 108.56 kB), and a full backend suite in
**662 s**. Targeted Part 16 + Part 18 tests pass 38/38.

**Verified in the running app** (screenshots `189_part18_01_not_ready_verdict` through
`197_part18_09_one_file_html`): the exact Not-yet verdict with nine numbered next panels, the
100/100 ready verdict, weighted scorecard, ten export buttons, selected basket scope, catalogue back
link, UI-confirmed finalization with S15 `complete`, and the actual one-file HTML with its embedded
figure, filled manuscript prose and captioned table. `SHOTS_OK` recorded three export OpenAPI paths,
an inline HTML header, zero console errors, and zero `NaN`, `Infinity`, `undefined` or
`PLACEHOLDER` strings in the rendered HTML. Part 19 closes the build below.

## Part 19 — hardening, docs & deploy (final closure)

Part 19 is the acceptance pass for CRAFT §5.15 / §7 / §8. It does not add a stage;
the 16-stage `S0 … S15` contract, the `ResultBase` explanation contract, the
`session_dep` header, privacy rules and statistical safeguards are unchanged.

### Quality gates and reference numbers

| Gate | Evidence |
|---|---|
| Golden fixtures | `backend/tests/fixtures/golden_models.json` freezes 13 model-family fits; `airpassengers_reference.json` freezes statsmodels ADF / KPSS / SARIMAX / ETS numbers and the app's ADF / KPSS outputs. `scripts/generate_golden.py` regenerates them reproducibly. |
| Property & regression tests | MASE(perfect) = 0, MASE(naive) = 1, a 25-seed sweep, advisor regressions for non-positive series, λ intervals, stationary series and white-noise / unit-root next-step rankings. |
| Coverage | `pytest -q --cov=app/ts --cov-fail-under=80` → **91.71%** on 82 `ts/` files (12,584 / 13,722 statements); `pyproject.toml` sets the 80% gate. |
| Full suite | **990 / 990 backend tests** in 609 s without coverage; **990 / 990** in 936 s with the coverage gate. 41 golden, 12 demo, 31 static-deploy tests. |
| Frontend | `npx tsc --noEmit` clean; **12 files / 52 tests** pass; production build clean (73 modules, 31.31 kB CSS, 502.40 kB JS). |
| 10k rows | paste + profile + STL decompose = **1.73–2.27 s including ingestion**; the normal suite asserts `<3.0 s`. The coverage-instrumented run uses a documented 12 s ceiling because a line tracer is not an interactive runtime. |

### Performance hardening

S4 used to hang for >300 s on 10,000 points because Mann–Kendall and Theil–Sen
built ~50 million pairwise comparisons. Part 19 replaces the large-n paths with
an exact Fenwick-tree Mann–Kendall S, deterministic bounded Theil–Sen thinning
(exact for ordinary small series), a vectorised `_tail_outliers`, one merged
Plotly `update_layout` per figure, and a small LRU cache that shares the
identical weekly-period robust STL fit between the seasonality and outlier
passes. `test_robust_stl_cache_is_exact_and_reuses_the_weekly_fit` pins
bit-identical components plus the cache hit; small-n statistical results are
unchanged.

### Beginner and accessibility features

* **242-term glossary** generated from one source into `GLOSSARY.md` and
  `frontend/src/copy/glossary.ts`, with hover- and focus-reachable tooltips.
  `GlossaryText` auto-links arbitrary prose; `GlossarySelection` shows a
  definition for a selected phrase anywhere in the app, including table text.
* **Per-panel "What is this?" drawer** with stage-specific reading / try-this /
  common-misunderstanding prose and glossary chips; reachable from every results
  header and the top bar.
* **Plain-English error library**: 43 searchable articles plus prefix families;
  a failing toast links directly to its own code's explanation, fix and "what
  not to do" guidance.
* **Undo / redo UI** in the top bar, Step 3 options pane and applied-step strip,
  with Ctrl/Cmd+Z, Ctrl/Cmd+Shift+Z and Ctrl/Cmd+Y shortcuts; the API clears the
  redo stack on a fresh branch.
* **Stepper Back / Next** controls, Arrow/Home/End stage navigation, and the
  existing WAI-ARIA one-tab-stop ribbon. "Simplify view" keeps exactly eight
  essential panels; plain-language mode hides the technical layer.
* Screen-reader figure / table value summaries, captions, `:focus-visible`,
  `aria-live`, colour-plus-text verdicts and `prefers-reduced-motion` support.

### Demo, docs and deployment

Three narrated demo scenarios replay the real S0→S15 pipeline in 17 steps with
progress, per-step timings, cooperative cancel, job deletion and a one-hour TTL
sweep; all three reach S15 `complete` at 100/100. Nineteen seed datasets are
bundled. `docker-compose.yml`, `backend/Dockerfile`, `frontend/Dockerfile`,
`frontend/nginx.conf`, `Dockerfile.single`, `frontend/vercel.json`,
`.env.example`, `DEPLOY.md`, `METHODS.md`, `GLOSSARY.md`, `CONTRIBUTING.md` and
`CHANGELOG.md` ship together. The static deploy suite checks YAML validity,
Dockerfile `COPY` sources, nginx `/api/` proxying, env variables, docs, glossary
counts and accessibility primitives. **Caveat:** no Docker binary exists in this
sandbox, so `docker compose up` is documented and statically checked, not run in
a live container. `frontend/vercel.json` still contains a placeholder backend
host that must be replaced with the deployed HF Space / Fly / Render URL.

**Verified live:** routes `/api/v1/demo/*` and `/preprocess/redo`, all three demo
replays, glossary search (242 rows), keyboard ribbon navigation, Simplify view
(24 → 8 commands), plain-language mode, guided tour, demo completion 17/17,
per-panel help, error library search and toast deep link, stepper Back/Next,
undo/redo against a real session, inline and selection glossary tooltips, and
shots `198_part19_01_beginner_toolbar` through
`213_part19_16_error_toast_article`. No `NaN`, `Infinity` or `undefined` strings
and no uncaught application errors were observed. The deliberately expired
session in the last smoke produced expected HTTP 400/404 network logs only.

---

## Repository layout

```
chronos-studio/
├── backend/                     Python 3.11 · FastAPI · statsmodels
│   ├── app/
│   │   ├── main.py              app, CORS, error handling, router mounting
│   │   ├── api/
│   │   │   ├── session.py       session lifecycle, pipeline map
│   │   │   ├── gate.py          S8 endpoints: assess / read / set horizon
│   │   │   ├── forecast.py      ★ Part 14: S13 forecast / horizon / hierarchy
│   │   │   ├── metrics.py       ★ Part 15: S14 rolling-origin evaluation / readback
│   │   │   ├── ai.py            ★ Part 17: config / preview / history / SSE chat
│   │   │   ├── catalog.py       ★ Part 16: S15 catalogue / basket / per-item downloads
│   │   │   ├── export.py        ★ Part 18: S15 scorecard / whole-report exports
│   │   │   └── datasets.py      S0 endpoints: upload / paste / example / remap
│   │   ├── ai/                  ★ Part 17: digest, prompts, grounding, client, state
│   │   ├── ts/report/           ★ Part 18: HTML/DOCX/PDF/LaTeX/XLSX, prose, archive, scripts
│   │   │   ├── schemas.py      AnalysisDigest subclass + request/response contracts
│   │   │   ├── digest.py       computed summaries only; raw values need explicit opt-in
│   │   │   ├── prompts.py      system prompt, exact fallback, panel labels
│   │   │   ├── grounding.py    result-id citations + numeral allowlist/sanitizer
│   │   │   ├── narrative.py    deterministic offline generator
│   │   │   ├── client.py       optional Gemini streaming, low temperature / seed
│   │   │   └── state.py        history, quota, usage, cancellation and deletion
│   │   ├── core/
│   │   │   ├── schemas.py       ★ the API contract (results, advisor, registry, digest)
│   │   │   ├── session.py       ★ pipeline state machine + in-memory store
│   │   │   ├── errors.py        every failure, with plain text + fix
│   │   │   ├── config.py        settings & optional-feature detection
│   │   │   └── seeds.py         deterministic RNG derived from tags
│   │   ├── ts/gate/             ★ Part 7: the readiness gate
│   │   │   └── readiness.py     GateFacts → 10 criteria → one verdict; pure, no I/O
│   │   ├── ts/select/           ★ Part 13: rolling-origin CV, comparisons, shortlist
│   │   ├── ts/diagnose/         ★ Part 13: residual battery and adequacy verdict
│   │   ├── ts/metrics/          ★ Part 15: policy, point, variance, PMF, comparative, protocols
│   │   ├── ts/forecast/         ★ Part 14: horizons, bands, back-transform, combine, reconcile
│   │   │   └── core.py          ForecastBundle: every S13 number + honesty field
│   │   ├── charts/              ★ single-source Plotly theme + Part 16 catalogue
│   │   │   ├── theme.py         palette / fonts / dashes / modes / title-legend geometry
│   │   │   ├── primitives.py    shared chart and table builders; no ad-hoc styling
│   │   │   ├── completeness.py  49 plots + 49 tables across nine catalogue groups
│   │   │   ├── exporters.py     local 300 dpi PNG / SVG / PDF / XLSX / CSV
│   │   │   └── evaluation.py   Part 15 metric figures
│   │   ├── ts/correlate/        ★ Part 6: ACF/PACF, CCF, preview engine
│   │   │   ├── _acf.py          ACF, PACF (Durbin–Levinson), Bartlett bands,
│   │   │   │                    pattern reading, order hints, variance ratio
│   │   │   ├── _ccf.py          cross-correlation with pre-whitening
│   │   │   ├── _preview.py      "what happens if I do this?" on a copy
│   │   │   └── _figures.py      ACF/PACF grid, transformation comparison, CCF
│   │   ├── ts/decompose/        Part 4: methods, components, figures
│   │   │   ├── methods.py       STL / classical / MSTL / UCM / HP / band-pass
│   │   │   ├── components.py    MK, Sen, Cox–Stuart, Fisher's g, scorecard
│   │   │   └── figures.py       component grid, overlay, index bars, periodogram
│   │   ├── ts/io/               ingestion (Part 1)
│   │   │   ├── readers.py       format dispatch, encoding & delimiter sniffing
│   │   │   ├── timeindex.py     date parsing, frequency inference, regularisation
│   │   │   ├── validate.py      numeric coercion, data class, quality report
│   │   │   └── ingest.py        the S0 orchestrator + mapping assistant
│   │   ├── ts/models/          registry + wrappers (Parts 8–11)
│   │   │   ├── predictive.py    ★ Part 10: the discrete PMF, CRPS/log score/PIT
│   │   │   ├── ingarch.py       INGARCH(1,1) and INAR(1)
│   │   │   ├── intermittent.py  Croston / SBA / SY, TSB, ADIDA
│   │   │   ├── pointprocess.py  renewal (exp/Weibull) and Hawkes
│   │   │   ├── volatility.py    ★ Part 11: ARCH/GARCH/GJR/EGARCH × normal/t, ARIMA–GARCH
│   │   │   └── registry.py      48 rows, 43 implemented
│   │   └── assets/examples/     19 teaching datasets + manifest.json
│   ├── scripts/gen_examples.py  rebuilds the bundled datasets
│   ├── scripts/generate_golden.py  regenerates the Part 19 golden fixtures
│   ├── app/ts/eda/stl_cache.py  shared robust STL fit (Part 19 performance)
│   └── tests/                   990 tests (Parts 0–19)
├── frontend/                    React 18 · TypeScript · Vite
│   └── src/
│       ├── App.tsx              shell: top bar → ribbon → stepper → panes
│       ├── components/          Ribbon, Stepper, ResultsPane, ResultCard, Toasts,
│       │                        AssistantDrawer (Part 17), GlossaryTip / GlossaryPanel /
│       │                        GlossaryText / GlossarySelection, HelpDrawer,
│       │                        ErrorLibraryPanel, DemoPanel, HistoryControls (Part 19)
│       ├── panels/              DataPanel (S0), EdaPanel (S1), AdvicePanel (S2–S3),
│       │                        DecomposePanel (S4), StationarityPanel (S5),
│       │                        CorrelatePanel (S6–S7), SelectPanel (S11–S12),
│       │                        ForecastPanel (S13), EvaluationPanel (S14),
│       │                        CatalogPanel (S15 / Part 16 catalogue),
│       │                        ReportPanel (S15 / Part 18 finalization)
│       ├── state/               store, typed API client, ai.ts (Part 17), mirrored types
│       ├── copy/strings.ts      every user-facing sentence (i18n-ready)
│       ├── copy/glossary.ts      generated 242-term glossary
│       ├── copy/beginner.ts      per-panel "What is this?" copy
│       ├── copy/errors.ts        plain-English error library
│       └── styles/              design tokens + layout
├── docker-compose.yml / Dockerfile.single
├── DEPLOY.md / METHODS.md / GLOSSARY.md / CONTRIBUTING.md / CHANGELOG.md
└── CRAFT_PROMPT.md              the full 16-stage specification / build plan
```

---

## The two contracts everything else depends on

**1. Every result explains itself.** Nothing may be rendered as a bare number.
`core/schemas.py::ResultBase` forces `plain` (one sentence a beginner understands),
`why` (why it matters), `technical` (formal definition), `method` (exactly how it
was computed), `caveats`, `citations` and a runnable `reproducible_snippet`.
The frontend renders those fields — it never invents a value.

**2. Every recommendation cites its evidence.** `AdvisorReport` + `RuleOut` mean a
suggestion is a *record*: what it recommends, the computed numbers behind it
(`facts_used`), the result ids it consumed, what happens if you ignore it, and the
exact click path to do it. The Part 17 assistant receives **only** a compact `AnalysisDigest`: stage
verdicts, `facts_used`, result ids and computed numbers, warnings, caveats and series metadata — no
fitted objects or raw values by default. A post-hoc validator requires result-id citations and flags
numerals absent from that digest, which keeps the assistant from inventing statistics.

The 16-stage pipeline (`S0` ingest → `S15` write-up) is a state machine in
`core/session.py`. Re-running a stage marks everything downstream `stale`; the UI
greys it out rather than silently reusing old numbers.

---

## Testing

```bash
cd backend && PYTHONPATH=. pytest -q  # 990 passed (Parts 0–19)
PYTHONPATH=. pytest -q --cov=app/ts --cov-fail-under=80  # 91.71% on ts/, ≥80% gate
cd frontend && npx tsc --noEmit       # typecheck
cd frontend && npm test               # 12 files / 52 tests
cd frontend && npm run build          # production bundle; Plotly is a local UMD asset
```

### Debug sweep (Parts 0–13)

`backend/tests/test_debug_regressions.py` holds 31 tests named after defects that were
actually observed in the running app, not after features. The sweep covered all 19 bundled
examples × 17 endpoints plus constant, all-zero, one-row, NaN and almost-constant series.
It pinned: the Part 1 ingest verdict winning over integrality (so `wwwusage` is continuous,
not Poisson); the one-click data-class override in **Column mapping**; method-aware
decomposition errors and period hints that no longer promise a season on 100 points;
HP/CF and constant-series crash fixes; the readiness gate's travelling caveat; and a single
p-value convention. Backend `core/formatting.py::p_phrase()` and frontend
`state/format.ts::pPhrase()` return the complete comparison (`p < 0.0001` or
`p = 0.0075`): prose no longer prints the impossible `p = 0.000`, `0.0e+00`, or the
ungrammatical `p = < 0.0001`. Backend p-value table columns carry `format="p"` and the
universal table renderer uses `pNumber()`, so a fit-comparison cell reads `< 0.0001` rather
than `0.000`. The production build loads `public/plotly.min.js` as a local
script instead of asking Rollup to bundle the 4.4 MB minified library, which used to exhaust
the sandbox's memory. An explicit `POST /analysis/decompose {"method":"stl"}` still refuses
with a measured fix when there are fewer than two seasonal cycles — silently running a
different method would break reproducibility — while the UI greys out the unavailable
methods and pre-selects one that works.

The ingestion tests are behavioural — each encodes a real file shape that people
actually upload (German decimals, `1821-01-01` in a column named `Year`,
twice-monthly dates, duplicates, a text column called `date`), plus the failure
message a non-expert needs to see when it goes wrong.

The EDA tests (`tests/test_part2_eda.py`) pin the three ways a profile can be
*confidently wrong*: a seasonal test that cannot see seasonality through a trend
(AirPassengers must give p ≈ 1e-17, not 0.43), a seasonal profile reported in
detrended units instead of the units the user reads, and a narrator whose sentences
contain numbers that were never computed. The last one is enforced mechanically by
`assert_grounded()`; the Part 17 offline generator enforces the same boundary with
`validate_grounding()` and `sanitize_numbers()`.

The advisor tests (`tests/test_part3_advisor.py`) are written as *decisions*, because
advice is the part of this app that can do the most damage with a plausible sentence:

| series | must say | must not say |
|---|---|---|
| AirPassengers | take logs; keep a seasonal term | anything about smoothing the model series |
| stable series, λ ≈ 1 | nothing about transforms | "take logs" because λ exists |
| series with zeros | Yeo–Johnson (and why a log is undefined) | "take logs" |
| 20-period gap | **blocker**: decide or shorten the window | "interpolate it" |
| noise | no seasonal terms | "seasonal difference" |

Every recommendation's `because` sentence is checked against `facts_used`, so a rule
cannot cite a number that was not computed. `prep.smooth.not_for_modelling` is
unconditional and tested as such.

The stationarity tests (`tests/test_part5_stationarity.py`, 42 tests) are written as the
specific ways this step goes wrong, because each one was a real defect found while building it:

| trap | what the tests pin |
|---|---|
| wrong tail | a sup-Wald needs an **upper**-tail critical value; with a lower-tail one, white noise cleared a "5%" line of 1.49 |
| spurious break | a random walk must reconcile to `unit_root`, never "stationary around a break" |
| segment-by-segment testing | one level shift must give **one** break, not three |
| self-normalised CUSUM | a step must score *above* its critical value, not below it |
| inverted OCSB | a deterministic seasonal pattern must **reject** the seasonal-unit-root null |
| degenerate regressions | a constant series is omitted with a reason, not reported as 1e13 |
| over-differencing | a stationary verdict must forbid `diff()`, in words |

The Part 13 tests (`tests/test_part13_selection_diagnostics.py`, 64) are written so that a *plausible*
answer fails them. The comparability guard is tested with all four facts it compares, because a guard
that only checks `n` passes the GARCH-vs-ARIMA case it exists to catch. The CV engine is tested for the
three ways a fold silently stops being a fold: test data inside the training window (asserted on the
indices, per fold), a MASE denominator computed on the full series (the test observation must not reach
the scale), and a requested fold that was never built — which is how the 10-of-28 bug shipped. The
verdict tests come in pairs: white noise must be **ADEQUATE** and an AR(1) must be **INADEQUATE** with
a named remedy, because a battery that always says "revise" is as useless as one that never does. The
combination tests assert that its weights came from held-out origins and that its components are the
fitted models rather than the benchmark ladder.

The Part 14 tests (`tests/test_part14_forecast.py`, 31) are written around the five ways a forecast
lies while looking correct: a band with no construction, a horizon that outruns the data, a silently
exponentiated log forecast, simulation that could leak the test period, and a forecast without its S12
adequacy verdict. They pin the seeded 2,000-path floor, the exact monotone back-transformation of
quantiles, integer PMF intervals, the nested fan's `data_behind` table, coherent reconciliation, the
long-horizon acknowledgement gate, stale-fit refusals, and the browser regressions found in the live
UI (duplicate React keys, duplicated callouts, midnight timestamps and a hierarchy forwarded to an
exogenous model).

The Part 15 tests (`tests/test_part15_metrics.py`, 32) are written around the six ways an evaluation
panel can be confidently wrong: in-sample fit relabelled as accuracy, a MASE denominator that saw the
future, a continuous density score on count atoms, a generic metric list, a figure without its table,
and a report with no usable API/UI contract. They pin perfect-forecast **MASE = 0**, naive-forecast
**MASE = 1**, poisoned-test invariance, every policy branch, the exact discrete PMF and the disclosed
intermittent empirical PMF, all ten chart builders, strict finite JSON, stage gates, explicit
back-transformation, warning suppression and the single-character-caveat UI regression. Four frontend
tests cover the S14 request contract and primary-score readback without a DOM testing library.

The Part 16 tests (`tests/test_part16_catalog.py`, 15) are written around the ways a catalogue can look
complete while being unusable: a scorecard that cannot name its items, a caption / alt / data-behind
claim with no backing rows, a PNG that is only the interactive modebar, a dark / print mode that keeps
light ink, a hidden legend resurrected by a restyle, a two-line title clipped or covered by a wrapped
legend, and a whole-report router smuggled in under Part 16's name. They pin 49 / 49 plots / tables in all
three modes, strict finite JSON, all five real download formats at 300 dpi, basket CRUD, the three
catalogue OpenAPI paths, the fixed 16-stage contract, and the boundary that Part 18, not Part 16, owns
whole-report exports. Six frontend
tests cover catalogue filtering, scorecard groups, export labels, defensive cells and the authoritative
report-basket state transition without a DOM testing library.

The Part 12 tests (`tests/test_part12_multivariate_ml.py`, 57) are written around the four ways a
multivariate or ML result can be *confidently wrong*: a VAR fitted to cointegrated levels without
ever asking Johansen, a Granger result that is only the shared trend (so every cause is re-tested on
differences and flagged `spurious_risk`), a model card that reports a shuffled split as validation,
and a threshold model quoting an invalid F-test. The regime tests simulate a known SETAR and assert
recovery to within 0.4%, alongside a linear AR(2) that must be **refused** at p = 0.13 rather than
accepted at the naive p = 0.0085. The API tests walk the real stage order — profile, decompose,
tests, correlate, gate, suggest, fit — because `/analysis/fit` returns 409 otherwise, which is the
gate doing its job; and one test pins that the AI key cannot reach a response body or a `repr`; the
Part 17 tests extend that boundary to payload previews, streams and chat history.

The gate tests (`tests/test_part7_gate.py`, 52 of them) are mostly *integrity* tests rather than
arithmetic, because a verdict is only as good as its refusal to guess: a check that could not run
must never pass, `unknown` must never be reported as "fine", the verdict must be traceable to the
checklist printed under it, a stored verdict must go visibly stale when the series changes, and
every travelling caveat must stand alone with no checklist nearby. The one place a threshold
genuinely matters — the seasonal-cycle count — is tested with the numbers that decide it
(24 cycles pass at `MIN_CYCLES = 2`, 1.5 cycles block). Several of these tests are regressions on
defects the browser run exposed: a break-based verdict crashing on `break_dates`, the horizon caveat
printed twice, the horizon ratio worded backwards, and a superseded horizon caveat staying in the
report.


---

## Optional extras

Everything below is detected at runtime (`/api/v1/health` → `optional_features`)
and the UI degrades gracefully when absent.

```bash
pip install ".[auto]"      # pmdarima  → auto_arima
pip install ".[garch]"     # arch → optional cross-check only; Part 11's GARCH family is
                           #        hand-written and needs nothing here
pip install ".[count]"     # (statsmodels already covers Poisson/NB)
pip install ".[ml]"        # scikit-learn (+ lightgbm/xgboost/torch, all opt-in)
                           # Part 12's six ML rows run on scikit-learn alone; without
                           # those libraries the LightGBM/XGBoost/LSTM rows stay
                           # visible and are refused by their measured import error
pip install ".[report]"    # python-docx, kaleido, playwright (+ optional weasyprint) — Parts 16/18
pip install ".[readers]"   # pyreadstat, pyreadr → .sav/.dta/.rds
pip install ".[ai]"        # google-genai                  (Part 17)
```

Live Gemini is optional. A **free** Google AI Studio key
([aistudio.google.com/apikey](https://aistudio.google.com/apikey)) may be exported as
`GEMINI_API_KEY` or written to `backend/.env` (already gitignored; `chmod 600` it); it
may also be typed into Assistant Settings, where it stays in that browser tab's
`sessionStorage`. The typed key is sent per request in `X-Gemini-Key`, never in a
payload, response, log, history or server-side session: `has_gemini_key()` answers a
boolean. `google-genai` is only needed for live requests. Without both the key and the
SDK, **No API key? Use offline mode** runs the same analysis with a deterministic
narrative generator — the AI adds explanation, never results.

---

## Privacy

Data is held in server memory for the life of the session (default 6 h) and
deleted on demand with **Delete session**. There is no database and no
persistence. The AI assistant sees a compact digest of computed results, not your
raw values. **Send data values** is OFF by default, with a payload preview before
first use and a visible banner whenever it is ON. **Clear chat** and **Delete my
session** remove assistant history; a typed key is never persisted server-side.

---

## Licence

MIT.
