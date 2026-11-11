# Chronos Studio — CRAFT Prompt (for building a JASP-like Time-Series Analysis App)

> **How to use this file**
> 1. Copy **everything inside the fence below** into a fresh chat with a coding agent (Claude Code, Cursor, Codex, Gemini CLI, ChatGPT, etc.).
> 2. The agent must answer with **Part 0 only** first. Then you type `Generate Part 1`, `Generate Part 2`, … to build the app in pieces.
> 3. Keep the conversation alive — the agent remembers earlier parts. If a chat dies, paste `HANDOFF` (see §9) plus the last part you generated.
> 4. Working name used throughout: **Chronos Studio** — rename it by telling the agent once.

---

````markdown
# CRAFT PROMPT — "Chronos Studio": A Free, Web-Based, Beginner-Friendly Time-Series Analysis Workbench (JASP for Time Series)

## 0. OPERATING CONTRACT (read before anything else)

You are building a real, runnable software product — not a sketch. Obey these rules for the entire session:

**R0.1 — Part-gated generation.** Generate code **only** for the part I name. Never pre-generate, "while I'm here", or bundle later parts. When I type `Generate Part X`, output only what §8 assigns to Part X.

**R0.2 — No code dumps without a manifest.** Before any code in a part, print a `DELIVERABLES` table: file path → one-line purpose → estimated LOC. Then write the files.

**R0.3 — Every part must run.** At the end of each part, give exact commands to launch/verify what exists so far. If a part depends on a future module, create a **stub with a typed interface** and register it in a running `DEFERRED_INTERFACES` list — never a silent `pass`.

**R0.4 — No placeholders in delivered code.** No `TODO`, no `...`, no "rest is similar". If a part is too large for one response, split it yourself as `Part 8.1`, `Part 8.2` and say so at the top.

**R0.5 — Statistical honesty is a hard requirement.** Every recommendation the app shows to a user must be backed by a *computed number that is displayed next to it*. The app must never print "your data looks stationary" without the test statistic and p-value on screen. Where a threshold is heuristic, label it as a default and let the user change it.

**R0.6 — Beginner-first UX.** Assume the user has never seen an ACF plot. Every statistic gets: plain-language meaning (1 sentence), a rule-of-thumb, a "why it matters", and an expandable technical definition.

**R0.7 — Reproducibility.** Every analysis stores its settings, random seed, library versions, and data hash. Any result screen can export a self-contained `.py` (and optionally `.R`) script that reproduces it.

**R0.8 — Free-tier only.** No paid APIs, no paid hosting requirement, no proprietary binaries required for the core path. Optional heavy extras must be feature-flagged and degrade gracefully.

**R0.9 — Fail loud, fail kind.** Errors reach the user as one plain-English sentence + one suggested fix + a collapsible technical trace. No stack traces as the primary message.

---

## 1. CONTEXT (C)

JASP is a free, point-and-click desktop GUI over R that lets non-programmers do classical statistics: load a dataset, tick checkboxes, get publication-ready tables and plots. It has no serious time-series module comparable to its ANOVA/regression coverage.

**The gap:** analysts, students, and thesis writers who need time-series analysis currently have to (a) learn Python/R, or (b) stitch together five separate tools, and (c) still guess at decisions like *"should I difference or detrend?"*, *"is this stationary now?"*, *"which model?"*, *"is my evaluation correct?"*.

**What we are building:** a **free, browser-based, click-through time-series workbench** that carries a non-expert from *raw CSV* to *defensible forecast* to *manuscript-ready results*, and that explains every decision it recommends.

**Why now / why web:** zero install, works on a student laptop or a lab Chromebook, shareable via link, and — critically — it can host an **AI assistant** (Google AI Studio / Gemini free tier) that interprets the *actual computed outputs* for the user.

**Scope of v1 (in):** single-series univariate analysis end-to-end; panel/multivariate limited to description + basic VAR/exogenous regression; report export.
**Out of scope for v1:** real-time streaming, distributed compute, deep-learning forecasting as a default path, user accounts/billing, mobile-native apps.

---

## 2. ROLE (R)

Act as a **trio working as one**:

1. **A time-series statistician** (the rigor of Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*, Shumway & Stoffer, and the X-13ARIMA-SEATS / STL / STAMP traditions) who knows which tests are valid under which assumptions, and when a diagnostic is being misapplied.
2. **A senior full-stack engineer** who ships maintainable code: typed, tested, containerized, with clean module boundaries and no notebook spaghetti.
3. **A UX designer for novices** who has watched a first-year student cry at an ACF plot — and designs so that doesn't happen.

Where these conflict, **statistical correctness beats visual polish**, and **clarity for the beginner beats both** — but only by simplifying the *presentation*, never the *computation*.

---

## 3. AUDIENCE (A)

Primary users, in priority order:

| Persona | Context | What they need | Failure mode to design against |
|---|---|---|---|
| **Thesis / dissertation student** | Has a 60-row monthly series in Excel, needs Chapter 4 | Guided pipeline, APA-style tables, copy-paste results, methods paragraph | Doesn't know what to report; gets lost in jargon |
| **Practitioner / analyst** | Wants a fast credible forecast without writing code | Speed, sensible defaults, model comparison, uncertainty intervals | Distrust; wants to see the numbers behind the recommendation |
| **Instructor** | Teaching a forecasting course | Reproducible demos, dataset bundles, viewable analysis steps | Tool too opaque to explain in class |
| **Reviewer-minded supervisor** | Will ask "why did you difference?" | Full audit trail, exported script, stated assumptions | Black-box output |

Design implications: **progressive disclosure**, **plain-language mode on by default**, **every panel has a "Why am I seeing this?" link**, **example datasets one click away**, **nothing destructive without an undo**, **analysis state is visible as a pipeline the user can walk backward**.

---

## 4. FORMAT (F)

### 4.1 Product architecture (build to this unless I say otherwise)

```
chronos-studio/
├── backend/                        # Python 3.11 + FastAPI
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, router mounting
│   │   ├── api/                    # thin HTTP layer (no stats logic here)
│   │   │   ├── datasets.py  analysis.py  models.py
│   │   │   ├── forecast.py  advisor.py   ai.py  export.py
│   │   ├── core/
│   │   │   ├── session.py          # analysis-session state machine
│   │   │   ├── schemas.py          # Pydantic contracts (the API's truth)
│   │   │   ├── errors.py  config.py  logging.py  seeds.py
│   │   ├── ts/                     # THE STATISTICAL CORE (pure, testable)
│   │   │   ├── io/                 # readers, validators, time-index repair
│   │   │   ├── eda/                # descriptives, plots, autocorrelation
│   │   │   ├── preprocess/         # impute, smooth, transform, outliers
│   │   │   ├── decompose/          # classical, STL, MSTL, X-13 adapter
│   │   │   ├── tests/              # unit-root, trend, seasonality, breaks, ARCH
│   │   │   ├── models/             # ETS, ARIMA, Theta, TBATS, GARCH, ML, VAR
│   │   │   ├── select/             # IC comparison, rolling-origin CV, DM test
│   │   │   ├── diagnose/           # residual batteries, QQ, PIT, RQR
│   │   │   ├── forecast/           # point + interval + simulation forecasting
│   │   │   ├── metrics/            # point, relative, probabilistic, coverage
│   │   │   └── report/             # HTML/DOCX/PDF/LaTeX/XLSX builders
│   │   ├── advisor/                # RULE ENGINE: every "we suggest X" lives here
│   │   │   ├── rules_*.py          # declarative rules, one file per stage
│   │   │   └── narrative.py        # rule output -> plain-English sentences
│   │   ├── ai/                     # Gemini assistant: grounding, prompts, tools
│   │   └── assets/                 # example datasets, glossary, templates
│   ├── tests/                      # pytest; each ts/ module has a twin
│   ├── pyproject.toml  Dockerfile  requirements.txt
└── frontend/                       # React 18 + TypeScript + Vite
    ├── src/
    │   ├── app/                    # routes, shell, layout
    │   ├── components/             # JASP-like ribbon, left options panel,
    │   │                           #   right results pane, collapsible output cards
    │   ├── panels/                 # one folder per pipeline stage
    │   ├── charts/                 # Plotly wrappers (single source of chart styling)
    │   ├── state/                  # store, session persistence, API client
    │   ├── copy/                   # ALL user-facing strings + glossary (i18n-ready)
    │   └── styles/
    └── package.json  vite.config.ts  Dockerfile
```

**Stack (locked unless I override):** FastAPI + Pydantic v2 · pandas / numpy / scipy / statsmodels / `pmdarima` / `arch` / `plotly` (server-side figure JSON) · `kaleido` or `plotly.js` client-side for export · React + TypeScript + Vite + Tailwind + Plotly.js + TanStack Query.
**Optional, feature-flagged:** `prophet`, `statsforecast`, `sktime`, `lightgbm`, `x13as` binary, `weasyprint` (PDF), `python-docx` (DOCX), `rpy2`.
**State:** in-memory session store keyed by `session_id`, TTL 6 h, optional Redis. Uploads to a temp dir, auto-purged. No database in v1. No auth in v1 (add later behind a flag).

### 4.2 The pipeline as a state machine (this is the spine of the whole app)

Implement as `core/session.py::PipelineState`. The UI renders it as a horizontal stepper; a stage can be re-run and everything downstream is invalidated (and visibly greyed out).

```
S0  INGEST        load file → detect schema → parse time index → validate
S1  PROFILE       EDA: descriptives, quality report, plots
S2  ADVISE_PREP   recommend impute / smooth / transform / outlier handling
S3  PREPARE       apply chosen preprocessing (versioned, reversible)
S4  DECOMPOSE     trend, seasonal, cyclic, remainder + strength measures
S5  TEST_STATION  unit-root + seasonality + break battery
S6  CORRELATE     ACF / PACF / CCF + interpretation
S7  ADVISE_NEXT   detrend / difference / seasonal adjust / variance-stabilize
S8  GATE          "Is it stationary and model-ready?" verdict + blockers
S9  SUGGEST_MODEL ranked model families with reasons
S10 FIT           fit selected candidates, store full objects
S11 SELECT        IC + rolling-origin CV + DM test → winner
S12 DIAGNOSE      residual battery, QQ, PIT/RQR, adequacy verdict
S13 FORECAST      point + intervals, horizon guidance, scenarios
S14 EVALUATE      accuracy metrics + backtest report + benchmark comparison
S15 FINALIZE      completeness scorecard → "ready for manuscript?" + exports
```

Every transition returns an `AdvisorReport` (see §4.4). The AI assistant reads **only** these objects.

### 4.3 The universal result contract (Pydantic — get this right, everything else follows)

Every computed result is one of: `TableResult`, `FigureResult`, `StatResult`, `NarrativeResult`, `VerdictResult`. All carry:

```python
class ResultBase(BaseModel):
    id: str                     # stable, e.g. "stat.adf.diff1"
    stage: str                  # "S5"
    title: str                  # "Augmented Dickey-Fuller test (after 1 difference)"
    plain: str                  # 1-sentence beginner meaning
    why: str                    # why it matters
    technical: str              # formal definition, LaTeX allowed
    method: str                 # exact procedure + library + settings used
    inputs: dict[str, Any]      # series version/hash, params, seed
    citations: list[str]        # e.g. "Dickey & Fuller (1979)"
    caveats: list[str]          # assumption violations detected
    warnings: list[str]
    reproducible_snippet: str   # the Python that regenerates THIS result
```

Derived types add their payload (`dataframe_json`, `figure_json`, `value`, `p_value`, `effect_size`, `interpretation_band`, `verdict`). The frontend never invents a number; it renders `ResultBase` + payload. **Enforce with a test.**

### 4.4 The Advisor pattern (used by every recommendation in the app)

No recommendation may be hardcoded into a React component or a sentence. All advisor output flows from declarative rules:

```python
@dataclass
class Rule:
    id: str                     # "prep.diff.needed"
    stage: str
    when: Callable[[AnalysisContext], bool]     # pure function over computed facts
    recommend: str              # "Apply first difference"
    confidence: Literal["high","medium","low"]
    because: Callable[[AnalysisContext], str]   # MUST interpolate real values
    provenance: list[str]       # result ids the rule consumed
    alternative: Callable[[AnalysisContext], str | None]
    risk_if_ignored: str
    how_to: str                 # exact click path in the UI
```

```python
@dataclass
class AdvisorReport:
    stage: str
    headline: str                        # one sentence
    recommendations: list[Rule]
    blockers: list[Rule]                 # things that MUST be resolved
    facts_used: dict[str, float | str]   # every number cited, machine-readable
    confidence: Literal["high","medium","low"]
    next_stage_ready: bool
```

`facts_used` is the *only* grounding the AI assistant gets. Rules must be unit-tested with synthetic series whose correct advice is known (e.g. a pure random walk must trigger `prep.diff.needed`; an AR(1) with φ=0.3 and no trend must **not**).

### 4.5 Response format when I type `Generate Part X`

Use exactly this skeleton — no preamble, no summary of what you're about to do:

```
=== PART X — <Name> ===
STATUS        what exists now / what this part adds / what it unlocks
DELIVERABLES  | path | purpose | ~LOC |
DEFERRED      interfaces stubbed for later parts
CODE          <one fenced block per file, with the path as the first line comment>
VERIFY        exact shell commands + expected output/screenshot
TESTS         the tests included in this part, and how to run them
RISKS         anything I should sanity-check as a statistician
NEXT          one line: what `Generate Part X+1` will add
```

Additional keyword commands you must honor all session long:

| Command | Effect |
|---|---|
| `Generate Part X` / `Generate Part X.Y` | Produce only that part |
| `Manifest` | Re-print the full deliverables table for all parts, marking what's done |
| `Status` | Current pipeline coverage, deferred interfaces, failing tests |
| `Audit` | Review already-generated code for R0.5 violations, stubs, leakage, bad stats |
| `Refactor <path>` | Rewrite one file without touching others |
| `Explain <module or rule id>` | Prose explanation, no code |
| `Test <module>` | Produce/extend pytest suite only |
| `UI <panel>` | Produce/extend only that frontend panel |
| `Doc` | Generate `README.md`, `METHODS.md`, `GLOSSARY.md`, `CHANGELOG.md` |
| `Deploy` | Dockerfile + compose + free-tier deployment guide |
| `Handoff` | Emit a compact state block so the build can resume in a new chat |

---

## 5. TASK (T)

Build **Chronos Studio**. Below is the full functional specification, partitioned into the parts you will generate on command. §5 defines *what the software must do*; §8 defines *what goes in which part*.

### 5.1 Ingestion — must read almost anything

- **Formats:** `.csv`, `.tsv`, `.xlsx`/`.xls`, `.json`, `.txt`, `.parquet`, `.dta` (Stata), `.sav` (SPSS), `.rds` (best-effort), pasted clipboard text, and **direct URL fetch**. Bonus: `.ods`.
- **Time index detection** (the #1 beginner failure point — spend real effort here):
  - Parse datetime columns (multiple formats, `dayfirst` ambiguity flagged), `year`+`month` / `year`+`quarter` columns, `YYYY-MM`, `YYYYQn`, ISO week, epoch seconds/millis, Excel serial dates, and **regular index inference from row order** with a confirmation prompt.
  - Infer frequency via `pd.infer_freq` **plus** a robust fallback (median spacing + Fourier/periodogram check for monthly-vs-quarterly confusion). If ambiguous, ask the user with a picture of two candidate calendars.
  - Detect and report: duplicate timestamps, gaps, irregular spacing, DST/timezone hazards, leap-year artifacts, trailing whitespace, thousand separators, `1.234,56` European decimals, mixed date locales.
- **Assisted column mapping:** guess target series, time column, and exogenous columns; show a 3-row preview; let the user override with one click.
- **Missing data audit:** per-column counts, runs of consecutive gaps, MCAR/MAR hints (Little's test when feasible), and a plain-English "what this means for your analysis".
- **Data quality report:** constant runs, zero-variance stretches, impossible values, out-of-range flags, negative values when a log transform was proposed, outliers (z, MAD, IQR, STL-remainder), duplicates.
- **Built-in example datasets** (bundled, ~12): AirPassengers, USAccDeaths, Lynx, WWWusage, JohnsonJohnson, sunspots, beer production, electricity demand, retail sales, COVID cases, daily temperature, an intermittent-demand series, a random walk, a trending+seasonal + structural break series. Each with a suggested "what you'll learn" note.
- **Large-data guard:** >200k rows → downsample for *interactive* plots only, never for computation; warn explicitly.

### 5.2 EDA — describe before you model

Automatic on load, no clicks required. Produce:

- **Descriptives table:** n, span, frequency, min/max, mean, median, SD, IQR, CV, skewness, kurtosis, % missing, #zeros, #negatives, first/last values, total change, CAGR.
- **Distribution:** histogram + KDE, boxplot, ECDF, violin-by-season, normality (Shapiro/JB/Anderson-Darling) *with the caveat that normality of the raw series is rarely required* — say so.
- **Time plots:** raw series, with rolling mean ± rolling SD band, with a smoothed overlay (LOESS/HP), calendar heatmap (day-of-week × week for daily; month × year for monthly), seasonal subseries plot, month/quarter boxplot grid, lag plot matrix (lags 1–12 default).
- **Outlier / anomaly panel:** STL-remainder outliers, level shifts vs spikes vs temporary changes (label them differently), each with a click-to-inspect.
- **The "just tell me what this is" card:** a generated 3–5 sentence plain-English description of the series (shape, direction, seasonality, volatility, gaps) — this is the same generator the AI assistant will narrate, so build it once.
- **Every plot** supports: hover, zoom, PNG/SVG export, "download data behind this plot", and a one-click "explain this plot" hook.

### 5.3 Preprocessing advisor — suggest, don't impose

For each of the following, compute the diagnostic evidence **first**, then recommend with the evidence attached:

| Decision | Evidence the app must compute | Correct default policy |
|---|---|---|
| **Imputation** | gap length, gap position, missingness pattern, seasonality strength, stationarity of remainder | <2 consecutive gaps in a seasonal series → interpolation (linear/seasonal); 2–12 → seasonal-mean or Kalman/`statsmodels` state-space smoother; long gaps or non-random missingness → **block**, don't silently impute; always offer "leave as NA and shorten" |
| **Smoothing** | signal-to-noise ratio, STL remainder variance share, autocorrelation of remainder | **Never smooth the series you will model** — smoothing destroys autocorrelation structure and invalidates MASE. Recommend smoothing for *presentation only*, and say why in plain words. Allow: MA, EMA, Savitzky-Golay, LOESS, HP filter (with the λ caveat), and STL-remainder smoothing. |
| **Transformation** | Box-Cox λ estimate + CI, variance-vs-level correlation (Park–Ruprecht / Dickey–Fuller-GLS heteroskedasticity check), any zeros/negatives, seasonal amplitude growth | log if λ≈0 and data strictly positive; sqrt if λ≈0.5; Box–Cox otherwise; **never** log series containing ≤0 (offer `log(y+c)` with a stated c, or Yeo–Johnson); warn that the back-transformed forecast mean is *biased* and offer bias adjustment (naive / smearing / lognormal) |
| **Outlier handling** | type (spike / level shift / transitory), magnitude, count, effect on ADF/KPSS | default = **flag, don't remove**; offer "add as intervention dummy (ARIMAX pulse/step)" — which is the statistically defensible route and teaches the user something; hard-delete allowed only with a warning |
| **Seasonality strength** | STL `F_S`, seasonal-difference stationarity, seasonal unit-root tests | if `F_S` > 0.64 (configurable) → seasonal difference or seasonal term likely needed |
| **Aggregation / frequency change** | n per candidate frequency, information loss | offer but warn about induced autocorrelation (temporal aggregation → MA structure) |
| **Windowing/trimming** | influence of endpoints, COVID-era style shocks | offer explicit "analysis window" control, always shown in outputs |

### 5.4 Decomposition & components

- **Method menu:** classical additive/multiplicative, **STL** (default; robust + configurable `s_window`, `t_window`, `robust`), **MSTL** (multiple seasonalities), **X-13ARIMA-SEATS** (if `x13as` present, else greyed out with install instructions), state-space unobserved-components (`UnobservedComponents`), HP filter, Christiano–Fitzgerald band-pass (optional).
- **For each component, quantify:**
  - **Trend:** direction, magnitude over the window (units *and* %), Mann–Kendall τ + p, Sen's slope + CI, Cox–Stuart, piecewise/segmented trend (allow breakpoints), trend strength `F_T` (Hyndman), % of variance explained.
  - **Seasonality:** which period(s), `F_S`, seasonal amplitude (peak–trough, absolute and % of mean), seasonal index/pattern table, timing of peak and trough, Kruskal–Wallis across seasonal groups, seasonal dummies F-test, evidence of *changing* seasonality (rolling seasonal amplitude), and whether it is additive or multiplicative (compare decomposition residuals).
  - **Cyclic (non-seasonal) variation:** periodogram / spectral density, top-5 peak periods with power and significance (Fisher's g-test), band-pass filtered component, spectral entropy, distinction from seasonality explained in plain words ("a cycle has no fixed calendar").
  - **Irregular / noise:** remainder variance share, ACF of remainder, Ljung–Box on remainder, normality of remainder, ARCH-LM (→ conditional heteroskedasticity is *structure*, not noise — say so), Hurst exponent if requested, approximate entropy / sample entropy as a complexity descriptor.
- **Decomposition scorecard:** one table ranking how much of the series is trend vs season vs cycle vs noise, with an auto-written interpretation.
- **Guardrail:** multiplicative decomposition on data with values ≤0 must be blocked with a clear explanation.

### 5.5 Stationarity & structural tests

Run the full battery and present it as a **verdict matrix**, not a wall of p-values:

- **Unit roots:** ADF (constant / constant+trend / none — results for all three shown, with the correct one highlighted given the visual/statistical trend evidence), Phillips–Perron, KPSS (level & trend), DF–GLS, **Zivot–Andrews** (unit root with one structural break) and **Lee–Strazicich** if feasible, seasonal unit roots (Canova–Hansen / OCSB, HEGY for quarterly).
- **Trend:** Mann–Kendall, Sen's slope, Cox–Stuart, and a note that trend ≠ non-stationarity.
- **Structural breaks:** Chow (known date), Quandt–Andrews (sup-Wald, unknown date), Bai–Perron multiple breaks, CUSUM/CUSUMSQ, and a "break candidates" overlay on the plot.
- **Variance/heteroskedasticity:** White, Breusch–Pagan, ARCH-LM, rolling SD plot, variance-ratio test.
- **Nonlinearity (optional, flagged):** BDS, Teräsvireli's neural test, threshold hints.
- **The reconciliation logic** (this is the part most tools get wrong — implement carefully): ADF and KPSS test opposite nulls. The app must render a **2×2 reconciliation matrix** (ADF says unit root? × KPSS says stationary?) and explain each of the four cells in plain words, including the "contradictory" cells where the honest answer is *"inconclusive — here are the three plausible reasons"*.
- **Already-stationary guardrail:** if the series is already stationary, explicitly **prohibit** (or at minimum hard-warn against) differencing. Detect over-differencing: ACF at lag 1 strongly negative (≈ −0.5 for MA(1)-like), variance *rising* after differencing, unit-root test still rejecting, and report the diagnosis.

### 5.6 ACF / PACF & the "what next" advisor

- **ACF and PACF** at lags 1…min(4·period, n/5): bars, confidence bands (white-noise ±1.96/√n *and* the better Bartlett bands), value table, per-lag hover interpretation.
- **Automatic pattern reading** shown as a badge on the plot: *white-noise-like*, *short-memory AR-like*, *MA-like*, *slowly decaying → non-stationary / long memory*, *seasonal spike at s*, *alternating/negative lag-1 → possible overdifferencing*, *significant spike at lag q only → MA(q) hint*.
- **Model-order hints:** candidate `(p,d,q)(P,D,Q)s` derived from ACF/PACF cut-offs, EACF if feasible, and cross-checks against the unit-root results — displayed as hints, never as a decision.
- **Ljung–Box / Box–Pierce** at multiple lags (including seasonal lag 2s), with the caveat that it loses power on seasonal spikes.
- **CCF** and pre-whitening when exogenous columns exist.
- **The "what next" advisor (S7):** consumes S5+S6 and produces a ranked action list — detrend (which method and why), first difference, seasonal difference, log then difference, seasonal adjustment (STL/X-13), leave alone — each with the triggering evidence, the expected effect on ACF, and a one-click *preview* showing the post-action ACF and test results **before** the user commits. Preview is mandatory: it prevents the classic "differenced it three times" disaster.

### 5.7 Model-readiness gate (S8)

A single, unambiguous verdict card with:
- **Verdict:** `READY` / `READY WITH CAVEATS` / `NOT READY` / `INCONCLUSIVE`.
- **A checklist of objective criteria**, each pass/fail with its number: stationarity (which tests agree), sufficient length vs horizon, no unresolved gaps, outliers addressed or flagged, variance stability, seasonality handled, no evidence of unmodeled structural break, adequate signal.
- **Blockers** that must be cleared, each with a button that jumps to the fix.
- **Caveats** that travel with every downstream output and into the final report.
- Never a bare "stationary: yes".

### 5.8 Model suggestion & fitting (S9–S10) — the full model registry

**Suggestion engine.** For each candidate, state *why it is suggested here* and *what would have to be true for it to win*. The registry below is the minimum; every entry must be implemented behind a uniform `ModelSpec`/`ModelWrapper` interface with `fit / predict / forecast / simulate / intervals / params_table / diagnostics / ic`.

#### A. Baselines & white-noise references (always fit, never optional)

| Model | Role | Implementation notes |
|---|---|---|
| **White noise / i.i.d.(0,σ²)** | The null every model must beat. Forecast = μ, intervals = σ·z. | Fit `mean` + residual test. If Ljung–Box fails to reject at every lag, **the honest answer is "no model needed"** — the app must be willing to say this. |
| **Gaussian white noise** | Probabilistic reference with normal predictive distribution | for CRPS/log-score baselines |
| **Naive (random walk)** | Mandatory benchmark; MASE denominator (non-seasonal) | `ŷ_{t+h}=y_t`; intervals from residual SD × √h (drift-free) or from simulation |
| **Seasonal naive** | Mandatory benchmark; MASE denominator (seasonal) | `ŷ_{t+h}=y_{t+h-m}` |
| **Mean / global average** | Floor benchmark | flat forecast, σ constant |
| **Drift** | Random walk with drift | `ŷ_{t+h}=y_t + h·(y_n−y_1)/(n−1)` |
| **Random walk with drift = ARIMA(0,1,0)+drift** | note the equivalence in the UI — a teaching moment | |

#### B. Mean / conditional-mean models (classical)

| Model | Class | Notes & guards |
|---|---|---|
| **AR(p)** | ARMA family | PACF cut-off driven; Yule–Walker / MLE; stationarity root check |
| **MA(q)** | ARMA family | invertibility root check; ACF cut-off driven |
| **ARMA(p,q)** | ARMA family | grid or IC search |
| **ARIMA(p,d,q)** | integrated | `auto_arima` if `pmdarima` present, else bounded grid; d from S5/S7 with override warning |
| **SARIMA(p,d,q)(P,D,Q)s** | seasonal | seasonal unit-root driven; expose all tested orders |
| **SARIMAX / ARIMAX** | with exogenous | intervention dummies (pulse/step from S2 outlier flags), future exogenous paths, scenario mode |
| **ARFIMA / fractional d** | long memory | when ACF decays hyperbolically and d̂ ∈ (0,0.5) (GPH / Whittle estimator); **flag that AIC is not comparable to ARMA** |
| **ARIMA + GARCH** | mean + variance | see D |
| **Theta (θ=0..2, optimal θ)** | M3/M4 champion | strong, stable seasonality; short series |
| **ETS / Holt–Winters** (A/N/M × N/A/M × damped) | exponential smoothing | `auto_ets`-style grid with IC; block multiplicative on non-positive data |
| **UCM / structural (local level, local linear trend, BSM)** | state space | interpretable components, smooth intervals, handles missing values natively (Kalman) |
| **TBATS / MSTL+ETS** | multiple seasonalities | flagged; degrade to MSTL+ETS if `tbats` absent |
| **Prophet (optional)** | trend+seasonality+holidays | off by default; always benchmarked |
| **Forecast combination** (mean / median / IC-weighted / Granger–Ramanathan regression) | ensemble | often wins; must be in the candidate set |

#### C. Point-process / count / intermittent models (the ones most tools omit)

| Model | When | Notes |
|---|---|---|
| **Poisson (time-varying λ)** | count data, mean ≈ variance | log link; overdispersion check is mandatory before recommending |
| **Negative Binomial (NB2)** | count data, variance > mean (overdispersed) | estimate α; compare to Poisson via LR test / AIC — **never** default to Poisson without the dispersion check |
| **Poisson / NB with trend & seasonality** | count + calendar effects | GLM with time and harmonic/seasonal terms |
| **INAR(1) / INGARCH (integer GARCH)** | count series with serial dependence | optional, flagged; report as advanced |
| **Croston (classic / SBA / SY)** | intermittent demand, few non-zero periods | MASE vs seasonal naive |
| **TSB (Teunter–Syntetos–Babai)** | intermittent with obsolescence risk | |
| **ADIDA / Aggregate–Disaggregate Intermittent Demand** | lumpy demand | optional |
| **Zero-inflated / hurdle count models** | excess zeros structurally | distinguish *structural* zeros from *missing* — say which |
| **Hawkes / self-exciting point process (advanced)** | event timestamps rather than values | flagged; requires event-level input mode |
| **Renewal / inter-arrival models (exponential, Weibull)** | event intervals | flagged; report rate and hazard |

#### D. Volatility / variance models

| Model | When | Notes |
|---|---|---|
| **ARCH(q)** | ARCH-LM rejects, volatility clustering | |
| **GARCH(p,q)** | volatility clustering, persistent variance | |
| **EGARCH / GJR-GARCH** | leverage effect (negative shocks raise variance more) | |
| **GARCH-t / GED errors** | fat-tailed standardized residuals | |
| **ARIMA–GARCH / ARMAX–GARCH** | mean dynamics + conditional variance | forecast the **mean with a variance-aware interval**, and separately the variance; must be explained in plain words |
| **Stochastic volatility (optional)** | slow-moving variance, no ARCH signature | flagged |
| **Realized volatility (optional)** | intraday returns available | flagged |

#### E. Regression-with-AR-errors & multivariate

| Model | When | Notes |
|---|---|---|
| **ARDL / dynamic regression with ARMA errors** | exogenous drivers, autocorrelated residuals | lags selected by IC; bounds test for levels relationship |
| **VAR / VARX** | multivariate system | Granger causality, IRF (with bootstrap bands), FEVD, stability roots |
| **VECM / Johansen; Engle–Granger** | cointegrated non-stationary levels | must run the cointegration test first and **refuse** a spurious VAR-in-differences story without it |
| **Panel / pooled (optional)** | multiple entities | flagged |

#### F. Machine-learning & nonlinear (opt-in only, always benchmarked)

LightGBM/XGBoost, Random Forest, k-NN, SVR, **LSTM/GRU/TFT (flagged, heavy)**, Gaussian Process, MARS, **self-exciting threshold AR (SETAR)**, **Markov-switching AR**, **neural network autoregression (NNAR)**. Requirements for all ML entrants: leakage-safe feature construction (target lags only, recursive vs direct multi-step declared), a **mandatory tabular-model card** stating features/lags/validation scheme, and a comparison against ETS/ARIMA that must be shown even if the ML model loses.

**Registry mechanics (how the app must treat this list):**
- One `models/registry.py` mapping `model_id → ModelSpec` with `family`, `required_data_properties`, `supports_exogenous`, `supports_seasonality`, `supports_probabilistic`, `supports_missing_values`, `is_baseline`, `feature_flag`, `min_n`, `assumption_checks`.
- The suggestion engine **filters the registry** by data facts (count vs continuous, #zeros, overdispersion, seasonality strength, stationarity, n, exogeneity presence) — it does not offer a model whose assumptions the data visibly violates without saying so.
- **Auto-detection of data class** drives the whole branch: `continuous` → B/C excluded; `count` → Poisson/NB branch offered first, and a dispersion test decides which; `intermittent` (>30 % zeros default) → Croston/TSB promoted, MAPE banned; `binary/proportion` → Beta/Logit-normal or binomial GLM; `event-timestamps` → point-process branch.
- **Fitting requirements:** store full fitted objects, convergence warnings (esp. ETS/ARIMA parameter-bound hits, GARCH non-convergence, GLM separation), estimated parameters with SEs, information criteria, and a **fit-failure path** for every single entry: catch, log the reason, fall back to the next candidate, and never show an empty panel.
- **Count/volatility special rule:** for count and volatility models, the *evaluation* battery switches (§5.12) — CRPS/PIT on the discrete predictive distribution, and for count data MAPE is forbidden outright.

### 5.9 Model selection & diagnostics (S11–S12)

- **Information criteria:** AIC, AICc, BIC, HQ — with the correct caveat that ICs are **only comparable on identical data and identical differencing**, and that IMA/ARFIMA models' ICs are not comparable to ARMA's. Enforce the comparability rule in code and refuse invalid comparisons.
- **Rolling-origin cross-validation** (default: 3–5 folds, expanding or sliding, horizon = forecast horizon): MAPE/sMAPE/MASE/RMSSE/CRPS per fold per model, plus a fold-level boxplot. **This is the primary selection criterion**; ICs are secondary. State that explicitly in the UI.
- **Diebold–Mariano** (and **Clark–West** / **MGN** for nested models) between the top two.
- **Model confidence set** (optional) if feasible.
- **Diagnostics battery per candidate:**
  - Residual time plot, residual vs fitted, residual histogram + KDE + normal QQ (with **shaded tolerance bands**, not just a line), ACF of residuals with bounds, Ljung–Box at multiple lags (including 2s), Box–Pierce, ARCH-LM (with the correct advice: ARCH ≠ model failure → consider ARIMA-GARCH), Jarque–Bera / Shapiro / Anderson–Darling with **the correct message that non-normal residuals do not invalidate the point forecast, only the naive normal intervals**, runs test / sign test, CUSUM of residuals, parameter stability (recursive coefficients / rolling estimation), and Ljung–Box *inverse* checks.
  - **QQ plot must be correct:** de-trended, standardized residuals, with a `qqline` and CI envelope, plus a note distinguishing raw vs standardized vs randomized-quantile residuals.
  - **Outlier/leverage in residuals:** flag residual spikes with their dates and their effect on IC.
- **Adequacy verdict:** `ADEQUATE` / `ADEQUATE WITH CAVEATS` / `INADEQUATE — revise` with the specific failing test named and the suggested remedy (add MA term, add seasonal term, model the variance, add intervention dummy, consider nonlinear).
- **Selection report:** a single table + a ranked shortlist + a "why this model won" paragraph template filled with real numbers.

### 5.10 The complete plots & tables catalog (S12.5 — build it as a factory)

Every item is exportable (PNG 300dpi, SVG, PDF), has a caption, and can be inserted into the report. Assume ≥35 plots and ≥30 tables; the catalog must be *complete*, not a sample. Groups:

1. **Data & quality:** raw series, series + rolling stats, missingness map, outlier overlay, calendar heatmap, distribution panel, seasonal subseries, month×year heatmap.
2. **Decomposition:** classical/STL/MSTL component grid, decomposition with different methods side by side, seasonal index bar chart, seasonal amplitude over time, trend with CI, periodogram, spectral density, cycle component.
3. **Stationarity:** ADF/KPSS/PP result forest-plot (a great beginner visual), rolling mean & SD, ADF over a grid of lags, structural break overlays, CUSUM/CUSUMSQ, variance ratio.
4. **Correlation:** ACF, PACF, ACF+PACF grid (raw vs differenced vs seasonally differenced — side by side is the teaching moment), CCF, cross-correlation heatmap, EACF table/plot.
5. **Model fit:** actual vs fitted, fitted with intervals, parameter table, IC comparison bar chart (with a 2-unit "indistinguishable" band drawn on it), CV fold chart, rolling-origin prediction paths (the "fan of history" plot).
6. **Diagnostics:** residual time plot, QQ, residual ACF, residual histogram, residuals vs fitted, PIT histogram, RQR QQ plot, ACF of squared residuals, CUSUM.
7. **Forecast:** forecast fan chart (multiple interval levels), forecast + holdout overlay, all-models forecast comparison, forecast with prediction intervals in original units after back-transformation, horizon-recommendation plot (accuracy vs horizon), scenario comparison, decomposition of the forecast.
8. **Evaluation:** metric comparison table + heatmap, MASE by horizon, fold-level accuracy boxplots, CRPS/Pinball loss curves, PIT histograms per model, coverage vs nominal coverage calibration plot, DM test matrix, hit-rate/confusion for directional accuracy.
9. **Tables:** every test in §5.5, every model's parameter table, IC table, CV results, metric table, coverage table, break table, outlier table, seasonal index table, data quality summary, glossary, methods paragraph, results paragraph.

**Consistency rule:** all charts go through one styling module (palette, fonts, colorblind-safe, dark/light, print mode with black-and-white-safe linestyles). No ad-hoc figure code in API handlers.

### 5.11 Forecasting (S13)

- Point forecasts at 1…H (default H from user; **advisor warns if H > n/5**, explains why, and can still proceed with an explicit "I understand" flag).
- **Prediction intervals** at 50/66/80/90/95 %, from the model's analytic intervals *and* from simulation (≥2,000 paths, seeded). If the model is non-Gaussian or the data is intermittent, simulation is the default.
- **Back-transformation** with bias adjustment options (naive, smearing, lognormal) — user-visible, because silently exponentiating a log model is one of the most common published errors.
- **Reconciliation** of aggregate vs disaggregate forecasts when the user supplies a hierarchy (optional).
- **Scenario/exogenous paths** for ARIMAX: user edits future exogenous values and sees the forecast change.
- **Combination forecasts:** simple mean / median / IC-weighted combination of the top models — often wins, include it as a candidate.
- **Forecast honesty:** if the residual diagnostics were inadequate, the forecast panel must carry a visible caveat banner.

### 5.12 Evaluation metrics (S14) — implement all of these correctly

**Point accuracy:** ME, RMSE, MAE, MPE, MAPE *with warnings for near-zero series*, sMAPE *with the note that its 0–200% scale is not a "percentage error"*, **MASE** (seasonal MASE with m = period; in-sample naive denominator, clearly documented), **RMSSE**, GMRAE, **MdRAE**, relative-to-benchmark ratios, and **MASE per horizon**.
**Probabilistic / interval:** **CRPS** (empirical, from simulated/quantile forecasts — implement carefully, cite Gneiting & Raftery 2007), average **pinball/quantile loss** per quantile level, **Log score** (with the continuous-log-score caveat and `log score = -mean(log p(y))`), **Winkler/interval score**, **PI coverage** (empirical vs nominal), average interval width (sharpness), **calibration curve** (coverage vs nominal), and **sharpness subject to calibration** as the stated evaluation philosophy.
**PIT and RQR:** PIT histogram + uniformity test (KS, and Cramér–von Mises) per model, with the diagnostic reading guide (U-shape = too narrow, hill = too wide, skew = bias); **Randomized Quantile Residuals (RQR)** — randomized, so seeded and repeated R times with averaged QQ and a combined uniformity test, plus the honest caveat about randomization.
**Comparative tests:** **Diebold–Mariano** (with HAC variance), **Clark–West** and **Mincer–Zarnowitz** for nested models, **Model Confidence Set** if feasible, and **directional accuracy / hit rate** with its (in)famous low power noted.
**Forecast evaluation protocols:** train/test split with the split shown on the chart, rolling-origin (multiple folds), and a clearly labeled *out-of-sample only* zone. **Leakage is a defect**: enforce that no test-period data informs parameters, differencing orders, Box-Cox λ, imputation, or outlier decisions — and write a test that proves it.
**Discrete / count / point-process evaluation (do not reuse continuous formulas blindly):** for Poisson/NB/INAR/INGARCH/Croston-type models, score on the **discrete predictive PMF** — CRPS for count forecasts, log score with the PMF (not the density), PIT computed from the discrete CDF (with randomised PIT to remove the discreteness spikes, and say that you did), and **Dunn–Smyth randomized quantile residuals** for model checking. MAPE and sMAPE are forbidden on count and intermittent data; MAE is reported only alongside MASE.
**Automatic metric-selection policy** — the app must *choose metrics appropriately for the situation* and say why: intermittent/zero-heavy → MASE + RMSSE + CRPS, never MAPE; **count/event data → discrete CRPS + log score + randomised PIT, no MAPE**; **volatility-only targets → QLIKE + MSE of variance + VaR backtests (Kupiec, Christoffersen)**; short series → MASE + MAE; probabilistic forecaster available → CRPS + pinball + PIT lead; else → MAE/RMSE + MASE with a note that the intervals are approximate. Display the *reason* next to the metric set.

### 5.13 Finalization & manuscript readiness (S15)

A **completeness scorecard** with weighted items, each linking to its result:
- data quality resolved or documented ✓/✗
- preprocessing justified in writing ✓/✗
- components characterized ✓/✗
- stationarity established with ≥2 agreeing tests ✓/✗
- ACF/PACF reported ✓/✗
- ≥2 candidate models + benchmarks compared ✓/✗
- out-of-sample evaluation done ✓/✗
- residual adequacy verdict issued ✓/✗
- forecast with uncertainty intervals ✓/✗
- assumptions & caveats recorded ✓/✗
Then one of: **"Analysis complete — ready for manuscript writing"** / **"Nearly ready — 2 items outstanding"** / **"Not yet — see blockers"**, plus a numbered "what to do next" list with the exact panels to visit.

**Exports (all in this part):**
- **HTML report** (self-contained, one file, embedded figures) — the default share/download.
- **DOCX** with numbered tables/figures, captions, and pre-written template prose (Methods, Results, Model Selection, Diagnostics, Forecast Accuracy) with **[PLACEHOLDER: statistic]** slots auto-filled from real results.
- **PDF** (via print-CSS or weasyprint).
- **LaTeX snippets** for tables, ready for a thesis.
- **XLSX workbook** with one sheet per table + a README sheet.
- **PNG/SVG/PDF** for every figure at publication resolution.
- **Reproducible script**: a single `.py` that re-does the whole analysis, and (nice-to-have) a `.R` equivalent.
- **Session archive** (`.chronos` zip: data hash, all settings, all results JSON) so a reviewer can reopen it.

### 5.14 The AI Assistant (free Google AI Studio / Gemini)

**Setup.** `google-genai` SDK, API key from env `GEMINI_API_KEY`, model configurable (`gemini-2.5-flash` default, with a `gemini-2.5-pro` opt-in), `temperature` ≤ 0.3, seeded where supported. Key entered once in a Settings panel (stored in the browser session, sent per-request, never logged, never persisted server-side). A **"No API key? Use offline mode"** path that runs the deterministic narrative generator alone — the app must be fully usable without the AI.

**Grounding contract (non-negotiable).** The assistant never sees raw data by default. It receives:
1. a compact `AnalysisDigest` JSON (schema in `ai/schemas.py`): stage verdicts, `facts_used`, top results with their numbers, warnings, caveats, series metadata (n, freq, dates, ranges — no values unless the user opts in);
2. the user's question;
3. the current panel id.
Its system prompt must forbid inventing any number, require it to cite the specific result id for every claim, require plain language with the technical term in parentheses, and require it to say **"the app has not computed that yet — open the X panel"** rather than guess. Include a post-hoc **number-grounding check** that extracts numerals from the reply and flags any that don't appear in the digest.

**Features:**
- **Ask anything about my results** (chat, streaming, cancels cleanly, per-session history).
- **Explain this** button on every result card (pre-seeds the panel context).
- **Interpret this plot** with an image or the figure's data payload.
- **What should I do next?** — must answer from the advisor's ranked rules, not free-form.
- **Write it up** — generate Methods / Results / Diagnostics / Discussion paragraphs from real numbers, labelled "AI-drafted, verify before submission".
- **Rewrite for my audience** — beginner / peer / reviewer.
- **"Explain like I'm 15"** toggle.
- **Data-safety controls:** explicit toggle for sending data values (default OFF), a visible banner when values are sent, free-tier rate-limit handling with graceful degradation and a queue, cost/token counter, and a hard request cap per session.
- **Never** let the assistant mutate the analysis; it may only *recommend*, and any action it suggests must be rendered as a clickable app action that the user confirms.

### 5.15 Cross-cutting product requirements

- **Beginner features:** glossary with 120+ terms (tooltip everywhere), guided tour, "Example analyses" that replay a full study step by step, plain-language mode toggle, per-panel "what is this?" drawer, error-message library in plain English, undo/redo, pipeline stepper with back navigation, and a **"Simplify view"** that hides all but the essential 8 panels.
- **Accessibility:** WCAG 2.1 AA, keyboard-navigable ribbon, screen-reader labels on every figure and table with the values summarized in the `alt`, colorblind-safe palettes, no color-only encoding, reduced-motion support.
- **Performance:** 10k-row series end-to-end interactive in <3 s on a laptop; heavy models (TBATS/ML/CV grids) run as background jobs with progress + cancel; never block the UI.
- **Privacy:** no server-side persistence of user data beyond the session TTL; explicit delete-my-session button; a plain-English privacy note; AI payload preview ("exactly what will be sent") before first use.
- **Quality gates:** pytest coverage ≥80% on `ts/`, golden-file tests comparing against known-good outputs (statsmodels R-comparison fixtures for ADF/KPSS/ARIMA/ETS on AirPassengers), property tests for the metrics (MASE of the perfect forecast = 0; naive-forecast MASE = 1), and statistical regression tests for the advisor rules.
- **Docs to generate at the end:** `README.md` (with screenshots), `METHODS.md` (exactly which test/implementation/assumption is used — the document a reviewer reads), `GLOSSARY.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and a one-page "Deploy for free" guide (Docker → Render/Railway/HF Spaces + Vercel, or a single-container Fly.io free tier).

---

## 6. NON-NEGOTIABLE STATISTICAL SAFEGUARDS

Implement these as **enforced code**, not UI text. Each has a test.

1. Never difference a series that at least two unit-root/KPSS tests jointly indicate is already stationary — warn hard, require explicit override.
2. Never difference more than twice without an explicit, recorded justification.
3. Never apply log/Box–Cox to non-positive values; block multiplicitative decomposition likewise.
4. Never compute MASE/relative metrics without first fitting the naive (or seasonal naive) benchmark.
5. Never compare AIC/BIC across models with different differencing or different effective sample sizes — refuse and explain.
6. Never report in-sample fit as forecast accuracy.
7. Never let any test-period observation influence any modeling choice (test it).
8. Never present an interval without stating its coverage and how it was constructed.
9. Always show `n`, `frequency`, and the exact evaluation window next to any accuracy number.
10. Always attach the "residual adequacy" verdict to any forecast shown or exported.
11. Always flag when the forecast horizon exceeds ~1/5 of the training length, and when `n < 2 periods` of data per season.
12. Never silently drop rows; every dropped row is counted and reported.

---

## 7. DEFINITION OF DONE (whole product)

- A non-programmer can go from uploading a CSV to a downloadable, publication-formatted DOCX/HTML report with forecasts and diagnostics, without touching code or reading documentation.
- Every recommendation on screen is traceable to a displayed statistic.
- The AI assistant adds explanation and drafting, and is *provably* grounded in computed results — removing it leaves a fully functional app.
- The pytest suite passes, the golden-file comparisons match, `docker compose up` gives a working app, and a free-tier deployment guide exists.
- `METHODS.md` is accurate enough that a thesis committee could defend it.

---

## 8. PART-BY-PART GENERATION PLAN (what goes in which part)

Generate **only** the named part. Parts are ordered by dependency; do not skip.

| Part | Name | Contents (generate exactly this) | Unlocks |
|---|---|---|---|
| **0** | **Scaffold & Contracts** | Repo tree, `pyproject.toml`/`package.json`, FastAPI app with health + CORS, React/Vite/Tailwind shell with the JASP-like ribbon/left-options/right-results layout, `core/schemas.py` (ResultBase + all result types + PipelineState + AdvisorReport), `core/session.py` skeleton, `core/errors.py`, Dockerfiles, `.env.example`, example dataset placeholders, `README` stub, pytest + vitest setup. **No analysis logic.** | Everything |
| **1** | **Ingestion & Data Layer** | `ts/io/`: readers (csv/tsv/xlsx/json/parquet/dta/sav/url/paste), time-index parser + frequency inference, column mapping, validation, quality report, missing-data audit, `SessionData` model, upload/preview endpoints, dataset manager UI panel, 12 bundled example datasets. | S0 → S1 |
| **2** | **EDA & Plot Factory** | `ts/eda/` descriptives, distribution, time plots, seasonal panels, outliers, plain-English series description; `charts/` single styling module; Plotly config + export; EDA panel UI with export buttons. | S1 |
| **3** | **Preprocessing Advisor & Engine** | `ts/preprocess/` (impute, smooth, transform/Box-Cox, outliers, aggregate), `advisor/rules_prep.py`, preview-before-commit endpoint, S2/S3 panels with accept/reject/undo. | S2, S3 |
| **4** | **Decomposition & Components** | `ts/decompose/` (classical, STL, MSTL, UCM, HP, X-13 adapter), component quantification (trend/season/cycle/noise metrics incl. `F_T`/`F_S`, Mann–Kendall, Sen, periodogram, ARCH-LM on remainder), scorecard + interpretation, S4 UI. | S4 |
| **5** | **Stationarity & Structural Tests** | `ts/tests/` ADF/PP/KPSS/DF-GLS/Zivot–Andrews/Canova–Hansen/OCSB/HEGY, trend tests, break tests (Chow, Quandt–Andrews, Bai–Perron, CUSUM), heteroskedasticity, 2×2 reconciliation logic, `AdvisorReport`, S5 UI with the verdict matrix. | S5 |
| **6** | **ACF/PACF & Next-Step Advisor** | ACF/PACF/CCF with Bartlett bands, pattern classifier, order hints, Ljung–Box battery, `advisor/rules_next` (detrend/difference/seasonal-adjust/leave-alone), **preview** endpoint that re-runs tests on the candidate series, S6/S7 UI. | S6, S7 |
| **7** | **Readiness Gate** | `advisor/rules_gate.py`, checklist + blockers + caveats object, gate card UI, caveat propagation into downstream result payloads. | S8 |
| **8** | **Models — Baselines, Registry & ARMA family** | `ts/models/registry.py` (the `ModelSpec` registry from §5.8, incl. data-class auto-detection), `ts/models/base.py` (`ModelWrapper` interface + `FitDiagnostics` + fit-failure path), **baselines**: white noise (with the "no model needed" verdict), Gaussian WN, naive, seasonal naive, mean, drift; **AR/MA/ARMA/ARIMA/SARIMA** (auto_arima + bounded-grid fallback + stationarity/invertibility root checks); **ARFIMA** (long-memory branch). **Split 8.1 = registry+base+baselines, 8.2 = ARMA family, 8.3 = ARFIMA+orders audit.** | S9, S10 |
| **9** | **Models — Smoothing, Structural, Exogenous** | ETS/Holt–Winters auto grid (add/mult/damped, non-positive guard), Theta (θ=0..2 + optimal), UCM/structural (local level, LLT, BSM, Kalman-native missing values), SARIMAX/ARIMAX with pulse/step intervention dummies and future-exogenous scenario paths, ARDL / regression-with-ARMA-errors. | S9, S10 |
| **10** | **Models — Point Process, Count & Intermittent** | Data-class branch for count/intermittent/event data: Poisson (time-varying λ, log link) with the **mandatory overdispersion pre-check**, Negative Binomial (NB2) + LR/AIC comparison, Poisson/NB with trend+harmonics, zero-inflated/hurdle, INAR(1)/INGARCH (flagged), Croston (classic/SBA/SY), TSB, ADIDA, renewal/inter-arrival and Hawkes (advanced, event-timestamp input mode). Includes the discrete-predictive-distribution plumbing used by CRPS/PIT in Part 12. | S9, S10 |
| **11** | **Models — Volatility** | ARCH(q), GARCH(p,q), EGARCH, GJR-GARCH, GARCH-t/GED, ARIMA–GARCH, stochastic volatility + realized volatility (flagged). Produces **two forecasts** (mean with variance-aware intervals, and the variance path), leverage-effect reporting, and non-convergence fallbacks. | S9, S10 |
| **12** | **Models — Multivariate, Nonlinear & ML (opt-in)** | VAR/VARX + Granger/IRF(bootstrap bands)/FEVD/stability roots, VECM/Johansen/Engle–Granger (with the spurious-regression guard), ARDL bounds test, panel (flagged); SETAR, Markov-switching AR, NNAR, Gaussian Process; LightGBM/XGB/RF/k-NN/SVR with leakage-safe feature builder + mandatory model card; LSTM/TFT flagged; **forecast combination** (mean/median/IC-weighted/Granger–Ramanathan). Every ML entrant must be benchmarked against ETS/ARIMA even when it loses. | S9, S10 |
| **13** | **Selection & Diagnostics** | `ts/select/` IC computation + comparability guard, rolling-origin CV engine with leakage tests, DM/Clark–West/MZ tests; `ts/diagnose/` residual battery, QQ with bands, PIT, RQR, count-model residual variants (Dunn–Smyth / PIT for discrete), ARCH-LM; adequacy verdict; S11/S12 UI. | S11, S12 |
| **14** | **Forecasting** | `ts/forecast/` point + analytic + simulation intervals, discrete/count prediction intervals, back-transformation with bias adjustment, horizons advisor, scenarios/exogenous paths, hierarchical reconciliation; S13 UI with fan charts. | S13 |
| **15** | **Evaluation Metrics** | `ts/metrics/` all point, relative, probabilistic (CRPS, pinball, log score, Winkler, coverage, calibration), discrete-prediction CRPS/log-score, PIT/RQR uniformity tests, DM matrix, **metric-selection policy engine** (bans MAPE on intermittent/zero-heavy, switches to CRPS when a probabilistic forecaster exists) with reasons; S14 UI + all evaluation plots/tables. | S14 |
| **16** | **Plots & Tables Completeness Pass** | Audit against the §5.10 catalog; fill every missing plot/table incl. per-family plots (GARCH variance path, count PMF ribbons, VAR IRF/FEVD); captions; export PNG/SVG/PDF/XLSX; single-source styling; accessibility `alt` text. | Full output coverage |
| **17** | **AI Assistant** | `ai/` Gemini client, AnalysisDigest builder, system prompts, streaming chat, number-grounding validator, "Explain this"/"Write it up"/"What next"/"Simplify", offline narrative mode, rate-limit + cost handling, privacy controls + payload preview; assistant drawer UI. | AI layer |
| **18** | **Reports & Finalization** | `ts/report/` HTML/DOCX/PDF/LaTeX/XLSX builders, manuscript prose templates, session archive, reproducible-script exporter; completeness scorecard engine + S15 UI; "ready for manuscript" verdict. | S15 |
| **19** | **Hardening, Docs & Deploy** | Coverage pass + golden-file fixtures per model family, performance pass, accessibility pass, glossary (120+ terms), `README`/`METHODS`/`GLOSSARY`/`CONTRIBUTING`/`CHANGELOG`, Dockerfile + compose, free-tier deployment guide, demo mode, seed data. | Ship |

**Dependency note:** the plots/tables completeness pass (Part 16) intentionally runs *after* all model and evaluation parts so it can fill gaps against reality. If asked for it early, say so and refuse. **Part 8 must ship the full model registry before any model is implemented** — the registry is what Parts 9–12 filter against, and adding a model later must be a one-file change plus a registry row.

---

## 9. HANDOFF BLOCK

When I type `Handoff`, emit this filled in, ≤40 lines, so the build can resume in a new chat:

```
CHRONOS-STUDIO HANDOFF
Name/stack:
Parts completed: [list]
Parts deferred/stubbed: [interface → file → expected consumer]
Pipeline stages implemented: [S0..S15 with % coverage]
Key contracts (do not change without migration): [ResultBase fields, AdvisorReport fields, ModelWrapper methods, AnalysisDigest schema]
Test status: [pass/fail counts, known failures]
Known deviations from the CRAFT prompt: [list]
Next part: [X — name]
Files that must be pasted to resume: [minimal set]
```

---

## 10. START

Respond now with **Part 0 only**, in the §4.5 format. Do not generate ingestion, statistics, or UI panel code. End by telling me to type `Generate Part 1`.
````

---

## Appendix — notes for *you* (not part of the prompt)

**Two corrections to your spec, folded into the prompt already:**
- **"CPRS" → CRPS** (Continuous Ranked Probability Score) — the standard probabilistic accuracy metric; I also added pinball loss, log score, Winkler score and coverage/calibration, which together with PIT and RQR form the correct probabilistic battery.
- **PIT and RQR are diagnostics, not accuracy metrics** — they test whether the predictive distribution is calibrated. The prompt uses them that way, and adds RQR's randomized-sampling caveat.

**Three additions I made that you didn't ask for but that will save you later:**
1. **The `AdvisorReport` + declarative rule engine** — without it, every "we suggest differencing" sentence gets hardcoded into a React component and becomes untestable/unaswerable when a supervisor asks "why?".
2. **Preview-before-commit for differencing/transformation** — the single best guard against the classic novice error of over-differencing. The prompt makes it mandatory.
3. **The reconciliation matrix for ADF vs KPSS** — the two tests have opposite nulls; most student-built tools report them side by side and leave users confused. Naming the four cells explicitly is what makes this feel like a real product.

**Suggested build order if you're time-constrained:** Parts 0→1→2→3→5→6→7→8.1(registry+baselines)→8.2(ARMA)→9(ETS/Theta)→13(selection+diagnostics)→14(forecast)→15(metrics)→18(reports). That is a genuinely useful app — no exotic models, no AI — in about 10 generations. Then bolt on 10 (count/intermittent), 11 (GARCH), 4 (decomposition), 17 (AI), 19 (deploy) in whatever order your users demand.

**If a part overflows the model's output limit:** the agent is instructed to split (e.g. `8.1`/`8.2`/`8.3`). You can also pre-emptively ask `Generate Part 8.1` yourself.

**On the expanded model registry (§5.8 A–F):** the important design decision is *not* that ~60 model entries exist — it's that the **registry is data, not code paths**. A `ModelSpec` row declares what a model needs (`required_data_properties`) and provides (`supports_probabilistic`, `supports_missing_values`). The suggestion engine then filters the registry by the facts computed in Parts 1–7, so "which models do I show this user?" is a query over data rather than a growing `if`-ladder. Adding VAR next year should be one file plus one registry row — test that claim early, in Part 8, or it will quietly stop being true.

**Where the count/point-process branch earns its keep:** a tool that offers ARIMA on a series of monthly event counts is doing the user harm. The data-class detector (continuous / count / intermittent / binary / event-timestamp) must run in Part 1's quality report and gate the *entire* downstream branch — including banning MAPE, switching to discrete CRPS, and promoting Croston/TSB over ETS. Put the detector in early; retrofitting it later means rewriting the model panels.
