"""
ts/models/registry.py — the model registry (§5.8).

The registry is **data**. A model is a `ModelSpec` row; whether it is offered to a user is a query
over that row and the facts measured from the data. No model is added by editing a branch.

That claim is testable, and Part 8 tests it: `test_adding_a_row_offers_a_model` inserts a new
`ModelSpec` at runtime and asserts it becomes a suggestion with the right reasons, with no code
change anywhere else. If that test ever needs a `if model_id == ...` to pass, the design has
quietly failed.

Two kinds of refusal, kept distinct because they are different facts:

* **incompatible** — the model cannot be fitted at all. A Poisson model on a series with −3 in it is
  not "risky"; it is undefined. These are hidden from the candidate set but still listed, greyed
  out, with the reason.
* **assumption warning** — the model can be fitted, but the data visibly violates a premise the
  user should know about before reading its output. These stay in the candidate set, badged.

Collapsing the two would either hide a real warning or ban a usable model.
"""

from __future__ import annotations

from dataclasses import dataclass
from app.core.formatting import p_phrase

from app.core.schemas import (
    DataClass,
    ModelCapabilities,
    ModelFamily,
    ModelRequirements,
    ModelSpec,
)
from app.ts.models.facts import DataFacts

# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #
_SPECS: list[ModelSpec] = []


def _spec(**kw) -> ModelSpec:
    s = ModelSpec(**kw)
    _SPECS.append(s)
    return s


# --- A. Baselines: always fitted, never optional (§5.8 A) ------------------- #
MEAN = _spec(
    model_id="mean",
    display_name="Mean (global average)",
    family=ModelFamily.BASELINE,
    is_baseline=True,
    description="Forecast every future value as the sample mean, with a constant interval.",
    plain_description="Ignore time entirely and predict the average. The floor every model must clear.",
    required=ModelRequirements(min_n=3),
    capabilities=ModelCapabilities(supports_intervals=True, supports_probabilistic=True),
    triggered_when="Always — it is the floor benchmark.",
    assumption_checks=["none — this model assumes nothing, which is why it is the floor"],
)

NAIVE = _spec(
    model_id="naive",
    display_name="Naive (random walk)",
    family=ModelFamily.BASELINE,
    is_baseline=True,
    description="Forecast ŷ(t+h) = y(t); intervals widen with √h.",
    plain_description="Assume tomorrow looks like today. Beating this is the minimum bar.",
    required=ModelRequirements(min_n=3),
    capabilities=ModelCapabilities(supports_intervals=True, supports_probabilistic=True),
    triggered_when="Always — it is the mandatory benchmark and the MASE denominator.",
    assumption_checks=["nothing to check; it is the benchmark"],
)

SEASONAL_NAIVE = _spec(
    model_id="seasonal_naive",
    display_name="Seasonal naive",
    family=ModelFamily.BASELINE,
    is_baseline=True,
    description="Forecast ŷ(t+h) = y(t+h−m) at the seasonal period m.",
    plain_description="Assume next month looks like the same month last year.",
    required=ModelRequirements(min_n=10),
    capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
    triggered_when="Always where a seasonal period exists — the second mandatory benchmark.",
    assumption_checks=["needs at least two full cycles for the interval to mean anything"],
)

DRIFT = _spec(
    model_id="drift",
    display_name="Random walk with drift",
    family=ModelFamily.BASELINE,
    is_baseline=True,
    description="Naive plus a constant slope estimated from the first and last values.",
    plain_description="Today's value, plus the average change per step, repeated forward.",
    required=ModelRequirements(min_n=5),
    capabilities=ModelCapabilities(supports_intervals=True),
    triggered_when="Always — the cheapest model that can extrapolate a trend.",
    assumption_checks=["a single global slope is a strong claim on a long series"],
)

WHITE_NOISE = _spec(
    model_id="white_noise",
    display_name="White noise (no model)",
    family=ModelFamily.BASELINE,
    is_baseline=True,
    description="i.i.d.(μ, σ²): forecast μ with a constant interval, and say so.",
    plain_description=(
        "The honest answer when there is no pattern: predict the average and state that no model "
        "is needed. A tool that cannot say this is overfitting."
    ),
    required=ModelRequirements(min_n=10),
    capabilities=ModelCapabilities(supports_intervals=True, supports_probabilistic=True),
    triggered_when="Whenever the residual autocorrelation battery finds nothing at any lag.",
    assumption_checks=["Ljung–Box must fail to reject at every tested lag"],
)

# --- B. Mean / conditional-mean family -------------------------------------- #
ARIMA = _spec(
    model_id="arima",
    display_name="ARIMA(p,d,q)",
    family=ModelFamily.ARMA,
    description=(
        "Auto-selected ARIMA by information criterion over a bounded grid, with d fixed by the "
        "stationarity verdict. Includes the pure AR, MA and ARMA cases as special cases."
    ),
    plain_description=(
        "The workhorse: each value depends on recent values and recent errors. The orders are "
        "chosen by IC, not by taste."
    ),
    required=ModelRequirements(min_n=20),
    capabilities=ModelCapabilities(supports_intervals=True, supports_probabilistic=True),
    order_params=["p", "d", "q"],
    triggered_when="Continuous data with autocorrelation and no strong seasonality.",
    assumption_checks=["stationarity/invertibility roots", "residual whiteness", "constant variance"],
    citations=["Hyndman & Athanasopoulos, FPP3 §9"],
)

SARIMA = _spec(
    model_id="sarima",
    display_name="SARIMA(p,d,q)(P,D,Q)m",
    family=ModelFamily.ARMA,
    description="ARIMA with a seasonal AR/MA structure at period m; D from the seasonal unit-root test.",
    plain_description=(
        "ARIMA with a seasonal memory: each value also depends on the same season in prior years."
    ),
    required=ModelRequirements(min_n=30),
    capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
    order_params=["p", "d", "q", "P", "D", "Q"],
    triggered_when="A seasonal period exists and the seasonal unit-root test called for a seasonal difference.",
    assumption_checks=["stationarity/invertibility roots", "residual whiteness", "≥ 2 full cycles"],
    citations=["Hyndman & Athanasopoulos, FPP3 §9.9"],
)

ARFIMA = _spec(
    model_id="arfima",
    display_name="ARFIMA (fractional d)",
    family=ModelFamily.ARMA,
    description="Fractionally integrated ARMA; d̂ estimated by GPH from the periodogram slope.",
    plain_description=(
        "For series whose memory decays slowly but is not a full unit root — d between 0 and 0.5. "
        "Its AIC is not comparable with ARIMA's, and the app says so."
    ),
    required=ModelRequirements(min_n=50),
    capabilities=ModelCapabilities(supports_intervals=True),
    order_params=["p", "d_frac", "q"],
    is_default_candidate=False,
    triggered_when="The ACF decays hyperbolically and the GPH estimate of d lands in (0, 0.5).",
    assumption_checks=["GPH estimate with its standard error", "AIC is not comparable with ARMA"],
    citations=["Geweke & Porter-Hudak (1983)"],
)

ETS = _spec(
    model_id="ets",
    display_name="ETS / Holt–Winters",
    family=ModelFamily.SMOOTHING,
    description="Exponential smoothing (error × trend × season), selected by AICc over the valid grid.",
    plain_description=(
        "Weighted averages that put more weight on recent observations, with optional trend and "
        "season terms."
    ),
    required=ModelRequirements(min_n=20),
    capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
    order_params=["error", "trend", "season", "damped"],
    triggered_when="Continuous data; especially strong trend or seasonality with modest n.",
    assumption_checks=["multiplicative components need strictly positive data", "parameter bounds"],
    citations=["Hyndman et al., Forecasting with Exponential Smoothing"],
)

THETA = _spec(
    model_id="theta",
    display_name="Theta method",
    family=ModelFamily.SMOOTHING,
    description="The M3/M4-winning theta decomposition with an optimally chosen θ.",
    plain_description=(
        "Averages several smoothed versions of the series, one of them extrapolated with a trend. "
        "Unfashionable and very hard to beat on seasonal data."
    ),
    required=ModelRequirements(min_n=20),
    capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
    order_params=["theta"],
    triggered_when="Stable, strongly seasonal series, especially with short samples.",
    assumption_checks=["series must be non-degenerate"],
    citations=["Assimakopoulos & Nikolopoulos (2000)"],
)

# --- C. Count and intermittent (§5.8 C) ------------------------------------ #
POISSON = _spec(
    model_id="poisson",
    display_name="Poisson regression (trend + season)",
    family=ModelFamily.COUNT,
    description="GLM with a log link, a linear trend and harmonic terms at the detected period.",
    plain_description=(
        "Counts are modelled as a rate: the forecast is an average count per period, and the "
        "interval comes from the Poisson distribution itself."
    ),
    required=ModelRequirements(min_n=20, requires_non_negative=True),
    capabilities=ModelCapabilities(
        supports_seasonality=True, supports_intervals=True, supports_probabilistic=True,
        discrete_predictive=True,
    ),
    triggered_when="Count data whose variance is close to its mean.",
    assumption_checks=["overdispersion ratio must be near 1 — otherwise use the negative binomial"],
)

NEG_BINOMIAL = _spec(
    model_id="neg_binomial",
    display_name="Negative binomial (NB2)",
    family=ModelFamily.COUNT,
    description="Log-link GLM with a quadratic mean–variance relation; α estimated by MLE.",
    plain_description=(
        "Like the Poisson, but allowed to be noisier than its own mean — which is what most real "
        "count data is."
    ),
    required=ModelRequirements(min_n=20, requires_non_negative=True),
    capabilities=ModelCapabilities(
        supports_seasonality=True, supports_intervals=True, supports_probabilistic=True,
        discrete_predictive=True,
    ),
    triggered_when="Count data with variance clearly above the mean (overdispersion).",
    assumption_checks=["dispersion ratio > 1", "α estimable away from its boundary"],
)

ZERO_INFLATED = _spec(
    model_id="zero_inflated",
    display_name="Zero-inflated count (hurdle)",
    family=ModelFamily.COUNT,
    description="Two-part model: a process that decides whether an event happened, then a count.",
    plain_description=(
        "Separates 'nothing happened' from 'something happened, this many times'. Only right when "
        "the zeros are structural, not when the rate is simply low."
    ),
    required=ModelRequirements(min_n=30, requires_non_negative=True),
    capabilities=ModelCapabilities(supports_intervals=True, discrete_predictive=True),
    is_default_candidate=False,
    triggered_when="The observed zero share is far above what the mean rate would produce.",
    assumption_checks=["zero excess computed against the Poisson rate"],
)

CROSTON = _spec(
    model_id="croston",
    display_name="Croston (SBA)",
    family=ModelFamily.INTERMITTENT,
    description="Separate exponential smoothing of demand sizes and inter-arrival intervals, "
    "with the Syntetos–Boylan bias correction.",
    plain_description=(
        "For spare parts and other things that are usually zero: it smooths *when* demand happens "
        "and *how much* separately, so the zero months do not drag the forecast down."
    ),
    required=ModelRequirements(min_n=12, requires_non_negative=True, max_pct_zeros=1.0),
    capabilities=ModelCapabilities(supports_intervals=True),
    triggered_when="Intermittent series: mostly zeros with occasional non-zero demand.",
    assumption_checks=["≥ 3 non-zero observations", "MAPE is banned for this model"],
    citations=["Syntetos & Boylan (2005)"],
)

TSB = _spec(
    model_id="tsb",
    display_name="TSB (Teunter–Syntetos–Babai)",
    family=ModelFamily.INTERMITTENT,
    description="Croston with a smoothed occurrence probability that decays when demand stops.",
    plain_description=(
        "Like Croston, but it notices when demand has stopped entirely — which Croston never does."
    ),
    required=ModelRequirements(min_n=12, requires_non_negative=True),
    capabilities=ModelCapabilities(supports_intervals=True),
    triggered_when="Intermittent series with a risk of obsolescence (long zero runs).",
    assumption_checks=["≥ 3 non-zero observations"],
    citations=["Teunter, Syntetos & Babai (2011)"],
)

# --- Part 10: serial dependence in counts, lumpy demand, and event data ------ #
INGARCH = _spec(
    feature_flag="advanced",  # §5.8 "optional, flagged; report as advanced"
    model_id="ingarch",
    display_name="INGARCH(1,1)",
    family=ModelFamily.COUNT,
    description="Integer GARCH: the conditional count rate follows its own recursion in the last count "
    "and the last rate.",
    plain_description=(
        "The count analogue of a volatility model: a busy period raises the rate for the next few, so "
        "outbreaks cluster instead of scattering."
    ),
    required=ModelRequirements(min_n=30, requires_count=True, requires_non_negative=True),
    capabilities=ModelCapabilities(supports_intervals=True, discrete_predictive=True),
    triggered_when="Counts are autocorrelated — the residual ACF in Step 6 does not die out.",
    assumption_checks=[
        "conditional count distribution matches the dispersion (Poisson or NB2)",
        "persistence alpha+beta below 1",
    ],
    citations=["Ferland, Latour & Oraichi (2006); Heinen (2003)"],
)

INAR = _spec(
    model_id="inar",
    display_name="INAR(1)",
    family=ModelFamily.COUNT,
    description="Binomial-thinning autoregression: each count survives to the next period with a "
    "probability, and new arrivals add to it.",
    plain_description=(
        "Says what share of one period's cases is still present the next period — a mechanism you can "
        "interpret, not just a fit."
    ),
    required=ModelRequirements(min_n=30, requires_count=True, requires_non_negative=True),
    capabilities=ModelCapabilities(supports_intervals=True, discrete_predictive=True),
    is_default_candidate=False,
    triggered_when="Counts with serial dependence where persistence has a population interpretation.",
    assumption_checks=[
        "thinning probability identifiable away from 1",
        "AIC is on a Gaussian quasi-likelihood, so NOT comparable with the Poisson models",
    ],
    citations=["McKenzie (1985); Al-Osh & Alzaid (1987)"],
)

ADIDA = _spec(
    model_id="adida",
    display_name="ADIDA (aggregate–disaggregate)",
    family=ModelFamily.INTERMITTENT,
    description="Aggregates lumpy demand into blocks, smooths the block series, then disaggregates.",
    plain_description=(
        "For demand that is both rare and erratic: adds periods together until there is about one "
        "demand per block, forecasts that easier series, then divides back down."
    ),
    required=ModelRequirements(min_n=12, requires_non_negative=True),
    capabilities=ModelCapabilities(supports_intervals=True),
    is_default_candidate=False,
    triggered_when="Lumpy intermittent demand, where demand sizes vary too much for one smoothed average.",
    assumption_checks=["at least 4 aggregate blocks", "MAPE is banned for this model"],
    citations=["Nikolopoulos et al. (2011)"],
)

CROSTON_SY = _spec(
    model_id="croston_sy",
    display_name="Croston (SY-corrected)",
    family=ModelFamily.INTERMITTENT,
    description="Croston with the Syntetos–Boylan correction applied only above the 1.32-interval "
    "threshold.",
    plain_description=(
        "Croston again, but the bias correction is skipped when demand is frequent enough that "
        "correcting it would understate the forecast."
    ),
    required=ModelRequirements(min_n=12, requires_non_negative=True),
    capabilities=ModelCapabilities(supports_intervals=True),
    is_default_candidate=False,
    triggered_when="Intermittent demand whose mean interval sits near the SBA correction's cut-off.",
    assumption_checks=["mean interval reported so the correction's applicability is visible"],
    citations=["Syntetos & Boylan (2005)"],
)

RENEWAL = _spec(
    feature_flag="advanced",  # §5.8 "flagged; report rate and hazard"
    model_id="renewal",
    display_name="Renewal / inter-arrival",
    family=ModelFamily.POINT_PROCESS,
    description="Exponential or Weibull renewal process fitted to the gaps between events.",
    plain_description=(
        "Models how long it usually is between events, and whether waiting longer makes the next one "
        "more or less likely."
    ),
    required=ModelRequirements(min_n=10),
    capabilities=ModelCapabilities(supports_intervals=True, discrete_predictive=True),
    triggered_when="The dataset is a list of event dates rather than values.",
    assumption_checks=["at least 10 usable intervals", "count over the horizon approximated as Poisson"],
    citations=["Cox (1962); Weibull (1951)"],
)

HAWKES = _spec(
    feature_flag="advanced",  # §5.8 "flagged; requires event-level input mode"
    model_id="hawkes",
    display_name="Hawkes (self-exciting)",
    family=ModelFamily.POINT_PROCESS,
    description="Self-exciting point process with an exponential kernel; estimates the branching ratio.",
    plain_description=(
        "Asks whether events cause more events — aftershocks, contagion — and reports what share of "
        "the activity is self-generated."
    ),
    required=ModelRequirements(min_n=10),
    capabilities=ModelCapabilities(supports_intervals=True, discrete_predictive=True),
    is_default_candidate=False,
    triggered_when="Event dates that cluster in bursts far more than a constant rate would produce.",
    assumption_checks=[
        "branching ratio below 0.95 (near-critical is not identified)",
        "clustering attributed to self-excitation cannot be distinguished from an unobserved driver",
    ],
    citations=["Hawkes (1971); Ogata (1988)"],
)

# --- Rows this build cannot fit yet ----------------------------------------- #

MARKOV = _spec(
    model_id="markov_switching",
    display_name="Markov-switching AR",
    family=ModelFamily.NONLINEAR,
    description="Two-regime autoregression: separate mean, variance and persistence per regime, with "
    "a labelled regime history.",
    plain_description=(
        "Finds periods that behaved differently from the rest — calm versus turbulent, say — and "
        "labels them, so you can check the labels against what actually happened."
    ),
    required=ModelRequirements(min_n=40),
    capabilities=ModelCapabilities(supports_intervals=True),
    is_default_candidate=False,
    triggered_when="Volatility clustering, or a level that appears to switch between two states.",
    assumption_checks=[
        "both regimes own at least 5% of the sample",
        "two regimes is an assumption, not a finding",
    ],
    citations=["Hamilton (1989)"],
)

# --- Rows this build cannot fit yet ----------------------------------------- #
# Kept in the registry because the *suggestion* is useful now even though the fit is not built
# yet: a user who needs a GARCH model deserves to be told this app has not built one rather than
# silently seeing only ARIMA. `feature_flag` carries "not yet implemented", and `build.py` refuses
# to fit them (`NOT_IMPLEMENTED`).
def _pending(**kw) -> ModelSpec:
    kw.setdefault("is_default_candidate", False)
    kw.setdefault("feature_flag", "pending")
    return _spec(**kw)


PENDING = [
    # -- Part 9: exogenous and structural models ---------------------------- #
    # These three *were* pending rows in Part 8 and are fitted now. The id, family and trigger are
    # unchanged, so the excluded-list prose from Part 8 ("planned for part 9") becomes a fitted model
    # with no engine change anywhere — which is the point of the registry being data.
    _spec(
        model_id="arimax",
        display_name="SARIMAX / dynamic regression",
        family=ModelFamily.EXOGENOUS,
        description="ARIMA errors with exogenous regressors, intervention dummies and scenario paths.",
        plain_description=(
            "A model that also uses outside information — promotions, weather, holidays — alongside "
            "the series' own history."
        ),
        required=ModelRequirements(min_n=30),
        capabilities=ModelCapabilities(supports_exogenous=True, supports_seasonality=True,
                                       supports_intervals=True),
        is_default_candidate=False,
        triggered_when="The dataset carries a numeric column that could drive the target.",
        assumption_checks=[
            "future driver paths must be supplied, not invented",
            "a trending driver needs the spurious-regression caveat",
        ],
        citations=["Box, Jenkins & Reinsel (2015); Hyndman & Athanasopoulos (2021) §9.2"],
    ),
    _spec(
        model_id="ucm",
        display_name="Unobserved components (local level / BSM)",
        family=ModelFamily.STRUCTURAL,
        description="Kalman-filtered structural model with interpretable level, trend and seasonal states.",
        plain_description=(
            "Splits the series into a level, a trend and a season, each with its own noise — so the "
            "split itself is an output you can look at."
        ),
        required=ModelRequirements(min_n=20, allows_missing=True),
        capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True,
                                       supports_missing_values=True),
        triggered_when=(
            "Missing values that must be handled natively, or a need for smooth, interpretable "
            "intervals and a stated level."
        ),
        assumption_checks=["state variance estimable", "convergence"],
        citations=["Harvey (1989); Commandeur & Koopman (2007)"],
    ),
    _spec(
        model_id="ardl",
        display_name="ARDL / dynamic regression with ARMA errors",
        family=ModelFamily.EXOGENOUS,
        description="Lagged regression on the target and a driver, with a bounds test for a long-run "
        "relationship.",
        plain_description=(
            "Asks whether the target and a driver move together in the long run as well as the short "
            "run — a different question from whether the driver predicts."
        ),
        required=ModelRequirements(min_n=30),
        capabilities=ModelCapabilities(supports_exogenous=True, supports_intervals=True),
        is_default_candidate=False,
        triggered_when="A driver column exists and the user needs a long-run relationship test.",
        assumption_checks=[
            "no p-value on the bounds test — the tabulated band is the reference",
            "recursive multi-step forecasts compound their own errors",
        ],
        citations=["Pesaran, Shin & Smith (2001)"],
    ),
    _spec(
        model_id="var",
        display_name="VAR",
        family=ModelFamily.MULTIVARIATE,
        description=(
            "Vector autoregression over the target and its mapped companion series, with Granger "
            "tests, bootstrap-banded impulse responses, a forecast-error variance decomposition and "
            "a stability check."
        ),
        plain_description=(
            "Models several series together so each can forecast the others, and shows how a shock to "
            "one travels through the rest."
        ),
        required=ModelRequirements(min_n=40),
        capabilities=ModelCapabilities(supports_exogenous=True, supports_intervals=True),
        # Opt-in: a VAR needs a second mapped column, and offering it by default on a single-series
        # dataset produces a row that can only fail.
        is_default_candidate=False,
        triggered_when=(
            "Two or more numeric columns are mapped, so there is a second series for the target to "
            "move with."
        ),
        assumption_checks=[
            "stability: every companion root inside the unit circle",
            "cointegration tested first — a level VAR on unrelated random walks is spurious",
            "Cholesky ordering stated with the FEVD, because it changes the answer",
            "lag order chosen by BIC where it disagrees with AIC",
        ],
        citations=["Sims (1980)", "Granger (1969)", "Lütkepohl (2005)"],
    ),
    _spec(
        model_id="vecm",
        display_name="VECM (Johansen)",
        family=ModelFamily.MULTIVARIATE,
        description=(
            "Vector error correction: short-run dynamics plus an estimated long-run relationship, "
            "with the adjustment speeds that say whether the relationship actually pulls."
        ),
        plain_description=(
            "For series that wander but stay tied together: estimates the long-run relationship and "
            "how fast a deviation from it is corrected."
        ),
        required=ModelRequirements(min_n=50),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Johansen finds at least one cointegrating relation at the 5% level.",
        assumption_checks=[
            "rank chosen at the 5% level from the trace test",
            "implied level system has exactly k − rank unit roots",
            "at least one series must adjust toward the relation with a negative speed",
        ],
        citations=["Johansen (1988)", "Engle & Granger (1987)"],
    ),
    _spec(
        model_id="gradient_boost",
        display_name="Gradient-boosted trees",
        family=ModelFamily.ML,
        description=(
            "Gradient boosting on leakage-safe lag and calendar features, with a time-ordered holdout "
            "and expanding folds."
        ),
        plain_description=(
            "A machine-learning model that learns patterns from recent values and the calendar. "
            "Opt-in, and always shown next to the classical models it has to beat."
        ),
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Opt-in: a long series where a flexible tabular model might beat ARIMA.",
        assumption_checks=[
            "features are lags, calendar harmonics and lag-1 drivers only — no leakage",
            "multi-step is recursive, and declared",
            "validation is a time-ordered holdout plus expanding folds, never a shuffled split",
        ],
    ),
    _spec(
        model_id="random_forest",
        display_name="Random forest",
        family=ModelFamily.ML,
        description="Bagged regression trees on the same leakage-safe lag design.",
        plain_description="An average of many decision trees, each trained on a different slice of the history.",
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Opt-in, alongside the other ML rows.",
        assumption_checks=["leakage-safe features", "time-ordered validation"],
    ),
    _spec(
        model_id="knn",
        display_name="k-nearest neighbours",
        family=ModelFamily.ML,
        description="Nearest-neighbour regression in lag space.",
        plain_description="Finds past periods that looked like today and averages what happened next.",
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Opt-in; useful as a deliberately simple nonlinear benchmark.",
        assumption_checks=[
            "features standardised on the training block only",
            "accuracy is measured on expanding-window refits, because k-NN reproduces its training rows",
        ],
    ),
    _spec(
        model_id="svr",
        display_name="Support vector regression",
        family=ModelFamily.ML,
        description="RBF-kernel SVR on lag features.",
        plain_description="Fits a smooth nonlinear curve through the recent history, ignoring small errors.",
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Opt-in, alongside the other ML rows.",
        assumption_checks=["leakage-safe features", "time-ordered validation"],
    ),
    _spec(
        model_id="nnar",
        display_name="Neural network AR (NNAR)",
        family=ModelFamily.ML,
        description=(
            "A single-hidden-layer tanh network over lag features. Not a recurrent network: no deep "
            "learning framework is installed in this build, and the row is named for what it is."
        ),
        plain_description="A small neural network that learns a nonlinear autoregression.",
        required=ModelRequirements(min_n=80),
        capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Opt-in, and only where the classical models have been fitted too.",
        assumption_checks=[
            "deterministic solver with a fixed seed, so the same data gives the same answer",
            "solver convergence warnings surfaced as caveats rather than printed to a log",
        ],
    ),
    _spec(
        model_id="gaussian_process",
        display_name="Gaussian process",
        family=ModelFamily.ML,
        description="GP regression with an RBF + white-noise kernel on lag features.",
        plain_description=(
            "A model that treats the forecast as a distribution over smooth curves, so its uncertainty "
            "comes from the fit itself."
        ),
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_intervals=True, supports_seasonality=True),
        is_default_candidate=False,
        triggered_when="Opt-in; strongest of the tabular rows on smooth seasonal series.",
        assumption_checks=[
            "cubic cost: the series is subsampled to 400 rows beyond that, and says so",
            "kernel hyperparameters optimised by marginal likelihood with one restart",
        ],
    ),
    _spec(
        model_id="setar",
        display_name="SETAR (2-regime threshold AR)",
        family=ModelFamily.NONLINEAR,
        description=(
            "Two-branch autoregression split at an estimated threshold on the series' own past, with a "
            "linearity test and a threshold interval."
        ),
        plain_description=(
            "Behaves one way below a level and differently above it — a floor, a ceiling, or a point "
            "where something starts reacting."
        ),
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when="A level that seems to change the dynamics, or asymmetric mean reversion.",
        assumption_checks=[
            "both branches own at least 5% of the sample",
            "the linearity test's p-value is a lower bound, because the threshold was estimated",
            "threshold reported with a 2-AIC interval, not as a constant",
        ],
        citations=["Tong & Lim (1980)", "Hansen (1999)"],
    ),
    _pending(
        model_id="lightgbm",
        planned_part=12,
        display_name="LightGBM",
        family=ModelFamily.ML,
        description="Gradient-boosted trees on lag features, with leakage-safe construction.",
        plain_description="A machine-learning model that learns patterns from past values and calendar features.",
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_exogenous=True, supports_seasonality=True),
        triggered_when="Long series with many potential drivers; opt-in only.",
        assumption_checks=["leakage-safe features", "must be benchmarked against ETS/ARIMA"],
    ),
    _pending(
        model_id="xgboost",
        planned_part=12,
        display_name="XGBoost",
        family=ModelFamily.ML,
        description="Gradient-boosted trees on the same leakage-safe lag design.",
        plain_description="A machine-learning model that learns patterns from recent values and the calendar.",
        required=ModelRequirements(min_n=60),
        capabilities=ModelCapabilities(supports_exogenous=True, supports_seasonality=True),
        triggered_when="Opt-in, alongside the other ML rows.",
        assumption_checks=["leakage-safe features", "time-ordered validation"],
    ),
    _pending(
        model_id="lstm",
        planned_part=12,
        display_name="LSTM / GRU",
        family=ModelFamily.ML,
        description="Recurrent neural network on the lag sequence.",
        plain_description="A neural network that carries a memory of the recent past.",
        required=ModelRequirements(min_n=120),
        capabilities=ModelCapabilities(supports_seasonality=True, supports_intervals=True),
        triggered_when="Opt-in, and only where the classical models have been fitted too.",
        assumption_checks=["needs a deep-learning framework", "leakage-safe feature construction"],
    ),
    _spec(
        model_id="combination",
        display_name="Forecast combination",
        family=ModelFamily.COMBINATION,
        description=(
            "Mean, median, IC-weighted or Granger–Ramanathan combination of the other fitted "
            "candidates. Weights are estimated on the components' held-out forecasts, never on their "
            "in-sample fits."
        ),
        plain_description=(
            "Averages the forecasts of the good models; usually beats choosing one. The averaging rule "
            "is picked in Step 13."
        ),
        required=ModelRequirements(min_n=20, min_two_models=True),
        capabilities=ModelCapabilities(supports_intervals=True),
        triggered_when="Once at least two other models have fitted — it has nothing to combine otherwise.",
        assumption_checks=[
            "the components must differ meaningfully; averaging six near-identical forecasts is one forecast",
            "its weights need held-out forecasts, so it is built after cross-validation",
        ],
    ),

    # --- Part 11: conditional variance (§5.8 D) --------------------------- #
    # Seven fitted rows and two honest refusals. The family exists because a series can be perfectly
    # predictable in its *level* and wildly uncertain in its *spread*: a GARCH fit answers "how sure
    # are we", which no mean model in this registry reports.
    _spec(
        model_id="arch",
        display_name="ARCH(q)",
        family=ModelFamily.VOLATILITY,
        description=(
            "Engle's ARCH: variance depends only on recent squared shocks, with no memory of the "
            "variance itself."
        ),
        plain_description=(
            "Says that a big move makes the next few periods more uncertain, with no memory beyond "
            "the last shock."
        ),
        required=ModelRequirements(min_n=80),
        capabilities=ModelCapabilities(supports_intervals=True),
        triggered_when="ARCH-LM rejects and the variance shows short-lived bursts.",
        assumption_checks=[
            "ARCH effects present in the *differenced* series, not only the level",
            "no variance term, so a persistent series will be under-modelled",
        ],
        citations=["Engle (1982)"],
    ),
    _spec(
        model_id="garch",
        display_name="GARCH(p,q)",
        family=ModelFamily.VOLATILITY,
        description=(
            "Bollerslev's GARCH(1,1): variance follows its own recursion as well as the last squared "
            "shock."
        ),
        plain_description=(
            "Models how uncertain the series is, not what its next value will be: a shock raises "
            "uncertainty, and uncertainty fades back to a long-run level."
        ),
        required=ModelRequirements(min_n=80),
        capabilities=ModelCapabilities(supports_intervals=True),
        triggered_when="ARCH-LM rejects and the effects are persistent.",
        assumption_checks=[
            "ARCH effects present",
            "persistence below 1 at which the long-run variance does not exist",
            "standardised residuals not still clustered",
        ],
        citations=["Bollerslev (1986)"],
    ),
    _spec(
        model_id="garch_t",
        display_name="GARCH-t",
        family=ModelFamily.VOLATILITY,
        description="GARCH(1,1) with Student-t innovations; nu estimated.",
        plain_description=(
            "The same variance model, with fat tails allowed for, so the rare large move is not "
            "treated as impossible."
        ),
        required=ModelRequirements(min_n=80),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Residual kurtosis says the Gaussian version will understate the tail.",
        assumption_checks=[
            "nu above 2, and above 30 means it is indistinguishable from normal",
        ],
        citations=["Bollerslev (1987)"],
    ),
    _spec(
        model_id="gjr",
        display_name="GJR-GARCH",
        family=ModelFamily.VOLATILITY,
        description=(
            "Asymmetric variance: a negative shock raises future volatility more than a positive "
            "shock of the same size."
        ),
        plain_description=(
            "Allows bad news to raise uncertainty more than good news does, the leverage effect."
        ),
        required=ModelRequirements(min_n=80),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when=(
            "Symmetric GARCH residuals still show structure, or falls look more volatile than rises."
        ),
        assumption_checks=[
            "gamma estimated and reported; a gamma near zero means the asymmetry is not there",
            "variance recursion stays positive at every step",
        ],
        citations=["Glosten, Jagannathan & Runkle (1993)"],
    ),
    _spec(
        model_id="egarch",
        display_name="EGARCH",
        family=ModelFamily.VOLATILITY,
        description=(
            "Nelson's exponential GARCH: the log of the variance is modelled, so it cannot be "
            "negative and the leverage effect enters with a sign."
        ),
        plain_description=(
            "The same leverage idea as GJR, written so that uncertainty can never come out negative."
        ),
        required=ModelRequirements(min_n=80),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when="Leverage matters and a positivity guarantee is wanted.",
        assumption_checks=[
            "log-variance recursion stays finite (clipped at +/-30)",
            "gamma's sign is the leverage direction",
        ],
        citations=["Nelson (1991)"],
    ),
    _spec(
        model_id="arima_garch",
        display_name="ARIMA-GARCH",
        family=ModelFamily.VOLATILITY,
        description=(
            "Mean dynamics from an AR equation and conditional variance from GARCH, estimated "
            "jointly; both paths are returned separately."
        ),
        plain_description=(
            "Two forecasts from one fit: what the next values will be, and how uncertain each one is."
        ),
        required=ModelRequirements(min_n=80),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when="The mean has its own dynamics *and* the variance clusters.",
        assumption_checks=[
            "two parameter blocks estimated jointly, so the mean's AR order matters",
            "the variance path is reported alongside the mean path, never merged into it",
        ],
        citations=["Bollerslev (1986); Engle (2001)"],
    ),
    _spec(
        model_id="gjr_t",
        display_name="GJR-GARCH-t",
        family=ModelFamily.VOLATILITY,
        description="GJR-GARCH with Student-t innovations: asymmetry and fat tails together.",
        plain_description=(
            "Both refinements at once: bad news raises uncertainty more, and the rare large move is "
            "allowed for."
        ),
        required=ModelRequirements(min_n=120),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when="The Gaussian asymmetric fit leaves fat-tailed standardised residuals.",
        assumption_checks=[
            "needs enough data for both an asymmetry and a tail parameter",
            "gamma and nu are reported separately",
        ],
        citations=["Glosten, Jagannathan & Runkle (1993); Bollerslev (1987)"],
    ),
    _spec(
        model_id="egarch_t",
        display_name="EGARCH-t",
        family=ModelFamily.VOLATILITY,
        description="EGARCH with Student-t innovations.",
        plain_description="Leverage with a guaranteed-positive variance and fat tails allowed for.",
        required=ModelRequirements(min_n=120),
        capabilities=ModelCapabilities(supports_intervals=True),
        is_default_candidate=False,
        triggered_when="The Gaussian EGARCH leaves fat-tailed standardised residuals.",
        assumption_checks=["log-variance recursion stays finite", "nu above 2"],
        citations=["Nelson (1991); Bollerslev (1987)"],
    ),
    _pending(
        model_id="sv",
        planned_part=11,
        display_name="Stochastic volatility",
        family=ModelFamily.VOLATILITY,
        description="Latent variance process estimated by MCMC.",
        plain_description="A slower-moving, unobserved uncertainty process.",
        required=ModelRequirements(min_n=100),
        capabilities=ModelCapabilities(supports_intervals=True),
        triggered_when="Slow-moving variance with no ARCH signature.",
        assumption_checks=["needs a Bayesian fitting layer"],
    ),
    _pending(
        model_id="realized",
        planned_part=11,
        display_name="Realized volatility",
        family=ModelFamily.VOLATILITY,
        description="Variance computed from intraday returns.",
        plain_description="Volatility measured from within-day prices rather than estimated.",
        required=ModelRequirements(min_n=50),
        triggered_when="Intraday return data is available.",
        assumption_checks=["needs an intraday panel, which this app does not ingest"],
    ),
]

#: Rows that are pending for a *measured* reason rather than a schedule. Imported lazily from the
#: families that produced the measurement, so the registry never restates a fact it did not check.
def _measured_refusals() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        from app.ts.models.ml import UNIMPLEMENTED_ML

        out.update(UNIMPLEMENTED_ML)
    except Exception:  # pragma: no cover - the ML module must import, but a registry must survive
        pass
    try:
        from app.ts.models.multivariate import UNIMPLEMENTED_MULTIVARIATE

        out.update(UNIMPLEMENTED_MULTIVARIATE)
    except Exception:  # pragma: no cover
        pass
    # Part 11's two refusals, stated with the reason the engine has carried since it shipped.
    out.setdefault("sv", _sv_reason())
    out.setdefault("realized", _realized_reason())
    return out


def _sv_reason() -> str:
    return (
        "it needs a latent variance process estimated by MCMC over a state space, and this build has "
        "no Bayesian fitting layer — the ARCH/GARCH family covers the same question with a recursion "
        "that can be written down and checked"
    )


def _realized_reason() -> str:
    return (
        "it computes variance from intraday returns, and this app ingests one value per period — there "
        "are no within-period observations to compute a realized measure from"
    )


_MEASURED_REFUSAL: dict[str, str] = _measured_refusals()

REGISTRY: dict[str, ModelSpec] = {s.model_id: s for s in _SPECS}

#: The ids the registry itself declares as not-yet-fitted. Kept for introspection only — the live
#: answer is `is_implemented()`, which reads the row's `feature_flag`. A second, cached copy of
#: "what is available" is exactly the duplicated fact that goes stale (this one did, the first time a
#: row was registered at runtime).
NOT_IMPLEMENTED: set[str] = {s.model_id for s in PENDING}

#: Baselines are compulsory: the comparison is meaningless without them.
BASELINE_IDS: tuple[str, ...] = tuple(s.model_id for s in _SPECS if s.is_baseline)


def is_implemented(model_id: str) -> bool:
    """
    Can this build fit the model *as currently registered*?

    A row registered at runtime (or a plugin row) that is not marked `pending` is fit-able, even if a
    placeholder row with that id existed earlier — the registry is the source of truth, and a
    second copy of "what is available" is the kind of duplicated fact that goes stale.
    """
    spec = REGISTRY.get(model_id)
    if spec is None:
        return False
    return spec.feature_flag != "pending"


def get_spec(model_id: str) -> ModelSpec:
    if model_id not in REGISTRY:
        raise KeyError(
            f"'{model_id}' is not in the model registry. Known ids: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[model_id]


def register(spec: ModelSpec) -> None:
    """
    Add a row at runtime.

    This exists so the "adding a model is a row" claim can be *tested* rather than asserted. It is
    also the hook a plugin would use; nothing in the suggestion engine needs to change.
    """
    _SPECS.append(spec)
    REGISTRY[spec.model_id] = spec


def all_specs() -> list[ModelSpec]:
    return list(_SPECS)


def implemented_specs() -> list[ModelSpec]:
    return [s for s in _SPECS if is_implemented(s.model_id)]


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    """Why a model is or is not offered. Both lists are shown to the user."""

    compatible: bool
    reasons: list[str]
    warnings: list[str]


def _fmt(x: float | None, dp: int = 2) -> str:
    return "—" if x is None else f"{x:.{dp}f}"


def evaluate_spec(spec: ModelSpec, f: DataFacts) -> Verdict:
    """
    Query one registry row against the measured facts.

    Every refusal names the number that caused it. "Not suitable for this data" teaches the user
    nothing; "the series contains 3 negative values, and a Poisson rate cannot be negative" is the
    kind of sentence that makes the next decision obvious.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    # --- data class ------------------------------------------------------- #
    # Every row is checked, including the ones this build cannot fit: a user reading the excluded
    # list should see the real reason (a GARCH model on a continuous series *is* applicable; it is
    # missing, not wrong) rather than a generic "not available".
    if not spec.supports_data_class(f.data_class):
        reasons.append(
            f"it belongs to the {spec.family.value} family, which does not apply to "
            f"{f.data_class.value.replace('_', ' ')} data"
        )

    # --- length ----------------------------------------------------------- #
    if f.n_non_missing < spec.required.min_n:
        reasons.append(
            f"it needs at least {spec.required.min_n} observations and there are {f.n_non_missing}"
        )

    # --- positivity / sign ------------------------------------------------ #
    if spec.required.requires_positive and (f.min_value is not None and f.min_value <= 0):
        reasons.append(
            f"it requires strictly positive values and the minimum is {_fmt(f.min_value, 3)}"
        )
    if spec.required.requires_non_negative and (f.min_value is not None and f.min_value < 0):
        reasons.append(
            f"it requires non-negative values and the minimum is {_fmt(f.min_value, 3)}"
        )

    # --- counts ----------------------------------------------------------- #
    if spec.family is ModelFamily.COUNT and not f.is_integer_valued:
        reasons.append(
            "it models integer counts and this series has fractional values"
        )
    if spec.model_id in {"ingarch", "inar"}:
        acf1 = f.autocorr_lag1
        if f.data_class not in {DataClass.COUNT, DataClass.INTERMITTENT}:
            reasons.append(
                f"it models counts that depend on earlier counts, and this series is "
                f"{f.data_class.value.replace('_', ' ')} — there are no count values to recurse on"
            )
        elif acf1 is not None and abs(acf1) < 0.20:
            warnings.append(
                f"lag-1 autocorrelation of the counts is {_fmt(acf1)}, so there is almost no serial "
                "dependence for the recursion to capture — the plain Poisson regression is simpler"
            )
    if spec.model_id == "poisson" and f.dispersion_ratio is not None:
        if f.dispersion_ratio > 1.5:
            warnings.append(
                f"variance is {_fmt(f.dispersion_ratio)}× the mean, so the data is overdispersed and "
                "a Poisson interval will be too narrow — the negative binomial is the honest choice"
            )
        elif f.dispersion_ratio < 0.6:
            warnings.append(
                f"variance is {_fmt(f.dispersion_ratio)}× the mean — underdispersed, which a Poisson "
                "cannot represent either"
            )
    if spec.model_id == "neg_binomial" and f.dispersion_ratio is not None:
        if f.dispersion_ratio <= 1.2:
            warnings.append(
                f"variance is only {_fmt(f.dispersion_ratio)}× the mean, so the extra parameter buys "
                "little over a Poisson and costs one degree of freedom"
            )
    if spec.model_id == "zero_inflated":
        if f.zero_excess is None or f.zero_excess < 2.0:
            warnings.append(
                "the zero share is"
                + (f" {_fmt(f.zero_excess)}× " if f.zero_excess is not None else " ")
                + "what a plain count model would produce, so a hurdle adds a parameter for nothing"
            )

    # --- conditional variance (§5.8 D, Part 11) ---------------------------- #
    # The spec's trigger is explicit: GARCH is offered when ARCH-LM rejects on the *residuals*. Getting
    # that object right is most of the work, and two simpler readings are both wrong — the level test
    # rejects on every random walk because the mean moves, and a blanket first-difference test rejects on
    # 96% of white-noise draws because differencing induces an MA(1) root. `arch_lm_resid_p` is the test
    # run after the differencing S5 chose, which is the series the mean model is actually fitted to.
    if spec.family is ModelFamily.VOLATILITY:
        if f.data_class is not DataClass.CONTINUOUS:
            reasons.append(
                f"it models the variance of a continuous series and this one is "
                f"{f.data_class.value.replace('_', ' ')}"
            )
        p_resid = f.arch_lm_resid_p
        p_level = f.arch_lm_level_p
        if p_resid is None:
            reasons.append(
                "the ARCH-LM test needs at least 24 observations after the differencing step and this "
                "series is too short to run it"
            )
        elif p_resid > 0.10:
            detail = (
                f" The test on the raw level does reject ({p_phrase(p_level, 3)}), but on a series with d = "
                f"{f.d_hint} that is the mean moving, not the variance clustering."
                if p_level is not None and p_level <= 0.05 and f.d_hint
                else ""
            )
            reasons.append(
                f"it needs volatility clustering and ARCH-LM on the mean-adjusted series returns "
                f"{p_phrase(p_resid, 3)}, which is no evidence of it.{detail}"
            )
        elif p_resid > 0.05:
            warnings.append(
                f"ARCH-LM on the mean-adjusted series gives {p_phrase(p_resid, 3)} — weak evidence of "
                "clustering, so the variance path may be close to flat"
            )

    # --- event timestamps -------------------------------------------------- #
    if f.data_class is DataClass.EVENT_TIMESTAMP:
        # The event branch is not "the continuous branch with a flag". A renewal model on a monthly
        # count series has no intervals to fit; a Poisson regression on event dates has no periods.
        # Each side names what it needs, and both sentences are quotable.
        if spec.family is not ModelFamily.POINT_PROCESS:
            reasons.append(
                "the dataset is a list of event dates, so there is no value series to model — only the "
                "arrival process is observable"
            )
    elif spec.family is ModelFamily.POINT_PROCESS:
        n_periods = f.n_non_missing
        reasons.append(
            f"it models the timing of individual events and this series has {n_periods} recorded "
            "periods rather than event timestamps"
        )

    # --- multivariate: it needs a *second series*, not a driver ------------- #
    # The distinction matters and is easy to get wrong: `n_exogenous` counts extra numeric columns, and
    # a multivariate model consumes those columns as co-equal series, while an exogenous model consumes
    # them as inputs to one series. Refusing a VAR because "there is no driver" would be a category
    # error, and offering it on a single-column dataset produces a row that can only fail.
    if spec.family is ModelFamily.MULTIVARIATE:
        if f.n_exogenous < 1:
            reasons.append(
                f"it models several series together and this dataset has 1 numeric column — "
                f"{f.n_exogenous} companion series to model with it. Map a second numeric column in "
                "Column mapping to give it something to move with."
            )
        elif f.n_non_missing < spec.required.min_n:
            pass  # already reported by the length check above
        else:
            warnings.append(
                f"the {f.n_exogenous} companion column(s) are used as co-equal series, which means this "
                "fit answers a joint question. Its information criteria cover the whole system and are "
                "not comparable with the single-series rows in the same table."
            )

    # --- cointegration: the VECM's whole premise --------------------------- #
    if spec.model_id == "vecm":
        if f.stationarity_verdict and str(f.stationarity_verdict).startswith("stationary"):
            reasons.append(
                f"it exists to model a long-run relationship between non-stationary series and Step 5 "
                f"found the series {str(f.stationarity_verdict).replace('_', ' ')} — there is no trend "
                "for a cointegrating relation to cancel"
            )

    # --- the ML family: no likelihood, so the IC ranking does not apply ----- #
    if spec.family is ModelFamily.ML:
        warnings.append(
            "this row has no likelihood, so it carries no AIC and cannot be ranked against the "
            "classical models on one. Compare it on holdout and fold error, and read the model card "
            "for its features, lags and validation scheme."
        )
        if f.n_non_missing < 80:
            warnings.append(
                f"there are {f.n_non_missing} observations; a flexible tabular model on this little "
                "history will mostly memorise it"
            )

    # --- exogenous -------------------------------------------------------- #
    if spec.required.requires_exogenous or spec.family is ModelFamily.EXOGENOUS:
        if f.n_exogenous <= 0:
            reasons.append(
                f"it needs at least 1 outside (exogenous) column and this dataset has {f.n_exogenous} "
                "— the target column cannot be its own driver"
            )
        elif not spec.capabilities.supports_exogenous:
            reasons.append(
                f"it does not accept exogenous inputs and {f.n_exogenous} are available"
            )

    # --- nonlinear -------------------------------------------------------- #
    if spec.family is ModelFamily.NONLINEAR:
        # A regime model needs variance contrast that a *trend* does not explain. Both numbers are
        # reported: on this build's promotion dataset the raw statistic is inflated by the trend (a
        # rising series has rising windows), so testing only the raw version would offer a regime
        # model for a series whose whole story is a straight line.
        raw = f.nonlinear_signal
        det = f.nonlinear_signal_detrended if f.nonlinear_signal_detrended is not None else raw
        if det is not None and det < 1.30:
            reasons.append(
                f"once its trend is removed, its most variable stretch is only {_fmt(det)}× as "
                "variable as noise of this length and persistence would be, which is too little "
                "contrast for a second regime to attach to (needs 1.30"
                + (f"; with the trend included the ratio is {_fmt(raw)}" if raw is not None else "")
                + ")"
            )
        if f.period is not None and (f.seasonal_strength or 0) >= 0.6:
            warnings.append(
                f"seasonal strength is {_fmt(f.seasonal_strength)}, and a two-regime model has no "
                "seasonal term — a regime will very likely absorb the season instead of a real shift"
            )

    # --- intermittency ---------------------------------------------------- #
    if spec.family is ModelFamily.INTERMITTENT:
        n_nonzero = f.n - f.n_zeros if f.n else 0
        if n_nonzero < 3:
            reasons.append(
                f"it needs at least 3 non-zero observations to estimate a demand size and there are {n_nonzero}"
            )
        # Sparse is the whole premise. Below the intermittent threshold there is no "when does it
        # happen" question to answer, and a Croston fit on 3%-zero data is just a bad mean model.
        if spec.model_id == "adida" and f.cv_of_sizes is not None and f.cv_of_sizes < 0.5:
            warnings.append(
                f"the non-zero demand sizes are fairly consistent (CV {_fmt(f.cv_of_sizes)}), so "
                "aggregating them into blocks buys little over Croston on the original period"
            )
        if f.pct_zeros < 0.20:
            # A zero share of exactly 0 must not print "only 0% of the values are zero" — that is
            # self-contradictory prose. The no-zeros case is its own sentence.
            if f.n_zeros == 0:
                reasons.append(
                    "no value in the series is zero, so there is no intermittency to model — Croston "
                    "and TSB would collapse to a slow mean"
                )
            else:
                reasons.append(
                    f"only {f.pct_zeros * 100:.0f}% of the values are zero ({f.n_zeros} of {f.n}); an "
                    "intermittent model needs a mostly-empty series to have anything to model"
                )

    # --- seasonality ------------------------------------------------------ #
    # Baselines are exempt from the "no seasonal term" warning on purpose. A benchmark's job is to
    # be beaten, not to explain the season; badging all five with the same warning buries the one
    # model where it matters (ARIMA) under advice the user cannot act on.
    if (
        not spec.is_baseline
        and not spec.capabilities.supports_seasonality
        and (f.seasonal_strength or 0) >= 0.30
    ):
        warnings.append(
            f"seasonal strength is {_fmt(f.seasonal_strength)}, which this model has no term for — "
            "its residuals will carry the season"
        )
    if spec.capabilities.supports_seasonality and f.period is None and spec.family is not ModelFamily.ARMA:
        warnings.append("no seasonal period was detected, so the seasonal terms will be dropped")
    if spec.model_id == "seasonal_naive" and f.period is None:
        reasons.append(
            "there is no seasonal period, so 'the same season last time' does not exist"
        )
    if spec.model_id == "seasonal_naive" and f.period and f.n_non_missing < f.period + 2:
        reasons.append(
            f"it needs more than one full cycle (period {f.period}) and there are {f.n_non_missing} "
            "observations"
        )
    if spec.family is ModelFamily.ARMA and spec.model_id == "sarima":
        if f.period is None:
            reasons.append("it needs a detected seasonal period and none was found")
        elif f.n_non_missing < 2 * f.period:
            reasons.append(
                f"it needs at least two full cycles ({2 * f.period} observations) and there are "
                f"{f.n_non_missing}"
            )

    # --- stationarity ----------------------------------------------------- #
    if spec.required.requires_stationarity and (f.stationarity_verdict or "").startswith("unit_root"):
        reasons.append(
            f"it requires a stationary series and Step 5 found a {f.stationarity_verdict.replace('_', ' ')}"
        )
    if spec.model_id == "arfima":
        if f.n_non_missing < 50:
            reasons.append(f"the GPH estimator needs at least 50 observations and there are {f.n_non_missing}")
        if f.autocorr_lag1 is not None and abs(f.autocorr_lag1) < 0.4:
            warnings.append(
                f"lag-1 autocorrelation is only {_fmt(f.autocorr_lag1)}, so there is little "
                "long-memory signature to estimate a fractional d from"
            )

    # --- breaks ----------------------------------------------------------- #
    if f.n_breaks >= 2 and spec.family in {ModelFamily.ARMA, ModelFamily.SMOOTHING}:
        warnings.append(
            f"{f.n_breaks} structural breaks were detected; a single fixed-parameter model spans "
            "all of them, so its early residuals and late ones are not comparable"
        )

    # --- implementation status -------------------------------------------- #
    if spec.feature_flag == "pending":
        # A *measured* reason beats a plan. Where the row is pending because a library is missing, the
        # probe result is quoted; "planned for part 11" on a part that has shipped, or on a part that
        # is currently being built, tells the user nothing true.
        measured = _MEASURED_REFUSAL.get(spec.model_id)
        if measured:
            reasons.append(measured)
        else:
            reasons.append(
                "this build has no fitting code for it yet — it is planned for part "
                f"{spec.planned_part if spec.planned_part is not None else 'a later build'}, so there "
                "is nothing to fit today"
            )

    compatible = not reasons
    # A model that cannot be fitted has nothing to warn about; mixing the two makes the excluded
    # list read like a list of concerns rather than a list of answers.
    return Verdict(compatible=compatible, reasons=reasons, warnings=warnings if compatible else [])


__all__ = [
    "BASELINE_IDS",
    "NOT_IMPLEMENTED",
    "REGISTRY",
    "Verdict",
    "all_specs",
    "evaluate_spec",
    "get_spec",
    "implemented_specs",
    "is_implemented",
    "register",
]
