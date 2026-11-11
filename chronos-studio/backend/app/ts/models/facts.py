"""
ts/models/facts.py — the numbers the suggestion engine queries.

The claim §5.8 makes is that the registry is *data, not code paths*: "which models do I show this
user?" must be a query over facts, and adding a model must mean adding a row, not editing an
`if`-ladder. That only holds if the facts are explicit and complete, so they live in one flat
dataclass whose every field is either measured here or carried over from an earlier stage.

Everything expensive is computed once, in `build_facts`, and nothing in it needs a fitted model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.schemas import DataClass


@dataclass
class DataFacts:
    """What the data is, measured once."""

    n: int = 0
    n_non_missing: int = 0
    n_missing: int = 0
    period: int | None = None
    data_class: DataClass = DataClass.CONTINUOUS

    # --- distribution ------------------------------------------------------ #
    mean: float | None = None
    sd: float | None = None
    cv: float | None = None                 # sd / |mean| — scale-free dispersion
    min_value: float | None = None
    max_value: float | None = None
    n_zeros: int = 0
    pct_zeros: float = 0.0
    n_negative: int = 0
    n_positive: int = 0
    is_integer_valued: bool = False
    n_unique: int = 0

    # --- count-data diagnostics -------------------------------------------- #
    #: var/mean. ~1 = Poisson, >1 = overdispersed (NB), <1 = underdispersed.
    dispersion_ratio: float | None = None
    dispersion_p_value: float | None = None
    #: Fraction of zeros relative to what the Poisson mean would predict. Structural zeros show up
    #: as a large excess; "this is just a low rate" does not.
    zero_excess: float | None = None

    # --- dependence -------------------------------------------------------- #
    autocorr_lag1: float | None = None
    seasonal_strength: float | None = None
    trend_r2: float | None = None
    stationarity_verdict: str | None = None
    d_hint: int = 0
    D_hint: int = 0
    n_breaks: int = 0
    arqu_lm_p: float | None = None          # ARCH-LM p-value: volatility clustering
    #: ARCH-LM on the series *after* the mean equation implied by `d_hint` — the level when the series
    #: is taken as stationary, the first (or d-th) difference when it is not. This is the "on the
    #: residuals" test the volatility trigger needs. See `_arch_lm_for_differencing`.
    arch_lm_resid_p: float | None = None
    #: ARCH-LM on the raw level, kept for reporting: on a trending series it rejects because the *mean*
    #: moves, and quoting it as evidence of clustering is exactly the false positive this field exists
    #: to expose.
    arch_lm_level_p: float | None = None
    #: Ratio of the SD in the most variable fifth of the series to the SD in the median fifth. ~1
    #: means the series is equally noisy throughout (one regime will do); 2+ means there is something
    #: for a second regime to explain. Deliberately not the ARCH-LM p-value: that test answers "is
    #: there *any* serial dependence in the variance", which is true of almost every real series.
    nonlinear_signal: float | None = None
    #: Coefficient of variation of the non-zero values — how erratic the demand *sizes* are. ADIDA's
    #: whole premise is that they are too variable for one smoothed average, so this is the number that
    #: makes its suggestion checkable. None for a series with fewer than two non-zero values.
    cv_of_sizes: float | None = None

    #: The same statistic after removing a linear trend, because a trending series has windows that
    #: naturally differ and would otherwise look like regimes. A genuine regime shift survives
    #: detrending with roughly the same score; a trend does not.
    nonlinear_signal_detrended: float | None = None
    skewness: float | None = None
    excess_kurtosis: float | None = None

    # --- what the user has already done ------------------------------------- #
    transform_stack: list[str] = field(default_factory=list)
    horizon: int | None = None
    n_exogenous: int = 0

    # --- environment ------------------------------------------------------- #
    available_features: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["data_class"] = self.data_class.value
        return out


def _safe(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def dispersion_test(y: np.ndarray) -> tuple[float | None, float | None]:
    """
    Cameron–Trivedi-style overdispersion check, reduced to what is defensible with n < 100.

    Regress `(y - ȳ)² - y` on ȳ and test the intercept: for a Poisson, E[(y-μ)²-μ] = 0. A positive
    intercept means variance grows faster than the mean, which is the condition that makes a
    negative binomial the honest model and a Poisson a lie about the intervals.

    Returns `(ratio, p_value)` where `ratio` is var/mean — the number a reader can sanity-check.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return None, None
    mu = float(y.mean())
    if mu <= 0:
        return None, None
    ratio = float(y.var(ddof=1) / mu)
    aux = (y - mu) ** 2 - y
    # One-sample t-test of the auxiliary residual against zero: the simple, assumption-light version
    # of Cameron–Trivedi, and the p-value it produces is honest about small n.
    if np.allclose(aux, aux[0]):
        return ratio, None
    try:
        p = float(stats.ttest_1samp(aux, 0.0, alternative="greater").pvalue)
    except Exception:  # pragma: no cover - defensive
        p = None
    return ratio, _safe(p)


def zero_excess(y: np.ndarray) -> float | None:
    """
    Observed zero share ÷ the zero share of the fitted conditional rate.

    The reference has to be the *conditional* mean, not the pooled mean. Zeros are ordinary whenever
    the rate is low, and a pooled mean hides that: a series where most periods have λ ≈ 0.5 legitimately
    produces ~60% zeros. Using the Poisson GLM's fitted rate as the reference is what makes the ratio
    a test for *excess* zeros rather than a test for a small number.

    ~1 means the zeros are what the rate predicts; 2+ means there is a structural-zero mechanism (a
    hurdle model is the right shape). Below ~0.6 says there are fewer zeros than the rate implies,
    which is usually a rounding artefact rather than a finding.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 6:
        return None
    # Reference = the zero share implied by the fitted rate of a simple Poisson GLM (intercept plus a
    # scaled trend), averaged over the sample.
    t = np.linspace(-1.0, 1.0, n)
    X = np.column_stack([np.ones(n), t])
    try:
        if np.all(y == 0):
            return None
        beta, *_ = np.linalg.lstsq(X, np.log(np.maximum(y, 0.5)), rcond=None)
        lam = np.exp(np.clip(X @ beta, -20, 20))
    except Exception:  # pragma: no cover - defensive
        lam = np.full(n, max(float(y.mean()), 1e-9))
    expected = float(np.mean(np.exp(-lam)))
    if expected <= 1e-12:
        return None
    return float(np.mean(y == 0) / expected)


def arch_lm(y: np.ndarray, lags: int = 5) -> float | None:
    """ARCH-LM p-value on the residual-like series. Low p = volatility clustering worth modelling."""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < lags + 10:
        return None
    x = y - y.mean()
    sq = x ** 2
    # Regress sq_t on its own lags and test the R² as n·R² ~ chi²(lags).
    Y = sq[lags:]
    X = np.column_stack([np.ones(Y.size)] + [sq[lags - k - 1: Y.size + lags - k - 1] for k in range(lags)])
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        ss_tot = float(((Y - Y.mean()) ** 2).sum())
        if ss_tot <= 0:
            return None
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
        stat = Y.size * r2
        return _safe(stats.chi2.sf(stat, lags))
    except Exception:  # pragma: no cover - defensive
        return None


def _arch_lm_after_differencing(y: np.ndarray, d_hint: int) -> float | None:
    """
    ARCH-LM on the series in the form the mean model will see it.

    The spec's volatility trigger is "ARCH-LM rejects on the residuals", and the point of the word
    *residuals* is that the test must be run on an object with the mean removed. Two simpler choices are
    both wrong, and this function exists because both were tried:

    * **On the level.** A trending or random-walk series rejects violently — not because the variance
      clusters but because the mean moves. Read literally, that would offer a GARCH on every random walk.
    * **On the first difference, always.** This was the second attempt and it fails in the opposite
      direction: differencing a *stationary* white-noise series produces an MA(1) with θ = −1, whose
      *squares* are strongly autocorrelated by construction. On 300 white-noise draws the test rejected
      **96%** of the time — a gate that passes almost everything is worse than no gate, because it looks
      like evidence.

    So the differencing order comes from `d_hint`, which S5 already decided: no difference for a
    stationary series, d differences for one that is not. The value is reported alongside the level
    p-value so a reader can see which object was tested.
    """
    x = np.asarray(y, dtype=float)
    for _ in range(max(int(d_hint), 0)):
        if x.size < 24:
            return None
        x = np.diff(x)
    if x.size < 24:
        return None
    return arch_lm(x)


def _nonlinear_signal(y: np.ndarray, window: int = 6, n_sim: int = 200) -> float | None:
    """
    How much more variable is the most variable stretch than a flat series would look?

    The naive statistic — max window SD ÷ median window SD — was the first thing tried, and it is
    wrong in a way worth recording: on *white noise* it sits around **1.6**, not 1.0, because taking a
    maximum over ~n overlapping windows of a noisy quantity has a large positive bias. A threshold of
    1.15 would therefore have passed every flat series it saw, which is exactly the vacuous-pass bug
    this project keeps hunting.

    So the statistic is reported relative to its own null: simulate Gaussian noise of the same length
    and memory (an AR(1) with the sample's lag-1 autocorrelation, because overlapping windows on an
    autocorrelated series are far more variable than on independent ones), compute the same ratio, and
    divide. **1.0 now means "exactly as lumpy as noise this shape"**, and a real regime structure
    scores well above it. The seeded RNG means the same series always gets the same number.

    The window is **6**, not 12, and that is a measured choice rather than a tidy one: on a generated
    two-regime series (220 points, σ 1 vs 5) this scores 1.77 against 1.12 for white noise at window 6,
    but only 1.25 against 1.10 at window 12 — a window as long as a regime fragment averages the
    contrast away. Short enough to see a regime, still long enough that a single spike cannot own a
    window.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < 3 * window:
        return None

    def ratio(v: np.ndarray) -> float | None:
        sds = np.array([np.std(v[i: i + window], ddof=1) for i in range(0, v.size - window + 1)])
        sds = sds[np.isfinite(sds)]
        if sds.size < 3:
            return None
        med = float(np.median(sds))
        if med <= 0:
            return None
        return float(np.max(sds) / med)

    observed = ratio(y)
    if observed is None:
        return None
    # Null: AR(1) with the sample's own first-order persistence, so the reference has the same
    # smoothness — otherwise a trending series would look "lumpy" only because its windows overlap.
    centred = y - y.mean()
    denom = float(np.sum(centred[:-1] ** 2))
    phi = float(np.sum(centred[:-1] * centred[1:]) / denom) if denom > 1e-12 else 0.0
    phi = float(np.clip(phi, -0.9, 0.95))
    rng = np.random.default_rng(20260101)
    eps = rng.normal(0.0, float(np.std(centred)) or 1.0, size=(n_sim, n))
    sims = np.empty_like(eps)
    sims[:, 0] = eps[:, 0]
    for t in range(1, n):
        sims[:, t] = phi * sims[:, t - 1] + eps[:, t]
    null = np.array([r for r in (ratio(row) for row in sims) if r is not None])
    if null.size < 20:
        return None
    ref = float(np.median(null))
    if ref <= 0:
        return None
    return float(observed / ref)


def build_facts(
    y: pd.Series,
    *,
    data_class: DataClass = DataClass.CONTINUOUS,
    period: int | None = None,
    seasonal_strength: float | None = None,
    trend_r2: float | None = None,
    stationarity_verdict: str | None = None,
    d_hint: int = 0,
    D_hint: int = 0,
    n_breaks: int = 0,
    transform_stack: list[str] | None = None,
    horizon: int | None = None,
    n_exogenous: int = 0,
    available_features: dict[str, bool] | None = None,
) -> DataFacts:
    """
    Measure everything the registry filter needs, from the series alone plus what earlier stages
    already reported.

    Deliberately does *not* re-run stationarity tests: those answers belong to S5, and a second copy
    computed here could disagree with the verdict the gate travelled with.
    """
    yv = pd.Series(y).astype(float)
    finite = yv.dropna().to_numpy()
    n_all = int(yv.notna().sum())
    vals = finite[np.isfinite(finite)]

    f = DataFacts(
        n=int(yv.size),
        n_non_missing=n_all,
        n_missing=int(yv.isna().sum()),
        period=period,
        data_class=data_class,
        seasonal_strength=seasonal_strength,
        trend_r2=trend_r2,
        stationarity_verdict=stationarity_verdict,
        d_hint=d_hint,
        D_hint=D_hint,
        n_breaks=n_breaks,
        transform_stack=list(transform_stack or []),
        horizon=horizon,
        n_exogenous=n_exogenous,
        available_features=dict(available_features or {}),
    )
    if vals.size == 0:
        return f

    f.mean = _safe(vals.mean())
    f.sd = _safe(vals.std(ddof=1)) if vals.size > 1 else None
    if f.mean not in (None, 0.0) and f.sd is not None:
        f.cv = _safe(f.sd / abs(f.mean))
    f.min_value = _safe(vals.min())
    f.max_value = _safe(vals.max())
    f.n_zeros = int((vals == 0).sum())
    f.pct_zeros = float(f.n_zeros / vals.size) if vals.size else 0.0
    f.n_negative = int((vals < 0).sum())
    f.n_positive = int((vals > 0).sum())
    f.n_unique = int(np.unique(vals).size)
    f.is_integer_valued = bool(np.allclose(vals, np.round(vals), atol=1e-9))

    if vals.size >= 3:
        f.skewness = _safe(stats.skew(vals))
        f.excess_kurtosis = _safe(stats.kurtosis(vals))
    if vals.size >= 4:
        sd = float(vals.std(ddof=0))
        if sd > 0:
            a = vals[:-1] - vals[:-1].mean()
            b = vals[1:] - vals[1:].mean()
            denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
            if denom > 0:
                f.autocorr_lag1 = _safe(float((a * b).sum() / denom))
    if vals.size >= 12:
        f.arqu_lm_p = arch_lm(vals)
        f.arch_lm_level_p = f.arqu_lm_p
        f.arch_lm_resid_p = _arch_lm_after_differencing(vals, d_hint)
    # Measured for every series rather than only when a nonlinear model is being considered: the
    # registry filter runs over *all* rows, so a fact that is only computed for some callers is a
    # silent "not applicable" the excluded list would then have to invent a reason for.
    nz = vals[vals != 0]
    if nz.size >= 2:
        m = float(nz.mean())
        f.cv_of_sizes = abs(float(nz.std(ddof=1)) / m) if m != 0 else None
    if vals.size >= 36:
        f.nonlinear_signal = _nonlinear_signal(vals)
        t = np.arange(vals.size, dtype=float)
        slope, intercept = np.polyfit(t, vals, 1)
        f.nonlinear_signal_detrended = _nonlinear_signal(vals - (intercept + slope * t))

    if data_class in {DataClass.COUNT, DataClass.INTERMITTENT} or (
        f.is_integer_valued and f.n_zeros > 0 and (f.min_value or 0) >= 0
    ):
        f.dispersion_ratio, f.dispersion_p_value = dispersion_test(vals)
        f.zero_excess = zero_excess(vals)

    return f


def classify_data_class(y: pd.Series, declared: DataClass | None = None) -> tuple[DataClass, str]:
    """
    Decide the branch, preferring the ingest-time verdict but correcting it when the numbers
    disagree in a way that would change which models are offered.

    Three corrections are worth making, all of them about *sparsity* or *range*, never about mere
    integrality:

    * a declared `count` series with 60 % zeros is intermittent by any useful definition — getting
      that wrong sends a spare-parts series to Poisson, whose intervals then assume a variance it
      does not have;
    * values inside [0, 1] that are not whole numbers are proportions;
    * a declared binary series must not fall through the intermittent rule just because it has many
      zeros.

    Integrality alone is **not** a correction. `detect_data_class` in Part 1 already applies the
    large-magnitude rule (whole-numbered passenger counts, monthly deaths and minute-by-minute usage
    are measured quantities, not event counts). Overruling that here used to reclassify eight bundled
    teaching datasets as COUNT and offer Poisson / negative-binomial / INGARCH for them. When an
    ingest verdict exists, it wins; the count branch is a default only when there is no verdict.
    """
    yv = pd.Series(y).astype(float).dropna().to_numpy()
    if yv.size == 0:
        return declared or DataClass.CONTINUOUS, "the series is empty"
    pct_zero = float(np.mean(yv == 0))
    non_neg_int = bool(np.allclose(yv, np.round(yv), atol=1e-9) and yv.min() >= 0)
    fraction = bool(np.all((yv >= 0) & (yv <= 1)) and not np.allclose(yv, np.round(yv)))

    if declared is DataClass.EVENT_TIMESTAMP:
        return declared, "the input is event timestamps"
    if declared is DataClass.BINARY_OR_PROPORTION:
        # A 0/1 series is whole-numbered and often has >30% zeros; letting the intermittent rule
        # below see it would turn a Bernoulli outcome into spare-parts demand.
        return declared, "the series was declared binary or proportional at load time"
    if fraction:
        return DataClass.BINARY_OR_PROPORTION, (
            f"every value lies in [0, 1] and the values are not whole numbers"
        )
    if non_neg_int and pct_zero > 0.30:
        return DataClass.INTERMITTENT, (
            f"{pct_zero * 100:.0f}% of values are zero, so this is intermittent demand "
            f"rather than a count series"
        )
    if declared is DataClass.CONTINUOUS:
        # Part 1's `detect_data_class` already made this call, with the large-magnitude rule:
        # passenger counts, monthly deaths and minute-by-minute usage are whole numbers but are
        # measured quantities, not event counts, and every forecasting package treats them as
        # continuous. Overruling that verdict here reclassified all of those bundled examples as
        # COUNT and offered Poisson / NB / INGARCH for them. Respect the ingest verdict; the
        # count branch stays available when the user declares it explicitly.
        return DataClass.CONTINUOUS, (
            "declared continuous at load time; whole-number values are rounded measurements "
            "in this magnitude range, not event counts"
        )
    if non_neg_int and declared is None:
        # No ingest verdict was supplied (a direct library call). Whole, non-negative values are
        # then the only evidence available, so the count branch is the safer default.
        ratio = float(yv.var(ddof=1) / yv.mean()) if yv.mean() > 0 else None
        return DataClass.COUNT, (
            f"all values are whole numbers with a minimum of {yv.min():.0f}"
            + (f" and a variance {ratio:.2f}× the mean" if ratio is not None else "")
        )
    if declared is DataClass.COUNT and non_neg_int:
        return declared, f"whole numbers with a minimum of {yv.min():.0f}"
    if declared is DataClass.INTERMITTENT and pct_zero <= 0.30:
        return (
            DataClass.INTERMITTENT,
            f"declared intermittent; {pct_zero * 100:.0f}% of values are zero",
        )
    return declared or DataClass.CONTINUOUS, "as declared at load time"


__all__ = [
    "DataFacts",
    "arch_lm",
    "build_facts",
    "classify_data_class",
    "dispersion_test",
    "zero_excess",
]
