"""
advisor/rules_prep.py — S2: the preprocessing advisor (§5.3).

Every "you should …" in this app is produced here, from a declarative rule whose
`when` is a pure function over *computed facts* and whose `because` interpolates
real numbers. No recommendation lives in a React component, and none is written
by a language model.

Design decisions worth stating:

* **Facts, not the series.** Rules receive `PrepContext`, which holds the numbers
  S1 computed (`eda_facts`) plus a handful of extra diagnostics that only make
  sense for a preprocessing decision (Box–Cox λ, gap runs, stationarity of the
  remainder). A rule may not touch the raw series: that keeps every rule testable
  with a dict, and it guarantees the advice is explicable from what the user can
  see on screen.

* **Blockers are separate from recommendations.** A blocker is something that must
  be decided before the pipeline can honestly continue (long gap, non-positive
  values for a log, mixed frequency). Recommendations are choices, and every one
  of them carries its alternative — because "don't" is usually a real option.

* **`prep.smooth.not_for_modelling` is unconditional.** Smoothing before modelling
  destroys the autocorrelation the model needs and makes MASE nonsense. The app
  will happily smooth for *pictures*, and says why that is different.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from app.core.schemas import Confidence

# The book's rule-of-thumb boundary for "strong" seasonality (Hyndman's F_S).
STRONG_SEASONALITY = 0.64

#: Below this many observations per gap run, interpolation is defensible.
SHORT_GAP = 2
#: Above this, a gap is an event, not a nuisance: block, do not interpolate.
LONG_GAP = 12
#: A run this long of identical values is nearly always a stall or a truncation.
STALL_RUN = 12


# --------------------------------------------------------------------------- #
# The context a rule sees
# --------------------------------------------------------------------------- #
@dataclass
class PrepContext:
    """Everything a preprocessing rule is allowed to know. All numbers, no series."""

    # identity
    n: int = 0
    period: int | None = None
    frequency_label: str = "observation"
    data_class: str = "continuous"
    series_name: str = "your series"

    # from S1 (`eda_facts`)
    n_missing: int = 0
    pct_zeros: float = 0.0
    n_zeros: int = 0
    n_negatives: int = 0
    n_gaps: int = 0
    longest_gap: int = 0
    n_outliers: int = 0
    n_level_shifts: int = 0
    seasonal_strength: float | None = None
    seasonal_amplitude: float | None = None
    seasonal_amplitude_growing: bool | None = None
    seasonal_p: float | None = None
    cv: float = 0.0
    skewness: float = 0.0
    mean: float | None = None
    min_value: float | None = None
    trend_r2: float = 0.0
    trend_pct_over_window: float = 0.0
    variance_ratio: float | None = None
    n_unique: int = 0
    stall_run: int = 1

    # extra diagnostics computed for S2 itself
    gap_runs: list[int] = field(default_factory=list)      # lengths of consecutive gaps
    boxcox_lambda: float | None = None
    boxcox_ci: tuple[float, float] | None = None
    variance_level_corr: float | None = None
    remainder_acf1: float | None = None                   # |ρ1| of the STL remainder
    remainder_sd: float | None = None
    signal_to_noise: float | None = None                   # var(seasonal+trend)/var(remainder)
    seasonal_diff_needed: bool | None = None
    n_per_year_at_finer: int | None = None                 # if aggregation is possible
    is_positive: bool = True

    # bookkeeping
    already_applied: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    def f(self, name: str, default: float | None = None) -> float | None:
        v = getattr(self, name, None)
        if v is None:
            return default
        try:
            fv = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return fv if math.isfinite(fv) else default

    def has(self, op: str) -> bool:
        return op in self.already_applied


# --------------------------------------------------------------------------- #
# A rule
# --------------------------------------------------------------------------- #
@dataclass
class Rule:
    id: str
    stage: str
    recommend: str                                    # the action, in imperative form
    when: Callable[[PrepContext], bool]
    because: Callable[[PrepContext], str]             # MUST interpolate real values
    confidence: Confidence = Confidence.MEDIUM
    provenance: list[str] = field(default_factory=list)
    alternative: Callable[[PrepContext], str | None] | None = None
    risk_if_ignored: str = ""
    how_to: str = ""
    kind: Literal["recommend", "blocker", "note"] = "recommend"
    op: str = ""                                      # preprocessing step it implies

    def to_out(self, ctx: PrepContext) -> dict[str, Any]:
        """Serialise against the live context — `because` is evaluated here, once."""
        alt = None
        if self.alternative is not None:
            try:
                alt = self.alternative(ctx)
            except Exception:                     # pragma: no cover - defensive
                alt = None
        return {
            "id": self.id,
            "stage": self.stage,
            "recommend": self.recommend,
            "confidence": self.confidence.value if hasattr(self.confidence, "value")
            else str(self.confidence),
            "because": self.because(ctx),
            "provenance": list(self.provenance),
            "alternative": alt,
            "risk_if_ignored": self.risk_if_ignored,
            "how_to": self.how_to,
        }


def _num(x: float | None, digits: int = 2) -> str:
    """
    A number as it should appear inside a sentence.

    No thousands separators: "1,450" is read as two numbers by a person skimming and by
    the grounding check alike. Large values switch to exponent notation, which is also
    the honest way to quote a signal-to-noise ratio of 1450.
    """
    if x is None or not np.isfinite(x):
        return "—"
    ax = abs(x)
    if ax >= 10000:
        return f"{x:.2e}"
    if ax >= 1000:
        return f"{round(x):d}"
    if ax >= 100:
        return f"{round(x, 1):g}"
    return f"{round(x, digits):g}"


def _p(x: float | None) -> str:
    """p-values are reported the way readers expect, never as a confident zero."""
    if x is None or not np.isfinite(x):
        return "—"
    if x < 0.001:
        return "< 0.001"
    return f"{x:.3f}"


def _pct(x: float | None) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{_num(x * 100 if abs(x) <= 1.5 else x, 1)}%"


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def _impute_short_gap_applies(c: PrepContext) -> bool:
    return (
        0 < c.longest_gap <= SHORT_GAP
        and bool(c.seasonal_strength and c.seasonal_strength >= STRONG_SEASONALITY)
    )


def _impute_short_gap_because(c: PrepContext) -> str:
    return (
        f"{c.n_missing} value{'s are' if c.n_missing != 1 else ' is'} missing, and the "
        f"longest run is {c.longest_gap} observation{'s' if c.longest_gap != 1 else ''}. "
        f"The series is strongly seasonal (F_S = {_num(c.seasonal_strength)}), so the same "
        f"{c.frequency_label.lower().replace(' (start of month)', '')} position last year is a "
        "better guide than the neighbouring values are."
    )


def _impute_medium_gap_applies(c: PrepContext) -> bool:
    return SHORT_GAP < c.longest_gap <= LONG_GAP


def _impute_medium_gap_because(c: PrepContext) -> str:
    return (
        f"The longest gap is {c.longest_gap} observations (out of {c.n_zeros + c.n + 0} positions). "
        "Linear interpolation across that many points invents a straight line that no "
        "model will be able to distinguish from data."
    )


def _impute_blank_applies(c: PrepContext) -> bool:
    return c.longest_gap > LONG_GAP


def _impute_blank_because(c: PrepContext) -> str:
    return (
        f"The longest gap is {c.longest_gap} observations — that is "
        f"{_num(100 * c.longest_gap / max(1, c.n), 0)}% of the series in one stretch. "
        "Filling it would create data that never existed exactly where the model is "
        "most likely to be judged."
    )


def _zero_run_applies(c: PrepContext) -> bool:
    # In intermittent demand, runs of zeros are the data, not a defect.
    if c.data_class == "intermittent" or c.pct_zeros >= 0.30:
        return False
    return c.stall_run >= STALL_RUN and c.n_unique > 1


def _zero_run_because(c: PrepContext) -> str:
    return (
        f"There is a run of {c.stall_run} identical consecutive values. That is usually a "
        "stalled sensor, a truncated export, or a placeholder — not {0} real periods of no activity."
    )


def _needs_variance_fix(c: PrepContext) -> bool:
    """Is there *evidence* that the variance depends on the level?

    A λ estimate on its own is not evidence: on pure noise it is a random number.
    We require the spread to actually move with the level (rolling SD ratio or a
    blocked level–spread correlation), or a seasonal swing that visibly widens.
    """
    unstable = bool(c.variance_ratio and c.variance_ratio > 1.25)
    corr = c.f("variance_level_corr")
    # the sign matters: variance that *shrinks* as the level rises is not fixed by a log
    corr_ok = bool(corr is not None and corr >= 0.5)
    # A widening seasonal swing is only evidence of a multiplicative structure *at a
    # level*, not after the level has been logged: on a logged series a still-widening
    # swing means the transform did not finish the job, which ordinary variance
    # diagnostics should then decide — not this flag on its own.
    growing = c.seasonal_amplitude_growing is True and not _already_logged(c)
    return unstable or corr_ok or growing


def _already_logged(c: PrepContext) -> bool:
    """
    Has this series already been log-transformed in this session?

    Detected from the recorded steps rather than from `already_applied` names, because a
    Box–Cox with λ ≈ 0 is a log in everything but name.
    """
    return "transform" in c.already_applied


def preferred_transform(c: PrepContext) -> str | None:
    """
    The single transform to recommend, chosen by λ when λ is informative.

    Exactly one of log / sqrt / Box–Cox is ever offered, because "take logs *and* a
    square root" is not advice. When the λ interval spans both 0 and 0.5 the
    simplest defensible choice wins (log), and the interval is printed with it.
    """
    if not _needs_variance_fix(c):
        return None
    if _already_logged(c):
        return None
    if c.n_zeros or c.n_negatives or not c.is_positive:
        return "yeojohnson"

    lam = c.f("boxcox_lambda")
    if lam is None:
        return "log"
    lo, hi = c.boxcox_ci if c.boxcox_ci else (lam, lam)
    # λ = 1 means "no transform". An interval that contains 1 means the data cannot tell
    # the difference from the original scale, and a transform that *amplifies* skew is
    # not variance stabilisation — so the honest answer is to leave the scale alone.
    if lo <= 1.0 <= hi or lam >= 0.85:
        return None
    # an interval containing 0 makes the log the simplest defensible choice
    if lo <= 0.0 <= hi or abs(lam) <= 0.25:
        return "log"
    if 0.25 < lam <= 0.85 or lo <= 0.5 <= hi:
        return "sqrt"
    return "boxcox"


def _log_applies(c: PrepContext) -> bool:
    return preferred_transform(c) == "log"


def _log_because(c: PrepContext) -> str:
    bits = [
        f"Box–Cox puts λ at {_num(c.boxcox_lambda)}"
        + (f" (95% interval {_num(c.boxcox_ci[0])} to {_num(c.boxcox_ci[1])})" if c.boxcox_ci else "")
    ]
    if c.variance_ratio and c.variance_ratio > 1.25:
        bits.append(f"the rolling spread grows by {_num((c.variance_ratio - 1) * 100)}% across the series")
    if c.seasonal_amplitude_growing:
        bits.append("the seasonal swing widens as the level rises (multiplicative, not additive)")
    if c.variance_level_corr:
        bits.append(f"spread and level move together (correlation {_num(c.variance_level_corr)})")
    return "Taking logs is worth it because " + "; ".join(bits) + "."


def _sqrt_applies(c: PrepContext) -> bool:
    return preferred_transform(c) == "sqrt"


def _sqrt_because(c: PrepContext) -> str:
    ci = f" (95% interval {_num(c.boxcox_ci[0])} to {_num(c.boxcox_ci[1])})" if c.boxcox_ci else ""
    return (
        f"λ = {_num(c.boxcox_lambda)}{ci} sits in the square-root region: the variance grows "
        f"with the level (spread ratio {_num(c.variance_ratio)}), but less sharply than a log assumes."
    )


def _boxcox_applies(c: PrepContext) -> bool:
    return preferred_transform(c) == "boxcox"


def _boxcox_because(c: PrepContext) -> str:
    ci = f" (95% interval {_num(c.boxcox_ci[0])} to {_num(c.boxcox_ci[1])})" if c.boxcox_ci else ""
    return (
        f"λ = {_num(c.boxcox_lambda)}{ci} is neither 0 (log) nor 0.5 (square root), and the "
        f"rolling spread moves by {_num((c.variance_ratio - 1) * 100 if c.variance_ratio else None)}% "
        "across the series; Box–Cox at that λ matches the variance behaviour most closely."
    )


def _no_log_applies(c: PrepContext) -> bool:
    return preferred_transform(c) == "yeojohnson"


def _no_log_because(c: PrepContext) -> str:
    bad = c.n_zeros + c.n_negatives
    spread = (
        f"{_num((c.variance_ratio - 1) * 100)}% growth in the rolling spread"
        if c.variance_ratio else "a spread that still tracks the level"
    )
    return (
        f"{bad} observation{'s are' if bad != 1 else ' is'} zero or negative, so a plain log is "
        f"undefined. The variance still needs stabilising ({spread}), so Yeo–Johnson — which "
        "behaves like a log for positive values and stays defined at zero — is the equivalent move."
    )


def _flag_not_delete_applies(c: PrepContext) -> bool:
    # One flagged point in an otherwise clean series is a note, not a modelling decision.
    return c.n_outliers >= 2 or c.n_level_shifts >= 1


def _single_outlier_applies(c: PrepContext) -> bool:
    return c.n_outliers == 1 and c.n_level_shifts == 0


def _single_outlier_because(c: PrepContext) -> str:
    return (
        "One observation is unusual once trend and season are removed — about what chance "
        "produces in a series this length, so it is not worth special treatment yet."
    )


def _flag_not_delete_because(c: PrepContext) -> str:
    shifts = (
        f", {c.n_level_shifts} of them level shift{'s' if c.n_level_shifts != 1 else ''}"
        if 0 < c.n_level_shifts < c.n_outliers else
        ", all of them level shifts" if c.n_level_shifts else ""
    )
    return (
        f"{c.n_outliers} observation{'s look' if c.n_outliers != 1 else ' looks'} unusual once "
        f"trend and season are removed{shifts}. Deleting them quietly removes information the "
        "model should be told about, and biases the fit towards the pattern you already believe."
    )


def _seasonal_term_applies(c: PrepContext) -> bool:
    s = c.f("seasonal_strength")
    return c.period is not None and c.period > 1 and s is not None and s >= STRONG_SEASONALITY


def _seasonal_term_because(c: PrepContext) -> str:
    return (
        f"Seasonal strength F_S = {_num(c.seasonal_strength)} at period {c.period} with a "
        f"Kruskal–Wallis p of {_p(c.seasonal_p)} (the conventional boundary for 'strong' "
        f"seasonality is F_S = {STRONG_SEASONALITY}). The pattern is real, so a seasonal term "
        "will pay for itself."
    )


def _no_seasonal_term_applies(c: PrepContext) -> bool:
    s = c.f("seasonal_strength")
    p = c.f("seasonal_p")
    return c.period is not None and c.period > 1 and (
        (s is not None and s < 0.3) or (p is not None and p >= 0.05)
    )


def _no_seasonal_term_because(c: PrepContext) -> str:
    return (
        f"Seasonal strength is F_S = {_num(c.seasonal_strength)} with a Kruskal–Wallis p of "
        f"{_p(c.seasonal_p)}: the calendar positions do not differ more than chance. "
        "Adding seasonal terms here would be fitting noise, and would make the forecast "
        "interval falsely narrow."
    )


def _smooth_not_for_modelling_because(c: PrepContext) -> str:
    snr = c.signal_to_noise
    tag = f" The signal-to-noise ratio here is {_num(snr)}, which is why smoothing is tempting." if snr else ""
    return (
        "Smoothing before modelling removes exactly the short-run autocorrelation that ARIMA "
        "and ETS are built to learn, and it makes MASE against a naive benchmark meaningless." + tag
    )


def _smooth_for_presentation_applies(c: PrepContext) -> bool:
    snr = c.f("signal_to_noise")
    if snr is not None:
        # a smooth-to-noise ratio above 10 means the shape is already plain to see
        return snr < 10.0
    return bool(c.cv and c.cv > 0.5)


def _smooth_for_presentation_because(c: PrepContext) -> str:
    if c.signal_to_noise is not None:
        evidence = f"signal-to-noise {_num(c.signal_to_noise)}"
    else:
        evidence = f"coefficient of variation {_num(c.cv)}"
    return (
        f"The series is noisy relative to its pattern ({evidence}). A moving average or "
        "LOESS line on top of the raw series will make the shape easier to see — for "
        "reading, not for fitting."
    )


def _trim_applies(c: PrepContext) -> bool:
    return c.n > 0 and bool(c.trend_r2 and c.trend_r2 > 0.9) and abs(c.trend_pct_over_window) > 400


def _trim_because(c: PrepContext) -> str:
    return (
        f"The window covers a {_num(abs(c.trend_pct_over_window))}% move and a straight line "
        f"already explains R² = {_num(c.trend_r2)} of it. The earliest observations describe a "
        "different regime from the one you are forecasting, and they dominate the fit."
    )


def _diff_already_applied_because(c: PrepContext) -> str:
    return (
        "This series has already been differenced once in this session. Differencing twice is "
        "one of the most common ways to turn a workable series into an unusable one."
    )


def _aggregate_applies(c: PrepContext) -> bool:
    return bool(c.n and c.n > 30 and c.period == 1 and c.n_per_year_at_finer)


def _aggregate_because(c: PrepContext) -> str:
    return (
        f"{c.n} observations with no seasonal period detected. If the raw data is finer than "
        f"{c.frequency_label.lower()}, aggregating to {c.n_per_year_at_finer} periods per year would "
        "expose a seasonal shape — at the cost of introducing moving-average autocorrelation."
    )


def _short_series_applies(c: PrepContext) -> bool:
    return 0 < c.n < 30


def _short_series_because(c: PrepContext) -> str:
    return (
        f"There are only {c.n} observations. Any preprocessing choice (impute, transform, "
        "difference) costs degrees of freedom, and with this few points a wrong choice is "
        "unrecoverable — prefer the simplest option available."
    )


RULES: list[Rule] = [
    # ---- blockers ------------------------------------------------------- #
    Rule(
        id="prep.gap.long_blocker",
        stage="S2",
        kind="blocker",
        op="impute",
        recommend="Decide what to do about the long gap before continuing",
        when=_impute_blank_applies,
        because=_impute_blank_because,
        confidence=Confidence.HIGH,
        provenance=["table.descriptives", "fig.series"],
        alternative=lambda c: (
            "Shorten the analysis window so the gap falls outside it, or keep it as NA "
            "and accept a shorter series."
        ),
        risk_if_ignored=(
            "Interpolating a gap of this length invents a trend that did not happen, and every "
            "downstream test will treat it as evidence."
        ),
        how_to="Step 2 → Imputation: choose 'Leave as missing' or set an analysis window.",
    ),
    Rule(
        id="prep.log.undefined_blocker",
        stage="S2",
        kind="blocker",
        op="transform",
        recommend="Choose a transform that is defined for zero or negative values",
        when=_no_log_applies,
        because=_no_log_because,
        confidence=Confidence.HIGH,
        provenance=["table.descriptives", "fig.rolling_stats"],
        alternative=lambda c: "Leave the series on its original scale and let the model handle the level.",
        risk_if_ignored="A log of a non-positive value is undefined; packages will silently drop or clip rows.",
        how_to="Step 2 → Transform: pick 'Yeo–Johnson' or 'log(y + c)' and state c.",
    ),
    Rule(
        id="prep.data.stalled",
        stage="S2",
        kind="blocker",
        recommend="Check the run of identical values with whoever produced the data",
        when=_zero_run_applies,
        because=_zero_run_because,
        confidence=Confidence.MEDIUM,
        provenance=["table.descriptives"],
        alternative=lambda c: "Mark those periods as missing so the model does not treat the stall as signal.",
        risk_if_ignored="A stalled stretch flattens the seasonal estimate and understates forecast uncertainty.",
        how_to="Step 2 → Data treatment: mark the run as missing.",
    ),
    # ---- imputation ----------------------------------------------------- #
    Rule(
        id="prep.impute.seasonal_interpolation",
        stage="S2",
        op="impute",
        recommend="Interpolate the short gaps using the same season from other years",
        when=_impute_short_gap_applies,
        because=_impute_short_gap_because,
        confidence=Confidence.HIGH,
        provenance=["table.descriptives", "stat.kruskalwal1is_seasonal"],
        alternative=lambda c: "Leave them missing: most models handle NA, and no value is invented.",
        risk_if_ignored="A one- or two-point hole will make most models drop those rows or refuse to fit.",
        how_to="Step 2 → Imputation → 'Seasonal interpolation' → Apply.",
    ),
    Rule(
        id="prep.impute.state_space",
        stage="S2",
        op="impute",
        recommend="Use a state-space (Kalman) smoother for the mid-length gaps",
        when=_impute_medium_gap_applies,
        because=_impute_medium_gap_because,
        confidence=Confidence.MEDIUM,
        provenance=["table.descriptives"],
        alternative=lambda c: "Seasonal means from other years, which is simpler and nearly as good.",
        risk_if_ignored="Linear interpolation across several points makes the remainder look far tidier than it is.",
        how_to="Step 2 → Imputation → 'Kalman smoother' → Apply.",
    ),
    # ---- transform ------------------------------------------------------ #
    Rule(
        id="prep.transform.log",
        stage="S2",
        op="transform",
        recommend="Take logs before modelling",
        when=_log_applies,
        because=_log_because,
        confidence=Confidence.HIGH,
        provenance=["fig.rolling_stats", "table.seasonal_profile", "fig.series"],
        alternative=lambda c: "Model on the original scale and accept that the spread grows with the level.",
        risk_if_ignored=(
            "A model fitted to a series whose variance grows will over-weight the recent, large "
            "values and produce intervals that are too narrow early and too wide late."
        ),
        how_to="Step 2 → Transform → 'Natural log' → Apply (a preview is shown first).",
    ),
    Rule(
        id="prep.transform.sqrt",
        stage="S2",
        op="transform",
        recommend="Take a square root rather than a log",
        when=_sqrt_applies,
        because=_sqrt_because,
        confidence=Confidence.MEDIUM,
        provenance=["fig.rolling_stats"],
        alternative=lambda c: "Log, which stabilises variance more aggressively.",
        risk_if_ignored="The variance still grows with the level; intervals will be unevenly calibrated.",
        how_to="Step 2 → Transform → 'Square root' → Apply.",
    ),
    Rule(
        id="prep.transform.boxcox",
        stage="S2",
        op="transform",
        recommend="Apply a Box–Cox transform at the estimated λ",
        when=_boxcox_applies,
        because=_boxcox_because,
        confidence=Confidence.MEDIUM,
        provenance=["fig.rolling_stats"],
        alternative=lambda c: "Log — simpler to explain, and nearly as good when λ is between 0 and 0.5.",
        risk_if_ignored="Using the wrong transform leaves residual heteroskedasticity in the model.",
        how_to="Step 2 → Transform → 'Box–Cox' (λ pre-filled).",
    ),
    # ---- outliers ------------------------------------------------------- #
    Rule(
        id="prep.outlier.flag_not_delete",
        stage="S2",
        op="outlier",
        recommend="Keep the unusual points and add them to the model as intervention dummies",
        when=_flag_not_delete_applies,
        because=_flag_not_delete_because,
        confidence=Confidence.HIGH,
        provenance=["table.outliers", "fig.outliers"],
        alternative=lambda c: "Leave them untouched: a robust model or automatic outlier handling may absorb them.",
        risk_if_ignored=(
            "An unmodelled spike inflates the residual variance, which widens every forecast "
            "interval; an unmodelled level shift is worse — it biases the trend."
        ),
        how_to="Step 2 → Outliers → 'Add as intervention' (pulse for spikes, step for level shifts).",
    ),
    Rule(
        id="prep.outlier.single_note",
        stage="S2",
        kind="note",
        op="",
        recommend="Nothing to do about the one unusual observation",
        when=_single_outlier_applies,
        because=_single_outlier_because,
        confidence=Confidence.MEDIUM,
        provenance=["table.outliers"],
        alternative=lambda c: "Flag it as an intervention dummy anyway if you know the event was real.",
        risk_if_ignored="None — modelling it would cost a degree of freedom for nothing.",
        how_to="Nothing to click.",
    ),
    # ---- seasonality ---------------------------------------------------- #
    Rule(
        id="prep.seasonal.term",
        stage="S2",
        op="",
        recommend="Keep a seasonal term in whatever model you choose",
        when=_seasonal_term_applies,
        because=_seasonal_term_because,
        confidence=Confidence.HIGH,
        provenance=["stat.kruskalwal1is_seasonal", "table.seasonal_profile"],
        alternative=lambda c: "Seasonally adjust first (STL), then model the adjusted series.",
        risk_if_ignored="Without a seasonal term the forecast will be flat at the annual average, which is wrong every month.",
        how_to="Step 8 → Suggest a model: seasonal candidates will be ranked first.",
    ),
    Rule(
        id="prep.seasonal.absent",
        stage="S2",
        op="",
        recommend="Do not add seasonal terms or a seasonal difference",
        when=_no_seasonal_term_applies,
        because=_no_seasonal_term_because,
        confidence=Confidence.HIGH,
        provenance=["stat.kruskalwal1is_seasonal"],
        alternative=lambda c: "Still allow a seasonal candidate if domain knowledge says the pattern exists but is masked.",
        risk_if_ignored="Seasonal difference on a non-seasonal series creates a unit root at the seasonal lag and destroys real signal.",
        how_to="Nothing to do — skip the seasonal options in Step 2.",
    ),
    # ---- smoothing ------------------------------------------------------ #
    Rule(
        id="prep.smooth.for_presentation",
        stage="S2",
        op="smooth",
        recommend="Smooth for the chart if the noise hides the shape — never for the model",
        when=_smooth_for_presentation_applies,
        because=_smooth_for_presentation_because,
        confidence=Confidence.LOW,
        provenance=["fig.rolling_stats", "fig.series"],
        alternative=lambda c: "Plot the raw series with a rolling mean on top (already done in Step 1).",
        risk_if_ignored="None for the analysis; the risk is misunderstanding a jagged chart as a broken one.",
        how_to="Step 2 → Smoothing → 'Presentation only'.",
    ),
    # ---- windowing ------------------------------------------------------ #
    Rule(
        id="prep.window.trim",
        stage="S2",
        op="trim",
        recommend="Trim the analysis window to the current regime",
        when=_trim_applies,
        because=_trim_because,
        confidence=Confidence.MEDIUM,
        provenance=["fig.series", "table.descriptives"],
        alternative=lambda c: "Keep everything and add a piecewise trend (Step 4).",
        risk_if_ignored="An upward-biased slope, and a forecast that continues a trend the series has left.",
        how_to="Step 2 → Window → set a start date.",
    ),
    Rule(
        id="prep.freq.aggregate",
        stage="S2",
        op="aggregate",
        recommend="Consider aggregating to a coarser frequency",
        when=_aggregate_applies,
        because=_aggregate_because,
        confidence=Confidence.LOW,
        provenance=["table.descriptives"],
        alternative=lambda c: "Keep the frequency and model the cycle with AR terms instead.",
        risk_if_ignored="A real seasonal pattern stays hidden, and every model looks equally bad.",
        how_to="Step 2 → Frequency → 'Aggregate to …'.",
    ),
    # ---- caution -------------------------------------------------------- #
    Rule(
        id="prep.transform.already_applied",
        stage="S2",
        kind="note",
        op="",
        recommend="A transform has already been applied — do not stack another one",
        when=lambda c: _already_logged(c),
        because=lambda c: (
            "The series has already been transformed in this session, so the variance "
            "diagnostics above describe the transformed data. If the spread still grows, "
            "the better move is a different model (one with a variance structure), not a "
            "second transform."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["fig.prep.variance_level"],
        alternative=lambda c: "Undo the transform in Step 3 and try a different one.",
        risk_if_ignored="Stacking logarithms makes coefficients uninterpretable and distorts the back-transform.",
        how_to="Step 3 → Applied → Undo last, then choose a different transform in Step 2.",
    ),
    Rule(
        id="prep.series.short",
        stage="S2",
        op="",
        recommend="Prefer the simplest preprocessing available",
        when=_short_series_applies,
        because=_short_series_because,
        confidence=Confidence.MEDIUM,
        provenance=["table.descriptives"],
        alternative=None,
        risk_if_ignored="Each extra transformation costs degrees of freedom you do not have.",
        how_to="Nothing to click — this is guidance for the choices below.",
    ),
    Rule(
        id="prep.diff.already_applied",
        stage="S2",
        op="",
        recommend="Do not difference again",
        when=lambda c: c.has("difference"),
        because=_diff_already_applied_because,
        confidence=Confidence.HIGH,
        provenance=[],
        alternative=lambda c: "If a unit root is still suspected in Step 5, test it there with the full battery.",
        risk_if_ignored="Over-differencing produces a series whose ACF at lag 1 is ≈ −0.5 and whose variance rises.",
        how_to="Nothing to click.",
    ),
]

#: Unconditional rules: they hold for every series, and the app says so out loud.
UNCONDITIONAL: list[Rule] = [
    Rule(
        id="prep.smooth.not_for_modelling",
        stage="S2",
        op="",
        recommend="Never smooth the series you will model",
        when=lambda c: True,
        because=_smooth_not_for_modelling_because,
        confidence=Confidence.HIGH,
        provenance=["fig.rolling_stats"],
        alternative=lambda c: "Smooth a *copy* for presentation, and say on the chart that it is smoothed.",
        risk_if_ignored=(
            "Smoothing destroys autocorrelation structure and makes every accuracy metric "
            "(MASE, RMSE against a naive benchmark) compare against an artificially easy target."
        ),
        how_to="Nothing to click — the app will not offer to smooth the modelling series.",
    ),
]

_FILE_ORDER = {r.id: i for i, r in enumerate(RULES)}


def rank(c: PrepContext) -> list[Rule]:
    """
    Rules whose condition holds, ordered so the most consequential comes first:
    blockers, then high confidence, then the order they are declared in.
    """
    hits = [r for r in RULES if _safe(r.when, c)]
    conf_rank = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
    hits.sort(
        key=lambda r: (
            0 if r.kind == "blocker" else 1,
            conf_rank.get(r.confidence, 3),
            _FILE_ORDER.get(r.id, 99),
        )
    )
    return hits


def _safe(fn: Callable[[PrepContext], bool], c: PrepContext) -> bool:
    try:
        return bool(fn(c))
    except Exception:                             # pragma: no cover - a rule must never crash a stage
        return False


__all__ = ["PrepContext", "Rule", "RULES", "UNCONDITIONAL", "rank", "STRONG_SEASONALITY"]
