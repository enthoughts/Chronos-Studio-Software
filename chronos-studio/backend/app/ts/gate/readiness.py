"""
ts/gate/readiness.py — S8: the model-readiness gate (§5.7).

The gate answers one question: *can a model be fitted to this series, and if so what must the user
be told when they read the output?* Everything downstream inherits its answer, so the failure modes
here are the expensive ones.

Three rules shape the whole module.

1. **Every criterion is a number, not a tick.** "Stationary: yes" is the thing the spec forbids, and
   for good reason: it is unfalsifiable. Each check returns the statistic it used, the threshold it
   was compared against, and which test said what.

2. **`passed = None` is a third state and it is not a failure.** A criterion that could not be
   evaluated — too short, no seasonal period, no Step 5 yet — is "not assessable". Collapsing that
   into `False` would fail every short series for lacking a seasonal test; collapsing it into `True`
   would pass them for the same reason. The verdict distinguishes the two: a series with unknowns
   is `INCONCLUSIVE`, not `NOT_READY`.

3. **Severity is separate from pass/fail.** A failed criterion that merely narrows what the output
   can claim is a *caveat*; one that makes the output wrong is a *blocker*. The verdict is then a
   pure function of the two lists, so it can never disagree with the checklist above it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from app.core.formatting import p_phrase

CriterionState = Literal["pass", "fail", "unknown", "not_applicable"]
Severity = Literal["blocker", "caveat"]


@dataclass
class Criterion:
    """One line of the checklist: what was checked, with what number, against what line."""

    key: str
    label: str
    state: CriterionState
    value: str
    detail: str
    #: What this criterion would need in order to matter at all. Shown when state == "unknown".
    why_unknown: str = ""
    #: Where the user goes to fix it.
    jump_to_stage: str | None = None
    #: Only consulted when state == "fail".
    severity: Severity = "caveat"
    #: Applies to the whole series' fitness for modelling, regardless of pass/fail.
    weight: float = 1.0

    @property
    def passed(self) -> bool | None:
        """`None` covers both "could not assess" and "not applicable" — neither is a failure."""
        return {"pass": True, "fail": False, "unknown": None, "not_applicable": None}[self.state]

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "passed": self.passed,
            "state": self.state,
            "value": self.value,
            "detail": self.detail,
            "why_unknown": self.why_unknown,
            "jump_to_stage": self.jump_to_stage,
            "severity": self.severity,
        }


@dataclass
class GateFacts:
    """
    Everything the gate is allowed to know, all of it computed upstream.

    Deliberately a flat dataclass of numerics rather than a reference to the session: the gate must
    be testable without building a whole session, and a reviewer must be able to see every input.
    """

    # identity / size
    n: int = 0
    n_original: int = 0
    n_missing: int = 0
    longest_gap: int = 0
    period: int | None = None
    data_class: str = "continuous"
    horizon: int | None = None
    horizon_requested: bool = False

    # S5
    stationarity_verdict: str | None = None
    stationarity_confidence: str | None = None
    n_unit_root_tests: int = 0
    n_agreeing: int = 0
    #: When the verdict is a *seasonal* unit root, the regular battery answers a different question.
    #: Kept separately so the checklist can say which tests agreed with what.
    n_regular_supporting: int = 0
    n_regular_tests: int = 0
    n_contradictions: int = 0
    differencing_needed: bool = False
    seasonal_differencing_needed: bool = False
    break_dates: list[str] = field(default_factory=list)
    break_count: int = 0
    #: Did *any* break test actually run? Without this the check passed on "we found no breaks",
    #: which is also what a check that never ran reports — the exact vacuous pass this gate exists
    #: to prevent.
    breaks_tested: bool = False

    #: Did Step 1 (profile) actually run? "No gaps, no outliers" is also what an unrun check
    #: reports, so the checks that key on these counts consult this first. Defaults to False: a fact
    #: set that does not say Step 1 ran has not earned a pass on gaps or outliers, and the caller
    #: that really did profile the series states so explicitly.
    profile_ran: bool = False

    # S1/S2 preprocessing facts
    n_outliers: int = 0
    outliers_addressed: bool = False
    variance_ratio: float | None = None          # level-to-spread correlation
    variance_growth_pct: float | None = None
    signal_to_noise: float | None = None
    seasonal_strength: float | None = None
    trend_r2: float | None = None
    skewness: float | None = None
    excess_kurtosis: float | None = None

    # S4
    remainder_share: float | None = None
    remainder_lb_p: float | None = None
    remainder_arch_p: float | None = None

    # S6
    correlogram_pattern: str | None = None
    ljung_box_p: float | None = None

    # what has been done to the series
    n_transformations: int = 0
    seasonality_handled: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- #
# Thresholds, named and justified
# --------------------------------------------------------------------------- #
#: §5.11 warns when the horizon exceeds n/5, because a model cannot extrapolate far beyond the
#: variation it has seen. This is the same line, applied before the forecast is made rather than
#: after.
HORIZON_RATIO = 5.0

#: Below this, most model families cannot estimate a seasonal component: you need several complete
#: cycles for the seasonal terms to mean anything. Two full cycles is the bare minimum.
MIN_CYCLES = 2

#: Absolute floor for any model with more than a mean and a variance.
MIN_N = 20

#: Hyndman's strength threshold for "a seasonal term earns its parameter".
SEASONAL_STRONG = 0.30

#: A remainder this large means the model has little structure to explain.
REMAINDER_HEAVY = 0.80


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #
def _length(f: GateFacts) -> Criterion:
    if f.n < MIN_N:
        cycles = (f.n / f.period) if f.period and f.period >= 2 else None
        return Criterion(
            key="length", label="Enough observations to fit a model", state="fail",
            value=f"{f.n} observations"
                  + (f" = {cycles:.1f} seasonal cycles of {f.period}" if cycles else ""),
            detail=(
                f"A model needs at least about {MIN_N} observations before its parameters are "
                f"estimable at all; this series has {f.n}"
                + (f", which is {cycles:.1f} complete seasonal cycles of {f.period}" if cycles else "")
                + ". Anything fitted to it would be a curve drawn through a handful of points."
            ),
            jump_to_stage="S0", severity="blocker",
        )
    usable = f.n
    note = ""
    if f.period and f.period >= 2:
        cycles = usable / f.period
        if cycles < MIN_CYCLES:
            # A caveat, not a blocker. This check is about *seasonal* terms, and the detail below
            # explicitly says a non-seasonal model is still possible. Marking it "blocker" made
            # the gate return `not_ready`, which made S10 refuse every fit — so the bundled
            # 100-minute wwwusage series (nominal daily period 1440) could see seven candidates
            # and fit none of them. A missing seasonal option narrows the candidate set; it does
            # not make drift, ARIMA or ETS invalid. The travelling caveat carries the restriction.
            return Criterion(
                key="length", label="Enough observations for a seasonal term", state="fail",
                value=f"{usable} observations = {cycles:.1f} seasonal cycles of {f.period}",
                detail=(
                    f"Only {cycles:.1f} complete seasonal cycles are available. A seasonal term "
                    f"estimated from fewer than {MIN_CYCLES} cycles is mostly extrapolation: the "
                    "model cannot tell a stable seasonal shape from a coincidence. Seasonal models "
                    "are excluded from the candidate set; non-seasonal models are still possible, "
                    "or aggregate to a coarser frequency. This is a restriction on what may be "
                    "fitted, not a reason to stop."
                ),
                jump_to_stage="S2", severity="caveat",
            )
        note = f" ({cycles:.1f} complete cycles of {f.period})"
    return Criterion(
        key="length", label="Enough observations", state="pass",
        value=f"{usable} observations" + (f", horizon {f.horizon}" if f.horizon else ""),
        detail=(
            f"{usable} usable observations{note}. "
            + (
                f"The requested horizon of {f.horizon} is {usable / f.horizon:.1f}× — "
                + ("comfortably inside" if usable / f.horizon >= HORIZON_RATIO else
                   "beyond")
                + f" the {HORIZON_RATIO:.0f}× the data supports."
                if f.horizon else "No forecast horizon has been requested yet."
            )
        ),
        jump_to_stage=None,
    )


def _horizon(f: GateFacts) -> Criterion:
    if not f.horizon:
        # Not a gap in the evidence — a question that has not been asked. An unset horizon cannot
        # be too long, so it neither passes nor blocks; it is simply not applicable until the user
        # has a forecast to make. Treating this as `unknown` would make every series "inconclusive"
        # merely for not having planned its forecast yet, which would train users to ignore the
        # word.
        return Criterion(
            key="horizon", label="Forecast horizon is inside what the data supports",
            state="not_applicable", value="no horizon chosen yet",
            detail=(
                "No forecast horizon has been set, so there is nothing to check. Set one in "
                "Step 10 and this line will report whether the sample can support it."
            ),
            jump_to_stage="S13",
        )
    # Written as "n is 2.4 horizons" rather than "the horizon is 2.4×": the guide is a *minimum
    # sample per step ahead*, so the multiplier belongs to the sample. The earlier wording had it
    # backwards, which made a failing horizon read like a passing one ("12.0× ... inside the guide").
    ratio = f.n / f.horizon
    reach = f"the sample is {ratio:.1f} forecast horizons long"
    if ratio < HORIZON_RATIO:
        return Criterion(
            key="horizon", label="Forecast horizon is inside what the data supports",
            state="fail",
            value=f"horizon {f.horizon} with {f.n} observations ({ratio:.1f}×)",
            detail=(
                f"Forecasting {f.horizon} steps ahead means {reach}, and the guide asks for about "
                f"{HORIZON_RATIO:.0f}×. Past that point intervals widen faster than the point "
                "forecast improves and the model has usually stopped adding information. Shorten "
                "the horizon, or lengthen the series."
            ),
            jump_to_stage="S13", severity="caveat",
        )
    return Criterion(
        key="horizon", label="Forecast horizon is inside what the data supports", state="pass",
        value=f"horizon {f.horizon} with {f.n} observations ({ratio:.1f}×)",
        detail=(
            f"{reach.capitalize()} — above the {HORIZON_RATIO:.0f}× guide, so the sample can "
            "support a forecast this far ahead."
        ),
    )


def _gaps(f: GateFacts) -> Criterion:
    if not f.profile_ran:
        return Criterion(
            key="gaps", label="No unresolved gaps", state="unknown", value="not measured",
            detail=(
                "Step 1 has not profiled the series, so \"no missing values\" is unconfirmed. "
                "Missingness changes which models are estimable and whether an imputation has to "
                "be declared."
            ),
            why_unknown="Step 1 (Explore) has not been run.",
            jump_to_stage="S1",
        )
    if f.n_missing == 0:
        return Criterion(
            key="gaps", label="No unresolved gaps", state="pass", value="no missing values",
            detail="Every observation in the window has a value.",
        )
    if f.longest_gap and f.longest_gap >= max(3, 0.1 * f.n):
        return Criterion(
            key="gaps", label="No unresolved gaps", state="fail",
            value=f"{f.n_missing} missing, longest run {f.longest_gap}",
            detail=(
                f"{f.n_missing} values are missing and the longest unbroken run is "
                f"{f.longest_gap} — over a tenth of the series. Imputing a run that long invents "
                "most of a season, and every test computed across it describes data that is not "
                "there."
            ),
            jump_to_stage="S2", severity="blocker",
        )
    return Criterion(
        key="gaps", label="No unresolved gaps", state="fail",
        value=f"{f.n_missing} missing, longest run {f.longest_gap}",
        detail=(
            f"{f.n_missing} values are missing (longest run {f.longest_gap}). A short gap can be "
            "imputed honestly, but the imputation must be recorded and the model's intervals will "
            "be slightly too narrow wherever it was used."
        ),
        jump_to_stage="S2", severity="caveat",
    )


def _stationarity(f: GateFacts) -> Criterion:
    label = "Stationarity established with agreeing tests"
    if not f.stationarity_verdict:
        return Criterion(
            key="stationarity", label=label, state="unknown", value="Step 5 has not run",
            detail="The gate will not judge stationarity it has not tested.",
            why_unknown="Step 5 (Stationarity) has not been run on this series.",
            jump_to_stage="S5",
        )
    v = f.stationarity_verdict
    if v in {"conflicting"} or (f.n_contradictions >= 2 and f.n_agreeing < 2):
        return Criterion(
            key="stationarity", label=label, state="fail",
            value=f"{v} — {f.n_agreeing} agreeing, {f.n_contradictions} contradicting",
            detail=(
                f"The tests conflict ({f.n_contradictions} contradicting). Modelling a series whose "
                "order of integration is unknown is not a small risk: d = 0 and d = 1 give "
                "different forecasts, different intervals and different residual diagnostics, and "
                "the data has not said which."
            ),
            jump_to_stage="S5", severity="blocker",
        )
    if v == "inconclusive":
        return Criterion(
            key="stationarity", label=label, state="fail",
            value="inconclusive",
            detail=(
                "The tests ran but did not agree on an answer. This is a legitimate outcome, and it "
                "means the model's order of integration is a judgement call rather than a finding "
                "— which must be stated in any report that uses it."
            ),
            jump_to_stage="S5", severity="caveat",
        )
    if f.n_agreeing == 0:
        return Criterion(
            key="stationarity", label=label, state="fail",
            value=f"{v.replace('_', ' ')} with no supporting test",
            detail=(
                "Step 5 produced a verdict that no test in the battery supports, so there is "
                "nothing here to quote. Re-run Step 5 — and if it says the same thing again, treat "
                "the order of integration as undecided rather than as known."
            ),
            jump_to_stage="S5", severity="blocker",
        )
    if f.n_agreeing < 2:
        return Criterion(
            key="stationarity", label=label, state="fail",
            value=f"{v.replace('_', ' ')} on a single test",
            detail=(
                f"The verdict is {v.replace('_', ' ')}, but only {f.n_agreeing} test supports it. "
                "One test is a hypothesis, not a finding — ADF and KPSS have opposite nulls, and "
                "agreement between them is what makes a verdict worth quoting."
            ),
            jump_to_stage="S5", severity="caveat",
        )
    n = f.n_agreeing
    witnesses = "1 seasonal witness" if n == 1 else f"{n} seasonal witnesses"
    plural = "" if n == 1 else "s"
    if v == "seasonal_unit_root":
        value = (
            f"seasonal unit root ({witnesses}"
            + (f"; {f.n_regular_supporting} of {f.n_regular_tests} regular tests also find a unit root"
               if f.n_regular_tests else "")
            + f"), confidence {f.stationarity_confidence or 'unstated'}"
        )
        parts = [f"{witnesses} put the unit root at the seasonal lag."]
        if f.n_regular_tests:
            parts.append(
                f"{f.n_regular_supporting} of the {f.n_regular_tests} regular unit-root tests also "
                "fail to reject a unit root, which is consistent with — but a different claim from "
                "— the seasonal finding."
            )
        parts.append(
            "A seasonal difference at the detected period is what removes it."
            if f.differencing_needed
            else "Whether to take a seasonal difference as well is a judgement, not a finding."
        )
        detail = " ".join(parts)
        return Criterion(
            key="stationarity", label=label, state="pass", value=value, detail=detail,
            jump_to_stage="S5",
        )
    return Criterion(
        key="stationarity", label=label, state="pass",
        value=f"{v.replace('_', ' ')} ({n} test{plural} agree, confidence "
              f"{f.stationarity_confidence or 'unstated'})",
        detail=(
            f"{n} tests agree on {v.replace('_', ' ')}"
            + (f", with {f.n_contradictions} dissenting" if f.n_contradictions else
               " and none dissents")
            + (". Differencing is needed before modelling."
               if f.differencing_needed else ". The series can be modelled in levels.")
        ),
        jump_to_stage="S5",
    )


def _breaks(f: GateFacts) -> Criterion:
    label = "No unmodelled structural break"
    if not f.breaks_tested:
        return Criterion(
            key="breaks", label=label, state="unknown", value="not measured",
            detail=(
                "No structural-break test has run, so \"no break found\" is not yet a finding. "
                "A level shift left in the series fakes a unit root and inflates residual variance, "
                "which changes which models are even appropriate."
            ),
            why_unknown="Step 5 (Stationarity) has not run its break tests.",
            jump_to_stage="S5",
        )
    if f.break_count == 0:
        return Criterion(
            key="breaks", label=label, state="pass", value="no break detected",
            detail="No structural-break test found a date that survives its critical value.",
            jump_to_stage="S5",
        )
    dates = ", ".join(f.break_dates[:3])
    return Criterion(
        key="breaks", label=label, state="fail",
        value=f"{f.break_count} break(s): {dates}",
        detail=(
            f"A break was detected at {dates}. This is not a data-quality problem — it is "
            "information, and the fix is a step dummy handed to the model, not deletion. A model "
            "fitted across an unmodelled shift will spread the jump into its residuals and "
            "mis-state its own uncertainty."
        ),
        jump_to_stage="S5", severity="caveat",
    )


def _outliers(f: GateFacts) -> Criterion:
    label = "Outliers addressed or explicitly flagged"
    if not f.profile_ran:
        return Criterion(
            key="outliers", label=label, state="unknown", value="not measured",
            detail=(
                "Step 1 has not been run, so nothing has looked for outliers. \"None found\" is "
                "not the same as \"none looked for\"."
            ),
            why_unknown="Step 1 (Explore) has not been run.",
            jump_to_stage="S1",
        )
    if f.n_outliers == 0:
        return Criterion(
            key="outliers", label=label, state="pass", value="no outliers flagged",
            detail="No observation is far enough from its neighbours to be flagged.",
            jump_to_stage="S1",
        )
    if f.outliers_addressed:
        return Criterion(
            key="outliers", label=label, state="pass",
            value=f"{f.n_outliers} flagged, treatment applied",
            detail=(
                f"{f.n_outliers} observations were flagged and a treatment has been applied and "
                "recorded, so the model sees a defined series."
            ),
            jump_to_stage="S1",
        )
    return Criterion(
        key="outliers", label=label, state="fail",
        value=f"{f.n_outliers} flagged, untreated",
        detail=(
            f"{f.n_outliers} observations were flagged and nothing has been done about them. That "
            "is a defensible choice — an outlier may be real — but it must be a *choice*: the "
            "diagnostic plots will be dominated by those points and the residual variance will be "
            "inflated by them."
        ),
        jump_to_stage="S1", severity="caveat",
    )


def _variance(f: GateFacts) -> Criterion:
    label = "Variance stable enough for a constant-variance model"
    if f.variance_ratio is None and f.variance_growth_pct is None:
        return Criterion(
            key="variance", label=label, state="unknown", value="not measured",
            detail="The variance-stability check needs Step 1's profile.",
            why_unknown="Step 1 (Explore) has not produced variance facts.",
            jump_to_stage="S1",
        )
    corr = f.variance_ratio
    growth = f.variance_growth_pct
    detail_bits = []
    if corr is not None:
        detail_bits.append(f"level-to-spread rank correlation {corr:.2f}")
    if growth is not None:
        # "grew -19%" is the kind of sign error a reader has to decode; state the direction once.
        detail_bits.append(
            f"spread grew {growth:.0f}% across the window" if growth >= 0
            else f"spread shrank {abs(growth):.0f}% across the window"
        )
    if (corr is not None and corr > 0.5) or (growth is not None and growth > 50):
        return Criterion(
            key="variance", label=label, state="fail",
            value="; ".join(detail_bits),
            detail=(
                "The spread tracks the level, so a constant-variance model will be wrong at one "
                "end of the series whichever end it is fitted to. A log or Box–Cox transform is "
                "the usual fix — and if one is applied, the forecasts must be back-transformed "
                "with a bias correction."
            ),
            jump_to_stage="S2", severity="caveat",
        )
    return Criterion(
        key="variance", label=label, state="pass", value="; ".join(detail_bits) or "stable",
        detail="The spread does not systematically track the level.",
        jump_to_stage="S1",
    )


def _seasonality(f: GateFacts) -> Criterion:
    label = "Seasonality handled or shown to be absent"
    if not f.period or f.period < 2:
        # Not a pass: there is nothing to have handled. Reporting "pass" would put a green tick
        # against a question the data never asked, and would hide a Step 0 calendar miss.
        return Criterion(
            key="seasonality", label=label, state="not_applicable",
            value="no seasonal period detected",
            detail=(
                "No calendar period was detected, so there is no seasonal structure to handle. If "
                "the data does repeat on a cycle this app did not detect, that is a Step 0 mapping "
                "problem rather than a modelling one."
            ),
            jump_to_stage="S1",
        )
    if f.seasonality_handled:
        return Criterion(
            key="seasonality", label=label, state="pass",
            value=f"period {f.period}, handled",
            detail=f"A seasonal treatment or seasonal model term covers the period-{f.period} cycle.",
            jump_to_stage="S4",
        )
    if f.seasonal_strength is None:
        # No measurement is not the same as a weak season. Reporting "pass (weak)" here would be
        # the exact failure the spec forbids — a tick with no number behind it.
        return Criterion(
            key="seasonality", label=label, state="unknown",
            value=f"period {f.period}, seasonal strength not measured",
            detail=(
                f"A period of {f.period} was detected, but its strength has not been measured, so "
                "the gate cannot say whether an unhandled season matters."
            ),
            why_unknown="Step 4 (Decompose) has not measured the seasonal strength.",
            jump_to_stage="S4",
        )
    if f.seasonal_strength >= SEASONAL_STRONG:
        return Criterion(
            key="seasonality", label=label, state="fail",
            value=f"period {f.period}, F_S = {f.seasonal_strength:.2f}",
            detail=(
                f"The series repeats every {f.period} observations and the seasonal strength is "
                f"{f.seasonal_strength:.2f}, above the {SEASONAL_STRONG:.2f} threshold where a "
                "seasonal term earns its parameter. Nothing has handled it yet, so a non-seasonal "
                "model would put the whole seasonal swing into its residuals."
            ),
            jump_to_stage="S4", severity="blocker",
        )
    return Criterion(
        key="seasonality", label=label, state="pass",
        value=f"period {f.period}, F_S = {f.seasonal_strength:.2f} (below "
              f"{SEASONAL_STRONG:.2f})",
        detail=(
            f"A period of {f.period} was detected but the seasonal strength is below "
            f"{SEASONAL_STRONG:.2f}, so leaving it unhandled costs little."
        ),
        jump_to_stage="S4",
    )


def _signal(f: GateFacts) -> Criterion:
    label = "Adequate signal for a model to find"
    if f.signal_to_noise is None and f.remainder_share is None:
        return Criterion(
            key="signal", label=label, state="unknown", value="not measured",
            detail=(
                "Neither the signal-to-noise ratio (Step 2) nor the decomposition's remainder share "
                "(Step 4) has been computed, so the gate cannot say how much of this series is "
                "predictable."
            ),
            why_unknown="Steps 2 and 4 have not measured how much of the variation is structural.",
            jump_to_stage="S4",
        )
    if f.signal_to_noise is not None and f.signal_to_noise < 0.2:
        return Criterion(
            key="signal", label=label, state="fail",
            value=f"signal-to-noise {f.signal_to_noise:.2f}",
            detail=(
                f"The predictable part of this series is {f.signal_to_noise:.2f} of its total "
                "variation. Anything fitted here will track the noise as much as the signal, and "
                "no model will beat a well-chosen baseline by much — which is worth knowing "
                "*before* spending effort on the search."
            ),
            jump_to_stage="S4", severity="caveat",
        )
    if f.remainder_share is not None and f.remainder_share > REMAINDER_HEAVY:
        return Criterion(
            key="signal", label=label, state="fail",
            value=f"remainder is {f.remainder_share * 100:.0f}% of the variation",
            detail=(
                f"{f.remainder_share * 100:.0f}% of the variation is in the remainder. That is not "
                "necessarily noise — check the remainder card in Step 4 for whether it is white — "
                "but there is little for trend or seasonal terms to explain."
            ),
            jump_to_stage="S4", severity="caveat",
        )
    bits = []
    if f.signal_to_noise is not None:
        bits.append(f"signal-to-noise {f.signal_to_noise:.2f}")
    if f.remainder_share is not None:
        bits.append(f"remainder {f.remainder_share * 100:.0f}% of variation")
    return Criterion(
        key="signal", label=label, state="pass", value="; ".join(bits) or "not measured",
        detail="There is structure for a model to find, beyond what the remainder holds.",
        jump_to_stage="S4",
    )


def _correlogram(f: GateFacts) -> Criterion:
    label = "Residual-relevant dependence examined (ACF/PACF)"
    if f.correlogram_pattern is None:
        return Criterion(
            key="correlogram", label=label, state="unknown", value="Step 6 has not run",
            detail="The gate cannot say whether dependence remains without the correlogram.",
            why_unknown="Step 6 (ACF/PACF) has not been run.",
            jump_to_stage="S6",
        )
    return Criterion(
        key="correlogram", label=label, state="pass",
        value=f"{f.correlogram_pattern.replace('_', ' ')}"
              + (f", Ljung–Box {p_phrase(f.ljung_box_p)}" if f.ljung_box_p is not None else ""),
        detail=(
            "The correlogram has been read and is reported alongside the model, so the choice of "
            "AR/MA terms has stated evidence rather than a guess behind it."
        ),
        jump_to_stage="S6",
    )


CHECKS = (
    _length, _horizon, _gaps, _stationarity, _breaks,
    _outliers, _variance, _seasonality, _signal, _correlogram,
)


# --------------------------------------------------------------------------- #
# The verdict
# --------------------------------------------------------------------------- #
@dataclass
class Gate:
    verdict: str                     # ready | ready_with_caveats | not_ready | inconclusive
    headline: str
    plain: str
    checklist: list[Criterion] = field(default_factory=list)
    blockers: list[Criterion] = field(default_factory=list)
    caveats: list[Criterion] = field(default_factory=list)
    unknown: list[Criterion] = field(default_factory=list)
    caveats_travelling: list[str] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checklist if c.state == "pass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "headline": self.headline,
            "plain": self.plain,
            "checklist": [c.to_dict() for c in self.checklist],
            "blockers": [c.to_dict() for c in self.blockers],
            "caveats": [c.to_dict() for c in self.caveats],
            "unknown": [c.to_dict() for c in self.unknown],
            "caveats_travelling": list(self.caveats_travelling),
            "n_pass": self.passed,
            "n_fail": sum(1 for c in self.checklist if c.state == "fail"),
            "n_unknown": len(self.unknown),
            "n_not_applicable": sum(1 for c in self.checklist if c.state == "not_applicable"),
            "n_total": len(self.checklist),
        }


def _travelling_caveats(g: Gate, f: GateFacts) -> list[str]:
    """
    The caveats that must follow every downstream number into the report.

    Written as complete sentences that stand alone, because they will appear attached to a
    forecast, a table of metrics and a methods paragraph — three places with no checklist in sight.
    A caveat that only makes sense next to the gate is not a travelling caveat.
    """
    out: list[str] = []
    seen: set[str] = set()
    for c in g.blockers + g.caveats:
        # The horizon is also written as its own sentence below (with the ratio and the
        # consequence), so listing it here as well printed the same fact twice in a list whose
        # whole job is to be read carefully.
        if c.key == "horizon":
            continue
        text = f"{c.label}: {c.value}."
        if text not in seen:
            seen.add(text)
            out.append(text)
    if f.stationarity_verdict:
        out.append(
            f"Stationarity verdict for this series: "
            f"{f.stationarity_verdict.replace('_', ' ')}"
            + (f" with {f.n_agreeing} tests agreeing" if f.n_agreeing else "")
            + "."
        )
    if f.differencing_needed:
        out.append(
            "All models are fitted to the differenced series; point forecasts are of changes and "
            "have been integrated back to the original scale."
        )
    if f.break_count:
        out.append(
            f"{f.break_count} structural break(s) are modelled with step dummies rather than "
            "removed; the break dates are estimates, not known dates."
        )
    if f.n_missing:
        out.append(
            f"{f.n_missing} missing observations were imputed; intervals are slightly too narrow "
            "wherever imputed values were used."
        )
    if f.horizon and f.n / max(1, f.horizon) < HORIZON_RATIO:
        out.append(
            f"The forecast horizon of {f.horizon} is long relative to the {f.n} observations "
            "available; accuracy should be expected to degrade with horizon."
        )
    return out


def assess(f: GateFacts) -> Gate:
    """
    Run every check and turn the results into one verdict.

    The mapping is deliberate, so the headline can always be traced to the lines above it:

    * any **blocker** failed                              → `not_ready`
    * no blockers, but **any check unassessed**           → `inconclusive`
    * no blockers, no unknowns, but **caveats**           → `ready_with_caveats`
    * nothing failed, nothing unknown                     → `ready`

    **Unknowns outrank caveats**, which is the less obvious ordering and the important one. If
    stationarity has not been tested, "ready with caveats" would be a false assurance: the caveat
    list is complete *for the checks that ran*, and the most consequential check of all is simply
    missing. The gate reports `inconclusive` and names the stages still to run. Once they have run,
    the same series will settle into `ready`, `ready_with_caveats` or `not_ready` on the evidence.

    Caveats are still listed under `inconclusive` — they are true regardless — but they do not
    upgrade the verdict.
    """
    checklist = [check(f) for check in CHECKS]

    blockers = [c for c in checklist if c.state == "fail" and c.severity == "blocker"]
    caveats = [c for c in checklist if c.state == "fail" and c.severity == "caveat"]
    unknown = [c for c in checklist if c.state == "unknown"]
    n_pass = sum(1 for c in checklist if c.state == "pass")

    if blockers:
        verdict = "not_ready"
    elif unknown:
        verdict = "inconclusive"
    elif caveats:
        verdict = "ready_with_caveats"
    else:
        verdict = "ready"

    if verdict == "not_ready":
        names = "; ".join(c.label.lower() for c in blockers)
        headline = f"Not ready: {len(blockers)} blocker" + ("s" if len(blockers) != 1 else "") \
                   + f" must be cleared first"
        plain = (
            f"{len(blockers)} of the {len(checklist)} readiness checks failed in a way that would "
            f"make a model's output wrong rather than merely limited: {names}. Clear "
            + ("them" if len(blockers) > 1 else "it")
            + " in the stage each line points to, then run the gate again."
        )
    elif verdict == "ready_with_caveats":
        headline = f"Ready with caveats: {len(caveats)} thing" \
                   + ("s" if len(caveats) != 1 else "") + " to carry into the report"
        plain = (
            f"Nothing blocks modelling. {n_pass} of the {len(checklist)} checks pass outright, and "
            f"{len(caveats)} fail in ways that limit what the output can claim rather than "
            "invalidating it: " + "; ".join(c.label.lower() for c in caveats) + ". These caveats "
            "travel with every downstream result and into the final report."
        )
    elif verdict == "inconclusive":
        headline = f"Inconclusive: {len(unknown)} check" \
                   + ("s" if len(unknown) != 1 else "") + " could not be assessed"
        plain = (
            f"{len(unknown)} of the {len(checklist)} checks could not be evaluated because the "
            "stage that measures them has not run: "
            + "; ".join(c.label.lower() for c in unknown)
            + ". "
            + (
                f"{len(caveats)} check(s) already fail and will still apply: "
                + "; ".join(c.label.lower() for c in caveats) + ". "
                if caveats else ""
            )
            + "Running the missing stages is the only way to turn this into a verdict — the gate "
            "will not assume they would pass."
        )
    else:
        # `not_applicable` is a real state, so a headline that counts them as passes would claim
        # ten passing checks on a screenshot that shows nine and one "n/a".
        n_na = sum(1 for c in checklist if c.state == "not_applicable")
        headline = (
            f"Ready: all {n_pass} checks pass"
            + (f" ({n_na} not applicable)" if n_na else "")
        )
        plain = (
            "Every readiness check passes with its number attached. Stationarity is established by "
            f"{max(1, f.n_agreeing)} agreeing tests, "
            + (
                f"the {f.horizon}-period horizon is within what {f.n} observations can support, "
                if f.horizon else
                "no horizon has been set yet so there is none to be too long, "
            )
            + "and there are no unhandled gaps, breaks or variance problems. Modelling can proceed."
        )

    gate = Gate(
        verdict=verdict,
        headline=headline,
        plain=plain,
        checklist=checklist,
        blockers=blockers,
        caveats=caveats,
        unknown=unknown,
    )
    gate.caveats_travelling = _travelling_caveats(gate, f)
    return gate


__all__ = [
    "Criterion", "GateFacts", "Gate", "assess", "CHECKS",
    "HORIZON_RATIO", "MIN_CYCLES", "MIN_N", "SEASONAL_STRONG", "REMAINDER_HEAVY",
]
