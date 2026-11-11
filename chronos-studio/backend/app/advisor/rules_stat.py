"""
advisor/rules_stat.py — S5: what the stationarity and break results mean for the next step (§5.5).

Same contract as `rules_prep`: a rule is a pure function over computed facts, and its `because`
interpolates the numbers that were actually computed. Nothing here re-runs a test; the rules read
the `StatContext` that `api/tests_stat.py` fills from `ts/tests/`.

Three rules carry more weight than the rest:

* `stat.already_stationary.no_diff` — the guardrail. If the series is already stationary,
  differencing is not neutral: it introduces a moving-average unit root, inflates the residual
  variance, and makes forecasts worse. The app says so explicitly rather than offering
  differencing as a harmless default.

* `stat.break.dummy_not_delete` — a detected break is a *feature*, not dirt. Deleting the
  affected observations or differencing through the shift both destroy information the model
  needs. The recommendation is a step dummy handed forward to the model stage.

* `stat.seasonal_unit.diff_at_lag` — when the unit root is seasonal, `diff(1)` does not fix it.
  The recommendation names the lag, because "difference it" with the wrong lag is worse than
  no advice at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.schemas import Confidence
from app.core.formatting import p_phrase


# --------------------------------------------------------------------------- #
# The context a rule sees
# --------------------------------------------------------------------------- #
@dataclass
class StatContext:
    """Every number an S5 rule is allowed to know. All computed, none typed by hand."""

    # identity
    series_name: str = "your series"
    n: int = 0
    period: int | None = None
    frequency_label: str = "observation"

    # the reconciled verdict
    verdict: str = "inconclusive"
    confidence: str = "low"
    differencing_needed: bool = False

    # the raw test numbers the verdict cites
    adf_stat: float | None = None
    adf_p: float | None = None
    adf_rejects: bool | None = None
    kpss_stat: float | None = None
    kpss_p: float | None = None
    kpss_rejects: bool | None = None
    dfgls_stat: float | None = None

    # seasonal
    ocsb_stat: float | None = None
    ocsb_cv5: float | None = None
    ocsb_rejects: bool | None = None

    # breaks
    break_dates: list[str] = field(default_factory=list)
    n_breaks: int = 0
    qa_stat: float | None = None
    qa_rejects: bool | None = None
    za_stat: float | None = None
    za_rejects: bool | None = None

    # how many other tests dissent from the headline
    n_contradictions: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class StatRule:
    id: str
    stage: str
    recommend: str
    when: Any                                        # Callable[[StatContext], bool]
    because: Any                                     # Callable[[StatContext], str]
    confidence: Confidence = Confidence.MEDIUM
    provenance: list[str] = field(default_factory=list)
    alternative: str | None = None
    risk_if_ignored: str = ""
    how_to: str = ""
    kind: str = "recommend"                          # recommend | blocker | protect | note

    def applies(self, c: StatContext) -> bool:
        try:
            return bool(self.when(c))
        except Exception:                            # pragma: no cover - a rule must never crash a stage
            return False

    def to_out(self, c: StatContext) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "recommend": self.recommend,
            "because": self.because(c),
            "confidence": self.confidence.value,
            "provenance": list(self.provenance),
            "alternative": self.alternative,
            "risk_if_ignored": self.risk_if_ignored,
            "how_to": self.how_to,
            "kind": self.kind,
        }


def _num(x: float | None, digits: int = 2) -> str:
    """Numbers in citable sentences never carry thousands separators (R0.5)."""
    if x is None:
        return "not computed"
    return f"{x:.{digits}f}"


def _dates(c: StatContext) -> str:
    if not c.break_dates:
        return "the detected date"
    return ", ".join(c.break_dates[:3]) + (" and others" if len(c.break_dates) > 3 else "")


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
RULES: list[StatRule] = [
    StatRule(
        id="stat.already_stationary.no_diff",
        stage="S5",
        kind="protect",
        recommend="Do not difference this series",
        when=lambda c: c.verdict == "stationary" and not c.differencing_needed,
        because=lambda c: (
            f"ADF does not find a unit root here (statistic {_num(c.adf_stat)}"
            + (f", {p_phrase(c.adf_p, 3)}" if c.adf_p is not None else "")
            + f") and KPSS does not reject a constant mean (statistic {_num(c.kpss_stat)}). "
            "Differencing a stationary series is not a harmless precaution: it adds a "
            "moving-average unit root, inflates the residual variance, and makes every "
            "interval that follows wider than it needed to be."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.adf.constant_trend", "stat.kpss.level"],
        alternative="If you want the *change* rather than the level, difference deliberately "
                    "and say so — that is a different question, not a prerequisite.",
        risk_if_ignored="Forecasts get noisier and intervals widen for no gain; MASE will look "
                        "worse than the correctly specified model.",
        how_to="Step 4 → keep 'no differencing' selected and go straight to the model stage.",
    ),
    StatRule(
        id="stat.unitroot.diff_once",
        stage="S5",
        recommend="Difference the series once (d = 1)",
        when=lambda c: c.verdict == "unit_root" and c.differencing_needed,
        because=lambda c: (
            f"ADF does not reject a unit root (statistic {_num(c.adf_stat)}"
            + (f", {p_phrase(c.adf_p, 3)}" if c.adf_p is not None else "")
            + f") and KPSS rejects stability of the mean (statistic {_num(c.kpss_stat)}). "
            "Two tests with opposite nulls agreeing is the strongest statement this battery "
            "can make. With a unit root the level is a running total of past shocks, so a "
            "model has to be built on the changes."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.adf.constant_trend", "stat.kpss.level"],
        alternative="A model that handles integration internally (ARIMA with d = 1, or ETS on "
                    "the level) does the same thing without a separate differencing step.",
        risk_if_ignored="Regressing a unit-root series on its own lags produces spurious "
                        "significance and intervals that are far too narrow.",
        how_to="Step 5 → 'Difference once', then re-run these tests on the differenced series.",
    ),
    StatRule(
        id="stat.unitroot.check_second_diff",
        stage="S5",
        kind="note",
        recommend="Check the differenced series before differencing again",
        when=lambda c: c.verdict == "unit_root" and c.differencing_needed,
        because=lambda c: (
            "One difference is the default, not a rule: about half of series that need "
            "differencing were already stationary afterwards. Re-run this battery on the "
            "differenced series — if ADF rejects there, d = 1 is enough."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.adf.constant_trend"],
        alternative=None,
        risk_if_ignored="Over-differencing creates a moving-average unit root that is hard to "
                        "fit and easy to mistake for a wrong model order.",
        how_to="Step 5 → after differencing, the battery runs again automatically on the new series.",
    ),
    StatRule(
        id="stat.break.dummy_not_delete",
        stage="S5",
        kind="blocker",
        recommend="Add a step dummy at the break date instead of transforming it away",
        when=lambda c: c.verdict == "stationary_around_break",
        because=lambda c: (
            f"The level changes at {_dates(c)} and, allowing for that change, Zivot–Andrews "
            "rejects the unit-root null — so this is a stable series with one interruption, "
            "not a wandering one. Differencing would turn the jump into a single spike and "
            "delete the level information the model needs."
        ),
        confidence=Confidence.HIGH,
        provenance=["stat.quandt_andrews.trend", "stat.bai_perron", "stat.za.both"],
        alternative="Model the two regimes separately if the relationship itself — not just the "
                    "level — changed at the break.",
        risk_if_ignored="Differencing a broken series makes ADF and KPSS disagree downstream and "
                        "produces a forecast that never recovers the new level.",
        how_to="Step 8 → add a step dummy at the break date (the app supplies it as a column).",
    ),
    StatRule(
        id="stat.seasonal_unit.diff_at_lag",
        stage="S5",
        recommend="Take a seasonal difference, not a plain one",
        when=lambda c: c.verdict == "seasonal_unit_root",
        because=lambda c: (
            f"OCSB does not reject a seasonal unit root (statistic {_num(c.ocsb_stat)} against a "
            f"5% critical value of {_num(c.ocsb_cv5)}): each period's value carries forward from "
            f"the same period last year. With a period of {c.period}, a plain first difference "
            "leaves that dependence untouched — it has to be differenced at the seasonal lag."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.ocsb", "stat.adf.constant_trend"],
        alternative="Use a model with seasonal differencing built in (SARIMA with D = 1) rather "
                    "than differencing the data by hand.",
        risk_if_ignored="The seasonal pattern is treated as a fixed shape when it is actually "
                        "accumulating error; forecasts inherit last year's deviation forever.",
        how_to="Step 5 → choose 'seasonal difference', then re-run the battery.",
    ),
    StatRule(
        id="stat.trend_stationary.detrend",
        stage="S5",
        recommend="Include a trend term rather than differencing",
        when=lambda c: c.verdict == "trend_stationary",
        because=lambda c: (
            f"ADF rejects a unit root (statistic {_num(c.adf_stat)}) but KPSS rejects a constant "
            f"mean (statistic {_num(c.kpss_stat)}) while not rejecting trend-stationarity. The "
            "series returns to a straight path: it is the path that slopes, not the shocks that "
            "persist."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.adf.constant_trend", "stat.kpss.trend"],
        alternative="If the trend itself is expected to bend, a damped trend or a structural "
                    "model is safer than an indefinite straight line.",
        risk_if_ignored="Differencing a trend-stationary series removes the trend and leaves "
                        "noise; the forecast then has no direction at all.",
        how_to="Step 8 → include 'trend' in the model specification.",
    ),
    StatRule(
        id="stat.inconclusive.no_credible_verdict",
        stage="S5",
        kind="note",
        recommend="Treat the stationarity verdict as provisional",
        when=lambda c: c.verdict in {"inconclusive", "conflicting"} or c.confidence == "low",
        because=lambda c: (
            f"The tests do not agree closely enough to settle this on {c.n} observations"
            + (f"; {c.n_contradictions} of the supporting tests point the other way"
               if c.n_contradictions else "")
            + ". A confident recommendation here would be manufactured."
        ),
        confidence=Confidence.LOW,
        provenance=["stat.adf.constant_trend", "stat.kpss.level"],
        alternative="Fit both a levels and a differenced model and compare them on held-out "
                    "error — the comparison is better evidence than the tests.",
        risk_if_ignored="A wrong differencing decision is expensive and hard to see in-sample; "
                        "comparison catches it, a single test does not.",
        how_to="Step 5 → note both options; Step 11 lets you compare them side by side.",
    ),
    StatRule(
        id="stat.ambiguous.carry_both",
        stage="S5",
        recommend="Carry both readings forward and let the model decide",
        when=lambda c: c.verdict == "unit_root_or_break",
        because=lambda c: (
            "ADF and KPSS read a unit root (ADF "
            f"{_num(c.adf_stat)}, KPSS {_num(c.kpss_stat)}) while Zivot–Andrews reads a single "
            f"level shift at {_dates(c)}. On this sample the two are observationally close, so "
            "the honest move is to fit both"
            + (f" — difference once and keep a step dummy at {_dates(c)}" if c.break_dates else "")
            + " — and compare them on held-out error."
        ),
        confidence=Confidence.LOW,
        provenance=["stat.adf.constant_trend", "stat.kpss.level", "stat.za.both"],
        alternative="Choose the unit-root reading if the level genuinely has no anchor (prices, "
                    "cumulative flows); choose the break reading if the change coincides with a "
                    "known event.",
        risk_if_ignored="Committing to one reading makes the forecast interval wrong in one "
                        "direction — too wide forever, or too narrow after the shift.",
        how_to="Step 11 → compare the differenced model against the break-dummy model.",
    ),
    StatRule(
        id="stat.contradiction.report",
        stage="S5",
        kind="note",
        recommend="Read the dissenting test before choosing",
        when=lambda c: c.n_contradictions > 0,
        because=lambda c: (
            f"{c.n_contradictions} test(s) in this battery disagree with the headline verdict. "
            "The headline is the majority reading, not a proof — the dissent is listed with its "
            "statistic so you can judge it."
        ),
        confidence=Confidence.MEDIUM,
        provenance=["stat.pp.constant_trend", "stat.dfgls.constant"],
        alternative=None,
        risk_if_ignored="Taking the headline as settled when the evidence is split is exactly how "
                        "a differencing decision gets made for the wrong reason.",
        how_to="Step 5 → open 'Where the tests disagree'.",
    ),
]

#: Rules that hold regardless of the data. Stated so the UI can show them once.
UNCONDITIONAL: list[str] = [
    "A test that could not run is not a test that passed; missing tests are shown as missing.",
    "Rejecting a null is evidence, not proof. Failing to reject is even weaker evidence.",
    "Every verdict here is about the series you have now — after any preprocessing you applied.",
]

#: Declaration order, so equally-ranked rules keep a stable, readable order in the UI.
_FILE_ORDER = {r.id: i for i, r in enumerate(RULES)}


def rank(c: StatContext) -> list[StatRule]:
    """Blockers first, then by confidence, then declaration order."""
    hits = [r for r in RULES if r.applies(c)]
    conf_rank = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}
    hits.sort(key=lambda r: (
        0 if r.kind == "blocker" else (2 if r.kind == "note" else 1),
        conf_rank.get(r.confidence, 3),
        _FILE_ORDER.get(r.id, 99),
    ))
    return hits


__all__ = ["StatContext", "StatRule", "RULES", "UNCONDITIONAL", "rank"]
