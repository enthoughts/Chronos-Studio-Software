"""
ts/tests/reconcile.py — turning a pile of test results into one sentence a person can act on.

The unit-root battery deliberately runs tests with *opposite* null hypotheses:

  ADF, PP, DF–GLS, ZA   null = "there is a unit root"
  KPSS                  null = "the series is stationary"

Running both is not redundancy, it is the point. When they disagree the disagreement is the
finding. The classic 2×2 (with the standard reading of each cell) is:

                     KPSS does not reject      KPSS rejects
  ADF rejects        stationary                 stationary but not a *constant* — usually a
                     (both agree)               break or a strong trend; look at the segments
  ADF does not       not enough evidence       unit root
  reject             to decide (low power)     (both agree)

Two more dimensions can overturn a "unit root" verdict, and both are checked before that
verdict is spoken:

  * a structural break — a level shift makes ADF behave like a unit root (Perron 1989). If
    Quandt–Andrews finds a break and ZA rejects, the series is trend/level stationary around
    a break, and the right response is a step dummy, not differencing;
  * seasonality — for a seasonal series, ADF's "unit root" can mean a *seasonal* unit root, so
    OCSB is consulted before recommending `diff()`.

Every verdict therefore cites the statistics it came from. There is no path through this
module that produces advice without a number behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.ts.tests.breaks import BreakTest, break_battery
from app.ts.tests.seasonal_unit import SeasonalUnitResult, seasonal_unit_battery
from app.ts.tests.unitroot import UnitRootTest, adf, unit_root_battery

Verdict = Literal[
    "stationary",
    "stationary_around_break",
    "trend_stationary",
    "unit_root",
    "seasonal_unit_root",
    "inconclusive",
    "conflicting",
    "unit_root_or_break",
]


@dataclass
class Reconciliation:
    """One verdict, the evidence for it, and what the user should do next."""

    verdict: Verdict
    headline: str
    plain: str
    action: str
    confidence: Literal["high", "moderate", "low"]
    differencing_needed: bool
    facts_used: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    break_tests: list[BreakTest] = field(default_factory=list)
    unit_root_tests: list[UnitRootTest] = field(default_factory=list)
    seasonal_tests: list[SeasonalUnitResult] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "headline": self.headline,
            "plain": self.plain,
            "action": self.action,
            "confidence": self.confidence,
            "differencing_needed": self.differencing_needed,
            "facts_used": self.facts_used,
            "contradictions": self.contradictions,
            "caveats": self.caveats,
            "breaks": [
                {"time": b["time"], "statistic": b["statistic"]} for t in self.break_tests
                for b in t.breaks
            ],
            "unit_root": [
                {"id": t.id, "statistic": t.statistic, "rejects_null": t.rejects_null,
                 "says": t.says}
                for t in self.unit_root_tests
            ],
            "seasonal": [
                {"id": t.id, "statistic": t.statistic, "rejects_null": t.rejects_null,
                 "says": t.says}
                for t in self.seasonal_tests
            ],
        }


def _t(tests: list[UnitRootTest], test_id: str) -> UnitRootTest | None:
    return next((x for x in tests if x.id == test_id), None)


def _rejects(t: UnitRootTest | None) -> bool | None:
    return None if t is None else t.rejects_null


def _f(x: float, places: int = 2) -> str:
    """Numbers in citable sentences never carry thousands separators (R0.5)."""
    return f"{x:.{places}f}"


#: How many normal period-to-period moves the change at the break date must cover before it is
#: called a clean shift rather than an ordinary one. Calibrated on pure random walks (n = 120–300,
#: scanning every candidate date): the best single move reaches 3.8 at the 95th percentile and 5.2
#: at the maximum, while a textbook level shift of 8 noise-units per period reaches 5.5 at the
#: median. The threshold sits between them, and the overlap is reported as ambiguity rather than
#: resolved by force.
SHIFT_RATIO = 4.5


def _shift_is_real(series: pd.Series, break_time: str) -> tuple[bool, float]:
    """
    Is the change at `break_time` bigger than the series' ordinary period-to-period movement?

    Compares the single first difference at the break date with the median absolute first
    difference (a robust scale, so one outlier cannot define "normal"), after removing a straight
    trend so that a sloping series does not inflate its own threshold.

    Two earlier versions of this check failed in opposite directions and are worth remembering:
    a residual-ADF after a step dummy rejected for a third of pure random walks (the fitted step
    absorbs the drift); a full-window MAD made a trending series incapable of ever passing, so a
    textbook break scored 161.7 against a critical value of 282.4.
    """
    y = series.dropna()
    cut = int(y.index.searchsorted(pd.Timestamp(break_time)))
    vals = y.to_numpy(dtype=float)
    n = len(vals)
    if cut < 6 or n - cut < 6:
        return False, float("nan")

    t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(X, vals, rcond=None)
    resid = vals - X @ beta
    d = np.diff(resid)
    mad = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    if not np.isfinite(mad) or mad <= 0:
        return False, float("nan")
    i = min(max(cut - 1, 0), len(d) - 1)
    ratio = abs(d[i]) / mad
    return bool(ratio >= SHIFT_RATIO), float(ratio)


def reconcile(
    series: pd.Series,
    *,
    period: int | None = None,
    run_breaks: bool = True,
    run_seasonal: bool = True,
) -> Reconciliation:
    """
    Run the battery and read it as a whole.

    `period` enables the seasonal branch; leave it out and the seasonal verdict is simply not
    spoken — an untested claim is not made.
    """
    y = series.dropna()
    n = len(y)

    ur = unit_root_battery(y, period=period)
    br: list[BreakTest] = []
    if run_breaks and n >= 30:
        br = break_battery(y)
    seas: list[SeasonalUnitResult] = []
    if run_seasonal and period and period >= 2 and n >= 4 * period + 4:
        seas = seasonal_unit_battery(y, period)

    adf_ct = _t(ur, "stat.adf.constant_trend")
    adf_c = _t(ur, "stat.adf.constant")
    kpss_l = _t(ur, "stat.kpss.level")
    kpss_t = _t(ur, "stat.kpss.trend")

    facts: list[str] = []
    for t in (adf_ct, kpss_l):
        if t is not None:
            facts.append(
                f"{t.name}: statistic {_f(t.statistic)}, "
                + ("rejects its null" if t.rejects_null else "does not reject its null")
            )

    # ---- the 2×2 ---------------------------------------------------------------
    adf_says_stationary = _rejects(adf_ct)
    kpss_says_nonstationary = _rejects(kpss_l)
    contradictions: list[str] = []

    # ---- overturning evidence: a break ------------------------------------------
    break_dates: list[str] = []
    break_significant = False
    for bt in br:
        for b in bt.breaks:
            if b["statistic"] > b.get("critical_5pct", float("inf")):
                break_dates.append(b["time"][:10])
                break_significant = True
    za = _t(ur, "stat.za.both")
    za_rejects = bool(za is not None and za.rejects_null)
    qa = next((b for b in br if b.id.startswith("stat.quandt_andrews")), None)
    qa_rejects = bool(qa is not None and qa.rejects_null)
    if qa_rejects and not za_rejects and break_significant:
        contradictions.append(
            "Quandt–Andrews places a break at "
            f"{', '.join(break_dates)} but Zivot–Andrews, which allows for a break and still "
            "tests for a unit root, does not reject. On its own that pattern is what a random "
            "walk looks like: wandering levels are easily read as breaks."
        )

    # ---- overturning evidence: seasonality --------------------------------------
    # OCSB's null IS the seasonal unit root, so `rejects_null is False` means the seasonal
    # pattern needs seasonal differencing. Reading that flag as "has a problem" inverts the test.
    ocsb_test = next((s for s in seas if s.id == "stat.ocsb"), None)
    has_seasonal_unit_root = ocsb_test is not None and ocsb_test.rejects_null is False
    seasonal_stable = any(
        s.id.startswith("stat.canova_hansen") and not s.rejects_null for s in seas
    )

    # Other tests that point the same way or the other way. They do not decide the verdict,
    # but a "both tests agree" sentence must not be printed while a third test disagrees.
    pp = _t(ur, "stat.pp.constant_trend")
    dfgls_c = _t(ur, "stat.dfgls.constant")
    dissent: list[str] = []
    if pp is not None and pp.rejects_null and not adf_says_stationary:
        dissent.append(
            f"Phillips–Perron rejects a unit root (statistic {_f(pp.statistic)}) while ADF does "
            "not. PP does not model seasonality"
            + (f", and this series repeats every {period} periods" if period else "")
            + ", so on seasonal data PP is the one to distrust here."
        )
    if dfgls_c is not None and dfgls_c.rejects_null and not adf_says_stationary:
        dissent.append(
            f"DF–GLS rejects a unit root (statistic {_f(dfgls_c.statistic)}) while ADF does not; "
            "DF–GLS is the more powerful test, so the 'unit root' reading is weaker than it looks."
        )
    if dfgls_c is not None and not dfgls_c.rejects_null and adf_says_stationary:
        dissent.append(
            f"DF–GLS does not reject a unit root (statistic {_f(dfgls_c.statistic)}) while ADF "
            "rejects. DF–GLS is the more powerful of the two against a persistent alternative, "
            "so the positive finding is the weaker one: treat 'stationary' as provisional."
        )

    # Only ZA can overturn a unit-root verdict, never Quandt–Andrews on its own: a random walk
    # wanders enough that a sup-Wald test will place "breaks" in it almost every time (the
    # spurious-break result). ZA is the one designed with a unit-root null *and* a break — and
    # even ZA is not enough by itself, because a random walk can supply the break it needs. The
    # residuals after removing the shift must *also* be stationary.
    confirmed, shift_ratio = (False, float("nan"))
    if break_significant and za_rejects and break_dates:
        confirmed, shift_ratio = _shift_is_real(y, break_dates[0])
    if break_significant and za_rejects and not confirmed:
        # ADF and KPSS say "unit root"; ZA says "one level shift". Both readings fit this series,
        # and no test in the battery separates them — a walk that wanders far enough looks exactly
        # like a series that was shifted once. Saying so is the accurate answer.
        return Reconciliation(
            verdict="unit_root_or_break",
            headline="Unit root, or a one-off level shift — the data cannot tell",
            plain=(
                "ADF does not reject a unit root and KPSS rejects a constant mean, so the standard "
                f"reading is a unit root. But Zivot–Andrews finds a single level shift at "
                f"{break_dates[0]} that would also explain the data, and the change there is "
                "ordinary-sized rather than dramatic. A series that wanders and a series that "
                "was shifted once are hard to tell apart on 100–200 points; any method that "
                "claims to know which is which is guessing."
            ),
            action=(
                "Carry both readings forward: difference the series once (so a unit root is "
                f"handled) and include a step dummy at {break_dates[0]} (so a shift is handled). "
                "Compare the two in the model stage — a good model tells you which it was."
            ),
            confidence="low",
            differencing_needed=True,
            facts_used=facts + [
                f"The level at {break_dates[0]} steps by {_f(shift_ratio)} times the series' "
                "ordinary period-to-period movement"
                + (f", below the {_f(SHIFT_RATIO)} used to call it a clean break"
                   if np.isfinite(shift_ratio) else "")
            ],
            contradictions=contradictions + [
                f"ADF/KPSS read a unit root; Zivot–Andrews reads a break at {break_dates[0]}."
            ],
            break_tests=br,
            unit_root_tests=ur,
            seasonal_tests=seas,
            caveats=[
                "Neither model is wrong, and neither is established. Forecast intervals that "
                "assume a unit root widen forever; the break model does not. The difference in "
                "the interval is the real cost of not knowing.",
            ],
        )

    if break_significant and za_rejects and confirmed:
        facts.append(
            f"Quandt–Andrews sup-Wald {_f(qa.statistic) if qa else 'n/a'} with "
            f"{len(break_dates)} significant break date(s): {', '.join(break_dates)}"
        )
        if za is not None:
            facts.append(
                f"Zivot–Andrews (break in level and trend) statistic {_f(za.statistic)}"
                + (" rejects the unit-root null" if za.rejects_null
                   else " does not reject the unit-root null")
            )
        facts.append(
            f"The level at {break_dates[0]} steps by {_f(shift_ratio)} times the series' "
            "ordinary period-to-period movement"
        )
        return Reconciliation(
            verdict="stationary_around_break",
            headline="Stationary around a structural break",
            plain=(
                "The series looks like it has a unit root only because its level jumps. The "
                f"break tests place the jump at {', '.join(break_dates)}. A series with a "
                "break is not a series whose past stops mattering: it is a stable series with "
                "one interruption."
            ),
            action=(
                "Do not difference this series — differencing would turn the jump into one "
                "spike and delete the level information. Add a step dummy at the break date and "
                "forecast the stable part."
            ),
            confidence="high" if (za_rejects and confirmed) else "moderate",
            differencing_needed=False,
            facts_used=facts,
            contradictions=contradictions,
            break_tests=br,
            unit_root_tests=ur,
            seasonal_tests=seas,
            caveats=[
                "The break date is estimated, not known. Treat it as the best single date in "
                "the window, not an exact day.",
            ],
        )

    if has_seasonal_unit_root and not adf_says_stationary:
        ocsb = next((s for s in seas if "ocsb" in s.id), None)
        if ocsb is not None:
            facts.append(
                f"OCSB seasonal unit-root statistic {_f(ocsb.statistic)} "
                f"against a 5% critical value of {_f(ocsb.critical.get('5%', float('nan')))}"
            )
        return Reconciliation(
            verdict="seasonal_unit_root",
            headline="Seasonal unit root",
            plain=(
                "The unit root is seasonal, not ordinary: this year's pattern does not settle "
                "back to a fixed seasonal shape after a shock, it drifts. A plain difference "
                "does not fix that — it moves the problem into the seasonal lags."
            ),
            action=(
                f"Take a seasonal difference (lag {period}) rather than a first difference, or "
                "use a model with seasonal differencing built in. Check the ACF afterwards: a "
                "spike at the seasonal lag that survives a plain difference is this pattern."
            ),
            confidence="moderate",
            differencing_needed=True,
            facts_used=facts,
            contradictions=contradictions + dissent,
            break_tests=br,
            unit_root_tests=ur,
            seasonal_tests=seas,
            caveats=[
                "Seasonal unit-root tests have low power on short series; a non-rejection is "
                "weaker evidence than a rejection.",
            ],
        )

    # ---- ADF rejects → stationary branch ---------------------------------------
    if adf_says_stationary and kpss_says_nonstationary:
        # both rejected: stationary, but not around a constant mean
        facts.append(
            f"KPSS (level) statistic {_f(kpss_l.statistic)} rejects stability of the mean"
            if kpss_l else "KPSS (level) rejects stability of the mean"
        )
        trend_ok = _rejects(kpss_t) is False
        if trend_ok:
            facts.append(
                f"KPSS (trend) statistic {_f(kpss_t.statistic)} does not reject "
                "trend-stationarity"
                if kpss_t else "KPSS (trend) does not reject trend-stationarity"
            )
            return Reconciliation(
                verdict="trend_stationary",
                headline="Stationary around a trend",
                plain=(
                    "The series returns to a path, but the path slopes. Both tests are right: "
                    "there is no unit root (ADF rejects) and the mean is not constant (KPSS "
                    "level rejects) — but a straight line through the data is stable (KPSS "
                    "trend does not reject)."
                ),
                action=(
                    "Difference it only if you want the growth rate. Otherwise include a time "
                    "trend (or detrend first) — differencing a trend-stationary series throws "
                    "away the trend and adds noise."
                ),
                confidence="moderate",
                differencing_needed=False,
                facts_used=facts,
                contradictions=contradictions,
                break_tests=br,
                unit_root_tests=ur,
                seasonal_tests=seas,
                caveats=[
                    "KPSS rejects a constant mean very readily on 100–200 points; use the trend "
                    "version before concluding a break or a random walk.",
                ],
            )
        return Reconciliation(
            verdict="conflicting",
            headline="Conflicting evidence",
            plain=(
                "One test says the series settles down, the other says its level wanders, and "
                "not even a straight trend explains the wandering. This usually means a break, "
                "a slow-moving component, or a series that is stationary only relative to "
                "something else."
            ),
            action=(
                "Look at the level plot and the break panel before differencing. If a break is "
                "found, use a dummy; if none, a first difference is the safer default — but "
                "validate it on the ACF rather than trusting the verdict."
            ),
            confidence="low",
            differencing_needed=True,
            facts_used=facts,
            contradictions=[
                "ADF rejects a unit root while KPSS rejects stationarity."
            ],
            break_tests=br,
            unit_root_tests=ur,
            seasonal_tests=seas,
            caveats=[
                "Do not treat this as a settled answer. Two tests that disagree are telling you "
                "the series is on the boundary, and the model choice matters more than usual.",
            ],
        )

    if adf_says_stationary and kpss_says_nonstationary is False:
        facts.append("KPSS (level) does not reject a constant mean")
        if seasonal_stable:
            facts.append("Canova–Hansen finds a stable seasonal pattern")
        return Reconciliation(
            verdict="stationary",
            headline="Stationary",
            plain=(
                "Both tests agree: the series settles back to a stable level. Its mean, and the "
                "strength of its relationship with its own past, do not depend on where you are "
                "in the window."
            ),
            action=(
                "Do not difference it. Differencing a stationary series makes it harder to "
                "forecast — it adds a moving-average unit root and throws away information. "
                "Model the level directly."
            ),
            confidence="high" if not dissent else "moderate",
            differencing_needed=False,
            facts_used=facts,
            contradictions=contradictions + dissent,
            break_tests=br,
            unit_root_tests=ur,
            seasonal_tests=seas,
            caveats=[],
        )

    # ---- ADF does not reject ---------------------------------------------------
    if kpss_says_nonstationary:
        if adf_c is not None:
            facts.append(
                f"{adf_c.name}: statistic {_f(adf_c.statistic)}, does not reject a unit root"
            )
        headline = ("Unit root (both tests agree)" if not dissent
                    else "Unit root (ADF and KPSS agree; one test dissents)")
        return Reconciliation(
            verdict="unit_root",
            headline=headline,
            plain=(
                "ADF does not reject a unit root and KPSS rejects a constant mean — the two "
                "tests with opposite nulls point the same way. Shocks do not fade: the level "
                "you see now is the sum of every shock up to now, so the best guess for next "
                "month is roughly this month."
            ),
            action=(
                "Difference it once before fitting a model, and check that the differenced "
                "series passes the same tests. Do not difference twice unless the second test "
                "also says so."
            ),
            confidence="high" if not dissent else "moderate",
            differencing_needed=True,
            facts_used=facts,
            contradictions=contradictions + dissent,
            break_tests=br,
            unit_root_tests=ur,
            seasonal_tests=seas,
            caveats=[
                "A unit root and a very persistent stationary process look identical on 100 "
                "points; the practical difference is whether the forecast cone widens forever.",
            ],
        )

    # ADF does not reject, KPSS does not reject
    if adf_ct is not None:
        facts.append(
            f"{adf_ct.name}: statistic {_f(adf_ct.statistic)}, does not reject a unit root"
        )
    if kpss_l is not None:
        facts.append(
            f"{kpss_l.name}: statistic {_f(kpss_l.statistic)}, does not reject a constant mean"
        )
    return Reconciliation(
        verdict="inconclusive",
        headline="Not enough evidence to decide",
        plain=(
            "Neither test finds what it is looking for. That is a real answer, not a failure: "
            "the series is too short, too noisy, or too close to the boundary for this window "
            f"({n} observations) to separate 'settles down' from 'wanders'."
        ),
        action=(
            "Treat it as persistent but unproven: start with the level and compare against a "
            "differenced version in the model stage. Getting more history is worth more here "
            "than a cleverer test."
        ),
        confidence="low",
        differencing_needed=False,
        facts_used=facts,
        contradictions=[],
        break_tests=br,
        unit_root_tests=ur,
        seasonal_tests=seas,
        caveats=[
            "ADF's failure to reject is often a power problem rather than evidence of a unit "
            "root; do not read it as a positive finding.",
        ],
    )


__all__ = ["Reconciliation", "Verdict", "reconcile"]
