"""
ts/diagnose/verdict.py — `ADEQUATE` / `ADEQUATE WITH CAVEATS` / `INADEQUATE — revise` (§5.9).

The verdict is the part of a diagnostics panel that people act on, so it is the part that must not be
a score. Three rules govern it:

**1. A verdict names its reason.** Every `INADEQUATE` carries the test that produced it, the number,
and the remedy. "The model is inadequate" without a named failure is an opinion.

**2. Twelve tests are not twelve pieces of evidence.** At 5% each, a perfect model trips at least one
test 46% of the time. So a *single* rejection, with no mechanism behind it and no corroboration,
produces `ADEQUATE WITH CAVEATS` — not `INADEQUATE`. Two independent rejections pointing at the same
mechanism do produce `INADEQUATE`, because that is no longer a multiple-testing artefact.

**3. Some failures are not about the mean at all.** Non-normal residuals leave the point forecast
untouched and only affect normal-theory intervals; ARCH in the residuals leaves the mean model correct
and asks for a variance model. Both are caveats, not revisions. Saying so is the difference between a
useful verdict and a checklist that always says "revise".

The remedies are specific on purpose: "add an MA term" is actionable, "consider a better model" is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
from app.core.formatting import p_phrase

from app.ts.diagnose.battery import DiagnosticBattery


class Adequacy(str, Enum):
    ADEQUATE = "adequate"
    ADEQUATE_WITH_CAVEATS = "adequate_with_caveats"
    INADEQUATE = "inadequate"


#: A finding that is evidence about the mean's structure.
STRUCTURAL = "structural"
#: A finding about the distribution or the variance — real, but not a reason to change the mean model.
DISTRIBUTIONAL = "distributional"


@dataclass
class Finding:
    """One failed check: what failed, the number, what it means, what to do."""

    test: str
    kind: str
    severity: str
    """`high` = the mean model is misspecified; `medium` = worth a look; `low` = informational."""
    detail: str
    remedy: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "test": self.test,
            "kind": self.kind,
            "severity": self.severity,
            "detail": self.detail,
            "remedy": self.remedy,
        }


@dataclass
class Verdict:
    """The three-state verdict plus every finding that produced it."""

    adequacy: Adequacy
    headline: str
    reasons: list[Finding] = field(default_factory=list)
    remedies: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    n_tests_run: int = 0
    n_rejected: int = 0
    expected_false_positives_at_5pct: float = 0.0
    """The multiple-testing floor, printed so a reader can see when the rejections are within it."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "adequacy": self.adequacy.value,
            "headline": self.headline,
            "reasons": [r.as_dict() for r in self.reasons],
            "remedies": self.remedies,
            "notes": self.notes,
            "n_tests_run": self.n_tests_run,
            "n_rejected": self.n_rejected,
            "expected_false_positives": round(self.expected_false_positives_at_5pct, 2),
        }


def _fmt(p: float | None) -> str:
    """Deprecated local spelling; kept as an alias so existing callers keep working."""
    return p_phrase(p)


def judge(
    battery: DiagnosticBattery,
    *,
    period: int | None = None,
    is_volatility_model: bool = False,
    is_count_model: bool = False,
    nominal_coverage: float = 0.9,
    observed_coverage: float | None = None,
) -> Verdict:
    """
    Turn the battery into a verdict.

    `is_volatility_model` and `is_count_model` change the reading, not the arithmetic: a GARCH row is
    *expected* to leave ARCH in its standardised residuals' squares only if it is misspecified, while a
    mean model is not held to that at all.
    """
    findings: list[Finding] = []
    notes: list[str] = []
    n_tests = 0
    n_rejected = 0

    if battery.n_resid < 12:
        return Verdict(
            adequacy=Adequacy.ADEQUATE_WITH_CAVEATS,
            headline=(
                f"No verdict: only {battery.n_resid} residuals. There is not enough here to test "
                "anything, and calling it adequate would be a claim the data cannot support."
            ),
            notes=list(battery.notes),
        )

    # -- autocorrelation ------------------------------------------------ #
    lb = battery.ljung_box or {}
    results = lb.get("results") or {}
    rejects = list(lb.get("rejects") or [])
    for lag, res in results.items():
        n_tests += 1
        p = res.get("p")
        if p is not None and p < 0.05:
            n_rejected += 1
    if rejects:
        worst = min(rejects, key=lambda L: results[str(L)].get("p") or 1.0)
        p = results[str(worst)].get("p")
        seasonal = bool(period and worst in (period, 2 * period))
        findings.append(
            Finding(
                test=f"Ljung–Box (lag {worst})",
                kind=STRUCTURAL,
                severity="high",
                detail=(
                    f"{_fmt(p)} at lag {worst}"
                    + (
                        f", the seasonal lag for a period of {period}. Autocorrelation remains where "
                        "the seasonal term should have absorbed it."
                        if seasonal
                        else ". The residuals still contain a linear pattern the model did not capture."
                    )
                ),
                remedy=(
                    f"Add a seasonal term at lag {period} (SAR/MA seasonal order, or a seasonal "
                    "difference) — the failure is seasonal."
                    if seasonal
                    else "Increase the AR or MA order by one and refit; if that does not clear it, the "
                    "pattern is nonlinear and a regime or threshold model is the next step."
                ),
            )
        )
    bp = battery.box_pierce or {}
    bp_p = bp.get("p")
    if bp_p is not None:
        n_tests += 1
        if bp_p < 0.05:
            n_rejected += 1
        elif rejects and bp_p >= 0.05:
            notes.append(
                f"Box–Pierce does not reject ({_fmt(bp_p)}) while Ljung–Box does. That is the "
                "expected signature of small but real autocorrelation: Ljung–Box is the more powerful "
                "test and the one to act on, but the size of the effect is modest."
            )

    # -- heteroskedasticity --------------------------------------------- #
    arch = battery.arch_lm or {}
    arch_p = arch.get("p")
    if arch_p is not None:
        n_tests += 1
        if arch_p < 0.05:
            n_rejected += 1
            if is_volatility_model:
                findings.append(
                    Finding(
                        test="ARCH-LM on standardised residuals",
                        kind=STRUCTURAL,
                        severity="high",
                        detail=(
                            f"{_fmt(arch_p)}. A variance model is fitted and ARCH is *still* in "
                            "its standardised residuals, which means the variance equation is "
                            "misspecified rather than merely absent."
                        ),
                        remedy="Add a GARCH order, or switch to an asymmetric form (GJR/EGARCH) if the "
                        "leverage sign points that way.",
                    )
                )
            else:
                findings.append(
                    Finding(
                        test="ARCH-LM on residuals",
                        kind=DISTRIBUTIONAL,
                        severity="low",
                        detail=(
                            f"{_fmt(arch_p)}. The variance has its own structure. The point "
                            "forecast is unaffected; intervals that assume a constant spread are not."
                        ),
                        remedy="Fit an ARIMA-GARCH row (Part 11) and use its variance path for the "
                        "intervals. This is not a reason to change the mean model's order.",
                    )
                )

    # -- normality ------------------------------------------------------- #
    jb = (battery.normality or {}).get("jarque_bera") or {}
    jb_p = jb.get("p")
    if jb_p is not None:
        n_tests += 1
        if jb_p < 0.05:
            n_rejected += 1
            kurt = jb.get("kurtosis")
            findings.append(
                Finding(
                    test="Jarque–Bera",
                    kind=DISTRIBUTIONAL,
                    severity="low",
                    detail=(
                        f"{_fmt(jb_p)}; excess kurtosis {kurt:+.2f}."
                        if isinstance(kurt, (int, float))
                        else f"{_fmt(jb_p)}."
                    )
                    + " The residuals are not normal.",
                    remedy=(
                        "Nothing to do for the point forecast. For intervals, use the empirical "
                        "bands the app reports alongside the normal ones, or a t-innovation model — "
                        "non-normal residuals do not invalidate the forecast, only a normal-theory "
                        "interval."
                    ),
                )
            )

    # -- bias and sign patterns ------------------------------------------ #
    signs = battery.signs or {}
    sp = signs.get("p")
    if sp is not None:
        n_tests += 1
        if sp < 0.05:
            n_rejected += 1
            pos, n = signs.get("positive"), signs.get("n")
            findings.append(
                Finding(
                    test="Sign test",
                    kind=STRUCTURAL,
                    severity="high",
                    detail=(
                        f"{_fmt(sp)}: {pos} of {n} residuals are positive. The model is biased in "
                        "one direction — it is systematically under- or over-predicting."
                    ),
                    remedy="Check for an unmodelled level shift or an intervention (promotion, outage, "
                    "policy change) spanning that stretch; a bias this persistent is usually a date.",
                )
            )
    runs = battery.runs or {}
    rp = runs.get("p")
    if rp is not None:
        n_tests += 1
        if rp < 0.05:
            n_rejected += 1
            observed = runs.get("runs", 0)
            expected = runs.get("expected") or 0
            too_few = observed < expected
            findings.append(
                Finding(
                    test="Runs test",
                    kind=STRUCTURAL,
                    severity="medium",
                    detail=(
                        f"{_fmt(rp)}: {observed} runs against {expected} expected. "
                        + (
                            "The residuals stay on one side of zero for too long — the model is "
                            "under-differenced or missing a slow component."
                            if too_few
                            else "The signs alternate too regularly — the model is over-differencing."
                        )
                    ),
                    remedy=(
                        "Differencing is too shallow: consider d+1, or a model with a longer memory."
                        if too_few
                        else "Differencing is too deep: consider d−1. Over-differencing inflates the "
                        "forecast variance for no gain."
                    ),
                )
            )

    # -- CUSUM ------------------------------------------------------------ #
    cusum = battery.cusum or {}
    if cusum.get("breach"):
        n_tests += 1
        # A CUSUM boundary crossing with no *other* sign of drift is a caveat, not a structural
        # finding. The path is a cumulative sum, so it wanders; a single crossing near the end of the
        # series is what a random walk does. It is promoted to `structural` only when the runs test or
        # the sign test also says the residual mean moved — that corroboration is the whole point of
        # the verdict layer, and without it this check alone made clean white noise "with caveats".
        drift_corroborated = bool(
            (battery.signs or {}).get("p") is not None and (battery.signs or {})["p"] < 0.05
        ) or bool((battery.runs or {}).get("p") is not None and (battery.runs or {})["p"] < 0.05)
        where = cusum.get("breach_where") or {}
        since = where.get("since")
        if not drift_corroborated:
            notes.append(
                "The CUSUM path crosses its 5% boundary"
                + (f" at about {since:.0%} of the series" if isinstance(since, float) else "")
                + ", but no other check finds a drift in the residual mean. A cumulative sum wanders, "
                "and a single crossing is what that looks like — it is recorded here rather than "
                "counted as a failure."
            )
        if drift_corroborated:
            findings.append(
                Finding(
                    test="CUSUM of residuals",
                    kind=STRUCTURAL,
                    severity="medium",
                    detail=(
                        "The CUSUM path crosses the 5% boundary"
                        + (
                            f" from about {since:.0%} of the way through the series onwards — the "
                            "residual mean drifts after that point."
                            if isinstance(since, float)
                            else "."
                        )
                    ),
                    remedy="Look for a structural break at that date: a level shift needs an "
                    "intervention dummy, a trend change needs a different trend term.",
                )
            )

    # -- coverage --------------------------------------------------------- #
    if observed_coverage is not None:
        n_tests += 1
        # Binomial tolerance rather than a fixed 5 points: with 40 intervals, 85% is inside noise.
        n_int = max(int(round(40)), 20)
        tol = 1.96 * math.sqrt(nominal_coverage * (1 - nominal_coverage) / n_int)
        if abs(observed_coverage - nominal_coverage) > tol:
            n_rejected += 1
            findings.append(
                Finding(
                    test="Interval coverage",
                    kind=DISTRIBUTIONAL,
                    severity="medium",
                    detail=(
                        f"Coverage is {observed_coverage:.1%} against a nominal "
                        f"{nominal_coverage:.0%}, outside ±{tol:.1%} of sampling noise for this many "
                        "intervals."
                    ),
                    remedy=(
                        "The intervals are too narrow — the predictive distribution understates the "
                        "spread. Widen them or fit a model with time-varying variance."
                        if observed_coverage < nominal_coverage
                        else "The intervals are too wide — they cover more than they claim and carry "
                        "less information than their width suggests."
                    ),
                )
            )

    # -- outliers ---------------------------------------------------------- #
    outliers = battery.outliers or []
    influential = [o for o in outliers if (o.get("influence") or 0) >= 0.1]
    if influential:
        first = influential[0]
        findings.append(
            Finding(
                test="Residual outliers",
                kind=DISTRIBUTIONAL,
                severity="medium",
                detail=(
                    f"{len(influential)} point(s) carry 10% or more of the residual sum of squares; "
                    f"the largest is {first.get('when') or f'position {first.get('position')}'} at "
                    f"{first.get('influence', 0):.0%}."
                ),
                remedy="Check that date in the source data. An intervention dummy for a real event is "
                "legitimate; removing a point because it is inconvenient is not.",
            )
        )

    # -- the verdict -------------------------------------------------------- #
    expected = n_tests * 0.05
    structural = [f for f in findings if f.kind == STRUCTURAL and f.severity in {"high", "medium"}]
    # Independent mechanisms: a Ljung–Box rejection and a runs rejection are the same story told twice,
    # so they do not corroborate each other. Distinct *test families* do.
    families = {f.test.split(" (")[0] for f in structural}
    corroborated = len(families) >= 2
    distributional_only = not structural and bool(findings)

    remedies = list(dict.fromkeys(f.remedy for f in findings))
    notes.append(
        f"{n_tests} checks ran and {n_rejected} rejected at the 5% level. With {n_tests} independent "
        f"tests, about {expected:.1f} rejection(s) are expected from chance alone."
    )

    if not findings:
        verdict = Adequacy.ADEQUATE
        headline = (
            f"No diagnostic failed across {n_tests} checks: the residuals show no remaining "
            "autocorrelation, no bias, no drift and no variance structure worth modelling."
        )
    elif distributional_only:
        verdict = Adequacy.ADEQUATE_WITH_CAVEATS
        headline = (
            "The mean model's structure holds — the failures are about the distribution, not the "
            "forecast. " + " ".join(f.detail for f in findings[:2])
        )
    elif len(structural) == 1 and n_rejected <= max(1, round(expected)):
        verdict = Adequacy.ADEQUATE_WITH_CAVEATS
        headline = (
            f"One structural check failed ({structural[0].test}) out of {n_tests} run, which is "
            f"within what chance produces at this number of tests ({expected:.1f} expected). "
            "Treat it as something to check, not a verdict."
        )
    elif corroborated:
        verdict = Adequacy.INADEQUATE
        headline = (
            f"INADEQUATE — revise: {len(families)} independent checks fail in the same direction "
            f"({', '.join(sorted(families))}). That is not a multiple-testing artefact; the mean model "
            "is missing structure."
        )
    else:
        verdict = Adequacy.ADEQUATE_WITH_CAVEATS
        headline = (
            "The structural failures do not corroborate each other, so they are reported as caveats "
            "rather than as a revision. Read them, act on any that match something you know about the "
            "series, and do not refit purely because a p-value was low."
        )

    if is_count_model:
        notes.append(
            "This is a count model: the normality checks are not meaningful (residuals are discrete) "
            "and are reported for completeness only. Read the discrete PIT in Step 12 instead."
        )

    return Verdict(
        adequacy=verdict,
        headline=headline,
        reasons=findings,
        remedies=remedies,
        notes=notes,
        n_tests_run=n_tests,
        n_rejected=n_rejected,
        expected_false_positives_at_5pct=expected,
    )


def diagnose_model(
    outcome: Any,
    *,
    period: int | None = None,
    is_count_model: bool = False,
    is_volatility_model: bool = False,
) -> dict[str, Any]:
    """
    Run the battery and the verdict for one stored `FitOutcome`, and return the wire shape.

    Kept here rather than in the API module because it is a *modelling* decision which residuals to
    pass (standardised where the model defines them) and what counts as a count model.
    """
    from app.ts.diagnose.battery import run_battery

    resid = None
    try:
        resid = outcome.wrapper.residuals()
    except Exception:  # pragma: no cover - a wrapper without residuals is handled below
        resid = None
    summary = outcome.summary
    if resid is None or len(resid) < 12:
        n_have = 0 if resid is None else len(resid)
        if summary.family.value == "combination":
            why = (
                f"This row has {n_have} held-out residuals — one per forecast origin — and the battery "
                "needs 12. The combination has no one-step residual to test: it is built from other "
                "models' forecasts, and its evidence is the held-out accuracy in Step 11, not a "
                "residual ACF."
            )
        elif summary.family.value == "ml":
            why = (
                f"This row has {n_have} residuals the battery can read. A tabular model forecasts "
                "recursively rather than by a one-step prediction, so the standard residual battery "
                "does not apply — its measured out-of-sample accuracy and interval coverage are the "
                "evidence instead."
            )
        else:
            why = (
                f"This fit produced {n_have} usable residuals, and the battery needs at least 12. "
                "There is nothing to test."
            )
        return {
            "model_id": summary.model_id,
            "display_name": summary.display_name,
            "status": "unavailable",
            "message": why,
            "battery": None,
            "verdict": None,
        }

    r = np.asarray(resid, dtype=float)
    index = None
    try:
        y = getattr(outcome.wrapper, "_y", None)
        if y is not None:
            index = y.index[-r.size :]
    except Exception:  # pragma: no cover
        index = None

    battery = run_battery(
        r,
        model_id=summary.model_id,
        display_name=summary.display_name,
        period=period,
        index=index,
        residual_sd=getattr(outcome.wrapper.diagnostics, "residual_sd", None),
    )
    verdict = judge(
        battery,
        period=period,
        is_volatility_model=is_volatility_model,
        is_count_model=is_count_model,
    )
    return {
        "model_id": summary.model_id,
        "display_name": summary.display_name,
        "status": "ok",
        "battery": battery.as_dict(),
        "verdict": verdict.as_dict(),
    }


__all__ = [
    "STRUCTURAL",
    "Adequacy",
    "Finding",
    "Verdict",
    "diagnose_model",
    "judge",
]
