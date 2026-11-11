"""
api/tests_stat.py — S5: stationarity, seasonal unit roots, structural breaks (§5.5).

  POST /analysis/tests              run the battery on the current series, store it
  GET  /analysis/tests              read back the stored battery (a reload costs nothing)
  POST /analysis/tests/reconcile    the 2×2 verdict plus the rules that follow from it

Why two steps rather than one: the battery is expensive (a few seconds of simulation) and the
user may want to read the table, then ask for the verdict. Splitting them also means the
verdict is always computed from the battery the screen is showing.

Every result that could not run is absent. Nothing here fabricates a verdict for a test the
data cannot support — the reconciliation says "inconclusive" in that case, in words.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.advisor.rules_stat import StatContext, rank
from app.api.deps import session_dep
from app.core.errors import ChronosError
from app.core.schemas import (
    AnalysisResponse,
    Confidence,
    NarrativeResult,
    RuleOut,
    Stage,
    StatResult,
    TableColumn,
    TableResult,
    AdvisorReport,
    FigureResult,
)
from app.core.session import Session
from app.ts.tests.breaks import BreakTest
from app.ts.tests.reconcile import Reconciliation, reconcile
from app.ts.tests.seasonal_unit import SeasonalUnitResult
from app.ts.tests.unitroot import UnitRootTest, unit_root_battery

router = APIRouter(tags=["analysis"])

BATTERY = "stat_battery"
VERDICT = "stat_verdict"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_series(session: Session) -> pd.Series:
    series: pd.Series | None = session.objects.get("series")
    if session.state.dataset is None or series is None:
        raise ChronosError(
            plain="There is no data loaded in this session yet.",
            fix="Load a file or an example dataset in Step 0 first.",
        )
    return series


def _period(session: Session) -> int | None:
    ds = session.state.dataset
    return ds.frequency.period if ds is not None else None


def _conf(x: str) -> Confidence:
    return {"high": Confidence.HIGH, "moderate": Confidence.MEDIUM,
            "medium": Confidence.MEDIUM, "low": Confidence.LOW}.get(x, Confidence.MEDIUM)


def _row(t: UnitRootTest | SeasonalUnitResult) -> dict[str, Any]:
    """One table row per test. `rejects` is spelled out, because the two nulls are opposite."""
    return {
        "test": t.name,
        "id": t.id,
        "null": t.null,
        "statistic": round(float(t.statistic), 3),
        "p_value": None if t.p_value is None else round(float(t.p_value), 4),
        "cv_5pct": round(float(t.critical.get("5%", float("nan"))), 3)
        if t.critical.get("5%") is not None else None,
        "rejects_null": bool(t.rejects_null),
        "reading": t.says,
    }


def _unit_table(tests: list[UnitRootTest]) -> TableResult | None:
    if not tests:
        return None
    return TableResult(
        id="table.stationarity",
        stage=Stage.S5_TEST_STATION,
        title="Unit-root tests",
        plain=(
            "ADF, PP, DF–GLS and ZA ask 'is there a unit root?' and hope to say no; KPSS asks "
            "'is the series stable?' and hopes to say yes. Reading them together is the point."
        ),
        why=(
            "The choice between modelling the level and modelling the changes is the single "
            "biggest decision in the pipeline, and it is decided here."
        ),
        method=(
            "statsmodels adfuller (AIC lag selection), kpss (Bartlett, "
            "nlags = ceil(4·(n/100)^0.25)), zivot_andrews; Phillips–Perron and DF–GLS "
            "implemented in app/ts/tests/unitroot.py"
        ),
        columns=[
            TableColumn(key="test", label="Test"),
            TableColumn(key="null", label="Its null hypothesis"),
            TableColumn(key="statistic", label="Statistic", dtype="number", align="right"),
            TableColumn(key="cv_5pct", label="5% critical value", dtype="number", align="right"),
            TableColumn(key="p_value", label="p-value", dtype="number", format="p", align="right"),
            TableColumn(key="reading", label="What it says"),
        ],
        rows=[_row(t) for t in tests],
        caption="A test that is missing could not run on this series; it did not pass.",
        footnote="Rejecting the null is evidence, not proof. Opposite nulls mean disagreement "
                 "is informative, not an error.",
        reproducible_snippet=(
            "import pandas as pd\n"
            "from statsmodels.tsa.stattools import adfuller, kpss\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "print(adfuller(s, autolag='AIC', regression='ct')[:2])\n"
            "print(kpss(s, regression='level', nlags='auto')[:2])\n"
        ),
        caveats=[
            "ADF has low power on short series: a high p-value is 'no evidence', not 'a unit root'.",
            "KPSS p-values are clipped to the 0.01–0.10 table range and printed as such.",
        ],
    )


def _seasonal_table(tests: list[SeasonalUnitResult]) -> TableResult | None:
    if not tests:
        return None
    return TableResult(
        id="table.seasonal_unit",
        stage=Stage.S5_TEST_STATION,
        title="Seasonal unit-root tests",
        plain=(
            "These ask a narrower question: does the seasonal pattern repeat at a fixed shape, "
            "or does each year inherit last year's deviation?"
        ),
        why="If the pattern drifts, a plain difference does not fix it and the seasonal lag must "
            "be differenced instead.",
        method="OSCB regression and Canova–Hansen season-specific trend test, "
               "app/ts/tests/seasonal_unit.py; critical values simulated at this n and period.",
        columns=[
            TableColumn(key="test", label="Test"),
            TableColumn(key="statistic", label="Statistic", dtype="number", align="right"),
            TableColumn(key="cv_5pct", label="5% critical value", dtype="number", align="right"),
            TableColumn(key="reading", label="What it says"),
        ],
        rows=[_row(t) for t in tests],
        caption="Canova–Hansen is read in the opposite direction: large values favour stability.",
        reproducible_snippet=(
            "import numpy as np, pandas as pd\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "ds = s.diff(periods=12).dropna()\n"
            "print(len(ds), ds.std())\n"
        ),
        caveats=["Seasonal unit-root tests have low power on short series."],
    )


def _break_table(tests: list[BreakTest]) -> TableResult | None:
    if not tests:
        return None
    rows = []
    for t in tests:
        # Bai–Perron reports no critical value when the BIC chooses zero breaks. A 0.000 in that
        # column reads as a line the statistic sits exactly on, so an absent value is shown as
        # absent rather than as a number.
        cv = t.critical.get("5%")
        rows.append({
            "test": t.name,
            "statistic": round(float(t.statistic), 3),
            "cv_5pct": (round(float(cv), 3)
                        if cv is not None and np.isfinite(float(cv)) and float(cv) > 0
                        else None),
            "breaks": ", ".join(b["time"][:10] for b in t.breaks) or "none found",
            "reading": t.says,
        })
    return TableResult(
        id="table.breaks",
        stage=Stage.S5_TEST_STATION,
        title="Structural-break tests",
        plain="A break is a change in the rules, not noise. These tests look for one and say where.",
        why="A level shift makes ADF look like a unit root. Finding the date turns a "
            "misleading verdict into a dummy variable.",
        method="Chow (pre-specified dates only), Quandt–Andrews sup-Wald and Bai–Perron "
               "(simulated critical values), CUSUM of recursive residuals.",
        columns=[
            TableColumn(key="test", label="Test"),
            TableColumn(key="statistic", label="Statistic", dtype="number", align="right"),
            TableColumn(key="cv_5pct", label="5% critical value", dtype="number", align="right"),
            TableColumn(key="breaks", label="Break date(s)"),
            TableColumn(key="reading", label="What it says"),
        ],
        rows=rows,
        caption="Break dates are offered to the model stage as step dummies; they are never "
                "deleted from the data.",
        reproducible_snippet=(
            "import numpy as np, pandas as pd\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "r = s - s.rolling(12).mean()\n"
            "print(r.abs().nlargest(3))\n"
        ),
        caveats=["Quandt–Andrews and Bai–Perron critical values are simulated here "
                 "(400 replications, seed 20260101) and are reproducible."],
    )


def _verdict_result(rec: Reconciliation) -> NarrativeResult:
    return NarrativeResult(
        id="narrative.stationarity_verdict",
        stage=Stage.S5_TEST_STATION,
        title=rec.headline,
        plain=rec.plain,
        why=rec.action,
        method="app/ts/tests/reconcile.py — ADF × KPSS 2×2, overturned only by evidence that "
               "survives a break (Zivot–Andrews) or a seasonal unit root (OCSB).",
        paragraphs=[rec.plain, rec.action] + (
            ["Where the tests disagree:"] + [f"• {c}" for c in rec.contradictions]
            if rec.contradictions else []
        ),
        numbers_cited={
            "n_breaks": len(rec.break_tests and [1] or []),
        },
        caveats=rec.caveats,
        reproducible_snippet=(
            "import pandas as pd\n"
            "from app.ts.tests.reconcile import reconcile\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "print(reconcile(s).verdict)\n"
        ),
    )


def _context(rec: Reconciliation, session: Session) -> StatContext:
    def find(tests, test_id):
        return next((t for t in tests if t.id == test_id), None)

    adf = find(rec.unit_root_tests, "stat.adf.constant_trend")
    adf_c = find(rec.unit_root_tests, "stat.adf.constant")
    kpss = find(rec.unit_root_tests, "stat.kpss.level")
    dfgls = find(rec.unit_root_tests, "stat.dfgls.constant")
    ocsb = find(rec.seasonal_tests, "stat.ocsb")
    qa = next((t for t in rec.break_tests if t.id.startswith("stat.quandt_andrews")), None)
    dates = [b["time"][:10] for t in rec.break_tests for b in t.breaks
             if b["statistic"] > b.get("critical_5pct", float("inf"))]
    ds = session.state.dataset
    return StatContext(
        series_name=ds.name if ds else "your series",
        n=len(session.objects.get("series")),
        period=_period(session),
        frequency_label=ds.frequency.label if ds else "observation",
        verdict=rec.verdict,
        confidence=rec.confidence,
        differencing_needed=rec.differencing_needed,
        adf_stat=(adf or adf_c).statistic if (adf or adf_c) else None,
        adf_p=(adf or adf_c).p_value if (adf or adf_c) else None,
        adf_rejects=(adf or adf_c).rejects_null if (adf or adf_c) else None,
        kpss_stat=kpss.statistic if kpss else None,
        kpss_p=kpss.p_value if kpss else None,
        kpss_rejects=kpss.rejects_null if kpss else None,
        dfgls_stat=dfgls.statistic if dfgls else None,
        ocsb_stat=ocsb.statistic if ocsb else None,
        ocsb_cv5=ocsb.critical.get("5%") if ocsb else None,
        ocsb_rejects=ocsb.rejects_null if ocsb else None,
        break_dates=dates,
        n_breaks=len(dates),
        qa_stat=qa.statistic if qa else None,
        qa_rejects=qa.rejects_null if qa else None,
        n_contradictions=len(rec.contradictions),
    )


# --------------------------------------------------------------------------- #
# POST /analysis/tests
# --------------------------------------------------------------------------- #
class TestsRequest(BaseModel):
    audience: str = "beginner"
    run_breaks: bool = True


class TestsResponse(AnalysisResponse):
    verdict: str | None = None
    headline_verdict: str | None = None


@router.post("/analysis/tests", response_model=AnalysisResponse)
def run_tests(
    body: TestsRequest | None = None,
    session: Session = Depends(session_dep),
) -> AnalysisResponse:
    """
    Run the S5 battery. Idempotent: re-running replaces the stored results.

    Reaching this stage needs S1 (the profile) because the battery's caveats refer to the EDA
    facts — seasonality in particular. It does not need S2 or S3: a user is allowed to ask
    "is this stationary?" before deciding anything about preprocessing.
    """
    t0 = time.perf_counter()
    series = _require_series(session)
    session.require_stage(Stage.S5_TEST_STATION, allow_stale=True)

    period = _period(session)
    rec = reconcile(
        series,
        period=period,
        run_breaks=(body.run_breaks if body else True),
    )

    session.results[Stage.S5_TEST_STATION] = {}
    for result in (
        _unit_table(rec.unit_root_tests),
        _seasonal_table(rec.seasonal_tests),
        _break_table(rec.break_tests),
        _verdict_result(rec),
    ):
        if result is not None:
            session.add_result(result)

    session.objects[BATTERY] = {
        "unit_root": [_row(t) for t in rec.unit_root_tests],
        "seasonal": [_row(t) for t in rec.seasonal_tests],
        "breaks": [
            {
                "test": t.name, "id": t.id, "statistic": t.statistic,
                "cv_5pct": t.critical.get("5%"), "says": t.says,
                "breaks": t.breaks, "caveats": t.caveats,
            }
            for t in rec.break_tests
        ],
    }
    session.objects[VERDICT] = rec.as_dict()

    session.mark_stage(
        Stage.S5_TEST_STATION, "complete",
        f"{rec.verdict} ({rec.confidence}) on {len(series)} observations",
    )
    for cav in rec.caveats:
        if cav not in session.state.caveats:
            session.state.caveats.append(cav)

    return AnalysisResponse(
        stage=Stage.S5_TEST_STATION,
        advisor=None,          # /analysis/tests/reconcile produces the recommendations
        results=list(session.results_for(Stage.S5_TEST_STATION)),
        pipeline=session.state.progress(),
        caveats=rec.caveats,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


# --------------------------------------------------------------------------- #
# POST /analysis/tests/reconcile
# --------------------------------------------------------------------------- #
@router.post("/analysis/tests/reconcile")
def run_reconcile(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """Turn the stored battery into the 2×2 verdict and the rules that follow from it."""
    t0 = time.perf_counter()
    stored = session.objects.get(VERDICT)
    if not stored:
        raise ChronosError(
            plain="The stationarity tests have not been run on this series yet.",
            fix="Press 'Run the tests' in Step 4 — it takes a few seconds.",
            code="tests.not_run",
        )

    series = _require_series(session)
    rec = reconcile(series, period=_period(session))
    ctx = _context(rec, session)
    hits = rank(ctx)

    recs = [
        RuleOut(
            id=r.id, stage=Stage.S5_TEST_STATION, recommend=r.recommend,
            confidence=r.confidence, because=r.because(ctx), provenance=r.provenance,
            alternative=r.alternative, risk_if_ignored=r.risk_if_ignored, how_to=r.how_to,
        )
        for r in hits if r.kind != "blocker"
    ]
    blockers = [
        RuleOut(
            id=r.id, stage=Stage.S5_TEST_STATION, recommend=r.recommend,
            confidence=r.confidence, because=r.because(ctx), provenance=r.provenance,
            alternative=r.alternative, risk_if_ignored=r.risk_if_ignored, how_to=r.how_to,
        )
        for r in hits if r.kind == "blocker"
    ]

    report = AdvisorReport(
        stage=Stage.S5_TEST_STATION,
        headline=rec.headline,
        recommendations=recs,
        blockers=blockers,
        facts_used={
            "adf_statistic": round(ctx.adf_stat, 4) if ctx.adf_stat is not None else 0.0,
            "kpss_statistic": round(ctx.kpss_stat, 4) if ctx.kpss_stat is not None else 0.0,
            "n_breaks": ctx.n_breaks,
            "differencing_needed": ctx.differencing_needed,
            "n": ctx.n,
        },
        confidence=_conf(rec.confidence),
        next_stage_ready=not blockers,
        narrative=rec.plain + " " + rec.action,
    )
    session.advice[Stage.S5_TEST_STATION] = report

    return {
        "stage": Stage.S5_TEST_STATION.value,
        "status": "complete",
        "verdict": rec.verdict,
        "headline": rec.headline,
        "plain": rec.plain,
        "action": rec.action,
        "confidence": rec.confidence,
        "differencing_needed": rec.differencing_needed,
        "facts_used": rec.facts_used,
        "contradictions": rec.contradictions,
        "caveats": rec.caveats,
        "blockers": [b.model_dump(mode="json") for b in blockers],
        "recommendations": [r.model_dump(mode="json") for r in recs],
        "duration_ms": int((time.perf_counter() - t0) * 1000),
    }


# --------------------------------------------------------------------------- #
# GET /analysis/tests
# --------------------------------------------------------------------------- #
@router.get("/analysis/tests")
def read_tests(session: Session = Depends(session_dep)) -> dict[str, Any]:
    battery = session.objects.get(BATTERY)
    if not battery:
        return {
            "stage": Stage.S5_TEST_STATION.value,
            "status": "not_started",
            "unit_root": [],
            "seasonal": [],
            "breaks": [],
            "verdict": None,
            "results": [],
        }
    return {
        "stage": Stage.S5_TEST_STATION.value,
        "status": session.state.stages[Stage.S5_TEST_STATION.value].status,
        "unit_root": battery.get("unit_root", []),
        "seasonal": battery.get("seasonal", []),
        "breaks": battery.get("breaks", []),
        "verdict": session.objects.get(VERDICT),
        "results": [r.model_dump(mode="json")
                    for r in session.results_for(Stage.S5_TEST_STATION)],
        "recommendations": [
            r.model_dump(mode="json")
            for r in (session.advice.get(Stage.S5_TEST_STATION).recommendations
                      if session.advice.get(Stage.S5_TEST_STATION) else [])
        ],
        "blockers": [
            r.model_dump(mode="json")
            for r in (session.advice.get(Stage.S5_TEST_STATION).blockers
                      if session.advice.get(Stage.S5_TEST_STATION) else [])
        ],
    }


__all__ = ["router"]
