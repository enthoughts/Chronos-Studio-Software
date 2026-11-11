"""
api/gate.py — S8: the model-readiness gate (§5.7).

  POST /analysis/gate          run every readiness check and store the verdict
  GET  /analysis/gate          read the stored verdict
  POST /analysis/gate/horizon  set the intended forecast horizon, then re-assess

Design notes:

* **The gate reads, it does not recompute.** Every number here comes from a stage that already ran
  (S1 profile, S2 advice, S4 decomposition, S5 tests, S6 correlogram). If a stage has not run, the
  corresponding check is `unknown` — never silently skipped, never assumed to pass. That is the
  whole reason `GateFacts` is a flat dataclass: it makes "where did this number come from" answerable
  by reading one function.

* **Staleness is visible.** The verdict records the series version it was computed for. A verdict
  that describes a different version of the series than the one about to be modelled is worse than
  no verdict, so `read` reports `stale: true` and refuses to call it current.

* **The horizon can be set here.** §5.11 warns when H > n/5, but by then the user has committed to
  a forecast. Setting the intended horizon at the gate moves that warning to where it can still
  change the plan.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import session_dep
from app.core.errors import ChronosError
from app.core.schemas import (
    AdvisorReport,
    AnalysisResponse,
    ChecklistItem,
    Confidence,
    NarrativeResult,
    RuleOut,
    Stage,
    TableColumn,
    TableResult,
    Verdict,
    VerdictResult,
)
from app.core.session import Session
from app.ts.gate.readiness import Gate, GateFacts, assess

router = APIRouter(tags=["analysis"])

GATE = "gate_verdict"

VERDICT_TO_ENUM: dict[str, Verdict] = {
    "ready": Verdict.READY,
    "ready_with_caveats": Verdict.READY_WITH_CAVEATS,
    "not_ready": Verdict.NOT_READY,
    "inconclusive": Verdict.INCONCLUSIVE,
}

STAGE_OF_CHECK: dict[str, Stage] = {
    "length": Stage.S2_ADVISE_PREP,
    "horizon": Stage.S13_FORECAST,
    "gaps": Stage.S2_ADVISE_PREP,
    "stationarity": Stage.S5_TEST_STATION,
    "breaks": Stage.S5_TEST_STATION,
    "outliers": Stage.S1_PROFILE,
    "variance": Stage.S2_ADVISE_PREP,
    "seasonality": Stage.S4_DECOMPOSE,
    "signal": Stage.S4_DECOMPOSE,
    "correlogram": Stage.S6_CORRELATE,
}


# --------------------------------------------------------------------------- #
# Assembling the facts
# --------------------------------------------------------------------------- #
def _require_series(session: Session) -> pd.Series:
    series: pd.Series | None = session.objects.get("series")
    if session.state.dataset is None or series is None:
        raise ChronosError(
            plain="There is no data loaded in this session yet.",
            fix="Load a file or an example dataset in Step 0 first.",
            code="gate.no_data",
        )
    return series


def _num(d: dict[str, Any], *keys: str, default: float | None = None) -> float | None:
    """Read the first key that exists and is a finite number; None otherwise."""
    for k in keys:
        v = d.get(k)
        if v is None or isinstance(v, bool):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            return fv
    return default


def _build_facts(session: Session, series: pd.Series) -> GateFacts:
    """
    Pull every number the gate needs out of the stages that produced it.

    Each block is written so that a missing stage leaves `None`/absent — the gate's own `unknown`
    state then handles it. Nothing here supplies a default that would look like evidence.
    """
    ds = session.state.dataset
    period = ds.frequency.period if ds is not None else None
    period = int(period) if period and period >= 2 else None

    eda = session.objects.get("eda_facts") or {}
    prep = session.objects.get("prep_facts") or {}
    decomp = session.objects.get("decomposition") or {}
    battery = session.objects.get("stat_battery") or {}
    verdict = session.objects.get("stat_verdict") or {}
    correlogram = (session.objects.get("correlogram") or {}).get("payload") or {}

    series_now = series.dropna()
    n = int(series_now.notna().sum())

    # ---- S5: how many tests agree, and do the seasonal ones say anything? ----
    #
    # "Agreeing" has to mean *supporting the same verdict*, not "rejecting its null". The two
    # families have opposite nulls — ADF rejecting is evidence FOR stationarity, KPSS rejecting is
    # evidence AGAINST it — so a raw count of rejections would score them backwards and report a
    # contested series as unanimous.
    ur = battery.get("unit_root") or []
    seas = battery.get("seasonal") or []
    v = str(verdict.get("verdict") or "") or None

    # Break dates are read before the agreement count, which credits a break-based verdict with
    # the breaks that support it — reading them afterwards crashed on exactly the
    # series the gate exists to flag.
    break_dates: list[str] = []
    for t in battery.get("breaks") or []:
        cv = t.get("cv_5pct")
        for b in t.get("breaks") or []:
            stat = b.get("statistic")
            if cv and stat is not None and float(stat) > float(cv):
                day = str(b.get("time", ""))[:10]
                if day and day not in break_dates:
                    break_dates.append(day)

    def _supports(row: dict[str, Any], target: str) -> bool:
        """Does this test row point at `target`? Encodes each test's own null direction."""
        tid = str(row.get("id", ""))
        rejects = bool(row.get("rejects_null"))
        says = str(row.get("says", "")).lower()
        if target in {"stationary", "trend_stationary", "stationary_around_break"}:
            if "kpss" in tid or "ocsb" in tid or "canova" in tid:
                return not rejects
            return rejects
        if target in {"unit_root", "unit_root_or_break"}:
            if "kpss" in tid:
                return rejects
            if "ocsb" in tid or "canova" in tid:
                return not rejects
            return not rejects
        if target == "seasonal_unit_root":
            # Only the seasonal battery can speak to a seasonal unit root. ADF has no opinion.
            if "ocsb" in tid:
                return not rejects          # OCSB's null IS the seasonal unit root
            if "canova" in tid:
                return "seasonal" in says or not rejects
            return False
        return False

    n_agree = 0
    if v:
        pool = ur + seas
        if v == "seasonal_unit_root":
            # Only the seasonal battery addressed the seasonal question, so only it is counted.
            # The regular unit-root tests are reported separately in the criterion's `value`, since
            # adding them together would produce one impressive number describing two different
            # claims.
            n_agree = sum(1 for r in seas if _supports(r, v))
            n_regular = sum(1 for r in ur if _supports(r, "unit_root"))
            n_regular_total = len(ur)
        else:
            n_agree = sum(1 for r in pool if _supports(r, v))
        if v in {"stationary_around_break", "unit_root_or_break"}:
            n_agree = max(1, n_agree + len(break_dates))
    n_tests = len(ur) + len(seas)
    n_regular = 0
    n_regular_total = 0

    # ---- S1/S2 facts ----
    # `n_outliers == 0` means "none found" only if something looked. `eda_facts` is an empty dict
    # until Step 1 runs, so its absence is checked here rather than inferred from the count.
    profile_ran = "n_outliers" in eda and "n_missing" in eda
    n_outliers = int(_num(eda, "n_outliers", default=0) or 0)
    outlier_treated = any(
        p.step == "outlier" for p in session.state.preprocessing
    )

    # ---- S4 ----
    shares = decomp.get("shares") or {}
    findings = {f.get("id"): f for f in (decomp.get("findings") or [])}
    remainder = findings.get("stat.remainder") or {}

    # ---- what has been done to the series ----
    steps = session.state.preprocessing
    seasonality_handled = bool(
        any(p.step == "difference" and (p.params or {}).get("lag", 1) not in (1, None)
            for p in steps)
        or (period is not None and _num(decomp, "seasonal_strength") is not None
            and (_num(decomp, "seasonal_strength") or 0) >= 0.30)
        or (period is not None and _num(eda, "seasonal_strength") is not None
            and (_num(eda, "seasonal_strength") or 0) >= 0.30)
    )

    horizon = session.objects.get("forecast_horizon")

    return GateFacts(
        n=n,
        n_original=int(session.objects.get("series_original", series).notna().sum())
        if session.objects.get("series_original") is not None else n,
        profile_ran=profile_ran,
        n_missing=int(eda.get("n_missing") or prep.get("n_missing") or 0),
        longest_gap=int(_num(eda, "longest_gap", default=0) or 0),
        period=period,
        data_class=str(prep.get("data_class") or "continuous"),
        horizon=int(horizon) if horizon else None,
        horizon_requested=bool(horizon),
        stationarity_verdict=v,
        stationarity_confidence=(verdict.get("confidence") or None),
        n_unit_root_tests=n_tests,
        n_agreeing=int(n_agree),
        n_regular_supporting=int(n_regular),
        n_regular_tests=int(n_regular_total),
        n_contradictions=len(verdict.get("contradictions") or []),
        differencing_needed=bool(verdict.get("differencing_needed")),
        seasonal_differencing_needed=bool(
            (next((r for r in seas if "ocsb" in str(r.get("id"))), None) or {})
            .get("rejects_null") is False
        ) if seas else False,
        break_dates=break_dates,
        break_count=len(break_dates),
        breaks_tested=any(
            row.get("statistic") is not None and not row.get("error")
            for row in (battery.get("breaks") or [])
        ),
        n_outliers=n_outliers,
        outliers_addressed=outlier_treated,
        # Two different quantities share the name `variance_ratio` upstream:
        #   * in `eda_facts` it is the late/early rolling-SD ratio (a *growth multiple*, ~1 = stable)
        #   * in `prep_facts` it is the same growth multiple, while `variance_level_corr` is the
        #     rank correlation between the rolling level and the rolling spread (0 = stable).
        # Reading `eda_facts["variance_ratio"]` as if it were the correlation would call a series
        # with a 3.4× growing spread "stable" — so each is mapped to the field that means it, and
        # the fallback is the growth ratio, converted to a percentage.
        variance_ratio=_num(prep, "variance_level_corr"),
        variance_growth_pct=(
            _num(prep, "variance_growth_pct")
            if _num(prep, "variance_growth_pct") is not None
            else (lambda r: (r - 1.0) * 100.0 if r is not None else None)(
                _num(eda, "variance_ratio"))
        ),
        signal_to_noise=_num(prep, "signal_to_noise"),
        seasonal_strength=_num(decomp, "seasonal_strength") or _num(eda, "seasonal_strength"),
        trend_r2=_num(eda, "trend_r2"),
        skewness=_num(eda, "skewness"),
        excess_kurtosis=_num(eda, "excess_kurtosis"),
        remainder_share=_num(shares, "remainder"),
        remainder_lb_p=_num(remainder, "p_value"),
        remainder_arch_p=None,
        correlogram_pattern=(correlogram.get("pattern") or {}).get("kind"),
        ljung_box_p=(
            (_num(row, "p_value") if (row := next(
                (r for r in (correlogram.get("ljung_box") or [])
                 if r.get("lag") == min(10, correlogram.get("max_lag") or 10)), None)) else None)
        ),
        n_transformations=sum(1 for p in steps if p.step in {"transform", "difference"}),
        seasonality_handled=seasonality_handled,
    )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def _checklist_table(gate: Gate) -> TableResult:
    rows = []
    for c in gate.checklist:
        rows.append({
            "criterion": c.label,
            "state": {
                "pass": "pass",
                "fail": "fail",
                "unknown": "not assessed",
                "not_applicable": "not applicable",
            }[c.state],
            "value": c.value,
            "what_it_means": c.detail,
        })
    return TableResult(
        id="table.gate_checklist",
        stage=Stage.S8_GATE,
        title="Readiness checklist",
        plain=(
            "Ten objective checks, each with the number it was judged on. A check that could not be "
            "run says 'not assessed' rather than passing by default."
        ),
        why=(
            "A model fitted to data that fails these checks produces numbers that look like results "
            "and are not. The checklist is the difference between 'the model said so' and 'the "
            "model said so, given these limitations'."
        ),
        method=(
            "Thresholds: at least 20 observations; at least 2 complete seasonal cycles for a "
            "seasonal model; horizon within n/5 of the sample; stationarity established by two or "
            "more agreeing tests; seasonal strength F_S >= 0.30 to require a seasonal term."
        ),
        columns=[
            TableColumn(key="criterion", label="Criterion"),
            TableColumn(key="state", label="Result"),
            TableColumn(key="value", label="The number"),
            TableColumn(key="what_it_means", label="What it means"),
        ],
        rows=rows,
        caption="Thresholds are fixed and shown, so a reader can disagree with them explicitly "
                "rather than silently.",
        footnote="'Not assessed' is not a pass. It means the stage that would have measured this "
                 "has not run.",
        caveats=[
            "A pass here means the data can support a model; it says nothing about whether a "
            "particular model is a good one.",
        ],
        reproducible_snippet=(
            "import pandas as pd\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "print('n =', s.notna().sum(), '| missing =', int(s.isna().sum()))\n"
            "print('cycles =', len(s) / 12)\n"
        ),
    )


def _verdict_result(gate: Gate) -> VerdictResult:
    return VerdictResult(
        id="verdict.readiness",
        stage=Stage.S8_GATE,
        title="Model readiness",
        plain=gate.plain,
        why=(
            "Everything downstream — model suggestions, forecasts, accuracy metrics and the final "
            "report — inherits this verdict. It is stated once, in one place, so it can be quoted "
            "rather than reconstructed."
        ),
        method="Rule-based: any failed blocker gives NOT READY; otherwise any failed caveat gives "
               "READY WITH CAVEATS; otherwise any unassessed check gives INCONCLUSIVE; otherwise "
               "READY.",
        verdict=VERDICT_TO_ENUM[gate.verdict],
        headline=gate.headline,
        checklist=[
            ChecklistItem(
                key=c.key,
                label=c.label,
                passed=c.passed,
                value=c.value,
                detail=c.detail if c.state != "unknown" else c.why_unknown or c.detail,
                jump_to_stage=STAGE_OF_CHECK.get(c.key),
            )
            for c in gate.checklist
        ],
        blockers=[f"{c.label} — {c.value}" for c in gate.blockers],
        caveats_travelling=gate.caveats_travelling,
        caveats=[
            "A readiness verdict is about the data, not about the model that will be chosen next.",
        ],
        reproducible_snippet=(
            "import pandas as pd\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "print('n =', s.notna().sum(), '| missing =', int(s.isna().sum()))\n"
        ),
    )


def _narrative(gate: Gate, facts: GateFacts) -> NarrativeResult:
    name = facts.as_dict().get("series_name") or "the series"
    paras = [
        gate.plain,
        f"The verdict is {gate.verdict.replace('_', ' ').upper()}. "
        f"{gate.passed} of {len(gate.checklist)} checks pass, "
        f"{len(gate.blockers)} block and {len(gate.caveats)} limit what the output can claim.",
    ]
    if gate.caveats_travelling:
        paras.append(
            "The following must appear with any result produced from this series: "
            + " ".join(gate.caveats_travelling)
        )
    return NarrativeResult(
        id="narrative.readiness",
        stage=Stage.S8_GATE,
        title="Is this series ready for modelling?",
        plain="The verdict and its travelling caveats, in prose, for the methods section.",
        why="A report must state its own limitations before its results, not after.",
        method="Composed from the checklist; every number also appears in the table.",
        paragraphs=paras,
        numbers_cited={
            k: float(v) for k, v in {
                "n": facts.n, "n_missing": facts.n_missing,
                "n_agreeing": facts.n_agreeing, "n_contradictions": facts.n_contradictions,
                "seasonal_strength": facts.seasonal_strength,
                "signal_to_noise": facts.signal_to_noise,
            }.items() if v is not None
        },
        caveats=["The verdict describes the current version of the series. Re-running any earlier "
                 "stage invalidates it."],
        reproducible_snippet=(
            "import pandas as pd\n"
            "s = pd.read_csv('series.csv', index_col=0, parse_dates=True).iloc[:, 0]\n"
            "print(len(s), int(s.isna().sum()))\n"
        ),
    )


def _advisor(gate: Gate) -> AdvisorReport:
    """Blockers and caveats as RuleOut records, so the assistant and the report can cite them."""
    def _rule(c: Any, kind: str) -> RuleOut:
        return RuleOut(
            id=f"gate.{kind}.{c.key}",
            stage=Stage.S8_GATE,
            recommend=c.label,
            confidence=Confidence.LOW if kind == "blocker" else Confidence.MEDIUM,
            because=c.detail,
            provenance=[STAGE_OF_CHECK.get(c.key, Stage.S8_GATE).value],
            alternative=None,
            risk_if_ignored=(
                "A model fitted anyway will produce numbers that look like results and are not."
                if kind == "blocker" else
                "The limitation will be missing from the report, which is how limitations become "
                "errors."
            ),
            how_to=f"Step {STAGE_OF_CHECK.get(c.key, Stage.S8_GATE).value[1:]}",
        )

    return AdvisorReport(
        stage=Stage.S8_GATE,
        headline=gate.headline,
        recommendations=[_rule(c, "caveat") for c in gate.caveats],
        blockers=[_rule(c, "blocker") for c in gate.blockers],
        facts_used={
            c.key: c.value for c in gate.checklist
        },
        confidence=Confidence.HIGH if gate.verdict in {"ready", "not_ready"} else Confidence.MEDIUM,
        next_stage_ready=gate.verdict in {"ready", "ready_with_caveats"},
        narrative=gate.plain,
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
class HorizonRequest(BaseModel):
    horizon: int = Field(..., ge=1, le=1000, description="Steps ahead the forecast must reach")


def _sync_session_caveats(session: Session, gate: Gate) -> list[str]:
    """
    Keep the session's caveat list in step with the *current* verdict.

    Two failure modes this closes:

    * a caveat that only lives in this stage's response never reaches the report, which is exactly
      what §5.7 forbids;
    * a caveat from a superseded verdict (horizon 60 → horizon 12) lingers and keeps warning about
      a plan the user has already abandoned.

    So the gate records which caveats it contributed and replaces only its own on each run; other
    stages' caveats in the same list are left untouched.

    Returns the caveats it contributed, which the caller stores *with* the payload — reading the
    previous list out of `session.objects[GATE]` after writing the new payload would find nothing to
    remove, which is how the first version of this left a horizon-60 warning alive at horizon 12.
    """
    previous = list((session.objects.get(GATE) or {}).get("session_caveats") or [])
    for cav in previous:
        if cav in session.state.caveats:
            session.state.caveats.remove(cav)
    added: list[str] = []
    for cav in gate.caveats_travelling:
        if cav not in session.state.caveats:
            session.state.caveats.append(cav)
            added.append(cav)
    return added


@router.post("/analysis/gate", response_model=AnalysisResponse)
def run_gate(session: Session = Depends(session_dep)) -> AnalysisResponse:
    """
    Assess readiness, store the verdict, and attach the travelling caveats to the session.

    The caveats are pushed into `session.state.caveats` deliberately: every later stage already
    appends its own caveats there, and the report reads that list. A caveat that lives only in this
    stage's output would not reach the report.
    """
    t0 = time.perf_counter()
    series = _require_series(session)
    facts = _build_facts(session, series)
    gate = assess(facts)

    results: list[Any] = [_verdict_result(gate), _checklist_table(gate), _narrative(gate, facts)]
    for r in results:
        session.add_result(r)

    verdict = _verdict_result(gate)
    contributed = _sync_session_caveats(session, gate)
    session.objects[GATE] = {
        "version": session.state.series_version,
        "payload": gate.to_dict(),
        "facts": facts.as_dict(),
        "at": time.time(),
        "session_caveats": contributed,
    }
    session.mark_stage(
        Stage.S8_GATE, "complete",
        f"{gate.verdict} — {gate.passed}/{len(gate.checklist)} pass, "
        f"{len(gate.blockers)} blocker(s)",
    )

    return AnalysisResponse(
        stage=Stage.S8_GATE,
        advisor=_advisor(gate),
        results=results,
        pipeline=session.state.progress(),
        caveats=gate.caveats_travelling,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/gate")
def read_gate(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    The stored verdict. `stale` is reported rather than hidden: a verdict about a different version
    of the series must never be presented as current.
    """
    stored = session.objects.get(GATE)
    if not stored:
        return {
            "stage": Stage.S8_GATE.value,
            "status": "not_started",
            "verdict": None,
            "message": "The readiness gate has not been run.",
            "stale": False,
        }
    return {
        "stage": Stage.S8_GATE.value,
        "status": "complete",
        "stale": stored.get("version") != session.state.series_version,
        "computed_for_version": stored.get("version"),
        "current_version": session.state.series_version,
        **stored["payload"],
        "facts": stored["facts"],
        "caveats_travelling": stored["payload"].get("caveats_travelling", []),
    }


@router.post("/analysis/gate/horizon")
def set_horizon(body: HorizonRequest, session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    Record the intended forecast horizon and re-assess.

    Doing this at the gate rather than at the forecast is the point: if the horizon is too long for
    the sample, the user learns it while the plan can still change.
    """
    series = _require_series(session)
    session.objects["forecast_horizon"] = int(body.horizon)
    facts = _build_facts(session, series)
    gate = assess(facts)
    # A horizon set here changes which caveats are true, so the report must see the new list — not
    # the one from the last full assessment. Synced before the payload is replaced so the caveats
    # contributed by the previous verdict can be identified and withdrawn.
    contributed = _sync_session_caveats(session, gate)
    session.objects[GATE] = {
        "version": session.state.series_version,
        "payload": gate.to_dict(),
        "facts": facts.as_dict(),
        "at": time.time(),
        "session_caveats": contributed,
    }
    horizon_check = next((c for c in gate.checklist if c.key == "horizon"), None)
    return {
        "stage": Stage.S8_GATE.value,
        "horizon": int(body.horizon),
        "verdict": gate.verdict,
        "headline": gate.headline,
        "horizon_check": horizon_check.to_dict() if horizon_check else None,
        "ratio": round(facts.n / max(1, body.horizon), 2),
        "caveats_travelling": gate.caveats_travelling,
    }


__all__ = ["router"]
