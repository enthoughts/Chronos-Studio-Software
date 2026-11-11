"""The weighted S15 completeness scorecard and its stored ResultBase cards."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Iterable

from app.core.schemas import (
    ChecklistItem,
    NarrativeResult,
    STAGE_LABELS,
    Stage,
    TableColumn,
    TableResult,
    Verdict,
    VerdictResult,
)
from app.ts.report.schemas import ReportItem, ReportNextStep, ReportScorecard
from app.ts.report.utils import as_dict, fallback_for, safe_json, stage_label

READY_TEXT = "Analysis complete — ready for manuscript writing"
NEARLY_TEXT = "Nearly ready — 2 items outstanding"
NOT_READY_TEXT = "Not yet — see blockers"

# The Part 16 catalogue is expensive to rebuild for every status read. A cached
# id list is valid only for the exact stored-result / pipeline fingerprint that
# produced it: changing the data, results or a stage status invalidates it.
CATALOG_CACHE_KEY = "report_catalog_ids"
CATALOG_FINGERPRINT_KEY = "report_catalog_fingerprint"

# Ten weighted checks from §5.13. The weights sum to 100; missing evidence never
# contributes points, and each row keeps the result ids and panel it came from.
CHECK_SPECS: tuple[tuple[str, str, int, str, tuple[str, ...], str], ...] = (
    ("data_quality", "Data quality resolved or documented", 10, "S1",
     ("table.descriptives", "narr.series_description"),
     "Open Explore (EDA) and record or resolve the data-quality findings."),
    ("preprocessing", "Preprocessing justified in writing", 10, "S2",
     ("table.prep.evidence", "fig.prep.variance_level"),
     "Open Preprocessing advice and record why each treatment was accepted or rejected."),
    ("components", "Components characterized", 10, "S4",
     ("narrative.decomp_scorecard", "table.decomp_scorecard", "fig.decomposition"),
     "Open Decompose and characterize trend, seasonality, cycle and remainder."),
    ("stationarity", "Stationarity established with ≥2 agreeing tests", 12, "S5",
     ("table.stationarity", "narrative.stationarity_verdict"),
     "Open Test stationarity and reconcile at least two unit-root tests."),
    ("acf_pacf", "ACF / PACF reported", 8, "S6",
     ("table.correlogram_acf", "table.correlogram_pacf", "correlation.acf_stems"),
     "Open ACF / PACF and report the correlogram with its confidence bands."),
    ("candidates", "≥2 candidate models + benchmarks compared", 10, "S11",
     ("models.fit_comparison", "select.shortlist", "models.suggestions"),
     "Fit at least two candidates, including the benchmark, then open Select model."),
    ("evaluation", "Out-of-sample evaluation done", 12, "S14",
     ("evaluate.metric_table", "evaluate.verdict", "fig.evaluate.split",
      "table.evaluation_metric_rows"),
     "Open Evaluate and run the rolling-origin, out-of-sample evaluation."),
    ("adequacy", "Residual adequacy verdict issued", 10, "S12",
     ("diagnose.verdicts", "forecast.adequacy"),
     "Open Diagnostics and issue the residual-adequacy verdict."),
    ("forecast", "Forecast with uncertainty intervals", 10, "S13",
     ("forecast.points", "forecast.intervals", "fig.forecast.fan"),
     "Open Forecast and produce point forecasts with stated uncertainty intervals."),
    ("caveats", "Assumptions & caveats recorded", 8, "S8",
     ("verdict.readiness", "table.gate_checklist", "narrative.readiness"),
     "Open Ready to model? and carry the assumptions and caveats into the report."),
)

SNIPPET_START = (
    "# Rebuild the manuscript-readiness scorecard from the stored result JSON.\n"
    "import json\n"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stage_complete(session: Any, stage: str) -> bool:
    st = session.state.stages.get(stage)
    return str(getattr(st, "status", "") or "") == "complete"


def catalog_fingerprint(session: Any, result_ids: Iterable[str]) -> str:
    """A stable digest of the session state that determines the catalogue."""
    state = session.state
    ids = sorted(str(x) for x in result_ids)
    stages = ",".join(
        f"{key}:{getattr(value, 'status', '')}"
        for key, value in sorted(getattr(state, "stages", {}).items())
    )
    payload = "|".join((
        str(getattr(state, "series_version", 0)),
        str(getattr(state, "updated_at", "")),
        str(len(ids)),
        ",".join(ids),
        stages,
    ))
    return sha256(payload.encode("utf-8")).hexdigest()


def cache_catalog_ids(
    session: Any,
    item_ids: Iterable[str],
    result_ids: Iterable[str] | None = None,
) -> list[str]:
    """Publish catalogue ids to the scorecard with their invalidation digest."""
    ids = [str(x) for x in item_ids]
    stored = {r.id for r in session.all_results()} if result_ids is None else set(result_ids)
    objects = getattr(session, "objects", {})
    objects[CATALOG_CACHE_KEY] = ids
    objects[CATALOG_FINGERPRINT_KEY] = catalog_fingerprint(session, stored)
    return ids


def _all_ids(session: Any) -> set[str]:
    """Stored result ids plus a fingerprint-validated Part 16 catalogue cache."""
    ids = {r.id for r in session.all_results()}
    objects = getattr(session, "objects", {})
    fingerprint = catalog_fingerprint(session, ids)
    cached = objects.get(CATALOG_CACHE_KEY)
    if (
        isinstance(cached, (list, tuple, set))
        and objects.get(CATALOG_FINGERPRINT_KEY) == fingerprint
    ):
        ids.update(str(x) for x in cached)
        return ids
    # A fresh session has no catalogue items; there is no reason to build one.
    if not ids:
        return ids
    # A genuine catalogue failure on a real analysis must be visible, not turned
    # into an empty report that still claims manuscript readiness.
    from app.charts import completeness as C
    catalog = C.build_catalog(session, mode="print")
    item_ids = [str(x["id"]) for x in catalog["plots"] + catalog["tables"]]
    cache_catalog_ids(session, item_ids, ids)
    ids.update(item_ids)
    return ids


def _result(session: Any, result_id: str) -> Any | None:
    return session.get_result(result_id)


def _rows(session: Any, result_id: str) -> list[dict[str, Any]]:
    r = _result(session, result_id)
    return [dict(x) for x in getattr(r, "rows", [])] if r is not None else []


def _plain(session: Any, result_id: str) -> str:
    r = _result(session, result_id)
    return str(getattr(r, "plain", "") or "")


def _status(session: Any, stage: str) -> str:
    st = session.state.stages.get(stage)
    return str(getattr(st, "status", "") or "")


def _stationarity_agreement(session: Any) -> tuple[bool, str]:
    verdict = as_dict(session.objects.get("stat_verdict"))
    tests = list(verdict.get("unit_root") or []) + list(verdict.get("seasonal") or [])
    counts: dict[str, int] = {}
    for test in tests:
        item = as_dict(test)
        says = str(item.get("says") or "").strip().lower()
        if says:
            counts[says] = counts.get(says, 0) + 1
    best = max(counts.values(), default=0)
    passed = _stage_complete(session, "S5") and best >= 2
    detail = (
        f"{best} agreeing stationarity tests: " + ", ".join(f"{k} × {v}" for k, v in counts.items())
        if counts else fallback_for("S5")
    )
    return passed, detail


def _candidate_check(session: Any) -> tuple[bool, str]:
    fits = as_dict(session.objects.get("fits"))
    shortlist = _rows(session, "select.shortlist")
    fit_rows = _rows(session, "models.fit_comparison")
    benchmark_names = {"mean", "drift", "naive", "seasonal_naive", "white_noise", "gaussian_wn"}
    fit_names = {str(name).lower() for name in fits}
    has_benchmark = any(name in benchmark_names for name in fit_names)
    for row in fit_rows:
        model_id = str(row.get("model_id") or row.get("model") or "").lower()
        status = str(row.get("status") or "").lower()
        family = str(row.get("family") or "").lower()
        if model_id in benchmark_names or status == "benchmark" or family == "baseline":
            has_benchmark = True
    n_candidates = max(len(fits), len(shortlist), len(fit_rows))
    passed = _stage_complete(session, "S11") and n_candidates >= 2 and has_benchmark
    return passed, f"{n_candidates} candidates; benchmark {'present' if has_benchmark else 'missing'}"


def _evaluation_check(session: Any) -> tuple[bool, str]:
    evaluation = as_dict(session.objects.get("evaluation"))
    report = as_dict(evaluation.get("report"))
    inputs = as_dict(report.get("inputs"))
    passed = _stage_complete(session, "S14") and bool(evaluation)
    window = str(inputs.get("evaluation_window") or "")
    n_test = inputs.get("n_pooled_test_observations")
    detail = (
        f"out-of-sample {window}, {n_test} held-out observations" if window
        else fallback_for("S14")
    )
    return passed, detail


def _adequacy_check(session: Any) -> tuple[bool, str]:
    adequacy = _plain(session, "forecast.adequacy") or _plain(session, "diagnose.verdicts")
    passed = _stage_complete(session, "S12") and bool(adequacy)
    return passed, adequacy or fallback_for("S12")


def _forecast_check(session: Any) -> tuple[bool, str]:
    forecast = as_dict(session.objects.get("forecast"))
    bundle = forecast.get("bundle")
    bundle_dict = as_dict(bundle)
    bands = getattr(bundle, "bands", None) or bundle_dict.get("bands") or []
    horizon = getattr(bundle, "horizon", None) or bundle_dict.get("horizon")
    passed = (
        _stage_complete(session, "S13")
        and bool(bands)
        and bool(_result(session, "forecast.intervals") or _result(session, "forecast.points"))
    )
    return passed, f"horizon {horizon}, {len(bands)} interval bands" if bands else fallback_for("S13")


def _check(session: Any, item_id: str) -> tuple[bool, str]:
    dataset = session.state.dataset
    present = _all_ids(session)
    if item_id == "data_quality":
        documented = bool(
            dataset and (
                dataset.quality.plain_summary
                or dataset.quality.issues
                or any("descript" in x or "quality" in x for x in present)
            )
        )
        detail = dataset.quality.plain_summary if dataset else fallback_for("S1")
        return _stage_complete(session, "S1") and documented, detail or fallback_for("S1")
    if item_id == "preprocessing":
        advice = bool(
            session.advice.get(Stage.S2_ADVISE_PREP)
            or session.advice.get("S2")
            or any("prep" in x for x in present)
        )
        applied = len(session.state.preprocessing)
        return advice, (
            f"{applied} versioned preprocessing step(s) applied; written advice retained"
            if advice else fallback_for("S2")
        )
    if item_id == "components":
        found = [x for x in ("narrative.decomp_scorecard", "table.decomp_scorecard", "fig.decomposition") if x in present]
        return _stage_complete(session, "S4") and bool(found), (
            "trend, seasonality, cycle and remainder results are stored" if found else fallback_for("S4")
        )
    if item_id == "stationarity":
        return _stationarity_agreement(session)
    if item_id == "acf_pacf":
        found = [x for x in present if "acf" in x.lower() or "pacf" in x.lower() or "correl" in x.lower()]
        return _stage_complete(session, "S6") and bool(found), (
            f"{len(found)} ACF / PACF result(s) reported" if found else fallback_for("S6")
        )
    if item_id == "candidates":
        return _candidate_check(session)
    if item_id == "evaluation":
        return _evaluation_check(session)
    if item_id == "adequacy":
        return _adequacy_check(session)
    if item_id == "forecast":
        return _forecast_check(session)
    if item_id == "caveats":
        travelling = list(session.state.caveats)
        found = [x for x in ("verdict.readiness", "table.gate_checklist", "narrative.readiness") if x in present]
        return bool(travelling) and bool(found), (
            f"{len(travelling)} travelling caveats recorded" if travelling else fallback_for("S8")
        )
    return False, fallback_for("S15")


def build_scorecard(session: Any) -> ReportScorecard:
    """Compute ten weighted checks from stored results, never from fresh statistics."""
    present = _all_ids(session)
    items: list[ReportItem] = []
    for item_id, label, weight, panel, candidates, fix in CHECK_SPECS:
        passed, evidence = _check(session, item_id)
        available = [rid for rid in candidates if rid in present]
        items.append(ReportItem(
            id=item_id,
            label=label,
            weight=weight,
            passed=passed,
            evidence=evidence,
            result_ids=list(candidates),
            available_result_ids=available,
            panel_id=panel,
            panel_label=stage_label(panel),
            fix=fix,
        ))
    score = sum(x.weight for x in items if x.passed)
    max_score = sum(x.weight for x in items)
    outstanding = [x for x in items if not x.passed]
    if not outstanding:
        status: str = "ready"
        verdict = READY_TEXT
    elif len(outstanding) <= 2:
        status = "nearly"
        # §5.13 names the three verdict strings verbatim. The numbered next-step
        # list and the scorecard rows carry the exact outstanding count.
        verdict = NEARLY_TEXT
    else:
        status = "not_ready"
        verdict = NOT_READY_TEXT
    next_steps = [
        ReportNextStep(
            number=i,
            item_id=x.id,
            text=f"{x.fix} Visit {x.panel_id} — {x.panel_label}.",
            panel_id=x.panel_id,
            panel_label=x.panel_label,
            result_ids=x.available_result_ids or x.result_ids,
        )
        for i, x in enumerate(outstanding, start=1)
    ]
    return ReportScorecard(
        generated_at=_utc_now(),
        items=items,
        score=score,
        max_score=max_score,
        percent=round(100 * score / max_score, 1),
        outstanding=[x.label for x in outstanding],
        blockers=[x.label for x in outstanding] if status == "not_ready" else [],
        next_steps=next_steps,
        verdict=verdict,
        status=status,  # type: ignore[arg-type]
        ready=status == "ready",
    )


def _snippet(scorecard: ReportScorecard) -> str:
    payload = safe_json({"verdict": scorecard.verdict, "score": scorecard.score, "items": [
        {"id": x.id, "passed": x.passed, "weight": x.weight} for x in scorecard.items
    ]})
    return SNIPPET_START + f"scorecard = json.loads({payload!r})\nprint(scorecard['verdict'])\n"


def scorecard_results(scorecard: ReportScorecard) -> list[Any]:
    """Three S15 cards: verdict, weighted table and numbered next steps."""
    snippet = _snippet(scorecard)
    common = dict(
        stage=Stage.S15_FINALIZE,
        technical="Weighted manuscript-readiness checklist defined in §5.13",
        method="Read stored result ids, stage statuses and travelling caveats; no statistic is recomputed",
        inputs={"stage": "S15", "max_score": scorecard.max_score},
        citations=sorted({rid for x in scorecard.items for rid in x.available_result_ids}),
        reproducible_snippet=snippet,
        # Session.add_result appends ResultBase.caveats to the session's
        # travelling caveats. Report cards must not add their own outstanding
        # labels there: a later readiness read would count the report card as
        # evidence for the caveats check. The verdict's caveats_travelling field
        # and the numbered next-steps card carry that information instead.
        caveats=[],
    )
    verdict_value = {
        "ready": Verdict.READY,
        "nearly": Verdict.READY_WITH_CAVEATS,
        "not_ready": Verdict.NOT_READY,
    }[scorecard.status]
    verdict = VerdictResult(
        id="report.readiness",
        title="Ready for manuscript writing?",
        plain=scorecard.verdict,
        why="A manuscript should not quote an incomplete analysis; every outstanding item names its panel.",
        verdict=verdict_value,
        headline=scorecard.verdict,
        checklist=[
            ChecklistItem(
                key=x.id,
                label=x.label,
                passed=x.passed,
                value=f"{x.weight} points",
                detail=x.evidence,
                jump_to_stage=Stage(x.panel_id),
            )
            for x in scorecard.items
        ],
        blockers=list(scorecard.blockers),
        caveats_travelling=list(scorecard.outstanding),
        warnings=[] if scorecard.ready else ["Verify outstanding items before submitting the manuscript."],
        **common,
    )
    table = TableResult(
        id="report.scorecard",
        title="Manuscript-readiness scorecard",
        plain=f"{scorecard.score} of {scorecard.max_score} weighted points are evidenced by stored results.",
        why="The weighted checklist links each claim to a result id and the exact panel to visit.",
        columns=[
            TableColumn(key="check", label="Check", dtype="string"),
            TableColumn(key="weight", label="Weight", dtype="integer", align="right"),
            TableColumn(key="status", label="Status", dtype="string", align="center"),
            TableColumn(key="panel", label="Panel", dtype="string"),
            TableColumn(key="results", label="Result links", dtype="string"),
            TableColumn(key="evidence", label="Evidence", dtype="string"),
        ],
        rows=[{
            "check": x.label,
            "weight": x.weight,
            "status": "✓" if x.passed else "✗",
            "panel": f"{x.panel_id} — {x.panel_label}",
            "results": ", ".join(x.available_result_ids or x.result_ids),
            "evidence": x.evidence,
        } for x in scorecard.items],
        caption=f"Weighted S15 scorecard: {scorecard.score}/{scorecard.max_score}. {scorecard.verdict}.",
        footnote="✓ = evidenced by a stored result; ✗ = open the named panel before submission.",
        **common,
    )
    paragraphs = (
        [f"{x.number}. {x.text}" for x in scorecard.next_steps]
        or ["No outstanding items: all ten weighted checks link to stored results."]
    )
    narrative = NarrativeResult(
        id="report.next_steps",
        title="What to do next",
        plain="The numbered list points to the exact panels needed before manuscript submission.",
        why="A readiness verdict is useful only when every gap has a concrete, user-confirmed next action.",
        paragraphs=paragraphs,
        audience="beginner",
        ai_drafted=False,
        numbers_cited={"score": float(scorecard.score), "max_score": float(scorecard.max_score)},
        **common,
    )
    return [verdict, table, narrative]


def finalize_session(session: Any) -> ReportScorecard:
    """Store the scorecard cards, complete S15 and remember the result for exports."""
    scorecard = build_scorecard(session)
    for result in scorecard_results(scorecard):
        session.add_result(result)
    session.objects["report"] = scorecard.model_dump(mode="json")
    session.state.blockers = list(scorecard.blockers)
    session.mark_stage(Stage.S15_FINALIZE, "complete", scorecard.verdict)
    return scorecard


__all__ = [
    "CATALOG_CACHE_KEY", "CATALOG_FINGERPRINT_KEY", "CHECK_SPECS",
    "NEARLY_TEXT", "NOT_READY_TEXT", "READY_TEXT",
    "build_scorecard", "cache_catalog_ids", "catalog_fingerprint",
    "finalize_session", "scorecard_results",
]
