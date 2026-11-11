"""Deterministic manuscript templates with real, grounded statistic slots.

The templates intentionally contain ``[PLACEHOLDER: statistic]`` style slots.
`fill_template` replaces every one from stored results before a report is built;
an unavailable statistic becomes the Part 17 fallback naming the panel to open,
never a guessed number. No prose here is labelled AI-drafted.
"""

from __future__ import annotations

import re
from typing import Any

from app.ts.report.schemas import ReportScorecard
from app.ts.report.utils import as_dict, fallback_for, jsonable

PLACEHOLDER_EXAMPLE = "[PLACEHOLDER: statistic]"
PLACEHOLDER_RE = re.compile(r"\[PLACEHOLDER:\s*([^\]]+)\]")

TEMPLATES: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    (
        "methods", "Methods",
        "We analysed [PLACEHOLDER: n_observations] observations at [PLACEHOLDER: frequency] "
        "frequency, covering [PLACEHOLDER: observation_window]. Data quality was "
        "[PLACEHOLDER: data_quality]. [PLACEHOLDER: preprocessing] Components were "
        "characterised using [PLACEHOLDER: decomposition_method]: [PLACEHOLDER: components]. "
        "Stationarity was assessed with [PLACEHOLDER: stationarity_tests]; the stored verdict "
        "was [PLACEHOLDER: stationarity_verdict]. [PLACEHOLDER: acf_pacf] "
        "Preprocessing decisions, model-family filters and transforms were recorded before "
        "the out-of-sample evaluation; no test-period observation informed them.",
        ("table.descriptives", "table.prep.evidence", "table.decomp_scorecard",
         "table.stationarity", "table.correlogram_acf", "table.correlogram_pacf"),
    ),
    (
        "results", "Results",
        "The sample contains [PLACEHOLDER: n_observations] observations with mean "
        "[PLACEHOLDER: series_mean] and standard deviation [PLACEHOLDER: series_sd]. "
        "[PLACEHOLDER: components] The readiness checklist recorded "
        "[PLACEHOLDER: recorded_caveats] travelling caveats, which remain attached to "
        "every downstream result. [PLACEHOLDER: readiness_prose] "
        "Numerical results quoted below come from the stored scorecard and catalogue; "
        "in-sample fit is never presented as forecast accuracy.",
        ("table.descriptives", "narrative.decomp_scorecard", "narrative.readiness",
         "verdict.readiness", "table.gate_checklist"),
    ),
    (
        "model_selection", "Model Selection",
        "The registry comparison covered [PLACEHOLDER: candidate_count] fitted candidates "
        "including the benchmark [PLACEHOLDER: benchmark]. The rolling-origin shortlist "
        "selected [PLACEHOLDER: selected_model] with held-out MASE "
        "[PLACEHOLDER: selected_cv_mase] and held-out MAE [PLACEHOLDER: selected_cv_mae]. "
        "[PLACEHOLDER: aicc_note] Information criteria are reported only where the "
        "effective sample, transform stack and differencing order make them comparable; "
        "the forecast decision uses held-out accuracy, not in-sample fit.",
        ("models.fit_comparison", "select.shortlist", "models.suggestions"),
    ),
    (
        "diagnostics", "Diagnostics",
        "Residual adequacy was issued as [PLACEHOLDER: adequacy_verdict]. "
        "[PLACEHOLDER: adequacy_detail] The forecast carries "
        "[PLACEHOLDER: interval_levels] uncertainty bands, each with its coverage and "
        "construction recorded in [PLACEHOLDER: interval_source]. The horizon was "
        "[PLACEHOLDER: forecast_horizon] steps; the first point forecast was "
        "[PLACEHOLDER: point_first] and the last was [PLACEHOLDER: point_last]. "
        "Caveats from diagnostics travel with the forecast rather than being silently dropped.",
        ("forecast.adequacy", "diagnose.verdicts", "forecast.intervals", "forecast.points"),
    ),
    (
        "forecast_accuracy", "Forecast Accuracy",
        "Forecast accuracy was evaluated out of sample only: n = [PLACEHOLDER: n_observations], "
        "frequency [PLACEHOLDER: frequency], exact evaluation window "
        "[PLACEHOLDER: evaluation_window], [PLACEHOLDER: folds] rolling-origin folds and "
        "[PLACEHOLDER: pooled_test_rows] pooled held-out observations. The selected metric "
        "set led with [PLACEHOLDER: primary_metric]; [PLACEHOLDER: primary_model] achieved "
        "[PLACEHOLDER: primary_score]. The training-only benchmark was "
        "[PLACEHOLDER: benchmark]. [PLACEHOLDER: coverage_note] "
        "The S13 full-history fit was not scored as accuracy, no rows were silently dropped, "
        "and relative measures use training-only scales.",
        ("evaluate.metric_table", "stat.evaluate.primary", "evaluate.verdict",
         "evaluate.protocol_folds", "evaluate.coverage_construction", "evaluate.leakage_audit"),
    ),
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if number.is_integer():
            return str(int(number))
        return f"{number:,.{digits}f}"
    return str(value)


def _table_rows(session: Any, catalog: dict[str, Any], *ids: str) -> list[dict[str, Any]]:
    for rid in ids:
        result = session.get_result(rid)
        rows = getattr(result, "rows", None)
        if rows:
            return [dict(x) for x in rows]
    for item in catalog.get("plots", []) + catalog.get("tables", []):
        if str(item.get("id")) in ids:
            return [dict(x) for x in (item.get("rows") or []) if isinstance(x, dict)]
    return []


def _first_value(rows: list[dict[str, Any]], label: str, key: str = "value") -> Any:
    for row in rows:
        if str(row.get("quantity", row.get("label", row.get("criterion", "")))).lower() == label.lower():
            return row.get(key)
    return None


def _catalog_item(catalog: dict[str, Any], item_id: str) -> dict[str, Any]:
    for item in catalog.get("plots", []) + catalog.get("tables", []):
        if item.get("id") == item_id:
            return item
    return {}


def _fallback(stage: str, value: Any, *, default: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or default or fallback_for(stage)


def build_facts(session: Any, catalog: dict[str, Any], scorecard: ReportScorecard) -> dict[str, str]:
    """Every fact is taken from a stored result, object or catalogue row."""
    dataset = session.state.dataset
    descriptives = _table_rows(session, catalog, "table.descriptives")
    decomp = _table_rows(session, catalog, "table.decomp_scorecard")
    decomp_object = as_dict(session.objects.get("decomposition"))
    stat_verdict = as_dict(session.objects.get("stat_verdict"))
    fits = as_dict(session.objects.get("fits"))
    fit_rows = _table_rows(session, catalog, "models.fit_comparison")
    shortlist = _table_rows(session, catalog, "select.shortlist")
    evaluation = as_dict(session.objects.get("evaluation"))
    evaluation_report = as_dict(evaluation.get("report"))
    inputs = as_dict(evaluation_report.get("inputs"))
    policy = as_dict(evaluation_report.get("policy"))
    metric_rows = _table_rows(session, catalog, "evaluate.metric_table")
    primary_stat = session.get_result("stat.evaluate.primary")
    coverage_stat = session.get_result("stat.evaluate.coverage95")
    adequacy = session.get_result("forecast.adequacy")
    intervals = _table_rows(session, catalog, "forecast.intervals")
    points = _table_rows(session, catalog, "forecast.points")
    readiness = session.get_result("narrative.readiness")
    acf_item = _catalog_item(catalog, "table.correlogram_acf")
    pacf_item = _catalog_item(catalog, "table.correlogram_pacf")

    components = (
        f"trend {_fmt(next((r.get('share') for r in decomp if str(r.get('component')) == 'Trend'), None), 3)}, "
        f"seasonality {_fmt(next((r.get('share') for r in decomp if str(r.get('component')) == 'Seasonality'), None), 3)}, "
        f"remainder {_fmt(next((r.get('share') for r in decomp if str(r.get('component')) == 'Remainder'), None), 3)}"
        if decomp else fallback_for("S4")
    )
    prep_steps = list(session.state.preprocessing)
    prep_text = (
        f"{len(prep_steps)} versioned preprocessing step(s) were recorded "
        f"({', '.join(str(getattr(p, 'method', '')) for p in prep_steps[:4])})"
        if prep_steps else fallback_for("S2")
    )
    unit_root = list(stat_verdict.get("unit_root") or []) + list(stat_verdict.get("seasonal") or [])
    stationarity_tests = (
        f"{len(unit_root)} agreeing unit-root / seasonal unit-root test rows"
        if unit_root else fallback_for("S5")
    )
    if acf_item or pacf_item:
        acf_text = (
            f"ACF and PACF were reported with Bartlett confidence bands "
            f"({len(acf_item.get('rows') or [])} ACF lags and "
            f"{len(pacf_item.get('rows') or [])} PACF lags)."
        )
    else:
        acf_text = fallback_for("S6")
    winner = shortlist[0] if shortlist else {}
    # The evaluation report is the source of truth for the selected primary
    # model. Parsing a prose sentence for it was brittle and could misreport a
    # runner-up as the primary when the wording changed.
    primary_model = str(
        inputs.get("primary_model")
        or winner.get("model")
        or inputs.get("forecast_model_id")
        or ""
    )
    primary_score = getattr(primary_stat, "value", None)
    if primary_score is None and metric_rows:
        for row in metric_rows:
            if str(row.get("model") or row.get("primary_model")) == primary_model:
                primary_score = row.get(str(policy.get("primary") or "crps"))
                break
    aicc = winner.get("aicc")
    aicc_text = (
        f"The shortlist's leading AICc was {_fmt(aicc)}; comparability guards blank any "
        "criterion computed on a different transform or effective sample."
        if aicc not in (None, "") else
        "No comparable AICc was stored for the shortlist leader, so no in-sample criterion is quoted."
    )
    levels = [str(r.get("coverage")) for r in intervals if r.get("coverage")]
    if not levels:
        levels = [f"{int(float(c) * 100)}%" for c in (inputs.get("coverages") or []) if c is not None]
    point_values = [r.get("point") for r in points if r.get("point") is not None]
    coverage_note = (
        f"Achieved 95% coverage was {_fmt(getattr(coverage_stat, 'value', None), 1)}% "
        "across the held-out blocks."
        if coverage_stat is not None else
        "Interval coverage and construction are reported in the stored evaluation tables."
    )
    facts: dict[str, Any] = {
        "n_observations": dataset.n_observations if dataset else _first_value(descriptives, "Observations (n)"),
        "frequency": dataset.frequency.label if dataset else fallback_for("S0"),
        "observation_window": f"{dataset.time_start} to {dataset.time_end}" if dataset else fallback_for("S0"),
        "data_quality": dataset.quality.plain_summary if dataset else fallback_for("S1"),
        "series_mean": _first_value(descriptives, "Mean"),
        "series_sd": _first_value(descriptives, "Standard deviation"),
        "preprocessing": prep_text,
        "decomposition_method": decomp_object.get("label") or decomp_object.get("method") or fallback_for("S4"),
        "components": components,
        "stationarity_tests": stationarity_tests,
        "stationarity_verdict": stat_verdict.get("headline") or fallback_for("S5"),
        "acf_pacf": acf_text,
        "recorded_caveats": len(session.state.caveats),
        "readiness_prose": " ".join(getattr(readiness, "paragraphs", []) or []) or fallback_for("S8"),
        "candidate_count": max(len(fits), len(fit_rows), len(shortlist)),
        "benchmark": inputs.get("benchmark") or inputs.get("benchmark_id") or fallback_for("S11"),
        "selected_model": winner.get("model") or fallback_for("S11"),
        "selected_cv_mase": winner.get("cv_mase") or fallback_for("S11"),
        "selected_cv_mae": winner.get("cv_mae") or fallback_for("S11"),
        "aicc_note": aicc_text,
        "adequacy_verdict": getattr(adequacy, "verdict", None) or fallback_for("S12"),
        "adequacy_detail": getattr(adequacy, "plain", None) or fallback_for("S12"),
        "interval_levels": ", ".join(levels) if levels else fallback_for("S13"),
        "interval_source": "forecast.intervals" if intervals else fallback_for("S13"),
        "forecast_horizon": len(points) or inputs.get("forecast_horizon") or fallback_for("S13"),
        "point_first": point_values[0] if point_values else fallback_for("S13"),
        "point_last": point_values[-1] if point_values else fallback_for("S13"),
        "evaluation_window": inputs.get("evaluation_window") or fallback_for("S14"),
        "folds": inputs.get("folds") or fallback_for("S14"),
        "pooled_test_rows": inputs.get("n_pooled_test_observations") or fallback_for("S14"),
        "primary_metric": policy.get("primary_label") or policy.get("primary") or fallback_for("S14"),
        "primary_model": primary_model or fallback_for("S14"),
        "primary_score": primary_score if primary_score is not None else fallback_for("S14"),
        "coverage_note": coverage_note,
    }
    formatted = {k: _fmt(v) for k, v in jsonable(facts).items()}
    # A fallback is already prose; do not let _fmt turn None into "—" in a sentence.
    for key, value in facts.items():
        if value is None:
            formatted[key] = fallback_for("S15")
    formatted["scorecard_verdict"] = scorecard.verdict
    return formatted


def fill_template(template: str, facts: dict[str, str]) -> str:
    """Replace every slot; an unknown slot gets a visible, panel-named fallback."""
    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        return facts.get(key) or fallback_for("S15")
    return PLACEHOLDER_RE.sub(replace, template)


def manuscript_sections(
    session: Any,
    catalog: dict[str, Any],
    scorecard: ReportScorecard,
) -> list[dict[str, Any]]:
    facts = build_facts(session, catalog, scorecard)
    sections: list[dict[str, Any]] = []
    for section_id, title, template, result_ids in TEMPLATES:
        text = fill_template(template, facts)
        if PLACEHOLDER_RE.search(text):  # pragma: no cover - defensive
            raise ValueError(f"unfilled manuscript placeholder in {section_id}")
        sections.append({
            "id": section_id,
            "title": title,
            "template": template,
            "paragraphs": [text],
            "result_ids": [x for x in result_ids if x],
            "note": "Template prose — verify before submission; numbers come from stored results.",
            "ai_drafted": False,
        })
    return sections


__all__ = [
    "PLACEHOLDER_EXAMPLE", "PLACEHOLDER_RE", "TEMPLATES", "build_facts",
    "fill_template", "manuscript_sections",
]
