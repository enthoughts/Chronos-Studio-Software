"""
api/forecast.py — Part 14, S13 forecasting (§5.11, §6.7–11).

Endpoints:

    GET  /analysis/forecast                 the stored forecast + the menu the UI may offer
    POST /analysis/forecast                build (or rebuild) the S13 forecast bundle
    GET  /analysis/forecast/horizon        the H > n/5 advisor, before spending time forecasting
    POST /analysis/forecast/hierarchy      optional aggregate <-> disaggregate reconciliation
    GET  /analysis/forecast/hierarchy      the stored reconciliation

Design rules, in the order they bite:

1. **The stored fits are the only source of numbers.** Nothing is refitted here except
   the user-supplied hierarchy's component series. Re-fitting the main series would
   quietly change the model Step 11 selected.
2. **A changed series is a refusal, not a stale forecast.** `fits["version"]` must
   equal the session's current `series_version`.
3. **An unsafe horizon is a refusal.** H > n/5 needs `acknowledge_long_horizon=True`;
   the advisor says why and by how much.
4. **A transform needs an explicit bias-adjustment choice.** No silent exponentiation.
5. **Every result carries the S12 adequacy verdict**, coverage and construction for
   every band, and the warnings the forecast earned. §6.8/10/11 are not UI prose;
   they are fields on this bundle and on every `ResultBase` produced here.
6. **No test-period observation is read.** The bundle records `n_train`, the exact
   training window and `test_observations_used = 0`.

Scenario paths come from the existing `POST /analysis/scenario` (Step 10): this
endpoint attaches the stored frame to the exogenous wrapper at predict time and
reports the held-flat drivers and a baseline-vs-scenario comparison.
"""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field, field_validator
from scipy import stats

from app.api.deps import session_dep
from app.api.models import FITS, _exog, _frequency_code, _future_index, _period, _series
from app.charts.primitives import fan_chart
from app.charts.theme import DASHES
from app.core.errors import ChronosError
from app.core.schemas import (
    AdvisorReport,
    AnalysisResponse,
    ChecklistItem,
    Confidence,
    FigureResult,
    NarrativeResult,
    RuleOut,
    Stage,
    TableColumn,
    TableResult,
    Verdict,
    VerdictResult,
)
from app.core.session import Session
from app.ts.forecast.backtransform import METHODS as BACK_TRANSFORM_METHODS
from app.ts.forecast.backtransform import detect_transform
from app.ts.forecast.combine import METHODS as COMBINATION_METHODS
from app.ts.forecast.combine import Component, combine_components
from app.ts.forecast.core import ForecastBundle, ForecastCore, build_bundle, single_core
from app.ts.forecast.horizons import COVERAGES, advise_horizon
from app.ts.forecast.intervals import MIN_PATHS, make_draws
from app.ts.forecast.reconcile import METHODS as RECONCILE_METHODS
from app.ts.forecast.reconcile import reconcile
from app.ts.models.fit import DISCRETE_PREDICTIVE, fit_one
from app.core.schemas import DataClass

router = APIRouter(tags=["analysis"])

FORECAST = "forecast"
HIERARCHY = "forecast_hierarchy"
DIAGNOSTICS = "diagnostics"
SELECTION = "selection"
_Z975 = float(stats.norm.ppf(0.975))


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class ForecastRequest(BaseModel):
    model_id: str | None = Field(None, description="Stored fit to forecast from; defaults to the Step 11 winner.")
    horizon: int = Field(12, ge=1, le=240)
    levels: list[float] = Field(
        default_factory=lambda: list(COVERAGES),
        description="Which of 50/66/80/90/95% to draw in the fan and tables. All five are computed.",
    )
    interval_method: Literal["analytic", "simulation", "auto"] = "auto"
    n_paths: int = Field(MIN_PATHS, ge=MIN_PATHS, le=20_000)
    seed: int | None = Field(None, description="Defaults to the session seed, so reruns reproduce.")
    acknowledge_long_horizon: bool = False
    back_transform: Literal["naive", "smearing", "lognormal"] | None = Field(
        None,
        description="Required when a transform is active. No exponentiation happens without it.",
    )
    combination: Literal["none", "mean", "median", "ic_weighted"] = "none"
    models: list[str] = Field(default_factory=list, description="Combination components; defaults to the Step 11 shortlist.")

    @field_validator("levels")
    @classmethod
    def _levels_allowed(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("Choose at least one interval level.")
        allowed = {round(float(c), 4) for c in COVERAGES}
        bad = [x for x in v if round(float(x), 4) not in allowed]
        if bad:
            raise ValueError(
                f"Unknown interval level(s) {bad}; allowed: {[round(c, 4) for c in COVERAGES]}."
            )
        return [float(x) for x in dict.fromkeys(round(float(x), 4) for x in v)]


class HierarchyRequest(BaseModel):
    bottom: dict[str, list[float]] = Field(
        ..., description="Bottom-level history, one numeric array per series (equal lengths)."
    )
    total: list[float] | None = Field(None, description="Aggregate history; defaults to the sum of the bottoms.")
    model_id: str | None = None
    horizon: int = Field(6, ge=1, le=60)
    method: Literal["bottom_up", "top_down", "ols"] = "ols"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _require_fits(session: Session) -> dict[str, Any]:
    stored = session.objects.get(FITS)
    if not stored or not stored.get("outcomes"):
        raise ChronosError(
            plain="No model has been fitted, so there is nothing to forecast from.",
            fix="Open Step 10 (Fit) and fit at least one model, then run Step 11 (Select).",
            code="forecast.no_fits",
        )
    if stored.get("version") != session.state.series_version:
        raise ChronosError(
            plain="The series changed after these models were fitted, so a forecast would mix two "
                  "different scales.",
            fix="Re-fit the models in Step 10 and re-run Step 11 before forecasting.",
            code="forecast.stale_fits",
        )
    return stored


def _available(stored: dict[str, Any]) -> dict[str, Any]:
    return {
        o.summary.model_id: o
        for o in stored["outcomes"]
        if o.wrapper is not None and not o.summary.failed
    }


def _winner(session: Session, available: dict[str, Any]) -> str:
    sel = session.objects.get(SELECTION) or {}
    report = sel.get("report")
    mid = getattr(report, "winner", None) if report is not None else None
    if mid and mid in available:
        return str(mid)
    for key in sorted(available):
        if key != "combination":
            return key
    return next(iter(available), "")


def _shortlist(session: Session, limit: int = 3) -> list[str]:
    sel = session.objects.get(SELECTION) or {}
    report = sel.get("report")
    entries = getattr(report, "shortlist", None) or []
    return [str(getattr(e, "model_id", "")) for e in entries[:limit] if getattr(e, "model_id", "")]


def _non_gaussian(stored: dict[str, Any], outcome: Any) -> bool:
    dc = str(stored.get("data_class") or "")
    family = outcome.summary.family.value if outcome.summary else ""
    return (
        dc in {"count", "intermittent", "event_timestamp", "serial_count", "point_process"}
        or family in {"count", "intermittent", "serial_count", "point_process", "volatility"}
        or outcome.summary.model_id in DISCRETE_PREDICTIVE
    )


def _transform(session: Session) -> tuple[Any, list[str]]:
    return detect_transform(session.state.preprocessing)


def _frequency(session: Session) -> str:
    """Human-readable frequency for the UI; the raw code is kept for `fit_one`."""
    ds = session.state.dataset
    freq = getattr(ds, "frequency", None) if ds is not None else None
    code = getattr(freq, "code", None)
    label = getattr(freq, "label", None)
    if code and label and str(label) != str(code):
        return f"{label} ({code})"
    return str(code or "unspecified")


def _adequacy(session: Session, model_id: str) -> dict[str, Any]:
    """The S12 verdict, attached to every forecast shown or exported (§6.10)."""
    stored = session.objects.get(DIAGNOSTICS) or {}
    blocks = stored.get("blocks") or {}
    block = blocks.get(model_id) or {}
    verdict = block.get("verdict") or {}
    adequacy = str(verdict.get("adequacy") or "")
    if not adequacy:
        return {
            "status": "not_evaluated",
            "adequacy": "not_evaluated",
            "headline": (
                "Residual adequacy is INCONCLUSIVE: Step 12 has not been run for this fit. Run "
                "Diagnostics before quoting these intervals."
            ),
            "reasons": [],
            "remedies": ["Run Step 12 (Diagnostics) on this fit."],
            "model_id": model_id,
            "source": "S12 /analysis/diagnose",
            "stale": bool(stored and stored.get("version") != session.state.series_version),
        }
    return {
        "status": "evaluated",
        "adequacy": adequacy,
        "headline": str(verdict.get("headline") or f"Residual adequacy: {adequacy}."),
        "reasons": [str(r.get("detail", "")) for r in (verdict.get("reasons") or [])],
        "remedies": [str(x) for x in (verdict.get("remedies") or [])],
        "model_id": model_id,
        "source": "S12 /analysis/diagnose",
        "stale": bool(stored.get("version") != session.state.series_version),
    }


def _attach_exog(wrapper: Any, session: Session, series: pd.Series) -> dict[str, Any]:
    """
    Put the Step 10 scenario on the wrapper at *predict* time (as `/fit/forecast` does),
    and report which drivers were supplied versus held flat.
    """
    exog, future = _exog(session)
    supports = bool(getattr(wrapper, "supports_exog", False))
    scenario: dict[str, Any] = {
        "applies": supports and exog is not None,
        "model_supports_exog": supports,
        "drivers": [],
        "supplied_paths": [],
        "held_flat": [],
        "note": "This model uses only the target series' own history; no future driver path applies.",
        "horizon_mismatch": False,
        "scenario_steps": 0 if future is None else int(len(future)),
        "comparison": None,
    }
    if not supports:
        return scenario
    drivers = [str(c) for c in (exog.columns if exog is not None else [])]
    supplied = [str(c) for c in (future.columns if future is not None else [])]
    held = [c for c in drivers if c not in supplied]
    scenario.update(
        drivers=drivers,
        supplied_paths=supplied,
        held_flat=held,
        note=(
            f"Forecast is conditional on {len(supplied)} supplied driver path(s); {len(held)} "
            "driver(s) are held at their last observed value — a scenario, not a prediction."
            if drivers else "No driver columns are mapped."
        ),
        horizon_mismatch=False,
    )
    wrapper._exog_raw = exog
    wrapper._future_raw = future
    wrapper._exog = wrapper._prepare_exog(pd.Series(series).index)
    note = getattr(wrapper, "scenario_caveat", None)
    if isinstance(note, str) and note:
        scenario["note"] = note
    return scenario


def _scenario_comparison(wrapper: Any, horizon: int, future: pd.DataFrame | None) -> dict[str, Any] | None:
    """Held-flat baseline vs the supplied scenario, both from the same fitted wrapper."""
    if future is None or not getattr(wrapper, "supports_exog", False):
        return None
    original = getattr(wrapper, "_future_raw", None)
    try:
        wrapper._future_raw = None
        base = np.asarray(wrapper.predict(horizon, alpha=0.05)["mean"], dtype=float)
        wrapper._future_raw = future
        scen = np.asarray(wrapper.predict(horizon, alpha=0.05)["mean"], dtype=float)
    except Exception as exc:  # pragma: no cover - a broken exog wrapper must not break S13
        return {"error": f"The scenario comparison could not be computed: {exc}"}
    finally:
        wrapper._future_raw = original
    change = scen - base
    return {
        "baseline": [float(v) for v in base],
        "scenario": [float(v) for v in scen],
        "change": [float(v) for v in change],
        "changed": bool(np.max(np.abs(change)) > 1e-9),
        "max_abs_change": float(np.max(np.abs(change))),
        "construction": (
            "Two predict(h, alpha=0.05) calls on the same fitted wrapper: one with every future "
            "driver held at its last observed value, one with the stored /analysis/scenario paths. "
            "The difference is the effect of the scenario, not a forecast of the driver."
        ),
        "coverage": 0.95,
    }


def _index_labels(session: Session, horizon: int) -> list[str]:
    """Future timestamps without a meaningless `00:00:00` tail, for charts and tables."""
    out: list[str] = []
    for value in _future_index(session, horizon):
        if hasattr(value, "strftime"):
            fmt = "%Y-%m-%d" if (value.hour, value.minute, value.second) == (0, 0, 0) else "%Y-%m-%d %H:%M"
            out.append(value.strftime(fmt))
        else:
            out.append(str(value))
    return out


#: A caveat is context; a warning is something that can change the reading of the
#: numbers. Showing the same list twice (in yellow and in red) teaches nobody.
_URGENT = (
    "one fifth", "full seasons", "short seasonal", "INADEQUATE", "not been run",
    "Simulation could not", "Simulation was requested", "differencing operation",
    "held at", "clipped", "INCONCLUSIVE",
)


def _travelling(adequacy: dict[str, Any], warnings: list[str]) -> list[str]:
    """Deduplicate the travelling text, including the bundle's prefixed verdict.

    The bundle also exports `Residual adequacy verdict for this fit: …` for report
    writers. On a `ResultBase` card that sentence would merely repeat the headline
    already printed above it, so the redundant copy is dropped here rather than
    weakening the payload.
    """
    headline = adequacy["headline"]
    values = [headline, *warnings]
    values = [
        w for w in values
        if not (w.startswith("Residual adequacy verdict for this fit:") and headline in w)
    ]
    return list(dict.fromkeys(values))


def _common(stage: Stage, adequacy: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    travelling = _travelling(adequacy, warnings)
    urgent = [w for w in travelling if any(k in w for k in _URGENT)]
    return {
        "stage": stage,
        "caveats": travelling,
        "warnings": urgent,
        "inputs": {
            "residual_adequacy": adequacy["adequacy"],
            "residual_adequacy_source": adequacy["source"],
            "test_observations_used": 0,
        },
        "reproducible_snippet": (
            "from app.ts.forecast.core import build_bundle, single_core\n"
            "# Re-run with POST /api/v1/analysis/forecast and the stored request.\n"
            "print('Every band carries coverage and construction')\n"
        ),
    }


# --------------------------------------------------------------------------- #
# Result builders (ResultBase, §4.3)
# --------------------------------------------------------------------------- #
def _horizon_result(bundle: ForecastBundle, common: dict[str, Any]) -> NarrativeResult:
    a = bundle.horizon_advice
    return NarrativeResult(
        id="forecast.horizon",
        title="How far ahead this forecast reaches",
        plain=a.get("advice", ""),
        why="A horizon longer than a fifth of the history is a structural claim, not an observed one.",
        method=a.get("method", ""),
        paragraphs=[a.get("advice", "")] + [str(w) for w in a.get("warnings", [])],
        numbers_cited={
            "n": float(a.get("n", 0)),
            "horizon": float(a.get("horizon", 0)),
            "recommended_max": float(a.get("recommended_max", 0)),
        },
        **common,
    )


def _adequacy_result(bundle: ForecastBundle, common: dict[str, Any]) -> VerdictResult:
    a = bundle.adequacy
    mapping = {
        "adequate": Verdict.READY,
        "adequate_with_caveats": Verdict.READY_WITH_CAVEATS,
        "inadequate": Verdict.READY_WITH_CAVEATS,
        "not_evaluated": Verdict.INCONCLUSIVE,
    }
    verdict = mapping.get(str(a.get("adequacy")), Verdict.INCONCLUSIVE)
    advice = bundle.horizon_advice
    checklist = [
        ChecklistItem(
            key="residual_adequacy",
            label="S12 residual-adequacy verdict attached to this forecast",
            passed=a.get("status") == "evaluated",
            value=str(a.get("adequacy", "")).replace("_", " "),
            detail=a.get("headline", ""),
            jump_to_stage=Stage.S12_DIAGNOSE,
        ),
        ChecklistItem(
            key="interval_disclosure",
            label="Every band states its coverage and construction",
            passed=bool(bundle.bands) and all(b.construction for b in bundle.bands),
            value=f"{len(bundle.bands)} levels",
            detail="§6.8: no bare interval.",
        ),
        ChecklistItem(
            key="horizon",
            label="Horizon within about n/5 (or explicitly acknowledged)",
            passed=bool(advice.get("allowed")),
            value=f"H={bundle.horizon}, n/5={advice.get('recommended_max')}",
            detail="§6.11: long horizons need an explicit 'I understand' flag.",
        ),
        ChecklistItem(
            key="season",
            label="At least two full seasons of history",
            passed=not bool(advice.get("season_short")),
            value=f"n={advice.get('n')}, period={advice.get('period') or 'none'}",
            detail="§6.11: short seasonal history is flagged.",
        ),
        ChecklistItem(
            key="leakage",
            label="Test-period observations used in this forecast",
            passed=True,
            value="0",
            detail="§6.7: the final fit saw the training window only.",
        ),
        ChecklistItem(
            key="back_transform",
            label="Back-transformation choice stated",
            passed=True,
            value=bundle.transform.method,
            detail=bundle.transform.formula,
        ),
    ]
    travelling = _travelling(a, list(bundle.warnings))
    return VerdictResult(
        id="forecast.adequacy",
        title="Forecast honesty: can these intervals be trusted?",
        plain=a["headline"],
        why=(
            "A forecast without its residual verdict, coverage and construction is a number nobody "
            "can audit; all three travel into the report and the export."
        ),
        method="S12 adequacy verdict + §6.8/10/11 safeguards attached to the S13 bundle.",
        verdict=verdict,
        headline=(
            f"Residual adequacy: {str(a.get('adequacy', '')).replace('_', ' ').upper()}. "
            + a["headline"]
        ),
        checklist=checklist,
        blockers=[] if a.get("status") == "evaluated" else [
            "Run Step 12 (Diagnostics) to replace INCONCLUSIVE with a real adequacy verdict.",
        ],
        caveats_travelling=list(dict.fromkeys(travelling)),
        **common,
    )


def _band_rows(bundle: ForecastBundle) -> list[dict[str, Any]]:
    return [
        {
            "coverage": f"{b.coverage:.0%}",
            "construction": b.source,
            "source": b.source,
            "lower_first": float(b.lower[0]),
            "upper_first": float(b.upper[0]),
            "lower_last": float(b.lower[-1]),
            "upper_last": float(b.upper[-1]),
            "mean_width": float(np.mean(b.upper - b.lower)),
            "detail": b.construction,
        }
        for b in bundle.bands
    ]


def _intervals_table(bundle: ForecastBundle, common: dict[str, Any]) -> TableResult:
    return TableResult(
        id="forecast.intervals",
        title="Prediction intervals: coverage and how each was made",
        plain="Each band names its target coverage and the exact construction that produced it.",
        why="A 95% band from a normal formula and a 95% band from 2,000 simulated paths are not "
            "interchangeable; the reader has to be able to tell which one is on screen.",
        method="§5.11: repeated ModelWrapper.predict(alpha=1-coverage) and seeded simulation.",
        columns=[
            TableColumn(key="coverage", label="Coverage"),
            TableColumn(key="construction", label="Construction"),
            TableColumn(key="lower_first", label="h=1 lower", dtype="number", format="0.000", align="right"),
            TableColumn(key="upper_first", label="h=1 upper", dtype="number", format="0.000", align="right"),
            TableColumn(key="lower_last", label="h=H lower", dtype="number", format="0.000", align="right"),
            TableColumn(key="upper_last", label="h=H upper", dtype="number", format="0.000", align="right"),
            TableColumn(key="mean_width", label="Mean width", dtype="number", format="0.000", align="right"),
            TableColumn(key="detail", label="Exact method"),
        ],
        rows=_band_rows(bundle),
        caption=f"{bundle.interval_method} intervals, {bundle.display_name}, H={bundle.horizon}.",
        footnote=(
            f"Primary construction: {bundle.interval_method}. "
            + (bundle.simulation or {}).get("construction", "Analytic intervals from the fitted model.")
        ),
        **common,
    )


def _points_table(bundle: ForecastBundle, common: dict[str, Any]) -> TableResult:
    cols: list[TableColumn] = [
        TableColumn(key="step", label="h", dtype="integer", align="right"),
        TableColumn(key="index", label="When"),
        TableColumn(key="point", label="Point", dtype="number", format="0.000", align="right"),
    ]
    for b in bundle.bands:
        key = int(round(b.coverage * 100))
        cols.append(TableColumn(key=f"lower_{key}", label=f"{key}% lower", dtype="number", format="0.000", align="right"))
        cols.append(TableColumn(key=f"upper_{key}", label=f"{key}% upper", dtype="number", format="0.000", align="right"))
    rows: list[dict[str, Any]] = []
    for j in range(bundle.horizon):
        row: dict[str, Any] = {
            "step": j + 1,
            "index": bundle.index[j],
            "point": float(bundle.point[j]),
        }
        for b in bundle.bands:
            key = int(round(b.coverage * 100))
            row[f"lower_{key}"] = float(b.lower[j])
            row[f"upper_{key}"] = float(b.upper[j])
        rows.append(row)
    return TableResult(
        id="forecast.points",
        title="Point forecasts and intervals, step 1 to H",
        plain=f"The {bundle.horizon}-step path on the {bundle.transform.scale} scale, with every band.",
        why="The table behind the fan chart makes every number copyable and checkable.",
        method=bundle.construction,
        columns=cols,
        rows=rows,
        caption=f"{bundle.display_name}; n={bundle.inputs.get('n_train')}, frequency={bundle.inputs.get('frequency')}.",
        footnote=(
            f"Coverage/construction: {bundle.interval_method}. "
            f"Scale: {bundle.transform.formula}. "
            "Training window: all in-sample observations; test observations used: 0."
        ),
        **common,
    )


def _fan_result(
    bundle: ForecastBundle,
    observed: pd.Series,
    common: dict[str, Any],
) -> FigureResult:
    shown = [
        b for b in bundle.bands
        if any(abs(b.coverage - c) < 1e-9 for c in bundle.coverages)
    ] or list(bundle.bands)
    chart = fan_chart(
        observed,
        index=bundle.index,
        point=bundle.point,
        bands=[
            {
                "coverage": b.coverage,
                "lower": b.lower,
                "upper": b.upper,
                "construction": b.construction,
                "source": bundle.interval_method,
            }
            for b in shown
        ],
        title=f"{bundle.display_name}: {bundle.horizon}-step forecast fan",
        subtitle=(
            f"{bundle.interval_method} intervals at "
            + ", ".join(f"{b.coverage:.0%}" for b in shown)
            + f" · {bundle.transform.scale} scale · S12: {bundle.adequacy['adequacy'].replace('_', ' ')}"
        ),
        y_title=bundle.transform.scale,
        comparison=(
            [{
                "name": "Held-flat scenario",
                "y": bundle.scenario["comparison"]["baseline"],
                "dash": DASHES[2],
            }]
            if bundle.scenario.get("comparison") and "baseline" in bundle.scenario["comparison"]
            else None
        ),
    )
    behind_cols: list[TableColumn] = [
        TableColumn(key="step", label="h", dtype="integer", align="right"),
        TableColumn(key="index", label="When"),
        TableColumn(key="point", label="Point", dtype="number", format="0.000", align="right"),
    ]
    for b in shown:
        key = int(round(b.coverage * 100))
        behind_cols += [
            TableColumn(key=f"lower_{key}pct", label=f"{key}% lower", dtype="number", format="0.000", align="right"),
            TableColumn(key=f"upper_{key}pct", label=f"{key}% upper", dtype="number", format="0.000", align="right"),
        ]
    behind = TableResult(
        id="forecast.fan.data",
        title="Numbers behind the fan chart",
        plain="Every point and band endpoint drawn above, as numbers.",
        why="A figure that cannot be read as a table is not auditable.",
        method="§4.3: FigureResult.data_behind is a TableResult.",
        columns=behind_cols,
        rows=chart.data_behind,
        **common,
    )
    return FigureResult(
        id="fig.forecast.fan",
        title=f"{bundle.display_name}: forecast fan",
        plain=(
            f"The observed series followed by {bundle.horizon} forecast steps and nested "
            f"{', '.join(f'{b.coverage:.0%}' for b in shown)} intervals."
        ),
        why="Uncertainty grows with the horizon; nested bands make that growth visible instead of "
            "hiding it behind one 95% ribbon.",
        method=(
            f"{bundle.interval_method} intervals; "
            + (bundle.simulation or {}).get("construction", "model's analytic predict(alpha) intervals")
        ),
        figure_json=chart.figure_json,
        caption=(
            f"{bundle.display_name}, H={bundle.horizon}, {bundle.interval_method} construction. "
            f"Residual adequacy: {bundle.adequacy['adequacy'].replace('_', ' ')}."
        ),
        alt_text=chart.alt_text,
        data_behind=behind,
        **common,
    )


def _construction_result(bundle: ForecastBundle, common: dict[str, Any]) -> NarrativeResult:
    paragraphs = [
        bundle.construction,
        "Coverage and construction for every band: "
        + " ".join(f"{b.coverage:.0%} ({b.source}): {b.construction}" for b in bundle.bands),
        f"Back-transformation: {bundle.transform.method}. {bundle.transform.formula} "
        f"{bundle.transform.point_definition}.",
        *bundle.transform.notes,
        (
            f"Seeded simulation: {bundle.simulation['n_paths']:,} paths, seed {bundle.simulation['seed']}, "
            f"{bundle.simulation['distribution']} innovations, {bundle.simulation['coupling']} coupling. "
            + bundle.simulation["construction"]
            if bundle.simulation else "Simulation was not available for this fit."
        ),
        bundle.scenario.get("note", ""),
        (
            f"Combination: {bundle.combination['method']} over "
            f"{', '.join(bundle.combination['component_ids'])}; weights "
            + ", ".join(f"{k}={v:.4f}" for k, v in bundle.combination["weights"].items())
            if bundle.combination else "No combination was requested; one fitted model was forecast."
        ),
        f"Training window: {bundle.inputs.get('n_train')} observations at frequency "
        f"{bundle.inputs.get('frequency')}; test observations used: 0 (§6.7).",
    ]
    return NarrativeResult(
        id="forecast.construction",
        title="Exactly how this forecast was constructed",
        plain="The full method, seed, scale change, scenario and adequacy disclosure in one place.",
        why="A forecast that cannot be reproduced from its own description is an opinion.",
        method="§6.8–11 safeguards, recorded as fields on the S13 bundle.",
        paragraphs=[p for p in paragraphs if p],
        **common,
    )


def _optional_tables(bundle: ForecastBundle, common: dict[str, Any]) -> list[Any]:
    out: list[Any] = []
    d = bundle.discrete
    if d:
        out.append(TableResult(
            id="forecast.discrete",
            title="Discrete predictive distribution behind the count intervals",
            plain="Each step has a probability mass function; the interval is its integer quantiles.",
            why="A count forecast is a distribution over whole numbers, not a symmetric band.",
            method=d.get("interval_construction", ""),
            columns=[
                TableColumn(key="step", label="h", dtype="integer", align="right"),
                TableColumn(key="mean", label="PMF mean", dtype="number", format="0.000", align="right"),
                TableColumn(key="lower", label="95% lower (integer)", dtype="number", format="0", align="right"),
                TableColumn(key="upper", label="95% upper (integer)", dtype="number", format="0", align="right"),
                TableColumn(key="p_at_least_one", label="P(Y ≥ 1)", dtype="number", format="0.000", align="right"),
                TableColumn(key="truncated", label="Tail truncated"),
            ],
            rows=[
                {
                    "step": int(i) + 1,
                    "mean": float(d["mean"][i]),
                    "lower": float(d["lower"][i]),
                    "upper": float(d["upper"][i]),
                    "p_at_least_one": float(d["p_at_least_one"][i]),
                    "truncated": bool(d["truncated"][i]),
                }
                for i in range(min(len(d["index"]), bundle.horizon))
            ],
            footnote="PMF support and probabilities are shipped in the readback payload for Part 15.",
            **common,
        ))
    comp = bundle.scenario.get("comparison")
    if comp and "baseline" in comp and "change" in comp:
        out.append(TableResult(
            id="forecast.scenario",
            title="What the future-driver scenario changed",
            plain="The same fit forecast twice: drivers held flat, then drivers following your paths.",
            why="A scenario is an assumption about the drivers; the difference is the only part it buys.",
            method=comp["construction"],
            columns=[
                TableColumn(key="step", label="h", dtype="integer", align="right"),
                TableColumn(key="index", label="When"),
                TableColumn(key="baseline", label="Held flat", dtype="number", format="0.000", align="right"),
                TableColumn(key="scenario", label="Your scenario", dtype="number", format="0.000", align="right"),
                TableColumn(key="change", label="Change", dtype="number", format="0.000", align="right"),
            ],
            rows=[
                {
                    "step": j + 1,
                    "index": bundle.index[j],
                    "baseline": float(comp["baseline"][j]),
                    "scenario": float(comp["scenario"][j]),
                    "change": float(comp["change"][j]),
                }
                for j in range(bundle.horizon)
            ],
            footnote=f"Maximum absolute change: {comp['max_abs_change']:.4f}.",
            **common,
        ))
    return out


# --------------------------------------------------------------------------- #
# S13 — the forecast
# --------------------------------------------------------------------------- #
@router.post("/analysis/forecast", response_model=AnalysisResponse)
def run_forecast(body: ForecastRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    t0 = time.perf_counter()
    session.require_stage(Stage.S13_FORECAST, allow_stale=True)
    stored = _require_fits(session)
    series = _series(session)
    n = int(pd.Series(series).dropna().size)
    period = _period(session)
    advice = advise_horizon(
        n, body.horizon, period=period, acknowledged=body.acknowledge_long_horizon
    )
    if not advice.allowed:
        raise ChronosError(
            plain=(
                f"A {body.horizon}-step horizon exceeds one fifth of the {n} training "
                f"observations (about {advice.recommended_max} steps)."
            ),
            fix=(
                f"Set the horizon to {advice.recommended_max} or less, or tick 'I understand' so "
                "acknowledge_long_horizon=true is sent with the request."
            ),
            code="forecast.horizon_acknowledgement",
            context={
                "n": n,
                "horizon": body.horizon,
                "recommended_max": advice.recommended_max,
                "period": period,
            },
        )

    spec, transform_warnings = _transform(session)
    differenced = any("differencing operation" in w for w in transform_warnings)
    if spec is None and body.back_transform is not None:
        raise ChronosError(
            plain="No transform is active on this series, so there is nothing to bias-adjust.",
            fix="Omit back_transform (or send null) and the forecast stays in the original units.",
            code="forecast.back_transform_unneeded",
        )
    if spec is not None and body.back_transform is None:
        raise ChronosError(
            plain=(
                f"The modelled series has an active {spec.describe()}. Its forecast must not be "
                "silently exponentiated back to the original scale."
            ),
            fix=(
                "Choose one of naive (median, no adjustment), smearing (empirical mean, Duan), or "
                "lognormal (parametric mean) and send it as back_transform."
            ),
            code="forecast.back_transform_required",
            context={"methods": list(BACK_TRANSFORM_METHODS), "transform": spec.as_dict()},
        )
    if spec is not None and differenced:
        raise ChronosError(
            plain=(
                "The series was both transformed and differenced, so a back-transformed value "
                "would be a ratio of changes, not a level forecast."
            ),
            fix="Undo the differencing operation in Step 3 and re-fit, or choose a model without a transform.",
            code="forecast.differenced_transform",
        )

    available = _available(stored)
    if not available:
        raise ChronosError(
            plain="Every fitted model failed, so there is no successful fit to forecast from.",
            fix="Read the failure reasons in Step 10 and fit a model that converges.",
            code="forecast.no_successful_fits",
        )

    winner = _winner(session, available)
    primary_id = body.model_id or winner
    if primary_id not in available:
        raise ChronosError(
            plain=f"'{primary_id}' is not a successful fit in this session.",
            fix=f"Choose one of: {', '.join(sorted(available))}.",
            code="forecast.unknown_model",
        )
    primary = available[primary_id]

    wanted = list(dict.fromkeys(body.models)) or _shortlist(session)
    if body.combination != "none":
        wanted = [m for m in wanted if m in available and m != "combination"]
        if len(wanted) < 2:
            raise ChronosError(
                plain=f"A {body.combination} combination needs at least two successful component fits.",
                fix="Fit (and select) at least two models, or list them explicitly in 'models'.",
                code="forecast.combination_needs_two",
                context={"available": sorted(available)},
            )
        components_outcomes = [available[m] for m in wanted]
    else:
        components_outcomes = [primary]

    # Scenario/exogenous paths, attached at predict time (not fit time).
    exog, future = _exog(session)
    scenario = _attach_exog(primary.wrapper, session, series)
    for o in components_outcomes:
        if o.wrapper is not primary.wrapper:
            _attach_exog(o.wrapper, session, series)
    if future is not None and len(future) < body.horizon:
        scenario["horizon_mismatch"] = True
        scenario["note"] += (
            f" The stored scenario covers {len(future)} steps but this forecast is {body.horizon} "
            "steps; steps beyond the scenario fall back to the held-flat assumption."
        )
    comparison = _scenario_comparison(primary.wrapper, body.horizon, future)
    scenario["comparison"] = comparison

    seed = int(body.seed if body.seed is not None else session.seed)
    n_resid = 0
    for o in components_outcomes:
        r = o.wrapper.residuals() if hasattr(o.wrapper, "residuals") else None
        if r is not None:
            n_resid = max(n_resid, int(np.asarray(r).size))
    draws = make_draws(seed, body.n_paths, body.horizon, n_resid)
    core_warnings: list[str] = []
    combination_payload: dict[str, Any] | None = None

    non_gaussian = _non_gaussian(stored, primary)
    detail = (
        f"Family: {primary.summary.family.value}; data class: {stored.get('data_class')}."
    )
    if body.combination == "none":
        core, warns = single_core(
            primary.wrapper,
            horizon=body.horizon,
            coverages=COVERAGES,
            draws=draws,
            interval_method=body.interval_method,
            non_gaussian=non_gaussian,
            detail=detail,
        )
        core_warnings.extend(warns)
        model_id, display_name = primary_id, primary.summary.display_name
    else:
        comps: list[Component] = []
        for o in components_outcomes:
            c, warns = single_core(
                o.wrapper,
                horizon=body.horizon,
                coverages=COVERAGES,
                draws=draws,
                interval_method=body.interval_method,
                non_gaussian=_non_gaussian(stored, o),
                detail=detail,
            )
            core_warnings.extend(warns)
            if c.paths is None:
                raise ChronosError(
                    plain=f"{o.summary.display_name} produced no simulated paths, so it cannot be combined.",
                    fix="Choose analytic intervals or drop this model from the combination.",
                    code="forecast.combination_no_paths",
                )
            comps.append(Component(
                model_id=o.summary.model_id,
                display_name=o.summary.display_name,
                point=c.point,
                bands=c.analytic_bands,
                paths=c.paths,
                aicc=o.summary.aicc,
                ic_comparable=bool(o.summary.ic_comparable),
                discrete=c.is_discrete,
            ))
        try:
            combo = combine_components(comps, body.combination, coverages=COVERAGES)
        except ValueError as exc:
            raise ChronosError(
                plain=str(exc),
                fix="Use mean or median, or fit components whose AICc values are comparable.",
                code="forecast.combination_refused",
            ) from exc
        combination_payload = combo.as_dict()
        combination_payload["notes"] = combo.info["notes"]
        combination_payload["construction"] = combo.info["construction"]
        core = ForecastCore(
            point=combo.point,
            analytic_bands=combo.analytic_bands,
            simulation_bands=combo.simulation_bands,
            paths=combo.paths,
            simulation_info={
                "n_paths": combo.info["n_paths"],
                "seed": seed,
                "distribution": "pooled",
                "coupling": "shared seeded draws across components",
                "construction": combo.info["construction"],
                "notes": combo.info["notes"],
            },
            discrete_info=None,
            construction=(
                f"Combination of {len(comps)} stored fits ({', '.join(wanted)}). "
                + combo.info["construction"]
            ),
            is_discrete=all(c.discrete for c in comps),
        )
        model_id = f"combination:{body.combination}"
        display_name = f"{body.combination.replace('_', ' ').title()} combination ({len(comps)} models)"

    adequacy = _adequacy(session, primary_id)
    resid_for_smearing = primary.wrapper.residuals() if hasattr(primary.wrapper, "residuals") else None
    index = _index_labels(session, body.horizon)
    caveats = [adequacy["headline"]] + list(advice.warnings) + list(transform_warnings)
    if adequacy["adequacy"] == "inadequate":
        caveats.append(
            "The residual diagnostics were INADEQUATE. These intervals assume the model's error "
            "structure is correct; treat their width as optimistic until the model is revised."
        )
    if adequacy["status"] == "not_evaluated":
        caveats.append("Run Step 12 (Diagnostics) to replace the INCONCLUSIVE adequacy verdict.")
    if scenario["held_flat"]:
        caveats.append(scenario["note"])
    caveats.extend(core_warnings)

    try:
        bundle = build_bundle(
            core,
            model_id=model_id,
            display_name=display_name,
            horizon=body.horizon,
            index=index,
            coverages=body.levels,
            interval_method=body.interval_method,
            non_gaussian=non_gaussian,
            seed=seed,
            n_paths=body.n_paths,
            transform_spec=spec,
            back_transform=body.back_transform,
            residuals=resid_for_smearing,
            adequacy=adequacy,
            horizon_advice=advice.as_dict(),
            scenario=scenario,
            warnings=caveats,
            caveats=caveats,
            inputs={
                "n_train": n,
                "n_obs": int(pd.Series(series).size),
                "frequency": _frequency(session),
                "period": period,
                "training_window": f"all {n} in-sample observations",
                "evaluation_window": "none — this is a forecast, not an accuracy score",
                "data_class": str(stored.get("data_class") or ""),
                "model_ids": wanted if body.combination != "none" else [primary_id],
                "requested_levels": list(body.levels),
            },
            combination=combination_payload,
            transform_warnings=transform_warnings,
        )
    except ValueError as exc:
        raise ChronosError(
            plain=str(exc),
            fix="Choose another bias-adjustment method, or undo the transform in Step 3.",
            code="forecast.back_transform_refused",
            context={"methods": list(BACK_TRANSFORM_METHODS)},
        ) from exc

    session.objects[FORECAST] = {
        "version": session.state.series_version,
        "at": time.time(),
        "bundle": bundle,
        "request": body.model_dump(),
        "index": index,
    }

    common = _common(Stage.S13_FORECAST, adequacy, list(dict.fromkeys(bundle.warnings)))
    results: list[Any] = [
        _adequacy_result(bundle, common),
        _horizon_result(bundle, common),
        _fan_result(bundle, pd.Series(series), common),
        _points_table(bundle, common),
        _intervals_table(bundle, common),
        _construction_result(bundle, common),
        *_optional_tables(bundle, common),
    ]
    for result in results:
        session.add_result(result)

    message = (
        f"{display_name}: {body.horizon}-step forecast with {len(bundle.bands)} stated "
        f"interval levels ({bundle.interval_method}); S12 verdict {adequacy['adequacy']}."
    )
    session.mark_stage(Stage.S13_FORECAST, "complete", message)
    advisor = AdvisorReport(
        stage=Stage.S13_FORECAST,
        headline=advice.advice,
        recommendations=[
            RuleOut(
                id="forecast.read_intervals",
                stage=Stage.S13_FORECAST,
                recommend="Read the point path together with the widest stated band",
                confidence=Confidence.HIGH,
                because=(
                    f"{len(bundle.bands)} coverage levels, {bundle.n_paths:,} seeded paths and "
                    f"an S12 verdict of {adequacy['adequacy']} are attached."
                ),
                provenance=["fig.forecast.fan", "forecast.intervals", "forecast.adequacy"],
                alternative="Switch interval_method to see the other construction.",
                risk_if_ignored="A point forecast read alone looks more certain than it is.",
                how_to="Open Step 13 and compare the nested bands.",
            )
        ],
        facts_used={
            "n": n,
            "horizon": body.horizon,
            "n_paths": bundle.n_paths,
            "levels": len(bundle.bands),
            "adequacy": adequacy["adequacy"],
        },
        confidence=Confidence.MEDIUM if advice.long_horizon else Confidence.HIGH,
        next_stage_ready=True,
        narrative=message,
    )
    return AnalysisResponse(
        stage=Stage.S13_FORECAST,
        advisor=advisor,
        results=results,
        pipeline=session.state.progress(),
        caveats=list(dict.fromkeys(caveats)),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/forecast")
def get_forecast(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """The stored bundle, re-rendered. Never re-runs a fit or a simulation."""
    stored = session.objects.get(FORECAST)
    series = session.objects.get("series")
    n = int(pd.Series(series).dropna().size) if series is not None else 0
    period = _period(session)
    spec, transform_warnings = _transform(session)
    fits = session.objects.get(FITS) or {}
    outcomes = fits.get("outcomes") or []
    exog, _ = _exog(session)
    drivers = [str(c) for c in (exog.columns if exog is not None else [])]
    models = [
        {
            "model_id": o.summary.model_id,
            "display_name": o.summary.display_name,
            "family": o.summary.family.value,
            "failed": bool(o.summary.failed),
            "supports_exog": bool(getattr(o.wrapper, "supports_exog", False)) if o.wrapper else False,
            "aicc": o.summary.aicc,
            "ic_comparable": bool(o.summary.ic_comparable),
        }
        for o in outcomes
    ]
    available = _available(fits) if fits else {}
    max_h = advise_horizon(max(n, 1), 1, period=period).recommended_max
    options = {
        "models": models,
        "winner": _winner(session, available) if available else None,
        "coverages": list(COVERAGES),
        "interval_methods": ["auto", "analytic", "simulation"],
        "back_transform_methods": list(BACK_TRANSFORM_METHODS),
        "combination_methods": ["none", *COMBINATION_METHODS],
        "reconciliation_methods": list(RECONCILE_METHODS),
        "min_paths": MIN_PATHS,
        "transform_active": spec.as_dict() if spec else None,
        "transform_required": spec is not None,
        "transform_warnings": list(transform_warnings),
        "drivers": drivers,
        "scenario_endpoint": "/analysis/scenario",
        "n": n,
        "period": period,
        "frequency": _frequency(session),
        "recommended_max_h": max_h,
    }
    if not stored:
        return {
            "stage": Stage.S13_FORECAST.value,
            "status": "not_started",
            "message": "No forecast has been produced yet.",
            "forecast": None,
            "options": options,
        }
    return {
        "stage": Stage.S13_FORECAST.value,
        "status": "complete",
        "stale": stored["version"] != session.state.series_version,
        "computed_for_version": stored["version"],
        "current_version": session.state.series_version,
        "forecast": stored["bundle"].as_dict(),
        "request": stored["request"],
        "options": options,
    }


@router.get("/analysis/forecast/horizon")
def horizon_advice(
    horizon: int = Query(12, ge=1, le=240),
    acknowledged: bool = Query(False),
    session: Session = Depends(session_dep),
) -> dict[str, Any]:
    """The advisor the UI calls while the horizon field is being edited."""
    series = session.objects.get("series")
    n = int(pd.Series(series).dropna().size) if series is not None else 0
    if n < 1:
        raise ChronosError(
            plain="There is no series yet, so a horizon cannot be judged.",
            fix="Load a dataset in Step 0 first.",
            code="forecast.no_series",
        )
    advice = advise_horizon(n, horizon, period=_period(session), acknowledged=acknowledged)
    return {"stage": Stage.S13_FORECAST.value, **advice.as_dict()}


# --------------------------------------------------------------------------- #
# Optional hierarchy reconciliation
# --------------------------------------------------------------------------- #
def _hierarchy_fits(
    request: HierarchyRequest,
    stored: dict[str, Any],
    model_id: str,
    session: Session,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.Series]:
    from app.ts.models.fit import EXOGENOUS_IDS, ML_IDS, MULTIVARIATE_IDS

    if model_id in {"combination", *EXOGENOUS_IDS, *MULTIVARIATE_IDS, *ML_IDS}:
        raise ChronosError(
            plain=f"'{model_id}' cannot be refitted on standalone hierarchy series.",
            fix="Choose a univariate fit such as ets, arima, theta, naive or drift.",
            code="forecast.hierarchy_model_unsupported",
        )
    names = list(request.bottom.keys())
    arrays = [np.asarray(v, dtype=float) for v in request.bottom.values()]
    lengths = {int(a.size) for a in arrays}
    if len(lengths) != 1:
        raise ChronosError(
            plain="The bottom series have different lengths, so they cannot share a time index.",
            fix="Supply the same number of observations for every bottom series.",
            code="forecast.hierarchy_lengths",
        )
    n = int(next(iter(lengths)))
    if n < 16 or n > 600:
        raise ChronosError(
            plain=f"A hierarchy series needs between 16 and 600 observations; {n} were supplied.",
            fix="Supply a longer history, or fewer observations.",
            code="forecast.hierarchy_n",
        )
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ChronosError(
            plain="A hierarchy series contains a missing or infinite value.",
            fix="Impute or trim the series before reconciling.",
            code="forecast.hierarchy_nonfinite",
        )
    B = np.vstack(arrays)
    total = (
        np.asarray(request.total, dtype=float)
        if request.total is not None else B.sum(axis=0)
    )
    if total.size != n or not np.all(np.isfinite(total)):
        raise ChronosError(
            plain=f"The aggregate history must have {n} finite values, one per bottom observation.",
            fix="Omit 'total' to use the sum of the bottoms, or supply the matching length.",
            code="forecast.hierarchy_total",
        )

    period = stored.get("period") or _period(session)
    dc = str(stored.get("data_class") or "continuous")
    data_class = DataClass(dc) if dc in {d.value for d in DataClass} else DataClass.CONTINUOUS
    idx = pd.RangeIndex(n)
    series_list = [pd.Series(a, index=idx) for a in arrays] + [pd.Series(total, index=idx)]
    pts: list[np.ndarray] = []
    sds: list[np.ndarray] = []
    failures: list[str] = []
    for label, s in zip([*names, "total"], series_list):
        outcome = fit_one(
            model_id, s, period=period, data_class=data_class,
            frequency_code=_frequency_code(session),
        )
        if outcome.summary.failed or outcome.wrapper is None:
            failures.append(f"{label}: {outcome.summary.failure_reason}")
            pts.append(np.full(request.horizon, float(s.mean())))
            sds.append(np.full(request.horizon, float(s.std(ddof=1)) or 1.0))
            continue
        frame = outcome.wrapper.predict(request.horizon, alpha=0.05)
        pts.append(np.asarray(frame["mean"], dtype=float))
        sds.append(
            np.maximum(
                (np.asarray(frame["upper"], float) - np.asarray(frame["lower"], float)) / (2 * _Z975),
                1e-6,
            )
        )
    if failures:
        raise ChronosError(
            plain="The hierarchy could not produce honest base forecasts for every series.",
            fix="; ".join(failures[:3]) + ". Choose a simpler univariate model.",
            code="forecast.hierarchy_fit_failed",
        )
    return (
        np.vstack(pts[:-1]),
        pts[-1],
        np.vstack(sds[:-1]),
        sds[-1],
        names,
        pd.Series(total, index=idx),
    )


@router.post("/analysis/forecast/hierarchy", response_model=AnalysisResponse)
def run_hierarchy(body: HierarchyRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    t0 = time.perf_counter()
    session.require_stage(Stage.S13_FORECAST, allow_stale=True)
    stored = _require_fits(session)
    available = _available(stored)
    model_id = body.model_id or _winner(session, available)
    if model_id not in available:
        raise ChronosError(
            plain=f"'{model_id}' is not a successful fit in this session.",
            fix=f"Choose one of: {', '.join(sorted(available))}.",
            code="forecast.unknown_model",
        )
    if len(body.bottom) < 2 or len(body.bottom) > 8:
        raise ChronosError(
            plain="Reconciliation needs between 2 and 8 bottom series.",
            fix="Supply a JSON object with 2–8 named numeric arrays.",
            code="forecast.hierarchy_count",
        )
    B, a, sdB, sdA, names, total_history = _hierarchy_fits(body, stored, model_id, session)
    rec = reconcile(
        bottom_names=names,
        base_bottom=B,
        base_total=a,
        bottom_sd=sdB,
        total_sd=sdA,
        method=body.method,
    )
    payload = rec.as_dict()
    session.objects[HIERARCHY] = {
        "version": session.state.series_version,
        "at": time.time(),
        "reconciliation": payload,
        "request": body.model_dump(),
    }
    adequacy = _adequacy(session, model_id)
    common = _common(Stage.S13_FORECAST, adequacy, [rec.construction])
    index = _index_labels(session, body.horizon)
    payload["index"] = index

    table = TableResult(
        id="forecast.hierarchy.total",
        title=f"Reconciled aggregate ({body.method})",
        plain="The aggregate forecast after making the bottom series add up to it.",
        why="Independent base forecasts almost never add up; this is the coherent version.",
        method=rec.construction,
        columns=[
            TableColumn(key="step", label="h", dtype="integer", align="right"),
            TableColumn(key="base_total", label="Base total", dtype="number", format="0.000", align="right"),
            TableColumn(key="reconciled_total", label="Reconciled total", dtype="number", format="0.000", align="right"),
            TableColumn(key="change", label="Change", dtype="number", format="0.000", align="right"),
            TableColumn(key="lower95", label="95% lower", dtype="number", format="0.000", align="right"),
            TableColumn(key="upper95", label="95% upper", dtype="number", format="0.000", align="right"),
        ],
        rows=[
            {
                "step": j + 1,
                "base_total": float(rec.base_total[j]),
                "reconciled_total": float(rec.reconciled_total[j]),
                "change": float(rec.reconciled_total[j] - rec.base_total[j]),
                "lower95": float(rec.total_lower95[j]),
                "upper95": float(rec.total_upper95[j]),
            }
            for j in range(body.horizon)
        ],
        footnote=(
            f"95% interval: {payload['interval_construction']} "
            f"Max coherence error: {rec.coherence_error:.3g}."
        ),
        **common,
    )
    bottom_table = TableResult(
        id="forecast.hierarchy.bottom",
        title="Bottom series, before and after reconciliation",
        plain="How much each disaggregate forecast had to move.",
        why="A coherent total built by moving one series wildly is not a good reconciliation.",
        method=rec.construction,
        columns=[
            TableColumn(key="series", label="Series"),
            TableColumn(key="base_mean", label="Base mean", dtype="number", format="0.000", align="right"),
            TableColumn(key="reconciled_mean", label="Reconciled mean", dtype="number", format="0.000", align="right"),
            TableColumn(key="change", label="Change", dtype="number", format="0.000", align="right"),
        ],
        rows=[
            {
                "series": name,
                "base_mean": float(np.mean(rec.base_bottom[i])),
                "reconciled_mean": float(np.mean(rec.reconciled_bottom[i])),
                "change": float(np.mean(rec.reconciled_bottom[i] - rec.base_bottom[i])),
            }
            for i, name in enumerate(names)
        ],
        footnote=f"Base incoherence (max |total − sum(bottoms)|): {rec.base_coherence_error:.4f}.",
        **common,
    )
    chart = fan_chart(
        total_history,
        index=index,
        point=rec.reconciled_total,
        bands=[{
            "coverage": 0.95,
            "lower": rec.total_lower95,
            "upper": rec.total_upper95,
            "construction": payload["interval_construction"],
            "source": body.method,
        }],
        title=f"Reconciled aggregate fan ({body.method})",
        subtitle=f"95% analytic interval · {len(names)} bottom series · coherence error {rec.coherence_error:.2g}",
        comparison=[{"name": "Base total", "y": rec.base_total, "dash": DASHES[2]}],
    )
    figure = FigureResult(
        id="fig.forecast.hierarchy",
        title=f"Reconciled aggregate forecast ({body.method})",
        plain=f"The {len(names)} bottom forecasts and the aggregate now add up coherently.",
        why="A hierarchy is only useful when its levels agree.",
        method=rec.construction,
        figure_json=chart.figure_json,
        alt_text=chart.alt_text,
        caption=f"Coverage 95%; {payload['interval_construction']}",
        **common,
    )
    narrative = NarrativeResult(
        id="forecast.hierarchy.method",
        title="How the hierarchy was reconciled",
        plain=rec.construction,
        why="Diagonal-W OLS and top-down proportions make different assumptions; both are stated.",
        method=rec.construction,
        paragraphs=[rec.construction, *rec.notes],
        **common,
    )
    results = [table, bottom_table, figure, narrative]
    for result in results:
        session.add_result(result)
    session.mark_stage(
        Stage.S13_FORECAST, "complete",
        f"Reconciled {len(names)} bottom series ({body.method}); coherence error "
        f"{rec.coherence_error:.3g}",
    )
    return AnalysisResponse(
        stage=Stage.S13_FORECAST,
        results=results,
        pipeline=session.state.progress(),
        caveats=[adequacy["headline"], *rec.notes],
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/forecast/hierarchy")
def get_hierarchy(session: Session = Depends(session_dep)) -> dict[str, Any]:
    stored = session.objects.get(HIERARCHY)
    if not stored:
        return {
            "stage": Stage.S13_FORECAST.value,
            "status": "not_started",
            "message": "No hierarchy has been supplied yet.",
            "reconciliation": None,
        }
    return {
        "stage": Stage.S13_FORECAST.value,
        "status": "complete",
        "stale": stored["version"] != session.state.series_version,
        "reconciliation": stored["reconciliation"],
        "request": stored["request"],
    }


__all__ = ["router"]
