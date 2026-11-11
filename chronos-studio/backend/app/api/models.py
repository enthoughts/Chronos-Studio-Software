"""
api/models.py — S9 (suggest) and S10 (fit) (§5.8).

  GET  /analysis/suggest            the candidate set, with reasons and exclusions
  GET  /analysis/registry           every registry row and whether this build can fit it
  POST /analysis/fit                fit the requested candidates (baselines always included)
  GET  /analysis/fits               the stored fits, ranked, in a table the UI can render
  POST /analysis/fit/forecast       forecast from one stored fit (used by S13 later)

Why S9 and S10 share a router: the suggestion set is a *plan* and the fit set is that plan executed.
Splitting them across two modules would mean re-deriving the differencing hints and the period twice,
and the moment those two derivations disagree the app fits a model the user was not shown.

The gate is a hard dependency (see `STAGE_DEPENDENCIES`). If the verdict is `not_ready`, fitting is
refused rather than merely discouraged: §5.7 says blockers must stop modelling, and an app that shows
a big red NOT READY card and then fits anyway has taught the user that the card is decoration.
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
    Confidence,
    DataClass,
    FigureResult,
    FitSummary,
    ModelFamily,
    RuleOut,
    Stage,
    TableColumn,
    TableResult,
)
from app.core.session import Session
from app.ts.gate.readiness import GateFacts, assess
from app.ts.models.facts import build_facts, classify_data_class
from app.ts.models.fit import (
    DISCRETE_PREDICTIVE,
    IMPLEMENTED,
    FitOutcome,
    best_by_aicc,
    fit_many,
    summarise_failures,
    to_schema,
)
from app.ts.models.registry import all_specs, implemented_specs, is_implemented
from app.ts.models.suggest import suggest

router = APIRouter()

SUGGESTIONS = "model_suggestions"
FITS = "fits"

STAGE_OF_FAMILY: dict[ModelFamily, Stage] = {
    ModelFamily.BASELINE: Stage.S10_FIT,
    ModelFamily.SMOOTHING: Stage.S10_FIT,
    ModelFamily.ARMA: Stage.S10_FIT,
    ModelFamily.COUNT: Stage.S10_FIT,
    ModelFamily.INTERMITTENT: Stage.S10_FIT,
    ModelFamily.STRUCTURAL: Stage.S10_FIT,
    ModelFamily.EXOGENOUS: Stage.S10_FIT,
    ModelFamily.NONLINEAR: Stage.S10_FIT,
    ModelFamily.POINT_PROCESS: Stage.S10_FIT,
}


# --------------------------------------------------------------------------- #
# Shared context
# --------------------------------------------------------------------------- #
def _series(session: Session) -> pd.Series:
    """
    The series to model.

    For event-timestamp data this is **rebuilt from the raw timestamps** rather than read from the
    stored counts. The two can differ because Steps 1-7 work on the counts (per day, say) while a
    point-process model needs event-level timing; rebuilding here means the count branch sees counts at
    the right resolution even if the ingest layer initially bucketed them wider.
    """
    ds = session.state.dataset
    if ds is not None and ds.data_class is DataClass.EVENT_TIMESTAMP:
        times = session.objects.get("event_times")
        if times is not None and len(times) >= 3:
            return _counts_from_events(pd.DatetimeIndex(times))
    series = session.objects.get("series")
    if series is None:
        raise ChronosError(
            plain="There is no series to model yet.",
            fix="Load a dataset in Step 0 first.",
        )
    return pd.Series(series).astype(float)


def _counts_from_events(times: pd.DatetimeIndex) -> pd.Series:
    """
    Event timestamps → a count series at the finest resolution that still gives an analysable sample.

    Daily for short spans, monthly for long ones: 400 events over six years are useless as a daily
    series (mostly zeros, and no season to speak of), but perfectly ordinary as a monthly count. The
    threshold is the one place this choice is made, so it is stated rather than buried.
    """
    t = pd.DatetimeIndex(sorted(pd.DatetimeIndex(times).dropna()))
    span_days = (t[-1] - t[0]).days if t.size > 1 else 0
    freq = "MS" if span_days > 730 else "D"
    counts = pd.Series(1.0, index=t).resample(freq).sum().fillna(0.0)
    counts.index.freq = None
    try:
        counts.index = pd.DatetimeIndex(counts.index)
        counts = counts.asfreq(pd.tseries.frequencies.to_offset(freq), fill_value=0.0)
    except Exception:  # pragma: no cover - defensive
        pass
    return counts.astype(float)


def _frequency_code(session: Session) -> str | None:
    """The calendar code ("W", "MS"), which is what tells a point process how long a period is."""
    ds = session.state.dataset
    freq = getattr(ds, "frequency", None) if ds is not None else None
    return getattr(freq, "code", None)


def _period(session: Session) -> int | None:
    ds = session.state.dataset
    if ds is None or ds.frequency is None:
        return None
    p = ds.frequency.period
    return int(p) if p and p >= 2 else None


def _d_hint(session: Session) -> tuple[int, int, str]:
    """
    Differencing orders, from Step 5 — never from the plot.

    `d` is 1 when Step 5 found a unit root (or when a difference has already been applied, in which
    case the series in hand is differenced and the model must not difference again). `D` follows the
    seasonal verdict. The third return value is the sentence used in the advisor's `because`, so the
    number's provenance is visible where the number is.
    """
    verdict = session.objects.get("stat_verdict") or {}
    v = str(verdict.get("verdict") or "")
    d = 1 if v.startswith("unit_root") or v in {"unit_root", "unit_root_or_break"} else 0
    D = 1 if v == "seasonal_unit_root" else 0
    steps = session.state.preprocessing
    applied_d = [p for p in steps if p.step == "difference"]
    note_bits: list[str] = []
    if applied_d:
        lag1 = [p for p in applied_d if (p.params or {}).get("lag", 1) in (1, None)]
        seasonal = [p for p in applied_d if (p.params or {}).get("lag", 1) not in (1, None)]
        d = 0 if lag1 else d
        D = 0 if seasonal else D
        note_bits.append(
            f"{len(applied_d)} difference(s) are already applied to the series, so the model must "
            "not difference again"
        )
    if not verdict:
        note_bits.append("Step 5 has not run, so d = 0 and the fit assumes stationarity")
    else:
        note_bits.append(f"Step 5 verdict: {v.replace('_', ' ')}")
    return d, D, "; ".join(note_bits)


def _data_class(session: Session, series: pd.Series) -> tuple[DataClass, str]:
    ds = session.state.dataset
    declared = ds.data_class if ds is not None else DataClass.CONTINUOUS
    return classify_data_class(series, declared)


def _facts(session: Session):
    series = _series(session)
    d, D, _ = _d_hint(session)
    data_class, _ = _data_class(session, series)
    eda = session.objects.get("eda_facts") or {}
    decomp = session.objects.get("decomposition") or {}
    battery = session.objects.get("stat_battery") or {}
    horizons = session.objects.get("forecast_horizon")
    f = build_facts(
        series,
        data_class=data_class,
        period=_period(session),
        seasonal_strength=decomp.get("seasonal_strength") or eda.get("seasonal_strength"),
        trend_r2=eda.get("trend_r2"),
        stationarity_verdict=(session.objects.get("stat_verdict") or {}).get("verdict"),
        d_hint=d,
        D_hint=D,
        n_breaks=sum(len(t.get("breaks") or []) for t in (battery.get("breaks") or [])),
        transform_stack=[p.step for p in session.state.preprocessing],
        horizon=int(horizons) if horizons else None,
        n_exogenous=len(session.state.dataset.exogenous_columns) if session.state.dataset else 0,
        available_features={mid: True for mid in IMPLEMENTED},
    )
    return f


def _exog(session: Session) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    The driver columns, past and future.

    Two things are returned: the observed drivers (as a frame aligned to the target) and the *future*
    paths if the user has supplied any. Future paths live in `session.objects["future_exogenous"]`
    and are written by the scenario endpoint; when they are absent the exogenous models hold each
    driver at its last value and label that as a scenario, which is the honest default.
    """
    ex = session.objects.get("exogenous")
    if ex is None:
        return None, None
    past = pd.DataFrame(ex)
    if past.shape[1] == 0:
        return None, None
    future = session.objects.get("future_exogenous")
    fut = pd.DataFrame(future) if future is not None else None
    return past, fut


def _interventions(session: Session) -> list[dict[str, Any]]:
    """
    Pulse/step candidates from Step 2's outlier flags.

    Only flagged points become dummies, and only the largest few: an intervention list that grows to
    cover every unusual point is the model explaining its own residuals. Step 5's break tests decide
    whether a flag is a pulse or a step; where they did not run, a pulse is assumed and the fit's
    caveat says so.
    """
    flags = session.objects.get("outliers") or session.objects.get("eda_outliers") or []
    if isinstance(flags, dict):
        flags = flags.get("outliers") or flags.get("points") or []
    series = session.objects.get("series")
    index = list(pd.Series(series).index) if series is not None else []
    pos_of = {t: i for i, t in enumerate(index)}
    breaks = session.objects.get("stat_battery") or {}
    step_positions = set()
    for b in (breaks.get("breaks") or []):
        at = b.get("break_index")
        if isinstance(at, int):
            step_positions.add(at)
    out: list[dict[str, Any]] = []
    for item in flags[:8]:
        t = item.get("time")
        pos = None
        if isinstance(t, str):
            try:
                pos = pos_of.get(pd.Timestamp(t))
            except Exception:
                pos = None
        elif isinstance(item.get("position"), int):
            pos = int(item["position"])
        if pos is None:
            continue
        out.append({"position": pos, "kind": "step" if pos in step_positions else "pulse"})
    return out


# --------------------------------------------------------------------------- #
# S9: suggestions
# --------------------------------------------------------------------------- #
@router.get("/analysis/registry")
def registry_rows() -> dict[str, Any]:
    """The registry, with an honest flag for what this build can and cannot fit."""
    rows = []
    for s in all_specs():
        rows.append(
            {
                "model_id": s.model_id,
                "display_name": s.display_name,
                "family": s.family.value,
                "is_baseline": s.is_baseline,
                "implemented": is_implemented(s.model_id),
                "min_n": s.required.min_n,
                "requires_count": s.required.requires_count,
                "supports_seasonality": s.capabilities.supports_seasonality,
                "supports_exogenous": s.capabilities.supports_exogenous,
                "supports_probabilistic": s.capabilities.supports_probabilistic,
                "discrete_predictive": s.capabilities.discrete_predictive,
                "advanced": s.feature_flag == "advanced",
                "planned_part": s.planned_part,
                "triggered_when": s.triggered_when,
                "assumption_checks": s.assumption_checks,
                "plain_description": s.plain_description,
            }
        )
    return {
        "n_rows": len(rows),
        "n_implemented": len(implemented_specs()),
        "rows": rows,
    }


@router.get("/analysis/suggest")
def get_suggestions(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    The S9 candidate set. Cheap enough to compute on every request: it is a filter, not a fit.
    """
    session.require_stage(Stage.S9_SUGGEST_MODEL, allow_stale=True)
    s = suggest(_facts(session))
    session.objects[SUGGESTIONS] = s.model_dump(mode="json")
    return s.model_dump(mode="json")


@router.post("/analysis/suggest", response_model=AnalysisResponse)
def run_suggest(session: Session = Depends(session_dep)) -> AnalysisResponse:
    """Run S9 and record it as a stage, with a table the user can read and a ranked advisor."""
    t0 = time.perf_counter()
    session.require_stage(Stage.S9_SUGGEST_MODEL, allow_stale=True)
    f = _facts(session)
    s = suggest(f)
    session.objects[SUGGESTIONS] = s.model_dump(mode="json")

    table = _suggestion_table(s)
    session.add_result(table)
    session.add_result(_suggestion_narrative(s, f))
    session.mark_stage(
        Stage.S9_SUGGEST_MODEL, "complete",
        f"{len(s.suggestions)} candidates from {s.n_registry} registry rows",
    )
    return AnalysisResponse(
        stage=Stage.S9_SUGGEST_MODEL,
        advisor=_suggest_advisor(s, f),
        results=[table],
        pipeline=session.state.progress(),
        caveats=s.caveats,
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


# --------------------------------------------------------------------------- #
# S10: fitting
# --------------------------------------------------------------------------- #
class FitRequest(BaseModel):
    model_ids: list[str] = Field(default_factory=list, description="Empty = the suggested set")
    allow_baselines_omitted: bool = False
    combination_method: str = Field(
        "mean",
        description="Which rule the `combination` row uses: mean, median, ic_weighted or "
        "granger_ramanathan. Ignored unless `combination` is in `model_ids`.",
    )
    combination_horizon: int | None = Field(
        None,
        description="Horizon the combination's weights are estimated over. Defaults to 12.",
    )


class ModelIdRequest(BaseModel):
    """A request that names one stored fit — used where the fit id is the whole input."""

    model_id: str


class ForecastRequest(BaseModel):
    model_id: str
    horizon: int = 12
    alpha: float = 0.05


@router.post("/analysis/fit", response_model=AnalysisResponse)
def run_fit(body: FitRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    """
    Fit the candidates.

    Refused outright when the gate says `not_ready` — see the module docstring. `inconclusive` is
    *not* refused, because an unassessed check is a gap in the record rather than evidence against
    modelling; the gate's caveats travel into the fit table instead.
    """
    t0 = time.perf_counter()
    session.require_stage(Stage.S10_FIT, allow_stale=True)
    states = session.state.stages
    if states[Stage.S9_SUGGEST_MODEL.value].status not in {"complete", "stale"}:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain="The model suggestions have not been run, so there is no candidate list to fit.",
            fix="Open Step 9 (Suggest) and run it first — it is the step that decides which models apply.",
        )

    gate = _gate_check(session)
    stored = session.objects.get(SUGGESTIONS) or {}
    ids = list(body.model_ids) or list(stored.get("default_fit_ids") or [])
    unknown = [i for i in ids if i not in IMPLEMENTED]
    if unknown:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain=f"{', '.join(unknown)} cannot be fitted in this build.",
            fix="Only models listed as implemented in Step 9 can be fitted. "
                f"Available: {', '.join(sorted(IMPLEMENTED))}.",
            context={"requested": ids, "unknown": unknown},
        )

    series = _series(session)
    d, D, d_note = _d_hint(session)
    data_class, class_note = _data_class(session, series)
    exog, future_exog = _exog(session)
    from app.ts.models.combination import METHODS as _COMBO_METHODS

    if body.combination_method not in _COMBO_METHODS:
        from app.core.errors import StageOrderError

        raise StageOrderError(
            plain=f"'{body.combination_method}' is not a combination rule this build knows.",
            fix=f"Use one of: {', '.join(_COMBO_METHODS)}.",
            context={"combination_method": body.combination_method, "known": list(_COMBO_METHODS)},
        )

    outcomes = fit_many(
        ids, series, period=_period(session), d_hint=d, D_hint=D, data_class=data_class,
        exog=exog, future_exog=future_exog, interventions=_interventions(session),
        event_times=session.objects.get("event_times"),
        frequency_code=_frequency_code(session),
        combination_method=body.combination_method,
        combination_horizon=body.combination_horizon,
    )

    session.objects[FITS] = {
        "version": session.state.series_version,
        "at": time.time(),
        "outcomes": outcomes,
        "data_class": data_class.value,
        "d": d,
        "D": D,
        "period": _period(session),
    }

    table = _fit_table(outcomes)
    narrative_results = [_fit_diagnostic_table(outcomes)]
    session.add_result(table)
    for r in narrative_results:
        session.add_result(r)

    n_ok = sum(1 for o in outcomes if not o.summary.failed)
    session.mark_stage(
        Stage.S10_FIT, "complete",
        f"{n_ok}/{len(outcomes)} models fitted; best AICc "
        + (best_by_aicc(outcomes)[0].summary.display_name if n_ok else "none"),
    )

    caveats = [c for o in outcomes if not o.summary.failed for c in o.summary.caveats]
    return AnalysisResponse(
        stage=Stage.S10_FIT,
        advisor=_fit_advisor(outcomes, d_note, class_note, gate),
        results=[table, *narrative_results],
        pipeline=session.state.progress(),
        caveats=list(dict.fromkeys(caveats)),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.get("/analysis/fits")
def get_fits(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    The stored fits, ranked, with every number the S11 comparison needs.

    Returns an empty `status` rather than an error when nothing has been fitted: a GET that 404s is
    a GET the UI has to special-case, and special cases are where the "panel is blank" bugs live.
    """
    stored = session.objects.get(FITS)
    if not stored:
        return {
            "stage": Stage.S10_FIT.value,
            "status": "not_started",
            "fits": [],
            "message": "No models have been fitted yet.",
        }
    outcomes: list[FitOutcome] = stored["outcomes"]
    ranked = best_by_aicc(outcomes)
    return {
        "stage": Stage.S10_FIT.value,
        "status": "complete",
        "stale": stored["version"] != session.state.series_version,
        "computed_for_version": stored["version"],
        "current_version": session.state.series_version,
        "data_class": stored["data_class"],
        "d": stored["d"],
        "D": stored["D"],
        "period": stored["period"],
        "failure_summary": summarise_failures(outcomes),
        "fits": [_fit_row(o) for o in ranked],
        "all": [_fit_row(o) for o in outcomes],
    }


@router.post("/analysis/fit/forecast")
def fit_forecast(body: ForecastRequest, session: Session = Depends(session_dep)) -> dict[str, Any]:
    """Forecast from one stored fit. Used by the comparison chart and, later, by S13."""
    stored = session.objects.get(FITS)
    if not stored:
        raise ChronosError(
            plain="No model has been fitted, so there is nothing to forecast from.",
            fix="Open Step 10 (Fit) and fit at least one model.",
        )
    match = [o for o in stored["outcomes"] if o.summary.model_id == body.model_id]
    if not match:
        raise ChronosError(
            plain=f"{body.model_id} is not among the fitted models.",
            fix="Fit it in Step 10 first, or choose one of the models already fitted.",
        )
    o = match[0]
    if o.summary.failed:
        raise ChronosError(
            plain=f"{o.summary.display_name} did not fit, so it cannot forecast.",
            fix=o.summary.failure_reason or "Choose another model.",
        )
    if body.horizon < 1 or body.horizon > 1000:
        raise ChronosError(
            plain=f"A horizon of {body.horizon} is not usable.",
            fix="Choose a horizon between 1 and 1000 periods.",
        )
    # The exogenous models read the scenario at forecast time, so the path the UI just set is the path
    # this forecast uses. Passing it here (rather than only at fit time) means a scenario change does
    # not silently require a re-fit and does not silently do nothing either.
    exog, future_exog = _exog(session)
    if getattr(o.wrapper, "supports_exog", False):
        o.wrapper._exog_raw = exog
        o.wrapper._future_raw = future_exog
        o.wrapper._exog = o.wrapper._prepare_exog(pd.Series(_series(session)).index)
    fc = o.wrapper.predict(body.horizon, alpha=body.alpha)
    idx = _future_index(session, body.horizon)
    # A volatility fit answers a question no mean model answers: how uncertain is each step. The
    # variance path ships alongside the mean path rather than instead of it, because the spec's Part 11
    # asks for "two forecasts" and a reader who is shown only one of them would take the flat mean line
    # for the model's whole opinion.
    variance_path = None
    if hasattr(o.wrapper, "variance_forecast"):
        try:
            vp = o.wrapper.variance_forecast()  # type: ignore[attr-defined]
            variance_path = {
                "variance": [float(v) for v in vp["variance"][: max(body.horizon, 1)]],
                "sd": [float(v) for v in vp["sd"][: max(body.horizon, 1)]],
                "unconditional_sd": vp.get("unconditional_sd"),
                "persistence": getattr(o.wrapper, "persistence", None),
                "leverage": _leverage(o.wrapper),
            }
        except Exception as exc:  # pragma: no cover - defensive: a variance path is an extra, not the forecast
            variance_path = {"error": f"The variance path could not be produced: {exc}"}
    scenario_note = ""
    if getattr(o.wrapper, "supports_exog", False):
        sc = o.wrapper.scenario_caveat
        scenario_note = sc or ""
    return {
        "scenario": scenario_note,
        "model_id": body.model_id,
        "display_name": o.summary.display_name,
        "horizon": body.horizon,
        "alpha": body.alpha,
        "index": [str(i) for i in idx],
        "mean": [float(v) for v in fc["mean"].to_numpy()],
        "lower": [float(v) for v in fc["lower"].to_numpy()],
        "upper": [float(v) for v in fc["upper"].to_numpy()],
        "caveats": o.summary.caveats,
        # The discrete predictive, when the model has one. Parts 12 and 15 read CRPS, PIT and the
        # stocked-out probability off this array — shipping the PMF is what makes those metrics
        # reproducible from the stored result instead of only from a re-run.
        "predictive": (
            o.wrapper.predictive().as_dict()  # type: ignore[attr-defined]
            if hasattr(o.wrapper, "predictive") and o.wrapper.predictive() is not None  # type: ignore[attr-defined]
            else None
        ),
        "discrete": o.summary.model_id in DISCRETE_PREDICTIVE,
        "variance_path": variance_path,
    }


def _leverage(wrapper: Any) -> dict[str, Any] | None:
    """The asymmetry coefficient, when the fitted model has one, with the direction spelled out."""
    spec = getattr(wrapper, "_model_spec", None)
    theta = getattr(wrapper, "_var_theta", None)
    if spec is None or theta is None or getattr(spec, "variance", None) not in {"gjr", "egarch"}:
        return None
    gamma = float(theta[2])
    if spec.variance == "gjr":
        active = gamma > 0
        reading = (
            "negative shocks raise next-period variance more than positive ones"
            if active
            else "no leverage: the asymmetry term is not positive, so falls and rises move variance alike"
        )
    else:
        active = gamma < 0
        reading = (
            "negative shocks raise log-variance, the leverage effect"
            if active
            else "no leverage: gamma is not negative, so the sign of a shock does not change the response"
        )
    return {"gamma": gamma, "active": bool(active), "reading": reading}


# --------------------------------------------------------------------------- #
# Helpers: gate, index, tables
# --------------------------------------------------------------------------- #
class ScenarioRequest(BaseModel):
    """Future paths for the driver columns, plus the horizon they describe."""

    horizon: int = Field(..., ge=1, le=240)
    paths: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Column name → one value per forecast step. Omit a column to hold it at its last "
        "observed value, which the fit reports as a scenario rather than a prediction.",
    )


@router.post("/analysis/fit/multivariate")
def fit_multivariate(body: ModelIdRequest, session: Session = Depends(session_dep)) -> dict[str, Any]:
    """
    The Granger table, impulse responses, FEVD and (for a VECM) the long-run coefficients.

    Split out from `/analysis/fit/forecast` because it is a different question: that endpoint answers
    "where is the series going", this one answers "how do these series move each other". It is a POST
    with a model id rather than a GET with a query parameter so the stored-fit lookup has the same
    shape as every other fit endpoint.
    """
    stored = session.objects.get(FITS)
    if not stored:
        raise ChronosError(
            plain="No model has been fitted, so there is no multivariate result to show.",
            fix="Open Step 10, fit the VAR (it needs a second mapped numeric column).",
        )
    match = [o for o in stored["outcomes"] if o.summary.model_id == body.model_id]
    if not match:
        raise ChronosError(
            plain=f"{body.model_id} is not among the fitted models.",
            fix="Fit it in Step 10 first.",
        )
    o = match[0]
    if not hasattr(o.wrapper, "multivariate_payload"):
        raise ChronosError(
            plain=f"{o.summary.display_name} is not a multivariate model, so it has no impulse "
                  "responses or variance decomposition.",
            fix="Choose the VAR or VECM row — those are the ones that model several series together.",
        )
    if o.summary.failed:
        raise ChronosError(
            plain=f"{o.summary.display_name} did not fit, so it has no multivariate result.",
            fix=o.summary.failure_reason or "Choose another model.",
        )
    payload = o.wrapper.multivariate_payload()  # type: ignore[attr-defined]
    return {"status": "complete", **payload, "caveats": o.summary.caveats}


@router.post("/analysis/scenario", response_model=AnalysisResponse)
def set_scenario(body: ScenarioRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    """
    Record the future driver paths the exogenous models will forecast under.

    Refused when the dataset has no drivers, because accepting the request and silently ignoring it
    would leave the user believing a scenario had been applied when the model was still holding every
    driver flat. Extra columns not in the dataset are refused for the same reason.
    """
    exog, _ = _exog(session)
    if exog is None:
        raise ChronosError(
            plain="This dataset has no outside columns, so there is no scenario to set.",
            fix="Map an extra numeric column as exogenous in Column mapping, or use a model that "
                "needs only the series.",
            code="models.no_exogenous",
        )
    known = set(exog.columns)
    unknown = [c for c in body.paths if c not in known]
    if unknown:
        raise ChronosError(
            plain=f"'{', '.join(unknown)}' is not a driver column in this dataset.",
            fix=f"Use one of: {', '.join(sorted(known))}.",
            code="models.unknown_driver",
        )
    for col, vals in body.paths.items():
        if len(vals) != body.horizon:
            raise ChronosError(
                plain=f"The path for '{col}' has {len(vals)} values but the horizon is {body.horizon}.",
                fix="Supply exactly one value per forecast step, or omit the column to hold it flat.",
                code="models.path_length",
            )
    frame = pd.DataFrame({c: pd.Series(v) for c, v in body.paths.items()})
    session.objects["future_exogenous"] = frame
    held = sorted(known - set(body.paths))

    rows = [
        {
            "driver": c,
            "scenario": ("supplied path" if c in body.paths else "held at last value"),
            "mean": float(pd.Series(body.paths[c]).mean()) if c in body.paths
            else float(pd.Series(exog[c]).iloc[-1]),
            "n_steps": body.horizon,
        }
        for c in sorted(known)
    ]
    table = TableResult(
        id="models.scenario",
        stage=Stage.S10_FIT,
        title="The scenario every exogenous forecast now assumes",
        plain=(
            "These are the driver values the next forecast will be conditional on. A driver held at "
            "its last value is an assumption, not a forecast, and it is listed as one."
        ),
        why=(
            "An exogenous model's output is only as good as its inputs. Stating the inputs — including "
            "the ones left to default — is what separates a scenario from a claim."
        ),
        method="session.objects['future_exogenous']; consumed by SARIMAX and ARDL at predict time.",
        columns=[
            {"key": "driver", "label": "Driver", "type": "text"},
            {"key": "scenario", "label": "How it was set", "type": "text"},
            {"key": "mean", "label": "Average over the horizon", "type": "number", "format": "0.00"},
            {"key": "n_steps", "label": "Steps", "type": "number", "format": "0"},
        ],
        rows=rows,
        reproducible_snippet=(
            "future = {'PromoDays': [4, 4, 0, 7, 3, 0]}\n"
            "fit.predict(horizon=6, exog=future)   # conditional on this path, not a prediction"
        ),
    )
    session.add_result(table)
    return AnalysisResponse(
        stage=Stage.S10_FIT,
        advisor=AdvisorReport(
            stage=Stage.S10_FIT,
            headline=f"Scenario set for {len(known)} driver column(s).",
            recommendations=[
                RuleOut(
                    id="scenario.conditional",
                    stage=Stage.S10_FIT,
                    recommend="Read every exogenous forecast as conditional on this scenario",
                    confidence=Confidence.MEDIUM,
                    because=(
                        f"{body.horizon} forecast steps with {len(body.paths)} supplied path(s) and "
                        f"{len(held)} driver(s) held flat."
                    ),
                    provenance=["future_exogenous", "exogenous"],
                    alternative="Supply a path for the held-flat columns to remove the assumption.",
                    risk_if_ignored=(
                        "A held-flat driver is the most common way an exogenous forecast misleads: the "
                        "model is right about the relationship and wrong about the world."
                    ),
                    how_to="POST /analysis/scenario with a path for every driver, then re-forecast.",
                )
            ],
            facts_used={
                "horizon": body.horizon,
                "n_drivers": len(known),
                "n_supplied": len(body.paths),
                "n_held_flat": len(held),
            },
            confidence=Confidence.MEDIUM,
            next_stage_ready=True,
            narrative=(
                f"{len(body.paths)} of {len(known)} driver column(s) have a supplied path; "
                f"{len(held)} held at the last observed value"
                + (f" ({', '.join(held)})." if held else ".")
            ),
        ),
        results=[table],
        pipeline=session.state.progress(),
        caveats=(
            [f"'{c}' is held at its last observed value for the whole horizon — a scenario, not a "
             "forecast of the driver." for c in held]
        ),
        duration_ms=0,
    )


def _gate_check(session: Session):
    stored = session.objects.get("gate_verdict")
    if not stored:
        raise ChronosError(
            plain="The readiness gate has not been run.",
            fix="Open Step 8 and assess readiness — fitting a model to a series nobody has checked "
                "is how a wrong number becomes a confident one.",
        )
    verdict = stored["payload"]["verdict"]
    if verdict == "not_ready":
        from app.core.errors import StageOrderError

        blockers = [b for b in stored["payload"].get("blockers", [])]
        # A bare checklist label reads backwards when quoted as a reason: "not ready to model:
        # Enough observations to fit a seasonal model". Each blocker is named with the number it
        # failed on, so the refusal is a sentence about the data rather than a fragment of a table.
        reasons = "; ".join(
            f"{b.get('label', 'a readiness check')} — {b.get('value', 'no number recorded')}"
            for b in blockers
        )
        raise StageOrderError(
            plain=(
                "The readiness gate says this series is not ready to model: "
                + (reasons or "a blocker has not been cleared")
                + "."
            ),
            fix="Clear the blockers listed in Step 8, then re-assess and return here.",
            context={"blockers": [b.get("key") for b in blockers]},
        )
    return stored["payload"]


def _future_index(session: Session, horizon: int) -> list[Any]:
    series = _series(session)
    ds = session.state.dataset
    try:
        freq = series.index.freq or (pd.infer_freq(series.index) if len(series) > 2 else None)
        last = series.index[-1]
        return list(pd.date_range(start=last, periods=horizon + 1, freq=freq)[1:])
    except Exception:  # pragma: no cover - defensive
        return list(range(1, horizon + 1))


def _suggestion_table(s) -> TableResult:
    rows = []
    for x in s.suggestions:
        rows.append(
            {
                "rank": x.rank + 1,
                "model": x.display_name,
                "model_id": x.model_id,
                "family": x.family.value if hasattr(x.family, "value") else str(x.family),
                "because": x.because,
                "would_win_if": x.would_win_if,
                "warnings": " ".join(x.assumption_warnings),
                "order": x.order_hint or "",
                "recommended": "yes" if x.recommended else "",
            }
        )
    return TableResult(
        id="models.suggestions",
        stage=Stage.S9_SUGGEST_MODEL,
        title="Candidate models, and why each one is here",
        plain=(
            "Every model that applies to this series, in the order it should be considered. The "
            "column on the right says what would have to be true for each to win."
        ),
        why=(
            "A ranked list with no reasons is a guess with a number in front of it. Each row states "
            "the measurement that put the model on the list and the condition that would make it the "
            "best choice."
        ),
        method="app/ts/models/suggest.py: registry filter over app/ts/models/facts.py measurements.",
        reproducible_snippet=(
            "from chronos.models.registry import REGISTRY, evaluate_spec\n"
            "from chronos.models.suggest import suggest\n\n"
            "facts = measure(series)            # n, zeros, dispersion, autocorrelation, ...\n"
            "candidates = suggest(facts)        # filter the registry rows against the facts\n"
            "for c in candidates.suggestions:\n"
            "    print(c.model_id, '->', c.because)"
        ),
        inputs={"data_class": s.data_class.value, "n_candidates": len(s.suggestions),
                "n_registry": s.n_registry},
        columns=[
            TableColumn(key="rank", label="#", dtype="integer", align="right"),
            TableColumn(key="model", label="Model"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="because", label="Why it is suggested"),
            TableColumn(key="would_win_if", label="Would win if"),
            TableColumn(key="warnings", label="Watch out for"),
            TableColumn(key="order", label="Order implied"),
        ],
        rows=rows,
        caption=(
            f"{len(rows)} candidates from {s.n_registry} registry rows"
            f" ({len(s.excluded)} excluded — see the excluded list for the reason each was dropped)"
        ),
        footnote=(
            "Baselines are always included: without the naive forecast there is no way to tell "
            "whether a model earned its complexity."
        ),
    )


def _suggestion_narrative(s, f) -> TableResult:
    """The excluded list, as its own table — 'why can't I use X?' deserves an answer."""
    from app.core.schemas import NarrativeResult

    rows = [
        {
            "model": e.display_name,
            "family": e.family.value if hasattr(e.family, "value") else str(e.family),
            "reason": " ".join(e.incompatibilities),
        }
        for e in s.excluded
    ]
    return TableResult(
        id="models.excluded",
        stage=Stage.S9_SUGGEST_MODEL,
        title="Models that do not apply here, and the number that excluded them",
        plain=(
            "Each row is a model the registry knows about that was not offered for this series, "
            "with the measurement that ruled it out."
        ),
        why=(
            "A candidate list that only shows what is included leaves the user wondering about "
            "everything that is not — and wondering about a model is how people end up fitting it "
            "by hand."
        ),
        method="app/ts/models/registry.py: evaluate_spec() over app/ts/models/facts.py.",
        reproducible_snippet=(
            "for spec in REGISTRY.values():\n"
            "    verdict = evaluate_spec(spec, facts)\n"
            "    if not verdict.compatible:\n"
            "        print(spec.model_id, 'excluded because', '; '.join(verdict.reasons))"
        ),
        columns=[
            TableColumn(key="model", label="Model"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="reason", label="Why it does not apply"),
        ],
        rows=rows,
        caption=f"{len(rows)} excluded models",
    )


def _fit_table(outcomes: list[FitOutcome]) -> TableResult:
    """
    The comparison table. Sorted so that the models whose ICs are comparable come first and the rest
    follow with their own note — a mixed sort would rank a number against a blank.
    """
    ranked = best_by_aicc(outcomes)
    rows = []
    for i, o in enumerate(ranked):
        s = o.summary
        d = o.diagnostics or {}
        rows.append(
            {
                "rank": i + 1,
                "model": s.display_name,
                "model_id": s.model_id,
                "family": s.family.value if hasattr(s.family, "value") else str(s.family),
                "aicc": None if s.aicc is None else round(s.aicc, 2),
                "bic": None if s.bic is None else round(s.bic, 2),
                "k": s.n_params,
                "rmse": None if d.get("rmse") is None else round(float(d["rmse"]), 4),
                "mase": None if d.get("mase") is None else round(float(d["mase"]), 4),
                "ljung_box_p": (
                    None if d.get("ljung_box_p") is None else round(float(d["ljung_box_p"]), 4)
                ),
                "status": "failed" if s.failed else ("benchmark" if s.is_baseline else "fitted"),
                "note": s.failure_reason if s.failed else (s.ic_note or ""),
                "seconds": round(o.seconds, 2),
            }
        )
    n_ok = sum(1 for o in outcomes if not o.summary.failed)
    return TableResult(
        id="models.fit_comparison",
        stage=Stage.S10_FIT,
        title="Every fitted model, ranked by AICc",
        plain=(
            "AICc ranks how well each model explains the data, penalised for how many numbers it "
            "had to estimate. Lower is better, and only models with the same differencing are "
            "comparable."
        ),
        why=(
            "The ranking is a shortlist, not a verdict: in-sample fit cannot tell you which model "
            "will forecast best, and the models at the top are often the most overfitted."
        ),
        method=(
            "AICc = −2ℓ + 2k + 2k(k+1)/(n−k−1); MASE against the naive (or seasonal-naive) "
            "benchmark; Ljung–Box on the residuals."
        ),
        reproducible_snippet=(
            "import numpy as np\n"
            "from statsmodels.tsa.holtwinters import ExponentialSmoothing\n"
            "from statsmodels.tsa.statespace.sarimax import SARIMAX\n\n"
            "ets = ExponentialSmoothing(y, trend='add', seasonal=None).fit()\n"
            "arima = SARIMAX(y, order=(p, d, q), trend='n').fit(disp=False)\n"
            "print('AICc', ets.aicc, arima.aicc)   # lower is better, same d only"
        ),
        inputs={"n_fitted": n_ok, "n_attempted": len(outcomes)},
        columns=[
            TableColumn(key="rank", label="#", dtype="integer", align="right"),
            TableColumn(key="model", label="Model"),
            TableColumn(key="family", label="Family"),
            TableColumn(key="aicc", label="AICc", dtype="number", format="0.00", align="right"),
            TableColumn(key="bic", label="BIC", dtype="number", format="0.00", align="right"),
            TableColumn(key="k", label="Parameters", dtype="integer", align="right"),
            TableColumn(key="rmse", label="RMSE", dtype="number", format="0.000", align="right"),
            TableColumn(key="mase", label="MASE", dtype="number", format="0.000", align="right"),
            TableColumn(
                key="ljung_box_p", label="Ljung–Box p", dtype="number", format="p",
                align="right",
            ),
            TableColumn(key="status", label="Status"),
            TableColumn(key="seconds", label="Fit time (s)", dtype="number", format="0.00",
                        align="right"),
        ],
        rows=rows,
        caption=f"{n_ok} of {len(outcomes)} models fitted",
        footnote=(
            "A dash means the model has no comparable information criterion (Theta, Croston, "
            "ARFIMA) — ranking those by AICc would compare numbers on different scales."
        ),
    )


def _fit_diagnostic_table(outcomes: list[FitOutcome]) -> TableResult:
    """The one-step-ahead accuracy table, which is what a reader can actually interpret."""
    ranked = best_by_aicc(outcomes)
    rows = []
    for o in ranked:
        s, d = o.summary, (o.diagnostics or {})
        if s.failed:
            rows.append(
                {
                    "model": s.display_name,
                    "rmse": None, "mae": None, "mape": None, "mase": None,
                    "resid_sd": None, "lb_p": None, "jb_p": None,
                    "caveat": s.failure_reason,
                }
            )
            continue
        rows.append(
            {
                "model": s.display_name,
                "rmse": None if d.get("rmse") is None else round(float(d["rmse"]), 3),
                "mae": None if d.get("mae") is None else round(float(d["mae"]), 3),
                "mape": None if d.get("mape") is None else round(float(d["mape"]), 2),
                "mase": None if d.get("mase") is None else round(float(d["mase"]), 3),
                "resid_sd": (
                    None if d.get("residual_sd") is None else round(float(d["residual_sd"]), 3)
                ),
                "lb_p": (
                    None if d.get("ljung_box_p") is None else round(float(d["ljung_box_p"]), 4)
                ),
                "jb_p": (
                    None if d.get("jarque_bera_p") is None
                    else round(float(d["jarque_bera_p"]), 4)
                ),
                "caveat": " ".join(d.get("warnings") or [])[:200],
            }
        )
    return TableResult(
        id="models.fit_diagnostics",
        stage=Stage.S10_FIT,
        title="In-sample accuracy and residual checks",
        plain=(
            "How close each model's one-step predictions came, and whether what it left behind "
            "looks like noise."
        ),
        why=(
            "MASE is comparable across series because it divides by the naive forecast's error. "
            "An MASE above 1 means the model is worse than simply repeating the last value."
        ),
        method="app/ts/models/diagnostics.py: RMSE/MAE/MAPE/MASE, Ljung–Box, Jarque–Bera, ARCH-LM.",
        reproducible_snippet=(
            "import numpy as np\n"
            "from statsmodels.stats.diagnostic import acorr_ljungbox\n\n"
            "resid = y - fitted                      # one-step-ahead residuals\n"
            "scale = np.mean(np.abs(np.diff(y)))     # naive denominator for MASE\n"
            "print('MASE', np.mean(np.abs(resid)) / scale)\n"
            "print('Ljung-Box p', acorr_ljungbox(resid, lags=[10], return_df=True)['lb_pvalue'].iloc[0])"
        ),
        columns=[
            TableColumn(key="model", label="Model"),
            TableColumn(key="rmse", label="RMSE", dtype="number", format="0.000", align="right"),
            TableColumn(key="mae", label="MAE", dtype="number", format="0.000", align="right"),
            TableColumn(key="mape", label="MAPE %", dtype="number", format="0.00", align="right"),
            TableColumn(key="mase", label="MASE", dtype="number", format="0.000", align="right"),
            TableColumn(key="resid_sd", label="Residual SD", dtype="number", format="0.000",
                        align="right"),
            TableColumn(key="lb_p", label="Ljung–Box p", dtype="number", format="p",
                        align="right"),
            TableColumn(key="jb_p", label="Jarque–Bera p", dtype="number", format="p",
                        align="right"),
        ],
        rows=rows,
        caption="MAPE is blank wherever it is undefined — counts and intermittent series",
        footnote=(
            "Ljung–Box p above 0.05 is good here: it means nothing is left in the residuals. "
            "Jarque–Bera below 0.05 means the intervals will be too narrow in the tails."
        ),
    )


def _fit_row(o: FitOutcome) -> dict[str, Any]:
    # `to_schema` takes the *outcome*, not its summary — passing the summary crashed every GET on
    # the first fitted session, which is the kind of typo a type checker would have caught and a
    # smoke test did.
    s = to_schema(o)
    row = s.model_dump(mode="json")
    row["seconds"] = round(o.seconds, 3)
    row["diagnostics"] = o.diagnostics
    row["caveats"] = o.summary.caveats
    # §5.8 F's mandatory "tabular-model card": features, lags, validation scheme and hyperparameters,
    # shipped with the fit so the UI never has to guess how an ML row was built. Absent for models that
    # do not construct a design matrix, which is a fact about the model, not missing data.
    card = getattr(o.wrapper, "card", None)
    row["model_card"] = card or None
    return row


def _suggest_advisor(s, f) -> AdvisorReport:
    recs: list[RuleOut] = []
    for x in s.suggestions[:5]:
        recs.append(
            RuleOut(
                id=f"suggest.{x.model_id}",
                stage=Stage.S9_SUGGEST_MODEL,
                recommend=f"Consider {x.display_name}",
                confidence=Confidence.HIGH if x.recommended else Confidence.MEDIUM,
                because=x.because,
                provenance=["model_suggestions", "eda_facts", "stat_verdict", "decomposition"],
                alternative=x.would_win_if,
                risk_if_ignored=(
                    f"Skipping it means never finding out whether {x.model_id} would have won; the "
                    "comparison table is cheap."
                ),
                how_to="Step 9 → tick the models → Fit in Step 10",
            )
        )
    return AdvisorReport(
        stage=Stage.S9_SUGGEST_MODEL,
        headline=s.headline,
        recommendations=recs,
        facts_used={
            "data_class": s.data_class.value,
            "n_candidates": len(s.suggestions),
            "n_excluded": len(s.excluded),
            "n_observations": f.n_non_missing,
        },
        confidence=Confidence.MEDIUM,
        next_stage_ready=bool(s.default_fit_ids),
        narrative=s.plain,
    )


def _fit_advisor(outcomes: list[FitOutcome], d_note: str, class_note: str, gate: dict) -> AdvisorReport:
    ranked = best_by_aicc(outcomes)
    ok = [o for o in ranked if not o.summary.failed]
    best = ok[0] if ok else None
    recs: list[RuleOut] = []
    blockers: list[RuleOut] = []
    if best is not None:
        bench = [o for o in ok if o.summary.is_baseline]
        best_mase = None
        for b in bench:
            m = (b.diagnostics or {}).get("mase")
            if m is not None and (best_mase is None or float(m) < best_mase):
                best_mase = float(m)
        own_mase = (best.diagnostics or {}).get("mase")
        beats = (
            best_mase is None or own_mase is None or float(own_mase) < best_mase
        )
        recs.append(
            RuleOut(
                id="fit.best_by_aicc",
                stage=Stage.S10_FIT,
                recommend=f"Start with {best.summary.display_name}",
                confidence=Confidence.MEDIUM,
                because=(
                    f"Lowest AICc of {len(ok)} fits"
                    + (f" ({best.summary.aicc:.1f})" if best.summary.aicc is not None else "")
                    + (
                        f"; its in-sample MASE is {float(own_mase):.3f} against the best benchmark's "
                        f"{best_mase:.3f}." if own_mase is not None and best_mase is not None else "."
                    )
                ),
                provenance=["models.fit_comparison", "models.fit_diagnostics"],
                alternative="Let Step 11 compare them on a holdout — AICc ranks explanation, not "
                            "forecasting.",
                risk_if_ignored=(
                    "AICc rewards in-sample fit and will happily pick the most overfitted model on "
                    "offer; the holdout comparison in Step 11 is what settles it."
                ),
                how_to="Step 11 → Select",
            )
        )
        if not beats:
            blockers.append(
                RuleOut(
                    id="fit.worse_than_naive",
                    stage=Stage.S10_FIT,
                    recommend="Do not report this model as an improvement",
                    confidence=Confidence.HIGH,
                    because=(
                        f"The best AICc model's in-sample MASE is {float(own_mase):.3f}, which is "
                        f"worse than the best benchmark's {best_mase:.3f} — it fits the sample "
                        "worse than repeating the last value."
                        if own_mase is not None and best_mase is not None else
                        "No comparable MASE is available for the leading model."
                    ),
                    provenance=["models.fit_diagnostics"],
                    risk_if_ignored="An improvement claim that is false is the worst output this app can produce.",
                    how_to="Step 11 → compare on a holdout before choosing",
                )
            )

    failed = [o for o in outcomes if o.summary.failed]
    for o in failed[:3]:
        recs.append(
            RuleOut(
                id=f"fit.failed.{o.summary.model_id}",
                stage=Stage.S10_FIT,
                recommend=f"Inspect why {o.summary.display_name} failed",
                confidence=Confidence.LOW,
                because=o.summary.failure_reason,
                provenance=["models.fit_comparison"],
                alternative="The baselines still stand as benchmarks.",
                risk_if_ignored=(
                    "A model that fails silently shrinks the comparison without shrinking the "
                    "confidence in it."
                ),
                how_to="Read the note column in the comparison table",
            )
        )

    return AdvisorReport(
        stage=Stage.S10_FIT,
        headline=(
            f"{len(ok)} of {len(outcomes)} models fitted; "
            + (f"best AICc is {best.summary.display_name}" if best else "none fitted")
        ),
        recommendations=recs,
        blockers=blockers,
        facts_used={
            "n_fitted": len(ok),
            "n_failed": len(failed),
            "d": d_note[:120],
            "data_class": class_note[:120],
            "gate_verdict": str(gate.get("verdict")),
        },
        confidence=Confidence.MEDIUM,
        next_stage_ready=bool(ok),
        narrative=summarise_failures(outcomes) or (
            "Every requested model fitted. Step 11 compares them on data they have not seen, which "
            "is the only test that matters for forecasting."
        ),
    )


__all__ = ["router"]
