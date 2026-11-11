"""
api/metrics.py — Part 15, S14 evaluation (§5.12, §5.10, §6).

Endpoints:

    GET  /analysis/evaluate     the stored evaluation + the menu the UI may offer
    POST /analysis/evaluate     refit every candidate on rolling-origin training windows and score
                                only the held-out blocks

The S13 forecast is a *forecast*, not an accuracy score: it was fitted to the whole history. S14
therefore never uses its fitted values for accuracy. It replays preprocessing decisions from the
original training windows only, refits fresh wrappers, and scores the out-of-sample blocks. A
poisoned-test audit proves that no test-period observation could have informed d/D, Box–Cox λ,
imputation, outlier dummies, parameters, or future driver paths.

Every card is a `ResultBase` with real backend numbers. Every figure comes from `app/charts/`,
with a caption, generated alt text and a `TableResult` behind it. The S12 adequacy verdict, exact
evaluation window, n, frequency, coverage/construction, benchmark and row-drop counts travel with
every result.
"""

from __future__ import annotations

import math
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field, field_validator

from app.api.deps import session_dep
from app.api.forecast import (
    FORECAST,
    _adequacy,
    _available,
    _frequency,
    _require_fits,
    _shortlist,
    _transform,
    _winner,
)
from app.api.models import FITS, _exog, _frequency_code, _period, _series
from app.charts import evaluation as ev
from app.core.errors import ChronosError
from app.core.formatting import p_number, p_phrase
from app.core.schemas import (
    AdvisorReport,
    AnalysisResponse,
    ChecklistItem,
    Confidence,
    DataClass,
    FigureResult,
    NarrativeResult,
    RuleOut,
    Stage,
    StatResult,
    TableColumn,
    TableResult,
    Verdict,
    VerdictResult,
)
from app.core.session import Session
from app.ts.forecast.backtransform import METHODS as BACK_TRANSFORM_METHODS
from app.ts.forecast.horizons import COVERAGES, advise_horizon
from app.ts.forecast.intervals import MAX_PATHS, MIN_PATHS
from app.ts.metrics.policy import METRIC_LABELS, MetricPolicy, choose_policy
from app.ts.metrics.probabilistic import CRPS_CITATION
from app.ts.metrics.protocols import (
    EvaluationRun,
    TrainingContext,
    _check_records,
    run_evaluation,
)
from app.ts.models.facts import classify_data_class
from app.ts.models.fit import (
    DISCRETE_PREDICTIVE,
    MULTIVARIATE_IDS,
    build_multi_frame,
    build_wrapper,
    days_per_period,
)
from app.ts.select.cv import CVFold, MIN_TRAIN

router = APIRouter(tags=["analysis"])

EVALUATION = "evaluation"
DEFAULT_REPS = 5
MAX_FOLDS = 12
MAX_HORIZON = 60
MIN_FOLDS = 2

_P_METRICS = {
    "pit", "rqr", "randomized_pit", "kupiec", "christoffersen", "mcs_p",
}
_P_KEYS = {
    "p", "ks_p", "cvm_p", "combined_p", "kupiec_p", "christoffersen_p",
    "conditional_p", "adf_p", "mcs_p", "joint_p", "p_value",
}
_INT_KEYS = {
    "fold", "step", "n", "count", "n_train", "n_test", "train_start", "train_end",
    "test_start", "test_end", "covered", "truncated_steps", "violations", "reps",
    "horizon", "n_paths", "folds", "d", "D", "n_missing_test", "coverage_pct",
    "n_quantiles", "n_obs", "n_unique", "n_pooled",
}
_BOOL_KEYS = {"ok", "higher_is_better", "truncated", "applicable", "independence_ok"}
_NUMERIC_KEYS = {
    "raw_value", "relative_rank", "expected_uniform", "mean_width", "empirical",
    "nominal", "coverage", "lambda", "statistic", "hit_rate", "rate", "value",
    "theoretical", "sample", "crps", "pinball", "qlike", "mse_variance",
    "discrete_crps", "pmf_log_score", "mae", "rmse", "mase", "rmsse", "mpe",
    "mape", "smape", "gmrae", "mdrae", "rel_mae", "rel_rmse", "log_score",
    "winkler", "zero_share", "primary_metric_value", "mean_width_95",
}
_LABELS = {
    "model": "Model", "metric": "Metric", "metric_label": "Metric", "key": "Key",
    "status": "Status", "reason": "Why", "note": "Note", "raw_value": "Value",
    "relative_rank": "Relative rank", "higher_is_better": "Higher is better",
    "fold": "Fold", "train_window": "Training window", "test_window": "Test window",
    "zone": "Zone", "d": "d", "D": "D", "adf_p": "ADF p", "lambda": "λ",
    "benchmark": "Benchmark", "decision_digest": "Decision digest",
    "name": "Check", "ok": "Passed", "detail": "Detail",
    "coverage": "Nominal", "coverage_pct": "Nominal %", "empirical": "Empirical",
    "p": "p", "p_display": "p", "covered": "Covered", "n": "n",
    "mean_width": "Mean width", "construction": "Construction", "source": "Source",
    "row_model": "Row model", "col_model": "Column model", "statistic": "Statistic",
    "better": "Lower loss", "plain": "Reading", "test": "Test",
    "model_a": "Model A", "model_b": "Model B", "predicted": "Predicted",
    "actual": "Actual", "count": "Count", "hit_rate": "Hit rate",
    "theoretical": "Theoretical", "sample": "Mean RQR", "combined_p": "Combined p",
    "ks_p": "KS p", "cvm_p": "CvM p", "reps": "Repetitions",
    "stability": "Stability", "caveat": "Caveat", "reading": "Reading",
    "kind": "Kind", "bin": "PIT bin", "expected_uniform": "Uniform expectation",
    "time": "When", "value": "Value", "train_start": "Train start",
    "train_end": "Train end", "test_start": "Test start", "test_end": "Test end",
    "n_train": "n train", "n_test": "n test", "n_missing_test": "Missing test rows",
    "step": "h", "crps": "CRPS", "pinball": "Pinball", "series": "Series",
    "x": "x", "y": "y", "metric_label": "Metric", "model_id": "Model",
    "primary_model": "Primary model", "truncated_steps": "Truncated PMF steps",
    "discrete_crps": "Discrete CRPS", "pmf_log_score": "PMF log score",
    "pit_ks_p": "PIT KS p", "pit_cvm_p": "PIT CvM p",
    "rqr_combined_p": "RQR combined p", "qlike": "QLIKE",
    "mse_variance": "MSE variance", "violations": "VaR violations",
    "kupiec_p": "Kupiec p", "christoffersen_p": "Christoffersen p",
    "conditional_p": "Conditional p", "independence_ok": "Independence OK",
    "row": "Row", "col": "Column", "ci95": "95% CI",
}


# --------------------------------------------------------------------------- #
# JSON / column helpers
# --------------------------------------------------------------------------- #
def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _field(rec: Any, name: str) -> Any:
    return getattr(rec, name, None) if not isinstance(rec, dict) else rec.get(name)


def _cols(*specs: tuple[str, str, str, str | None, str]) -> list[TableColumn]:
    return [
        TableColumn(key=key, label=label, dtype=dtype, format=fmt, align=align)  # type: ignore[arg-type]
        for key, label, dtype, fmt, align in specs
    ]


def _behind_columns(rows: list[dict[str, Any]]) -> list[TableColumn]:
    if not rows:
        return []
    keys = list(dict.fromkeys(k for row in rows for k in row))
    out: list[TableColumn] = []
    for key in keys:
        values = [row.get(key) for row in rows if key in row and row.get(key) is not None]
        if key in _BOOL_KEYS or (values and all(isinstance(v, (bool, np.bool_)) for v in values)):
            dtype, fmt, align = "bool", None, "center"
        elif key in _P_KEYS:
            dtype, fmt, align = "number", "p", "right"
        elif key in _INT_KEYS or (values and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values)):
            dtype, fmt, align = "integer", None, "right"
        elif key in _NUMERIC_KEYS or (values and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in values)):
            dtype, fmt, align = "number", "0.000", "right"
        else:
            dtype, fmt, align = "string", None, "left"
        out.append(TableColumn(
            key=key, label=_LABELS.get(key, key.replace("_", " ").title()),
            dtype=dtype, format=fmt, align=align,  # type: ignore[arg-type]
        ))
    return out


# --------------------------------------------------------------------------- #
# Request / readback facts
# --------------------------------------------------------------------------- #
class EvaluationRequest(BaseModel):
    horizon: int = Field(6, ge=1, le=MAX_HORIZON)
    folds: int = Field(4, ge=MIN_FOLDS, le=MAX_FOLDS)
    scheme: Literal["expanding", "sliding"] = "expanding"
    models: list[str] = Field(default_factory=list)
    include_baselines: bool = True
    coverages: list[float] = Field(default_factory=lambda: list(COVERAGES))
    n_paths: int = Field(MIN_PATHS, ge=MIN_PATHS, le=MAX_PATHS)
    seed: int | None = Field(None, description="Defaults to the session seed, so reruns reproduce.")
    back_transform: Literal["naive", "smearing", "lognormal"] | None = Field(
        None, description="Required when a transform is active. No silent exponentiation."
    )
    reps: int = Field(DEFAULT_REPS, ge=1, le=20)

    @field_validator("coverages")
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
        out = sorted({round(float(c), 4) for c in v})
        if 0.95 not in out:
            out.append(0.95)
        return sorted(set(out))


def _default_adequacy() -> dict[str, Any]:
    return {
        "status": "not_evaluated", "adequacy": "not_evaluated",
        "headline": "Residual adequacy is INCONCLUSIVE: Step 12 has not been run for this fit.",
        "reasons": [], "remedies": ["Run Step 12 (Diagnostics) on this fit."],
        "model_id": "", "source": "S12 /analysis/diagnose", "stale": False,
    }


def _needs_original(records: Sequence[Any], spec: Any) -> tuple[bool, str]:
    if spec is not None:
        return True, spec.describe()
    for rec in records:
        step = str(_field(rec, "step") or "")
        method = str(_field(rec, "method") or "")
        if step == "impute" and method not in {"none", "leave_missing", ""}:
            return True, method
        if step == "difference" or method == "difference":
            # S14 must score the untouched levels, not the changes; each fold
            # re-decides differencing from its own training window.
            return True, method or step
        if method == "delete":
            return True, method
    return False, ""


def _raw_series(session: Session, current: pd.Series, records: Sequence[Any], spec: Any) -> tuple[pd.Series, str]:
    """Recover original-unit values; transform/imputation chains cannot be scored from a processed series."""
    needs, why = _needs_original(records, spec)
    original = session.objects.get("series_original")
    if needs and original is None:
        raise ChronosError(
            plain=f"The series has {why}, but the untouched original series is no longer in memory.",
            fix="Reload the dataset and re-run the pipeline, or undo the preprocessing step before evaluating.",
            code="evaluate.no_original",
            context={"reason": why},
        )
    c = pd.Series(current).astype(float)
    if original is None:
        return c.copy(), "current series (no transform or imputation to replay)"
    o = pd.Series(original).astype(float)
    if c.empty:
        return c.copy(), "series_original"
    if o.index.equals(c.index) and len(o) == len(c):
        raw = o.reindex(c.index)
    elif len(o) >= len(c):
        raw = pd.Series(o.to_numpy(dtype=float)[-len(c):], index=c.index)
    else:
        if needs:
            raise ChronosError(
                plain="The original series is shorter than the modelled series, so its training-only "
                      "decisions cannot be replayed.",
                fix="Reload the dataset and re-run the pipeline before evaluating.",
                code="evaluate.original_mismatch",
            )
        raw = c.copy()
    return raw.astype(float), "series_original"


def _session_facts(session: Session, *, horizon: int, models: list[str] | None = None) -> dict[str, Any]:
    try:
        series = pd.Series(_series(session)).astype(float)
    except ChronosError:
        series = pd.Series(dtype=float)
    n = int(series.size)
    period = _period(session)
    frequency = _frequency(session)
    frequency_code = _frequency_code(session)
    ds = session.state.dataset
    declared = ds.data_class if ds is not None else DataClass.CONTINUOUS
    if isinstance(declared, str):
        try:
            declared = DataClass(declared)
        except ValueError:
            declared = DataClass.CONTINUOUS
    if n:
        data_class, class_note = classify_data_class(series, declared)
    else:
        data_class, class_note = declared, "the series is empty"
    zero_pool = series.iloc[: max(1, int(n * 0.8))].dropna().to_numpy(dtype=float) if n else np.asarray([])
    zero_share = float(np.mean(zero_pool == 0)) if zero_pool.size else 0.0

    fits = session.objects.get(FITS) or {}
    outcomes = fits.get("outcomes") or []
    model_options = [
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
    winner = _winner(session, available) if available else None
    shortlist = _shortlist(session, limit=3)
    vol_ids = [mid for mid, o in available.items() if hasattr(o.wrapper, "variance_forecast")]
    requested = list(dict.fromkeys(models or [])) or shortlist or ([winner] if winner else [])
    discrete = (
        any(mid in DISCRETE_PREDICTIVE for mid in requested)
        or data_class in {DataClass.COUNT, DataClass.EVENT_TIMESTAMP}
    )
    target_kind = (
        "variance"
        if data_class == DataClass.CONTINUOUS and (winner in vol_ids or any(m in vol_ids for m in requested))
        else "level"
    )
    policy = choose_policy(
        data_class=data_class, n=n, period=period, horizon=horizon, zero_share=zero_share,
        probabilistic=True, discrete=discrete, target_kind=target_kind,
    )
    spec, transform_warnings = _transform(session)
    adequacy = _adequacy(session, winner) if winner else _default_adequacy()
    forecast_stored = session.objects.get(FORECAST)
    forecast_horizon = int(getattr(forecast_stored.get("bundle"), "horizon", 0)) if forecast_stored else None
    s13_status = session.state.stages[Stage.S13_FORECAST.value].status
    s13_stale = bool(forecast_stored and forecast_stored.get("version") != session.state.series_version)
    max_h = advise_horizon(max(n, 1), 1, period=period).recommended_max if n else 0
    benchmark_id = "seasonal_naive" if period and int(period) >= 2 else "naive"
    return {
        "series": series, "n": n, "period": period, "frequency": frequency,
        "frequency_code": frequency_code, "data_class_obj": data_class,
        "data_class": data_class.value, "class_note": class_note,
        "zero_share": zero_share, "model_options": model_options, "available": available,
        "winner": winner, "shortlist": shortlist, "vol_ids": vol_ids,
        "requested": requested, "discrete": discrete, "target_kind": target_kind,
        "policy_obj": policy, "policy": policy.as_dict(), "spec": spec,
        "transform_warnings": list(transform_warnings),
        "transform_active": spec.as_dict() if spec else None,
        "transform_required": spec is not None,
        "adequacy": adequacy, "forecast_stored": forecast_stored,
        "forecast_horizon": forecast_horizon, "s13_status": s13_status,
        "s13_stale": s13_stale, "recommended_max_h": max_h,
        "benchmark_id": benchmark_id,
    }


def _options(facts: dict[str, Any]) -> dict[str, Any]:
    pol = facts["policy"]
    default_h = min(6, max(1, int(facts["forecast_horizon"] or 6)))
    default_models = [m for m in [facts["winner"], facts["benchmark_id"]] if m]
    return {
        "models": facts["model_options"],
        "winner": facts["winner"],
        "shortlist": facts["shortlist"],
        "default_models": list(dict.fromkeys(default_models)),
        "coverages": list(COVERAGES),
        "schemes": ["expanding", "sliding"],
        "back_transform_methods": list(BACK_TRANSFORM_METHODS),
        "transform_active": facts["transform_active"],
        "transform_required": facts["transform_required"],
        "transform_warnings": facts["transform_warnings"],
        "min_paths": MIN_PATHS, "max_paths": MAX_PATHS,
        "min_folds": MIN_FOLDS, "max_folds": MAX_FOLDS,
        "min_train": MIN_TRAIN, "min_horizon": 1, "max_horizon": MAX_HORIZON,
        "reps_min": 1, "reps_max": 20,
        "default_horizon": default_h, "default_folds": 4,
        "default_scheme": "expanding", "default_n_paths": MIN_PATHS,
        "default_reps": DEFAULT_REPS,
        "n": facts["n"], "period": facts["period"], "frequency": facts["frequency"],
        "recommended_max_h": facts["recommended_max_h"],
        "data_class": facts["data_class"], "class_note": facts["class_note"],
        "zero_share": facts["zero_share"], "target_kind": facts["target_kind"],
        "discrete": facts["discrete"], "probabilistic": True,
        "benchmark_id": facts["benchmark_id"],
        "policy": pol,
        "metric_labels": pol["metric_labels"],
        "policy_reasons": pol["reasons"], "policy_notes": pol["notes"],
        "banned": pol["banned"],
        "adequacy": facts["adequacy"],
        "s13_status": facts["s13_status"],
        "s13_forecast_horizon": facts["forecast_horizon"],
        "s13_stale": facts["s13_stale"],
        "stage_dependencies": [Stage.S13_FORECAST.value],
        "next_stage_requires": [Stage.S14_EVALUATE.value, Stage.S12_DIAGNOSE.value],
        "evaluation_endpoint": "/analysis/evaluate",
    }


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #
def _count(value: Any) -> int:
    """A non-negative finite count, or zero for a missing/malformed value."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0
    return int(x) if math.isfinite(x) and x >= 0 else 0


def _integrity(report: dict[str, Any]) -> dict[str, Any]:
    """Verify the safeguards that a verdict must never simply assume.

    The report declares how many held-out rows the folds produced. Each model's
    point scorer must account for exactly that number when its finite and
    excluded rows are added back together. This catches a shortened forecast,
    benchmark or actual vector, not just a missing value.
    """
    inp = report.get("inputs") or {}
    point_rows = [v for v in (report.get("point") or {}).values() if isinstance(v, dict)]
    expected = _count(inp.get("n_pooled_test_observations"))
    n_excluded = max((_count(v.get("n_excluded")) for v in point_rows), default=0)
    n_scored = min((_count(v.get("n")) for v in point_rows), default=0)
    rows_accounted = bool(point_rows) and expected > 0 and all(
        _count(v.get("n")) + _count(v.get("n_excluded")) == expected for v in point_rows
    )
    n_missing = _count(inp.get("n_missing_test_observations"))
    failures = [x for x in (report.get("fold_failures") or []) if isinstance(x, dict)]
    failures_named = all(
        all(k in x for k in ("model", "fold", "reason")) for x in failures
    )
    leakage = report.get("leakage") or {}
    checks = [c for c in (leakage.get("checks") or []) if isinstance(c, dict)]
    fit_check_ok = any(
        c.get("name") == "factory_received_training_window_only" and bool(c.get("ok"))
        for c in checks
    )
    used_for_fitting = _count(inp.get("test_observations_used_for_fitting"))
    scored_declared = _count(inp.get("test_observations_scored")) or expected
    scored_by_accuracy = _count((report.get("accuracy") or {}).get("n_scored")) or expected
    out_of_sample = bool(
        expected > 0
        and bool(leakage.get("ok"))
        and fit_check_ok
        and used_for_fitting == 0
        and scored_declared == expected
        and scored_by_accuracy == expected
    )
    benchmark = str(inp.get("benchmark_id") or inp.get("benchmark") or "")
    benchmark_ok = bool(benchmark) and any(_count(v.get("n_scaled")) > 0 for v in point_rows)
    coverage_rows = [
        r for r in (report.get("coverage_construction") or []) if isinstance(r, dict)
    ]
    coverage_ok = bool(coverage_rows) and all(
        r.get("coverage") is not None and bool(r.get("construction")) for r in coverage_rows
    )
    rows_ok = bool(rows_accounted and failures_named)
    detail = (
        f"{n_missing} missing, {n_excluded} excluded, {len(failures)} failed fold(s); "
        f"{n_scored}/{expected} finite rows per model"
    )
    return {
        "expected": expected,
        "n_excluded": n_excluded,
        "n_scored": n_scored,
        "n_missing": n_missing,
        "fold_failures": failures,
        "rows_accounted": rows_accounted,
        "rows_ok": rows_ok,
        "out_of_sample": out_of_sample,
        "benchmark": benchmark,
        "benchmark_ok": benchmark_ok,
        "coverage_ok": coverage_ok,
        "coverage_levels": len(coverage_rows),
        "detail": detail,
        "warning": (
            ""
            if rows_ok else
            f"Row accounting is incomplete: {detail}. Do not quote a model ranking until "
            "every held-out row is either scored or explicitly excluded."
        ),
    }


def _common(
    *,
    run: EvaluationRun,
    report: dict[str, Any],
    adequacy: dict[str, Any],
    policy: MetricPolicy,
    body: EvaluationRequest,
    forecast_payload: dict[str, Any],
    facts: dict[str, Any],
    transform_warnings: list[str],
    raw_source: str,
) -> dict[str, Any]:
    inp = dict(report["inputs"])
    n = int(inp.get("n", 0))
    frequency = str(inp.get("frequency", "unspecified"))
    eval_window = str(inp.get("evaluation_window", "—"))
    integrity = _integrity(report)
    n_missing = integrity["n_missing"]
    n_excluded = integrity["n_excluded"]
    n_fold_failures = len(integrity["fold_failures"])
    n_checks = len(report["leakage"]["checks"])
    coverage_levels = len(report["coverage_construction"])
    primary = str(inp.get("primary_model", ""))
    construction = str((report["probabilistic"].get(primary) or {}).get("construction") or "")
    benchmark = str(inp.get("benchmark_id") or inp.get("benchmark") or "naive")
    zero_share = float(inp.get("zero_share", 0.0))
    dc = str(inp.get("data_class", "continuous"))
    horizon = int(inp.get("horizon", body.horizon))
    folds = int(inp.get("folds", body.folds))
    n_paths = int(inp.get("n_paths", body.n_paths))
    seed = int(inp.get("seed", 0))
    reps = int(inp.get("reps", body.reps))
    inputs = {
        **inp,
        "n": n, "frequency": frequency, "evaluation_window": eval_window,
        "n_excluded": n_excluded, "n_fold_failures": n_fold_failures,
        "n_leakage_checks": n_checks, "coverage_levels": coverage_levels,
        "rows_accounted": integrity["rows_accounted"],
        "out_of_sample_verified": integrity["out_of_sample"],
        "benchmark_verified": integrity["benchmark_ok"],
        "interval_construction": construction,
        "coverage_construction": report["coverage_construction"],
        "benchmark": benchmark, "primary_metric": policy.primary,
        "policy_cluster": policy.cluster, "adequacy": adequacy.get("adequacy"),
        "adequacy_status": adequacy.get("status"), "leakage_ok": bool(report["leakage"]["ok"]),
        "transform_warnings": list(transform_warnings),
        "raw_source": raw_source, "class_note": facts["class_note"],
        "forecast_model_id": forecast_payload.get("model_id"),
        "forecast_horizon": forecast_payload.get("horizon"),
        "forecast_interval_method": forecast_payload.get("interval_method"),
        "forecast_transform": forecast_payload.get("transform"),
        "forecast_test_observations_used": int((forecast_payload.get("inputs") or {}).get("test_observations_used", 0)),
        "s13_source": "POST /analysis/forecast (full-history forecast, NOT an accuracy score)",
        "request": _jsonable(body.model_dump()),
    }
    caveats = list(dict.fromkeys([
        adequacy["headline"],
        f"All accuracy is out-of-sample rolling-origin: n = {n}, frequency {frequency}, exact "
        f"evaluation window {eval_window}. The S13 full-history fit was never scored as accuracy (§6.7).",
        f"Every interval states its coverage and construction ({coverage_levels} nominal levels). "
        "Sharpness is read subject to calibration.",
        (f"No rows were silently dropped: {n_missing} missing test observations, {n_excluded} excluded "
         f"non-finite triples and {n_fold_failures} failed fold(s) are counted; every model "
         f"accounts for {integrity['expected']} held-out rows.")
         if integrity["rows_ok"] else integrity["warning"],
        f"MASE, RMSSE and every relative ratio require the {benchmark} benchmark fitted on each "
        "training window only (§6.4).",
        "No AIC/BIC comparison is made across folds, differencing orders or effective samples; the "
        "DM matrix uses the same out-of-sample targets.",
        *policy.notes,
        *transform_warnings,
        f"Training-only decisions were replayed from {raw_source}; future drivers were held flat at "
        "the last training row, not read from the test period.",
        f"Data class: {dc}; {facts['class_note']}.",
    ]))
    warnings = list(dict.fromkeys([
        *report["warnings"],
        *[f"Fold failure: {x['model']} fold {x['fold']}: {x['reason']}" for x in report["fold_failures"]],
        *transform_warnings,
        *([integrity["warning"]] if integrity["warning"] else []),
    ]))
    snippet = (
        "from app.ts.metrics.protocols import run_evaluation\n"
        "from app.ts.metrics.policy import choose_policy\n"
        f"policy = choose_policy(data_class={dc!r}, n={n}, period={facts['period']!r}, "
        f"horizon={horizon}, zero_share={zero_share:.6f}, probabilistic=True)\n"
        f"# Rolling origin: {folds} folds, {n_paths} seeded paths, seed {seed}, "
        f"{reps} PIT/RQR repetitions.\n"
        "print(policy.headline)\n"
    )
    return {
        "stage": Stage.S14_EVALUATE,
        "inputs": _jsonable(inputs),
        "citations": [
            CRPS_CITATION,
            "Hyndman & Koehler (2006), Another look at measures of forecast accuracy",
            "Diebold & Mariano (1995), Comparing Predictive Accuracy, JBES 13(3)",
            "Clark & West (2007), Tests for equal predictive accuracy in nested models, JBES 25(1)",
            "Mincer & Zarnowitz (1969), Evaluation of economic forecasts",
            "Dunn & Smyth (1996), Randomized quantile residuals",
            "Kupiec (1995); Christoffersen (1998), VaR coverage and independence backtests",
        ],
        "caveats": caveats,
        "warnings": warnings,
        "reproducible_snippet": snippet,
    }


def _table(
    *,
    id: str, title: str, plain: str, why: str, method: str,
    columns: list[TableColumn], rows: list[dict[str, Any]],
    caption: str = "", footnote: str = "", technical: str = "", common: dict[str, Any],
) -> TableResult:
    return TableResult(
        id=id, title=title, plain=plain, why=why, method=method, technical=technical,
        columns=columns, rows=_jsonable(rows), caption=caption, footnote=footnote, **common,
    )


def _figure(
    *,
    chart: Any, id: str, title: str, plain: str, why: str, method: str,
    caption: str, technical: str = "", common: dict[str, Any],
) -> FigureResult:
    behind_rows = _jsonable(list(chart.data_behind))
    behind = TableResult(
        id=f"{id}.data", title=f"Numbers behind {title}",
        plain="Every number drawn in the figure above, as an auditable table.",
        why="A figure that cannot be read as numbers is not auditable.",
        method=method,
        columns=_behind_columns(behind_rows), rows=behind_rows,
        caption=f"Data behind: {title}.", footnote="Generated from the backend evaluation payload.",
        **common,
    )
    return FigureResult(
        id=id, title=title, plain=plain, why=why, method=method, technical=technical,
        figure_json=_jsonable(chart.figure_json), caption=caption,
        alt_text=str(chart.alt_text), data_behind=behind, **common,
    )


def _narrative(
    *,
    id: str, title: str, plain: str, why: str, method: str,
    paragraphs: list[str], numbers_cited: dict[str, float],
    technical: str = "", common: dict[str, Any],
) -> NarrativeResult:
    return NarrativeResult(
        id=id, title=title, plain=plain, why=why, method=method, technical=technical,
        paragraphs=[p for p in paragraphs if p], audience="beginner", ai_drafted=False,
        numbers_cited={k: float(v) for k, v in numbers_cited.items() if v is not None and math.isfinite(float(v))},
        **common,
    )


def _stat(
    *,
    id: str, title: str, plain: str, why: str, method: str,
    value: float, statistic_name: str, interpretation: str,
    p_value: float | None = None, effect_size: float | None = None,
    technical: str = "", band: Literal["none", "weak", "moderate", "strong", "inconclusive"] = "none",
    common: dict[str, Any],
) -> StatResult:
    return StatResult(
        id=id, title=title, plain=plain, why=why, method=method, technical=technical,
        value=float(value), p_value=p_value, statistic_name=statistic_name,
        effect_size=effect_size, interpretation=interpretation, band=band, **common,
    )


def _verdict(
    *,
    headline: str, adequacy: dict[str, Any], report: dict[str, Any],
    common: dict[str, Any],
) -> VerdictResult:
    inp = report["inputs"]
    leakage = report["leakage"]
    checks = leakage["checks"]
    integrity = _integrity(report)
    n_pass = int(sum(bool(c.get("ok")) for c in checks))
    adequacy_ok = adequacy.get("status") == "evaluated"
    out_of_sample_ok = integrity["out_of_sample"]
    benchmark_ok = integrity["benchmark_ok"]
    coverage_ok = integrity["coverage_ok"]
    rows_ok = integrity["rows_ok"]
    hard_fail = (
        not (out_of_sample_ok and bool(leakage["ok"]) and benchmark_ok and rows_ok)
        or bool(integrity["fold_failures"])
    )
    if hard_fail or adequacy.get("adequacy") == "inadequate":
        verdict = Verdict.NOT_READY
    elif not adequacy_ok or not coverage_ok:
        verdict = Verdict.INCONCLUSIVE
    elif report["warnings"] or common["warnings"]:
        verdict = Verdict.READY_WITH_CAVEATS
    else:
        verdict = Verdict.READY
    checklist = [
        ChecklistItem(
            key="out_of_sample", label="Out-of-sample accuracy", passed=out_of_sample_ok,
            value=f"{integrity['expected']} pooled rows across {inp.get('folds', 0)} folds",
            detail=(f"Exact evaluation window {inp.get('evaluation_window', '—')}; S13 fitted values "
                    "were not scored and every fit received a training-only wrapper.")
                   if out_of_sample_ok else
                   "The held-out protocol could not be verified from the fold, fit-check and "
                   "accuracy counts; do not quote these scores as out-of-sample accuracy.",
            jump_to_stage=Stage.S14_EVALUATE,
        ),
        ChecklistItem(
            key="leakage", label="Training-only decision audit", passed=bool(leakage["ok"]),
            value=f"{n_pass}/{len(checks)} checks passed",
            detail="Poisoned test values leave d/D, λ, imputation and outlier decisions unchanged.",
            jump_to_stage=Stage.S14_EVALUATE,
        ),
        ChecklistItem(
            key="benchmark", label="Naive / seasonal-naive benchmark", passed=benchmark_ok,
            value=integrity["benchmark"] or "unavailable",
            detail=("Fitted on each training window before any MASE or relative ratio was computed; "
                    "scaled rows are counted.")
                   if benchmark_ok else
                   "No usable training-window benchmark scale was recorded for the scored rows.",
            jump_to_stage=Stage.S11_SELECT,
        ),
        ChecklistItem(
            key="adequacy", label="S12 residual adequacy", passed=adequacy_ok,
            value=str(adequacy.get("adequacy")), detail=str(adequacy.get("headline")),
            jump_to_stage=Stage.S12_DIAGNOSE,
        ),
        ChecklistItem(
            key="coverage", label="Interval coverage / construction", passed=coverage_ok,
            value=f"{integrity['coverage_levels']} stated levels",
            detail=("Every band carries nominal coverage and construction; width is read after calibration.")
                   if coverage_ok else
                   "At least one band is missing a nominal coverage or construction statement.",
            jump_to_stage=Stage.S13_FORECAST,
        ),
        ChecklistItem(
            key="dropped_rows", label="No silent row drops", passed=rows_ok,
            value=integrity["detail"],
            detail=(f"{len(integrity['fold_failures'])} failed fold(s) are named in the protocol table; "
                    "every model reconciles its finite and excluded rows with the fold count.")
                   if rows_ok else integrity["warning"],
            jump_to_stage=Stage.S14_EVALUATE,
        ),
    ]
    blockers: list[str] = []
    if not adequacy_ok:
        blockers.append("Run Step 12 (Diagnostics) to replace the INCONCLUSIVE adequacy verdict.")
    if not leakage["ok"]:
        blockers.append("The leakage audit failed; re-run after fixing the training-only protocol.")
    if not out_of_sample_ok:
        blockers.append("The out-of-sample protocol failed verification; do not quote a ranking.")
    if not benchmark_ok:
        blockers.append("A training-window benchmark scale was not verified for the scored rows.")
    if not rows_ok:
        blockers.append(integrity["warning"] or "Held-out row accounting is incomplete.")
    if not coverage_ok:
        blockers.append("State nominal coverage and construction for every interval band.")
    if integrity["fold_failures"]:
        blockers.append(f"{len(integrity['fold_failures'])} fold(s) failed; read the protocol table before quoting a ranking.")
    return VerdictResult(
        id="evaluate.verdict", title="Evaluation readiness verdict",
        plain="Out-of-sample accuracy, leakage, benchmark, adequacy and interval disclosure in one checklist.",
        why="A metric table without these safeguards is a number that cannot be trusted.",
        method="§6.7–11 safeguards enforced by ts/metrics/protocols.py and attached to every result.",
        technical="Leakage is a defect. A failed poisoned-test audit, an unverified training-only "
                  "protocol or incomplete row accounting refuses the evaluation verdict.",
        verdict=verdict, headline=headline, checklist=checklist,
        blockers=list(dict.fromkeys(blockers)),
        caveats_travelling=list(common["caveats"]), **common,
    )


# --------------------------------------------------------------------------- #
# Result builders
# --------------------------------------------------------------------------- #
def _metric_columns(policy: dict[str, Any]) -> list[TableColumn]:
    cols = [_cols(("model", "Model", "string", None, "left"))[0]]
    for m in policy["metrics"]:
        label = METRIC_LABELS.get(m, m)
        fmt = "p" if m in _P_METRICS else "0.000"
        cols.append(TableColumn(key=m, label=label, dtype="number", format=fmt, align="right"))
    cols.append(TableColumn(key="hit_rate", label="Hit rate", dtype="number", format="0.000", align="right"))
    return cols


def _primary_value(report: dict[str, Any]) -> tuple[str, float | None, dict[str, Any]]:
    primary = str(report["inputs"].get("primary_model", ""))
    metric = str(report["policy"]["primary"])
    row = next((r for r in report["metric_rows"] if str(r.get("model")) == primary), {})
    try:
        value = float(row.get(metric)) if row.get(metric) is not None else None
        if value is not None and not math.isfinite(value):
            value = None
    except (TypeError, ValueError):
        value = None
    return primary, value, row


def _results(
    *,
    run: EvaluationRun,
    report: dict[str, Any],
    raw: pd.Series,
    adequacy: dict[str, Any],
    policy: MetricPolicy,
    body: EvaluationRequest,
    forecast_payload: dict[str, Any],
    facts: dict[str, Any],
    transform_warnings: list[str],
    raw_source: str,
) -> tuple[list[Any], dict[str, Any]]:
    common = _common(
        run=run, report=report, adequacy=adequacy, policy=policy, body=body,
        forecast_payload=forecast_payload, facts=facts,
        transform_warnings=transform_warnings, raw_source=raw_source,
    )
    inp = report["inputs"]
    n = int(inp["n"])
    frequency = str(inp["frequency"])
    eval_window = str(inp["evaluation_window"])
    horizon = int(inp["horizon"])
    folds = int(inp["folds"])
    primary, primary_value, primary_row = _primary_value(report)
    ranked_all = list(report["accuracy"]["ranked"])
    leader = ranked_all[0] if ranked_all else primary
    policy_dict = report["policy"]
    metric_rows = report["metric_rows"]
    coverage_rows = report["coverage_construction"]
    calibration_rows = report["calibration"]["rows"]
    dm = report["comparative"]["dm"]
    hit = report["comparative"]["hit_rate"]
    comparative = report["comparative"]
    benchmark = str(inp.get("benchmark_id") or inp.get("benchmark") or "naive")
    integrity = _integrity(report)
    n_excluded = integrity["n_excluded"]
    n_missing = integrity["n_missing"]
    n_checks = len(report["leakage"]["checks"])
    results: list[Any] = []

    # 1. Verdict + policy.
    results.append(_verdict(
        headline=(
            f"{policy.headline} {folds} rolling-origin folds scored {inp['n_pooled_test_observations']} "
            f"out-of-sample rows from {eval_window}."
        ),
        adequacy=adequacy, report=report, common=common,
    ))
    results.append(_narrative(
        id="evaluate.policy", title="Why these metrics were chosen",
        plain=f"{policy.headline} The reason is computed from the data class, length and zero share.",
        why="MAPE on a count series and CRPS without a predictive distribution are both meaningless.",
        method="ts/metrics/policy.py chooses one metric cluster from measured facts; no UI toggle promotes a banned metric.",
        paragraphs=[
            policy.headline,
            *policy.reasons,
            *policy.notes,
            f"Banned here: {', '.join(policy.banned) if policy.banned else 'none'}.",
            f"n = {n}, frequency {frequency}, zero share {float(inp['zero_share']):.1%}, "
            f"horizon {horizon}, {folds} folds.",
        ],
        numbers_cited={
            "n": float(n), "horizon": float(horizon), "folds": float(folds),
            "zero_share": float(inp["zero_share"]),
            "n_pooled_test_observations": float(inp["n_pooled_test_observations"]),
        },
        technical="Branches: variance → QLIKE/MSE/VaR; count/event → PMF CRPS/log score/randomised PIT; "
                  "intermittent → MASE/RMSSE/CRPS; short → MASE/MAE; probabilistic → CRPS/pinball/PIT.",
        common=common,
    ))
    policy_table_rows: list[dict[str, Any]] = []
    reason0 = policy.reasons[0] if policy.reasons else policy.headline
    reason_last = policy.reasons[-1] if policy.reasons else policy.headline
    for m in policy.metrics:
        policy_table_rows.append({
            "metric": METRIC_LABELS.get(m, m), "key": m,
            "status": "Primary" if m == policy.primary else "Selected",
            "reason": reason0, "note": policy.headline,
        })
    for m in policy.banned:
        policy_table_rows.append({
            "metric": METRIC_LABELS.get(m, m), "key": m, "status": "Banned",
            "reason": reason_last, "note": "Never promoted into the headline table.",
        })
    results.append(_table(
        id="evaluate.policy.metrics", title="Metric policy: selected and banned",
        plain="Every legal metric and every refusal, with the reason visible.",
        why="A blank cell is ambiguous; a banned metric with a reason is a decision the reader can audit.",
        method="The backend policy object is rendered row by row; the UI invents no metric.",
        columns=_cols(
            ("metric", "Metric", "string", None, "left"),
            ("key", "Key", "string", None, "left"),
            ("status", "Status", "string", None, "left"),
            ("reason", "Why", "string", None, "left"),
            ("note", "Note", "string", None, "left"),
        ),
        rows=policy_table_rows,
        caption=policy.headline,
        footnote=" · ".join(policy.reasons) or "The policy reasons travel with every result.",
        common=common,
    ))

    # 2. Headline stat + metric table + heatmap.
    if primary_value is not None:
        results.append(_stat(
            id="stat.evaluate.primary", title=f"Primary metric: {METRIC_LABELS.get(policy.primary, policy.primary)}",
            plain=f"Primary model {primary} scores {primary_value:.4g} on "
                  f"{METRIC_LABELS.get(policy.primary, policy.primary)}; "
                  f"{leader} ranks first on that metric.",
            why="The policy's leading metric is the one number a reader should see first.",
            method="Pooled out-of-sample rolling-origin errors; missing rows and failed folds are counted, not dropped.",
            value=primary_value, statistic_name=METRIC_LABELS.get(policy.primary, policy.primary),
            interpretation=(
                f"Ranking by {policy.primary}: {', '.join(report['accuracy']['ranked'])}. "
                f"n = {n}, frequency {frequency}, evaluation window {eval_window}."
            ),
            effect_size=None if primary_row.get("rel_mae") is None else float(primary_row.get("rel_mae")),
            technical="Lower is better unless the leading metric is a p-value or hit rate; p-value columns "
                      "use the app-wide p formatting.",
            common=common,
        ))
    results.append(_table(
        id="evaluate.metric_table", title="Out-of-sample metric comparison",
        plain=f"Every candidate scored on the same held-out blocks from {eval_window}.",
        why="A single metric can be gamed; the policy's whole set shows point, distributional and calibration behaviour.",
        method="Pooled OOS errors, analytic interval bands, seeded simulation paths and exact PMFs where available.",
        columns=_metric_columns(policy_dict), rows=metric_rows,
        caption=(f"n = {n} · {frequency} · evaluation {eval_window} · {folds} folds · benchmark {benchmark}. "
                 f"{policy.headline}"),
        footnote="Lower is better for error scores and p-value columns are formatted as p; hit rate is higher-is-better "
                 "and shown for the primary model only. Missing values are refused, not replaced by zeros.",
        technical="MASE = mean(|e_t| / s_fold(t)) with a training-window-only scale; "
                  "RMSSE = sqrt(mean((e_t / s_fold(t))²)); CRPS = E|X−y| − ½E|X−X'|; "
                  "log score = −mean(log p(y)) and is refused outside a PMF's support.",
        common=common,
    ))
    results.append(_figure(
        chart=ev.metric_heatmap(metric_rows, policy=policy_dict,
                                calibration_by_model=report["probabilistic"]),
        id="fig.evaluate.metric_heatmap", title="Metric comparison heatmap",
        plain="Relative rank within each metric: 1 means best among the models actually evaluated.",
        why="Raw metric units differ; a ranking heatmap shows the pattern without pretending the columns are comparable.",
        method="Column-wise min–max rank from backend metric rows; coverage uses absolute deviation from nominal.",
        caption=f"{policy.headline} 1 = best in each column; lower error and higher hit rate are handled separately.",
        common=common,
    ))

    # 3. Protocol, split and leakage.
    results.append(_figure(
        chart=ev.split_chart(raw, report["folds"], n=n, frequency=frequency,
                             evaluation_window=eval_window),
        id="fig.evaluate.split", title="Rolling-origin train / test split",
        plain=f"The shaded zones are out-of-sample only: {eval_window}.",
        why="A train/test split you cannot see is the easiest place to hide leakage.",
        method="fold_origins from S11, with train_end < test_start asserted per fold.",
        caption=(f"n = {n} · {frequency} · {folds} folds · evaluation {eval_window}. "
                 f"Test observations used for fitting: {inp['test_observations_used_for_fitting']}; "
                 f"test observations scored: {inp['test_observations_scored']}."),
        technical="The model parameters, differencing order, Box–Cox λ, imputation and outlier dummies are "
                  "recomputed inside each training window only.",
        common=common,
    ))
    results.append(_table(
        id="evaluate.protocol_folds", title="Rolling-origin folds and training-only decisions",
        plain="Each fold's exact training and test window, benchmark and decision digest.",
        why="An accuracy number without its split is not reproducible.",
        method="Explicit CVFold origins; d/D from a training-window ADF, λ re-estimated on training data only.",
        columns=_cols(
            ("fold", "Fold", "integer", None, "right"),
            ("train_window", "Training window", "string", None, "left"),
            ("test_window", "Test window", "string", None, "left"),
            ("n_train", "n train", "integer", None, "right"),
            ("n_test", "n test", "integer", None, "right"),
            ("zone", "Zone", "string", None, "left"),
            ("d", "d", "integer", None, "right"),
            ("D", "D", "integer", None, "right"),
            ("adf_p", "ADF p", "number", "p", "right"),
            ("lambda", "λ", "number", "0.000", "right"),
            ("benchmark", "Benchmark", "string", None, "left"),
            ("n_missing_test", "Missing test rows", "integer", None, "right"),
            ("decision_digest", "Decision digest", "string", None, "left"),
        ),
        rows=report["folds"],
        caption=f"Out-of-sample-only zones from {eval_window}; {inp['n_unique_test_observations']} unique "
                f"test observations appear in {inp['n_pooled_test_observations']} pooled fold rows.",
        footnote="Overlapping folds are reported as pooled rows, not unique rows. No test value informs a decision digest.",
        common=common,
    ))
    leakage_rows = [
        {"name": str(c.get("name")), "ok": bool(c.get("ok")), "detail": str(c.get("detail", "")),
         "folds": folds}
        for c in report["leakage"]["checks"]
    ]
    results.append(_table(
        id="evaluate.leakage_audit", title="Leakage audit",
        plain=f"{n_checks} checks confirm that the test period did not train the model.",
        why="Leakage is a defect, not a caveat on a flattering accuracy number.",
        method="Poisoned-test decision digests, strict fold ordering, wrapper input-length checks and held-flat drivers.",
        columns=_cols(
            ("name", "Check", "string", None, "left"),
            ("ok", "Passed", "bool", None, "center"),
            ("detail", "Detail", "string", None, "left"),
            ("folds", "Folds", "integer", None, "right"),
        ),
        rows=leakage_rows,
        caption=f"Leakage audit {'passed' if report['leakage']['ok'] else 'FAILED'}; {n_checks} checks.",
        footnote=(f"Poison sentinel {report['leakage']['poison_value']:,.2f}; "
                  f"{len(report['leakage']['fold_digests'])} fold digests are stored in the readback payload."),
        technical="The digest covers d, D, ADF p, λ, imputation, outlier positions, transform parameters and "
                  "the training summary. It is unchanged when every test value is replaced.",
        common=common,
    ))

    # 4. Point / distributional figures.
    results.append(_figure(
        chart=ev.mase_by_horizon_chart(report["mase_by_horizon"], horizon),
        id="fig.evaluate.mase_horizon", title="MASE by forecast horizon",
        plain="Scaled error at each horizon, averaged across folds.",
        why="A model that is good at one step can fall apart at the horizon the user actually asked for.",
        method="Per-fold training-window naive / seasonal-naive denominator; errors grouped by horizon step.",
        caption=f"MASE by h = 1…{horizon}; benchmark {benchmark}; denominator uses training rows only.",
        common=common,
    ))
    results.append(_figure(
        chart=ev.fold_boxplots(report["fold_scores"], metric="mase"),
        id="fig.evaluate.fold_boxplots", title="Fold-level MASE boxplots",
        plain="Every fold's MASE is shown, not just the average.",
        why="One disastrous fold behind a good mean is an unstable model.",
        method="FoldScore from app/ts/select/cv.py, pooled per model into boxplots.",
        caption=f"{folds} rolling-origin folds per model; a wide box means the ranking depends on the origin.",
        common=common,
    ))
    results.append(_figure(
        chart=ev.loss_curves_chart(report["loss_curves"], horizon),
        id="fig.evaluate.loss_curves", title="CRPS / pinball loss curves",
        plain="Distributional loss at each horizon from the seeded predictive ensemble, with a "
              "dotted ranked-probability CRPS curve when a discrete PMF is available.",
        why="Point error cannot see a badly shaped predictive distribution; proper scores can.",
        method=f"{inp['n_paths']:,} seeded paths; empirical CRPS via the sorted-draw identity; "
               "pinball loss at every stated quantile; discrete CRPS uses the PMF sum form.",
        caption=f"CRPS and mean pinball loss by horizon, lower is better. {CRPS_CITATION} "
                + ("Dotted curves add disclosed empirical integer-PMF CRPS."
                   if any(c.get("discrete_crps") for c in report["loss_curves"].values()) else ""),
        technical="CRPS(F,y) = E|X−y| − ½E|X−X'|; quantile loss = max(q(y−f), (q−1)(y−f)).",
        common=common,
    ))
    pit_caveat = next((str(v.get("caveat", "")) for v in report["pit_histograms"].values() if v.get("caveat")), "")
    results.append(_figure(
        chart=ev.pit_histograms(report["pit_histograms"]),
        id="fig.evaluate.pit_histograms", title="PIT histograms by model",
        plain="Uniformity of the predictive CDF at the observed values, with a shape-reading guide.",
        why="Intervals can be the right width on average and still be systematically too narrow, too wide or biased.",
        method="Randomized PIT from the empirical ensemble or exact discrete PMF; KS and Cramér–von Mises tests.",
        caption="U-shape = too narrow; hill = too wide; skew = biased centre. " + pit_caveat,
        technical="KS and Cramér–von Mises test U(0,1); discrete atoms are randomized inside F(y−1)…F(y).",
        common=common,
    ))
    rqr_caveat = next((str(v.get("caveat", "")) for v in report["rqr_by_model"].values() if v.get("caveat")), "")
    results.append(_figure(
        chart=ev.rqr_qq(report["rqr_by_model"]),
        id="fig.evaluate.rqr_qq", title="Dunn–Smyth randomized quantile residual QQ",
        plain="Averaged randomized quantile residuals against the normal reference line.",
        why="A QQ plot of residuals catches distributional failures a histogram of PIT can soften.",
        method=f"One seeded RQR draw is tested; {inp['reps']} dependent repetitions report "
               "KS-p stability, not a Fisher-combined p-value.",
        caption="Points on the diagonal are calibrated. " + rqr_caveat,
        technical="RQR = Φ⁻¹(u), u drawn inside the predictive probability step; repetitions are combined, not cherry-picked.",
        common=common,
    ))
    results.append(_figure(
        chart=ev.calibration_curve(calibration_rows),
        id="fig.evaluate.calibration", title="Coverage vs nominal calibration",
        plain="Empirical interval coverage against the coverage the model claimed.",
        why="Sharpness only matters after calibration; a narrow but under-covering interval is not a win.",
        method="Binomial coverage test at each stated nominal level; average width reported beside the p-value.",
        caption="Sharpness subject to calibration: compare widths only where coverage is consistent with nominal. "
                f"{len(coverage_rows)} levels stated.",
        common=common,
    ))
    results.append(_table(
        id="evaluate.coverage_construction", title="Interval coverage, width and construction",
        plain="Nominal coverage, achieved coverage, binomial test, mean width and how each band was built.",
        why="§6.8 forbids a bare interval: the reader must know what the band claims and how it was made.",
        method="Analytic bands from each fold's training-window fit; exact monotone back-transformation when active.",
        columns=_cols(
            ("coverage", "Nominal", "number", "0.000", "right"),
            ("coverage_pct", "Nominal %", "integer", None, "right"),
            ("empirical", "Empirical", "number", "0.000", "right"),
            ("p", "Coverage p", "number", "p", "right"),
            ("mean_width", "Mean width", "number", "0.000", "right"),
            ("construction", "Construction", "string", None, "left"),
            ("source", "Source", "string", None, "left"),
        ),
        rows=coverage_rows,
        caption=f"Interval construction: {common['inputs']['interval_construction']}",
        footnote="Coverage tests are binomial; low power at small n is reported rather than read as a pass.",
        common=common,
    ))

    # 5. Comparative tests.
    min_dm_p = min([float(r["p"]) for r in dm["rows"] if r.get("p") is not None], default=None)
    results.append(_figure(
        chart=ev.dm_heatmap(dm),
        id="fig.evaluate.dm_matrix", title="Diebold–Mariano test matrix",
        plain="Pairwise p-values for equal predictive accuracy on the same out-of-sample errors.",
        why="A leader in the metric table is not automatically statistically distinguishable from the runner-up.",
        method=f"DM with Newey–West HAC variance, lag = horizon − 1 = {dm.get('lags', 0)}.",
        caption=(f"HAC lag {dm.get('lags', 0)}; smallest raw {p_phrase(min_dm_p)}. "
                 "Holm-adjusted p-values across unique pairs are in the comparison table; "
                 "a pre-specified single pair remains the strongest reading."),
        common=common,
    ))
    nested = comparative["nested"]
    cw = nested.get("clark_west") or {}
    mz = nested.get("mincer_zarnowitz") or {}
    mcs = comparative.get("mcs") or {}
    comparative_rows: list[dict[str, Any]] = []
    for r in dm["rows"]:
        comparative_rows.append({
            "test": "Diebold–Mariano", "model_a": r.get("model_a"), "model_b": r.get("model_b"),
            "statistic": r.get("statistic"), "p": r.get("p"), "p_display": p_number(r.get("p")),
            "p_holm": r.get("p_holm"), "p_holm_display": p_number(r.get("p_holm")),
            "n": r.get("n"), "plain": r.get("plain", ""), "note": dm.get("note", ""),
        })
    comparative_rows.append({
        "test": "Clark–West (nested)", "model_a": benchmark, "model_b": primary,
        "statistic": cw.get("statistic"), "p": cw.get("p"), "p_display": p_number(cw.get("p")),
        "n": nested.get("n"), "plain": cw.get("plain", ""),
        "note": "Canonical Clark & West (2007) MSPE-adjusted one-sided test: e_small² − e_large² "
                "+ (f_small − f_large)². The benchmark forecast supplies the gap explicitly.",
    })
    comparative_rows.append({
        "test": "Mincer–Zarnowitz", "model_a": primary, "model_b": "",
        "statistic": mz.get("joint_chi2"), "p": mz.get("p"), "p_display": p_number(mz.get("p")),
        "n": mz.get("n") if "n" in mz else nested.get("n"), "plain": mz.get("plain", ""),
        "note": "Joint Wald test of (intercept, slope) = (0, 1); a rejection is a calibration failure.",
    })
    comparative_rows.append({
        "test": "Model confidence set", "model_a": ", ".join(mcs.get("survivors") or []) or "—",
        "model_b": "", "statistic": mcs.get("mcs_p"), "p": mcs.get("mcs_p"),
        "p_display": p_number(mcs.get("mcs_p")), "n": mcs.get("n"),
        "plain": f"Applicable: {bool(mcs.get('applicable'))}",
        "note": mcs.get("note", ""),
    })
    comparative_rows.append({
        "test": "Directional accuracy", "model_a": primary, "model_b": benchmark,
        "statistic": hit.get("rate"), "p": hit.get("p"), "p_display": p_number(hit.get("p")),
        "n": hit.get("n"), "plain": hit.get("note", ""),
        "note": "Low power in short evaluation windows; a hit rate above 0.5 is not proof of skill.",
    })
    results.append(_table(
        id="evaluate.comparative_tests", title="Comparative and nested tests",
        plain="DM, Clark–West, Mincer–Zarnowitz, MCS and directional accuracy on the same held-out targets.",
        why="Different tests answer different questions: equal loss, nesting gain, calibration, surviving set and sign skill.",
        method="Verified S11 HAC/DM/CW/MZ helpers plus a transparent stationary-bootstrap MCS when n ≥ 30.",
        columns=_cols(
            ("test", "Test", "string", None, "left"),
            ("model_a", "Model A", "string", None, "left"),
            ("model_b", "Model B", "string", None, "left"),
            ("statistic", "Statistic", "number", "0.000", "right"),
            ("p", "p", "number", "p", "right"),
            ("p_display", "p display", "string", None, "left"),
            ("p_holm", "Holm p", "number", "p", "right"),
            ("p_holm_display", "Holm display", "string", None, "left"),
            ("n", "n", "integer", None, "right"),
            ("plain", "Reading", "string", None, "left"),
            ("note", "Note", "string", None, "left"),
        ),
        rows=comparative_rows,
        caption=f"DM HAC lag {dm.get('lags', 0)}; smallest {p_phrase(min_dm_p)}. All p-values use p_phrase/p_number.",
        footnote="DM p_holm is Holm-adjusted across unique model pairs; other tests are not part of "
                 "that family. Directional accuracy has low power below 30 pairs and excludes ties.",
        common=common,
    ))
    results.append(_figure(
        chart=ev.confusion_heatmap(hit),
        id="fig.evaluate.confusion", title="Directional hit-rate confusion",
        plain="Up/down calls from the forecast origin, with the exact binomial test.",
        why="A forecast can be accurate in level and still miss the direction a decision depends on.",
        method="Direction relative to the last observed training value at each rolling origin; exact binomial null 0.5.",
        caption=f"Hit rate {hit.get('rate')}; {p_phrase(hit.get('p'))}. Low power below 30 pairs.",
        common=common,
    ))

    # 6. Branch-specific PMF / variance tables.
    if report["discrete"] or policy.discrete:
        discrete_rows: list[dict[str, Any]] = []
        for mid, d in report["discrete"].items():
            pit_d = d.get("pit") or {}
            rqr_d = d.get("rqr") or {}
            discrete_rows.append({
                "model": mid, "discrete_crps": d.get("crps"), "pmf_log_score": d.get("log_score"),
                "pit_ks_p": pit_d.get("ks_p"), "pit_cvm_p": pit_d.get("cvm_p"),
                "rqr_combined_p": rqr_d.get("combined_p"),
                "truncated_steps": d.get("truncated_steps"),
                "pmf_source": "empirical simulated integer PMF" if d.get("approximate") else "exact predictive PMF",
                "caveat": (pit_d.get("caveat") or rqr_d.get("caveat") or "")
                          + (" " + str(d["construction"]) if d.get("construction") else ""),
                "note": "PMF log score uses P(Y = y), not a density; randomized PIT is disclosed."
                        + (" Empirical PMF scores are diagnostics; MASE + RMSSE + ensemble CRPS still rank."
                           if d.get("approximate") else ""),
            })
        if not discrete_rows:
            discrete_rows.append({
                "model": primary, "discrete_crps": None, "pmf_log_score": None,
                "pit_ks_p": None, "pit_cvm_p": None, "rqr_combined_p": None,
                "truncated_steps": 0, "pmf_source": "unavailable",
                "caveat": "No untransformed discrete PMF was available; MAPE/sMAPE remain banned and only "
                          "MAE/MASE are reported as point errors.",
                "note": "Discrete branch selected by policy.",
            })
        results.append(_table(
            id="evaluate.discrete_scores", title="Discrete predictive PMF scores",
            plain="Count CRPS, PMF log score, randomized PIT and Dunn–Smyth residuals.",
            why="A count forecast has atoms; a continuous density score on it is the wrong quantity.",
            method="Per-step PMF grid (exact or disclosed empirical simulation); ranked probability "
                   "score, −log P(Y=y), randomized CDF PIT and Dunn–Smyth residuals.",
            columns=_cols(
                ("model", "Model", "string", None, "left"),
                ("discrete_crps", "Discrete CRPS", "number", "0.000", "right"),
                ("pmf_log_score", "PMF log score", "number", "0.000", "right"),
                ("pit_ks_p", "PIT KS p", "number", "p", "right"),
                ("pit_cvm_p", "PIT CvM p", "number", "p", "right"),
                ("rqr_combined_p", "RQR combined p", "number", "p", "right"),
                ("truncated_steps", "Truncated PMF steps", "integer", None, "right"),
                ("pmf_source", "PMF source", "string", None, "left"),
                ("caveat", "Randomization caveat", "string", None, "left"),
                ("note", "Note", "string", None, "left"),
            ),
            rows=discrete_rows,
            caption="MAPE and sMAPE are banned on count / intermittent data; MAE is shown only beside MASE.",
            footnote="Randomized PIT and RQR are seeded and repeated; a single draw is not a test. "
                     "An empirical simulated integer PMF is disclosed as such and never called exact.",
            common=common,
        ))
    if report["variance"] or policy.variance:
        variance_rows: list[dict[str, Any]] = []
        for mid, d in report["variance"].items():
            bt = d.get("var_backtests") or {}
            variance_rows.append({
                "model": mid, "qlike": d.get("qlike"), "mse_variance": d.get("mse_variance"),
                "n": bt.get("n"), "violations": bt.get("violations"),
                "kupiec_p": bt.get("kupiec_p"), "christoffersen_p": bt.get("christoffersen_p"),
                "conditional_p": bt.get("conditional_p"),
                "independence_ok": bt.get("independence_ok"),
                "note": bt.get("note", ""),
            })
        if not variance_rows:
            variance_rows.append({
                "model": primary, "qlike": None, "mse_variance": None, "n": 0,
                "violations": 0, "kupiec_p": None, "christoffersen_p": None,
                "conditional_p": None, "independence_ok": None,
                "note": "No variance_forecast() wrapper was available for this selection.",
            })
        results.append(_table(
            id="evaluate.variance_scores", title="Volatility / variance-target scores",
            plain="QLIKE, MSE of variance and Kupiec / Christoffersen VaR backtests.",
            why="A variance target is not a mean forecast; scoring its point path would answer the wrong question.",
            method="Squared-return realised variance proxy; Gaussian lower-tail VaR from the forecast sd.",
            columns=_cols(
                ("model", "Model", "string", None, "left"),
                ("qlike", "QLIKE", "number", "0.000", "right"),
                ("mse_variance", "MSE variance", "number", "0.000", "right"),
                ("n", "n", "integer", None, "right"),
                ("violations", "VaR violations", "integer", None, "right"),
                ("kupiec_p", "Kupiec p", "number", "p", "right"),
                ("christoffersen_p", "Christoffersen p", "number", "p", "right"),
                ("conditional_p", "Conditional p", "number", "p", "right"),
                ("independence_ok", "Independence OK", "bool", None, "center"),
                ("note", "Note", "string", None, "left"),
            ),
            rows=variance_rows,
            caption="QLIKE is the proper variance score; squared returns are a noisy realised-variance proxy.",
            footnote="Kupiec tests unconditional coverage, Christoffersen adds independence; both have low power "
                     "in short samples.",
            common=common,
        ))

    # 7. Coverage stat, method disclosure.
    row95 = next((r for r in coverage_rows if abs(float(r.get("coverage") or 0) - 0.95) < 1e-9), None)
    if row95 and row95.get("empirical") is not None:
        cal95 = next((r for r in calibration_rows if abs(float(r.get("nominal") or 0) - 0.95) < 1e-9), {})
        results.append(_stat(
            id="stat.evaluate.coverage95", title="95% interval calibration",
            plain=f"Achieved 95% coverage was {float(row95['empirical']):.1%} across the held-out blocks.",
            why="Coverage is the licence to compare interval widths at all.",
            method="Exact binomial test of achieved against nominal coverage; construction attached to the band.",
            value=float(row95["empirical"]),
            p_value=row95.get("p"),
            statistic_name="95% empirical coverage",
            interpretation=(
                f"{cal95.get('covered', '—')} of {cal95.get('n', '—')} intervals covered; "
                f"mean width {row95.get('mean_width')}. {p_phrase(row95.get('p'))}."
            ),
            effect_size=None if row95.get("mean_width") is None else float(row95["mean_width"]),
            technical="Binomial test of H0: achieved coverage = nominal coverage.",
            common=common,
        ))
    results.append(_narrative(
        id="evaluate.method", title="Exactly how this evaluation was computed",
        plain="Every safeguard, construction and exclusion in one reproducible disclosure.",
        why="A metric table without its protocol is an opinion with decimals.",
        method="§5.12 metric battery, S11 rolling-origin CV, S12 adequacy verdict, §6 safeguards.",
        paragraphs=[
            *report["notes"],
            *report["warnings"],
            f"Protocol: {folds} {inp['scheme']} folds, horizon {horizon}, n = {n}, frequency {frequency}, "
            f"exact evaluation window {eval_window}.",
            f"Leakage: {n_checks} audit checks, ok = {report['leakage']['ok']}; "
            f"test observations used for fitting {inp['test_observations_used_for_fitting']}; "
            f"scored {inp['test_observations_scored']} pooled rows from "
            f"{inp['n_unique_test_observations']} unique observations.",
            (f"Rows: {n_missing} missing test observations, {n_excluded} excluded non-finite triples, "
             f"{len(report['fold_failures'])} failed folds — all counted, none silently dropped; "
             f"each model reconciles {integrity['expected']} held-out rows.")
            if integrity["rows_ok"] else f"Rows: {integrity['warning']}",
            f"Intervals: {len(coverage_rows)} nominal levels, {common['inputs']['interval_construction']}.",
            f"Back-transform: {inp.get('back_transform') or 'none required'}; transform active "
            f"{inp.get('transform_active')}. Raw source: {raw_source}.",
            f"PIT/RQR: {inp['reps']} seeded repetitions; {inp['n_paths']:,} predictive paths; "
            "randomization is disclosed rather than hidden.",
            f"S12 adequacy for this fit: {adequacy.get('adequacy')}. {adequacy.get('headline')}",
        ],
        numbers_cited={
            "n": float(n), "folds": float(folds), "horizon": float(horizon),
            "zero_share": float(inp["zero_share"]),
            "n_pooled_test_observations": float(inp["n_pooled_test_observations"]),
            "n_unique_test_observations": float(inp["n_unique_test_observations"]),
            "n_missing_test_observations": float(n_missing),
            "n_excluded": float(n_excluded),
            "rows_accounted": float(integrity["rows_accounted"]),
            "out_of_sample_verified": float(integrity["out_of_sample"]),
            "benchmark_verified": float(integrity["benchmark_ok"]),
            "n_leakage_checks": float(n_checks),
            "coverage_levels": float(len(coverage_rows)),
            "n_paths": float(inp["n_paths"]), "reps": float(inp["reps"]),
            "fold_failures": float(len(report["fold_failures"])),
        },
        technical="CRPS uses the Gneiting–Raftery ensemble identity; DM uses HAC lag h−1; discrete PMF scores "
                  "use the ranked probability score; QLIKE uses log(v_hat)+v_real/v_hat.",
        common=common,
    ))
    return results, common


def _advisor(
    *,
    report: dict[str, Any], adequacy: dict[str, Any], policy: MetricPolicy,
    common: dict[str, Any],
) -> AdvisorReport:
    inp = report["inputs"]
    integrity = _integrity(report)
    primary, primary_value, _ = _primary_value(report)
    label = METRIC_LABELS.get(policy.primary, policy.primary)
    ranked = report["accuracy"]["ranked"]
    leader = ranked[0] if ranked else primary
    n_checks = len(report["leakage"]["checks"])
    n_pass = int(sum(bool(c.get("ok")) for c in report["leakage"]["checks"]))
    min_dm_p = min([float(r["p"]) for r in report["comparative"]["dm"]["rows"] if r.get("p") is not None], default=None)
    min_dm_holm = min(
        [float(r["p_holm"]) for r in report["comparative"]["dm"]["rows"] if r.get("p_holm") is not None],
        default=None,
    )
    value_text = "—" if primary_value is None else f"{primary_value:.4g}"
    because = (
        f"{inp['n']} observations, {inp['folds']} folds, horizon {inp['horizon']} and "
        f"{inp['n_unique_test_observations']} unique out-of-sample rows give {primary} a {label} "
        f"of {value_text}; {leader} ranks first on that metric. "
        f"{n_pass}/{n_checks} leakage checks passed."
    )
    recommendations = [
        RuleOut(
            id="evaluate.read_primary", stage=Stage.S14_EVALUATE,
            recommend=f"Lead the write-up with {label}",
            confidence=Confidence.HIGH,
            because=because,
            provenance=["evaluate.metric_table", "fig.evaluate.metric_heatmap", "stat.evaluate.primary"],
            alternative="Read the fold boxplots before treating the ranking as stable.",
            risk_if_ignored="A mean over folds can hide one catastrophic origin.",
            how_to="Open Step 14 and compare the metric table with the fold-level boxplots.",
        ),
        RuleOut(
            id="evaluate.check_calibration", stage=Stage.S14_EVALUATE,
            recommend="Check calibration before preferring a narrow interval",
            confidence=Confidence.HIGH,
            because=(f"{len(report['coverage_construction'])} nominal coverage levels are stated and the "
                     f"95% binomial test is {p_phrase(next((r.get('p') for r in report['coverage_construction'] if abs(float(r.get('coverage') or 0)-0.95)<1e-9), None))}."),
            provenance=["fig.evaluate.calibration", "evaluate.coverage_construction"],
            alternative="Use the PIT histograms to see whether the failure is width or bias.",
            risk_if_ignored="Sharpness without calibration rewards an over-confident forecast.",
            how_to="Open Step 14 and read the calibration plot beside the interval-width table.",
        ),
        RuleOut(
            id="evaluate.read_dm", stage=Stage.S14_EVALUATE,
            recommend="Treat the leader as provisional until the DM matrix is read",
            confidence=Confidence.MEDIUM,
            because=(f"{len(report['comparative']['dm']['rows'])} paired comparisons use HAC lag "
                     f"{report['comparative']['dm'].get('lags', 0)}; smallest {p_phrase(min_dm_p)}. "
                     f"smallest Holm-adjusted {p_phrase(min_dm_holm)} across unique pairs."),
            provenance=["fig.evaluate.dm_matrix", "evaluate.comparative_tests"],
            alternative="Use Clark–West / Mincer–Zarnowitz when the benchmark is nested or calibration is in doubt.",
            risk_if_ignored="A small metric gap may be indistinguishable from forecast noise.",
            how_to="Open Step 14 and read the DM matrix with the comparative-test table.",
        ),
    ]
    blockers: list[RuleOut] = []
    if adequacy.get("status") != "evaluated":
        blockers.append(RuleOut(
            id="evaluate.adequacy_missing", stage=Stage.S14_EVALUATE,
            recommend="Run Step 12 before finalizing",
            confidence=Confidence.HIGH,
            because=f"S12 has 0 evaluated adequacy verdicts for this fit; status {adequacy.get('status')}.",
            provenance=["evaluate.verdict"],
            alternative="Carry the INCONCLUSIVE verdict into the report explicitly.",
            risk_if_ignored="Step 15 needs both S14 and S12; finalization will be refused.",
            how_to="Open Step 12 (Diagnostics), run it, then return to Step 14.",
        ))
    if report["fold_failures"]:
        blockers.append(RuleOut(
            id="evaluate.fold_failures", stage=Stage.S14_EVALUATE,
            recommend="Read the failed folds before quoting the ranking",
            confidence=Confidence.HIGH,
            because=f"{len(report['fold_failures'])} of {inp['folds'] * len(report['models'])} model-fold fits failed.",
            provenance=["evaluate.protocol_folds", "evaluate.leakage_audit"],
            alternative="Drop the unstable model from the requested list and re-run.",
            risk_if_ignored="The comparison would silently favour models that survived more folds.",
            how_to="Open Step 14 and expand the protocol table.",
        ))
    facts_used: dict[str, float | int | str | bool] = {
        "n": int(inp["n"]), "folds": int(inp["folds"]), "horizon": int(inp["horizon"]),
        "n_unique_test_observations": int(inp["n_unique_test_observations"]),
        "n_pooled_test_observations": int(inp["n_pooled_test_observations"]),
        "n_missing_test_observations": int(inp["n_missing_test_observations"]),
        "n_excluded": integrity["n_excluded"],
        "rows_accounted": integrity["rows_accounted"],
        "out_of_sample_verified": integrity["out_of_sample"],
        "benchmark_verified": integrity["benchmark_ok"],
        "leakage_ok": bool(report["leakage"]["ok"]),
        "adequacy": str(adequacy.get("adequacy")),
        "policy_cluster": policy.cluster,
        "primary_model": primary,
        "primary_metric": policy.primary,
        "n_paths": int(inp["n_paths"]), "reps": int(inp["reps"]),
        "fold_failures": len(report["fold_failures"]),
        "evaluation_window": str(inp["evaluation_window"]),
        "ranked": ", ".join(ranked),
    }
    if primary_value is not None:
        facts_used["primary_metric_value"] = float(primary_value)
    return AdvisorReport(
        stage=Stage.S14_EVALUATE,
        headline=(f"{policy.headline} {primary} scores {label} at {value_text} across "
                  f"{inp['folds']} out-of-sample folds; {leader} ranks first."),
        recommendations=recommendations, blockers=blockers, facts_used=facts_used,
        confidence=Confidence.HIGH if report["leakage"]["ok"] and not report["fold_failures"] and adequacy.get("status") == "evaluated" else Confidence.MEDIUM,
        next_stage_ready=bool(report["leakage"]["ok"] and adequacy.get("status") == "evaluated" and not report["fold_failures"]),
        narrative=(f"Scored {inp['n_pooled_test_observations']} pooled out-of-sample rows from "
                   f"{inp['evaluation_window']} with {n_pass}/{n_checks} leakage checks passed; "
                   f"S12 adequacy {adequacy.get('adequacy')}."),
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/analysis/evaluate")
def get_evaluation(session: Session = Depends(session_dep)) -> dict[str, Any]:
    """The stored evaluation report, re-rendered. Never re-runs any fold."""
    facts = _session_facts(session, horizon=6)
    options = _options(facts)
    stored = session.objects.get(EVALUATION)
    if not stored:
        return {
            "stage": Stage.S14_EVALUATE.value,
            "status": "not_started",
            "message": "No out-of-sample evaluation has been run yet.",
            "evaluation": None,
            "options": _jsonable(options),
        }
    return {
        "stage": Stage.S14_EVALUATE.value,
        "status": "complete",
        "stale": stored["version"] != session.state.series_version,
        "computed_for_version": stored["version"],
        "current_version": session.state.series_version,
        "evaluation": stored["report"],
        "request": stored["request"],
        "options": _jsonable(options),
    }


@router.post("/analysis/evaluate", response_model=AnalysisResponse)
def run_evaluate(body: EvaluationRequest, session: Session = Depends(session_dep)) -> AnalysisResponse:
    t0 = time.perf_counter()
    session.require_stage(Stage.S14_EVALUATE, allow_stale=True)
    fits = _require_fits(session)
    facts = _session_facts(session, horizon=body.horizon, models=body.models)
    series_current = _series(session)
    records = list(session.state.preprocessing)
    spec, transform_warnings = _transform(session)
    try:
        _check_records(records)
    except ValueError as exc:
        raise ChronosError(
            plain=str(exc),
            fix="Undo the offending Step 3 operation and re-fit, or reload the original dataset.",
            code="evaluate.preprocessing_unsupported",
        ) from exc
    raw, raw_source = _raw_series(session, series_current, records, spec)

    if spec is not None and body.back_transform is None:
        raise ChronosError(
            plain=f"The modelled series has an active {spec.describe()}. Its evaluation forecasts must not "
                  "be silently exponentiated back to the original scale.",
            fix="Choose naive, smearing or lognormal and send it as back_transform.",
            code="evaluate.back_transform_required",
            context={"methods": list(BACK_TRANSFORM_METHODS), "transform": spec.as_dict()},
        )
    if spec is None and body.back_transform is not None:
        raise ChronosError(
            plain="No transform is active on this series, so there is nothing to bias-adjust.",
            fix="Omit back_transform (or send null) and the evaluation stays in original units.",
            code="evaluate.back_transform_unneeded",
        )

    forecast_stored = session.objects.get(FORECAST)
    if not forecast_stored:
        raise ChronosError(
            plain="Step 13 has produced no forecast bundle, so Step 14 cannot verify the state machine.",
            fix="Open Step 13 (Forecast) and produce a forecast first.",
            code="evaluate.no_forecast",
        )
    if forecast_stored.get("version") != session.state.series_version:
        raise ChronosError(
            plain="The series changed after the Step 13 forecast was made, so the evaluation would mix versions.",
            fix="Re-run Step 13 (Forecast) on the current series before evaluating.",
            code="evaluate.stale_forecast",
        )
    bundle = forecast_stored["bundle"]
    forecast_payload = bundle.as_dict()
    available = _available(fits)
    if not available:
        raise ChronosError(
            plain="Every stored fit failed, so there is no model to evaluate out of sample.",
            fix="Read the failure reasons in Step 10 and fit a model that converges.",
            code="evaluate.no_successful_fits",
        )
    primary_id = _winner(session, available) or next(iter(available), "")
    adequacy = _adequacy(session, primary_id) if primary_id else _default_adequacy()
    benchmark_id = facts["benchmark_id"]
    outcomes = fits.get("outcomes") or []
    baseline_ids = {benchmark_id} | {
        str(o.summary.model_id) for o in outcomes
        if getattr(o.summary, "is_baseline", False)
    }
    requested = list(dict.fromkeys(body.models)) or _shortlist(session, limit=3) or ([primary_id] if primary_id else [])
    model_ids = [m for m in requested if m in available and m != "combination"]
    if primary_id and primary_id in available and primary_id not in model_ids:
        model_ids.insert(0, primary_id)
    if body.include_baselines:
        # The benchmark is always the training-window reference; including it as
        # a scored row is a user choice, not an obligation of the protocol.
        for baseline in [benchmark_id, *sorted(baseline_ids)]:
            if baseline in available and baseline not in model_ids:
                model_ids.append(baseline)
    else:
        model_ids = [m for m in model_ids if m not in baseline_ids or m == primary_id]
    model_ids = list(dict.fromkeys(model_ids))
    unknown_requested = [m for m in body.models if m not in available and m != "combination" and m != benchmark_id]
    if not model_ids:
        raise ChronosError(
            plain="No requested model is a successful stored fit, so there is nothing to compare.",
            fix=f"Choose from: {', '.join(sorted(available))}.",
            code="evaluate.unknown_models",
            context={"requested": body.models, "available": sorted(available)},
        )

    data_class = facts["data_class_obj"]
    zero_pool = raw.iloc[: max(1, int(raw.size * 0.8))].dropna().to_numpy(dtype=float)
    zero_share = float(np.mean(zero_pool == 0)) if zero_pool.size else 0.0
    discrete = (
        any(m in DISCRETE_PREDICTIVE for m in model_ids)
        or data_class in {DataClass.COUNT, DataClass.EVENT_TIMESTAMP}
    )
    vol_ids = facts["vol_ids"]
    target_kind = (
        "variance"
        if data_class == DataClass.CONTINUOUS and (primary_id in vol_ids or any(m in vol_ids for m in model_ids))
        else "level"
    )
    policy = choose_policy(
        data_class=data_class, n=int(raw.size), period=facts["period"], horizon=body.horizon,
        zero_share=zero_share, probabilistic=True, discrete=discrete, target_kind=target_kind,
    )
    seed = int(body.seed if body.seed is not None else session.seed)
    exog, _future = _exog(session)
    frequency_code = facts["frequency_code"]
    period = facts["period"]

    def factory(mid: str, fold: CVFold, ctx: TrainingContext) -> Any:
        wrapper = build_wrapper(
            mid, period=period, d_hint=ctx.d, D_hint=ctx.D,
            exog=ctx.exog_train, future_exog=ctx.future_exog,
            interventions=ctx.interventions,
        )
        if mid in MULTIVARIATE_IDS:
            wrapper._multi = build_multi_frame(ctx.train_model, ctx.exog_train)  # type: ignore[attr-defined]
        wrapper.period_days = days_per_period(frequency_code, period)
        return wrapper

    inputs = {
        "frequency": facts["frequency"], "data_class": data_class.value,
        "zero_share": zero_share, "target_kind": target_kind,
        "discrete": discrete, "probabilistic": True,
        "class_note": facts["class_note"], "raw_source": raw_source,
        "benchmark_id": benchmark_id, "forecast_model_id": forecast_payload.get("model_id"),
        "forecast_horizon": forecast_payload.get("horizon"),
        "forecast_interval_method": forecast_payload.get("interval_method"),
        "forecast_transform": forecast_payload.get("transform"),
        "forecast_test_observations_used": int((forecast_payload.get("inputs") or {}).get("test_observations_used", 0)),
        "s13_source": "POST /analysis/forecast (full-history forecast, NOT an accuracy score)",
        "adequacy": adequacy.get("adequacy"), "adequacy_status": adequacy.get("status"),
        "stage_requirements": [Stage.S13_FORECAST.value],
        "transform_warnings": list(transform_warnings),
        "cv_excluded": ["combination"],
        "include_baselines": bool(body.include_baselines),
        "baseline_models_available": sorted(baseline_ids),
    }
    try:
        run = run_evaluation(
            raw=raw, model_ids=model_ids, factory=factory, policy=policy, inputs=inputs,
            horizon=body.horizon, folds=body.folds, scheme=body.scheme,
            coverages=body.coverages, n_paths=body.n_paths, seed=seed, reps=body.reps,
            period=period, spec_template=spec, records=records, exog=exog,
            back_transform=body.back_transform, adequacy=adequacy,
            data_class=data_class, zero_share=zero_share, target_kind=target_kind,
            preference=primary_id,
        )
    except ValueError as exc:
        raise ChronosError(
            plain=str(exc),
            fix="Lower the horizon or fold count, choose a valid bias adjustment, or undo an unsupported "
                "preprocessing operation and re-fit.",
            code="evaluate.protocol_refused",
            context={"horizon": body.horizon, "folds": body.folds, "n": int(raw.size)},
        ) from exc
    report = run.as_dict()
    if unknown_requested:
        report["warnings"] = list(dict.fromkeys(report["warnings"] + [
            f"Requested model(s) {', '.join(unknown_requested)} were not successful stored fits and were "
            "not evaluated; the request was not silently substituted.",
        ]))
    if not body.include_baselines:
        report["warnings"] = list(dict.fromkeys(report["warnings"] + [
            "Baseline rows were excluded from the comparison by include_baselines=false; the "
            "training-window naive / seasonal-naive benchmark is still used for MASE and ratios.",
        ]))
    if not run.leakage.ok:
        raise ChronosError(
            plain="The leakage audit failed, so this evaluation is a defect and will not be reported.",
            fix="Inspect the audit checks, fix the training-only protocol, and re-run Step 14.",
            code="evaluate.leakage_detected",
            context={"checks": [c["name"] for c in run.leakage.checks if not c["ok"]]},
        )
    request = _jsonable(body.model_dump())
    request["seed_resolved"] = seed
    session.objects[EVALUATION] = {
        "version": session.state.series_version,
        "at": time.time(),
        "report": report,
        "run": run,
        "request": request,
        "raw_source": raw_source,
    }
    results, common = _results(
        run=run, report=report, raw=raw, adequacy=adequacy, policy=policy, body=body,
        forecast_payload=forecast_payload, facts=facts,
        transform_warnings=list(transform_warnings), raw_source=raw_source,
    )
    for result in results:
        session.add_result(result)
    primary, primary_value, _ = _primary_value(report)
    label = METRIC_LABELS.get(policy.primary, policy.primary)
    ranked = report["accuracy"]["ranked"]
    leader = ranked[0] if ranked else primary
    value_text = "—" if primary_value is None else f"{primary_value:.4g}"
    message = (
        f"{policy.cluster}: {len(model_ids)} models, {report['inputs']['folds']} folds, "
        f"{report['inputs']['n_pooled_test_observations']} OOS rows; {primary} scores {label} "
        f"at {value_text}; {leader} ranks first."
    )
    session.mark_stage(Stage.S14_EVALUATE, "complete", message)
    return AnalysisResponse(
        stage=Stage.S14_EVALUATE,
        advisor=_advisor(report=report, adequacy=adequacy, policy=policy, common=common),
        results=results,
        pipeline=session.state.progress(),
        caveats=list(common["caveats"]),
        duration_ms=int((time.perf_counter() - t0) * 1000),
    )


__all__ = ["router"]
