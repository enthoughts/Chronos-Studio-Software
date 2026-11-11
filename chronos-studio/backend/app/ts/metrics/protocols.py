"""
ts/metrics/protocols.py — the only way S14 is allowed to produce an accuracy number (§5.12, §6).

The S13 forecast is fitted to the whole history. Calling its fitted values "accuracy" would be the
original sin this app exists to prevent, so S14 refits every candidate on a training window and
scores only the following held-out block. The protocol has three jobs:

1. **Rolling-origin folds.** `fold_origins` from S11 chooses explicit, forward-moving origins. The
   last fold's test block ends at the last observation. Every score is out of sample.
2. **Training-only decisions.** Differencing orders, Box–Cox / Yeo–Johnson λ, imputation and
   outlier dummies are re-derived from the training window alone. No Step 5 verdict, no stored λ,
   no full-series intervention list, no observed future driver path reaches a fold's fit. A
   poisoned-test audit replaces every test value with a sentinel and proves that the decision
   digest does not move.
3. **Full predictive payload.** Each fold produces point forecasts, analytic bands, seeded
   simulation paths and — when the wrapper owns one — an exact discrete PMF. Back-transformation
   is explicit, so a transformed series is never silently exponentiated.

`LeakageAudit.ok=False` is a defect, not a caveat. The API refuses to mark S14 complete if it
happens.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from app.ts.diagnose.calibration import PIT_REPS, RQR_REPS, SEED
from app.ts.forecast.backtransform import (
    METHODS as BACK_TRANSFORM_METHODS,
    BackTransformInfo,
    TransformSpec,
    apply_back_transform,
)
from app.ts.forecast.horizons import COVERAGES
from app.ts.forecast.intervals import (
    MIN_PATHS,
    IntervalBand,
    analytic_bands,
    make_draws,
    simulate_bands,
)
from app.ts.metrics.comparative import (
    directional_accuracy,
    dm_matrix,
    model_confidence_set,
    nested_tests,
)
from app.ts.metrics.discrete import discrete_scores
from app.ts.metrics.point import mase_scale, point_metrics
from app.ts.metrics.policy import MetricPolicy, choose_policy
from app.ts.metrics.probabilistic import probabilistic_scores
from app.ts.metrics.variance import variance_scores
from app.ts.select.cv import CVFold, CVScheme, MIN_TRAIN, fold_origins, score_fold

_POISON = 1_234_567.89


# --------------------------------------------------------------------------- #
# JSON hygiene
# --------------------------------------------------------------------------- #
def _clean(value: Any) -> Any:
    """Convert numpy scalars and non-finite floats to strict-JSON values."""
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    return value


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(_clean(payload), sort_keys=True, default=str).encode()).hexdigest()[:16]


def _record_field(rec: Any, name: str) -> Any:
    return getattr(rec, name, None) if not isinstance(rec, dict) else rec.get(name)


def _label(index: pd.Index, position: int) -> str:
    if position < 0 or position >= len(index):
        return "—"
    v = index[position]
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except Exception:
        return str(v)


# --------------------------------------------------------------------------- #
# Training-only decisions
# --------------------------------------------------------------------------- #
def _check_records(records: Sequence[Any]) -> list[str]:
    """Reject only treatments that cannot be reconstructed from the original series.

    A seasonal/first difference plus training-only imputation is the documented
    S7 → S3 → S14 path. S14 scores the untouched original series and lets each
    fold re-decide differencing and imputation, so a whole-series difference is
    replayable without leaking the test window. Trimming, aggregation, smoothing
    and row deletion reindex or redefine observations and remain unsupported.
    """
    unsupported: list[str] = []
    for rec in records:
        step = str(_record_field(rec, "step") or "")
        method = str(_record_field(rec, "method") or "")
        if step in {"trim", "aggregate", "smooth"} or method in {
            "trim", "aggregate", "moving_average", "delete"
        }:
            unsupported.append(method or step)
    if unsupported:
        raise ValueError(
            "S14 needs the original regular series: "
            + ", ".join(sorted(set(unsupported)))
            + " change the observation set or the modelled quantity. Undo those Step 3 "
              "operations and re-fit before evaluating."
        )
    return [
        str(_record_field(rec, "method") or _record_field(rec, "step") or "")
        for rec in records
    ]


def _impute_train(train: pd.Series, methods: Sequence[str], period: int | None) -> tuple[pd.Series, dict[str, Any]]:
    from app.ts.preprocess.ops import kalman_smooth, seasonal_interpolate

    n_missing = int(train.isna().sum())
    out = train.astype(float).copy()
    chosen = "none"
    if n_missing:
        if "kalman_smoother" in methods:
            chosen = "kalman_smoother"
            out = kalman_smooth(out, period=period).series
        else:
            chosen = "seasonal_interpolation" if "seasonal_interpolation" in methods else "training_linear"
            out = seasonal_interpolate(out, period=period).series
        if out.isna().any():
            try:
                out = out.interpolate(method="time", limit_direction="both")
            except Exception:
                out = out.interpolate(method="linear", limit_direction="both")
    finite = out.to_numpy(dtype=float)[np.isfinite(out.to_numpy(dtype=float))]
    if out.isna().any():
        fill = float(np.median(finite)) if finite.size else 0.0
        out = out.fillna(fill)
        chosen += "+training_median"
    return out.astype(float), {
        "n_missing": n_missing,
        "method": chosen,
        "period": int(period) if period else None,
        "n_imputed": int(n_missing),
    }


def _outlier_positions(train: pd.Series) -> tuple[list[int], list[dict[str, Any]], dict[str, Any]]:
    v = train.to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    if finite.size < 8:
        return [], [], {"method": "mad_on_training_window", "n": int(finite.size), "positions": []}
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    if scale <= 1e-12:
        positions: list[int] = []
        zs = np.zeros_like(v)
    else:
        zs = np.abs(v - med) / scale
        positions = [int(i) for i in np.flatnonzero(np.nan_to_num(zs, nan=0.0) > 3.5)[:4]]
    interventions = [{"position": i, "kind": "pulse"} for i in positions]
    digest = {
        "method": "mad_on_training_window",
        "median": round(med, 6),
        "scale": round(scale, 6),
        "n": int(finite.size),
        "positions": positions,
        "z_at_flags": [round(float(zs[i]), 4) for i in positions],
    }
    return positions, interventions, digest


def _differencing(train: pd.Series) -> tuple[int, int, float | None, dict[str, Any]]:
    import warnings

    from statsmodels.tsa.stattools import adfuller

    v = train.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    p: float | None = None
    d = 0
    if v.size >= 12 and np.var(v) > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = adfuller(v, autolag="AIC", result_object=False)
                except TypeError:  # older statsmodels has no result_object argument
                    res = adfuller(v, autolag="AIC")
                p = float(res[1])
            if p > 0.05 and np.var(np.diff(v)) < np.var(v):
                d = 1
        except Exception:
            p = None
    # Seasonal differencing is deliberately not invented here: D=0 is a fixed protocol setting,
    # while the seasonal-naive benchmark below still uses m=period. That is honest and cannot see
    # the test block.
    return d, 0, p, {
        "method": "ADF on training window only; seasonal D fixed at 0",
        "d": d, "D": 0,
        "adf_p": None if p is None else round(p, 6),
    }


def _lambda(train: pd.Series, spec: TransformSpec | None) -> tuple[float | None, list[str]]:
    """
    Re-estimate a data-dependent λ on THIS training window only.

    The full-series `spec.params["lambda"]` is never a fallback: it was estimated
    from the test block too, and copying it into a fold would be leakage. If
    Box–Cox cannot be re-estimated the evaluation is refused; Yeo–Johnson falls
    back to λ = 1 (the identity), which is a fixed value, with a warning.
    """
    v = train.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    warnings: list[str] = []
    if spec is None:
        return None, warnings
    if spec.method == "boxcox":
        if v.size < 8:
            raise ValueError(
                "Box–Cox λ cannot be re-estimated from a training window with fewer than eight "
                "finite values; undo the transform or use a longer series before evaluating."
            )
        if np.any(v <= 0):
            raise ValueError(
                "A Box–Cox transform is active but this fold's training window is not strictly "
                "positive. Re-using the full-series λ would leak test values; undo the transform "
                "or use Yeo–Johnson."
            )
        try:
            lam = float(stats.boxcox_normmax(v, method="mle"))
        except Exception as exc:
            raise ValueError(
                f"Box–Cox λ could not be re-estimated on the training window ({exc}); the "
                "evaluation is refused rather than reusing the full-series λ."
            ) from exc
        return float(np.clip(lam, -2.0, 2.0)), warnings
    if spec.method == "yeojohnson":
        if v.size < 8:
            warnings.append(
                "Yeo–Johnson λ could not be re-estimated on a short training window; λ = 1 "
                "(identity) was used instead of the full-series λ."
            )
            return 1.0, warnings
        try:
            return float(np.clip(stats.yeojohnson_normmax(v), -2.0, 2.0)), warnings
        except Exception as exc:
            warnings.append(
                f"Yeo–Johnson λ re-estimation failed ({type(exc).__name__}); λ = 1 (identity) was "
                "used instead of the full-series λ."
            )
            return 1.0, warnings
    # log / sqrt have no data-dependent λ; their base/offset are user choices.
    return None, warnings


def _training_spec(spec: TransformSpec | None, lam: float | None) -> TransformSpec | None:
    if spec is None:
        return None
    params = dict(spec.params)
    if lam is not None:
        params["lambda"] = float(lam)
    return TransformSpec(method=spec.method, params=params, label=spec.label or spec.method)


def _forward(series: pd.Series, spec: TransformSpec | None) -> pd.Series:
    """Forward transform matching `TransformSpec.inverse` exactly."""
    if spec is None or spec.method == "none":
        return pd.Series(series).astype(float).copy()
    v = pd.Series(series).astype(float).to_numpy()
    if spec.method == "log":
        offset = float(spec.params.get("offset", 0.0) or 0.0)
        shifted = v + offset
        if np.any(shifted <= 0):
            raise ValueError("The log transform needs a strictly positive training window.")
        out = np.log10(shifted) if str(spec.params.get("base", "e")) in {"10", "10.0"} else np.log(shifted)
    elif spec.method == "sqrt":
        if np.any(v < 0):
            raise ValueError("The square-root transform needs a non-negative training window.")
        out = np.sqrt(v)
    elif spec.method == "boxcox":
        lam = float(spec.params.get("lambda", 0.0))
        if np.any(v <= 0) and abs(lam) > 1e-12:
            raise ValueError("Box–Cox needs a strictly positive training window.")
        out = np.log(v) if abs(lam) < 1e-12 else stats.boxcox(v, lam)
    elif spec.method == "yeojohnson":
        lam = float(spec.params.get("lambda", 1.0))
        out = stats.yeojohnson(v, lam)
    else:
        raise ValueError(f"Unknown transform '{spec.method}'.")
    return pd.Series(np.asarray(out, dtype=float), index=pd.Series(series).index)


def _benchmark(train_raw: pd.Series, horizon: int, period: int | None) -> tuple[np.ndarray, str]:
    v = train_raw.to_numpy(dtype=float)
    finite = v[np.isfinite(v)]
    last = float(finite[-1]) if finite.size else float("nan")
    if period and int(period) >= 2 and finite.size > int(period):
        cycle = finite[-int(period):]
        return np.asarray([cycle[(j - 1) % int(period)] for j in range(1, horizon + 1)], float), "seasonal_naive"
    return np.full(horizon, last, float), "naive"


def _returns(train_raw: pd.Series, test_raw: pd.Series) -> np.ndarray:
    last = train_raw.to_numpy(dtype=float)[np.isfinite(train_raw.to_numpy(dtype=float))]
    prev = float(last[-1]) if last.size else float("nan")
    path = np.concatenate([[prev], test_raw.to_numpy(dtype=float)])
    if np.all(np.isfinite(path)) and np.all(path > 0):
        return np.diff(np.log(path))
    return np.diff(path)


def _exog_fold(
    exog: pd.DataFrame | None, fold: CVFold, horizon: int
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, bool]:
    """Training drivers only; the future path is held flat at the last TRAINING row."""
    if exog is None or getattr(exog, "empty", True):
        return None, None, True
    e = pd.DataFrame(exog).astype(float).reset_index(drop=True)
    if len(e) <= fold.train_end:
        return None, None, False
    train_e = e.iloc[fold.train_start:fold.train_end + 1].reset_index(drop=True)
    last = train_e.iloc[-1]
    future = pd.DataFrame([last.to_dict()] * horizon)
    held_flat = bool(np.allclose(future.to_numpy(dtype=float), np.tile(last.to_numpy(dtype=float), (horizon, 1)), equal_nan=True))
    return train_e, future, held_flat


@dataclass
class TrainingContext:
    fold: CVFold
    train_raw: pd.Series
    test_raw: pd.Series
    train_model: pd.Series
    test_actual: np.ndarray
    spec: TransformSpec | None
    d: int
    D: int
    adf_p: float | None
    lambda_value: float | None
    benchmark: np.ndarray
    benchmark_id: str
    baseline: np.ndarray
    baseline_value: float
    scale: float | None
    exog_train: pd.DataFrame | None
    future_exog: pd.DataFrame | None
    interventions: list[dict[str, Any]]
    decisions: dict[str, Any]
    n_missing_test: int
    exog_held_flat: bool
    warnings: list[str] = field(default_factory=list)


def _decisions(
    train: pd.Series,
    *,
    spec_template: TransformSpec | None,
    methods: Sequence[str],
    period: int | None,
) -> dict[str, Any]:
    imputed, imp = _impute_train(train, methods, period)
    positions, interventions, out = _outlier_positions(imputed)
    d, D, adf_p, diff = _differencing(imputed)
    lam, lam_warn = _lambda(imputed, spec_template)
    spec = _training_spec(spec_template, lam)
    v = imputed.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return {
        "n_train": int(imputed.size),
        "train_last": round(float(v[-1]), 6) if v.size else None,
        "train_mean": round(float(v.mean()), 6) if v.size else None,
        "train_sd": round(float(v.std(ddof=1)), 6) if v.size > 1 else None,
        "imputation": imp,
        "outliers": out,
        "differencing": diff,
        "lambda": None if lam is None else round(float(lam), 6),
        "transform_method": None if spec is None else spec.method,
        "transform_params": _clean(dict(spec.params)) if spec else {},
        "interventions": positions,
        "warnings": list(lam_warn),
        "adf_p": None if adf_p is None else round(float(adf_p), 6),
        "d": d, "D": D,
    }


def _context(
    raw: pd.Series,
    fold: CVFold,
    *,
    spec_template: TransformSpec | None,
    methods: Sequence[str],
    period: int | None,
    horizon: int,
    exog: pd.DataFrame | None,
    back_transform: str | None,
) -> TrainingContext:
    assert fold.train_end < fold.test_start, (
        f"fold {fold.index} trains through index {fold.train_end} and scores from {fold.test_start}; "
        "this is a leakage defect."
    )
    train_slice = raw.iloc[fold.train_start:fold.train_end + 1]
    test_slice = raw.iloc[fold.test_start:fold.test_end + 1]
    dec = _decisions(train_slice, spec_template=spec_template, methods=methods, period=period)
    imputed, _ = _impute_train(train_slice, methods, period)
    spec = _training_spec(spec_template, dec["lambda"])
    if spec is not None and back_transform not in BACK_TRANSFORM_METHODS:
        raise ValueError(
            f"The series has a {spec.describe()}; choose one of {', '.join(BACK_TRANSFORM_METHODS)} "
            "before back-transforming. No exponentiation is performed without an explicit choice."
        )
    train_model = _forward(imputed, spec)
    benchmark, benchmark_id = _benchmark(imputed, horizon, period)
    finite_train = imputed.to_numpy(dtype=float)
    finite_train = finite_train[np.isfinite(finite_train)]
    baseline_value = float(finite_train[-1]) if finite_train.size else float("nan")
    exog_train, future_exog, held_flat = _exog_fold(exog, fold, horizon)
    return TrainingContext(
        fold=fold,
        train_raw=imputed,
        test_raw=test_slice,
        train_model=train_model,
        test_actual=test_slice.to_numpy(dtype=float),
        spec=spec,
        d=int(dec["d"]), D=int(dec["D"]),
        adf_p=dec["adf_p"],
        lambda_value=dec["lambda"],
        benchmark=benchmark,
        benchmark_id=benchmark_id,
        baseline=np.full(horizon, baseline_value, float),
        baseline_value=baseline_value,
        scale=mase_scale(imputed.to_numpy(dtype=float), period=period),
        exog_train=exog_train,
        future_exog=future_exog,
        interventions=[{"position": int(p), "kind": "pulse"} for p in dec["interventions"]],
        decisions=dec,
        n_missing_test=int(test_slice.isna().sum()),
        exog_held_flat=held_flat,
        warnings=list(dec["warnings"]),
    )


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
@dataclass
class LeakageAudit:
    ok: bool
    checks: list[dict[str, Any]] = field(default_factory=list)
    fold_digests: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    poison_value: float = _POISON

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "checks": _clean(self.checks),
            "fold_digests": _clean(self.fold_digests),
            "notes": list(self.notes), "poison_value": self.poison_value,
        }


def _params_match(spec: TransformSpec | None, decisions: dict[str, Any]) -> bool:
    """The fold's transform parameters must be the ones recomputed from training data."""
    if spec is None:
        return True
    recorded = dict(decisions.get("transform_params") or {})
    for key, value in spec.params.items():
        if key not in recorded:
            return False
        got = recorded[key]
        if isinstance(value, (int, float)) and isinstance(got, (int, float)):
            if abs(float(value) - float(got)) > 1e-6:
                return False
        elif str(value) != str(got):
            return False
    return True


def leakage_audit(
    *,
    raw: pd.Series,
    folds: Sequence[CVFold],
    contexts: Sequence[TrainingContext],
    spec_template: TransformSpec | None,
    methods: Sequence[str],
    period: int | None,
    fit_checks: Sequence[dict[str, Any]],
    expected_fits: int | None = None,
) -> LeakageAudit:
    checks: list[dict[str, Any]] = []
    digests: list[dict[str, Any]] = []
    order_ok = all(f.train_end < f.test_start for f in folds)
    checks.append({
        "name": "strict_train_test_order", "ok": order_ok,
        "detail": "Every fold's last training index is strictly before its first test index."
                  if order_ok else "A fold's training window overlaps its test block.",
    })
    counts_ok = len(folds) == len(contexts)
    checks.append({
        "name": "one_training_context_per_fold", "ok": counts_ok,
        "detail": f"{len(folds)} folds and {len(contexts)} training contexts were supplied."
                  if counts_ok else
                  f"Only {min(len(folds), len(contexts))} fold/context pairs exist; an audit that "
                  "silently zips to the shorter list is not an audit.",
    })

    poison_ok = counts_ok
    digest_ok = counts_ok
    transform_ok = True
    for fold, ctx in zip(folds, contexts):
        original = _decisions(
            raw.iloc[fold.train_start:fold.train_end + 1],
            spec_template=spec_template, methods=methods, period=period,
        )
        poisoned = raw.copy()
        poisoned.iloc[fold.test_start:fold.test_end + 1] = _POISON
        after = _decisions(
            poisoned.iloc[fold.train_start:fold.train_end + 1],
            spec_template=spec_template, methods=methods, period=period,
        )
        same = _digest(original) == _digest(after)
        poison_ok &= same
        digest_ok &= _digest(original) == _digest(ctx.decisions)
        transform_ok &= _params_match(ctx.spec, ctx.decisions)
        digests.append({
            "fold": fold.index,
            "train_start": fold.train_start, "train_end": fold.train_end,
            "test_start": fold.test_start, "test_end": fold.test_end,
            "decision_digest": _digest(original),
            "poisoned_digest": _digest(after),
            "context_digest": _digest(ctx.decisions),
            "decisions": original,
        })
    checks.append({
        "name": "poisoned_test_invariance", "ok": bool(poison_ok),
        "detail": (
            "Replacing every test-period value with a sentinel left the differencing order, "
            "Box–Cox / Yeo–Johnson λ, imputation summary and outlier decisions unchanged in "
            "every fold."
            if poison_ok else
            "A training-only decision changed when the test values were poisoned."
        ),
    })
    checks.append({
        "name": "fold_decisions_match_context", "ok": bool(digest_ok),
        "detail": "Each context's stored decision digest equals the digest recomputed from its "
                  "training slice." if digest_ok else
                  "A fold used decisions that differ from the training-only recomputation.",
    })
    checks.append({
        "name": "training_only_transform_parameters", "ok": bool(transform_ok),
        "detail": "Data-dependent transform λ values were re-estimated on each training window; "
                  "user-specified base/offset are carried unchanged." if transform_ok else
                  "A fold copied a full-series transform parameter instead of its training-only value.",
    })

    declared = expected_fits is not None
    want = int(expected_fits) if declared else int(len(fit_checks))
    complete = declared and len(fit_checks) == want
    fit_ok = complete and all(bool(c.get("ok")) for c in fit_checks)
    checks.append({
        "name": "factory_received_training_window_only", "ok": bool(fit_ok),
        "detail": (
            f"All {want} (model, fold) fits were audited and handed a series no longer than the "
            "training window, with no observed test-period driver path."
            if fit_ok else
            (
                f"Only {len(fit_checks)} of {want} fits were audited, or at least one fitted "
                "wrapper received a longer series than its training window."
                if declared else
                "The caller did not declare the expected number of (model, fold) fits, so "
                "completeness cannot be proved."
            )
        ),
        "rows": _clean(list(fit_checks)),
    })
    def _drivers_verified(ctx: TrainingContext) -> bool:
        if not bool(getattr(ctx, "exog_held_flat", False)):
            return False
        train_e = getattr(ctx, "exog_train", None)
        future = getattr(ctx, "future_exog", None)
        if train_e is None or future is None:
            return train_e is None and future is None
        if not len(train_e) or not len(future):
            return False
        try:
            last = train_e.iloc[-1].to_numpy(dtype=float)
            expected = np.tile(last, (len(future), 1))
            got = future.to_numpy(dtype=float)
            return bool(np.allclose(got, expected, equal_nan=True))
        except (TypeError, ValueError, IndexError):
            return False

    exog_ok = bool(contexts) and all(_drivers_verified(c) for c in contexts)
    checks.append({
        "name": "future_drivers_held_flat", "ok": exog_ok,
        "detail": "Future exogenous rows repeat the last TRAINING driver row; test-period drivers "
                  "were not supplied to the fit or forecast."
                  if exog_ok else
                  "The stored future-driver frame does not repeat the last training row, or the "
                  "training-only flag was not set; the audit does not trust an unverified flag.",
    })
    ok = all(bool(c["ok"]) for c in checks)
    return LeakageAudit(
        ok=ok, checks=checks, fold_digests=digests,
        notes=[
            "Leakage is a defect, not a caveat: a failed check refuses the whole evaluation.",
            f"Poison sentinel: {_POISON:,.2f}. The digest covers d, D, ADF p, λ, imputation, "
            "outlier positions, transform parameters and the training summary.",
            "Parameters were refit on each training window. The S13 full-history forecast is NOT "
            "used as an accuracy score.",
            "Fit checks are complete by construction: a factory that fails before it can be "
            "inspected is recorded as a failed audit row, not omitted.",
        ],
    )


# --------------------------------------------------------------------------- #
# Predictive payload for one fold
# --------------------------------------------------------------------------- #
def _nan_bands(coverages: Sequence[float], horizon: int, construction: str) -> list[IntervalBand]:
    nan = np.full(horizon, np.nan)
    return [IntervalBand(float(c), nan.copy(), nan.copy(), construction, "analytic", round(1 - float(c), 10))
            for c in coverages]


def _fold_predictive(
    wrapper: Any,
    ctx: TrainingContext,
    *,
    horizon: int,
    coverages: Sequence[float],
    n_paths: int,
    seed: int,
    back_transform: str | None,
) -> dict[str, Any]:
    levels = sorted({float(c) for c in coverages} | {0.95})
    predictor = getattr(wrapper, "predictive", None)
    pred = None
    is_discrete = False
    try:
        wrapper.predict(int(horizon), alpha=0.05)
        pred = predictor() if callable(predictor) else None
        is_discrete = pred is not None and hasattr(pred, "as_dict")
    except Exception:
        is_discrete = False

    # Croston/TSB/ADIDA own an occurrence/size simulation but no closed-form PMF. S14 asks for
    # their empirical integer predictive distribution explicitly; this does NOT replace the S13
    # `predictive()` contract or relabel the analytic interval as exact.
    empirical_predictor = getattr(wrapper, "evaluation_predictive", None)
    empirical_pred = None
    if not is_discrete and callable(empirical_predictor):
        try:
            empirical_pred = empirical_predictor(
                horizon=int(horizon), n_paths=int(n_paths),
                seed=int(seed) + 997 * int(ctx.fold.index),
            )
        except Exception:
            empirical_pred = None
    distribution_source = pred if is_discrete else empirical_pred

    point_z, analytic = analytic_bands(
        wrapper.predict, horizon, levels,
        discrete=is_discrete,
        detail="S14 fold fit; analytic intervals from the training-window fit only.",
    )
    if is_discrete:
        try:
            wrapper.predict(int(horizon), alpha=0.05)
            pred = predictor() if callable(predictor) else None
        except Exception:
            pred = None
    resid_fn = getattr(wrapper, "residuals", None)
    try:
        resid = resid_fn() if callable(resid_fn) else None
    except Exception:
        resid = None
    resid = np.asarray(resid if resid is not None else [], dtype=float).reshape(-1)
    resid = resid[np.isfinite(resid)]
    ref95 = next((b for b in analytic if abs(float(b.coverage) - 0.95) < 1e-9), None)
    draws = make_draws(int(seed) + 997 * int(ctx.fold.index), int(n_paths), int(horizon), int(resid.size))
    sim_bands: list[IntervalBand] = []
    paths_z: np.ndarray | None = None
    construction = "Analytic intervals; seeded simulation unavailable."
    try:
        sim = simulate_bands(
            point=point_z,
            lower95=ref95.lower if ref95 else None,
            upper95=ref95.upper if ref95 else None,
            residuals=resid,
            draws=draws,
            coverages=levels,
            discrete_predictive=distribution_source,
        )
        sim_bands, paths_z = list(sim.bands), np.asarray(sim.paths, dtype=float)
        construction = str(sim.info.get("construction", "")) + (
            " In S14 the residual pool and parameters come from this fold's training fit only; "
            "no test-period observation was used."
        )
        if empirical_pred is not None:
            construction += (
                " The per-step PMF is empirical: seeded draws from the model's own occurrence/size "
                "simulation were stochastically integerised to preserve the mean. It is not an "
                "exact parametric PMF, and its support is finite."
            )
    except Exception as exc:
        construction = f"Simulation unavailable ({type(exc).__name__}: {exc}); analytic bands remain labelled."

    all_bands = list(analytic) + list(sim_bands)
    bt: BackTransformInfo | None = None
    if ctx.spec is not None:
        point, all_bt, paths, info = apply_back_transform(
            point=point_z, bands=all_bands, paths=paths_z,
            spec=ctx.spec, method=back_transform, residuals=resid,
        )
        bt = info
        construction += f" Back-transformed through {ctx.spec.describe()} using {info.method}."
    else:
        point = np.asarray(point_z, dtype=float)
        paths = paths_z
    n_analytic = len(analytic)
    analytic_bt = list(all_bt[:n_analytic]) if ctx.spec is not None else list(analytic)
    sim_bt = list(all_bt[n_analytic:]) if ctx.spec is not None else list(sim_bands)

    pmf_payload: dict[str, Any] | None = None
    if ctx.spec is None:
        if pred is not None and hasattr(pred, "as_dict"):
            try:
                pmf_payload = pred.as_dict(max_steps=int(horizon))
                pmf_payload["truncated_any"] = bool(np.any(np.asarray(pred.truncated, dtype=bool)))
            except Exception:
                pmf_payload = None
        elif empirical_pred is not None:
            try:
                pmf_payload = empirical_pred.as_dict(max_steps=int(horizon))
                pmf_payload["truncated_any"] = False
                pmf_payload["approximate"] = True
                pmf_payload["evaluation_only"] = True
                pmf_payload["construction"] = (
                    f"Empirical integer PMF from {int(n_paths)} seeded draws of the fitted "
                    f"{getattr(wrapper, 'model_id', 'intermittent')} occurrence/size mechanism; "
                    "stochastic integerisation preserves the simulated mean, but this is not an "
                    "exact parametric predictive distribution."
                )
            except Exception:
                pmf_payload = None
    return {
        "point": np.asarray(point, dtype=float),
        "analytic_bands": analytic_bt,
        "simulation_bands": sim_bt,
        "paths": paths,
        "construction": construction,
        "is_discrete": is_discrete,
        "pmf": pmf_payload,
        "transform": bt.as_dict() if bt is not None else None,
        "n_paths": int(n_paths),
        "seed": int(seed) + 997 * int(ctx.fold.index),
    }


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _nan_like(length: int) -> np.ndarray:
    return np.full(int(length), np.nan, dtype=float)


def _pooled_bands(
    per_cov: dict[float, dict[str, list[np.ndarray]]],
    levels: Sequence[float],
    construction: str,
    total_len: int,
) -> list[IntervalBand]:
    out: list[IntervalBand] = []
    for cov in levels:
        lo_parts = per_cov.get(float(cov), {}).get("lower", [])
        hi_parts = per_cov.get(float(cov), {}).get("upper", [])
        lo = np.concatenate(lo_parts) if lo_parts else _nan_like(total_len)
        hi = np.concatenate(hi_parts) if hi_parts else _nan_like(total_len)
        if lo.size < total_len:
            lo = np.pad(lo, (0, total_len - lo.size), constant_values=np.nan)
            hi = np.pad(hi, (0, total_len - hi.size), constant_values=np.nan)
        out.append(IntervalBand(
            float(cov), lo[:total_len], hi[:total_len], construction, "analytic",
            round(1 - float(cov), 10),
        ))
    return out


def _by_horizon(pooled: Sequence[float], n_folds: int, horizon: int) -> list[float]:
    arr = np.asarray(pooled, dtype=float).reshape(-1)
    want = int(n_folds) * int(horizon)
    if arr.size < want:
        arr = np.pad(arr, (0, want - arr.size), constant_values=np.nan)
    mat = arr[:want].reshape(int(n_folds), int(horizon))
    with np.errstate(all="ignore"):
        out = np.nanmean(mat, axis=0)
    return [float(v) if math.isfinite(v) else float("nan") for v in out]


def _pinball_pooled(prob: dict[str, Any]) -> np.ndarray:
    """Mean pinball loss at each pooled observation, across every reported quantile."""
    steps = (prob.get("pinball") or {}).get("per_step") or {}
    arrays = [np.asarray(v, dtype=float) for v in steps.values()]
    if arrays:
        size = max(a.size for a in arrays)
        stack = np.vstack([np.pad(a, (0, size - a.size), constant_values=np.nan) for a in arrays])
        with np.errstate(all="ignore"):
            return np.nanmean(stack, axis=0)
    return np.asarray([], dtype=float)


def _primary_id(model_ids: Sequence[str], preference: str | None) -> str:
    if preference and preference in model_ids:
        return str(preference)
    for mid in model_ids:
        if mid not in {"naive", "seasonal_naive", "mean", "white_noise"}:
            return str(mid)
    return str(model_ids[0]) if model_ids else ""


# --------------------------------------------------------------------------- #
# Evaluation run
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationRun:
    policy: MetricPolicy
    inputs: dict[str, Any]
    folds: list[dict[str, Any]]
    models: list[str]
    point: dict[str, dict[str, Any]]
    probabilistic: dict[str, dict[str, Any]]
    discrete: dict[str, dict[str, Any]]
    variance: dict[str, dict[str, Any]]
    comparative: dict[str, Any]
    leakage: LeakageAudit
    accuracy: dict[str, Any]
    metric_rows: list[dict[str, Any]]
    mase_by_horizon: dict[str, list[float]]
    fold_scores: list[dict[str, Any]]
    loss_curves: dict[str, dict[str, list[float]]]
    pit_histograms: dict[str, dict[str, Any]]
    rqr_by_model: dict[str, dict[str, Any]]
    calibration: dict[str, Any]
    coverage_construction: list[dict[str, Any]]
    fold_failures: list[dict[str, Any]]
    warnings: list[str]
    notes: list[str]
    adequacy: dict[str, Any]
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return _clean({
            "policy": self.policy.as_dict(),
            "inputs": self.inputs,
            "folds": self.folds,
            "models": list(self.models),
            "point": self.point,
            "probabilistic": self.probabilistic,
            "discrete": self.discrete,
            "variance": self.variance,
            "comparative": self.comparative,
            "leakage": self.leakage.as_dict(),
            "accuracy": self.accuracy,
            "metric_rows": self.metric_rows,
            "mase_by_horizon": self.mase_by_horizon,
            "fold_scores": self.fold_scores,
            "loss_curves": self.loss_curves,
            "pit_histograms": self.pit_histograms,
            "rqr_by_model": self.rqr_by_model,
            "calibration": self.calibration,
            "coverage_construction": self.coverage_construction,
            "fold_failures": self.fold_failures,
            "warnings": self.warnings,
            "notes": self.notes,
            "adequacy": self.adequacy,
            "seconds": round(self.seconds, 3),
        })


def run_evaluation(
    *,
    raw: pd.Series,
    model_ids: Sequence[str],
    factory: Callable[[str, CVFold, TrainingContext], Any],
    policy: MetricPolicy | None = None,
    inputs: dict[str, Any] | None = None,
    horizon: int,
    folds: int,
    scheme: str,
    coverages: Sequence[float],
    n_paths: int,
    seed: int,
    reps: int,
    period: int | None,
    spec_template: TransformSpec | None,
    records: Sequence[Any],
    exog: pd.DataFrame | None,
    back_transform: str | None,
    adequacy: dict[str, Any],
    data_class: Any = "continuous",
    zero_share: float = 0.0,
    target_kind: str = "level",
    preference: str | None = None,
) -> EvaluationRun:
    t0 = time.perf_counter()
    raw = pd.Series(raw).astype(float).copy()
    n = int(raw.size)
    index = pd.Index(raw.index)
    levels = sorted({float(c) for c in coverages} | {0.95})
    if n_paths < MIN_PATHS:
        raise ValueError(f"At least {MIN_PATHS:,} seeded paths are required.")
    if spec_template is not None and back_transform not in BACK_TRANSFORM_METHODS:
        raise ValueError(
            f"The series has a {spec_template.describe()}; choose one of "
            f"{', '.join(BACK_TRANSFORM_METHODS)}. Nothing is exponentiated silently."
        )
    if spec_template is None and back_transform is not None:
        raise ValueError("No transform is active, so there is nothing to bias-adjust.")
    methods = _check_records(records)
    cv_scheme = CVScheme(scheme) if scheme in {s.value for s in CVScheme} else CVScheme.EXPANDING
    origins = fold_origins(n, horizon=int(horizon), folds=int(folds), scheme=cv_scheme)
    contexts = [
        _context(raw, f, spec_template=spec_template, methods=methods, period=period,
                 horizon=int(horizon), exog=exog, back_transform=back_transform)
        for f in origins
    ]
    train_window = f"{_label(index, origins[0].train_start)} … {_label(index, origins[-1].train_end)}"
    eval_window = f"{_label(index, origins[0].test_start)} … {_label(index, origins[-1].test_end)}"
    unique_test = sorted({i for f in origins for i in range(f.test_start, f.test_end + 1)})
    base_inputs = dict(inputs or {})
    base_inputs.update({
        "n": n,
        "frequency": base_inputs.get("frequency", "unspecified"),
        "period": period,
        "horizon": int(horizon),
        "folds": len(origins),
        "scheme": cv_scheme.value,
        "training_window": train_window,
        "evaluation_window": eval_window,
        "n_unique_test_observations": len(unique_test),
        "n_pooled_test_observations": int(sum(f.n_test for f in origins)),
        "n_missing_test_observations": int(sum(c.n_missing_test for c in contexts)),
        "benchmark": contexts[0].benchmark_id,
        "coverages": levels,
        "interval_method": "analytic bands for pinball / Winkler; seeded simulation for CRPS / PIT",
        "n_paths": int(n_paths),
        "seed": int(seed),
        "reps": int(reps),
        "back_transform": back_transform,
        "transform_active": spec_template is not None,
        "test_observations_used_for_fitting": 0,
        "test_observations_scored": int(sum(f.n_test for f in origins)),
        "data_class": str(getattr(data_class, "value", data_class)),
        "zero_share": float(zero_share),
        "target_kind": target_kind,
        "min_train": MIN_TRAIN,
    })
    pol = policy or choose_policy(
        data_class=data_class, n=n, period=period, horizon=horizon, zero_share=zero_share,
        probabilistic=True, discrete=base_inputs.get("discrete", False),
        target_kind=target_kind,
    )

    fold_meta: list[dict[str, Any]] = []
    for f, ctx in zip(origins, contexts):
        fold_meta.append({
            "fold": f.index,
            "train_start": f.train_start, "train_end": f.train_end,
            "test_start": f.test_start, "test_end": f.test_end,
            "n_train": f.n_train, "n_test": f.n_test,
            "train_window": f"{_label(index, f.train_start)} … {_label(index, f.train_end)}",
            "test_window": f"{_label(index, f.test_start)} … {_label(index, f.test_end)}",
            "zone": "out-of-sample only",
            "d": ctx.d, "D": ctx.D,
            "adf_p": ctx.adf_p,
            "lambda": ctx.lambda_value,
            "benchmark": ctx.benchmark_id,
            "n_missing_test": ctx.n_missing_test,
            "decision_digest": _digest(ctx.decisions),
        })

    # One audit row per (model, fold) from the start. A factory that raises
    # before it can be inspected stays an explicit FAILED row rather than
    # disappearing from an `all(...)` over a shorter list.
    fit_checks: dict[tuple[str, int], dict[str, Any]] = {
        (mid, ctx.fold.index): {
            "model": mid, "fold": ctx.fold.index,
            "train_rows": int(len(ctx.train_model)),
            "wrapper_rows": None, "ok": False,
            "reason": "factory or fit did not complete",
        }
        for mid in model_ids for ctx in contexts
    }
    expected_fits = len(fit_checks)
    fold_failures: list[dict[str, Any]] = []
    warnings: list[str] = []
    if len(unique_test) < int(sum(f.n_test for f in origins)):
        warnings.append(
            "Rolling-origin test blocks overlap: pooled rows include repeated observations. "
            "HAC lag and Holm-adjusted DM p-values are reported, but coverage/binomial and "
            "directional tests remain descriptive under overlap."
        )
    notes: list[str] = [
        "Accuracy is out-of-sample rolling-origin only; the S13 full-history fitted values are "
        "never reported as forecast accuracy (§6.7).",
        "MASE / RMSSE and every relative ratio use the naive / seasonal-naive benchmark fitted "
        "on each training window before the test block (§6.4).",
        "Every interval states its nominal coverage and construction; sharpness is read subject "
        "to calibration.",
        "Rows with a missing actual or a failed fold are counted, not silently dropped.",
        "No AIC/BIC comparison is made across folds or differencing orders; the DM matrix uses the "
        "same effective out-of-sample targets.",
    ]
    for c in contexts:
        warnings.extend(c.warnings)
    if contexts and not contexts[0].exog_held_flat:
        warnings.append("The exogenous future frame could not be built from training rows alone.")

    point_out: dict[str, dict[str, Any]] = {}
    prob_out: dict[str, dict[str, Any]] = {}
    discrete_out: dict[str, dict[str, Any]] = {}
    variance_out: dict[str, dict[str, Any]] = {}
    fold_scores: list[dict[str, Any]] = []
    mase_by_horizon: dict[str, list[float]] = {}
    loss_curves: dict[str, dict[str, list[float]]] = {}
    pit_histograms: dict[str, dict[str, Any]] = {}
    rqr_by_model: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    errors_by_model: dict[str, list[np.ndarray]] = {}
    pooled_actual: dict[str, np.ndarray] = {}
    pooled_forecast: dict[str, np.ndarray] = {}
    pooled_baseline: dict[str, np.ndarray] = {}
    pooled_benchmark: dict[str, np.ndarray] = {}
    pooled_benchmark_errors: dict[str, np.ndarray] = {}
    transform_payload: dict[str, Any] | None = None

    for mid in model_ids:
        actual_parts: list[np.ndarray] = []
        forecast_parts: list[np.ndarray] = []
        benchmark_parts: list[np.ndarray] = []
        baseline_parts: list[np.ndarray] = []
        errors_folds: list[np.ndarray] = []
        errors_matrix: list[np.ndarray] = []
        scales: list[float | None] = []
        paths_parts: list[np.ndarray] = []
        point_parts: list[np.ndarray] = []
        per_cov: dict[float, dict[str, list[np.ndarray]]] = {
            float(c): {"lower": [], "upper": []} for c in levels
        }
        construction = "Analytic intervals from the fold's training-window fit."
        grids: list[np.ndarray] = []
        pmfs: list[np.ndarray] = []
        truncated: list[bool] = []
        pmf_approximate = False
        realized_var_parts: list[np.ndarray] = []
        forecast_var_parts: list[np.ndarray] = []
        returns_parts: list[np.ndarray] = []
        forecast_sd_parts: list[np.ndarray] = []
        n_fold_failures = 0

        for ctx in contexts:
            h = int(ctx.fold.n_test)
            benchmark = np.asarray(ctx.benchmark[:h], dtype=float)
            baseline = np.asarray(ctx.baseline[:h], dtype=float)
            actual = np.asarray(ctx.test_actual[:h], dtype=float)
            scales.append(ctx.scale)
            benchmark_parts.append(benchmark)
            baseline_parts.append(baseline)
            actual_parts.append(actual)
            try:
                wrapper = factory(mid, ctx.fold, ctx)
                if getattr(wrapper, "_is_fitted", False) or getattr(wrapper, "summary", None) is not None:
                    raise RuntimeError("factory returned an already-fitted wrapper")
                wrapper.fit(ctx.train_model)
                y_fit = getattr(wrapper, "y", None)
                rows = int(np.asarray(y_fit).size) if y_fit is not None else int(len(ctx.train_model))
                fit_ok = rows <= int(len(ctx.train_model))
                row = fit_checks[(mid, ctx.fold.index)]
                row.update({
                    "wrapper_rows": rows, "ok": bool(fit_ok),
                })
                row.pop("reason", None)
                if not fit_ok:
                    raise RuntimeError("wrapper was handed more rows than its training window")
                if getattr(wrapper, "failed", False):
                    raise RuntimeError(str(getattr(wrapper.summary, "failure_reason", "the fold fit failed")))
                payload = _fold_predictive(
                    wrapper, ctx, horizon=h, coverages=levels, n_paths=n_paths,
                    seed=int(seed), back_transform=back_transform,
                )
                point = np.asarray(payload["point"], dtype=float)[:h]
                if point.size < h:
                    point = np.pad(point, (0, h - point.size), constant_values=np.nan)
                point_parts.append(point)
                forecast_parts.append(point)
                construction = str(payload["construction"])
                if payload.get("transform"):
                    transform_payload = payload["transform"]
                for b in payload["analytic_bands"]:
                    key = float(b.coverage)
                    lo = np.asarray(b.lower, dtype=float)[:h]
                    hi = np.asarray(b.upper, dtype=float)[:h]
                    if lo.size < h:
                        lo = np.pad(lo, (0, h - lo.size), constant_values=np.nan)
                        hi = np.pad(hi, (0, h - hi.size), constant_values=np.nan)
                    per_cov.setdefault(key, {"lower": [], "upper": []})["lower"].append(lo)
                    per_cov.setdefault(key, {"lower": [], "upper": []})["upper"].append(hi)
                paths = payload.get("paths")
                if paths is None or np.asarray(paths).ndim != 2:
                    paths = np.full((int(n_paths), h), np.nan)
                else:
                    paths = np.asarray(paths, dtype=float)
                    if paths.shape[1] < h:
                        paths = np.pad(paths, ((0, 0), (0, h - paths.shape[1])), constant_values=np.nan)
                    paths = paths[:int(n_paths), :h]
                paths_parts.append(paths)
                pmf = payload.get("pmf")
                if pmf:
                    pmf_approximate = pmf_approximate or bool(pmf.get("approximate"))
                    for j in range(h):
                        g = np.asarray(pmf["support"][j], dtype=float)
                        p = np.asarray(pmf["pmf"][j], dtype=float)
                        grids.append(g)
                        pmfs.append(p)
                        truncated.append(bool(pmf["truncated"][j]))
                else:
                    # One empty PMF per scored position keeps the PMF vector aligned with `actual`
                    # even when one fold failed to produce a distribution.
                    grids.extend(np.asarray([], dtype=float) for _ in range(h))
                    pmfs.extend(np.asarray([], dtype=float) for _ in range(h))
                    truncated.extend(False for _ in range(h))
                if hasattr(wrapper, "variance_forecast"):
                    vf = wrapper.variance_forecast()
                    var = np.asarray(vf.get("variance", []), dtype=float)[:h]
                    if var.size < h:
                        var = np.pad(var, (0, h - var.size), constant_values=np.nan)
                    r = _returns(ctx.train_raw, ctx.test_raw)[:h]
                    if r.size < h:
                        r = np.pad(r, (0, h - r.size), constant_values=np.nan)
                    realized_var_parts.append(r ** 2)
                    forecast_var_parts.append(var)
                    returns_parts.append(r)
                    forecast_sd_parts.append(np.sqrt(np.maximum(var, 0.0)))
            except Exception as exc:
                n_fold_failures += 1
                row = fit_checks[(mid, ctx.fold.index)]
                if not row.get("ok"):
                    row["reason"] = f"{type(exc).__name__}: {exc}"
                fold_failures.append({
                    "model": mid, "fold": ctx.fold.index,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                point = _nan_like(h)
                point_parts.append(point)
                forecast_parts.append(point)
                paths_parts.append(np.full((int(n_paths), h), np.nan))
                grids.extend(np.asarray([], dtype=float) for _ in range(h))
                pmfs.extend(np.asarray([], dtype=float) for _ in range(h))
                truncated.extend(False for _ in range(h))
                for c in levels:
                    per_cov[float(c)]["lower"].append(_nan_like(h))
                    per_cov[float(c)]["upper"].append(_nan_like(h))
                realized_var_parts.append(_nan_like(h))
                forecast_var_parts.append(_nan_like(h))
                returns_parts.append(_nan_like(h))
                forecast_sd_parts.append(_nan_like(h))
            err = actual - point
            errors_folds.append(err)
            errors_matrix.append(err)
            lower95 = per_cov.get(0.95, {}).get("lower", [np.full(h, np.nan)])[-1]
            upper95 = per_cov.get(0.95, {}).get("upper", [np.full(h, np.nan)])[-1]
            fs = score_fold(
                ctx.fold, actual, point,
                y_train=ctx.train_raw.to_numpy(dtype=float), period=period,
                lower=lower95, upper=upper95,
            )
            fold_scores.append({"model_id": mid, **fs.as_dict()})

        actual = np.concatenate(actual_parts) if actual_parts else np.asarray([], dtype=float)
        forecast = np.concatenate(forecast_parts) if forecast_parts else np.asarray([], dtype=float)
        benchmark = np.concatenate(benchmark_parts) if benchmark_parts else np.asarray([], dtype=float)
        baseline = np.concatenate(baseline_parts) if baseline_parts else np.asarray([], dtype=float)
        point = np.concatenate(point_parts) if point_parts else np.asarray([], dtype=float)
        errors_by_model[mid] = errors_folds
        pooled_actual[mid] = actual
        pooled_forecast[mid] = forecast
        pooled_baseline[mid] = baseline
        pooled_benchmark[mid] = benchmark
        pooled_benchmark_errors[mid] = actual - benchmark
        mat = np.vstack(errors_matrix) if errors_matrix else np.zeros((0, int(horizon)))
        pm = point_metrics(
            actual=actual, forecast=forecast, benchmark=benchmark, scales=scales,
            period=period, errors_by_horizon=mat,
            allow_mape=pol.allows("mape"), benchmark_id=contexts[0].benchmark_id if contexts else "naive",
        )
        point_out[mid] = pm.as_dict()
        mase_by_horizon[mid] = [float(v) if math.isfinite(v) else float("nan") for v in pm.mase_by_horizon]

        total_len = int(actual.size)
        bands = _pooled_bands(per_cov, levels, construction, total_len)
        paths = (
            np.concatenate(paths_parts, axis=1)
            if paths_parts and all(p.shape[0] == paths_parts[0].shape[0] for p in paths_parts)
            else np.full((int(n_paths), total_len), np.nan)
        )
        prob = probabilistic_scores(
            actual=actual, point=point, bands=bands, paths=paths,
            interval_method="analytic + seeded simulation",
            construction=construction, n_paths=int(n_paths), reps=int(reps), seed=int(seed),
        )
        prob_payload = prob.as_dict()
        prob_out[mid] = prob_payload
        crps_steps = list(prob.crps_by_step)
        pin_steps = _pinball_pooled(prob_payload)
        loss_curves[mid] = {
            "crps": _by_horizon(crps_steps, len(origins), int(horizon)),
            "pinball": _by_horizon(pin_steps, len(origins), int(horizon)) if pin_steps.size else [],
            "source": "seeded simulation ensemble",
        }
        pit = prob.pit or {}
        pit_histograms[mid] = {
            "hist": list(pit.get("hist") or []),
            "ks_p": pit.get("ks_p"),
            "cvm_p": pit.get("cvm_p"),
            "reading": pit.get("reading", ""),
            "kind": pit.get("kind", "empirical_ensemble"),
            "caveat": pit.get("caveat", ""),
            "note": pit.get("note", ""),
        }
        rqr = prob.rqr or {}
        rqr_by_model[mid] = {
            "qq_theoretical": list(rqr.get("qq_theoretical") or []),
            "qq_sample": list(rqr.get("qq_sample") or []),
            "ks_p": rqr.get("ks_p"),
            "combined_p": _prob_combined_p(rqr),
            "reps": rqr.get("reps", int(reps)),
            "stability": rqr.get("stability", ""),
            "caveat": rqr.get("caveat", ""),
            "note": rqr.get("note", ""),
        }

        has_pmf = (
            spec_template is None and len(grids) == int(actual.size)
            and any(g.size and p.size and float(np.nansum(p)) > 0 for g, p in zip(grids, pmfs))
        )
        if has_pmf:
            ds = discrete_scores(
                grids=grids, pmfs=pmfs, observed=actual, truncated=truncated,
                reps=int(reps), seed=int(seed),
            )
            ddict = ds.as_dict()
            if pmf_approximate:
                ddict["approximate"] = True
                ddict["construction"] = (
                    "Empirical integer PMF from the fitted model's seeded occurrence/size simulation; "
                    "stochastic integerisation preserves the mean, but the support is finite and the "
                    "distribution is not an exact parametric predictive law."
                )
                ddict["warnings"] = list(ddict.get("warnings") or []) + [
                    f"{mid}: discrete CRPS/log score/PIT/RQR use an empirical simulated integer PMF, "
                    "not a closed-form PMF. Randomized PIT remains necessary for count atoms."
                ]
                ddict["notes"] = list(ddict.get("notes") or []) + [
                    "The intermittent policy still ranks on MASE + RMSSE + ensemble CRPS; these "
                    "PMF scores are an additional distributional diagnostic, not a replacement."
                ]
            discrete_out[mid] = ddict
            if pol.discrete:
                # Count/event branch: the discrete PMF is the predictive law, so its randomized
                # PIT / RQR and its exact CRPS replace the continuous-ensemble diagnostics.
                pit_histograms[mid] = {
                    "hist": list(ds.pit.get("hist") or []),
                    "ks_p": ds.pit.get("ks_p"),
                    "cvm_p": ds.pit.get("cvm_p"),
                    "reading": ds.pit.get("reading", ""),
                    "kind": ds.pit.get("kind", "discrete"),
                    "caveat": ds.pit.get("caveat", ""),
                    "note": ds.pit.get("note", ""),
                }
                rqr_by_model[mid] = {
                    "qq_theoretical": list(ds.rqr.get("qq_theoretical") or []),
                    "qq_sample": list(ds.rqr.get("qq_sample") or []),
                    "ks_p": ds.rqr.get("ks_p"),
                    "combined_p": ds.rqr.get("combined_p"),
                    "reps": ds.rqr.get("reps", int(reps)),
                    "stability": ds.rqr.get("stability", ""),
                    "caveat": ds.rqr.get("caveat", ""),
                    "note": ds.rqr.get("note", ""),
                }
                loss_curves[mid]["crps"] = _by_horizon(ds.crps_by_step, len(origins), int(horizon))
                loss_curves[mid]["source"] = (
                    "empirical integer PMF" if pmf_approximate else "exact discrete PMF"
                )
            else:
                # Intermittent branch: keep the ensemble CRPS as the primary probabilistic metric,
                # but add the exact ranked-probability-score curve as a separate diagnostic.
                loss_curves[mid]["discrete_crps"] = _by_horizon(
                    ds.crps_by_step, len(origins), int(horizon)
                )
                loss_curves[mid]["source"] = (
                    "seeded simulation ensemble + empirical integer PMF"
                    if pmf_approximate else
                    "seeded simulation ensemble + discrete PMF"
                )
        elif pol.discrete:
            warnings.append(
                f"{mid}: a discrete PMF was requested but no untransformed PMF was available; "
                "MAPE/sMAPE remain banned and only MAE/MASE are reported as point errors."
            )
        elif pol.cluster == "intermittent_zero_heavy":
            warnings.append(
                f"{mid}: no untransformed integer predictive PMF was available, so the discrete "
                "CRPS/log-score/PIT/RQR table is omitted rather than reusing a continuous density; "
                "MASE + RMSSE + ensemble CRPS remain the ranking metrics."
            )

        if pol.variance and forecast_var_parts:
            vs = variance_scores(
                realized_var=np.concatenate(realized_var_parts),
                forecast_var=np.concatenate(forecast_var_parts),
                returns=np.concatenate(returns_parts),
                forecast_sd=np.concatenate(forecast_sd_parts),
            )
            variance_out[mid] = vs.as_dict()
        if n_fold_failures:
            warnings.append(f"{mid}: {n_fold_failures} of {len(contexts)} folds failed; the rows are counted above.")

    primary = _primary_id(model_ids, preference)
    errors_for_dm = {mid: [np.asarray(e, dtype=float) for e in folds] for mid, folds in errors_by_model.items()}
    dm = dm_matrix(errors_for_dm, horizon=int(horizon))
    hit = directional_accuracy(
        pooled_baseline.get(primary, np.asarray([])),
        pooled_actual.get(primary, np.asarray([])),
        pooled_forecast.get(primary, np.asarray([])),
    )
    nested = nested_tests(
        actual=pooled_actual.get(primary, np.asarray([])),
        forecast=pooled_forecast.get(primary, np.asarray([])),
        benchmark_forecast=pooled_benchmark.get(primary, np.asarray([])),
        benchmark_errors=pooled_benchmark_errors.get(primary, np.asarray([])),
        model_errors=np.concatenate(errors_for_dm.get(primary, [])) if errors_for_dm.get(primary) else np.asarray([]),
        horizon=int(horizon), model_name=primary,
    )
    mcs = model_confidence_set(errors_for_dm)
    comparative = {
        "dm": dm,
        "nested": nested,
        "mcs": mcs,
        "hit_rate": hit.as_dict(),
        "primary_model": primary,
        "citations": [
            "Diebold & Mariano (1995)",
            "Clark & West (2007)",
            "Mincer & Zarnowitz (1969)",
            "Hansen, Lunde & Nason (2011)",
            "Holm (1979) for the DM family-wise adjustment",
        ],
    }

    calibration_rows: list[dict[str, Any]] = []
    coverage_construction: list[dict[str, Any]] = []
    if primary in prob_out:
        calibration_rows = list(prob_out[primary].get("calibration", {}).get("rows") or [])
        coverage_construction = [
            {
                "coverage": r.get("coverage"),
                "coverage_pct": r.get("coverage_pct"),
                "construction": r.get("construction"),
                "source": r.get("source"),
                "mean_width": r.get("mean_width"),
                "empirical": r.get("empirical"),
                "p": r.get("p"),
            }
            for r in calibration_rows
        ]
        base_inputs["interval_construction"] = str(
            prob_out[primary].get("construction") or "coverage and construction stated per band"
        )
    else:
        base_inputs["interval_construction"] = "No probabilistic payload for the primary model."

    metric_rows = _metric_rows(
        model_ids=model_ids, policy=pol, point=point_out, prob=prob_out,
        discrete=discrete_out, variance=variance_out, hit=hit.as_dict(),
        primary=primary,
    )
    audit = leakage_audit(
        raw=raw, folds=origins, contexts=contexts, spec_template=spec_template,
        methods=methods, period=period, fit_checks=list(fit_checks.values()),
        expected_fits=expected_fits,
    )
    if not audit.ok:
        warnings.append("Leakage audit FAILED; S14 must not be reported as complete.")
    if base_inputs["n_missing_test_observations"]:
        warnings.append(
            f"{base_inputs['n_missing_test_observations']} test observations were missing and are "
            "counted as excluded, not silently dropped."
        )
    if transform_payload:
        base_inputs["transform"] = transform_payload
    base_inputs["leakage_ok"] = bool(audit.ok)
    base_inputs["n_fold_failures"] = len(fold_failures)
    base_inputs["benchmark_id"] = contexts[0].benchmark_id if contexts else "naive"
    base_inputs["primary_model"] = primary
    base_inputs["adequacy"] = adequacy.get("adequacy", "not_evaluated")

    return EvaluationRun(
        policy=pol, inputs=base_inputs, folds=fold_meta, models=list(model_ids),
        point=point_out, probabilistic=prob_out, discrete=discrete_out, variance=variance_out,
        comparative=comparative, leakage=audit,
        accuracy={"primary": primary, "primary_metric": pol.primary,
                  "ranked": _ranked_models(metric_rows, pol.primary),
                  "n_scored": base_inputs["n_pooled_test_observations"],
                  "evaluation_window": eval_window},
        metric_rows=metric_rows, mase_by_horizon=mase_by_horizon,
        fold_scores=fold_scores, loss_curves=loss_curves,
        pit_histograms=pit_histograms, rqr_by_model=rqr_by_model,
        calibration={"rows": calibration_rows,
                     "philosophy": "Sharpness subject to calibration: narrower wins only after "
                                   "empirical coverage is consistent with nominal."},
        coverage_construction=coverage_construction,
        fold_failures=fold_failures,
        warnings=list(dict.fromkeys(warnings)),
        notes=notes,
        adequacy=dict(adequacy),
        seconds=time.perf_counter() - t0,
    )


def _prob_combined_p(rqr: dict[str, Any]) -> float | None:
    """
    One inferential KS p-value, not a Fisher combination.

    Per-repetition p-values come from dependent randomizations of the same
    data; Fisher's chi-square would be invalid. `rqr_from_paths` already stores
    the first seeded draw's p in both `ks_p` and `combined_p`.
    """
    p = rqr.get("combined_p")
    if p is None:
        p = rqr.get("ks_p")
    if p is None:
        return None
    try:
        value = float(p)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


_HIGHER_BETTER = {
    "pit", "rqr", "randomized_pit", "kupiec", "christoffersen", "mcs_p", "hit_rate",
}


def _ranked_models(rows: list[dict[str, Any]], metric: str) -> list[str]:
    """Rank by the primary metric, respecting its direction."""
    higher = metric in _HIGHER_BETTER

    def key(r: dict[str, Any]):
        v = r.get(metric)
        missing = v is None or not math.isfinite(float(v))
        value = float(v) if v is not None else math.inf
        # Missing values always sort last; higher-is-better uses a negated key.
        return (missing, -value if higher and not missing else value, str(r.get("model", "")))

    return [str(r["model"]) for r in sorted(rows, key=key)]


def _metric_value(mid: str, metric: str, *, point, prob, discrete, variance, hit) -> float | None:
    p = point.get(mid, {})
    q = prob.get(mid, {})
    d = discrete.get(mid, {})
    v = variance.get(mid, {})
    if metric == "hit_rate" and mid == hit.get("primary_model"):
        return hit.get("rate")
    if metric in {"mae", "rmse", "mase", "rmsse", "mape", "smape", "mpe", "gmrae", "mdrae", "rel_mae", "rel_rmse"}:
        return p.get(metric)
    if metric == "crps":
        return q.get("crps")
    if metric == "pinball":
        return (q.get("pinball") or {}).get("mean")
    if metric == "log_score":
        return d.get("log_score") if d else (q.get("log_score") or {}).get("value")
    if metric == "winkler":
        return (q.get("winkler") or {}).get("mean")
    if metric in {"coverage", "calibration"}:
        # Calibration error, not mean coverage: 0.95 empirical at 0.50 nominal
        # is a 0.45 miss, not a good score.
        rows = (q.get("calibration") or {}).get("rows") or []
        deviations = [
            abs(float(r["empirical"]) - float(r.get("nominal") or r.get("coverage")))
            for r in rows
            if r.get("empirical") is not None and (r.get("nominal") or r.get("coverage")) is not None
        ]
        return float(np.mean(deviations)) if deviations else None
    if metric == "pit":
        return (q.get("pit") or {}).get("cvm_p")
    if metric == "rqr":
        return (q.get("rqr") or {}).get("ks_p")
    if metric == "discrete_crps":
        return d.get("crps")
    if metric == "randomized_pit":
        return (d.get("pit") or {}).get("cvm_p")
    if metric == "qlike":
        return v.get("qlike")
    if metric == "mse_variance":
        return v.get("mse_variance")
    if metric == "kupiec":
        return ((v.get("var_backtests") or {}).get("kupiec_p"))
    if metric == "christoffersen":
        return ((v.get("var_backtests") or {}).get("christoffersen_p"))
    return None


def _metric_rows(
    *,
    model_ids: Sequence[str],
    policy: MetricPolicy,
    point: dict[str, dict[str, Any]],
    prob: dict[str, dict[str, Any]],
    discrete: dict[str, dict[str, Any]],
    variance: dict[str, dict[str, Any]],
    hit: dict[str, Any],
    primary: str,
) -> list[dict[str, Any]]:
    hit_with_primary = {**hit, "primary_model": primary}
    metrics = list(policy.metrics) + ["hit_rate"]
    rows: list[dict[str, Any]] = []
    for mid in model_ids:
        row: dict[str, Any] = {"model": mid, "primary_model": primary}
        for metric in metrics:
            row[metric] = _metric_value(
                mid, metric, point=point, prob=prob, discrete=discrete,
                variance=variance, hit=hit_with_primary,
            )
        rows.append(row)
    return _clean(rows)


__all__ = [
    "EvaluationRun", "LeakageAudit", "TrainingContext", "choose_policy", "leakage_audit",
    "run_evaluation",
]
