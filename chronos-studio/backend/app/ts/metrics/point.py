"""
ts/metrics/point.py — point-accuracy metrics, with their refusals attached (§5.12, §6.4).

The two things that make a metric table trustworthy are only visible when a metric
is *absent*: a MAPE that silently skipped its zero actual, and a MASE computed
without fitting the naive benchmark. Both are refused here with a reason. Every
relative metric uses a training-window-only denominator; `scales` supplies one
denominator per fold, or one per observation when no fold matrix is available.

Pooled MASE/RMSSE are NOT `MAE / mean(scales)` and `RMSE / mean(scales)`. Those
expressions are only correct when all folds happen to share the same scale, and
would otherwise let a large-scale fold dominate both the numerator and the
denominator. Each out-of-sample error is divided by its own fold's training
scale, then averaged: MASE = mean(|e_t| / s_fold(t)), RMSSE =
sqrt(mean((e_t / s_fold(t))²)). No test value enters a denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

#: An actual closer to zero than this relative to the series' own scale makes a
#: percentage error meaningless. It is a warning, not an imputation excuse.
NEAR_ZERO_REL = 1e-8


def mase_scale(y_train: Sequence[float], period: int | None = None) -> float | None:
    """Mean absolute in-sample naive / seasonal-naive error, training window only."""
    y = np.asarray(y_train, dtype=float)
    y = y[np.isfinite(y)]
    if y.size < 3:
        return None
    if period and int(period) >= 2 and y.size > int(period) + 1:
        p = int(period)
        d = np.abs(y[p:] - y[:-p])
    else:
        d = np.abs(np.diff(y))
    s = float(d.mean()) if d.size else 0.0
    return s if s > 0 and math.isfinite(s) else None


@dataclass
class PointMetrics:
    n: int = 0
    me: float | None = None
    mae: float | None = None
    rmse: float | None = None
    mpe: float | None = None
    mape: float | None = None
    smape: float | None = None
    mase: float | None = None
    rmsse: float | None = None
    gmrae: float | None = None
    mdrae: float | None = None
    rel_mae: float | None = None
    rel_rmse: float | None = None
    mase_by_horizon: list[float] = field(default_factory=list)
    scale: float | None = None
    benchmark: str = "naive"
    n_excluded: int = 0
    n_near_zero: int = 0
    n_scaled: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "me": self.me, "mae": self.mae, "rmse": self.rmse,
            "mpe": self.mpe, "mape": self.mape, "smape": self.smape,
            "mase": self.mase, "rmsse": self.rmsse,
            "gmrae": self.gmrae, "mdrae": self.mdrae,
            "rel_mae": self.rel_mae, "rel_rmse": self.rel_rmse,
            "mase_by_horizon": list(self.mase_by_horizon),
            "scale": self.scale, "benchmark": self.benchmark,
            "n_excluded": self.n_excluded, "n_near_zero": self.n_near_zero,
            "n_scaled": self.n_scaled,
            "warnings": list(self.warnings), "notes": list(self.notes),
        }


def _finite(*arrays: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, int]:
    """Align vectors to the longest one, padding gaps with NaN.

    Truncating to the shortest vector would silently discard the extra rows.
    Padding makes every missing or non-finite value an explicit exclusion that
    can be reconciled with the number of held-out observations.
    """
    aligned_input = [np.asarray(a, dtype=float).reshape(-1) for a in arrays]
    n = max((a.size for a in aligned_input), default=0)
    aligned: list[np.ndarray] = []
    mask = np.ones(n, dtype=bool)
    for a in aligned_input:
        full = np.full(n, np.nan, dtype=float)
        if a.size:
            full[: a.size] = a
        aligned.append(full)
        mask &= np.isfinite(full)
    return aligned, mask, int((~mask).sum())


def _row_scales(
    scales: Sequence[float],
    n_rows: int,
    errors_by_horizon: np.ndarray | None,
) -> tuple[np.ndarray | None, str | None]:
    """Align one training-only scale to every scored row; refuse an ambiguous map."""
    values = np.asarray(
        [float(s) if s is not None and math.isfinite(float(s)) and float(s) > 0 else np.nan
         for s in scales], dtype=float
    )
    n_scales = int(values.size)
    if n_scales == 0:
        return None, "no finite positive training scales"

    # Preferred: an (folds × horizon) error matrix with one scale per fold.
    if errors_by_horizon is not None:
        mat = np.asarray(errors_by_horizon, dtype=float)
        if mat.ndim == 2 and mat.shape[0] == n_scales and mat.shape[1] * n_scales >= n_rows:
            row = np.repeat(values, mat.shape[1])
            if row.size >= n_rows:
                return row[:n_rows], None

    # Golden/property callers sometimes pass one scale per observation.
    if n_scales == n_rows:
        return values, None
    # A single scale is safe to broadcast: it is either the only fold or a scalar fixture.
    if n_scales == 1:
        return np.full(n_rows, float(values[0]), dtype=float), None
    return None, (
        f"{n_scales} training scales cannot be aligned with {n_rows} pooled rows; "
        "supply an (folds × horizon) error matrix or one scale per observation"
    )


def point_metrics(
    *,
    actual: Sequence[float],
    forecast: Sequence[float],
    benchmark: Sequence[float],
    scales: Sequence[float],
    period: int | None = None,
    errors_by_horizon: np.ndarray | None = None,
    allow_mape: bool = True,
    benchmark_id: str = "naive",
) -> PointMetrics:
    """
    Score pooled out-of-sample errors. `scales` has one training-only MASE
    denominator per fold; `errors_by_horizon` is (folds, horizon) for the
    MASE-by-horizon curve. No test value enters a denominator.
    """
    (a, f, b), mask, excluded = _finite(actual, forecast, benchmark)
    a, f, b = a[mask], f[mask], b[mask]
    out = PointMetrics(n=int(a.size), n_excluded=excluded, benchmark=benchmark_id)
    if excluded:
        out.warnings.append(
            f"{excluded} held-out row(s) were excluded for a missing or non-finite actual, "
            "forecast or benchmark value; shorter vectors were padded, not silently truncated."
        )
    if a.size == 0:
        out.warnings.append("No finite out-of-sample triples were available to score.")
        return out

    e = a - f
    be = a - b
    ae = np.abs(e)
    abe = np.abs(be)
    out.me = float(e.mean())
    out.mae = float(ae.mean())
    out.rmse = float(np.sqrt(np.mean(e**2)))

    positive = [float(s) for s in scales if s is not None and math.isfinite(float(s)) and float(s) > 0]
    out.scale = float(np.mean(positive)) if positive else None
    row_scales, refusal = _row_scales(scales, a.size, errors_by_horizon)
    if row_scales is None:
        out.warnings.append(f"MASE and RMSSE are unavailable: {refusal}.")
    else:
        valid_scale = np.isfinite(row_scales) & (row_scales > 0)
        if not np.all(valid_scale):
            out.warnings.append(
                f"{int((~valid_scale).sum())} row(s) had a zero or undefined training scale; "
                "they are excluded from MASE/RMSSE and counted, not replaced by the mean scale."
            )
        scaled = np.divide(e, row_scales, out=np.full_like(e, np.nan), where=valid_scale)
        finite_scaled = np.abs(scaled[np.isfinite(scaled)])
        out.n_scaled = int(finite_scaled.size)
        if finite_scaled.size:
            out.mase = float(finite_scaled.mean())
            out.rmsse = float(np.sqrt(np.mean(scaled[np.isfinite(scaled)] ** 2)))
            out.notes.append(
                "MASE and RMSSE divide every out-of-sample error by its own fold's "
                "training-window naive / seasonal-naive denominator before pooling; "
                "the displayed mean scale is descriptive only."
            )
        else:
            out.warnings.append("MASE and RMSSE are unavailable: no row had a usable training scale.")

    # ---- percentage errors: refuse rather than drop the awkward rows --------- #
    level = float(np.mean(np.abs(a))) or 1.0
    near = np.abs(a) < max(NEAR_ZERO_REL * level, 1e-12)
    out.n_near_zero = int(near.sum())
    if not allow_mape:
        out.warnings.append(
            "MAPE, MPE, sMAPE, GMRAE and MdRAE are banned by this data class's metric policy; "
            "MAE and MASE are the only point errors reported."
        )
    elif out.n_near_zero:
        out.warnings.append(
            f"MAPE and MPE are undefined: {out.n_near_zero} of {a.size} actual values are at or "
            "near zero. Rows were not dropped; the metric is simply not reported."
        )
    else:
        out.mpe = float(np.mean(e / a) * 100.0)
        out.mape = float(np.mean(ae / np.abs(a)) * 100.0)

    # sMAPE survives a zero actual but is capped at 200%, not a percentage error.
    if allow_mape:
        denom = np.abs(a) + np.abs(f)
        ok = denom > 1e-12
        if ok.any():
            out.smape = float(np.mean(2.0 * ae[ok] / denom[ok]) * 100.0)
            out.notes.append(
                "sMAPE is on a 0–200% scale and is not a percentage error; it is shown only for "
                "comparability with published tables."
            )
        else:
            out.warnings.append("sMAPE is undefined: every actual and forecast pair is exactly zero.")

    # ---- relative-to-benchmark ratios --------------------------------------- #
    bench_mae = float(abe.mean())
    bench_rmse = float(np.sqrt(np.mean(be**2)))
    if bench_mae > 0:
        out.rel_mae = float(out.mae / bench_mae)
    if bench_rmse > 0:
        out.rel_rmse = float(out.rmse / bench_rmse)
    if allow_mape:
        ratio_den_zero = abe <= 1e-12
        if np.any(ratio_den_zero & (ae > 1e-12)):
            out.warnings.append(
                "GMRAE is undefined: the benchmark made a perfect forecast at an origin where this "
                "model erred. MdRAE excludes that ratio and says how many it excluded."
            )
        else:
            ratios = np.divide(
                ae, abe, out=np.ones_like(ae), where=abe > 1e-12
            )
            ratios = np.maximum(ratios, 1e-12)
            out.gmrae = float(np.exp(np.mean(np.log(ratios))))
        finite_ratios = np.divide(
            ae, abe, out=np.full_like(ae, np.nan), where=abe > 1e-12
        )
        finite_ratios = finite_ratios[np.isfinite(finite_ratios)]
        if finite_ratios.size:
            out.mdrae = float(np.median(finite_ratios))
            if finite_ratios.size < a.size:
                out.notes.append(
                    f"MdRAE used {finite_ratios.size} of {a.size} ratios; zero benchmark errors were "
                    "excluded and counted."
                )
    out.notes.append(
        f"The {benchmark_id} benchmark was fitted on each training window before any relative "
        "metric was computed (§6.4)."
    )

    # ---- MASE per horizon ---------------------------------------------------- #
    if errors_by_horizon is not None:
        mat = np.asarray(errors_by_horizon, dtype=float)
        if mat.ndim == 2 and mat.shape[0] == len(scales) and all(
            s is not None and math.isfinite(float(s)) and float(s) > 0 for s in scales
        ):
            per = np.abs(mat) / np.asarray(scales, dtype=float)[:, None]
            with np.errstate(all="ignore"):
                out.mase_by_horizon = [float(v) for v in np.nanmean(per, axis=0)]
        elif mat.ndim == 2:
            out.notes.append("MASE by horizon needs one finite training scale per fold.")

    return out


__all__ = ["NEAR_ZERO_REL", "PointMetrics", "mase_scale", "point_metrics"]
