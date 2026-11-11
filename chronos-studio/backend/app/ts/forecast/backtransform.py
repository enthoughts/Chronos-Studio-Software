"""
ts/forecast/backtransform.py — bringing a transformed forecast back honestly (§5.11).

"Silently exponentiating a log model is one of the most common published errors."
This module refuses to do that. Three points, all enforced here:

1. **Nothing is inverted unless the caller names a method.** `apply_back_transform`
   takes `method` explicitly and the API refuses a forecast on a transformed series
   when the request omits it. There is no default exponentiation.
2. **Bias adjustment changes the POINT, not the quantiles.** g is monotone, so
   `q_p(Y) = g^{-1}(q_p(Z))` exactly. Multiplying interval endpoints by a smearing
   factor (a common shortcut) would break that identity. The bands are transformed
   directly; the naive/smearing/lognormal choice only changes which *mean* is quoted.
   * naive = g^{-1}(E[Z]) = the median on the original scale,
   * smearing = mean_r g^{-1}(E[Z] + e_r) (Duan 1983, generalised to any g),
   * lognormal = exp(E[Z] + Var(Z)/2) (only for a log-scale transform).
3. **An inverse has a domain.** Box–Cox and Yeo–Johnson paths can land outside it.
   Values are clipped to the nearest valid point and the count is reported, because
   a clipped tail that nobody mentions is a silently narrowed interval.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import stats

_Z975 = float(stats.norm.ppf(0.975))
METHODS: tuple[str, ...] = ("naive", "smearing", "lognormal")
_DOMAIN_EPS = 1e-12


def inverse_boxcox(z: Any, lam: float) -> np.ndarray:
    """Inverse Box–Cox. λ=0 is the log case. Domain-clipped, never NaN."""
    arr = np.asarray(z, dtype=float)
    lam = float(lam)
    if abs(lam) < _DOMAIN_EPS:
        return np.exp(np.clip(arr, -700.0, 700.0))
    base = np.maximum(1.0 + lam * arr, _DOMAIN_EPS)
    return np.power(base, 1.0 / lam)


def inverse_yeojohnson(z: Any, lam: float) -> np.ndarray:
    """
    Inverse Yeo–Johnson, implemented here because SciPy 1.17 ships no
    `special.inv_yeojohnson`. Both branches (positive and negative arguments) are
    handled; the two singular λ values 0 and 2 have their own closed forms.
    """
    arr = np.asarray(z, dtype=float)
    lam = float(lam)
    out = np.empty_like(arr)
    pos = arr >= 0
    if abs(lam) < _DOMAIN_EPS:
        out[pos] = np.expm1(np.clip(arr[pos], -700.0, 700.0))
    else:
        base = np.maximum(1.0 + lam * arr[pos], _DOMAIN_EPS)
        out[pos] = np.power(base, 1.0 / lam) - 1.0
    neg = ~pos
    if abs(lam - 2.0) < _DOMAIN_EPS:
        out[neg] = 1.0 - np.exp(np.clip(-arr[neg], -700.0, 700.0))
    else:
        base = np.maximum(1.0 - (2.0 - lam) * arr[neg], _DOMAIN_EPS)
        out[neg] = 1.0 - np.power(base, 1.0 / (2.0 - lam))
    return out


@dataclass(frozen=True)
class TransformSpec:
    """The transform that is *currently* applied to the modelled series."""

    method: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str = ""

    @property
    def is_log_scale(self) -> bool:
        if self.method == "log":
            return True
        if self.method == "boxcox":
            try:
                return abs(float(self.params.get("lambda", 1.0))) < _DOMAIN_EPS
            except (TypeError, ValueError):
                return False
        return False

    @property
    def log_k(self) -> float:
        """Natural-log coefficient of the inverse: g^{-1}(z) = exp(k z) - offset."""
        if self.method == "log":
            base = str(self.params.get("base", "e"))
            if base in {"10", "10.0"}:
                return math.log(10.0)
            return 1.0
        return 1.0

    @property
    def offset(self) -> float:
        if self.method == "log":
            try:
                return float(self.params.get("offset", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    def inverse(self, z: Any) -> tuple[np.ndarray, int]:
        """Back-transform values and report how many needed domain clipping."""
        arr = np.asarray(z, dtype=float)
        if self.method == "none":
            return arr.copy(), 0
        if self.method in {"log", "boxcox"} and self.is_log_scale:
            k, off = self.log_k, self.offset
            scaled = np.clip(k * arr, -700.0, 700.0)
            return np.exp(scaled) - off, int(np.count_nonzero(k * arr != scaled))
        if self.method == "boxcox":
            lam = float(self.params.get("lambda", 1.0))
            base = 1.0 + lam * arr
            clipped = int(np.count_nonzero(base < _DOMAIN_EPS))
            return inverse_boxcox(arr, lam), clipped
        if self.method == "yeojohnson":
            lam = float(self.params.get("lambda", 1.0))
            pos = arr >= 0
            base = np.where(pos, 1.0 + lam * arr, 1.0 - (2.0 - lam) * arr)
            clipped = int(np.count_nonzero(base < _DOMAIN_EPS))
            return inverse_yeojohnson(arr, lam), clipped
        if self.method == "sqrt":
            clipped = int(np.count_nonzero(arr < 0))
            return np.maximum(arr, 0.0) ** 2, clipped
        raise ValueError(f"Unknown transform method '{self.method}'.")

    def describe(self) -> str:
        if self.method == "log":
            return f"log transform (base {self.params.get('base', 'e')}, offset {self.offset:g})"
        if self.method in {"boxcox", "yeojohnson"}:
            return f"{self.method} transform (lambda {self.params.get('lambda', 1)})"
        if self.method == "sqrt":
            return "square-root transform"
        return "no transform"

    def as_dict(self) -> dict[str, Any]:
        return {"method": self.method, "params": dict(self.params), "label": self.label}


def detect_transform(records: Iterable[Any]) -> tuple[TransformSpec | None, list[str]]:
    """
    Read the *current* preprocessing chain, not the whole history.

    `records` is `session.state.preprocessing` — `undo` pops a record, so a replayed
    chain is the truth. A difference anywhere in that chain means the modelled
    quantity is a change, not a level: back-transforming a level is then meaningless
    and the caller is warned instead of being handed a wrong number.
    """
    recs = list(records)
    active: TransformSpec | None = None
    warnings: list[str] = []
    for rec in recs:
        step = getattr(rec, "step", None) or (rec.get("step") if isinstance(rec, dict) else None)
        method = getattr(rec, "method", None) or (rec.get("method") if isinstance(rec, dict) else None)
        params = getattr(rec, "params", None) or (rec.get("params") if isinstance(rec, dict) else None)
        if step == "transform" and method in {"log", "boxcox", "yeojohnson", "sqrt"}:
            active = TransformSpec(method=method, params=dict(params or {}), label=method)
        elif step == "transform" and method == "none" and active is not None:
            # Applying "no transform" does not undo an earlier log; keep the chain honest.
            warnings.append(
                "A 'leave on the original scale' operation was applied after a transform; it is a "
                "no-op, so the earlier transform is still active."
            )
    for rec in recs:
        step = getattr(rec, "step", None) or (rec.get("step") if isinstance(rec, dict) else None)
        method = getattr(rec, "method", None) or (rec.get("method") if isinstance(rec, dict) else None)
        if step == "difference" or method == "difference":
            warnings.append(
                "The modelled series contains a differencing operation, so forecasts are of "
                "changes (or log-changes), not of levels. Undo the differencing in Step 3 before "
                "asking for a back-transformed level forecast."
            )
            break
    return active, warnings


@dataclass
class BackTransformInfo:
    """Exactly what was done to the scale, for the UI, the report and the export."""

    applied: bool
    method: str
    transform: dict[str, Any] | None
    scale: str
    formula: str
    point_definition: str
    notes: list[str] = field(default_factory=list)
    factor: float | None = None
    clipped_bounds: int = 0
    clipped_paths: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "method": self.method,
            "transform": self.transform,
            "scale": self.scale,
            "formula": self.formula,
            "point_definition": self.point_definition,
            "factor": self.factor,
            "clipped_bounds": self.clipped_bounds,
            "clipped_paths": self.clipped_paths,
            "notes": list(self.notes),
            "silent_exponentiation": False,
        }


def apply_back_transform(
    *,
    point: np.ndarray,
    bands: Sequence[Any],
    paths: np.ndarray | None,
    spec: TransformSpec | None,
    method: str | None,
    residuals: np.ndarray | None,
) -> tuple[np.ndarray, list[Any], np.ndarray | None, BackTransformInfo]:
    """
    Transform a whole bundle (point, every band, every simulated path) to the
    original scale. `bands` are `IntervalBand`s; their `construction` is extended,
    never replaced, so coverage and construction remain attached (§6.8).
    """
    if spec is None:
        info = BackTransformInfo(
            applied=False,
            method="none",
            transform=None,
            scale="original",
            formula="y = z (no transform was applied to the modelled series)",
            point_definition="the model's own point forecast on the original scale",
            notes=[
                "No back-transformation was needed: the fitted series is on the original scale.",
                "Naive / smearing / lognormal apply only when a log, Box–Cox, Yeo–Johnson or "
                "square-root transform is active.",
            ],
        )
        return np.asarray(point, float).copy(), list(bands), paths, info

    if method not in METHODS:
        raise ValueError(
            f"The series has a {spec.describe()}; choose one of {', '.join(METHODS)} before "
            "back-transforming. No exponentiation is performed without an explicit choice."
        )

    z = np.asarray(point, dtype=float).reshape(-1)
    h = z.size
    # Variance on the transformed scale from the model's own 95% band.
    var_z: np.ndarray | None = None
    for b in bands:
        if abs(float(b.coverage) - 0.95) < 1e-9:
            width = np.maximum(np.asarray(b.upper, float) - np.asarray(b.lower, float), 0.0)
            var_z = (width / (2.0 * _Z975)) ** 2
            break

    r = np.asarray(residuals if residuals is not None else [], dtype=float).reshape(-1)
    r = r[np.isfinite(r)]

    if method == "lognormal" and not spec.is_log_scale:
        raise ValueError(
            "The lognormal bias adjustment assumes a log-scale transform (log, or Box–Cox with "
            "lambda about zero). This series uses "
            f"{spec.describe()}; choose 'smearing' or 'naive' instead."
        )
    if method == "lognormal" and var_z is None:
        raise ValueError(
            "The lognormal adjustment needs the model's 95% analytic interval to estimate "
            "Var(Z); it was not available. Choose 'smearing' or 'naive'."
        )
    if method == "smearing" and r.size < 8:
        raise ValueError(
            f"Smearing needs at least eight in-sample residuals on the transformed scale; {r.size} "
            "were available. Choose 'naive' or 'lognormal'."
        )

    # ---- the point: one of three estimators of E[Y] -------------------------- #
    factor: float | None = None
    if method == "naive":
        new_point, _ = spec.inverse(z)
        formula = "point = g^-1(point_z)"
        point_def = (
            "the inverse-transformed point is the MEDIAN on the original scale; it is not "
            "bias-adjusted and understates the mean when variance is appreciable"
        )
    elif method == "smearing":
        # Duan's smearing estimate, generalised: average g^-1(point_z + e_r) over the
        # in-sample residuals. For a log transform this is exactly exp(point_z)*mean(exp(e)).
        smeared, _ = spec.inverse(z[None, :] + r[:, None])
        new_point = np.asarray(smeared, float).mean(axis=0)
        if spec.is_log_scale:
            factor = float(np.mean(np.exp(np.clip(spec.log_k * r, -700.0, 700.0))))
        formula = "point[h] = mean_r g^-1(point_z[h] + e_r), e_r = in-sample residuals"
        point_def = (
            "Duan's nonparametric smearing estimate: the MEAN on the original scale, using the "
            f"empirical residual distribution ({r.size} residuals), with no normality assumption"
        )
    else:
        assert var_z is not None
        k = spec.log_k
        new_point = np.exp(np.clip(k * z + (k ** 2) * var_z / 2.0, -700.0, 700.0)) - spec.offset
        formula = "point[h] = exp(k*point_z[h] + k^2*Var(Z_h)/2) - offset"
        point_def = (
            "the MEAN of a lognormal predictive distribution: Var(Z_h) is estimated from the "
            "model's own 95% analytic interval, so it assumes normality on the log scale"
        )

    # ---- interval endpoints: exact monotone transforms, never rescaled -------- #
    out_bands: list[Any] = []
    clipped_bounds = 0
    for b in bands:
        lo, c1 = spec.inverse(np.asarray(b.lower, float))
        hi, c2 = spec.inverse(np.asarray(b.upper, float))
        clipped_bounds += c1 + c2
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        from app.ts.forecast.intervals import IntervalBand

        out_bands.append(
            IntervalBand(
                coverage=b.coverage,
                lower=lo,
                upper=hi,
                construction=(
                    f"{b.construction} Back-transformed through {spec.describe()} by the exact "
                    "monotone inverse (quantiles are transformed; they are NOT multiplied by the "
                    f"bias-adjustment factor). Bias adjustment method: {method}."
                ),
                source=b.source,
                alpha=b.alpha,
                nesting_adjusted=b.nesting_adjusted,
            )
        )

    # ---- simulated paths: transform each draw, so combinations still work ----- #
    new_paths: np.ndarray | None = None
    clipped_paths = 0
    if paths is not None:
        vals, clipped = spec.inverse(np.asarray(paths, float))
        new_paths = np.asarray(vals, float)
        clipped_paths = clipped

    notes = [
        f"Active transform: {spec.describe()}.",
        "The point and the interval endpoints answer different questions: bias adjustment changes "
        "the reported mean, while quantiles follow the monotone transform exactly.",
        "Interval endpoints are NOT multiplied by a smearing factor; doing so would break "
        "q_p(Y) = g^-1(q_p(Z)).",
        "This choice was supplied explicitly by the caller — nothing was exponentiated silently.",
    ]
    if method == "naive":
        notes.append("Naive: no bias adjustment. On a log scale this quotes the median.")
    if method == "smearing":
        notes.append("Smearing: no distributional assumption, but it needs representative residuals.")
    if method == "lognormal":
        notes.append("Lognormal: parametric; if the log-scale residuals are skewed, use smearing.")
    if clipped_bounds or clipped_paths:
        notes.append(
            f"{clipped_bounds} interval endpoint(s) and {clipped_paths} simulated path value(s) "
            "fell outside the inverse transform's domain and were clipped to the nearest valid "
            "value; the affected tails are conservative."
        )

    info = BackTransformInfo(
        applied=True,
        method=method,
        transform=spec.as_dict(),
        scale="original",
        formula=formula,
        point_definition=point_def,
        notes=notes,
        factor=factor,
        clipped_bounds=clipped_bounds,
        clipped_paths=clipped_paths,
    )
    return np.asarray(new_point, float), out_bands, new_paths, info


__all__ = [
    "METHODS",
    "BackTransformInfo",
    "TransformSpec",
    "apply_back_transform",
    "detect_transform",
    "inverse_boxcox",
    "inverse_yeojohnson",
]
