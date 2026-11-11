"""
ts/forecast/reconcile.py — optional aggregate <-> disaggregate reconciliation (§5.11).

A hierarchy is coherent when the bottom-level forecasts add up to the aggregate.
Independent base forecasts almost never do. Three standard repairs are offered:

    bottom_up   trust the disaggregate forecasts; aggregate = S b
    top_down    trust the aggregate; split it by the base bottom proportions
    ols         weighted least-squares projection (the diagonal-W MinT/OLS case),
                y~ = S (S' W^-1 S)^-1 S' W^-1 y_hat

`S = [I_b; 1_b']` is the summing matrix. W_h is diagonal: each series' forecast-error
variance at step h, estimated from that series' own 95% analytic interval. The
diagonal-W assumption (independent base errors) is stated in the result, because
ignoring cross-series covariance is the one approximation a reader cannot see.

Intervals are propagated with the same linear map G_h (`Var = G W G'`); no interval
is drawn by eye. Coherence is *checked* after the fact and the maximum residual is
reported rather than assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy import stats

_Z975 = float(stats.norm.ppf(0.975))
METHODS: tuple[str, ...] = ("bottom_up", "top_down", "ols")
ReconcileMethod = Literal["bottom_up", "top_down", "ols"]


@dataclass
class Reconciliation:
    method: str
    bottom_names: list[str]
    index: list[str]
    base_bottom: np.ndarray
    base_total: np.ndarray
    reconciled_bottom: np.ndarray
    reconciled_total: np.ndarray
    base_bottom_sd: np.ndarray
    base_total_sd: np.ndarray
    total_lower95: np.ndarray
    total_upper95: np.ndarray
    coherence_error: float
    base_coherence_error: float
    construction: str
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "bottom_names": list(self.bottom_names),
            "index": list(self.index),
            "base_bottom": [[float(v) for v in row] for row in self.base_bottom],
            "base_total": [float(v) for v in self.base_total],
            "reconciled_bottom": [[float(v) for v in row] for row in self.reconciled_bottom],
            "reconciled_total": [float(v) for v in self.reconciled_total],
            "total_lower95": [float(v) for v in self.total_lower95],
            "total_upper95": [float(v) for v in self.total_upper95],
            "coherence_error": self.coherence_error,
            "base_coherence_error": self.base_coherence_error,
            "construction": self.construction,
            "notes": list(self.notes),
            "coverage": 0.95,
            "interval_construction": (
                "95% analytic interval on the reconciled total: linear variance propagation "
                "Var(y~_h) = G_h W_h G_h' from each base series' own 95% analytic interval; "
                "diagonal W (independent base errors) and fixed top-down proportions are assumed."
            ),
        }


def _as_2d(values: Any, names: Sequence[str], horizon: int, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1 and len(names) == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape != (len(names), horizon):
        raise ValueError(
            f"{what} must have shape ({len(names)}, {horizon}) (one row per bottom series); "
            f"got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{what} contains a non-finite value; reconciliation needs real numbers.")
    return arr


def reconcile(
    *,
    bottom_names: Sequence[str],
    base_bottom: Any,
    base_total: Any,
    bottom_sd: Any | None = None,
    total_sd: Any | None = None,
    method: ReconcileMethod = "ols",
    index: Sequence[str] | None = None,
) -> Reconciliation:
    """
    Reconcile one two-level hierarchy. Pure function: no session, no fitting, no I/O.

    `bottom_sd`/`total_sd` are per-series, per-step standard deviations on the base
    forecasts (shape (b, H) and (H,)). When omitted they are inferred from the
    spread of the base bottom forecasts — a disclosed fallback, not a silent one.
    """
    names = [str(x) for x in bottom_names]
    b = len(names)
    if b < 2:
        raise ValueError("A hierarchy needs at least two bottom series to reconcile.")
    if b > 12:
        raise ValueError("This endpoint reconciles at most 12 bottom series at a time.")
    if method not in METHODS:
        raise ValueError(f"Unknown reconciliation method '{method}'; choose one of {', '.join(METHODS)}.")

    bt = np.asarray(base_total, dtype=float).reshape(-1)
    h = int(bt.size)
    if h < 1:
        raise ValueError("A reconciliation needs at least one forecast step.")
    if not np.all(np.isfinite(bt)):
        raise ValueError("The base aggregate forecast contains a non-finite value.")
    B = _as_2d(base_bottom, names, h, "Base bottom forecasts")

    if bottom_sd is None:
        # Disclosed fallback: use the cross-series spread of the base forecasts as a
        # common scale. It is not an error variance, and the note says so.
        spread = float(np.mean(np.std(B, axis=0, ddof=1)))
        sd_b = np.full((b, h), max(spread, 1e-6))
        sd_note = (
            "No base interval widths were supplied, so W was built from the cross-series spread of "
            "the base bottom forecasts — a scale proxy, not an estimated error variance."
        )
    else:
        sd_b = _as_2d(bottom_sd, names, h, "Base bottom standard deviations")
        sd_b = np.maximum(sd_b, 1e-6)
        sd_note = "W uses each bottom series' own 95% analytic interval width / (2*1.959964)."
    if total_sd is None:
        sd_a = np.sqrt(np.sum(sd_b ** 2, axis=0))
        sd_note += " The aggregate's variance is the sum of the bottom variances under diagonal W."
    else:
        sd_a = np.maximum(np.asarray(total_sd, dtype=float).reshape(-1), 1e-6)
        if sd_a.size != h:
            raise ValueError("The aggregate standard deviations must have one value per step.")

    S = np.vstack([np.eye(b), np.ones((1, b))])           # (b+1, b)
    yhat = np.vstack([B, bt[None, :]])                     # (b+1, h)
    new_bottom = np.empty((b, h), dtype=float)
    var_total = np.zeros(h, dtype=float)
    projected_total = np.zeros(h, dtype=float)
    equal_share_used = False

    for j in range(h):
        w = np.concatenate([sd_b[:, j], [sd_a[j]]]) ** 2   # diagonal variances
        winv = 1.0 / w
        sw = S.T @ (winv[:, None] * S)                     # (b, b)
        if method == "bottom_up":
            P = np.hstack([np.eye(b), np.zeros((b, 1))])   # ignore the aggregate
            G = S @ P
        elif method == "top_down":
            total_bottom = float(np.sum(B[:, j]))
            if abs(total_bottom) < 1e-12:
                props = np.full(b, 1.0 / b)
                prop_note = True
            else:
                props = B[:, j] / total_bottom
                prop_note = False
            if np.any(props < 0):
                props = np.maximum(props, 0.0)
                s = float(props.sum())
                props = props / s if s > 0 else np.full(b, 1.0 / b)
                prop_note = True
            P = np.zeros((b, b + 1))
            P[:, -1] = props
            G = S @ P
            if prop_note:
                equal_share_used = True
        else:
            P = np.linalg.pinv(sw) @ (S.T * winv[None, :])  # (b, b+1)
            G = S @ P
        ytilde = G @ yhat[:, j]                           # (b+1,)
        new_bottom[:, j] = ytilde[:b]
        projected_total[j] = ytilde[b]
        var_total[j] = float(G[-1] @ (w * G[-1]))

    total = new_bottom.sum(axis=0)
    # Numerical coherence check against the *projected* aggregate, not against a
    # total defined as the sum (which would be zero by construction and prove nothing).
    coherence = float(np.max(np.abs(projected_total - total)))
    base_coherence = float(np.max(np.abs(bt - B.sum(axis=0))))

    if equal_share_used:
        sd_note += (
            " At a zero-sum or negative-share step, top-down shares fell back to equal (1/b)."
        )
    half = _Z975 * np.sqrt(np.maximum(var_total, 0.0))
    construction = (
        f"{method} reconciliation of {b} bottom series and their aggregate over {h} steps. "
        "S = [I_b; 1_b']; W_h = diagonal base forecast-error variances from each series' own "
        "95% analytic interval. "
        + (
            "OLS: y~_h = S (S'W^-1 S)^-1 S'W^-1 y_hat_h (weighted least-squares projection, "
            "diagonal-W MinT)."
            if method == "ols" else
            "Bottom-up: the aggregate is replaced by the sum of the disaggregate base forecasts."
            if method == "bottom_up" else
            "Top-down: the aggregate is kept and split by the base bottom proportions at each step."
        )
    )
    return Reconciliation(
        method=method,
        bottom_names=names,
        index=[str(x) for x in (index if index is not None else range(1, h + 1))],
        base_bottom=B,
        base_total=bt,
        reconciled_bottom=new_bottom,
        reconciled_total=total,
        base_bottom_sd=sd_b,
        base_total_sd=sd_a,
        total_lower95=total - half,
        total_upper95=total + half,
        coherence_error=coherence,
        base_coherence_error=base_coherence,
        construction=construction,
        notes=[
            "Diagonal W assumes base forecast errors are independent across series; the true "
            "covariance is usually positive and would change the OLS weights.",
            sd_note,
            "Intervals are propagated through the same linear map as the point forecasts. "
            "Top-down proportions are treated as fixed, which understates their uncertainty.",
            "These are base forecasts computed by the app's own fitted models; no number was "
            "invented by the UI.",
        ],
    )


__all__ = ["METHODS", "Reconciliation", "reconcile"]
