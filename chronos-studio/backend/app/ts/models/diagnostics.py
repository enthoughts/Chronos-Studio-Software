"""
ts/models/diagnostics.py — one diagnostic block, computed identically for every model.

If each wrapper computed its own Ljung–Box, the comparison table would be comparing tests rather
than models. Everything here is deliberately the same code for a mean model and a SARIMAX.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.ts.models.base import FitDiagnostics


def _safe(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(v) else v


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float | None, int | None]:
    """Ljung–Box Q on a residual series. High p = nothing left to explain."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < lags + 5 or np.allclose(x, x[0]):
        return None, None
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 0:
        return None, None
    q = 0.0
    for k in range(1, lags + 1):
        rk = float((xc[k:] * xc[:-k]).sum() / denom)
        q += rk * rk / (n - k)
    q *= n * (n + 2)
    return _safe(stats.chi2.sf(q, lags)), lags


def jarque_bera(x: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8 or np.allclose(x, x[0]):
        return None
    try:
        return _safe(stats.jarque_bera(x).pvalue)
    except Exception:  # pragma: no cover
        return None


def arch_lm_p(x: np.ndarray, lags: int = 5) -> float | None:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < lags + 10:
        return None
    sq = (x - x.mean()) ** 2
    Y = sq[lags:]
    if np.allclose(Y, Y[0]):
        return None
    X = np.column_stack([np.ones(Y.size)] + [sq[lags - k - 1: Y.size + lags - k - 1] for k in range(lags)])
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        ss_tot = float(((Y - Y.mean()) ** 2).sum())
        if ss_tot <= 0:
            return None
        r2 = 1.0 - float((resid ** 2).sum()) / ss_tot
        return _safe(stats.chi2.sf(Y.size * r2, lags))
    except Exception:  # pragma: no cover
        return None


def _acf1(x: np.ndarray) -> float | None:
    if x.size < 4:
        return None
    a = x[:-1] - x[:-1].mean()
    b = x[1:] - x[1:].mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return _safe(float((a * b).sum() / denom)) if denom > 0 else None


def _mape(y: np.ndarray, fit: np.ndarray) -> tuple[float | None, str]:
    """
    MAPE, or an explicit refusal.

    A percentage error is undefined wherever the truth is zero and meaningless wherever it is near
    zero, which is every count and every intermittent series. Returning a number there would be a
    fabrication, so this returns `None` plus the reason.
    """
    n_zero = int((y == 0).sum())
    if n_zero:
        return None, f"MAPE is undefined: {n_zero} of the actual values are zero."
    denom = np.abs(y)
    if np.any(denom < 1e-8):
        return None, "MAPE is undefined: some actual values are effectively zero."
    return _safe(float(np.mean(np.abs((y - fit) / denom)) * 100)), ""


def naive_scale(y: np.ndarray, period: int | None) -> float | None:
    """
    The MASE denominator: mean absolute change of the appropriate naive forecast.

    Seasonal when a period exists and there is more than one full cycle, otherwise first differences
    — the definition Hyndman gives, and the reason MASE is comparable across series.
    """
    if y.size < 3:
        return None
    if period and period >= 2 and y.size > period + 1:
        d = np.abs(y[period:] - y[:-period])
    else:
        d = np.abs(np.diff(y))
    if d.size == 0:
        return None
    scale = float(d.mean())
    return scale if scale > 0 else None


def diagnose(
    y: pd.Series,
    *,
    fitted: np.ndarray | None,
    resid: np.ndarray | None,
    period: int | None = None,
    ban_mape: bool = False,
) -> FitDiagnostics:
    """
    The whole block. Absent pieces stay `None` — the table shows "—" and means it.
    """
    d = FitDiagnostics()
    yv = pd.Series(y).astype(float).to_numpy()

    if resid is not None:
        r = np.asarray(resid, dtype=float)
        r = r[np.isfinite(r)]
        if r.size >= 8:
            d.ljung_box_p, d.ljung_box_lag = ljung_box(r, lags=min(10, max(5, r.size // 5)))
            d.jarque_bera_p = jarque_bera(r)
            d.arch_lm_p = arch_lm_p(r)
            d.residual_sd = _safe(r.std(ddof=1))
            d.residual_mean = _safe(r.mean())
            d.residual_acf1 = _acf1(r)

    if fitted is not None:
        fv = np.asarray(fitted, dtype=float)
        # A one-step-ahead fit is one shorter than the series (there is no prediction for t=1), so
        # align from the end rather than assuming equal lengths.
        if fv.size != yv.size:
            yv = yv[-fv.size:] if fv.size < yv.size else yv
            fv = fv[-yv.size:]
        mask = np.isfinite(fv) & np.isfinite(yv)
        if mask.sum() >= 5:
            a, b = yv[mask], fv[mask]
            err = a - b
            d.rmse = _safe(np.sqrt(np.mean(err ** 2)))
            d.mae = _safe(np.mean(np.abs(err)))
            if ban_mape:
                d.mape = None
                d.warnings.append(
                    "MAPE is not reported: for count and intermittent series a percentage error is "
                    "undefined at zero and misleading near it."
                )
            else:
                d.mape, why = _mape(a, b)
                if why:
                    d.warnings.append(why)
            scale = naive_scale(a, period)
            if scale:
                d.mase = _safe(float(np.mean(np.abs(err)) / scale))
    return d


__all__ = ["arch_lm_p", "diagnose", "jarque_bera", "ljung_box", "naive_scale"]
