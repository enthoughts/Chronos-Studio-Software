"""
ts/correlate/_ccf.py — cross-correlation and pre-whitening (§5.6).

Cross-correlation between a series and an input is one of the most over-interpreted numbers in
applied time series. Two series that are both trending show a large CCF at every lag without any
causal link; two autocorrelated series produce spurious peaks at arbitrary lags. The fix is
**pre-whitening**: fit a filter to the input, apply the *same* filter to both series, and correlate
the residuals. That is what this module does, and it reports both the raw and the pre-whitened
version so the difference is visible rather than asserted.

The default is never to claim direction. A peak at a negative lag means the input *leads*; a peak
at a positive lag means it lags. Both are reported with that sentence spelled out, because "the
cross-correlation is 0.7 at lag −2" is not an answer to any question a user actually asked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CCF:
    """Cross-correlations at lags −max_lag … +max_lag."""

    lags: list[int]
    values: list[float]
    band: float
    n: int
    x_name: str
    y_name: str
    prewhitened: bool = False
    filter_ar: list[float] = field(default_factory=list)

    def peak(self) -> dict[str, Any]:
        i = int(np.argmax(np.abs(np.asarray(self.values, dtype=float))))
        lag = self.lags[i]
        val = self.values[i]
        return {
            "lag": lag,
            "value": float(val),
            "significant": bool(abs(val) > self.band),
            "leads_lags": _lead_phrase(lag, self.x_name, self.y_name),
        }

    def rows(self) -> list[dict[str, Any]]:
        return [
            {"lag": int(l), "ccf": round(float(v), 4), "band": round(float(self.band), 4),
             "significant": bool(abs(v) > self.band)}
            for l, v in zip(self.lags, self.values)
        ]


def _lead_phrase(lag: int, x_name: str, y_name: str) -> str:
    """
    Translate a lag into words, in the convention this module actually computes.

    The value at lag k is corr(x_t, y_{t+k}). So if y_t = x_{t−d} (y is x delayed by d), the peak
    sits at **k = +d** — a positive lag means the input *leads*. Getting this backwards is the
    single most common CCF error: it turns "the driver leads" into "the driver lags" and reads
    perfectly plausibly either way.
    """
    if lag == 0:
        return f"{x_name} and {y_name} move together with no lag"
    if lag > 0:
        return f"{x_name} leads {y_name} by {lag} observation(s)"
    return f"{x_name} lags {y_name} by {-lag} observation(s)"


def _align(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join on the index so a gap in either series cannot silently shift the alignment."""
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1, join="inner").dropna()
    if len(joined) < 10:
        raise ValueError("The two series share too few observations to correlate.")
    return (joined["x"].to_numpy(dtype=float), joined["y"].to_numpy(dtype=float))


def ccf(
    x: pd.Series,
    y: pd.Series,
    *,
    max_lag: int = 24,
    x_name: str = "input",
    y_name: str = "series",
    prewhiten: bool = True,
) -> CCF:
    """
    Cross-correlation of `y` against `x` at lags −h … +h, optionally pre-whitened.

    Pre-whitening fits a low-order AR model to the input `x` and filters *both* series with it. The
    CCF of the filtered residuals is the one that can be read as a relationship, because the
    shared autocorrelation has been removed from both sides.
    """
    xv, yv = _align(x, y)
    n = len(xv)
    max_lag = int(max(1, min(max_lag, n // 3)))

    ar_coefs: list[float] = []
    if prewhiten:
        xv, yv, ar_coefs = _prewhiten_pair(xv, yv)
        n = len(xv)
        max_lag = int(max(1, min(max_lag, n // 3)))

    def _norm(v: np.ndarray) -> np.ndarray:
        d = v - v.mean()
        sd = float(np.sqrt(np.sum(d * d)))
        return d / sd if sd > 0 else d

    xs, ys = _norm(xv), _norm(yv)

    lags: list[int] = []
    values: list[float] = []
    for k in range(-max_lag, max_lag + 1):
        if k < 0:
            v = float(np.dot(xs[-k:], ys[: n + k]))
        elif k == 0:
            v = float(np.dot(xs, ys))
        else:
            v = float(np.dot(xs[: n - k], ys[k:]))
        lags.append(k)
        values.append(v)

    band = float(1.96 / np.sqrt(n))
    return CCF(lags=lags, values=values, band=band, n=n, x_name=x_name, y_name=y_name,
               prewhitened=prewhiten, filter_ar=ar_coefs)


def _prewhiten_pair(x: np.ndarray, y: np.ndarray, order: int | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Fit AR(p) to x by Yule–Walker, filter x and y with the same coefficients.

    The order is the smallest p ≤ 5 for which the residual ACF of x is inside its band at the
    first 10 lags — i.e. the point at which the input has been whitened. If no order achieves that
    (possible on a unit-root input), the largest one is used and the caller is told the filter was
    imperfect through `filter_ar` plus the caveat the API attaches.
    """
    def fit(p: int) -> tuple[np.ndarray, np.ndarray]:
        z = x - x.mean()
        r = np.array([np.dot(z[:-k], z[k:]) / np.dot(z, z) for k in range(1, p + 1)])
        # Yule–Walker: R φ = r  (Levinson–Durbin)
        phi = np.zeros(p + 1)
        v = 1.0
        for k in range(1, p + 1):
            num = r[k - 1] - sum(phi[j] * r[k - j - 1] for j in range(1, k))
            den = 1.0 - sum(phi[j] * r[j - 1] for j in range(1, k))
            phi[k] = num / den if abs(den) > 1e-12 else 0.0
            for j in range(1, k):
                phi[j] = phi[j] - phi[k] * phi[k - j]
        coefs = phi[1:]

        def filt(a: np.ndarray) -> np.ndarray:
            out = np.zeros(len(a))
            for t in range(len(a)):
                acc = a[t]
                for j, c in enumerate(coefs, start=1):
                    acc -= c * (a[t - j] if t - j >= 0 else 0.0)
                out[t] = acc
            return out

        return filt(x), filt(y)

    best: tuple[np.ndarray, np.ndarray, list[float]] | None = None
    max_order = order if order is not None else 5
    for p in range(1, max_order + 1):
        fx, fy = fit(p)
        fx = fx[p:]
        fy = fy[p:]
        coefs = [float(c) for c in _last_coefs(p, x)]
        best = (fx, fy, coefs)
        rr = fx - fx.mean()
        den = float(np.dot(rr, rr))
        if den <= 0:
            break
        acf_res = np.array([float(np.dot(rr[:-k], rr[k:]) / den) for k in range(1, 11)])
        if np.all(np.abs(acf_res) < 1.96 / np.sqrt(len(fx))):
            break
    assert best is not None
    return best


def _last_coefs(p: int, x: np.ndarray) -> np.ndarray:
    """Refit to return the coefficients actually used (kept separate for clarity)."""
    z = x - x.mean()
    r = np.array([np.dot(z[:-k], z[k:]) / np.dot(z, z) for k in range(1, p + 1)])
    phi = np.zeros(p + 1)
    for k in range(1, p + 1):
        num = r[k - 1] - sum(phi[j] * r[k - j - 1] for j in range(1, k))
        den = 1.0 - sum(phi[j] * r[j - 1] for j in range(1, k))
        phi[k] = num / den if abs(den) > 1e-12 else 0.0
        for j in range(1, k):
            phi[j] = phi[j] - phi[k] * phi[k - j]
    return phi[1:]


def spurious_warning(c: CCF, raw: CCF | None = None) -> str | None:
    """
    Say plainly when a CCF peak is untrustworthy.

    Two cases matter: (a) pre-whitening changed the peak's lag, which means the raw peak was the
    shared autocorrelation; (b) the peak sits at a lag where the input's own autocorrelation is
    still large, which means the two series are correlated through their memory rather than at a
    shared event.
    """
    peak = c.peak()
    if raw is not None and raw.peak()["lag"] != peak["lag"]:
        return (
            f"Pre-whitening moved the peak from lag {raw.peak()['lag']} to lag {peak['lag']}. The "
            "raw peak was mostly the two series' own autocorrelation, which is why the "
            "pre-whitened version is the one to read."
        )
    if not peak["significant"]:
        return (
            f"No lag is outside the band (±{c.band:.3f}). That is a real answer: these two series "
            "show no linear relationship at the lags examined."
        )
    return None


__all__ = ["CCF", "ccf", "spurious_warning"]
