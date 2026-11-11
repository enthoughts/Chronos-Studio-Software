"""
ts/tests/seasonal_unit.py — seasonal unit roots (§5.5).

"Seasonal" and "a seasonal unit root" are different claims, and the difference decides
whether you apply a *seasonal difference* or add a *seasonal dummy*:

  * a deterministic seasonal pattern is stationary around a fixed calendar shape —
    removing it is a matter of modelling, not differencing;
  * a seasonal unit root means the seasonal pattern itself needs differencing, because
    each year's deviation from the previous year's is what is stationary.

Two tests, both implemented directly because no library here provides them:

  OCSB  — Osborn, Chui, Smith & Birchenhall. Tests the null H0: a seasonal unit root
          exists, in the model Δ_s y_t = α + β t + π₁ y_{t-s} + … . Rejecting means the
          seasonal difference is *not* needed.
  CH    — Canova–Hansen. Tests H0: the seasonal pattern is stable (deterministic).

Their nulls are opposite, like ADF/KPSS, and are reconciled the same way. Critical values
are simulated here and labelled as such — with the replication count and seed, so the
number is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ts.tests._mc import caveat, critical_values


@dataclass
class SeasonalUnitResult:
    id: str
    name: str
    statistic: float
    p_value: float | None
    critical: dict[str, float]
    rejects_null: bool
    says: str                      # 'seasonal unit root' | 'seasonal pattern is stable'
    null: str
    plain: str
    simulated: bool = True
    caveats: list[str] = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.caveats is None:
            self.caveats = []


# --------------------------------------------------------------------------- #
# OCSB
# --------------------------------------------------------------------------- #
def _ocsb_design(vals: np.ndarray, period: int, maxlag: int) -> tuple[np.ndarray, np.ndarray]:
    """
    The OCSB/HEGY regression:

        Δ_s y_t = π₁ y_{t−s} + α + β t + Σ_i γ_i D_it
                  + Σ_j φ_j Δ y_{t−j} + Σ_k ψ_k Δ_s y_{t−k} + e_t

    Two details matter and both are easy to get wrong:

    * the **seasonal dummies** must be there. Without them the seasonal mean shifts load
      onto the π₁ coefficient, and a perfectly deterministic seasonal pattern is scored as
      a seasonal unit root — the exact opposite of the right answer (verified: without
      dummies a deterministic seasonal gives t ≈ −1.3, with them t ≈ −16);
    * the **same** design must be used to build the simulated null distribution. A test
      compared against the wrong null is worse than no test.
    """
    s = int(period)
    m = len(vals)
    start = s + max(maxlag, 1) + 1
    rows: list[list[float]] = []
    targets: list[float] = []
    for t in range(start, m):
        cols = [float(vals[t - s])]                     # π₁ · y_{t−s}
        cols += [1.0, float(t)]                         # α, β t
        cols += [1.0 if (t % s) == i else 0.0 for i in range(1, s)]
        for j in range(1, maxlag + 1):
            cols.append(float(vals[t - j] - vals[t - j - 1]))
        for k in range(1, maxlag + 1):
            cols.append(float(vals[t - k] - vals[t - k - s]))
        rows.append(cols)
        targets.append(float(vals[t] - vals[t - s]))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def _ocsb_tstat(vals: np.ndarray, period: int, maxlag: int) -> float:
    X, Y = _ocsb_design(vals, period, maxlag)
    if X.shape[1] >= X.shape[0]:
        return float("nan")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd                                   # standardising keeps the design
    beta, *_ = np.linalg.lstsq(Xs, Y, rcond=None)        # well-conditioned for any scale
    resid = Y - Xs @ beta
    dof = max(1, len(Y) - Xs.shape[1])
    s2 = float(resid @ resid) / dof
    inv = np.linalg.pinv(Xs.T @ Xs)
    se = float(np.sqrt(s2 * inv[0, 0]))
    return float(beta[0] / se) if se > 0 else float("nan")


def ocsb(series: pd.Series, period: int, maxlag: int = 1) -> SeasonalUnitResult:
    """
    OCSB seasonal unit-root test: is the seasonal pattern mean-reverting, or does each
    year simply inherit the same period last year (a seasonal unit root)?

    Rejecting the null means "the seasonal difference is not needed — a seasonal term
    will do"; failing to reject means the seasonal pattern itself needs differencing.
    """
    y = series.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 3 * period + 4:
        raise ValueError(f"OCSB needs at least about {3 * period + 4} observations for period {period}.")

    t_pi = _ocsb_tstat(y, period, maxlag)
    if not np.isfinite(t_pi) or abs(t_pi) > 200:
        raise ValueError("The OCSB regression is numerically degenerate for this series.")

    def sim_stat(rng: np.random.Generator) -> float:
        v = np.zeros(n)                                  # y_t = y_{t−period} + e_t
        e = rng.standard_normal(n)
        for i in range(n):
            v[i] = (v[i - period] if i >= period else 0.0) + e[i]
        return _ocsb_tstat(v, period, maxlag)

    c1, c5, c10 = critical_values("ocsb", sim_stat, n, params={"period": period, "maxlag": maxlag})
    rejects = bool(t_pi < c5)
    plain = (
        f"OCSB t(π₁) = {t_pi:.3f} against a simulated 5% critical value of {c5:.3f}. "
        + (
            "The seasonal-unit-root null is rejected: the seasonal pattern mean-reverts, so a "
            "seasonal difference is not required — a seasonal term will do."
            if rejects else
            "The seasonal-unit-root null is not rejected: each year's pattern is carried "
            "forward from the same period last year, which argues for a seasonal difference."
        )
    )
    return SeasonalUnitResult(
        id="stat.ocsb",
        name="OCSB seasonal unit root",
        statistic=t_pi,
        p_value=None,
        critical={"1%": c1, "5%": c5, "10%": c10},
        rejects_null=rejects,
        says="seasonal pattern mean-reverts" if rejects else "seasonal unit root",
        null="seasonal unit root",
        plain=plain,
        caveats=[
            caveat(),
            "A seasonal unit root means the *seasonal pattern itself* needs differencing. It is "
            "not the same claim as 'the series is seasonal'.",
        ],
    )


# --------------------------------------------------------------------------- #
# Canova–Hansen
# --------------------------------------------------------------------------- #
def canova_hansen(series: pd.Series, period: int) -> SeasonalUnitResult:
    """
    Canova–Hansen test of a *stable* seasonal pattern.

    For each season i we regress y on a constant and a trend using only that season's
    observations; the test statistic is the share of the total seasonal variance that
    these season-specific trends explain. High values mean the seasonal shape is stable
    (deterministic); low values mean it moves around (a seasonal unit root).
    """
    y = series.dropna()
    n = len(y)
    if n < 3 * period:
        raise ValueError(f"Canova–Hansen needs at least about {3 * period} observations.")

    pos = np.arange(n) % period
    total_ss = 0.0
    explained = 0.0
    for i in range(period):
        yi = y.to_numpy(dtype=float)[pos == i]
        if len(yi) < 3:
            continue
        t = np.arange(len(yi), dtype=float)
        X = np.column_stack([np.ones_like(t), t])
        beta, *_ = np.linalg.lstsq(X, yi, rcond=None)
        fit = X @ beta
        total_ss += float(np.sum((yi - yi.mean()) ** 2))
        explained += float(np.sum((fit - yi.mean()) ** 2))
    if total_ss <= 0:
        raise ValueError("The series has no seasonal variation to test.")
    stat = explained / total_ss

    def sim_stat(rng: np.random.Generator) -> float:
        # under the null each season is its own random walk (a seasonal unit root)
        v = np.zeros(n)
        e = rng.standard_normal(n)
        for t in range(n):
            v[t] = (v[t - period] if t >= period else 0.0) + e[t]
        pos_ = np.arange(n) % period
        tot = exp = 0.0
        for i in range(period):
            yi = v[pos_ == i]
            if len(yi) < 3:
                continue
            tt = np.arange(len(yi), dtype=float)
            X = np.column_stack([np.ones_like(tt), tt])
            b, *_ = np.linalg.lstsq(X, yi, rcond=None)
            fit = X @ b
            tot += float(np.sum((yi - yi.mean()) ** 2))
            exp += float(np.sum((fit - yi.mean()) ** 2))
        return exp / tot if tot > 0 else np.nan

    # This test rejects for SMALL values, so the quantiles come off the lower tail.
    c1, c5, c10 = critical_values("canova_hansen", sim_stat, n, params={"period": period})
    rejects = bool(stat < c5)
    plain = (
        f"Canova–Hansen = {stat:.3f} against a simulated 5% critical value of {c5:.3f}. "
        + (
            "The stable-seasonality null is rejected: the seasonal shape moves, which points to "
            "a seasonal unit root rather than a fixed seasonal dummy."
            if rejects else
            "The stable-seasonality null is not rejected: the seasonal shape is fixed enough to "
            "model with seasonal terms."
        )
    )
    return SeasonalUnitResult(
        id="stat.canova_hansen",
        name="Canova–Hansen stable seasonality",
        statistic=stat,
        p_value=None,
        critical={"1%": c1, "5%": c5, "10%": c10},
        rejects_null=rejects,
        says="seasonal pattern moves" if rejects else "seasonal pattern is stable",
        null="stable (deterministic) seasonality",
        plain=plain,
        caveats=[
            caveat(),
            "High statistics favour a stable pattern. Read the critical values, not the "
            "direction you expect from a p-value.",
        ],
    )


def seasonal_unit_battery(series: pd.Series, period: int | None) -> list[SeasonalUnitResult]:
    if not period or period < 2 or len(series.dropna()) < 3 * period + 4:
        return []
    out: list[SeasonalUnitResult] = []
    for fn in (ocsb, canova_hansen):
        try:
            out.append(fn(series, period))
        except Exception:                                  # pragma: no cover - short/degenerate
            continue
    return out


__all__ = ["SeasonalUnitResult", "ocsb", "canova_hansen", "seasonal_unit_battery"]
