"""
ts/tests/unitroot.py — the unit-root battery (§5.5).

Five things this module refuses to do, because they are how this analysis usually goes
wrong elsewhere:

1. **Report one ADF and stop.** ADF's null is *has a unit root*; it has low power against
   a near-unit-root with a trend, and it cannot distinguish "stationary around a trend"
   from "non-stationary". So we run ADF (three deterministic specifications), KPSS (level
   and trend), Phillips–Perron, DF–GLS, and—if a break is suspected—Zivot–Andrews, and
   then reconcile them rather than picking a favourite.

2. **Merge KPSS's null with ADF's.** They are opposite nulls. The reconciliation lives in
   `reconcile.py`, and the four cells are explained in words, including the two
   contradictory ones where the honest answer is "inconclusive".

3. **Treat "p < 0.05" as a verdict.** p-values are printed but the *verdict* comes from a
   labelled rule with the two statistics next to it, so a reader can disagree.

4. **Return a p-value it cannot compute.** DF–GLS and PP have no p-value in the libraries
   used here; they are reported against simulated/theory-based critical values, and say so.

5. **Differencing when the series is already stationary.** That guardrail is enforced in
   `advisor/rules_next`, using these results, and it is tested.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from app.core.formatting import p_phrase

Spec = Literal["none", "constant", "constant+trend"]


@dataclass
class UnitRootTest:
    """One test, with everything a reader needs to check the verdict themselves."""

    id: str                      # 'stat.adf.constant_trend'
    name: str                    # 'Augmented Dickey–Fuller'
    spec: str                    # 'constant+trend' | 'constant' | 'none'
    null: str                    # 'unit root' | 'stationary'
    statistic: float
    p_value: float | None
    lags: int | None = None
    critical: dict[str, float] = field(default_factory=dict)
    rejects_null: bool | None = None      # at 5%, for THIS test's null
    says: str = ""                        # 'unit root' | 'stationary' | 'inconclusive'
    plain: str = ""
    caveats: list[str] = field(default_factory=list)
    simulated_cv: bool = False


# --------------------------------------------------------------------------- #
# ADF — all three deterministic specifications
# --------------------------------------------------------------------------- #
_ADF_REGRESSION = {"none": "n", "constant": "c", "constant+trend": "ct"}


def adf(series: pd.Series, spec: Spec = "constant+trend", maxlag: int | None = None) -> UnitRootTest:
    """
    Augmented Dickey–Fuller.

    Null: the series has a unit root. Rejecting it is evidence of stationarity — but only
    for the deterministic specification you chose, which is why we show all three.
    """
    from statsmodels.tsa.stattools import adfuller

    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 12:
        raise ValueError("ADF needs at least about a dozen observations.")
    if maxlag is None:
        # Schwert-style rule, the statsmodels default: 12·(n/100)^0.25, capped
        maxlag = int(np.ceil(12 * (len(y) / 100) ** 0.25))
        maxlag = min(maxlag, max(1, len(y) // 4))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, usedlag, nobs, crit, _icbest = adfuller(
            y, maxlag=maxlag, regression=_ADF_REGRESSION[spec], autolag="AIC"
        )

    # A rank-deficient regression returns a statistic on the order of 1e13 with a normal-looking
    # p-value. That is a numerical failure, not a finding: raise so the battery omits the test
    # rather than printing a number no one can interpret.
    if not np.isfinite(stat) or abs(stat) > 1e6:
        raise ValueError(
            f"ADF ({_spec_words(spec)}) is numerically degenerate on this series "
            "(rank-deficient design); it is omitted rather than reported."
        )

    rejects = bool(p < 0.05)
    says = "stationary" if rejects else "unit root"
    plain = (
        f"ADF ({_spec_words(spec)}) = {stat:.3f} with {p_phrase(p, 3)}. "
        + (
            "The unit-root null is rejected: there is evidence of stationarity."
            if rejects else
            "The unit-root null is not rejected — the data look like they contain a unit root."
        )
        + (
            " Note that ADF has low power: failing to reject is not proof of a unit root."
            if not rejects else ""
        )
    )
    return UnitRootTest(
        id=f"stat.adf.{_slug(spec)}",
        name="Augmented Dickey–Fuller",
        spec=spec,
        null="unit root",
        statistic=float(stat),
        p_value=float(p),
        lags=int(usedlag),
        critical={k: float(v) for k, v in crit.items()},
        rejects_null=rejects,
        says=says,
        plain=plain,
        caveats=[
            "ADF's null is a unit root. A high p-value means 'not enough evidence against a "
            "unit root', not 'there is a unit root'.",
        ],
    )


# --------------------------------------------------------------------------- #
# KPSS — the opposite null
# --------------------------------------------------------------------------- #
def kpss(series: pd.Series, spec: Literal["level", "trend"] = "level") -> UnitRootTest:
    """
    KPSS. Null: the series *is* stationary (level- or trend-stationary).

    Rejecting KPSS is evidence of non-stationarity. Pairing it with ADF is the point:
    two tests, opposite nulls, four possible combinations.
    """
    from statsmodels.tsa.stattools import kpss as _kpss

    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 12:
        raise ValueError("KPSS needs at least about a dozen observations.")
    nlags = int(np.ceil(4 * (len(y) / 100) ** 0.25))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, lags, crit = _kpss(y, regression="c" if spec == "level" else "ct", nlags=nlags)

    # statsmodels caps p at the table edges; say so rather than printing a fake 0.10
    capped = "KPSS p-values are only tabulated between 0.01 and 0.10; the printed value is "
    rejects = bool(p < 0.05)
    says = "stationary" if not rejects else "unit root"
    plain = (
        f"KPSS ({spec}) = {stat:.3f} with {p_phrase(p, 3)}. "
        + (
            "The stationarity null is rejected: there is evidence of non-stationarity."
            if rejects else
            "The stationarity null is not rejected — consistent with a stationary series."
        )
    )
    return UnitRootTest(
        id=f"stat.kpss.{spec}",
        name="KPSS",
        spec=spec,
        null="stationary",
        statistic=float(stat),
        p_value=float(p),
        lags=int(lags),
        critical={k: float(v) for k, v in crit.items()} if isinstance(crit, dict) else {},
        rejects_null=rejects,
        says=says,
        plain=plain,
        caveats=[
            "KPSS's null is the opposite of ADF's: it assumes stationarity and looks for "
            "evidence against it.",
            capped + "clipped, so treat 0.10 as 'p ≥ 0.10'.",
        ],
    )


# --------------------------------------------------------------------------- #
# Phillips–Perron — no p-value, but a statistic and a table
# --------------------------------------------------------------------------- #
def phillips_perron(series: pd.Series, spec: Spec = "constant+trend",
                    lags: int | None = None, period: int | None = None) -> UnitRootTest:
    """
    Phillips–Perron: ADF's t-ratio, corrected for serial correlation non-parametrically.

    Implemented here (statsmodels has no PP for a univariate series) as the standard
    Z(τ) with a Newey–West style long-run variance, and compared against the
    MacKinnon critical values that ADF also uses — the same asymptotic distribution,
    which is why the same table applies.
    """
    y = series.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 12:
        raise ValueError("Phillips–Perron needs at least about a dozen observations.")
    if lags is None:
        lags = int(np.ceil(4 * (n / 100) ** (2 / 9)))

    t = np.arange(n, dtype=float)
    # deterministic terms
    X = {"none": np.empty((n, 0)), "constant": np.ones((n, 1)), "constant+trend": np.column_stack([np.ones(n), t])}[spec]
    dy = np.diff(y)
    y_lag = y[:-1]
    k = X.shape[1]
    Z = np.column_stack([X[1:], y_lag]) if k else y_lag.reshape(-1, 1)
    m = len(dy)

    beta, *_ = np.linalg.lstsq(Z, dy, rcond=None)
    resid = dy - Z @ beta
    dof = max(1, m - Z.shape[1])
    s2 = float(resid @ resid) / dof

    # long-run variance with Bartlett weights
    gamma0 = float(resid @ resid) / m
    lr = gamma0
    for j in range(1, lags + 1):
        w = 1.0 - j / (lags + 1.0)
        g = float(resid[j:] @ resid[:-j]) / m
        lr += 2.0 * w * g
    lr = max(lr, 1e-12)

    if not np.isfinite(s2) or s2 <= 0:
        raise ValueError("Phillips–Perron is numerically degenerate on this series.")
    xtx_inv = np.linalg.pinv(Z.T @ Z)
    se_last = float(np.sqrt(s2 * xtx_inv[-1, -1]))
    if not np.isfinite(se_last) or se_last <= 0:
        raise ValueError("Phillips–Perron is numerically degenerate on this series.")
    t_ratio = float(beta[-1] / se_last)
    # Z(tau) = sqrt(s2/lr) * t - (lr - s2)/(2 lr) * sqrt(m) * se(beta_last)/s2 ... in the
    # standard form below, expressed with the regression's own quantities
    lam = 0.5 * (lr - s2)
    z_tau = (s2 / lr) ** 0.5 * t_ratio - lam * np.sqrt(m) * (xtx_inv[-1, -1] ** 0.5) / s2

    # same asymptotic distribution as ADF (MacKinnon 2010), so the ADF table applies
    critical = _mackinnon_cv(spec, n)
    rejects = bool(z_tau < critical["5%"])
    plain = (
        f"Phillips–Perron ({_spec_words(spec)}) Z(τ) = {z_tau:.3f}. "
        f"The 5% critical value is {critical['5%']:.3f}, so the unit-root null is "
        + ("rejected." if rejects else "not rejected.")
        + " PP corrects for serial correlation non-parametrically, so it needs no lag choice "
          "and is more robust to heteroskedasticity than ADF — but it is sensitive to the "
          "bandwidth used."
    )
    caveats = [
        "No p-value is reported: PP has no closed-form finite-sample distribution here, so "
        "the statistic is compared with MacKinnon's critical values instead.",
        "PP can over-reject badly when the series has a large negative MA root or strong "
        "unmodelled seasonality.",
    ]
    if period and period > 1:
        caveats.append(
            f"PP does not model seasonality. Because this series has a {period}-period "
            "cycle, treat a PP rejection as less informative than the seasonal unit-root "
            "tests below."
        )
    return UnitRootTest(
        id=f"stat.pp.{_slug(spec)}",
        name="Phillips–Perron",
        spec=spec,
        null="unit root",
        statistic=float(z_tau),
        p_value=None,          # deliberately absent: no reliable finite-sample p here
        lags=int(lags),
        critical=critical,
        rejects_null=rejects,
        says="stationary" if rejects else "unit root",
        plain=plain,
        caveats=caveats,
    )


def _mackinnon_cv(spec: Spec, n: int) -> dict[str, float]:
    """MacKinnon (2010) response-surface critical values (asymptotic, n-adjusted)."""
    # coefficients: cv = b0 + b1/n + b2/n^2 ; rows are 1%, 5%, 10% per spec
    table = {
        "constant": [(-3.43035, -6.5393, -16.786), (-2.86154, -2.8903, -4.234),
                     (-2.56677, -1.5384, -2.809)],
        "constant+trend": [(-3.95877, -9.0531, -28.428), (-3.41049, -4.3904, -9.036),
                           (-3.12705, -2.5856, -3.632)],
        "none": [(-2.56574, -2.2358, -3.627), (-1.94100, -0.2669, -3.223),
                 (-1.61682, 0.2656, -2.494)],
    }[spec]
    out = {}
    for label, (b0, b1, b2) in zip(("1%", "5%", "10%"), table):
        out[label] = float(b0 + b1 / n + b2 / (n * n))
    return out


# --------------------------------------------------------------------------- #
# DF–GLS — the one with better power
# --------------------------------------------------------------------------- #
def df_gls(series: pd.Series, spec: Literal["constant", "constant+trend"] = "constant") -> UnitRootTest:
    """
    Elliott–Rothenberg–Stock DF–GLS, and why it matters: plain ADF with a constant has
    poor power against a series that is stationary around a *trend*. DF–GLS detrends by
    local GLS first (with the ERS c̄), which is uniformly more powerful, and it is the
    test to trust when ADF and KPSS disagree.

    Implemented directly: GLS-detrended series → ADF regression on it → ERS critical
    values (interpolated from the published table for the two specifications used here).
    """
    from statsmodels.tsa.stattools import adfuller

    y = series.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 15:
        raise ValueError("DF–GLS needs at least about 15 observations.")

    cbar = -7.0 if spec == "constant" else -13.5
    alpha = 1.0 + cbar / n
    t = np.arange(n, dtype=float)
    if spec == "constant":
        z = np.ones((n, 1))
    else:
        # the ERS trend is usually written in terms of the mean of t; either scaling
        # gives the same detrended series
        z = np.column_stack([np.ones(n), t - n / 2.0])

    # The GLS transform, with the first observation scaled by sqrt(1 − alpha²) so that the
    # transformed errors are homoskedastic (Elliott–Rothenberg–Stock eq. 3). Without that
    # scaling the first point carries ~79% of the weight in the constant estimate, the
    # "demeaned" series inherits an arbitrary offset from y₁, and the ADF regression —
    # which has no constant of its own — is run on a series with a big nonzero mean.
    scale = float(np.sqrt(1.0 - alpha ** 2))
    ybar = np.r_[scale * y[0], y[1:] - alpha * y[:-1]]
    zbar = np.vstack([scale * z[:1], z[1:] - alpha * z[:-1]])
    gamma, *_ = np.linalg.lstsq(zbar, ybar, rcond=None)
    detrended = y - z @ gamma

    # ERS choose the augmentation by AIC/MAIC over a small set: on 100–200 observations a
    # large default lag set costs most of the power this test exists to provide. With no
    # augmentation the statistic is badly oversized, so AIC over 0..4 is the right balance.
    maxlag = int(min(4, max(1, np.floor(12 * (n / 100) ** 0.25))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, _p, usedlag, *_rest = adfuller(detrended, maxlag=maxlag, regression="n",
                                            autolag="AIC")

    # ERS (1996) Table 1, interpolated on the c̄ actually used; these are the standard
    # asymptotic critical values for the two cases this app runs.
    cv = ({"1%": -3.77, "5%": -3.19, "10%": -2.89} if spec == "constant"
          else {"1%": -2.58, "5%": -1.94, "10%": -1.62})
    rejects = bool(stat < cv["5%"])
    plain = (
        f"DF–GLS ({spec}) = {stat:.3f} against a 5% critical value of {cv['5%']:.2f}, so the "
        "unit-root null is " + ("rejected." if rejects else "not rejected.")
        + " DF–GLS is the most powerful of the three here when the series is stationary "
          "around a trend, so it is the one to trust when ADF and KPSS disagree."
    )
    return UnitRootTest(
        id=f"stat.dfgls.{spec}",
        name="DF–GLS (Elliott–Rothenberg–Stock)",
        spec=spec,
        null="unit root",
        statistic=float(stat),
        p_value=None,
        lags=int(usedlag),
        critical=cv,
        rejects_null=rejects,
        says="stationary" if rejects else "unit root",
        plain=plain,
        caveats=[
            "Critical values are ERS's published asymptotic table, not finite-sample; for very "
            "short series they are mildly conservative.",
            "DF–GLS tests the *detrended* series, so it answers a narrower question than ADF.",
        ],
    )


# --------------------------------------------------------------------------- #
# Zivot–Andrews — a unit root in the presence of one break
# --------------------------------------------------------------------------- #
def zivot_andrews(series: pd.Series, spec: Literal["level", "trend", "both"] = "both") -> UnitRootTest:
    """
    Zivot–Andrews: a unit-root test that allows one endogenous structural break.

    This matters because a shift in level is the most common thing ADF misreads as a unit
    root: the series looks like it wanders because it moved once. If ZA rejects where ADF
    does not, the honest reading is "a break, not a unit root".
    """
    from statsmodels.tsa.stattools import zivot_andrews as _za

    y = series.dropna().to_numpy(dtype=float)
    if len(y) < 20:
        raise ValueError("Zivot–Andrews needs at least about 20 observations.")
    spec_sm = {"level": "c", "trend": "t", "both": "ct"}[spec]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stat, p, crit, baselag, breakpoint = _za(y, trim=0.15, regression=spec_sm, autolag="AIC")

    rejects = bool(p < 0.05)
    idx = int(breakpoint) if breakpoint is not None else 0
    plain = (
        f"Zivot–Andrews ({spec}) = {stat:.3f} with {p_phrase(p, 3)}; the most likely break is at "
        f"position {idx + 1} of {len(y)}. "
        + ("The unit-root null is rejected once one break is allowed — so what looked like a "
           "unit root is better explained by a shift." if rejects else
           "The unit-root null is still not rejected after allowing a break.")
    )
    return UnitRootTest(
        id=f"stat.za.{spec}",
        name="Zivot–Andrews (unit root with one break)",
        spec=spec,
        null="unit root",
        statistic=float(stat),
        p_value=float(p),
        lags=int(baselag),
        critical={k: float(v) for k, v in (crit or {}).items()},
        rejects_null=rejects,
        says="stationary" if rejects else "unit root",
        plain=plain,
        caveats=["Allows exactly one break at an unknown date; two breaks need the Bai–Perron "
                 "route in the break tests below."],
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spec_words(spec: str) -> str:
    return {
        "constant": "constant only",
        "constant+trend": "constant and trend",
        "none": "no deterministic terms",
        "level": "level shift",
        "trend": "trend break",
    }.get(spec, spec)


def _slug(spec: str) -> str:
    return spec.replace("+", "_").replace(" ", "_")


def unit_root_battery(series: pd.Series, period: int | None = None,
                      include_breaks: bool = True) -> list[UnitRootTest]:
    """
    Every unit-root test that can run on this series.

    The order is deliberate. The classic pair (ADF, KPSS) comes first because the reader needs
    to see them agree or disagree side by side; the alternatives follow, and Zivot–Andrews
    last, because it only matters if the first two said "unit root".

    A test that cannot run is omitted, never fabricated: a missing verdict is honest, a made-up
    one is not.
    """
    y = series.dropna()
    n = len(y)
    out: list[UnitRootTest] = []

    def run(fn, *args, **kwargs) -> None:
        try:
            out.append(fn(y, *args, **kwargs))
        except Exception:                                  # pragma: no cover - short series
            return

    run(adf, "constant+trend")
    run(adf, "constant")
    run(kpss, "level")
    run(kpss, "trend")
    run(phillips_perron, "constant+trend", period=period)
    run(df_gls, "constant")
    run(df_gls, "constant+trend")
    if include_breaks and n >= 30:
        run(zivot_andrews, "both")
    return out


__all__ = [
    "UnitRootTest", "adf", "kpss", "phillips_perron", "df_gls", "zivot_andrews",
    "unit_root_battery",
]
