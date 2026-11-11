"""
ts/decompose/methods.py — S4: taking a series apart (§5.4).

Every method here answers the same question — "how much of this is trend, how much seasonal,
how much leftover?" — and each answers it differently, which is why the app runs several and
shows them side by side rather than picking one and presenting it as the truth.

The decisions worth stating:

* **STL is the default.** Classical decomposition uses a centred moving average, which flattens
  the ends of the trend (the first and last half-cycle are simply lost) and is not robust to
  outliers. STL keeps every observation, handles outliers with `robust=True`, and lets the
  seasonal pattern change slowly. Classical is still offered, because it is what the textbook
  shows and a user comparing them learns something.

* **Multiplicative decomposition is blocked on non-positive data**, not silently switched to
  additive: the mode is a statement about the series ("the seasonal swing grows with the level"),
  and quietly changing it would make the picture lie (R0.6, §5.4 guardrail).

* **A decomposition is a description, not a model.** The remainder of a good decomposition is
  SUPPOSED to look like noise. If the remainder still has autocorrelation or ARCH structure,
  the app says so — that structure is what the model stage has to capture, and pretending the
  remainder is white noise is how a poor model looks acceptable.

* **Cycles are not seasons.** A periodogram peak that is not at a calendar lag is a cycle: it
  has no fixed date. The app distinguishes them in words, because "period 41.3 months" is
  meaningless as a seasonal term and very meaningful as a business cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

Model = Literal["additive", "multiplicative"]
Method = Literal["stl", "classical", "mstl", "ucm", "hp", "cf"]


@dataclass
class Component:
    """One component of a decomposition, plus the numbers that describe it."""

    name: str                                    # trend | seasonal | remainder | cycle
    values: pd.Series
    share_of_variance: float                     # of the total of the three parts
    plain: str = ""


@dataclass
class Decomposition:
    """The result of one method, with everything a scorecard or a plot needs."""

    method: Method
    model: Model
    label: str
    observed: pd.Series
    trend: pd.Series
    seasonal: pd.Series
    remainder: pd.Series
    period: int | None = None
    period2: int | None = None
    fitted: pd.Series | None = None
    #: Share of variance in each of trend/seasonal/remainder. They are computed on the same
    #: series and sum to 1 by construction, which is why the scorecard can be a percentage bar.
    shares: dict[str, float] = field(default_factory=dict)
    #: Hyndman's strength measures (0–1): how much of the variation the component explains.
    trend_strength: float | None = None
    seasonal_strength: float | None = None
    caveats: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def component(self, name: str) -> pd.Series:
        return {"trend": self.trend, "seasonal": self.seasonal,
                "remainder": self.remainder}.get(name, self.trend)

    def as_table_rows(self) -> list[dict[str, Any]]:
        out = []
        for name in ("trend", "seasonal", "remainder"):
            s = self.component(name)
            out.append({
                "component": name,
                "share": round(self.shares.get(name, float("nan")), 4),
                "sd": round(float(np.nanstd(s.to_numpy(dtype=float))), 6),
                "min": round(float(np.nanmin(s.to_numpy(dtype=float))), 6),
                "max": round(float(np.nanmax(s.to_numpy(dtype=float))), 6),
            })
        return out


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #
def check_model(model: Model, series: pd.Series) -> None:
    """
    Multiplicative decomposition needs strictly positive data — the §5.4 guardrail.

    The check is a refusal, not a switch to additive. A user who asked for multiplicative
    because "the swings grow with the level" has said something about the series; converting
    silently would answer a different question than the one asked.
    """
    if model != "multiplicative":
        return
    y = series.dropna()
    n_nonpos = int((y <= 0).sum())
    if n_nonpos:
        raise ValueError(
            f"Multiplicative decomposition needs every value to be positive, but {n_nonpos} "
            f"of {len(y)} values are zero or negative (the smallest is {float(y.min()):.6g}). "
            "Use additive decomposition, or shift/transform the series first — for example a "
            "log transform, which turns a multiplicative pattern into an additive one."
        )


def _interp(values: pd.Series) -> pd.Series:
    """STL and friends cannot accept gaps; interpolate a copy and say nothing (the caller notes it)."""
    return values.interpolate(limit_direction="both").astype(float)


def _shares(trend: pd.Series, seasonal: pd.Series, remainder: pd.Series,
            model: Model = "additive") -> dict[str, float]:
    """
    Variance shares of the three parts.

    Computed on the components themselves, so they sum to exactly 1 and the scorecard bar is
    honest. For a multiplicative decomposition the parts are already ratios (seasonal ≈ 1), so
    they are centred on their own means first — otherwise "seasonal share" would be measuring
    the level, not the swing.
    """
    t = np.asarray(trend, dtype=float)
    s = np.asarray(seasonal, dtype=float)
    r = np.asarray(remainder, dtype=float)
    mask = np.isfinite(t) & np.isfinite(s) & np.isfinite(r)
    t, s, r = t[mask], s[mask], r[mask]
    if t.size == 0:
        return {"trend": float("nan"), "seasonal": float("nan"), "remainder": float("nan")}
    if model == "multiplicative":
        # The parts are levels around 1. Comparing them raw would score the trend as "the level"
        # and everything else as ~0 (the trend's variance swamps the ratios). They are put on the
        # same additive footing as the additive case by flattening the level: t/(mean t) − 1,
        # s − 1, r − 1 — each then measures deviation, and the shares are comparable across models.
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(t != 0, t / np.mean(t) - 1.0, 0.0)
        s = s - 1.0
        r = r - 1.0
    # centre each part: a constant offset belongs to none of them
    t = t - t.mean()
    s = s - s.mean()
    r = r - r.mean()
    total = float(np.var(t) + np.var(s) + np.var(r))
    if total <= 0:
        return {"trend": 0.0, "seasonal": 0.0, "remainder": 0.0}
    return {
        "trend": float(np.var(t) / total),
        "seasonal": float(np.var(s) / total),
        "remainder": float(np.var(r) / total),
    }


def _strengths(trend: pd.Series, seasonal: pd.Series, remainder: pd.Series) -> tuple[float, float]:
    """Hyndman's F_T and F_S: 1 − Var(remainder)/Var(remainder + component)."""
    r = np.asarray(remainder, dtype=float)
    t = np.asarray(trend, dtype=float)
    s = np.asarray(seasonal, dtype=float)
    mask = np.isfinite(r) & np.isfinite(t) & np.isfinite(s)
    if mask.sum() < 4:
        return float("nan"), float("nan")
    var_r = float(np.var(r[mask], ddof=1))
    var_tr = float(np.var((t + r)[mask], ddof=1))
    var_sr = float(np.var((s + r)[mask], ddof=1))
    f_t = max(0.0, 1.0 - var_r / var_tr) if var_tr > 0 else float("nan")
    f_s = max(0.0, 1.0 - var_r / var_sr) if var_sr > 0 else float("nan")
    return float(f_t), float(f_s)


# --------------------------------------------------------------------------- #
# Methods
# --------------------------------------------------------------------------- #
def stl(series: pd.Series, period: int, model: Model = "additive", *,
        robust: bool = True, seasonal_window: int | None = None,
        trend_window: int | None = None) -> Decomposition:
    """
    STL (Cleveland et al. 1990) — loess smoothing of the trend and of each seasonal subseries.

    `seasonal_window` must be odd and at least 7; it controls how fast the seasonal pattern is
    allowed to change. Larger = more stable pattern. The default (7) lets the pattern drift,
    which is usually what a real business series does.
    """
    from statsmodels.tsa.seasonal import STL

    check_model(model, series)
    y = series.dropna()
    if period < 2:
        raise ValueError("STL needs a seasonal period of at least 2.")
    if len(y) < 2 * period + 2:
        raise ValueError(
            f"STL needs at least two full cycles ({2 * period + 2} observations for a period "
            f"of {period}); this series has {len(y)}."
        )
    notes: list[str] = []
    if y.isna().any():                                        # pragma: no cover - dropna above
        notes.append("Missing observations were interpolated for the fit.")
    if model == "multiplicative":
        # STL is additive. A multiplicative decomposition is the additive one applied to logs,
        # then exponentiated — which is exact, and worth saying rather than hiding.
        fit = STL(np.log(_interp(y)), period=period, robust=robust,
                  seasonal=seasonal_window or 7, trend=trend_window).fit()
        trend = np.exp(fit.trend)
        seasonal = np.exp(fit.seasonal)
        remainder = np.exp(fit.resid)
        notes.append(
            "Multiplicative STL is fitted to the log of the series and exponentiated, which is "
            "exactly a multiplicative decomposition."
        )
    else:
        sw = seasonal_window or 7
        if sw % 2 == 0 or sw < 7:
            sw = max(7, sw + 1 if sw % 2 == 0 else sw)
        fit = STL(_interp(y), period=period, robust=robust, seasonal=sw, trend=trend_window).fit()
        trend = pd.Series(np.asarray(fit.trend), index=y.index)
        seasonal = pd.Series(np.asarray(fit.seasonal), index=y.index)
        remainder = pd.Series(np.asarray(fit.resid), index=y.index)

    trend.index = y.index
    seasonal.index = y.index
    remainder.index = y.index
    f_t, f_s = _strengths(trend, seasonal, remainder)
    return Decomposition(
        method="stl", model=model, label=f"STL ({model})",
        observed=y, trend=trend, seasonal=seasonal, remainder=remainder,
        period=period, fitted=trend * seasonal if model == "multiplicative" else trend + seasonal,
        shares=_shares(trend, seasonal, remainder, model),
        trend_strength=f_t, seasonal_strength=f_s,
        caveats=[
            "STL is a smoother, not a model: it has no likelihood, no standard errors and no "
            "forecast. Use it to *see* the components, then model them.",
        ] + ([
            "Robust fitting down-weights outliers, so a large spike will look smaller here than "
            "in the data. That is deliberate — but it means the remainder is not the full story.",
        ] if robust else []) + notes,
        notes=notes,
    )


def classical(series: pd.Series, period: int, model: Model = "additive") -> Decomposition:
    """
    The textbook decomposition: a centred moving average for the trend, then averages of the
    detrended values for the seasonal index.

    Its cost is visible and stated: the moving average cannot reach the ends, so the first and
    last `period/2` trend values are missing. That is why it is not the default.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    check_model(model, series)
    y = series.dropna()
    if period < 2:
        raise ValueError("Classical decomposition needs a seasonal period of at least 2.")
    if len(y) < 2 * period:
        raise ValueError(
            f"Classical decomposition needs two full cycles ({2 * period} observations for a "
            f"period of {period}); this series has {len(y)}."
        )
    fit = seasonal_decompose(_interp(y), model=model, period=period, extrapolate_trend=0)
    trend = pd.Series(np.asarray(fit.trend), index=y.index)
    seasonal = pd.Series(np.asarray(fit.seasonal), index=y.index)
    remainder = pd.Series(np.asarray(fit.resid), index=y.index)
    missing = int(trend.isna().sum())
    f_t, f_s = _strengths(trend, seasonal, remainder)
    return Decomposition(
        method="classical", model=model, label=f"Classical ({model})",
        observed=y, trend=trend, seasonal=seasonal, remainder=remainder,
        period=period, fitted=trend * seasonal if model == "multiplicative" else trend + seasonal,
        shares=_shares(trend, seasonal, remainder, model),
        trend_strength=f_t, seasonal_strength=f_s,
        caveats=[
            f"The centred moving average leaves the first and last {missing // 2} trend values "
            f"undefined ({missing} in total), so the ends of the trend are simply absent. STL "
            "does not have this problem.",
            "The moving average is not robust: one outlier drags the trend around it.",
        ],
    )


def mstl(series: pd.Series, periods: list[int], model: Model = "additive", *,
         robust: bool = True) -> Decomposition:
    """
    MSTL — STL repeated for several seasonal periods (e.g. weekly + annual on daily data).

    Returned as a single decomposition whose `seasonal` is the sum of all seasonal components,
    with the individual ones kept in `notes` as their variance shares, because "which period
    matters most?" is the question that follows immediately.
    """
    from statsmodels.tsa.seasonal import MSTL

    check_model(model, series)
    y = series.dropna()
    periods = sorted({int(p) for p in periods if p and p >= 2})
    if not periods:
        raise ValueError("MSTL needs at least one seasonal period of 2 or more.")
    if len(y) < 2 * max(periods) + 2:
        raise ValueError(
            f"MSTL needs two full cycles of the longest period ({2 * max(periods) + 2} "
            f"observations); this series has {len(y)}."
        )
    yy = np.log(_interp(y)) if model == "multiplicative" else _interp(y)
    fit = MSTL(yy, periods=periods, stl_kwargs={"robust": robust}).fit()
    seas_df: pd.DataFrame = fit.seasonal
    if isinstance(seas_df, pd.Series):
        seas_df = seas_df.to_frame()
    seasonal_total = seas_df.sum(axis=1)
    trend = pd.Series(np.asarray(fit.trend), index=y.index)
    seas = pd.Series(np.asarray(seasonal_total), index=y.index)
    rem = pd.Series(np.asarray(fit.resid), index=y.index)
    if model == "multiplicative":
        trend, seas, rem = np.exp(trend), np.exp(seas), np.exp(rem)
    f_t, f_s = _strengths(trend, seas, rem)
    per_period = {
        int(p): round(float(np.var(np.asarray(seas_df.iloc[:, i], dtype=float))), 6)
        for i, p in enumerate(periods)
    }
    top = max(per_period, key=lambda k: per_period[k]) if per_period else None
    return Decomposition(
        method="mstl", model=model, label=f"MSTL ({model}, periods {periods})",
        observed=y, trend=trend, seasonal=seas, remainder=rem,
        period=periods[0], period2=periods[1] if len(periods) > 1 else None,
        fitted=trend * seas if model == "multiplicative" else trend + seas,
        shares=_shares(trend, seas, rem, model),
        trend_strength=f_t, seasonal_strength=f_s,
        notes=[
            f"Seasonal variance by period: "
            + ", ".join(f"{p} → {v:.4g}" for p, v in per_period.items())
            + (f". Period {top} carries the most seasonal variation." if top else ""),
        ],
        caveats=[
            "MSTL's components are fitted sequentially, so a very strong longer cycle can absorb "
            "part of a shorter one.",
        ],
    )


def ucm(series: pd.Series, period: int | None = None, model: Model = "additive") -> Decomposition:
    """
    State-space unobserved components (local linear trend + optional stochastic seasonal).

    Different in kind from STL: this is a *model* with a likelihood, so its components come with
    uncertainty and can be forecast. It is also more opinionated — it assumes a specific form for
    the trend (level + slope, both allowed to drift), which is why its trend is smoother than
    STL's on the same data.
    """
    from statsmodels.tsa.statespace.structural import UnobservedComponents

    check_model(model, series)
    y = series.dropna()
    if len(y) < 20:
        raise ValueError("The state-space decomposition needs at least about 20 observations.")
    yy = np.log(_interp(y)) if model == "multiplicative" else _interp(y)
    slope_present = True
    kwargs: dict[str, Any] = {"level": True, "trend": True, "irregular": True}
    if period and period >= 2 and len(y) >= 2 * period:
        kwargs.update({"seasonal": period, "stochastic_seasonal": True})
    mod = UnobservedComponents(yy, **kwargs)
    res = mod.fit(disp=False, maxiter=200)
    # statsmodels exposes the state-space components differently from the STL API: the level and
    # slope are parts of `smoothed_state`, and the irregular is recovered as observed − fitted.
    state = np.asarray(res.smoothed_state, dtype=float)
    level = state[0]
    slope = state[1] if slope_present else np.zeros_like(level)
    trend = pd.Series(level + slope, index=y.index)
    if "seasonal" in kwargs:
        n_seas = period - 1
        seasonal = pd.Series(state[2:2 + n_seas].sum(axis=0), index=y.index)
    else:
        seasonal = pd.Series(0.0, index=y.index)
    irregular = yy - np.asarray(res.fittedvalues, dtype=float)
    remainder = pd.Series(np.asarray(irregular, dtype=float), index=y.index)
    if model == "multiplicative":
        trend, seasonal, remainder = np.exp(trend), np.exp(seasonal), np.exp(remainder)
    f_t, f_s = _strengths(trend, seasonal, remainder)
    return Decomposition(
        method="ucm", model=model, label=f"Unobserved components ({model})",
        observed=y, trend=trend, seasonal=seasonal, remainder=remainder,
        period=period if "seasonal" in kwargs else None,
        fitted=trend * seasonal if model == "multiplicative" else trend + seasonal,
        shares=_shares(trend, seasonal, remainder, model),
        trend_strength=f_t, seasonal_strength=f_s,
        notes=[f"Log-likelihood {res.llf:.2f}, {res.df_model} estimated parameters."],
        caveats=[
            "This is a fitted model: its components depend on assumptions about how the level, "
            "slope and seasonal terms are allowed to move, and those assumptions are choices.",
        ],
    )


def hp_filter(series: pd.Series, lamb: float | None = None) -> Decomposition:
    """
    Hodrick–Prescott: split into a smooth trend and the rest. No seasonal component at all.

    Offered for the trend picture, not for a seasonal analysis — and with its one real
    objection attached: the HP trend is an artefact of λ, it manufactures cycles in series that
    have none, and the ends are unreliable. It is included because half the literature uses it
    and a user will meet it.
    """
    from statsmodels.tsa.filters.hp_filter import hpfilter

    y = series.dropna()
    if len(y) < 10:
        raise ValueError("The HP filter needs at least about 10 observations.")
    lam = lamb if lamb is not None else (1600 if len(y) >= 100 else 100)
    cycle, trend_arr = hpfilter(_interp(y), lamb=lam)
    trend = pd.Series(np.asarray(trend_arr), index=y.index)
    cycle_s = pd.Series(np.asarray(cycle), index=y.index)
    remainder = pd.Series(0.0, index=y.index)
    lam = float(lam)
    return Decomposition(
        method="hp", model="additive", label=f"Hodrick–Prescott (λ = {lam:g})",
        observed=y, trend=trend, seasonal=cycle_s, remainder=remainder,
        fitted=trend,
        shares=_shares(trend, cycle_s, remainder, "additive"),
        trend_strength=None, seasonal_strength=None,
        notes=[
            f"λ = {lam:g} chosen by the usual rule for this sample size. The second panel is the "
            "cycle, not a seasonal component."
        ],
        caveats=[
            "HP has no seasonal component: any seasonality in this series is sitting in the "
            "'cycle' panel and will look like a repeating cycle.",
            "The HP trend is defined by λ, not by the data: a different λ gives a different "
            "trend, and the filter invents cycles in series with none (Hamilton 2018).",
        ],
    )


def cf_filter(series: pd.Series, low: int = 6, high: int = 32, drift: bool = True) -> Decomposition:
    """
    Christiano–Fitzgerald band-pass: keep the variation whose period lies between `low` and
    `high` observations, and call that the cycle.

    A band-pass filter is the honest way to look for a cycle, because it states the band it is
    looking in. The cost is at the ends: the filter is two-sided, so it loses `high` observations
    from each end.
    """
    from statsmodels.tsa.filters.cf_filter import cffilter

    y = series.dropna()
    if len(y) < 3 * high:
        raise ValueError(
            f"The band-pass filter needs about three times the longest period ({3 * high}) "
            f"observations; this series has {len(y)}."
        )
    cycle_arr, trend_arr = cffilter(_interp(y), low=low, high=high, drift=drift)
    cyc = cycle_arr
    cycle = pd.Series(np.asarray(cyc), index=y.index)
    total = _interp(y)
    trend = (total - cycle).rename("trend")
    remainder = pd.Series(0.0, index=y.index)
    return Decomposition(
        method="cf", model="additive", label=f"Band-pass ({low}–{high} observations)",
        observed=y, trend=trend, seasonal=cycle, remainder=remainder,
        fitted=trend,
        shares=_shares(trend, cycle, remainder, "additive"),
        notes=[f"Band: periods between {low} and {high} observations."],
        caveats=[
            f"Two-sided filter: the first and last ~{high} observations are unreliable and the "
            "estimate is only meaningful in the middle of the sample.",
        ],
    )


def decompose(series: pd.Series, method: Method = "stl", period: int | None = None,
              model: Model = "additive", **kwargs: Any) -> Decomposition:
    """One entry point, so the API and the tests agree about what 'decompose' means."""
    if method == "stl":
        if not period:
            raise ValueError("STL needs a seasonal period. Tell me the calendar of the series.")
        return stl(series, period, model, **kwargs)
    if method == "classical":
        if not period:
            raise ValueError("Classical decomposition needs a seasonal period.")
        return classical(series, period, model)
    if method == "mstl":
        periods = kwargs.pop("periods", None) or ([period] if period else [])
        return mstl(series, list(periods), model, **kwargs)
    if method == "ucm":
        return ucm(series, period, model)
    if method == "hp":
        return hp_filter(series, kwargs.get("lamb"))
    if method == "cf":
        return cf_filter(series, kwargs.get("low", 6), kwargs.get("high", 32))
    raise ValueError(f"Unknown decomposition method '{method}'.")


__all__ = [
    "Component", "Decomposition", "Model", "Method",
    "check_model", "stl", "classical", "mstl", "ucm", "hp_filter", "cf_filter", "decompose",
]
