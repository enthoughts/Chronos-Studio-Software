"""
ts/preprocess/ops.py — the preprocessing engine (§5.3, and the "suggest, don't impose"
rule).

S2 recommends; S3 applies. Every operation here:

  * is **pure** — it takes a series and params, returns a new series plus a record
    describing exactly what changed, so nothing is destructive and undo is free;
  * is **reversible or honest** — an operation that cannot be undone (dropping rows)
    says so in its record rather than pretending;
  * is **previewable** — the same function is called by the preview endpoint with
    `commit=False`, so what the user sees is literally what will be applied;
  * never silently changes the number of observations without recording it.

Smoothing of the modelling series is deliberately NOT here — it is never offered (see the
unconditional rule in `advisor/rules_prep.py`): a smoothed series has autocorrelation built into it
by construction, which is the opposite of what every test downstream is trying to detect.

Differencing lives here rather than in a separate module because S7's "what next" advisor proposes
it and Step 2 must be able to apply and undo it through the same versioned path. The *decision* to
difference belongs to S5/S7, where it is tested; this module only executes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class OpResult:
    """What an operation did, in terms a non-statistician can check."""

    series: pd.Series
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    plain: str = ""
    reversible: bool = True
    n_changed: int = 0
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #
def seasonal_interpolate(s: pd.Series, period: int | None = None, **_: Any) -> OpResult:
    """
    Fill short gaps from the same calendar position in neighbouring years, then from
    the series mean if that position has no history. Falls back to linear.
    """
    n_missing = int(s.isna().sum())
    if n_missing == 0:
        return OpResult(s.copy(), "seasonal_interpolation", {}, "Nothing was missing.", n_changed=0)
    if not period or period < 2:
        out = s.interpolate(method="time", limit_direction="both")
        return OpResult(out, "linear_interpolation", {"period": None},
                        f"Filled {n_missing} missing values by linear interpolation along the time index.",
                        n_changed=n_missing)

    filled = s.copy()
    seasonal_mean = s.groupby(np.arange(len(s)) % period).transform("mean")
    missing = s.isna()
    filled[missing] = seasonal_mean[missing]
    if filled.isna().any():
        filled = filled.interpolate(method="time", limit_direction="both")
    return OpResult(
        filled, "seasonal_interpolation", {"period": period},
        f"Filled {n_missing} missing value{'s' if n_missing != 1 else ''} with the average of "
        f"the same position in other cycles (period {period}).",
        n_changed=n_missing,
    )


def kalman_smooth(s: pd.Series, period: int | None = None, **_: Any) -> OpResult:
    """State-space (local linear trend + optional seasonal) smoother for mid-length gaps."""
    n_missing = int(s.isna().sum())
    if n_missing == 0:
        return OpResult(s.copy(), "kalman_smoother", {}, "Nothing was missing.", n_changed=0)
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        endog = s.astype(float)
        kwargs: dict[str, Any] = {"level": "local linear trend"}
        if period and period > 1 and len(s) >= 3 * period:
            kwargs["seasonal"] = period
        model = UnobservedComponents(endog, **kwargs)
        res = model.fit(disp=False)
        smoothed = pd.Series(res.fittedvalues, index=s.index)
        filled = s.copy()
        missing = s.isna()
        filled[missing] = smoothed[missing]
        return OpResult(
            filled, "kalman_smoother", {"period": period},
            f"Filled {n_missing} missing value{'s' if n_missing != 1 else ''} with a state-space "
            "smoother, which uses the neighbouring shape and the seasonal cycle together.",
            n_changed=n_missing,
        )
    except Exception as exc:                                     # pragma: no cover
        fallback = seasonal_interpolate(s, period)
        fallback.warnings.append(f"The state-space smoother failed ({type(exc).__name__}); seasonal interpolation was used instead.")
        return fallback


def leave_missing(s: pd.Series, period: int | None = None) -> OpResult:
    return OpResult(s.copy(), "leave_missing", {}, "Missing values were left as they are.", n_changed=0)


# --------------------------------------------------------------------------- #
# Transformation
# --------------------------------------------------------------------------- #
def _positive_or_warn(s: pd.Series, method: str) -> OpResult | None:
    if (s.dropna() <= 0).any():
        return OpResult(
            s.copy(), method, {}, f"{method} is undefined for zero or negative values; nothing was changed.",
            n_changed=0,
            warnings=["A log-type transform needs strictly positive data. Use Yeo–Johnson instead."],
        )
    return None


def log_transform(s: pd.Series, base: str = "e", offset: float = 0.0, **_: Any) -> OpResult:
    shifted = s + offset
    bad = _positive_or_warn(shifted, "log")
    if bad is not None:
        return bad
    out = np.log(shifted) if base == "e" else np.log10(shifted)
    return OpResult(
        out, "log", {"base": base, "offset": offset},
        "Logged the series" + (f" after adding {offset} to keep every value positive" if offset else "")
        + ". Differences are now proportional changes rather than absolute ones.",
        n_changed=int(s.notna().sum()),
    )


def difference(s: pd.Series, lag: int = 1, **_: Any) -> OpResult:
    """
    Replace the series with its change over `lag` periods.

    Two honesty requirements, both visible in the returned text:

    * the operation **removes `lag` observations** (they are undefined, not zero), and says how
      many — a difference that silently dropped rows would invalidate every count downstream;
    * the direction of each resulting value is the *change*, so "the series" after this call is a
      different quantity. Anything that quotes a level afterwards would be quoting the wrong number.

    Undo restores the previous version from the session, which is why this is reversible despite
    the lost observations.
    """
    lag = int(lag)
    if lag < 1:
        raise ValueError("A difference needs a lag of at least 1.")
    n_before = int(s.notna().sum())
    out = s.diff(lag)
    n_after = int(out.notna().sum())
    if n_after < 10:
        return OpResult(
            s.copy(), "difference", {"lag": lag},
            f"A difference of lag {lag} would leave only {n_after} observations, which is too few "
            "to analyse; nothing was changed.",
            n_changed=0,
            warnings=["The series is too short for this difference."],
        )
    return OpResult(
        out, "difference", {"lag": lag},
        f"Replaced the series with its change over {lag} period{'s' if lag != 1 else ''}. Each "
        f"value is now the difference between consecutive "
        f"{'observations' if lag == 1 else f'values {lag} apart'}, and the first {lag} "
        f"observation{'s' if lag != 1 else ''} "
        f"{'are' if lag != 1 else 'is'} undefined, so the series is {n_after} long rather than "
        f"{n_before}.",
        reversible=True,
        n_changed=n_after,
        warnings=[
            "Every value is now a change, not a level. Forecasts made from this series are "
            "forecasts of changes and must be integrated back to the original scale.",
        ],
    )


def sqrt_transform(s: pd.Series, **_: Any) -> OpResult:
    if (s.dropna() < 0).any():
        return OpResult(s.copy(), "sqrt", {}, "A square root needs non-negative data; nothing changed.",
                        warnings=["Negative values present."])
    return OpResult(np.sqrt(s), "sqrt", {}, "Took the square root of every value.", n_changed=int(s.notna().sum()))


def boxcox_transform(s: pd.Series, lam: float = 0.0, **_: Any) -> OpResult:
    from scipy import stats

    bad = _positive_or_warn(s, "boxcox")
    if bad is not None:
        return bad
    out = pd.Series(stats.boxcox(s.dropna().to_numpy(dtype=float), lam), index=s.index[s.notna()])
    return OpResult(
        out.reindex(s.index), "boxcox", {"lambda": lam},
        f"Applied a Box–Cox transform with λ = {lam:g}.", n_changed=int(s.notna().sum()),
    )


def yeojohnson_transform(s: pd.Series, lam: float | None = None, **_: Any) -> OpResult:
    from scipy import stats

    v = s.dropna().to_numpy(dtype=float)
    if v.size < 3:
        return OpResult(s.copy(), "yeojohnson", {}, "Too few observations for a Yeo–Johnson transform.")
    lam = float(stats.yeojohnson_normmax(v)) if lam is None else float(lam)
    out = pd.Series(stats.yeojohnson(v, lam), index=s.index[s.notna()]).reindex(s.index)
    return OpResult(
        out, "yeojohnson", {"lambda": lam},
        f"Applied a Yeo–Johnson transform with λ = {lam:.3g}, which behaves like a log for positive "
        "values and still works for zeros and negatives.",
        n_changed=int(s.notna().sum()),
    )


def none_transform(s: pd.Series, **_: Any) -> OpResult:
    return OpResult(s.copy(), "none", {}, "The series was left on its original scale.", n_changed=0)


# --------------------------------------------------------------------------- #
# Outlier handling — flag (default), dummy, or delete (with a warning)
# --------------------------------------------------------------------------- #
def flag_outliers(s: pd.Series, times: list[str] | None = None, **_: Any) -> OpResult:
    return OpResult(
        s.copy(), "flag_only", {"times": times or []},
        "The unusual points are recorded as flags so later steps can add them to a model, and "
        "left in the data unchanged.",
        n_changed=0,
        reversible=True,
    )


def delete_outliers(s: pd.Series, times: list[str] | None = None, **_: Any) -> OpResult:
    times = times or []
    idx = pd.DatetimeIndex(pd.to_datetime(times)) if times else pd.DatetimeIndex([])
    keep = ~s.index.isin(idx)
    out = s[keep]
    return OpResult(
        out, "delete", {"times": times},
        f"Removed {len(times)} observation{'s' if len(times) != 1 else ''} from the series. This "
        "changes the number of observations and cannot be undone by re-running the model — it can "
        "only be undone here.",
        reversible=True,          # undoable in this app, but not recoverable downstream
        n_changed=len(times),
        warnings=[
            "Deleting an observation is the least defensible treatment: it removes information the "
            "model could have used, and biases the fit towards the pattern you already believe. "
            "Adding an intervention dummy is preferred.",
        ],
    )


# --------------------------------------------------------------------------- #
# Presentation-only smoothing (never applied to the modelling series)
# --------------------------------------------------------------------------- #
def moving_average(s: pd.Series, window: int = 7, **_: Any) -> OpResult:
    return OpResult(
        s.rolling(window, center=True, min_periods=max(2, window // 2)).mean(),
        "moving_average", {"window": window},
        f"A centred {window}-point moving average, for display only.",
        n_changed=0,
        warnings=["Presentation only — this must not replace the series the model is fitted to."],
    )


def trim_window(s: pd.Series, start: str | None = None, end: str | None = None, **_: Any) -> OpResult:
    out = s.loc[start:end] if (start or end) else s.copy()
    return OpResult(
        out, "trim", {"start": start, "end": end},
        f"Restricted the analysis window to {out.index.min() if len(out) else '—'} … "
        f"{out.index.max() if len(out) else '—'} ({len(out)} observations).",
        n_changed=len(s) - len(out),
        warnings=["Everything downstream is re-run on the shorter window."],
    )


def aggregate(s: pd.Series, freq: str = "QS", how: str = "sum", **_: Any) -> OpResult:
    agg = s.resample(freq).sum() if how == "sum" else s.resample(freq).mean()
    return OpResult(
        agg, "aggregate", {"freq": freq, "how": how},
        f"Aggregated to {freq} by {how} ({len(agg)} observations). Aggregation introduces a moving-average "
        "autocorrelation structure, so AR terms may now be needed where none were before.",
        n_changed=len(agg),
    )


#: The registry. `op` is what the advisor rule id refers to.
REGISTRY: dict[str, Callable[..., OpResult]] = {
    "impute.seasonal": seasonal_interpolate,
    "impute.kalman": kalman_smooth,
    "impute.none": leave_missing,
    "transform.log": log_transform,
    "transform.sqrt": sqrt_transform,
    "transform.boxcox": boxcox_transform,
    "transform.yeojohnson": yeojohnson_transform,
    "transform.none": none_transform,
    "outlier.flag": flag_outliers,
    "outlier.delete": delete_outliers,
    "difference": difference,
    "smooth.ma": moving_average,
    "trim": trim_window,
    "aggregate": aggregate,
}

#: Which step each op belongs to (matches `PreprocessingRecord.step`).
STEP_OF: dict[str, str] = {
    "impute.seasonal": "impute", "impute.kalman": "impute", "impute.none": "impute",
    "transform.log": "transform", "transform.sqrt": "transform",
    "transform.boxcox": "transform", "transform.yeojohnson": "transform",
    "transform.none": "transform",
    "outlier.flag": "outlier", "outlier.delete": "outlier",
    "difference": "difference",
    "smooth.ma": "smooth", "trim": "trim", "aggregate": "aggregate",
}


def apply_op(op: str, series: pd.Series, params: dict[str, Any] | None = None) -> OpResult:
    fn = REGISTRY.get(op)
    if fn is None:
        raise KeyError(f"unknown preprocessing operation: {op}")
    return fn(series, **(params or {}))


__all__ = [
    "OpResult", "REGISTRY", "STEP_OF", "apply_op",
    "seasonal_interpolate", "kalman_smooth", "leave_missing",
    "log_transform", "sqrt_transform", "boxcox_transform", "yeojohnson_transform",
    "none_transform", "flag_outliers", "delete_outliers", "moving_average",
    "trim_window", "aggregate",
]
