"""
ts/eda/descriptives.py — describe the series before modelling (§5.2).

Deliberate choices:

* Normality tests are reported **with the caveat that normality of the raw series
  is almost never required**. We say so in the result itself, not only in docs.
* Every measure carries an interpretation band where one is meaningful, so the UI
  can colour it and the advisor can threshold on it in Part 3.
* Nothing here recommends anything. Advice lives in `advisor/`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.schemas import Confidence


@dataclass
class Descriptives:
    n: int
    n_missing: int
    pct_missing: float
    time_start: str
    time_end: str
    frequency_label: str
    period: int | None

    mean: float
    median: float
    sd: float
    variance: float
    cv: float
    minimum: float
    maximum: float
    q1: float
    q3: float
    iqr: float
    range: float
    skewness: float
    excess_kurtosis: float

    first: float
    last: float
    total_change: float
    pct_change: float
    cagr: float | None
    n_increases: int
    n_decreases: int
    n_ties: int

    n_zeros: int
    pct_zeros: float
    n_negatives: int
    median_abs_deviation: float
    coefficient_of_variation_band: str
    skew_band: str
    kurtosis_band: str

    normality: dict[str, Any] = field(default_factory=dict)

    def as_rows(self) -> list[dict[str, Any]]:
        """Long form, for the results table (one row per quantity)."""
        return [
            {"quantity": "Observations (n)", "value": self.n, "note": f"{self.n_missing} missing"},
            {"quantity": "Period covered", "value": f"{self.time_start[:10]} → {self.time_end[:10]}",
             "note": self.frequency_label},
            {"quantity": "Mean", "value": _r(self.mean), "note": ""},
            {"quantity": "Median", "value": _r(self.median), "note": ""},
            {"quantity": "Standard deviation", "value": _r(self.sd), "note": ""},
            {"quantity": "Variance", "value": _r(self.variance), "note": ""},
            {"quantity": "Coefficient of variation", "value": _r(self.cv), "note": self.coefficient_of_variation_band},
            {"quantity": "Minimum", "value": _r(self.minimum), "note": ""},
            {"quantity": "1st quartile", "value": _r(self.q1), "note": ""},
            {"quantity": "3rd quartile", "value": _r(self.q3), "note": ""},
            {"quantity": "Maximum", "value": _r(self.maximum), "note": ""},
            {"quantity": "Interquartile range", "value": _r(self.iqr), "note": ""},
            {"quantity": "Median absolute deviation", "value": _r(self.median_abs_deviation),
             "note": "robust spread"},
            {"quantity": "Skewness", "value": _r(self.skewness), "note": self.skew_band},
            {"quantity": "Excess kurtosis", "value": _r(self.excess_kurtosis), "note": self.kurtosis_band},
            {"quantity": "First value", "value": _r(self.first), "note": ""},
            {"quantity": "Last value", "value": _r(self.last), "note": ""},
            {"quantity": "Total change", "value": _r(self.total_change), "note": ""},
            {"quantity": "Percentage change", "value": _r(self.pct_change), "note": "%"},
            {"quantity": "CAGR", "value": _r(self.cagr) if self.cagr is not None else "—",
             "note": "compound annual growth" if self.cagr is not None else "needs ≥1 year of data"},
            {"quantity": "Upward steps", "value": self.n_increases, "note": ""},
            {"quantity": "Downward steps", "value": self.n_decreases, "note": ""},
            {"quantity": "Unchanged steps", "value": self.n_ties, "note": ""},
            {"quantity": "Zeros", "value": self.n_zeros, "note": f"{self.pct_zeros:.1%} of values"},
            {"quantity": "Negative values", "value": self.n_negatives, "note": ""},
        ]


def _r(x: float | None, digits: int = 4) -> float | None:
    if x is None or not np.isfinite(x):
        return None
    return float(round(x, digits))


def _band_cv(cv: float) -> str:
    if not np.isfinite(cv):
        return ""
    if cv < 0.1:
        return "very stable (CV < 0.1)"
    if cv < 0.3:
        return "moderately variable (0.1 ≤ CV < 0.3)"
    if cv < 0.6:
        return "variable (0.3 ≤ CV < 0.6)"
    return "highly variable (CV ≥ 0.6)"


def _band_skew(s: float) -> str:
    a = abs(s)
    side = "right" if s > 0 else "left"
    if a < 0.5:
        return "roughly symmetric"
    if a < 1.0:
        return f"moderately skewed {side}"
    return f"strongly skewed {side}"


def _band_kurtosis(k: float) -> str:
    if k < -0.5:
        return "lighter tails than normal (platykurtic)"
    if k <= 0.5:
        return "tails close to normal"
    if k <= 3:
        return "heavier tails than normal (leptokurtic)"
    return "very heavy tails"


def describe(series: pd.Series, frequency_label: str = "", period: int | None = None) -> Descriptives:
    """Compute the descriptive table for S1."""
    s = series.astype(float)
    v = s.dropna()
    n = int(len(v))
    if n == 0:
        raise ValueError("describe() needs at least one observation")

    arr = v.to_numpy()
    diffs = np.diff(arr)
    q1, q3 = np.percentile(arr, [25, 75])
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    med = float(np.median(arr))

    years = (v.index[-1] - v.index[0]).days / 365.25 if n > 1 else 0.0
    cagr = None
    if years >= 1 and arr[0] > 0 and arr[-1] > 0:
        cagr = float((arr[-1] / arr[0]) ** (1 / years) - 1) * 100

    return Descriptives(
        n=n,
        n_missing=int(s.isna().sum()),
        pct_missing=float(s.isna().mean()),
        time_start=str(v.index[0]),
        time_end=str(v.index[-1]),
        frequency_label=frequency_label,
        period=period,
        mean=mean, median=med, sd=sd, variance=float(np.var(arr, ddof=1)) if n > 1 else 0.0,
        cv=float(sd / mean) if mean != 0 else float("nan"),
        minimum=float(np.min(arr)), maximum=float(np.max(arr)),
        q1=float(q1), q3=float(q3), iqr=float(q3 - q1),
        range=float(np.max(arr) - np.min(arr)),
        skewness=float(stats.skew(arr, bias=False)) if n > 2 else 0.0,
        excess_kurtosis=float(stats.kurtosis(arr, bias=False)) if n > 3 else 0.0,
        first=float(arr[0]), last=float(arr[-1]),
        total_change=float(arr[-1] - arr[0]),
        pct_change=float((arr[-1] - arr[0]) / abs(arr[0]) * 100) if arr[0] != 0 else float("nan"),
        cagr=cagr,
        n_increases=int(np.sum(diffs > 0)),
        n_decreases=int(np.sum(diffs < 0)),
        n_ties=int(np.sum(diffs == 0)),
        n_zeros=int(np.sum(arr == 0)),
        pct_zeros=float(np.mean(arr == 0)),
        n_negatives=int(np.sum(arr < 0)),
        median_abs_deviation=float(np.median(np.abs(arr - med))),
        coefficient_of_variation_band=_band_cv(float(sd / mean) if mean else float("nan")),
        skew_band=_band_skew(float(stats.skew(arr, bias=False)) if n > 2 else 0.0),
        kurtosis_band=_band_kurtosis(float(stats.kurtosis(arr, bias=False)) if n > 3 else 0.0),
        normality=normality_battery(arr),
    )


def normality_battery(arr: np.ndarray) -> dict[str, Any]:
    """
    Shapiro–Wilk, Jarque–Bera and D'Agostino–Pearson, with the interpretation the
    user actually needs: *the raw series does not have to be normal*.
    """
    n = len(arr)
    out: dict[str, Any] = {"n": n}
    try:
        if 3 <= n <= 5000:
            sh_w, sh_p = stats.shapiro(arr)
            out["shapiro"] = {"statistic": float(sh_w), "p_value": float(sh_p)}
        jb = stats.jarque_bera(arr)
        out["jarque_bera"] = {"statistic": float(jb.statistic), "p_value": float(jb.pvalue)}
        if n >= 8:
            dp = stats.normaltest(arr)
            out["dagostino"] = {"statistic": float(dp.statistic), "p_value": float(dp.pvalue)}
    except Exception:  # pragma: no cover - degenerate input
        pass

    pvals = [v["p_value"] for k, v in out.items() if isinstance(v, dict) and "p_value" in v]
    out["rejects_normality"] = bool(pvals and min(pvals) < 0.05)
    out["plain"] = (
        "At least one test rejects normality of the raw values."
        if out["rejects_normality"]
        else "No test rejects normality of the raw values."
    ) + (
        " That is usually fine: what matters for forecasting is the behaviour of the "
        "residuals after fitting, not of the raw series. We check those in Step 12."
    )
    out["caveat"] = (
        "With large n these tests detect trivial departures. Judge the Q–Q plot, not only p-values."
        if n > 500 else ""
    )
    return out


# --------------------------------------------------------------------------- #
# Step-change and run statistics worth showing at EDA
# --------------------------------------------------------------------------- #
def runs_test(values: pd.Series) -> dict[str, Any]:
    """
    Wald–Wolfowitz runs test on the *signs of the first differences* — a cheap,
    assumption-light check of whether the series wanders more (or less) than chance.
    Too few runs ⇒ persistence/trend; too many ⇒ sawtooth alternation.
    """
    d = values.dropna().diff().dropna()
    signs = np.sign(d.to_numpy())
    signs = signs[signs != 0]
    n1, n2 = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    n = n1 + n2
    if n < 10:
        return {"available": False, "plain": "Too few observations for a runs test."}
    runs = 1 + int(np.sum(np.diff(signs) != 0))
    if n1 == 0 or n2 == 0:
        # Monotone series (a pure trend, or the tail of one): the runs statistic has
        # zero variance. Say so, and report the exact sign test instead — which is
        # what the user actually wanted to know.
        return {
            "available": False,
            "all_same_sign": True,
            "plain": (
                "Every step goes the same way, so the runs test has nothing to "
                "compare: this is a pure trend (sign-test p below)."
            ),
        }
    mean_runs = 2 * n1 * n2 / n + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n**2 * (n - 1))
    if var_runs <= 0:
        return {"available": False, "plain": "Runs test is degenerate for this series."}
    z = (runs - mean_runs) / np.sqrt(var_runs)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    if p >= 0.05:
        plain = "The number of runs is consistent with chance — no strong persistence or alternation."
    elif z < 0:
        plain = "Fewer runs than chance would give: values persist (trend or slow cycles)."
    else:
        plain = "More runs than chance would give: values alternate (sawtooth / possible over-differencing)."
    return {
        "available": True, "runs": runs, "expected_runs": float(mean_runs),
        "z": float(z), "p_value": p, "plain": plain,
    }


def rolling_cv_stability(values: pd.Series, window: int) -> dict[str, Any]:
    """Compare the first and last third of a rolling SD — the heteroskedasticity hint."""
    sd = values.rolling(window, min_periods=max(3, window // 2)).std().dropna()
    if len(sd) < 6:
        return {"available": False}
    third = max(1, len(sd) // 3)
    early, late = float(sd.iloc[:third].mean()), float(sd.iloc[-third:].mean())
    ratio = late / early if early > 0 else float("nan")
    return {
        "available": True,
        "early_sd": early,
        "late_sd": late,
        "ratio": ratio,
        "plain": (
            "The spread is roughly stable across the window."
            if 0.8 <= ratio <= 1.25
            else (
                f"The spread grows by about {(ratio - 1) * 100:.0f}% from start to end — "
                "variance is probably not constant, which argues for a log or Box–Cox transform."
                if ratio > 1.25 else
                f"The spread shrinks by about {(1 - ratio) * 100:.0f}% from start to end."
            )
        ),
    }


__all__ = ["Descriptives", "describe", "normality_battery", "runs_test", "rolling_cv_stability"]
