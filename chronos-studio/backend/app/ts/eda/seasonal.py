"""
ts/eda/seasonal.py — seasonality, calendar structure and the trend hint (§5.2).

Two measures here are used all the way downstream, so they are implemented to the
definitions in Hyndman & Athanasopoulos rather than invented:

    F_S = max(0, 1 − Var(remainder) / Var(remainder + seasonal))
    F_T = max(0, 1 − Var(remainder) / Var(remainder + trend))

They are variance *shares*: how much of the variation in (component + remainder)
the component explains. 0.64 is the book's rule-of-thumb boundary for "strong".
Both are reported with the periodogram evidence, because a variance share alone
cannot tell you *which* period is seasonal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from app.core.formatting import p_phrase
from app.ts.eda.stl_cache import robust_stl_components

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
QUARTER_LABELS = ["Q1", "Q2", "Q3", "Q4"]
DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def labels_for(freq_code: str, period: int | None) -> list[str]:
    """Human labels for one seasonal cycle, so no chart ever shows '0…11'."""
    if not period or period <= 1:
        return []
    if freq_code in {"MS", "ME", "M"} and period == 12:
        return MONTH_LABELS
    if freq_code in {"QS", "QE", "Q"} and period == 4:
        return QUARTER_LABELS
    if freq_code in {"D", "B", "W", "W-SUN"} and period == 7:
        return DAY_LABELS
    if freq_code in {"h", "H"} and period == 24:
        return [f"{h:02d}h" for h in range(24)]
    return [f"P{i + 1}" for i in range(period)]


@dataclass
class SeasonalProfile:
    available: bool
    period: int
    labels: list[str]
    indices: list[float]            # seasonal index per position (mean 1.0 for multiplicative view)
    means: list[float]              # mean value per position
    counts: list[int]
    amplitude: float                # peak − trough of the position means
    amplitude_pct_of_mean: float
    peak_label: str
    trough_label: str
    peak_value: float
    trough_value: float
    kw_statistic: float | None      # Kruskal–Wallis across positions
    kw_p_value: float | None
    f_statistic: float | None       # one-way ANOVA (reported alongside, with caveat)
    f_p_value: float | None
    strength: float | None          # F_S from STL
    amplitude_growing: bool | None
    amplitude_early: float | None
    amplitude_late: float | None
    plain: str


def seasonal_profile(
    values: pd.Series,
    period: int | None,
    freq_code: str = "",
) -> SeasonalProfile:
    """
    Profile one seasonal cycle: typical values by position, the swing, and
    whether the swing is stable, growing or shrinking.
    """
    s = values.dropna().astype(float)
    labels = labels_for(freq_code, period)
    if not period or period <= 1 or len(s) < period + 2 or s.nunique() < 3:
        return SeasonalProfile(
            available=False, period=period or 1, labels=labels, indices=[], means=[],
            counts=[], amplitude=float("nan"), amplitude_pct_of_mean=float("nan"),
            peak_label="", trough_label="", peak_value=float("nan"),
            trough_value=float("nan"), kw_statistic=None, kw_p_value=None,
            f_statistic=None, f_p_value=None, strength=None, amplitude_growing=None,
            amplitude_early=None, amplitude_late=None,
            plain="There is no seasonal cycle to profile at this frequency.",
        )

    # Position within the cycle, counted from the first observation so that a
    # series starting mid-year still aligns correctly.
    pos = (np.arange(len(s)) % period)
    # Test positional differences on the DETRENDED series. A rising series has huge
    # within-position spread, which swamps the between-position signal and makes a
    # perfectly seasonal series look flat (a real trap: AirPassengers gives p = 0.43
    # if you test the raw values). The seasonal profile itself is still reported in
    # original units, because that is what the user reads.
    detrended = _detrend_for_testing(s)

    # TWO frames on purpose: the reported profile must be in original units (that is
    # what a reader applies by hand), while the significance test must run on the
    # detrended values (otherwise a rising series hides its own seasonality).
    raw_frame = pd.DataFrame({"value": s.to_numpy(), "pos": pos})
    test_frame = pd.DataFrame({"value": detrended, "pos": pos})

    grouped = raw_frame.groupby("pos")["value"]
    means = grouped.mean().reindex(range(period))
    counts = grouped.size().reindex(range(period)).fillna(0).astype(int)

    overall = float(s.mean())
    if overall == 0:
        overall = float(np.nanmean(np.abs(s.to_numpy()))) or 1.0
    indices = (means / overall).tolist() if overall != 0 else [float("nan")] * period
    amplitude = float(np.nanmax(means.to_numpy()) - np.nanmin(means.to_numpy()))
    peak_i = int(np.nanargmax(means.to_numpy()))
    trough_i = int(np.nanargmin(means.to_numpy()))

    # Kruskal–Wallis: non-parametric, so it does not assume normality of the raw data.
    groups = [g["value"].to_numpy() for _, g in test_frame.groupby("pos") if len(g) >= 2]
    kw_stat = kw_p = None
    if len(groups) >= 3:
        try:
            res = stats.kruskal(*groups)
            kw_stat, kw_p = float(res.statistic), float(res.pvalue)
        except ValueError:  # pragma: no cover - all-identical values
            pass

    f_stat = f_p = None
    if len(groups) >= 3:
        try:
            res_f = stats.f_oneway(*groups)
            f_stat, f_p = float(res_f.statistic), float(res_f.pvalue)
        except ValueError:  # pragma: no cover
            pass

    strength = stl_strength(s, period)
    growing, early, late = amplitude_trend(s, period, labels)
    if kw_p is None or kw_p >= 0.05:
        # If the positions do not differ beyond chance, "the swing is growing" is
        # meaningless — it is noise being described as structure.
        growing = None

    plain = _profile_plain(
        labels=labels, means=means.to_numpy(), peak_i=peak_i, trough_i=trough_i,
        amplitude=amplitude, overall=overall, kw_p=kw_p, strength=strength,
        growing=growing, early=early, late=late,
    )

    return SeasonalProfile(
        available=True, period=period, labels=labels,
        indices=[float(x) for x in indices],
        means=[float(x) for x in means.to_numpy()],
        counts=[int(x) for x in counts.to_numpy()],
        amplitude=amplitude,
        amplitude_pct_of_mean=float(amplitude / overall * 100) if overall else float("nan"),
        peak_label=labels[peak_i] if peak_i < len(labels) else f"P{peak_i + 1}",
        trough_label=labels[trough_i] if trough_i < len(labels) else f"P{trough_i + 1}",
        peak_value=float(means.to_numpy()[peak_i]),
        trough_value=float(means.to_numpy()[trough_i]),
        kw_statistic=kw_stat, kw_p_value=kw_p,
        f_statistic=f_stat, f_p_value=f_p,
        strength=strength,
        amplitude_growing=growing,
        amplitude_early=early, amplitude_late=late,
        plain=plain,
    )


def _profile_plain(
    *, labels: list[str], means: np.ndarray, peak_i: int, trough_i: int,
    amplitude: float, overall: float, kw_p: float | None, strength: float | None,
    growing: bool | None, early: float | None, late: float | None,
) -> str:
    bits = []
    if kw_p is not None:
        bits.append(
            f"A Kruskal–Wallis test across the {len(labels)} seasonal positions gives "
            f"{p_phrase(kw_p, 4)}"
            + (" — the positions differ more than chance would explain." if kw_p < 0.05
               else " — no convincing difference between positions.")
        )
    if strength is not None:
        band = "strong" if strength >= 0.64 else ("moderate" if strength >= 0.3 else "weak")
        bits.append(f"Seasonal strength F_S = {strength:.2f} ({band}).")
    bits.append(
        f"The typical peak is {labels[peak_i] if peak_i < len(labels) else peak_i + 1} "
        f"({means[peak_i]:,.4g}) and the trough is "
        f"{labels[trough_i] if trough_i < len(labels) else trough_i + 1} "
        f"({means[trough_i]:,.4g}), a swing of {amplitude:,.4g}"
        + (f" ({amplitude / overall * 100:.1f}% of the average level)." if overall else ".")
    )
    if growing is True and early and late:
        bits.append(
            f"The swing is growing: {early:,.4g} early in the series versus {late:,.4g} late, "
            "which points to multiplicative seasonality."
        )
    elif growing is False and early and late:
        bits.append(f"The swing is shrinking: {early:,.4g} early versus {late:,.4g} late.")
    else:
        bits.append("The size of the swing is stable over time, which points to additive seasonality.")
    return " ".join(bits)


def _detrend_for_testing(values: pd.Series) -> np.ndarray:
    """
    Remove a linear (and, if long enough, seasonal) trend before comparing seasonal
    positions, so that drift does not mask the very pattern we are testing for.
    """
    x = np.arange(len(values), dtype=float)
    y = values.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return y - (intercept + slope * x)


def stl_strength(values: pd.Series, period: int) -> float | None:
    """
    F_S from an STL decomposition (Hyndman & Athanasopoulos §3.4).

    Returns None when there is not enough data for a reliable STL fit.
    """
    s = values.dropna().astype(float)
    if period <= 1 or len(s) < 2 * period + 2:
        return None
    try:
        # Part 19: share the identical weekly-period fit with the outlier pass.
        fit = robust_stl_components(s, period, seasonal=7)
    except Exception:  # pragma: no cover
        return None
    remainder = fit["resid"].to_numpy(dtype=float)
    seasonal = fit["seasonal"].to_numpy(dtype=float)
    var_r = float(np.var(remainder, ddof=1))
    var_rs = float(np.var(remainder + seasonal, ddof=1))
    if var_rs <= 0:
        return None
    return float(max(0.0, 1.0 - var_r / var_rs))


def amplitude_trend(
    values: pd.Series, period: int, labels: list[str]
) -> tuple[bool | None, float | None, float | None]:
    """
    Is the seasonal swing stable, growing or shrinking?

    Compares peak−trough within each *year*, so it answers the question the
    additive-vs-multiplicative choice actually depends on.
    """
    s = values.dropna().astype(float)
    if len(s) < 3 * period:
        return None, None, None
    frame = pd.DataFrame({"value": s.to_numpy(), "pos": np.arange(len(s)) % period})
    years = pd.Series(pd.to_datetime(s.index)).dt.year.to_numpy()
    frame["year"] = years
    amps: list[float] = []
    for _year, chunk in frame.groupby("year"):
        if chunk["pos"].nunique() >= max(2, int(0.75 * period)):
            amps.append(float(chunk["value"].max() - chunk["value"].min()))
    if len(amps) < 4:
        return None, None, None
    k = max(2, len(amps) // 3)
    early = float(np.mean(amps[:k]))
    late = float(np.mean(amps[-k:]))
    if early <= 0:
        return None, early, late
    ratio = late / early
    if ratio > 1.15:
        return True, early, late
    if ratio < 0.87:
        return False, early, late
    return None, early, late


# --------------------------------------------------------------------------- #
# Calendar cross-tabs (month × year heatmap)
# --------------------------------------------------------------------------- #
def month_year_matrix(values: pd.Series, period: int, freq_code: str) -> pd.DataFrame:
    """Rows = years, columns = seasonal positions. The month×year heatmap."""
    s = values.dropna().astype(float)
    labels = labels_for(freq_code, period)
    idx = pd.DatetimeIndex(s.index)
    frame = pd.DataFrame({"value": s.to_numpy(), "year": idx.year})
    if period == 12:
        frame["pos"] = idx.month
    elif period == 4:
        frame["pos"] = idx.quarter
    elif period == 7:
        frame["pos"] = idx.dayofweek + 1
    else:
        frame["pos"] = np.arange(len(s)) % period + 1
    pivot = frame.pivot_table(index="year", columns="pos", values="value", aggfunc="mean")
    pivot = pivot.reindex(columns=range(1, period + 1))
    pivot.columns = labels[: len(pivot.columns)]
    return pivot


# --------------------------------------------------------------------------- #
# Trend hint (a descriptor, NOT a stationarity test — that is S5)
# --------------------------------------------------------------------------- #
def trend_hint(values: pd.Series) -> dict[str, Any]:
    """
    How much of the movement is a persistent drift?

    Deliberately described as a *hint*: proper inference (Mann–Kendall, Sen's slope,
    unit-root tests) lives in Parts 4 and 5. Here it exists so the EDA narrative can
    say "this series is rising" without pretending to have tested anything.
    """
    s = values.dropna().astype(float)
    n = len(s)
    if n < 6:
        return {"available": False}
    x = np.arange(n, dtype=float)
    slope, intercept = np.polyfit(x, s.to_numpy(), 1)
    fitted = intercept + slope * x
    resid = s.to_numpy() - fitted
    var_fit = float(np.var(fitted, ddof=1))
    var_res = float(np.var(resid, ddof=1))
    r2 = float(1 - var_res / (var_fit + var_res)) if (var_fit + var_res) > 0 else 0.0
    diffs = np.diff(s.to_numpy())
    monotonic_share = float(np.mean(np.sign(diffs) == np.sign(slope))) if slope != 0 else 0.5
    total = float(slope * (n - 1))
    mean_level = float(np.mean(s.to_numpy()))
    return {
        "available": True,
        "slope_per_period": float(slope),
        "total_change": total,
        "pct_over_window": float(total / mean_level * 100) if mean_level else float("nan"),
        "r_squared": r2,
        "monotonic_share": monotonic_share,
        "plain": (
            f"A straight line through the series explains {r2:.0%} of its variation, and the "
            f"level changes by {total:,.4g} over the window."
            + (" That is a strong, persistent drift." if r2 > 0.7 else
               " The drift is weak or intermittent." if r2 < 0.3 else "")
        ),
    }


def calendar_heatmap_data(values: pd.Series, period: int, freq_code: str) -> pd.DataFrame:
    """Alias kept explicit for the chart layer's benefit."""
    return month_year_matrix(values, period, freq_code)


__all__ = [
    "SeasonalProfile", "seasonal_profile", "stl_strength", "amplitude_trend",
    "month_year_matrix", "calendar_heatmap_data", "trend_hint",
    "labels_for", "MONTH_LABELS", "QUARTER_LABELS", "DAY_LABELS",
]
