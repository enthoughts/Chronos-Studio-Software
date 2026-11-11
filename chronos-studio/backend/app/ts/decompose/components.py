"""
ts/decompose/components.py — quantifying what the decomposition found (§5.4).

A component plot is not evidence. Each one gets numbers, and each number gets a plain sentence:

  trend      direction, size in units and percent, Mann–Kendall τ, Sen's slope with CI,
             Cox–Stuart, and the share of variance it carries
  seasonal   which period(s), F_S, peak-to-trough amplitude (units and % of the mean), the
             index table, when the peak and trough fall, Kruskal–Wallis across seasonal
             groups, and whether the pattern is getting stronger or weaker
  cycle      periodogram peaks, Fisher's g-test, and the plain distinction from seasonality
  remainder  variance share, Ljung–Box, normality, ARCH-LM, and — the important one — whether
             the remainder is *actually* noise

Two rules this module follows, both of them from §5.4:

* **A "significant" cycle that is not on a calendar lag is not seasonality.** The periodogram
  will always produce a tallest peak; Fisher's g says whether that peak is more than the noise
  of a spectrum would produce, and the plain text distinguishes "every December" from "roughly
  every 3 years, whenever it happens".
* **Structure in the remainder is a finding, not a failure.** ARCH effects and autocorrelation
  in the residual are exactly what the modelling stage has to handle, so they are reported as
  structure to be explained, never smoothed away or dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from math import comb

from scipy import stats
from app.core.formatting import p_phrase

from app.ts.decompose.methods import Decomposition

#: Part 19 performance pass (§5.15: "10k-row series end-to-end interactive in <3 s").
#: The exact Mann–Kendall and Theil–Sen statistics are O(n²). On 10,000 points
#: Theil–Sen used to build a Python list of ~50,000,000 pairwise slopes (~400 MB
#: and minutes), which made S4 unusable. Both statistics stay *exact* for ordinary
#: series; above these sizes we use bounded, deterministic equivalents documented
#: in METHODS.md. Nothing here is random, so results remain reproducible.
MK_EXACT_LIMIT = 2000          # exact vectorised all-pairs path below this n
SEN_EXACT_LIMIT = 1500         # exact all-pairs Theil–Sen below this n
SEN_THIN_POINTS = 1000         # deterministic shape-preserving grid above the limit


def _mann_kendall_s_fenwick(x: np.ndarray) -> float:
    """
    Exact Mann–Kendall S in O(n log n), via a Fenwick tree over compressed ranks.

    For each observation we need the number of *earlier* values that are smaller
    minus the number that are larger (ties contribute sign 0). A binary-indexed
    tree answers both counts in log n, so S is identical to the naive double loop
    without materialising n² comparisons. See METHODS.md §Performance.
    """
    _, inv = np.unique(x, return_inverse=True)
    m = int(inv.max()) + 1
    bit = [0] * (m + 1)

    def add(k: int, v: int) -> None:
        k += 1
        while k <= m:
            bit[k] += v
            k += k & -k

    def prefix(k: int) -> int:
        # number of prior observations with compressed rank <= k; k may be -1
        k += 1
        total = 0
        while k > 0:
            total += bit[k]
            k -= k & -k
        return total

    s = 0.0
    for t in range(len(inv)):
        c = int(inv[t])
        less = prefix(c - 1)
        le = prefix(c)
        greater = t - le
        s += less - greater
        add(c, 1)
    return s


#: Hyndman's rule of thumb for calling a component "strong".
STRONG = 0.64
#: Below this, the component explains so little that it is not worth a term in a model.
WEAK = 0.30


@dataclass
class ComponentFinding:
    """One quantified component: the numbers, the sentence, and the caveats."""

    id: str
    name: str
    statistic: float | None
    p_value: float | None
    band: str                        # none | weak | moderate | strong
    plain: str
    detail: str = ""
    table: list[dict[str, Any]] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "statistic": self.statistic,
            "p_value": self.p_value, "band": self.band, "plain": self.plain,
            "detail": self.detail, "caveats": self.caveats,
        }


def _band(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "none"
    if x >= STRONG:
        return "strong"
    if x >= WEAK:
        return "moderate"
    if x > 0.05:
        return "weak"
    return "none"


def _num(x: float | None, digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "not computed"
    return f"{x:.{digits}f}"


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
def mann_kendall(values: pd.Series) -> tuple[float, float]:
    """
    Mann–Kendall trend test on the *trend component* (not the raw series).

    Kendall's τ over all pairs, with the normal approximation and a continuity correction. Used
    on the extracted trend because that is where "is it going up?" belongs: on the raw series a
    strong season can dominate the pair comparisons and hide the trend.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 4:
        return float("nan"), float("nan")
    if n > MK_EXACT_LIMIT:
        # Same exact S, O(n log n) instead of O(n²): keeps 10k rows interactive.
        s = _mann_kendall_s_fenwick(x)
    else:
        s = 0.0
        for i in range(n - 1):
            s += float(np.sum(np.sign(x[i + 1:] - x[i])))
    # variance with the tie correction
    _, counts = np.unique(x, return_counts=True)
    tie = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    var_s = (n * (n - 1) * (2 * n + 5) - tie) / 18.0
    if var_s <= 0:
        return float("nan"), float("nan")
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    tau = s / (0.5 * n * (n - 1))
    return float(tau), p


def sens_slope(values: pd.Series) -> tuple[float, float, float]:
    """
    Sen's slope: the median of all pairwise slopes, with a distribution-free 95% CI.

    Preferred over an OLS slope on a trend component because it is not dragged by the handful of
    large values that every real series has.
    """
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(y)
    y = y[mask]
    n = len(y)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    if n > SEN_EXACT_LIMIT:
        # Deterministic, shape-preserving thinning: keep the endpoints plus an
        # evenly spaced interior grid. Slopes still divide by the *original*
        # index distance, so units stay "per time step" and the endpoints anchor
        # the estimate. Bounded at ~500k pairs instead of n²; see METHODS.md.
        idx = np.unique(np.linspace(0, n - 1, SEN_THIN_POINTS).astype(int))
    else:
        idx = np.arange(n)
    x = idx.astype(float)
    ys = y[idx]
    i, j = np.triu_indices(len(idx), 1)
    diffs_arr = (ys[j] - ys[i]) / (x[j] - x[i])
    slope = float(np.median(diffs_arr))
    var_s = float(np.var(diffs_arr, ddof=1)) if len(diffs_arr) > 1 else 0.0
    # Kendall's approximation to the CI half-width. When every pairwise slope is identical (a
    # perfectly linear series) the spread is zero and the interval collapses onto the estimate —
    # returning NaN there would hide a perfectly good answer.
    half = float(1.96 * np.sqrt(var_s / max(1, len(diffs_arr))))
    return slope, slope - half, slope + half


def cox_stuart(values: pd.Series) -> tuple[float, float]:
    """
    Cox–Stuart: does the second half of the series sit persistently above the first?

    A sign test on paired differences, which is why it is insensitive to a single jump but
    excellent at catching a slow drift.
    """
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 6:
        return float("nan"), float("nan")
    half = n // 2
    d = y[half:half + half] - y[:half]
    pos = int(np.sum(d > 0))
    neg = int(np.sum(d < 0))
    m = pos + neg
    if m == 0:
        # No pair differs: there is no drift *or* drift in both directions. That is a clean
        # "no evidence", so return z = 0 and p = 1 rather than a NaN the caller has to special-case.
        return 0.0, 1.0
    # normal approximation to the sign test
    z = (pos - m / 2) / np.sqrt(m / 4) if m else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return float(z), p


def deseasonalised(dec: Decomposition) -> pd.Series:
    """
    The series with the seasonal component removed — what a trend test should be run on.

    Testing the *smoothed trend component* would be a mistake worth naming: Mann–Kendall assumes
    independent observations, and a loess-smoothed line is massively autocorrelated by
    construction, so its p-value is meaningless (it reported p < 0.0001 for a trend in white
    noise). Removing the seasonal component from the observed series leaves the noise in, which
    is what the test needs.
    """
    y = dec.observed.astype(float)
    s = dec.seasonal.reindex(y.index).astype(float)
    if dec.model == "multiplicative":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = y / s
        return out.replace([np.inf, -np.inf], np.nan).dropna()
    return (y - s).dropna()


def trend_finding(dec: Decomposition) -> ComponentFinding:
    """Everything worth saying about the trend component, in one object."""
    t = dec.trend.dropna()
    if len(t) < 8:
        return ComponentFinding(
            id="stat.trend", name="Trend", statistic=None, p_value=None, band="none",
            plain="There are not enough observations to describe a trend.",
        )
    # Tests run on the deseasonalised series (see `deseasonalised`); the *movement* reported is
    # the trend component's own, which is the thing the user is looking at.
    adjusted = deseasonalised(dec)
    tau, tau_p = mann_kendall(adjusted)
    slope, lo, hi = sens_slope(adjusted)
    z_cs, p_cs = cox_stuart(adjusted)
    first, last = float(t.iloc[0]), float(t.iloc[-1])
    delta = last - first
    pct = 100.0 * delta / abs(first) if first != 0 else float("nan")
    # Direction comes from the actual movement of the component, not from a separate slope, so
    # the words can never contradict the numbers in the same sentence.
    if delta > 0:
        direction = "upward"
    elif delta < 0:
        direction = "downward"
    else:
        direction = "flat"
    # F_T alone can call a trend "strong" on a series where Mann–Kendall finds no drift at all:
    # F_T divides by the remainder, so a tiny remainder makes any wiggle look strong. The band is
    # therefore capped by the test — a trend the test cannot distinguish from nothing is never
    # labelled better than "weak", whatever F_T says.
    band = _band(dec.trend_strength)
    drift_sig = bool(np.isfinite(tau_p) and tau_p < 0.05)
    if not drift_sig and band in {"strong", "moderate"}:
        band = "weak"

    sig = "significant" if drift_sig else "not significant"
    plain = (
        f"The trend is {direction}: it moves from {first:.4g} to {last:.4g} over the window "
        f"({delta:+.4g}"
        + (f", {pct:+.1f}% of where it started" if np.isfinite(pct) else "")
        + f"). On the seasonally adjusted series, Mann–Kendall τ = {_num(tau)} with "
        f"{p_phrase(tau_p)} — {sig} at 5% — and Sen's slope {_num(slope, 4)} per period (95% CI "
        f"{_num(lo, 4)} to {_num(hi, 4)}). Trend strength F_T = {_num(dec.trend_strength)} ({band})."
    )
    caveats = [
        "Mann–Kendall and Sen's slope are computed on the seasonally adjusted series, because "
        "both assume independent observations — running them on a smoothed trend line would give "
        "a spuriously tiny p-value.",
    ]
    if not drift_sig:
        caveats.append(
            "The trend is not statistically distinguishable from no trend; a model with a "
            "trend term may still be better for forecasting, but the evidence here is weak."
        )
        # HP and CF have no F_T at all (their trend is defined by a smoothing parameter, not by a
        # fitted model), so `dec.trend_strength` is None. `np.isfinite(None)` raises a TypeError,
        # which reached the API as a 500 on any HP/CF run whose Mann–Kendall test found no drift.
        if (dec.trend_strength is not None and np.isfinite(dec.trend_strength)
                and dec.trend_strength >= 0.30):
            caveats.append(
                f"F_T = {_num(dec.trend_strength)} is high even though the drift test is not "
                "significant — F_T divides by the remainder, so a small remainder inflates it. "
                "The band above follows the test, not F_T."
            )
    if dec.method in {"hp", "cf"}:
        caveats.append(
            f"{dec.method.upper()} trends are defined by a smoothing parameter, so the direction "
            "shown here is partly a choice rather than a property of the data."
        )
    if np.isfinite(p_cs) and p_cs < 0.05:
        if drift_sig:
            caveats.append(
                f"Cox–Stuart agrees there is drift (z = {_num(z_cs)}, {p_phrase(p_cs)}), which "
                "means the change is persistent rather than one jump."
            )
        else:
            caveats.append(
                f"The drift tests disagree: Cox–Stuart finds drift at 5% (z = {_num(z_cs)}, "
                f"{p_phrase(p_cs)}) while Mann–Kendall does not. With two tests at 5% on the "
                "same data, one false positive is expected about a tenth of the time — so this is "
                "weak evidence, not a trend."
            )
    return ComponentFinding(
        id="stat.trend", name="Trend",
        statistic=tau, p_value=tau_p, band=band, plain=plain,
        detail=(
            f"Δ over window: {delta:+.6g}"
            + (f" ({pct:+.2f}%)" if np.isfinite(pct) else "")
            + f"; Sen's slope {slope:.6g}/period (CI {lo:.6g}..{hi:.6g}); Cox–Stuart z {z_cs:.3f}."
        ),
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #
def seasonal_finding(dec: Decomposition, series: pd.Series, period: int | None) -> ComponentFinding:
    """
    The seasonal component: size, shape, timing, and whether it is stable.

    Amplitude is reported in the series' own units and as a percentage of the mean level, because
    "a swing of 40" means nothing until you know whether the series averages 50 or 5000.
    """
    s = dec.seasonal.dropna()
    if period is None or period < 2 or s.empty:
        return ComponentFinding(
            id="stat.seasonality", name="Seasonality", statistic=None, p_value=None, band="none",
            plain="No seasonal period was supplied, so no seasonal component was estimated.",
        )

    # HP and CF have **no seasonal component**: their second panel is a band-pass/HP *cycle*, and
    # the decomposition itself says so in `caveats`. Computing a seasonal shape, a peak position
    # and a Kruskal–Wallis p-value from that panel produced sentences like "Seasonality repeats
    # every 1440 observations and carries F_S = not computed" on a 100-point series — a measured-
    # sounding claim about a component that does not exist. Refuse, and say what the panel is.
    if dec.method in {"hp", "cf"}:
        return ComponentFinding(
            id="stat.seasonality", name="Seasonality", statistic=None, p_value=None, band="none",
            plain=(
                f"{dec.method.upper()} estimates no seasonal component. Its second panel is a "
                "cycle — variation inside a chosen frequency band — not a repeating calendar "
                "effect, so seasonal strength, peak position and a seasonal-shape test are not "
                "defined for it. Use STL or Classical for seasonal questions."
            ),
            caveats=[
                "A cycle and a season are different claims: a season repeats on the calendar, a "
                "cycle repeats at a data-dependent period. Do not read this panel as seasonality.",
            ],
        )
    # A state-space fit can decline to estimate a seasonal term (not enough complete cycles). Its
    # `dec.period` is None while the *calendar* `period` still holds a number; using that number
    # here would describe a component the fit never produced.
    if dec.period is None:
        return ComponentFinding(
            id="stat.seasonality", name="Seasonality", statistic=None, p_value=None, band="none",
            plain=(
                f"{dec.label} did not estimate a seasonal component: the calendar suggests a "
                f"period of {period}, but the series is too short for the fit to estimate one. "
                "No seasonal strength or shape is reported."
            ),
        )
    if len(s) < 2 * period:
        return ComponentFinding(
            id="stat.seasonality", name="Seasonality", statistic=None, p_value=None, band="none",
            plain=(
                f"The seasonal component covers {len(s)} observations against a period of "
                f"{period} — fewer than two full cycles, so a seasonal shape would be a "
                "coincidence rather than a measurement. No peak position is reported."
            ),
        )

    amplitude = float(s.max() - s.min())
    mean_level = float(np.nanmean(series.to_numpy(dtype=float)))
    amp_pct = 100.0 * amplitude / abs(mean_level) if mean_level != 0 else float("nan")
    band = _band(dec.seasonal_strength)

    # Shape: the average of each position in the cycle, as a deviation from the component mean.
    pos = np.arange(len(s)) % period
    shape = pd.Series(s.to_numpy(dtype=float)).groupby(pos).mean()
    shape = shape - shape.mean()
    peak_pos = int(shape.idxmax()) + 1
    trough_pos = int(shape.idxmin()) + 1

    # Kruskal–Wallis across the seasonal groups: is the pattern real, or within-group noise?
    groups = [s.to_numpy(dtype=float)[pos == k] for k in range(period)]
    groups = [g for g in groups if len(g) >= 2]
    kw_p: float | None = None
    kw_h: float | None = None
    if len(groups) >= 3:
        with np.errstate(invalid="ignore"):
            kw_h, kw_p = stats.kruskal(*groups)
            kw_h, kw_p = float(kw_h), float(kw_p)

    # Is the pattern getting bigger or smaller?
    cycles = [shape_block for shape_block in np.array_split(s.to_numpy(dtype=float), max(2, len(s) // period))]
    amps = [float(np.max(b) - np.min(b)) for b in cycles if len(b) >= period]
    growing: bool | None = None
    if len(amps) >= 3:
        rho, rho_p = stats.spearmanr(np.arange(len(amps)), amps)
        growing = bool(rho > 0 and rho_p < 0.10)

    plain = (
        f"Seasonality repeats every {period} observations and carries F_S = "
        f"{_num(dec.seasonal_strength)} ({band}) of the variation. The swing is {amplitude:.4g} "
        f"units peak to trough"
        + (f", which is {amp_pct:.1f}% of the series mean" if np.isfinite(amp_pct) else "")
        + f". The high point is position {peak_pos} of {period} and the low point is position "
        f"{trough_pos}"
        + (f"; Kruskal–Wallis {p_phrase(kw_p)} across the positions." if kw_p is not None
           else ".")
    )
    caveats: list[str] = []
    if growing:
        caveats.append(
            f"The seasonal swing is {'growing' if growing else 'shrinking'}: the amplitude rises "
            f"from {amps[0]:.3g} in the first cycle to {amps[-1]:.3g} in the last. A fixed "
            "seasonal term will under-predict the recent swings — a multiplicative model, or a "
            "seasonal term allowed to evolve, fits better."
        )
    if kw_p is not None and kw_p >= 0.05:
        caveats.append(
            "Kruskal–Wallis cannot separate the seasonal positions, so the 'seasonality' here may "
            "be the trend leaking into the seasonal component."
        )
    return ComponentFinding(
        id="stat.seasonality", name="Seasonality",
        statistic=dec.seasonal_strength, p_value=kw_p, band=band, plain=plain,
        detail=(
            f"Amplitude {amplitude:.6g} units ({amp_pct:.2f}% of the mean); peak position "
            f"{peak_pos}/{period}, trough {trough_pos}/{period}; Kruskal–Wallis H = "
            f"{_num(kw_h)}; cycle amplitudes {', '.join(f'{a:.3g}' for a in amps[:6])}."
        ),
        table=[
            {"position": int(i) + 1, "effect": round(float(v), 6)}
            for i, v in shape.items()
        ],
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# Cycle / spectrum
# --------------------------------------------------------------------------- #
def fisher_g(series: pd.Series) -> tuple[float, float, float]:
    """
    Fisher's g-test for a hidden periodicity: is the tallest periodogram ordinate more than a
    noise spectrum would produce? Returns (g, p, period of the peak).
    """
    y = np.asarray(series.dropna(), dtype=float)
    y = y - y.mean()
    n = len(y)
    if n < 8:
        return float("nan"), float("nan"), float("nan")
    power = np.abs(np.fft.rfft(y)) ** 2
    power = power[1:]                                    # drop the zero frequency
    total = float(power.sum())
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    g = float(power.max() / total)
    m = len(power)

    def _fisher_p(gval: float, mm: int) -> float:
        """P(G > g) = Σ_k (−1)^{k−1} C(m,k) (1 − k·g)^{m−1}, summed while k·g < 1."""
        acc = 0.0
        for k in range(1, int(np.floor(1.0 / gval)) + 1):
            term = float(comb(mm, k)) * (1.0 - k * gval) ** (mm - 1)
            acc += ((-1) ** (k - 1)) * term
        return float(min(1.0, max(0.0, acc)))

    p = _fisher_p(g, m)
    peak_index = int(np.argmax(power)) + 1
    period_peak = n / peak_index if peak_index else float("nan")
    return g, p, period_peak


def cycle_finding(series: pd.Series, period: int | None,
                  dec_trend: pd.Series | None = None) -> ComponentFinding:
    """
    Periodogram peaks and what they mean.

    The distinction the plain text has to make: a spectral peak at a calendar lag is seasonality;
    a peak anywhere else is a cycle, which has no fixed date and cannot be modelled with a
    seasonal term.
    """
    y = series.dropna()
    if len(y) < 16:
        return ComponentFinding(
            id="stat.cycle", name="Cycle", statistic=None, p_value=None, band="none",
            plain="The series is too short to look for cycles: a spectrum needs at least a few "
                  "dozen observations before a peak means anything.",
        )
    # The spectrum must be taken on the DETRENDED series. A trend is a huge amount of power at
    # the lowest frequency, and a periodogram of the raw series therefore always reports "the
    # strongest cycle has a period of the whole sample" — which is the trend wearing a disguise.
    detrended = (y - dec_trend.reindex(y.index)).dropna() if dec_trend is not None else y
    g, p, peak_period = fisher_g(detrended)
    vals = np.asarray(detrended, dtype=float) - float(np.mean(detrended))
    power = np.abs(np.fft.rfft(vals)) ** 2
    freqs = np.fft.rfftfreq(len(vals))
    order = np.argsort(power[1:])[::-1][:5] + 1
    peaks = [
        {"period": round(float(1 / freqs[i]), 2) if freqs[i] > 0 else None,
         "power": round(float(power[i]), 6)}
        for i in order
    ]
    top = peaks[0] if peaks else None
    on_calendar = bool(
        period and top and top["period"] and abs(float(top["period"]) - period) <= max(1.0, 0.15 * period)
    )
    # Fisher's g tests for ONE sharp hidden periodicity and is badly calibrated on short noise
    # spectra: on simulated white noise of this length it fires at ~7–8% when it claims 5%, and
    # it found a "cycle" in the app's own white-noise example. A peak is therefore only called a
    # cycle when it clears 1% as well as 5%, and the caveat says which test was used.
    significant = np.isfinite(p) and p < 0.01
    if not significant:
        plain = (
            f"No cycle clears the threshold: Fisher's g = {_num(g, 4)} with {p_phrase(p)} "
            "(1% is the bar here). The tallest periodogram peak is about where a spectrum of pure "
            "noise would put one, so there is no repeating pattern to model beyond seasonality."
        )
        band = "none"
    elif on_calendar:
        plain = (
            f"The strongest periodicity is at {top['period']:g} observations, which matches the "
            f"seasonal period — this is seasonality, already accounted for in the seasonal "
            f"component, not an extra cycle (Fisher's g = {_num(g, 4)}, {p_phrase(p)})."
        )
        band = "moderate"
    else:
        plain = (
            f"The strongest periodicity has a period of about {top['period']:g} observations "
            f"(Fisher's g = {_num(g, 4)}, {p_phrase(p)}), which is NOT the seasonal period "
            f"({period if period else 'none'}). That is a cycle, not a season: it has no fixed "
            "calendar position, so a seasonal term will not capture it."
        )
        band = "moderate"
    caveats = [
        "The spectrum is computed on the detrended series: on a trending series, power at the "
        "lowest frequency is the trend itself, and reporting it as a cycle is a classic mistake.",
        "Fisher's g looks for a single sharp periodicity. It over-rejects on short samples (about "
        "7% at a nominal 5%), which is why a 1% bar is used and why a marginal result is reported "
        "as 'no cycle' rather than as a weak one.",
    ]
    return ComponentFinding(
        id="stat.cycle", name="Cycle",
        statistic=g, p_value=p, band=band, plain=plain,
        detail="Top periodogram peaks: " + ", ".join(
            f"{pk['period']:g} obs ({pk['power']:.3g})" for pk in peaks if pk["period"]
        ),
        table=peaks,
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# Remainder
# --------------------------------------------------------------------------- #
def remainder_finding(dec: Decomposition) -> ComponentFinding:
    """
    Is the remainder actually noise? Three checks, each with a different consequence.

    Autocorrelation → the model needs AR/MA terms and a seasonal term.
    ARCH → the model needs a variance equation (Part 11), not a level equation change.
    Non-normality → prediction intervals from a Gaussian model will be the wrong width.
    """
    r = dec.remainder.dropna()
    n = len(r)
    if n < 20:
        return ComponentFinding(
            id="stat.remainder", name="Remainder", statistic=None, p_value=None, band="none",
            plain="Too few observations to test whether the remainder is noise.",
        )
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

    if float(np.nanstd(r.to_numpy(dtype=float))) <= 1e-12:
        return ComponentFinding(
            id="stat.remainder", name="Remainder", statistic=None, p_value=None, band="none",
            plain=(
                "The remainder is constant to machine precision — the decomposition reproduced the "
                "series almost exactly. That is a warning, not a triumph: it usually means the "
                "series is deterministic (a constructed example) or that the method has too much "
                "freedom for this many observations."
            ),
            caveats=[
                "Tests of 'is the remainder noise?' are undefined on a constant remainder, so they "
                "were not run.",
            ],
        )

    lag = min(10, max(1, n // 5))
    try:
        lb = acorr_ljungbox(r, lags=[lag], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[-1])
        lb_stat = float(lb["lb_stat"].iloc[-1])
    except Exception:                                     # pragma: no cover
        lb_stat, lb_p = float("nan"), float("nan")

    arch_p: float | None = None
    arch_stat: float | None = None
    try:
        with np.errstate(invalid="ignore"):
            arch_stat, arch_p, _, _ = het_arch(r, nlags=min(5, max(1, n // 10)))
        arch_stat, arch_p = float(arch_stat), float(arch_p)
    except Exception:                                     # pragma: no cover
        pass

    jb_stat, jb_p = stats.jarque_bera(r) if n >= 8 else (float("nan"), float("nan"))
    jb_stat, jb_p = float(jb_stat), float(jb_p)

    share = dec.shares.get("remainder", float("nan"))
    parts = [
        f"The remainder holds {share * 100:.1f}% of the variation"
        if np.isfinite(share) else "The remainder's share was not computed",
        f"Ljung–Box at lag {lag}: {p_phrase(lb_p)}"
        + (" — there is still autocorrelation left" if np.isfinite(lb_p) and lb_p < 0.05
           else " — no autocorrelation left"),
    ]
    if arch_p is not None:
        parts.append(
            f"ARCH-LM {p_phrase(arch_p)}"
            + (" — the variance is not constant (that is structure, not noise)"
               if arch_p < 0.05 else " — no evidence of changing variance")
        )
    parts.append(f"Jarque–Bera {p_phrase(jb_p)} " + ("— not normal" if jb_p < 0.05 else "— consistent with normal"))
    plain = ". ".join(parts) + "."

    caveats: list[str] = []
    if np.isfinite(lb_p) and lb_p < 0.05:
        caveats.append(
            "Autocorrelation in the remainder means the decomposition has not extracted all the "
            "predictable structure: the model stage needs those terms, and a decomposition is not "
            "a substitute for one."
        )
    if arch_p is not None and arch_p < 0.05:
        caveats.append(
            "Changing variance is a *feature* of the series, not dirt: a constant-variance model "
            "will produce intervals that are too narrow in volatile stretches and too wide in calm "
            "ones."
        )
    if jb_p < 0.05:
        caveats.append(
            "A non-normal remainder makes Gaussian prediction intervals the wrong width — check "
            "the empirical coverage before quoting an interval."
        )
    band = "strong" if (np.isfinite(lb_p) and lb_p < 0.05) else "weak"
    return ComponentFinding(
        id="stat.remainder", name="Remainder",
        statistic=lb_stat, p_value=lb_p, band=band, plain=plain,
        detail=(
            f"Variance share {share:.4f}; Ljung–Box {lb_stat:.3f} (p {lb_p:.4g}); ARCH-LM "
            f"{_num(arch_stat)} (p {_num(arch_p, 4)}); Jarque–Bera {jb_stat:.3f} (p {jb_p:.4g})."
        ),
        caveats=caveats,
    )


# --------------------------------------------------------------------------- #
# The scorecard
# --------------------------------------------------------------------------- #
def scorecard(dec: Decomposition, series: pd.Series, period: int | None) -> dict[str, Any]:
    """
    One table ranking trend / season / remainder, plus the sentence that reads it.

    This is the answer a non-expert actually wants — "what is this series mostly made of?" — and
    it is the reason the decomposition is worth running at all.
    """
    findings = [
        trend_finding(dec),
        seasonal_finding(dec, series, period),
        cycle_finding(series, period, dec.trend),
        remainder_finding(dec),
    ]
    shares = dec.shares
    # HP/CF's second panel is a cycle, not a season. Labelling its variance share "Seasonality"
    # contradicted both the finding above and the decomposition's own caveat, so the row takes the
    # name of the component that was actually estimated.
    season_name = "Cycle" if dec.method in {"hp", "cf"} else "Seasonality"
    season_word = "cycle" if dec.method in {"hp", "cf"} else "seasonal"
    # The remainder gets no F-band: Hyndman's F_T/F_S are strength measures for a *signal*, and
    # printing "strong" next to a remainder means the opposite of what a reader would assume.
    rows = [
        {"component": "Trend", "share": round(shares.get("trend", float("nan")), 4),
         "reading": findings[0].band},
        {"component": season_name, "share": round(shares.get("seasonal", float("nan")), 4),
         "reading": findings[1].band},
        {"component": "Remainder", "share": round(shares.get("remainder", float("nan")), 4),
         "reading": "not applicable"},
    ]
    ranked = sorted(rows, key=lambda r: -(r["share"] if np.isfinite(r["share"]) else -1))
    headline = "The decomposition could not be summarised."
    if not ranked or not np.isfinite(ranked[0]["share"]):
        plain = "The decomposition could not be summarised: the shares were not computable."
    else:
        by = {r["component"]: r["share"] for r in rows}
        def pct(x: float) -> str:
            v = 100 * x
            if v == 0:
                return "0%"
            if abs(v) < 0.1:
                return "<0.1%"
            return f"{v:.0f}%" if abs(v) >= 10 else f"{v:.1f}%"

        parts = (
            f"This series is {pct(by['Trend'])} trend, {pct(by[season_name])} "
            f"{season_word} and {pct(by['Remainder'])} remainder."
        )
        # Built explicitly rather than sliced out of `plain` later: the percentages contain
        # decimal points, so `plain.split(".")[0]` turns "2.9% trend" into the headline
        # "This series is 2."
        headline = parts
        if ranked[0]["component"] == "Remainder":
            parts += (
                " The remainder dominates, which means there is no strong trend or seasonal shape "
                "to model — the useful structure, if any, is in the serial dependence of the "
                "remainder itself (that is what Part 6 looks at)."
            )
            top_findings = [f for f in findings[:2] if f.band in {"strong", "moderate"}]
            if top_findings:
                parts += (
                    " Note that "
                    + " and ".join(f.name.lower() for f in top_findings)
                    + " still score above 'weak' — that is real, just small."
                )
        else:
            parts += (
                f" The {ranked[0]['component'].lower()} is the main structure, so a model that "
                "captures it will do most of the work; the remainder is what is left to forecast."
            )
        plain = parts
    return {
        "method": dec.method,
        "label": dec.label,
        "model": dec.model,
        "rows": rows,
        "plain": plain,
        "headline": headline,
        "findings": findings,
        "trend_strength": dec.trend_strength,
        "seasonal_strength": dec.seasonal_strength,
    }


__all__ = [
    "ComponentFinding", "mann_kendall", "sens_slope", "cox_stuart", "fisher_g",
    "trend_finding", "seasonal_finding", "cycle_finding", "remainder_finding", "scorecard",
    "STRONG", "WEAK",
]
