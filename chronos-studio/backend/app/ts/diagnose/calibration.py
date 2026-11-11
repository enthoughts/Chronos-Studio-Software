"""
ts/diagnose/calibration.py — PIT and randomized quantile residuals (§5.9).

These two test the thing nothing else in the app tests: whether the **predictive distribution** is
calibrated. A model can have perfect point forecasts and intervals that are half as wide as they
should be, and every residual-ACF plot will look fine.

**PIT must be randomized for a discrete model.** For a count model, PIT is `F(y)` — a step function —
and under a correct model the PIT values pile up on a countable set of atoms rather than filling [0,1].
The fix is the randomized PIT: draw `u ~ U(F(y−1), F(y))` per observation. ONE seeded draw is the
inferential sample; extra seeded draws quantify Monte Carlo stability. Averaging randomized draws
shrinks their variance toward the centre and would make a correct model look hill-shaped.

**RQR** applies the same randomization on the normal scale: `Φ⁻¹(u)`. The residuals are then exactly
N(0,1) under a correct model. The same one-draw rule applies: repetitions are not independent
tests and must not be Fisher-combined.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

#: §5.9: "randomized, so seeded and repeated R times". Repetitions are for a
#: stability statement, NOT for a combined p-value across dependent draws.
RQR_REPS = 5
PIT_REPS = 5
SEED = 20260101


@dataclass
class PITResult:
    """The probability integral transform, with two uniformity tests and draw stability."""
    ok: bool
    values: list[float] = field(default_factory=list)
    ks_stat: float | None = None
    ks_p: float | None = None
    cvm_stat: float | None = None
    """Cramér–von Mises — more power against U/hill/skew shapes than KS."""
    cvm_p: float | None = None
    kind: str = "continuous"
    hist: list[int] = field(default_factory=list)
    reading: str = ""
    caveat: str = ""
    note: str = ""
    reps: int = 0
    per_rep_ks_p: list[float] = field(default_factory=list)
    stability: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "values": self.values,
            "ks_stat": self.ks_stat,
            "ks_p": self.ks_p,
            "cvm_stat": self.cvm_stat,
            "cvm_p": self.cvm_p,
            "kind": self.kind,
            "hist": self.hist,
            "reading": self.reading,
            "caveat": self.caveat,
            "note": self.note,
            "reps": self.reps,
            "per_rep_ks_p": self.per_rep_ks_p,
            "stability": self.stability,
        }


@dataclass
class RQRResult:
    """Randomized quantile residuals from ONE seeded draw, with stability disclosed."""
    ok: bool
    values: list[float] = field(default_factory=list)
    qq_theoretical: list[float] = field(default_factory=list)
    qq_sample: list[float] = field(default_factory=list)
    ks_stat: float | None = None
    ks_p: float | None = None
    n: int = 0
    reps: int = 0
    per_rep_ks_p: list[float] = field(default_factory=list)
    stability: str = ""
    caveat: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "values": self.values,
            "qq_theoretical": self.qq_theoretical,
            "qq_sample": self.qq_sample,
            "ks_stat": self.ks_stat,
            "ks_p": self.ks_p,
            "n": self.n,
            "reps": self.reps,
            "per_rep_ks_p": self.per_rep_ks_p,
            "stability": self.stability,
            "caveat": self.caveat,
            "note": self.note,
        }


def _cvm_uniform(u: np.ndarray) -> tuple[float, float]:
    """
    Cramér–von Mises against U(0,1), with the asymptotic p-value (Stephens' T).

    T = n·ω² in Stephens' form; the survival function uses the standard
    Anderson–Darling (1952) series. The previous implementation contained a dead
    `sum(... * 0)` branch that looked like a tabulated small-sample p-value.
    """
    u = np.sort(np.clip(np.asarray(u, dtype=float), 1e-12, 1 - 1e-12))
    n = u.size
    i = np.arange(1, n + 1)
    t = n * (1.0 / (12.0 * n) + float(((u - (2 * i - 1) / (2.0 * n)) ** 2).sum()))
    if t <= 0.347:
        return float(t), 1.0
    return float(t), max(0.0, min(1.0, float(_cvm_sf(t))))


def _cvm_sf(t: float) -> float:
    """Survival function of the Cramér–von Mises limiting distribution."""
    if t <= 0:
        return 1.0
    total = 0.0
    for k in range(0, 40):
        term = (
            math.exp(-((4 * k + 1) ** 2) / (16.0 * t))
            * math.sqrt(4 * k + 1)
            * (math.gamma(k + 0.5) / (math.gamma(0.5) * math.factorial(k)))
        )
        total += ((-1) ** k) * term
    val = total / (math.pi * math.sqrt(t))
    return float(min(1.0, max(0.0, val)))


def _reading(hist: list[int], n: int) -> str:
    """What the shape means, because a p-value says only that something is wrong."""
    if not hist or n == 0:
        return ""
    shares = np.asarray(hist, dtype=float) / n
    first, last = shares[0], shares[-1]
    mid = float(shares[3:7].mean()) if len(shares) >= 7 else float(shares.mean())
    peak_bin = int(np.argmax(shares))
    if first > 1.6 * mid and last > 1.6 * mid:
        return (
            "U-shape: too many PIT values at both ends. The intervals are too narrow — the truth "
            "falls outside them more often than the nominal rate says it should."
        )
    if abs(peak_bin - 4) <= 1 and shares[peak_bin] > 1.6 * mid:
        return (
            "Hill-shape: too many PIT values in the middle. The intervals are too wide — they cover "
            "more than they claim, which wastes the information they carry."
        )
    if first > 1.6 * mid:
        return (
            "Left-skewed: too many PIT values near 0. The model's distribution sits above the data — "
            "a systematic over-prediction, not a width problem."
        )
    if last > 1.6 * mid:
        return (
            "Right-skewed: too many PIT values near 1. The model's distribution sits below the data — "
            "a systematic under-prediction."
        )
    return "Approximately uniform: the predictive distribution matches the outcomes that occurred."


def _stability(ps: list[float], reps: int) -> str:
    if not ps:
        return "No seeded repetitions were available."
    agree = max(ps) - min(ps) < 0.15 or (min(ps) >= 0.05 and max(ps) >= 0.05) or max(ps) < 0.05
    return (
        f"KS p-values across {reps} randomizations stayed in [{min(ps):.3f}, {max(ps):.3f}]; "
        "the verdict does not depend on the draw."
        if agree else
        f"KS p-values moved from {min(ps):.3f} to {max(ps):.3f} across {reps} randomizations; "
        "the verdict is inconclusive at this sample size."
    )


def _pad(values: Sequence[float] | np.ndarray, size: int, fill: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(size, fill, dtype=float)
    out[: min(size, arr.size)] = arr[: min(size, arr.size)]
    return out


def pit_from_quantiles(
    actual: Sequence[float],
    *,
    cdf: Sequence[float] | None = None,
    lower_cdf: Sequence[float] | None = None,
    kind: str = "continuous",
    reps: int = PIT_REPS,
    seed: int = SEED,
) -> PITResult:
    """
    PIT from the fitted predictive CDF at each actual value.

    * `cdf` — F(y_t) for a continuous model.
    * `lower_cdf` + `cdf` — F(y_t−1), F(y_t) for a discrete model. ONE seeded
      uniform draw inside the atom is tested; extra draws only report stability.

    Non-finite actuals or CDF bounds are excluded and counted in the note.
    """
    y = np.asarray(actual, dtype=float).reshape(-1)
    if y.size < 10:
        return PITResult(ok=False, kind=kind, note=f"Only {y.size} observations; PIT needs at least 10.")
    valid = np.isfinite(y)
    ks_ps: list[float] = []
    reps_used = 0
    caveat = ""
    if kind == "discrete":
        if lower_cdf is None or cdf is None:
            return PITResult(
                ok=False, kind=kind,
                note=(
                    "A discrete model needs both F(y−1) and F(y) to randomize its PIT. Without "
                    "the lower CDF the values land on the atoms of the PMF and the histogram "
                    "would look non-uniform even for a perfect model."
                ),
            )
        lo = np.clip(_pad(lower_cdf, y.size, 0.0), 0.0, 1.0)
        hi = np.clip(_pad(cdf, y.size, 1.0), 0.0, 1.0)
        hi = np.maximum(hi, lo)
        valid &= np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        lo, hi = lo[valid], hi[valid]
        if lo.size < 10:
            return PITResult(ok=False, kind=kind, note="Fewer than 10 finite randomized PIT values.")
        rng = np.random.default_rng(seed)
        vals: list[np.ndarray] = []
        reps_used = max(1, int(reps))
        for _ in range(reps_used):
            uu = lo + rng.random(lo.size) * (hi - lo)
            vals.append(uu)
            ks_ps.append(float(stats.kstest(uu, "uniform").pvalue))
        u = np.clip(vals[0], 1e-12, 1 - 1e-12)
        ks = stats.kstest(u, "uniform")
        caveat = (
            "Discrete model: ONE seeded randomized PIT draw is plotted and tested; "
            f"{reps_used} seeded draws quantify stability, not independent tests. A single "
            "draw alone is not a test unless its stability is read too."
        )
    else:
        if cdf is None:
            return PITResult(ok=False, kind=kind, note="No predictive CDF was supplied, so PIT cannot be computed.")
        c = _pad(cdf, y.size, np.nan)
        valid &= np.isfinite(c)
        u = np.clip(c[valid], 1e-12, 1 - 1e-12)
        if u.size < 10:
            return PITResult(ok=False, kind=kind, note="Fewer than 10 finite PIT values.")
        ks = stats.kstest(u, "uniform")

    hist = np.histogram(u, bins=10, range=(0.0, 1.0))[0].tolist()
    cvm_stat, cvm_p = _cvm_uniform(u)
    excluded = int(y.size - u.size)
    note = (
        f"PIT uses {u.size} finite observation(s); {excluded} were excluded. "
        "One seeded draw is the inferential sample; repetitions are dependent."
    )
    return PITResult(
        ok=True,
        values=[round(float(v), 3) for v in u],
        ks_stat=round(float(ks.statistic), 4),
        ks_p=round(float(ks.pvalue), 4),
        cvm_stat=round(cvm_stat, 4),
        cvm_p=round(cvm_p, 4),
        kind=kind,
        hist=hist,
        reading=_reading(hist, int(u.size)),
        caveat=caveat,
        note=note,
        reps=reps_used,
        per_rep_ks_p=[round(p, 4) for p in ks_ps],
        stability=_stability(ks_ps, reps_used) if ks_ps else "",
    )


def rqr_from_draws(
    actual: Sequence[float],
    draws: Sequence[Sequence[float]],
    *,
    reps: int = RQR_REPS,
    seed: int = SEED,
) -> RQRResult:
    """
    Randomized quantile residuals from predictive draws.

    For each observation `u = (rank + random jitter) / (n_finite_draws + 1)`,
    then `z = Φ⁻¹(u)`. NaNs in the ensemble are excluded per column. ONE seeded
    draw is tested/plotted; extra draws report KS-p stability. Averaging them
    would shrink the normal variance and invalidate the test.
    """
    y = np.asarray(actual, dtype=float).reshape(-1)
    d = np.asarray(draws, dtype=float)
    if d.ndim != 2:
        return RQRResult(ok=False, note="Predictive draws must be a 2-D array (draws × observations).")
    if d.shape[1] != y.size:
        return RQRResult(
            ok=False,
            note=f"{d.shape[1]} draw columns against {y.size} observations; they must align.",
        )
    if d.shape[0] < 20:
        return RQRResult(ok=False, note="At least 20 predictive draws are needed to estimate F(y).")

    finite_d = np.isfinite(d)
    # NaN draws compare False; the denominator is the per-column finite count.
    counts = (np.where(finite_d, d, np.inf) < y[None, :]).sum(axis=0)
    n_eff = finite_d.sum(axis=0)
    usable = np.isfinite(y) & (n_eff >= 20)
    idx = np.flatnonzero(usable)
    if idx.size < 10:
        return RQRResult(
            ok=False, note=f"Only {idx.size} columns have a finite actual and ≥20 finite draws.",
        )

    rng = np.random.default_rng(seed)
    per_rep: list[np.ndarray] = []
    ks_ps: list[float] = []
    reps_used = max(1, int(reps))
    for _ in range(reps_used):
        z = np.full(y.size, np.nan, dtype=float)
        u = (counts[idx] + rng.random(idx.size)) / (n_eff[idx] + 1.0)
        z[idx] = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        per_rep.append(z)
        ks_ps.append(float(stats.kstest(z[idx], "norm").pvalue))
    z = per_rep[0][idx]
    zs = np.sort(z)
    n = zs.size
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ks = stats.kstest(z, "norm")
    return RQRResult(
        ok=True,
        values=[round(float(v), 3) for v in z],
        qq_theoretical=[round(float(v), 3) for v in theo],
        qq_sample=[round(float(v), 3) for v in zs],
        ks_stat=round(float(ks.statistic), 4),
        ks_p=round(float(ks.pvalue), 4),
        n=int(n),
        reps=reps_used,
        per_rep_ks_p=[round(p, 4) for p in ks_ps],
        stability=_stability(ks_ps, reps_used),
        caveat=(
            "RQR is randomized by construction: ONE seeded draw inside each predictive "
            "probability step is plotted and tested; further seeded draws measure Monte "
            "Carlo stability. Averaging randomized residuals shrinks their variance and "
            "would invalidate the KS test."
        ),
        note="Per-column finite draw counts exclude NaNs; repetitions are dependent, not combined.",
    )


def coverage_tests(
    inside: Sequence[bool], *, nominal: float = 0.9
) -> dict[str, Any]:
    """
    Is the achieved coverage consistent with the nominal one?

    A binomial test, not a threshold. "Coverage 86% against 90% nominal" is not automatically a
    failure: with 40 intervals, 86% is well inside sampling variation.
    """
    a = np.asarray(inside, dtype=bool)
    n = a.size
    if n < 10:
        return {"n": n, "coverage": None, "p": None, "note": "Too few intervals to test coverage."}
    k = int(a.sum())
    res = stats.binomtest(k, n, nominal)
    lo, hi = res.proportion_ci(confidence_level=0.95).low, res.proportion_ci(confidence_level=0.95).high
    return {
        "n": n,
        "covered": k,
        "coverage": round(k / n, 4),
        "nominal": nominal,
        "p": round(float(res.pvalue), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "note": (
            f"{k} of {n} intervals contained the outcome ({k / n:.1%}) against a nominal {nominal:.0%}; "
            f"the 95% interval for that proportion is {lo:.1%}–{hi:.1%}."
        ),
    }


__all__ = [
    "PIT_REPS",
    "PITResult",
    "RQR_REPS",
    "RQRResult",
    "SEED",
    "coverage_tests",
    "pit_from_quantiles",
    "rqr_from_draws",
]
