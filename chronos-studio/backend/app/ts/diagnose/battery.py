"""
ts/diagnose/battery.py — the residual battery (§5.9).

Every test here answers a *different* question, and the docstrings say which one. The two most common
mistakes this module is written to prevent:

1. **Ljung–Box at a single lag.** Autocorrelation can hide at lag 1 and reappear at the seasonal lag.
   §5.9 asks for "multiple lags (including 2s)", so `ljung_box_multi` returns the full grid and the
   verdict reads it, rather than testing at `min(10, n/5)` and hoping.

2. **CUSUM of the raw residuals.** The spec asks for CUSUM *of residuals*, which detects a shift in
   the mean of the residuals — that is a bias, not "structural change". The recursive-coefficient
   version (parameter stability) is a separate column, and the two are labelled differently so a
   reader cannot mistake one for the other.

Where a test cannot run — too few observations, a constant series, a degenerate design — the result is
`None` with a reason. A missing number is honest; a p-value from an invalid test is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

#: §5.9's "including 2s": the seasonal lag and its second harmonic are always tested when a period is
#: known, because a model can leave a seasonal signature that lag 10 never sees.
DIAGNOSTIC_LAG_RULE = (
    "Ljung–Box is tested at 5, 10, 2s and s lags. A model that passes at 10 and fails at s has left "
    "seasonal autocorrelation behind, and quoting only the lag-10 p-value would hide it."
)


def _clean(x: Sequence[float]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a[np.isfinite(a)]


def _isf(p: float | None) -> float | None:
    if p is None:
        return None
    p = float(p)
    return p if math.isfinite(p) else None


# --------------------------------------------------------------------------- #
# Autocorrelation
# --------------------------------------------------------------------------- #
def acf_with_bounds(x: Sequence[float], max_lag: int = 24) -> dict[str, Any]:
    """
    ACF with the ±1.96/√n band, and the number of lags that poke out of it.

    The band is the white-noise band for *each* lag rather than a single one for all — the variance of
    an estimated autocorrelation grows with the lag, and a flat band overstates the evidence at long
    lags.
    """
    a = _clean(x)
    n = a.size
    if n < 8:
        return {"lags": [], "acf": [], "bounds": [], "n_exceed": 0, "note": "Too few residuals for an ACF."}
    max_lag = int(min(max_lag, max(1, n // 3)))
    ac = a - a.mean()
    denom = float((ac * ac).sum())
    if denom <= 0:
        return {"lags": [], "acf": [], "bounds": [], "n_exceed": 0, "note": "The residuals are constant."}
    lags = list(range(1, max_lag + 1))
    vals = [float((ac[k:] * ac[:-k]).sum() / denom) for k in lags]
    bounds = [float(1.96 / math.sqrt(max(n - k, 1))) for k in lags]
    return {
        "lags": lags,
        "acf": [round(v, 4) for v in vals],
        "bounds": [round(b, 4) for b in bounds],
        "n_exceed": int(sum(abs(v) > b for v, b in zip(vals, bounds))),
        "note": "",
    }


def ljung_box_multi(
    x: Sequence[float], *, period: int | None = None, extra_lags: Sequence[int] = (5, 10)
) -> dict[str, Any]:
    """
    Ljung–Box at several lags at once.

    Returns `{lag: {q, p, dof}}` plus the list of lags that reject at 5%, so the verdict can name
    *which* lag failed rather than reporting "autocorrelation present" from an undisclosed lag.
    """
    a = _clean(x)
    n = a.size
    if n < 12:
        return {"results": {}, "rejects": [], "note": "Too few residuals for Ljung–Box."}
    lags: list[int] = []
    for L in extra_lags:
        if isinstance(L, (int, np.integer)) and 1 <= int(L) < n - 4:
            lags.append(int(L))
    if period and period >= 2:
        for L in (period, 2 * period):
            if 1 <= L < n - 4:
                lags.append(int(L))
    lags = sorted(set(lags)) or [min(10, max(5, n // 5))]

    ac = a - a.mean()
    denom = float((ac * ac).sum())
    if denom <= 0:
        return {"results": {}, "rejects": [], "note": "The residuals are constant."}

    # One pass of the autocorrelations up to the largest lag; each test reuses the prefix.
    max_lag = max(lags)
    r = [float((ac[k:] * ac[:-k]).sum() / denom) for k in range(1, max_lag + 1)]
    results: dict[str, dict[str, float | int]] = {}
    rejects: list[int] = []
    for L in lags:
        q = float(n * (n + 2) * sum(r[k - 1] ** 2 / (n - k) for k in range(1, L + 1)))
        p = _isf(stats.chi2.sf(q, L))
        results[str(L)] = {"q": round(q, 4), "p": round(p, 4) if p is not None else None, "dof": L}
        if p is not None and p < 0.05:
            rejects.append(L)
    return {"results": results, "rejects": rejects, "note": ""}


def box_pierce(x: Sequence[float], lags: int = 10) -> dict[str, float | int | None]:
    """
    Box–Pierce, kept because §5.9 asks for it and because it is *less* powerful than Ljung–Box.

    Reporting both and letting them disagree is informative: when Ljung–Box rejects and Box–Pierce
    does not, the autocorrelation is real but small, which is a different situation from both
    rejecting strongly.
    """
    a = _clean(x)
    n = a.size
    if n < lags + 5:
        return {"q": None, "p": None, "lags": lags}
    ac = a - a.mean()
    denom = float((ac * ac).sum())
    if denom <= 0:
        return {"q": None, "p": None, "lags": lags}
    q = float(n * sum(((ac[k:] * ac[:-k]).sum() / denom) ** 2 for k in range(1, lags + 1)))
    return {"q": round(q, 4), "p": _isf(stats.chi2.sf(q, lags)), "lags": lags}


# --------------------------------------------------------------------------- #
# Normality
# --------------------------------------------------------------------------- #
def jarque_bera_test(x: Sequence[float]) -> dict[str, Any]:
    """Skewness + kurtosis. The reading matters: non-normality invalidates *normal intervals*, not the
    point forecast, and the verdict says so."""
    a = _clean(x)
    if a.size < 8 or np.allclose(a, a[0]):
        return {"statistic": None, "p": None, "skew": None, "kurtosis": None}
    try:
        res = stats.jarque_bera(a)
        return {
            "statistic": round(float(res.statistic), 4),
            "p": _isf(res.pvalue),
            "skew": round(float(stats.skew(a)), 4),
            "kurtosis": round(float(stats.kurtosis(a, fisher=True)), 4),
            "note": "Kurtosis is excess: 0 is Gaussian, positive means heavier tails than normal.",
        }
    except Exception:  # pragma: no cover
        return {"statistic": None, "p": None, "skew": None, "kurtosis": None}


def shapiro_test(x: Sequence[float]) -> dict[str, float | None]:
    """Shapiro–Wilk. Refused below 3 and above 5000 observations, where it is not defined or not useful."""
    a = _clean(x)
    if a.size < 3:
        return {"w": None, "p": None, "note": "Needs at least 3 residuals."}
    if a.size > 5000:
        return {
            "w": None,
            "p": None,
            "note": "Not run: Shapiro–Wilk is calibrated for n ≤ 5000, and at this size it rejects "
            "trivial departures. Jarque–Bera is the appropriate test here.",
        }
    res = stats.shapiro(a)
    return {"w": round(float(res.statistic), 4), "p": _isf(res.pvalue), "note": ""}


def anderson_darling_test(x: Sequence[float]) -> dict[str, Any]:
    """
    Anderson–Darling against the normal — more sensitive in the tails than either of the others, which
    is exactly why it is the one to read on a volatility series.

    `stats.anderson` returns critical values rather than a p-value; the p-value is interpolated from
    the tabulated breakpoints, and the interpolation is labelled as such.
    """
    a = _clean(x)
    if a.size < 8:
        return {"statistic": None, "p": None, "critical": {}, "note": "Too few residuals."}
    # SciPy 1.17 accepts `method="interpolate"` and returns a SignificanceResult
    # with a p-value. SciPy 1.16 and older do not accept that keyword and return
    # the tabulated critical values instead. The offline wheelhouse pins 1.16,
    # so both signatures have to work; a TypeError here used to become a 500 on
    # every S12 diagnostics run.
    try:
        res = stats.anderson(a, dist="norm", method="interpolate")
    except TypeError:  # SciPy < 1.17: no `method` keyword
        res = stats.anderson(a, dist="norm")
    # SciPy 1.17 returns a `SignificanceResult` (statistic + pvalue); older versions return the
    # tabulated critical values instead. Both shapes are handled, and the note says which was used.
    stat = float(res.statistic)
    if hasattr(res, "pvalue"):
        p: float | None = float(res.pvalue)
        crit = {str(round(x * 100)): None for x in (15.0, 10.0, 5.0, 2.5, 1.0)}
        note = (
            "The p-value is interpolated by SciPy from Anderson–Darling's tabulated critical values, "
            "so treat small differences around 5% as noise rather than as a threshold crossing."
        )
        approx = True
    else:  # pragma: no cover - SciPy < 1.17
        crit = {
            str(pct): round(float(v), 4)
            for pct, v in zip(res.significance_level, res.critical_values)
        }
        crits = np.asarray(res.critical_values, dtype=float)
        sigs = np.asarray(res.significance_level, dtype=float) / 100.0
        p = None
        if stat <= crits[0]:
            p = 0.25
        elif stat >= crits[-1]:
            p = float(max(0.0005, sigs[-1] / 4))
        else:
            for i in range(len(crits) - 1):
                if crits[i] <= stat <= crits[i + 1]:
                    frac = (stat - crits[i]) / (crits[i + 1] - crits[i])
                    p = float(sigs[i] + frac * (sigs[i + 1] - sigs[i]))
                    break
        note = (
            "The p-value is interpolated from the tabulated critical values; compare the statistic "
            "with the critical values directly where the decision matters."
        )
        approx = True

    return {
        "statistic": round(stat, 4),
        "p": round(p, 4) if p is not None else None,
        "critical": crit,
        "note": note,
        "approximate": approx,
    }


# --------------------------------------------------------------------------- #
# Runs, signs and CUSUM
# --------------------------------------------------------------------------- #
def runs_test(x: Sequence[float]) -> dict[str, Any]:
    """
    Wald–Wolfowitz runs test on signs: are the residuals alternating more or less than chance?

    Detects the failure the autocorrelation tests can miss — a *pattern* in signs with no linear
    correlation — and detects the opposite, residuals that are too smooth.
    """
    a = _clean(x)
    n = a.size
    pos = int((a > 0).sum())
    neg = int((a < 0).sum())
    if pos + neg < 10:
        return {"runs": None, "expected": None, "p": None, "note": "Too few non-zero residuals."}
    signs = np.sign(a[a != 0])
    runs = 1 + int((signs[1:] != signs[:-1]).sum())
    N = pos + neg
    expected = 1.0 + (2.0 * pos * neg) / N
    var = (2.0 * pos * neg * (2.0 * pos * neg - N)) / (N * N * (N - 1.0))
    if var <= 0:
        return {"runs": runs, "expected": round(expected, 2), "p": None}
    z = (runs - expected) / math.sqrt(var)
    return {
        "runs": runs,
        "expected": round(expected, 2),
        "z": round(float(z), 4),
        "p": _isf(2.0 * stats.norm.sf(abs(z))),
        "note": "Fewer runs than expected means the residuals stay on one side of zero — under-differenced.",
    }


def sign_test(x: Sequence[float]) -> dict[str, float | int | None]:
    """Binomial test that positive and negative residuals are equally likely — a pure bias check."""
    a = _clean(x)
    pos = int((a > 0).sum())
    n = int((a != 0).sum())
    if n < 8:
        return {"positive": pos, "n": n, "p": None}
    p = float(stats.binomtest(pos, n, 0.5).pvalue)
    return {"positive": pos, "n": n, "p": _isf(p)}


def cusum_path(x: Sequence[float]) -> dict[str, Any]:
    """
    CUSUM of the *standardised* residuals, with the 5% boundary.

    The path is `cumsum((r − mean)/sd) / sqrt(n)`, and the 5% line for a stable mean is ±0.948 (the
    Kolmogorov–Smirnov 5% value). A path that wanders outside that band indicates the residual mean
    drifted — a bias or a missing level shift, not "the model is wrong" in general.
    """
    a = _clean(x)
    n = a.size
    if n < 10:
        return {"path": [], "critical": None, "breach": False, "note": "Too few residuals."}
    sd = float(a.std(ddof=1))
    if sd <= 0:
        return {"path": [], "critical": None, "breach": False, "note": "The residuals are constant."}
    z = (a - a.mean()) / sd
    path = np.cumsum(z) / math.sqrt(n)
    crit = 0.948
    breach = bool(np.any(np.abs(path) > crit))
    # Where it breached, in position terms, so the UI can say "from 60% of the series onwards".
    where = None
    idx = np.where(np.abs(path) > crit)[0]
    if idx.size:
        where = {"start_index": int(idx[0]), "end_index": int(idx[-1]), "since": float(idx[0] / n)}
    return {
        "path": [round(float(v), 4) for v in path],
        "critical": crit,
        "breach": breach,
        "breach_where": where,
        "note": "CUSUM of residuals detects a drifting *mean*. Structural change in the coefficients "
        "is a different test (recursive estimation) and is reported separately.",
    }


# --------------------------------------------------------------------------- #
# The whole battery
# --------------------------------------------------------------------------- #
@dataclass
class DiagnosticBattery:
    """Every diagnostic for one model, plus what each one means. No aggregate score."""

    model_id: str
    display_name: str = ""
    n_resid: int = 0
    ljung_box: dict[str, Any] = field(default_factory=dict)
    box_pierce: dict[str, Any] = field(default_factory=dict)
    acf: dict[str, Any] = field(default_factory=dict)
    normality: dict[str, Any] = field(default_factory=dict)
    arch_lm: dict[str, Any] = field(default_factory=dict)
    runs: dict[str, Any] = field(default_factory=dict)
    signs: dict[str, Any] = field(default_factory=dict)
    cusum: dict[str, Any] = field(default_factory=dict)
    outliers: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "n_resid": self.n_resid,
            "ljung_box": self.ljung_box,
            "box_pierce": self.box_pierce,
            "acf": self.acf,
            "normality": self.normality,
            "arch_lm": self.arch_lm,
            "runs": self.runs,
            "signs": self.signs,
            "cusum": self.cusum,
            "outliers": self.outliers,
            "notes": self.notes,
        }


#: A residual beyond this many robust standard deviations is flagged. 3.5 on the *MAD* scale rather
#: than the sd, because a leverage point inflates its own sd and hides itself.
OUTLIER_Z = 3.5


def run_battery(
    resid: Sequence[float],
    *,
    model_id: str = "",
    display_name: str = "",
    period: int | None = None,
    index: Any = None,
    residual_sd: float | None = None,
) -> DiagnosticBattery:
    """
    Compute the battery once.

    `index` (the dates the residuals belong to) is used only to label outliers with their date. Without
    it the outlier list still appears, with positions instead — a flagged point the user cannot find in
    their data is not a finding.
    """
    r = _clean(resid)
    b = DiagnosticBattery(model_id=model_id, display_name=display_name, n_resid=int(r.size))
    if r.size < 12:
        b.notes.append(
            f"Only {r.size} residuals: the battery needs at least 12, so most of it is not computed. "
            "A p-value from 8 residuals is a random number with a decimal point."
        )
        return b

    b.ljung_box = ljung_box_multi(r, period=period)
    b.box_pierce = box_pierce(r)
    b.acf = acf_with_bounds(r, max_lag=min(24, max(8, r.size // 3)))
    b.normality = {
        "jarque_bera": jarque_bera_test(r),
        "shapiro": shapiro_test(r),
        "anderson_darling": anderson_darling_test(r),
        "note": "Non-normal residuals do not invalidate the point forecast. They invalidate intervals "
        "that assume normality — which is why the app reports empirical bands alongside normal ones.",
    }
    b.arch_lm = _arch_lm_block(r)
    b.runs = runs_test(r)
    b.signs = sign_test(r)
    b.cusum = cusum_path(r)
    b.outliers = _outliers(r, index=index, residual_sd=residual_sd)
    return b


def _arch_lm_block(r: np.ndarray, lags: int = 5) -> dict[str, Any]:
    """
    ARCH-LM on the residuals, with §5.9's advice attached.

    The advice is the part that matters: ARCH in the residuals does **not** mean the mean model is
    wrong. It means the variance has structure, and the answer is an ARIMA-GARCH row, not a different
    ARIMA order.
    """
    from app.ts.models.diagnostics import arch_lm_p

    p = arch_lm_p(r, lags=lags)
    if p is None:
        return {"p": None, "advice": "ARCH-LM could not be computed on these residuals."}
    advice = (
        "Heteroskedasticity is present in the residuals. This is not a failure of the mean model — it "
        "means the variance has its own structure. Fit an ARIMA-GARCH row (Part 11) if the intervals "
        "are what you use; the point forecast is unaffected."
        if p < 0.05
        else "No evidence of remaining heteroskedasticity; a constant-variance interval is adequate here."
    )
    return {"p": round(float(p), 4), "lags": lags, "advice": advice}


def _outliers(
    r: np.ndarray, *, index: Any = None, residual_sd: float | None = None
) -> list[dict[str, Any]]:
    """
    Flag residual spikes on the MAD scale, with their date and their influence.

    `influence` is the observation's share of the total squared residual. A point that is 4σ out but is
    1% of the residual sum of squares is a curiosity; one that is 2σ out and 25% of the sum is the
    reason the fit looks bad, and the difference matters to the verdict.
    """
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    scale = 1.4826 * mad if mad > 0 else float(r.std(ddof=1))
    if scale <= 0:
        return []
    z = (r - med) / scale
    total_sq = float((r * r).sum()) or 1.0
    out: list[dict[str, Any]] = []
    for i in np.where(np.abs(z) > OUTLIER_Z)[0]:
        share = float(r[i] ** 2 / total_sq)
        label = None
        try:
            if index is not None and len(index) > int(i):
                label = str(index[int(i)])[:10]
        except Exception:  # pragma: no cover
            label = None
        out.append(
            {
                "position": int(i),
                "when": label,
                "residual": round(float(r[int(i)]), 4),
                "z": round(float(z[int(i)]), 2),
                "influence": round(share, 4),
                "reading": (
                    f"This single point is {share * 100:.0f}% of the residual sum of squares."
                    if share >= 0.1
                    else "Large deviation, but a small share of the total residual variation."
                ),
            }
        )
    out.sort(key=lambda d: -d["influence"])
    return out[:10]


__all__ = [
    "DIAGNOSTIC_LAG_RULE",
    "OUTLIER_Z",
    "DiagnosticBattery",
    "acf_with_bounds",
    "anderson_darling_test",
    "box_pierce",
    "cusum_path",
    "jarque_bera_test",
    "ljung_box_multi",
    "run_battery",
    "runs_test",
    "shapiro_test",
    "sign_test",
]
