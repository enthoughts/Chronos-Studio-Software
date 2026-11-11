"""
ts/metrics/probabilistic.py — scoring the whole predictive distribution (§5.12).

CRPS is computed from the empirical predictive ensemble with the identity

    CRPS(F, y) = E|X − y| − ½ E|X − X'|,   X, X' ~ F independently,

which is the form Gneiting & Raftery (2007) call a proper scoring rule. The
second expectation is evaluated in O(n log n) from the sorted draws, not by
building an n×n matrix.

Every band score is aligned to the FULL number of evaluation rows. A shorter
band is padded with NaN and counted as excluded; it is never silently truncated
to the shortest array. PIT/RQR use one seeded inferential draw with repetitions
reported as stability, never Fisher-combined dependent p-values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

from app.ts.diagnose.calibration import coverage_tests, pit_from_quantiles, rqr_from_draws
from app.ts.forecast.intervals import IntervalBand

_Z975 = float(stats.norm.ppf(0.975))
CRPS_CITATION = "Gneiting & Raftery (2007), Strictly Proper Scoring Rules, JASA 102(477)"


def crps_ensemble(paths: np.ndarray, actual: Sequence[float]) -> tuple[np.ndarray, float]:
    """
    Per-observation CRPS from an ensemble, plus its mean. Lower is better.

    Uses the sorted-draw identity `Σ_{i,j}|x_i−x_j| = 2 Σ_k x_(k)(2k−n+1)`,
    so the ½E|X−X'| term is `Σ_k x_(k)(2k−n+1)/n²`.
    """
    p = np.asarray(paths, dtype=float)
    y = np.asarray(actual, dtype=float).reshape(-1)
    if p.ndim != 2 or p.shape[1] != y.size:
        raise ValueError(f"paths must be (draws × {y.size} observations); got {p.shape}.")
    out = np.full(y.size, np.nan, dtype=float)
    n_draws = p.shape[0]
    if n_draws < 2:
        return out, float("nan")
    for j in range(y.size):
        col = p[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2 or not math.isfinite(y[j]):
            continue
        xs = np.sort(col)
        n = xs.size
        kk = np.arange(n, dtype=float)
        pair_half = float(np.dot(xs, 2.0 * kk - n + 1.0) / (n * n))
        out[j] = float(np.mean(np.abs(xs - y[j])) - pair_half)
    finite = out[np.isfinite(out)]
    return out, float(finite.mean()) if finite.size else float("nan")


def _pinball(y: np.ndarray, f: np.ndarray, q: float) -> tuple[float, np.ndarray]:
    d = y - f
    per = np.maximum(q * d, (q - 1.0) * d)
    mask = np.isfinite(per)
    return (float(per[mask].mean()) if mask.any() else float("nan"),
            np.where(mask, per, np.nan))


def _band_arrays(
    y: np.ndarray, b: IntervalBand
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Full-length (actual, lower, upper, valid mask, excluded count)."""
    n = int(y.size)
    lo_raw = np.asarray(b.lower, dtype=float).reshape(-1)
    hi_raw = np.asarray(b.upper, dtype=float).reshape(-1)
    lo = np.full(n, np.nan, dtype=float)
    hi = np.full(n, np.nan, dtype=float)
    lo[: min(n, lo_raw.size)] = lo_raw[: min(n, lo_raw.size)]
    hi[: min(n, hi_raw.size)] = hi_raw[: min(n, hi_raw.size)]
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    valid = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    return y, lo, hi, valid, int((~valid).sum())


def pinball_from_bands(
    actual: Sequence[float], bands: Sequence[IntervalBand]
) -> dict[str, Any]:
    """Average pinball loss at every quantile implied by the two-sided bands."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    by_q: dict[str, float] = {}
    per_step: dict[str, list[float]] = {}
    n = int(y.size)
    n_excluded = 0
    for b in bands:
        c = float(b.coverage)
        qlo, qhi = (1.0 - c) / 2.0, (1.0 + c) / 2.0
        yy, ll, hh, valid, excluded = _band_arrays(y, b)
        n_excluded += excluded
        for q, endpoint in ((qlo, ll), (qhi, hh)):
            f = np.where(valid, endpoint, np.nan)
            val, per = _pinball(y, f, q)
            key = f"{q:.3f}"
            if math.isfinite(val):
                by_q[key] = val
            per_step[key] = [float(v) if math.isfinite(v) else float("nan") for v in per]
    vals = [v for v in by_q.values() if math.isfinite(v)]
    return {
        "by_quantile": by_q,
        "per_step": per_step,
        "mean": float(np.mean(vals)) if vals else None,
        "n_quantiles": len(vals),
        "n_expected": n,
        "n_excluded": n_excluded,
        "warnings": ([f"{n_excluded} band endpoint(s) were missing or invalid and excluded "
                      "from pinball loss; rows were not silently truncated."] if n_excluded else []),
        "citation": CRPS_CITATION,
        "note": "Each two-sided band contributes its lower and upper quantile; the mean is over "
                "the reported quantiles, not over coverage levels. Full-length alignment is enforced.",
    }


def log_score_normal(
    actual: Sequence[float], point: Sequence[float], sigma: Sequence[float]
) -> dict[str, Any]:
    """−mean(log p(y)) under a normal predictive density from the model's own sd."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    m = np.asarray(point, dtype=float).reshape(-1)
    s = np.asarray(sigma, dtype=float).reshape(-1)
    n = min(y.size, m.size, s.size)
    mask = np.isfinite(y[:n]) & np.isfinite(m[:n]) & np.isfinite(s[:n]) & (s[:n] > 0)
    if not mask.any():
        return {"value": None, "n": 0, "note": "No finite positive predictive sd was available."}
    lp = stats.norm.logpdf(y[:n][mask], loc=m[:n][mask], scale=s[:n][mask])
    return {
        "value": float(-np.mean(lp)),
        "n": int(mask.sum()),
        "formula": "log score = -mean(log p(y))",
        "note": "p(y) is a DENSITY, not a probability: the score can be negative and its units "
                "depend on the scale. The normal sd comes from the model's own 95% analytic "
                "band, so this is a stated approximation, not a full density forecast. For a "
                "count PMF use discrete.log_score instead.",
    }


def sigma_from_bands(bands: Sequence[IntervalBand], n: int | None = None) -> np.ndarray | None:
    """Normal predictive sd implied by the 95% analytic band: width / (2 z_0.975)."""
    for b in bands:
        if abs(float(b.coverage) - 0.95) < 1e-9:
            width = np.maximum(np.asarray(b.upper, float) - np.asarray(b.lower, float), 0.0)
            out = width / (2.0 * _Z975)
            if n is not None and out.size < n:
                out = np.pad(out, (0, n - out.size), constant_values=np.nan)
            return out[:n] if n is not None else out
    return None


def winkler_scores(actual: Sequence[float], bands: Sequence[IntervalBand]) -> dict[str, Any]:
    """Winkler / interval score for each stated coverage, with its alpha."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    by_cov: dict[str, float] = {}
    per_step: dict[str, list[float]] = {}
    n_excluded = 0
    for b in bands:
        cov = float(b.coverage)
        alpha = max(1.0 - cov, 1e-9)
        yy, ll, hh, valid, excluded = _band_arrays(y, b)
        n_excluded += excluded
        score = np.full(y.size, np.nan, dtype=float)
        with np.errstate(all="ignore"):
            inside = (
                (hh - ll)
                + (2.0 / alpha) * (ll - yy) * (yy < ll)
                + (2.0 / alpha) * (yy - hh) * (yy > hh)
            )
        score[valid] = inside[valid]
        key = f"{cov:.0%}"
        finite = score[np.isfinite(score)]
        if finite.size:
            by_cov[key] = float(finite.mean())
        per_step[key] = [float(v) if math.isfinite(v) else float("nan") for v in score]
    vals = [v for v in by_cov.values() if math.isfinite(v)]
    return {
        "by_coverage": by_cov,
        "per_step": per_step,
        "mean": float(np.mean(vals)) if vals else None,
        "n_expected": int(y.size),
        "n_excluded": n_excluded,
        "warnings": ([f"{n_excluded} interval row(s) were missing or invalid and excluded; "
                      "arrays were not silently truncated."] if n_excluded else []),
        "note": "Winkler = interval width + (2/alpha)·distance outside on either side; lower is "
                "better. Coverage and construction are stated for every band (§6.8).",
    }


def coverage_curve(actual: Sequence[float], bands: Sequence[IntervalBand]) -> dict[str, Any]:
    """Empirical vs nominal coverage, binomial test, and average width."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    rows: list[dict[str, Any]] = []
    for b in sorted(bands, key=lambda x: float(x.coverage)):
        cov = float(b.coverage)
        yy, ll, hh, valid, excluded = _band_arrays(y, b)
        inside = ((yy[valid] >= ll[valid]) & (yy[valid] <= hh[valid])).astype(bool)
        test = coverage_tests(inside, nominal=cov)
        rows.append({
            "coverage": cov,
            "coverage_pct": int(round(cov * 100)),
            "nominal": cov,
            "empirical": test.get("coverage"),
            "covered": test.get("covered"),
            "n": int(inside.size),
            "n_excluded": excluded,
            "p": test.get("p"),
            "ci95": test.get("ci95"),
            "mean_width": float(np.mean(hh[valid] - ll[valid])) if valid.any() else None,
            "construction": b.construction,
            "source": b.source,
            "note": test.get("note", "") + (f" {excluded} missing/invalid band row(s) excluded."
                                            if excluded else ""),
        })
    return {
        "rows": rows,
        "philosophy": "Sharpness subject to calibration: a narrower interval is better only after "
                      "its empirical coverage is consistent with the nominal level.",
        "citation": "Gneiting & Raftery (2007); Christoffersen (1998) for calibration testing",
    }


def empirical_pit(
    actual: Sequence[float], paths: np.ndarray, *, reps: int = 5, seed: int = 20260101
):
    """Randomized PIT from an empirical ensemble; finite draws only, one inferential draw."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    p = np.asarray(paths, dtype=float)
    if p.ndim != 2 or p.shape[1] != y.size:
        raise ValueError("The ensemble must have one column per observation.")
    finite = np.isfinite(p)
    n_eff = finite.sum(axis=0)
    lower = np.zeros(y.size, dtype=float)
    upper = np.ones(y.size, dtype=float)
    with np.errstate(all="ignore"):
        for j in range(y.size):
            if not math.isfinite(y[j]) or n_eff[j] == 0:
                continue
            col = p[:, j][finite[:, j]]
            lo = float((col < y[j]).sum()) / float(n_eff[j])
            hi = float((col <= y[j]).sum()) / float(n_eff[j])
            if hi <= lo:
                hi = min(lo + 1.0 / float(n_eff[j]), 1.0)
            lower[j], upper[j] = lo, hi
    res = pit_from_quantiles(
        y, lower_cdf=lower, cdf=upper, kind="discrete", reps=reps, seed=seed
    )
    res.kind = "empirical_ensemble"
    res.caveat = (
        "PIT is drawn inside the empirical CDF step from the per-column FINITE seeded "
        f"predictive draws, so a correct model with tied draws stays uniform. One draw is "
        f"tested; {reps} seeded draws quantify stability, not independent tests."
    )
    return res


def rqr_from_paths(
    actual: Sequence[float], paths: np.ndarray, *, reps: int = 5, seed: int = 20260101
) -> dict[str, Any]:
    """Dunn–Smyth-style randomized quantile residuals from predictive draws."""
    out = rqr_from_draws(actual, np.asarray(paths, dtype=float), reps=reps, seed=seed).as_dict()
    # The per-repetition KS p-values come from dependent randomizations of the
    # SAME data. Fisher's method would be invalid; use the first seeded draw.
    out["combined_p"] = out.get("ks_p")
    out["note"] = (
        "combined_p is the KS p-value of one seeded randomization, not a Fisher combination "
        "of dependent repetition p-values."
    )
    return out


@dataclass
class ProbabilisticScores:
    crps_mean: float | None = None
    crps_by_step: list[float] = field(default_factory=list)
    pinball: dict[str, Any] = field(default_factory=dict)
    log_score: dict[str, Any] = field(default_factory=dict)
    winkler: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    pit: dict[str, Any] = field(default_factory=dict)
    rqr: dict[str, Any] = field(default_factory=dict)
    interval_method: str = "analytic"
    construction: str = ""
    n_paths: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "crps": self.crps_mean,
            "crps_by_step": list(self.crps_by_step),
            "pinball": self.pinball,
            "log_score": self.log_score,
            "winkler": self.winkler,
            "calibration": self.calibration,
            "pit": self.pit,
            "rqr": self.rqr,
            "interval_method": self.interval_method,
            "construction": self.construction,
            "n_paths": self.n_paths,
            "warnings": list(self.warnings),
            "citation": CRPS_CITATION,
        }


def probabilistic_scores(
    *,
    actual: Sequence[float],
    point: Sequence[float],
    bands: Sequence[IntervalBand],
    paths: np.ndarray | None,
    interval_method: str,
    construction: str,
    n_paths: int,
    reps: int,
    seed: int,
) -> ProbabilisticScores:
    y = np.asarray(actual, dtype=float).reshape(-1)
    out = ProbabilisticScores(
        interval_method=interval_method, construction=construction, n_paths=int(n_paths)
    )
    if paths is not None and np.asarray(paths).ndim == 2 and np.asarray(paths).shape[1] == y.size:
        by_step, mean = crps_ensemble(np.asarray(paths, dtype=float), y)
        out.crps_by_step = [float(v) if math.isfinite(v) else float("nan") for v in by_step]
        out.crps_mean = float(mean) if math.isfinite(mean) else None
        try:
            out.pit = empirical_pit(y, np.asarray(paths, dtype=float), reps=reps, seed=seed).as_dict()
            out.rqr = rqr_from_paths(y, np.asarray(paths, dtype=float), reps=reps, seed=seed)
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            out.warnings.append(f"PIT / RQR were refused: {type(exc).__name__}: {exc}")
            out.pit = {"ok": False, "note": f"PIT refused: {exc}"}
            out.rqr = {"ok": False, "note": f"RQR refused: {exc}"}
    else:
        out.warnings.append(
            "CRPS, PIT and RQR need a seeded predictive ensemble aligned with the evaluation "
            "observations; no ensemble was available for this model."
        )
        out.pit = {"ok": False, "note": "No aligned predictive ensemble."}
        out.rqr = {"ok": False, "note": "No aligned predictive ensemble."}
    sigma = sigma_from_bands(bands, n=y.size)
    out.log_score = (
        log_score_normal(y, point, sigma)
        if sigma is not None else
        {"value": None, "n": 0, "note": "No 95% analytic band was available to estimate a density."}
    )
    out.pinball = pinball_from_bands(y, bands)
    out.winkler = winkler_scores(y, bands)
    out.calibration = coverage_curve(y, bands)
    for block in (out.pinball, out.winkler):
        out.warnings.extend(block.get("warnings") or [])
    return out


__all__ = [
    "CRPS_CITATION", "ProbabilisticScores", "coverage_curve", "crps_ensemble",
    "empirical_pit", "log_score_normal", "pinball_from_bands", "probabilistic_scores",
    "rqr_from_paths", "sigma_from_bands", "winkler_scores",
]
