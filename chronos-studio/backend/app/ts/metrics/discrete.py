"""
ts/metrics/discrete.py — count forecasts are scored on the PMF (§5.12).

CRPS on an arbitrary discrete support is the energy identity

    CRPS(F, y) = E|X − y| − ½ E|X − X'|,  X, X' ~ F independently,

evaluated in O(k log k) from sorted support points and prefix sums. The old
ranked-probability-sum shortcut was only valid for unit-spaced grids and added
an incorrect distance penalty outside the support; both are fixed here.

The log score is `−log P(Y = y)`. An observation outside the stored PMF support
has probability zero, so the log score is refused for the whole model rather
than replaced by a finite "edge penalty". PIT is randomized inside
`F(y−1) … F(y)` because otherwise even a perfect Poisson model looks
non-uniform. Dunn–Smyth residuals are displayed/tested from ONE seeded
randomization; extra repetitions quantify Monte Carlo stability, not
independent p-values (Fisher-combining them was invalid).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

from app.ts.diagnose.calibration import pit_from_quantiles

_PIT_CITATION = "Dunn & Smyth (1996); Czado, Gneiting & Held (2009)"


def _normalise(grid: np.ndarray, pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Finite, sorted, duplicate-merged, non-negative and L1-normalised support."""
    g = np.asarray(grid, dtype=float).reshape(-1)
    p = np.asarray(pmf, dtype=float).reshape(-1)
    n = min(g.size, p.size)
    g, p = g[:n], p[:n]
    ok = np.isfinite(g) & np.isfinite(p)
    g, p = g[ok], np.maximum(p[ok], 0.0)
    if g.size == 0:
        return g, p
    order = np.argsort(g)
    g, p = g[order], p[order]
    unique, inverse = np.unique(g, return_inverse=True)
    merged = np.zeros(unique.size, dtype=float)
    np.add.at(merged, inverse, p)
    total = float(merged.sum())
    return unique, (merged / total if total > 0 else merged)


def discrete_crps(grid: Sequence[float], pmf: Sequence[float], observed: float) -> float:
    """Exact CRPS for an arbitrary finite discrete distribution and any real y."""
    if not math.isfinite(observed):
        return float("nan")
    g, p = _normalise(np.asarray(grid, dtype=float), np.asarray(pmf, dtype=float))
    if g.size == 0 or p.sum() <= 0:
        return float("nan")
    mu = float(np.dot(p, np.abs(g - float(observed))))
    cum_p = np.concatenate([[0.0], np.cumsum(p)[:-1]])
    cum_px = np.concatenate([[0.0], np.cumsum(p * g)[:-1]])
    pair_half = float(np.dot(p, g * cum_p - cum_px))
    value = mu - pair_half
    return float(max(0.0, value)) if math.isfinite(value) else float("nan")


def discrete_log_score(grid: Sequence[float], pmf: Sequence[float], observed: float) -> float:
    """−log P(Y = y); NaN when y is non-integer or outside the stored support."""
    if not math.isfinite(observed):
        return float("nan")
    g, p = _normalise(np.asarray(grid, dtype=float), np.asarray(pmf, dtype=float))
    if g.size == 0:
        return float("nan")
    integer = abs(observed - round(observed)) < 1e-9
    if not integer:
        return float("nan")
    pos = np.flatnonzero(np.isclose(g, float(observed), rtol=0.0, atol=1e-12))
    if pos.size == 0:
        return float("nan")
    prob = float(p[pos[0]])
    if prob <= 0:
        return float("nan")
    return float(-math.log(prob))


def _cdf_bounds(
    grids: Sequence[Sequence[float]], pmfs: Sequence[Sequence[float]], observed: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(observed, dtype=float).reshape(-1)
    lower = np.zeros(y.size, dtype=float)
    upper = np.ones(y.size, dtype=float)
    for j in range(y.size):
        g, p = _normalise(np.asarray(grids[j]), np.asarray(pmfs[j]))
        if g.size == 0 or p.sum() <= 0:
            continue  # missing PMF: a full [0, 1] draw, explicitly not an atom
        if not math.isfinite(y[j]):
            lower[j], upper[j] = 0.0, 1.0
            continue
        cdf = np.cumsum(p)
        below = float(np.sum(p[g < y[j]]))
        at = float(np.sum(p[g == y[j]]))
        lower[j] = min(max(below, 0.0), 1.0)
        upper[j] = min(max(below + at, lower[j] + 1e-9), 1.0)
    return lower, upper


def randomized_pit(
    grids: Sequence[Sequence[float]],
    pmfs: Sequence[Sequence[float]],
    observed: Sequence[float],
    *,
    reps: int = 5,
    seed: int = 20260101,
):
    """PIT histogram, KS and Cramér–von Mises tests from the exact PMF atoms."""
    y = np.asarray(observed, dtype=float).reshape(-1)
    lower, upper = _cdf_bounds(grids, pmfs, y)
    res = pit_from_quantiles(
        y, lower_cdf=lower, cdf=upper, kind="discrete", reps=reps, seed=seed
    )
    res.caveat = (
        "Discrete PIT is drawn uniformly inside F(y−1) … F(y) from the exact PMF; without "
        "it a correct count model would pile up on atoms and the uniformity test would reject. "
        f"One seeded draw is plotted and tested; {reps} seeded draws quantify stability, not "
        "independent tests."
    )
    return res


@dataclass
class RQRDiagnostic:
    ok: bool = False
    values: list[float] = field(default_factory=list)
    qq_theoretical: list[float] = field(default_factory=list)
    qq_sample: list[float] = field(default_factory=list)
    ks_stat: float | None = None
    ks_p: float | None = None
    combined_p: float | None = None
    reps: int = 0
    per_rep_ks_p: list[float] = field(default_factory=list)
    stability: str = ""
    caveat: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "values": self.values,
            "qq_theoretical": self.qq_theoretical, "qq_sample": self.qq_sample,
            "ks_stat": self.ks_stat, "ks_p": self.ks_p,
            "combined_p": self.combined_p, "reps": self.reps,
            "per_rep_ks_p": self.per_rep_ks_p, "stability": self.stability,
            "caveat": self.caveat, "note": self.note,
        }


def dunn_smyth_rqr(
    grids: Sequence[Sequence[float]],
    pmfs: Sequence[Sequence[float]],
    observed: Sequence[float],
    *,
    reps: int = 5,
    seed: int = 20260101,
) -> RQRDiagnostic:
    """Averaged QQ display is replaced by ONE inferential seeded randomization."""
    y = np.asarray(observed, dtype=float).reshape(-1)
    lower, upper = _cdf_bounds(grids, pmfs, y)
    if y.size < 10:
        return RQRDiagnostic(note=f"Only {y.size} observations; RQR needs at least 10.")
    rng = np.random.default_rng(seed)
    per_rep: list[np.ndarray] = []
    ps: list[float] = []
    for _ in range(max(1, reps)):
        u = np.clip(lower + rng.random(y.size) * (upper - lower), 1e-6, 1 - 1e-6)
        z = stats.norm.ppf(u)
        per_rep.append(z)
        ps.append(float(stats.kstest(z, "norm").pvalue))
    z = per_rep[0]
    order = np.argsort(z)
    zs = z[order]
    n = zs.size
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    ks = stats.kstest(z, "norm")
    agree = max(ps) - min(ps) < 0.15 or (min(ps) >= 0.05 and max(ps) >= 0.05) or max(ps) < 0.05
    stability = (
        f"KS p-values across {len(per_rep)} randomizations stayed in "
        f"[{min(ps):.3f}, {max(ps):.3f}]; the verdict does not depend on the draw."
        if agree else
        f"KS p-values moved from {min(ps):.3f} to {max(ps):.3f} across {len(per_rep)} "
        "randomizations; the verdict is inconclusive at this sample size."
    )
    return RQRDiagnostic(
        ok=True,
        values=[round(float(v), 3) for v in z],
        qq_theoretical=[round(float(v), 3) for v in theo],
        qq_sample=[round(float(v), 3) for v in zs],
        ks_stat=round(float(ks.statistic), 4),
        ks_p=round(float(ks.pvalue), 4),
        combined_p=round(float(ps[0]), 4),
        reps=len(per_rep),
        per_rep_ks_p=[round(p, 4) for p in ps],
        stability=stability,
        caveat=(
            "Dunn–Smyth residuals are randomized by construction: each residual is a normal "
            "quantile of a uniform draw inside the PMF atom. ONE seeded draw is plotted and "
            "tested; further draws measure Monte Carlo stability. Averaging randomized "
            "residuals shrinks their variance and would invalidate the KS test."
        ),
        note=(
            "combined_p is the KS p-value of the first seeded randomization, NOT a Fisher "
            "combination of dependent per-repetition p-values."
        ),
    )


@dataclass
class DiscreteScores:
    n: int = 0
    crps: float | None = None
    crps_by_step: list[float] = field(default_factory=list)
    log_score: float | None = None
    log_score_by_step: list[float] = field(default_factory=list)
    pit: dict[str, Any] = field(default_factory=dict)
    rqr: dict[str, Any] = field(default_factory=dict)
    truncated_steps: int = 0
    n_out_of_support: int = 0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "crps": self.crps,
            "crps_by_step": list(self.crps_by_step),
            "log_score": self.log_score,
            "log_score_by_step": list(self.log_score_by_step),
            "pit": self.pit, "rqr": self.rqr,
            "truncated_steps": self.truncated_steps,
            "n_out_of_support": self.n_out_of_support,
            "warnings": list(self.warnings), "notes": list(self.notes),
            "citation": _PIT_CITATION,
        }


def discrete_scores(
    *,
    grids: Sequence[Sequence[float]],
    pmfs: Sequence[Sequence[float]],
    observed: Sequence[float],
    truncated: Sequence[bool] = (),
    reps: int = 5,
    seed: int = 20260101,
) -> DiscreteScores:
    y = np.asarray(observed, dtype=float).reshape(-1)
    if len(grids) != y.size or len(pmfs) != y.size:
        raise ValueError("One PMF support and one probability vector are needed per observation.")
    crps_steps = [discrete_crps(grids[j], pmfs[j], y[j]) for j in range(y.size)]
    log_steps = [discrete_log_score(grids[j], pmfs[j], y[j]) for j in range(y.size)]
    out_of_support = 0
    for j in range(y.size):
        if not math.isfinite(y[j]):
            continue
        g, p = _normalise(np.asarray(grids[j]), np.asarray(pmfs[j]))
        if g.size == 0 or p.sum() <= 0 or not np.any(np.isclose(g, float(y[j]), rtol=0.0, atol=1e-12)):
            out_of_support += 1
    out = DiscreteScores(
        n=int(y.size),
        crps_by_step=[float(v) if math.isfinite(v) else float("nan") for v in crps_steps],
        log_score_by_step=[float(v) if math.isfinite(v) else float("nan") for v in log_steps],
        truncated_steps=int(np.sum(np.asarray(truncated, dtype=bool))) if truncated else 0,
        n_out_of_support=int(out_of_support),
    )
    finite_crps = [v for v in crps_steps if math.isfinite(v)]
    finite_log = [v for v in log_steps if math.isfinite(v)]
    out.crps = float(np.mean(finite_crps)) if finite_crps else None
    if out_of_support:
        out.log_score = None
        out.warnings.append(
            f"{out_of_support} observed count(s) fell outside the stored PMF support; their "
            "predictive probability is zero, so the mean log score is refused rather than "
            "replaced by a finite edge penalty."
        )
    else:
        out.log_score = float(np.mean(finite_log)) if finite_log else None
    out.pit = randomized_pit(grids, pmfs, y, reps=reps, seed=seed).as_dict()
    out.rqr = dunn_smyth_rqr(grids, pmfs, y, reps=reps, seed=seed).as_dict()
    out.notes.extend([
        "Discrete CRPS uses the energy identity E|X−y| − ½E|X−X'| on the stored support, "
        "valid for irregular grids and observations beyond the support.",
        "Log score uses P(Y = y) from the PMF, not a density; non-integer observations and "
        "out-of-support counts are refused.",
        "Calibration uses the seeded randomized discrete PIT and Dunn–Smyth residuals; "
        "repetitions expose randomization variability without being treated as independent tests.",
        "MAPE and sMAPE are banned for this branch by policy.py.",
    ])
    if out.truncated_steps:
        out.warnings.append(
            f"{out.truncated_steps} PMF support(s) were truncated; tail probabilities are "
            "represented only up to the stored support and are reported as conservative."
        )
    return out


__all__ = [
    "DiscreteScores", "RQRDiagnostic", "discrete_crps", "discrete_log_score",
    "discrete_scores", "dunn_smyth_rqr", "randomized_pit",
]
