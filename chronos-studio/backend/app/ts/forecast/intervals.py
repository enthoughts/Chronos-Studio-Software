"""
ts/forecast/intervals.py — analytic intervals and seeded simulation (§5.11, §6.8).

Two constructions, both reported by name:

1. **Analytic.** The fitted wrapper already owns the model-specific formula — ETS/Theta
   use z * residual_sd * sqrt(h), ARIMA uses `get_forecast().conf_int(alpha)`, count
   models invert their PMF's CDF at alpha/2. The forecast layer does not re-derive any
   of them: it calls `ModelWrapper.predict(h, alpha=1-coverage)` once per requested
   coverage and labels what came back. Re-deriving them here is how a fan chart ends
   up with a Gaussian band around a negative-binomial forecast.

2. **Simulation.** Wrappers have no path simulator (§8's contract is `predict` only), so
   the seeded Monte Carlo lives here, not in `ts/models/`. For a continuous model it
   draws >=2,000 standardised *in-sample innovation* paths, scaled at each horizon by
   the model's own 95% analytic half-width (with a residual-sd * sqrt(h) fallback). For
   a discrete model it draws >=2,000 paths from the exact per-step PMF using a shared
   uniform per path. The coupling and the scale are stated in every band's
   `construction`; neither is passed off as a recursive model simulation.

Every band carries its coverage and construction. No band is ever returned bare (§6.8).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import stats

#: §5.11's floor. Enforced in `make_draws`, not merely documented.
MIN_PATHS = 2_000
MAX_PATHS = 20_000

_Z975 = float(stats.norm.ppf(0.975))


@dataclass
class IntervalBand:
    """One two-sided band. `construction` is mandatory and travels to the UI/export."""

    coverage: float
    lower: np.ndarray
    upper: np.ndarray
    construction: str
    source: Literal["analytic", "discrete_pmf", "simulation", "pooled_analytic", "pooled_simulation"]
    alpha: float = 0.05
    nesting_adjusted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "coverage": self.coverage,
            "coverage_pct": int(round(self.coverage * 100)),
            "alpha": round(1.0 - self.coverage, 6),
            "source": self.source,
            "construction": self.construction,
            "lower": [float(v) for v in self.lower],
            "upper": [float(v) for v in self.upper],
            "mean_width": float(np.mean(self.upper - self.lower)),
            "nesting_adjusted": self.nesting_adjusted,
        }


@dataclass
class SimulationDraws:
    """Shared random draws, so a combination pools *the same* randomness across models."""

    seed: int
    n_paths: int
    horizon: int
    normals: np.ndarray
    residual_index: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_paths": self.n_paths,
            "horizon": self.horizon,
            "shared_across_components": True,
        }


def make_draws(seed: int, n_paths: int, horizon: int, n_resid: int) -> SimulationDraws:
    """Create one seeded draw set. Refuses fewer than 2,000 paths (§5.11)."""
    n_paths, horizon = int(n_paths), int(horizon)
    if n_paths < MIN_PATHS:
        raise ValueError(
            f"Simulation needs at least {MIN_PATHS:,} seeded paths; {n_paths} were requested."
        )
    if n_paths > MAX_PATHS:
        raise ValueError(f"Simulation is capped at {MAX_PATHS:,} paths; {n_paths} were requested.")
    if horizon < 1:
        raise ValueError("A simulation needs at least one forecast step.")
    rng = np.random.default_rng(int(seed))
    return SimulationDraws(
        seed=int(seed),
        n_paths=n_paths,
        horizon=horizon,
        normals=rng.standard_normal((n_paths, horizon)),
        residual_index=rng.integers(0, max(int(n_resid), 1), size=(n_paths, horizon)),
    )


def _as_array(values: Any, horizon: int, what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != horizon:
        raise ValueError(
            f"The fitted model returned {arr.size} {what} values for a {horizon}-step forecast."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            f"The fitted model returned a non-finite {what} value; a forecast with NaNs "
            "cannot be shown or exported."
        )
    return arr


def _nest(bands: list[IntervalBand]) -> list[IntervalBand]:
    """
    Force the fan to be nested: a 95% band must contain the 90% band, which must
    contain the 80%, and so on. Some wrappers' approximations cross at long horizons;
    widening the inner bound of the outer band is the conservative, visible fix.
    """
    lo_acc: np.ndarray | None = None
    hi_acc: np.ndarray | None = None
    for b in bands:
        lo, hi = b.lower.copy(), b.upper.copy()
        if lo_acc is not None and hi_acc is not None:
            lo = np.minimum(lo, lo_acc)
            hi = np.maximum(hi, hi_acc)
        changed = not (np.array_equal(lo, b.lower) and np.array_equal(hi, b.upper))
        b.lower, b.upper, b.nesting_adjusted = lo, hi, changed
        lo_acc, hi_acc = lo, hi
    return bands


def analytic_bands(
    predict: Callable[..., pd.DataFrame],
    horizon: int,
    coverages: Sequence[float] = (0.50, 0.66, 0.80, 0.90, 0.95),
    *,
    discrete: bool = False,
    detail: str = "",
) -> tuple[np.ndarray, list[IntervalBand]]:
    """
    The model's own intervals at each coverage, obtained by repeated `predict(alpha)`.

    `point` is read from the first call. For a PMF model the same call that produces the
    band also refreshes the wrapper's cached predictive distribution; the caller reads
    it afterwards (always at 95%, because the loop is sorted ascending).
    """
    levels = sorted({float(c) for c in coverages})
    for c in levels:
        if not 0.0 < c < 1.0:
            raise ValueError(f"Coverage {c} is not a probability strictly between 0 and 1.")
    point: np.ndarray | None = None
    bands: list[IntervalBand] = []
    for c in levels:
        alpha = round(1.0 - c, 10)
        frame = predict(int(horizon), alpha=alpha)
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        mean = _as_array(frame["mean"], horizon, "mean")
        lo = _as_array(frame["lower"], horizon, "lower")
        hi = _as_array(frame["upper"], horizon, "upper")
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        if point is None:
            point = mean
        if discrete:
            construction = (
                f"Analytic {c:.0%} two-sided interval: exact integer quantiles of the fitted "
                f"model's discrete predictive PMF at alpha={alpha:g} (CDF inverted once per step; "
                "endpoints are attainable counts and the band contains at least the nominal mass)."
            )
            source: Literal["analytic", "discrete_pmf"] = "discrete_pmf"
        else:
            construction = (
                f"Analytic {c:.0%} two-sided interval from the fitted model's own "
                f"ModelWrapper.predict(h={horizon}, alpha={alpha:g}); the wrapper's predictive "
                "distribution supplies the multiplier (normal/t, state-space variance or PMF). "
                + detail
            ).strip()
            source = "analytic"
        bands.append(IntervalBand(c, lo, hi, construction, source, alpha))
    assert point is not None
    return point, _nest(bands)


@dataclass
class SimulationResult:
    bands: list[IntervalBand]
    paths: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"bands": [b.as_dict() for b in self.bands], **self.info}


def _bands_from_paths(
    paths: np.ndarray,
    coverages: Sequence[float],
    *,
    construction: str,
    source: Literal["simulation", "pooled_simulation"],
    discrete: bool,
) -> list[IntervalBand]:
    bands: list[IntervalBand] = []
    for c in sorted({float(x) for x in coverages}):
        qlo, qhi = (1.0 - c) / 2.0, (1.0 + c) / 2.0
        lo, hi = np.quantile(paths, [qlo, qhi], axis=0)
        if discrete:
            lo, hi = np.floor(lo), np.ceil(hi)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        bands.append(
            IntervalBand(
                c, lo, hi,
                f"Simulation {c:.0%} two-sided interval. {construction}",
                source, round(1.0 - c, 10),
            )
        )
    return _nest(bands)


def simulate_bands(
    *,
    point: np.ndarray,
    lower95: np.ndarray | None,
    upper95: np.ndarray | None,
    residuals: np.ndarray | None,
    draws: SimulationDraws,
    coverages: Sequence[float] = (0.50, 0.66, 0.80, 0.90, 0.95),
    distribution: Literal["bootstrap", "normal"] = "bootstrap",
    coupling: Literal["comonotonic", "independent"] = "comonotonic",
    discrete_predictive: Any | None = None,
) -> SimulationResult:
    """
    Seeded >=2,000-path Monte Carlo. Returns paths (for combinations / back-transforms)
    and the same five coverage bands as the analytic branch.

    The simulation is an honest *approximation* built from the fitted model's own
    outputs: the in-sample innovation distribution supplies the shape and the model's
    95% analytic interval supplies the horizon-dependent scale. It is not a recursive
    state-space simulation, and the construction string says so.
    """
    h = int(point.size)
    if draws.horizon != h:
        raise ValueError("The shared draws were made for a different horizon.")
    p = np.asarray(point, dtype=float)

    # ---- discrete predictive: sample the exact PMF, never a Gaussian band ------ #
    if discrete_predictive is not None:
        pmf = np.asarray(discrete_predictive.pmf, dtype=float)
        grid = np.asarray(discrete_predictive.grid, dtype=float)
        if pmf.ndim != 2 or pmf.shape[0] < h:
            raise ValueError("The discrete predictive PMF does not cover the requested horizon.")
        cdf = np.cumsum(pmf[:h], axis=1)
        cdf = cdf / np.maximum(cdf[:, -1:], 1e-300)
        # One uniform per path, shared across horizons: a comonotonic quantile coupling.
        u = stats.norm.cdf(draws.normals[:, 0])
        paths = np.empty((draws.n_paths, h), dtype=float)
        for j in range(h):
            idx = np.clip(np.searchsorted(cdf[j], u, side="left"), 0, grid.shape[1] - 1)
            paths[:, j] = grid[j, idx]
        construction = (
            f"Seeded Monte Carlo, {draws.n_paths:,} paths, seed={draws.seed}, inverse-CDF draws "
            "from the fitted model's exact per-step discrete predictive PMF; one shared uniform "
            "per path (comonotonic coupling across horizons, so paths are coherent but the "
            "cross-horizon dependence is assumed, not estimated). Endpoints are integer quantiles."
        )
        bands = _bands_from_paths(paths, coverages, construction=construction,
                                  source="simulation", discrete=True)
        return SimulationResult(
            bands, paths,
            {
                "n_paths": draws.n_paths,
                "seed": draws.seed,
                "distribution": "discrete_pmf",
                "coupling": "comonotonic",
                "construction": construction,
                "scale_source": "exact per-step PMF",
                "notes": [
                    "Interval endpoints are attainable integer counts, not interpolated positions "
                    "between impossible values.",
                    "PMF truncation, if any, is reported on the predictive payload and makes the "
                    "upper tail conservative.",
                ],
            },
        )

    # ---- continuous: standardised innovations × model's analytic scale -------- #
    r = np.asarray(residuals if residuals is not None else [], dtype=float).reshape(-1)
    r = r[np.isfinite(r)]
    sd = float(np.std(r, ddof=1)) if r.size >= 3 else 0.0
    if distribution == "bootstrap" and (r.size < 8 or not np.isfinite(sd) or sd <= 0):
        distribution = "normal"
        fallback_note = (
            "Fewer than eight usable in-sample residuals were available, so the innovation "
            "shape fell back to standard normal rather than pretending to bootstrap."
        )
    else:
        fallback_note = ""

    sigma: np.ndarray | None = None
    scale_source = ""
    if lower95 is not None and upper95 is not None:
        lo = np.asarray(lower95, dtype=float).reshape(-1)
        hi = np.asarray(upper95, dtype=float).reshape(-1)
        if lo.size == h and hi.size == h and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            width = np.maximum(hi - lo, 0.0)
            cand = width / (2.0 * _Z975)
            if np.any(cand > 0):
                sigma = np.where(cand > 0, cand, sd * np.sqrt(np.arange(1, h + 1)))
                scale_source = (
                    "each horizon's scale is the model's own 95% analytic half-width divided by "
                    f"z_0.975 = {_Z975:.4f} (with a residual-sd * sqrt(h) substitution at any "
                    "zero-width step)"
                )
    if sigma is None:
        sigma = np.full(h, sd) * np.sqrt(np.arange(1, h + 1, dtype=float))
        scale_source = (
            "the 95% analytic band was unavailable, so the scale is residual_sd * sqrt(h) — a "
            "random-walk approximation, disclosed rather than hidden"
        )

    if distribution == "bootstrap":
        z = (r - float(r.mean())) / sd
        eps = z[np.minimum(draws.residual_index, z.size - 1)]
        shape_note = (
            "innovations are drawn with replacement from the model's centred, standardised "
            "in-sample residuals (empirical shape, so skew/heavy tails survive)"
        )
    else:
        eps = draws.normals
        shape_note = "innovations are standard normal draws"

    if coupling == "comonotonic":
        eps = np.broadcast_to(eps[:, :1], (draws.n_paths, h))
        coupling_note = (
            "one draw per path is shared across horizons (comonotonic coupling): the marginal "
            "bands are correct, but the path-to-path dependence is an assumption"
        )
    else:
        coupling_note = "draws are independent across horizons and paths"

    paths = p[None, :] + eps * sigma[None, :]
    construction = (
        f"Seeded Monte Carlo, {draws.n_paths:,} paths, seed={draws.seed}. {shape_note}; "
        f"{scale_source}; {coupling_note}. This is a residual-based approximation built from "
        "the fitted model's predictive outputs, not a recursive simulation of the fitted states."
    )
    notes = [
        "Simulation is the default for non-Gaussian, count and intermittent data, where a "
        "symmetric analytic band is the wrong shape.",
        "No test-period observation is used: the innovation pool is in-sample residuals from "
        "the final fit only.",
    ]
    if fallback_note:
        notes.append(fallback_note)
    bands = _bands_from_paths(paths, coverages, construction=construction,
                              source="simulation", discrete=False)
    return SimulationResult(
        bands, paths,
        {
            "n_paths": draws.n_paths,
            "seed": draws.seed,
            "distribution": distribution,
            "coupling": coupling,
            "construction": construction,
            "scale_source": scale_source,
            "notes": notes,
        },
    )


def pooled_bands_from_paths(
    paths: np.ndarray,
    coverages: Sequence[float],
    *,
    construction: str,
    discrete: bool = False,
) -> list[IntervalBand]:
    """Quantiles of a *pooled* combination path array (see combine.py)."""
    return _bands_from_paths(paths, coverages, construction=construction,
                             source="pooled_simulation", discrete=discrete)


def half_width_from_sd(sd: float, horizon: int) -> np.ndarray:
    """Small helper for reconciliation: sigma_h = sd * sqrt(h)."""
    return float(max(sd, 0.0)) * np.sqrt(np.arange(1, int(horizon) + 1, dtype=float))


__all__ = [
    "MAX_PATHS",
    "MIN_PATHS",
    "IntervalBand",
    "SimulationDraws",
    "SimulationResult",
    "analytic_bands",
    "half_width_from_sd",
    "make_draws",
    "pooled_bands_from_paths",
    "simulate_bands",
]
