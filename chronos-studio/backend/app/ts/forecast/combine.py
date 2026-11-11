"""
ts/forecast/combine.py — mean / median / IC-weighted combination forecasts (§5.11).

"Simple mean / median / IC-weighted combination of the top models — often wins,
include it as a candidate."

Three rules keep the combination from being the row that quietly wins for the wrong
reason (the Part 13 combination fit was already excluded from CV for exactly this):

* **Weights are declared.** Equal, median (no weights), or Akaike weights from
  exp(-ΔAICc/2). AICc is an *in-sample* criterion: the payload says so, and says
  that rolling-origin CV in Step 11 remains the primary selection criterion.
* **Comparability is checked.** If any component's IC is flagged incomparable or
  missing, IC weighting refuses instead of quietly ranking likelihoods computed on
  different effective samples (§6.5).
* **Intervals come from pooled simulated paths**, not from averaging two numbers.
  Each component's seeded innovations share the same draw set, so the pooled paths
  are a genuine linear pool of random variables; the analytic bands are reported
  alongside as a separately labelled linear pool of the component intervals.

No test-period observation is read: the components are the full-series fits stored
by Step 10, and the top-model list comes from the stored Step 11 shortlist.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.ts.forecast.intervals import IntervalBand, pooled_bands_from_paths

METHODS: tuple[str, ...] = ("mean", "median", "ic_weighted")


@dataclass
class Component:
    model_id: str
    display_name: str
    point: np.ndarray
    bands: list[IntervalBand]
    paths: np.ndarray | None
    aicc: float | None = None
    ic_comparable: bool = True
    discrete: bool = False

    def band(self, coverage: float) -> IntervalBand | None:
        for b in self.bands:
            if abs(b.coverage - coverage) < 1e-9:
                return b
        return None


@dataclass
class CombinationResult:
    point: np.ndarray
    analytic_bands: list[IntervalBand]
    simulation_bands: list[IntervalBand]
    paths: np.ndarray
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_components": self.info.get("n_components"),
            "method": self.info.get("method"),
            "weights": self.info.get("weights"),
            "component_ids": self.info.get("component_ids", []),
            "n_paths": self.info.get("n_paths"),
            "coverages": self.info.get("coverages", []),
            "construction": self.info.get("construction"),
            "notes": self.info.get("notes", []),
        }


def ic_weights(components: Sequence[Component]) -> tuple[list[float], str, list[str]]:
    """Akaike weights from ΔAICc. Refuses when the ICs are not comparable."""
    bad = [
        c.model_id for c in components
        if c.aicc is None or not np.isfinite(float(c.aicc)) or not c.ic_comparable
    ]
    if bad:
        raise ValueError(
            "IC weighting needs a finite, comparable AICc for every component; "
            f"{', '.join(bad)} do(es) not have one (§6.5). Use 'mean' or 'median', or refit "
            "on a common effective sample."
        )
    vals = np.asarray([float(c.aicc) for c in components], dtype=float)
    delta = vals - float(vals.min())
    raw = np.exp(-0.5 * delta)
    w = raw / float(raw.sum())
    note = (
        "Akaike weights = exp(-ΔAICc/2), normalised. These are IN-SAMPLE information "
        "criteria, not held-out accuracy: rolling-origin CV (Step 11) remains the primary "
        "selection criterion."
    )
    return [float(x) for x in w], note, [f"delta_aicc={d:.4f}" for d in delta]


def combine_components(
    components: Sequence[Component],
    method: str = "mean",
    coverages: Sequence[float] = (0.50, 0.66, 0.80, 0.90, 0.95),
) -> CombinationResult:
    """Pool the component forecasts into one bundle-shaped forecast."""
    if method not in METHODS:
        raise ValueError(f"Unknown combination '{method}'; choose one of {', '.join(METHODS)}.")
    comps = list(components)
    if len(comps) < 2:
        raise ValueError("A combination needs at least two fitted component forecasts.")
    h = int(np.asarray(comps[0].point).size)
    levels = sorted({float(c) for c in coverages})
    for c in comps:
        if np.asarray(c.point).size != h:
            raise ValueError(f"{c.model_id} has a different horizon from the other components.")
        if c.paths is None or np.asarray(c.paths).shape[1] != h:
            raise ValueError(f"{c.model_id} has no simulated paths for this horizon.")
    n_paths = min(int(np.asarray(c.paths).shape[0]) for c in comps)

    notes: list[str] = [
        "Combination forecasts are built from the stored full-series fits; no test-period "
        "observation was used to choose or weight them.",
        "All components share one seeded draw set, so the pooled paths are a linear pool of "
        "random variables rather than an average of two unrelated intervals.",
    ]
    if method == "ic_weighted":
        weights, note, deltas = ic_weights(comps)
        notes.append(note)
        notes.append("ΔAICc: " + ", ".join(f"{c.model_id}={d}" for c, d in zip(comps, deltas)))
    elif method == "mean":
        weights = [1.0 / len(comps)] * len(comps)
        notes.append("Equal weights: no model is privileged by an in-sample criterion.")
    else:
        weights = [1.0 / len(comps)] * len(comps)
        notes.append("Median: a robust pool, insensitive to one component's extreme path.")

    points = np.vstack([np.asarray(c.point, float) for c in comps])
    paths_stack = np.stack([np.asarray(c.paths, float)[:n_paths] for c in comps], axis=0)

    if method == "median":
        point = np.median(points, axis=0)
        pooled_paths = np.median(paths_stack, axis=0)
        pool_phrase = "component-wise median"
    else:
        w = np.asarray(weights, dtype=float)
        point = (w[:, None] * points).sum(axis=0)
        pooled_paths = np.tensordot(w, paths_stack, axes=(0, 0))
        pool_phrase = "weighted linear pool with weights " + ", ".join(
            f"{c.model_id}={x:.4f}" for c, x in zip(comps, weights)
        )

    analytic: list[IntervalBand] = []
    for cov in levels:
        lows, highs, constructions = [], [], []
        for c in comps:
            band = c.band(cov)
            if band is None:
                raise ValueError(f"{c.model_id} has no {cov:.0%} band to pool.")
            lows.append(np.asarray(band.lower, float))
            highs.append(np.asarray(band.upper, float))
            constructions.append(band.construction)
        lo_arr, hi_arr = np.vstack(lows), np.vstack(highs)
        if method == "median":
            lo, hi = np.median(lo_arr, axis=0), np.median(hi_arr, axis=0)
        else:
            w = np.asarray(weights, float)
            lo, hi = (w[:, None] * lo_arr).sum(axis=0), (w[:, None] * hi_arr).sum(axis=0)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        analytic.append(
            IntervalBand(
                coverage=cov,
                lower=lo,
                upper=hi,
                construction=(
                    f"Analytic {cov:.0%} combination interval: a {pool_phrase} of each "
                    "component's own analytic interval endpoints (a linear pool of bounds, NOT a "
                    "mixture distribution). Each component's coverage and construction: "
                    + " | ".join(f"{c.model_id}: {s}" for c, s in zip(comps, constructions))
                ),
                source="pooled_analytic",
                alpha=round(1.0 - cov, 10),
            )
        )

    sim_construction = (
        f"Seeded Monte Carlo, {n_paths:,} paths; {pool_phrase} applied to every simulated path, "
        "then empirical quantiles of the pooled paths at each step."
    )
    simulation = pooled_bands_from_paths(
        pooled_paths, levels, construction=sim_construction,
        discrete=all(c.discrete for c in comps),
    )
    info = {
        "method": method,
        "n_components": len(comps),
        "weights": {c.model_id: float(w) for c, w in zip(comps, weights)},
        "component_ids": [c.model_id for c in comps],
        "n_paths": n_paths,
        "construction": sim_construction,
        "notes": notes,
        "coverages": levels,
    }
    return CombinationResult(point, analytic, simulation, pooled_paths, info)


__all__ = ["METHODS", "CombinationResult", "Component", "combine_components", "ic_weights"]
