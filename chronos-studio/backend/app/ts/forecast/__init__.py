"""
ts/forecast/ — Part 14: S13, forecasting (§5.11).

The forecast layer is deliberately *outside* `ts/models/`. A fitted wrapper answers
"what does this model say about step h?" (`ModelWrapper.predict(h, alpha)`); it does
not know about interval menus, seeded Monte Carlo, back-transformation, scenarios,
combinations or the honesty banner. Those are properties of a **reported forecast**,
not of a model, and keeping them here is what makes every one of the 48 registry
rows report the same five coverages with the same disclosure.

The six modules:

    horizons.py      H > n/5 and short-season advisor with an explicit acknowledgement
    intervals.py     analytic intervals (repeated predict) + seeded >=2,000-path simulation
    backtransform.py naive / smearing / lognormal back-transformation, never silent
    reconcile.py     optional aggregate <-> disaggregate reconciliation
    combine.py       mean / median / IC-weighted pools of top models
    core.py          the ForecastBundle: one JSON-shaped object every endpoint & UI reads

Nothing here reads a test-period observation. The bundle records `n_train`, the exact
window used, and `test_observations_used = 0` so the claim is checkable rather than
prose (§6.7).
"""

from app.ts.forecast.backtransform import (
    BackTransformInfo,
    TransformSpec,
    detect_transform,
    inverse_boxcox,
    inverse_yeojohnson,
)
from app.ts.forecast.combine import Component, combine_components, ic_weights
from app.ts.forecast.core import ForecastBundle, IntervalBand, build_bundle
from app.ts.forecast.horizons import (
    COVERAGES,
    HorizonAdvice,
    advise_horizon,
)
from app.ts.forecast.intervals import (
    MIN_PATHS,
    analytic_bands,
    make_draws,
    simulate_bands,
)
from app.ts.forecast.reconcile import (
    Reconciliation,
    reconcile,
)

__all__ = [
    "BackTransformInfo",
    "COVERAGES",
    "Component",
    "ForecastBundle",
    "HorizonAdvice",
    "IntervalBand",
    "MIN_PATHS",
    "Reconciliation",
    "TransformSpec",
    "advise_horizon",
    "analytic_bands",
    "build_bundle",
    "combine_components",
    "detect_transform",
    "ic_weights",
    "inverse_boxcox",
    "inverse_yeojohnson",
    "make_draws",
    "reconcile",
    "simulate_bands",
]
