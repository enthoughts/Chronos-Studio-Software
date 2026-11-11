"""
ts/forecast/core.py — the ForecastBundle, the one object S13 ships (§4.3, §5.11).

A forecast is not three arrays. It is three arrays **plus** the disclosure that makes
them usable: coverage and construction on every band, the interval method that was
selected and why, the seeded simulation's path count and seed, the back-transformation
choice and formula, the S12 residual-adequacy verdict, the horizon/season warnings,
the scenario the exogenous path assumed, and the training window (with zero test
observations). §6.8–11 are enforced by putting these fields in the bundle and
making `as_dict()` the only serialiser the API and UI are allowed to use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from app.ts.forecast.backtransform import (
    BackTransformInfo,
    TransformSpec,
    apply_back_transform,
)
from app.ts.forecast.horizons import COVERAGES
from app.ts.forecast.intervals import (
    IntervalBand,
    SimulationDraws,
    analytic_bands,
    simulate_bands,
)

IntervalMethod = Literal["analytic", "simulation", "auto"]


@dataclass
class ForecastCore:
    """Model-level numbers, before scale/adequacy/disclosure are attached."""

    point: np.ndarray
    analytic_bands: list[IntervalBand]
    simulation_bands: list[IntervalBand] | None
    paths: np.ndarray | None
    simulation_info: dict[str, Any] | None
    discrete_info: dict[str, Any] | None
    construction: str
    is_discrete: bool = False


@dataclass
class ForecastBundle:
    model_id: str
    display_name: str
    horizon: int
    index: list[str]
    point: np.ndarray
    bands: list[IntervalBand]                 # the selected construction, every coverage
    analytic_bands: list[IntervalBand]         # always reported alongside
    simulation_bands: list[IntervalBand] | None
    interval_method: Literal["analytic", "simulation"]
    coverages: list[float]
    simulation: dict[str, Any] | None
    discrete: dict[str, Any] | None
    transform: BackTransformInfo
    adequacy: dict[str, Any]
    horizon_advice: dict[str, Any]
    scenario: dict[str, Any]
    combination: dict[str, Any] | None
    inputs: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    construction: str = ""
    seed: int = 0
    n_paths: int = 0
    #: Kept in memory (not on the wire) for Part 15's CRPS/pinball/interval scores.
    paths: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "horizon": self.horizon,
            "index": list(self.index),
            "point": [float(v) for v in self.point],
            "bands": [b.as_dict() for b in self.bands],
            "analytic_bands": [b.as_dict() for b in self.analytic_bands],
            "simulation_bands": [b.as_dict() for b in (self.simulation_bands or [])],
            "interval_method": self.interval_method,
            "coverages": list(self.coverages),
            "simulation": self.simulation,
            "discrete": self.discrete,
            "transform": self.transform.as_dict(),
            "adequacy": dict(self.adequacy),
            "horizon_advice": dict(self.horizon_advice),
            "scenario": dict(self.scenario),
            "combination": self.combination,
            "inputs": dict(self.inputs),
            "warnings": list(self.warnings),
            "caveats": list(self.caveats),
            "notes": list(self.notes),
            "construction": self.construction,
            "seed": self.seed,
            "n_paths": self.n_paths,
            "scale": self.transform.scale,
            "intervals_stated": all(bool(b.construction) and b.coverage for b in self.bands),
            "test_observations_used": 0,
        }


def single_core(
    wrapper: Any,
    *,
    horizon: int,
    coverages: Sequence[float] = COVERAGES,
    draws: SimulationDraws,
    interval_method: IntervalMethod = "auto",
    non_gaussian: bool = False,
    detail: str = "",
) -> tuple[ForecastCore, list[str]]:
    """
    Analytic intervals (repeated `predict`) + seeded simulation for ONE stored fit.
    Returns the core and any warnings the simulation itself raised.
    """
    levels = sorted({float(c) for c in coverages})
    warnings: list[str] = []

    # A pre-call establishes whether this wrapper owns a discrete predictive PMF.
    # `predictive()` is only populated by `_forecast`, so asking before the first
    # predict would always answer "no" for a count model.
    wrapper.predict(int(horizon), alpha=0.05)
    predictor = getattr(wrapper, "predictive", None)
    pred = predictor() if callable(predictor) else None
    is_discrete = pred is not None

    point, analytic = analytic_bands(
        wrapper.predict, horizon, levels, discrete=is_discrete, detail=detail
    )

    # Refresh the PMF at the same horizon so the simulation samples the exact
    # distribution that produced these bands (not one from an earlier request).
    pred = None
    if is_discrete:
        wrapper.predict(int(horizon), alpha=0.05)
        pred = predictor() if callable(predictor) else None

    resid = None
    residuals_fn = getattr(wrapper, "residuals", None)
    if callable(residuals_fn):
        try:
            resid = residuals_fn()
        except Exception:  # pragma: no cover - a wrapper may not have residuals
            resid = None

    ref95 = next((b for b in analytic if abs(b.coverage - 0.95) < 1e-9), None)
    sim = None
    sim_bands: list[IntervalBand] | None = None
    paths: np.ndarray | None = None
    sim_info: dict[str, Any] | None = None
    try:
        sim = simulate_bands(
            point=point,
            lower95=ref95.lower if ref95 else None,
            upper95=ref95.upper if ref95 else None,
            residuals=resid,
            draws=draws,
            coverages=levels,
            discrete_predictive=pred if is_discrete else None,
        )
        sim_bands, paths, sim_info = sim.bands, sim.paths, sim.info
    except (ValueError, FloatingPointError) as exc:
        warnings.append(
            f"Simulation could not be produced ({exc}); the analytic intervals are still shown "
            "and are labelled as such."
        )

    discrete_info: dict[str, Any] | None = None
    if pred is not None and hasattr(pred, "as_dict"):
        payload = pred.as_dict(max_steps=min(horizon, 12))
        payload["coverage"] = 0.95
        payload["interval_construction"] = (
            "95% two-sided prediction interval: exact integer quantiles of the fitted model's "
            "per-step PMF at alpha=0.05 (at least nominal mass, no interpolation between counts)."
        )
        payload["truncated_any"] = bool(np.any(np.asarray(pred.truncated, dtype=bool)))
        discrete_info = payload

    if interval_method == "auto":
        chosen: Literal["analytic", "simulation"] = (
            "simulation" if (non_gaussian or is_discrete) else "analytic"
        )
        auto_reason = (
            "simulation is the default for non-Gaussian / count / intermittent / discrete-PMF "
            "models"
            if chosen == "simulation" else
            "the model reports a symmetric analytic predictive interval"
        )
    else:
        chosen = "analytic" if sim_bands is None else interval_method  # type: ignore[assignment]
        auto_reason = "chosen explicitly by the caller"

    construction = (
        f"Point path from ModelWrapper.predict(h={horizon}); analytic intervals at "
        f"{', '.join(f'{c:.0%}' for c in levels)} from repeated predict(alpha=1-coverage); "
        + (
            f"seeded {sim_info['n_paths']:,}-path simulation (seed {sim_info['seed']}). "
            if sim_info else "simulation unavailable. "
        )
        + f"Primary construction: {chosen} ({auto_reason})."
    )
    core = ForecastCore(
        point=point,
        analytic_bands=analytic,
        simulation_bands=sim_bands,
        paths=paths,
        simulation_info=sim_info,
        discrete_info=discrete_info,
        construction=construction,
        is_discrete=is_discrete,
    )
    return core, warnings


def build_bundle(
    core: ForecastCore,
    *,
    model_id: str,
    display_name: str,
    horizon: int,
    index: Sequence[str],
    coverages: Sequence[float],
    interval_method: IntervalMethod,
    non_gaussian: bool,
    seed: int,
    n_paths: int,
    transform_spec: TransformSpec | None,
    back_transform: str | None,
    residuals: np.ndarray | None,
    adequacy: dict[str, Any],
    horizon_advice: dict[str, Any],
    scenario: dict[str, Any],
    warnings: Sequence[str],
    caveats: Sequence[str],
    inputs: dict[str, Any],
    combination: dict[str, Any] | None = None,
    transform_warnings: Sequence[str] = (),
) -> ForecastBundle:
    """
    Attach scale, honesty and disclosure to a ForecastCore. Everything that reaches
    the UI goes through `as_dict()`; there is no second ad-hoc JSON shape.
    """
    levels = sorted({float(c) for c in coverages})

    # Back-transform BOTH constructions in one pass so the 95% analytic band is
    # always present for the lognormal variance estimate (an empty simulation
    # band list would otherwise make that method fail for no good reason).
    n_analytic = len(core.analytic_bands)
    all_bands = list(core.analytic_bands) + list(core.simulation_bands or [])
    an_point, transformed, new_paths, bt = apply_back_transform(
        point=core.point,
        bands=all_bands,
        paths=core.paths,
        spec=transform_spec,
        method=back_transform,
        residuals=residuals,
    )
    an_bands = transformed[:n_analytic]
    sim_bands = transformed[n_analytic:]

    if interval_method == "auto":
        primary: Literal["analytic", "simulation"] = (
            "simulation" if (non_gaussian or core.is_discrete) and sim_bands else "analytic"
        )
    else:
        primary = "simulation" if interval_method == "simulation" and sim_bands else "analytic"
    selected = sim_bands if primary == "simulation" else an_bands

    all_warnings = [str(w) for w in warnings] + list(transform_warnings) + bt.notes
    if not sim_bands and interval_method == "simulation":
        all_warnings.append(
            "Simulation was requested but unavailable; analytic intervals were used instead and "
            "are labelled 'analytic'."
        )
    if adequacy.get("adequacy") in {"inadequate", "adequate_with_caveats"}:
        all_warnings.append(
            f"Residual adequacy verdict for this fit: {adequacy['adequacy']}. "
            f"{adequacy.get('headline', '')}".strip()
        )
    # De-duplicate, preserving order.
    all_warnings = list(dict.fromkeys(all_warnings))

    sim_payload: dict[str, Any] | None = None
    if core.simulation_info:
        sim_payload = {
            **core.simulation_info,
            "draws": {
                "seed": seed,
                "n_paths": n_paths,
                "shared_across_components": True,
            },
            "coverages": list(levels),
        }

    bundle = ForecastBundle(
        model_id=model_id,
        display_name=display_name,
        horizon=int(horizon),
        index=[str(x) for x in index],
        point=np.asarray(an_point, dtype=float),
        bands=selected,
        analytic_bands=an_bands,
        simulation_bands=sim_bands,
        interval_method=primary,
        coverages=list(levels),
        simulation=sim_payload,
        discrete=core.discrete_info,
        transform=bt,
        adequacy=dict(adequacy),
        horizon_advice=dict(horizon_advice),
        scenario=dict(scenario),
        combination=combination,
        inputs={
            **inputs,
            "test_observations_used": 0,
            "coverages": list(levels),
            "interval_method": primary,
            "n_paths": n_paths,
            "seed": seed,
            "non_gaussian": bool(non_gaussian),
        },
        warnings=all_warnings,
        caveats=list(caveats),
        notes=[
            core.construction,
            "Every band states its coverage and construction; no band is shown bare (§6.8).",
            "The S12 residual-adequacy verdict is attached to this forecast and travels with "
            "any export (§6.10).",
        ],
        construction=core.construction + (
            f" Back-transformation: {bt.method} ({bt.formula})." if bt.applied else
            " Scale: original units (no transform applied)."
        ),
        seed=int(seed),
        n_paths=int(n_paths),
        paths=new_paths,
    )
    return bundle


__all__ = ["ForecastBundle", "ForecastCore", "build_bundle", "single_core"]
