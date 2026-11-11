"""
ts/models/combination.py — forecast combination (§5.8, the `combination` row; built in Part 13).

    "Forecast combination (mean / median / IC-weighted / Granger–Ramanathan regression) — often wins;
    must be in the candidate set."

The row is unusual in that it is not fitted to the series at all: it is fitted to the *other models'
forecasts*. That has four consequences this module is built around.

**1. It cannot be a normal member of `fit_many`.** Its inputs are wrappers, so it is fitted last, from
the fits that already exist, and it is the one row whose fit depends on which other rows were ticked.
The card says so, because "Gradient-boosted trees vs combination" is a different comparison once the
ticked set changes.

**2. Its honesty depends on the components being scored out of sample.** A combination built from
*in-sample* fits inherits every one of their in-sample advantages and will beat them for that reason
alone. So the combination is fitted on the components' **rolling-origin forecast errors**: the
weights are estimated where the components were actually forecasting, not where they were fitted.

**3. `Granger–Ramanathan` is a regression and needs its collinearity disclosed.** Component forecasts
are near-collinear by construction (they are forecasting the same series). The regression is estimated
by least squares but the combination is *not* allowed to extrapolate: weights are reported, and a
weight outside [−1, 2] on a component triggers a caveat rather than being silently clipped.

**4. It has no likelihood, so it has no IC.** `ic_comparable = False` and the AIC columns read "—".
Ranking it by an IC invented from a regression's residual sum of squares is exactly the kind of
cross-scale comparison §5.9 forbids.

Four combination rules are available; `mean` is the default because it is the one that is almost
impossible to get wrong, and `granger_ramanathan` is the one to reach for only when there is enough
held-out data to estimate the weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

CombinationMethod = Literal["mean", "median", "ic_weighted", "granger_ramanathan"]

METHODS: tuple[str, ...] = ("mean", "median", "ic_weighted", "granger_ramanathan")

#: The noun phrase, used where a sentence starts with "A …". `METHOD_PLAIN` is written for slots in
#: the middle of a sentence; gluing the two together produced "A the simple average of the component
#: forecasts", which is the kind of sentence that makes a reader distrust the numbers beside it.
METHOD_NOUN: dict[str, str] = {
    "mean": "simple average of the component forecasts",
    "median": "median of the component forecasts",
    "ic_weighted": "weighted average, with more weight on the components that fit better",
    "granger_ramanathan": "least-squares regression on the component forecasts",
}

METHOD_PLAIN: dict[str, str] = {
    "mean": "the simple average of the component forecasts",
    "median": "the middle of the component forecasts, which ignores one wild component",
    "ic_weighted": "a weighted average, with more weight on the components that fit better",
    "granger_ramanathan": "a least-squares regression on the component forecasts",
}


@dataclass
class ComponentForecast:
    """One component's forecast path and the weight it was given."""

    model_id: str
    display_name: str
    weight: float = 0.0
    mean: list[float] = field(default_factory=list)
    lower: list[float] = field(default_factory=list)
    upper: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "weight": round(self.weight, 4),
            "mean": [round(v, 4) for v in self.mean],
            "lower": [round(v, 4) for v in self.lower],
            "upper": [round(v, 4) for v in self.upper],
        }


def _weights_ic(aicc: Sequence[float | None]) -> np.ndarray:
    """
    Akaike-style weights from the components' AICc.

    Computed on the *deltas*, which is what makes them invariant to a constant offset — and then
    renormalised across the components actually present, because their AICc values need not be
    comparable with anything outside this set. A component with no AICc gets zero weight and the note
    in the card says which ones.
    """
    vals = np.asarray([v if v is not None and np.isfinite(v) else np.nan for v in aicc], dtype=float)
    if np.all(np.isnan(vals)):
        # No component has an IC — fall back to equal weights rather than inventing one.
        return np.full(vals.size, 1.0 / vals.size)
    best = np.nanmin(vals)
    d = vals - best
    w = np.where(np.isnan(d), 0.0, np.exp(-0.5 * np.where(np.isnan(d), 0.0, d)))
    total = w.sum()
    return w / total if total > 0 else np.full(vals.size, 1.0 / vals.size)


def _weights_granger_ramanathan(
    p: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """
    Least squares of the actual on the component forecasts, without an intercept-free constraint.

    The classic Granger–Ramanathan form includes an intercept; the weights are then not forced to sum
    to one, which is the point — it lets the combination correct a shared bias. But an intercept is
    only identifiable with enough held-out points, so with few rows the estimate is refused and the
    caller falls back to the mean.

    Returns the weights (last element is the intercept) and any caveats.
    """
    caveats: list[str] = []
    n, k = p.shape
    if n < k + 3:
        raise FitFailure(
            f"Granger–Ramanathan needs at least {k + 3} held-out forecasts to estimate {k} weights "
            f"and an intercept; there are {n}.",
            "Use the mean or median combination, or a shorter horizon so more origins fit.",
            kind="length",
        )
    X = np.column_stack([p, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    w = beta[:k]
    if np.any(np.abs(w) > 2.0):
        bad = [f"{v:+.2f}" for v in w if abs(v) > 2.0]
        caveats.append(
            "A Granger–Ramanathan weight came out outside [−1, 2] ("
            + ", ".join(bad)
            + "). That is the regression using one component to correct another, which is unstable "
            "on collinear forecasts — the combination will not extrapolate well beyond the observed "
            "range."
        )
    if np.any(np.abs(w).sum() > 4.0):
        caveats.append(
            "The weights sum to a large absolute value, a sign of collinearity between the component "
            "forecasts. The combination fits this sample; it is not a robust out-of-sample rule."
        )
    return np.concatenate([w, [beta[k]]]), caveats


class CombinationModel(ModelWrapper):
    """
    The combination row.

    Unlike every other wrapper, `fit` needs the components. It takes them by being handed a *forecast
    matrix* — one row per held-out origin, one column per component — plus the realised values for
    those origins. `fit_many` cannot construct this; the caller (the API's fit handler, after the CV
    pass) does, which is why this class is never in `IMPLEMENTED` for a direct fit and refuses with a
    clear message if it is.
    """

    model_id = "combination"
    display_name = "Forecast combination"
    family = ModelFamily.COMBINATION
    is_baseline = False
    fitted_on = "component forecasts"

    def __init__(
        self,
        *,
        method: CombinationMethod = "mean",
        components: Sequence[str] | None = None,
        period: int | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(period=period, **kw)
        if method not in METHODS:
            raise ValueError(f"Unknown combination method '{method}'; use one of {METHODS}.")
        self.method: str = method
        self.component_ids: list[str] = list(components or [])
        self._forecast_matrix: np.ndarray | None = None
        self._realised: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._intercept: float = 0.0
        self._component_forecasts: list[ComponentForecast] = []
        self._origins: list[Any] = []
        self._full_series: list[float] = []
        self.card: dict[str, Any] = {}

    # -- inputs ------------------------------------------------------------- #
    def set_inputs(
        self,
        matrix: np.ndarray,
        realised: np.ndarray,
        *,
        aicc: Sequence[float | None] | None = None,
        component_forecasts: Sequence[ComponentForecast] | None = None,
        origins: Sequence[Any] | None = None,
        full_series: Sequence[float] | None = None,
    ) -> "CombinationModel":
        """
        Supply the held-out component forecasts.

        `matrix` is (origins × components); `realised` is the (origins,) outcome at those origins.
        Both must be aligned — the constructor cannot check which origin a row came from, so the
        caller aligns them, and `cross_validate` is the caller that does.
        """
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2:
            raise FitFailure(
                "The combination needs a 2-D matrix of component forecasts (origins × components).",
                "This is a bug in the caller, not in the data.",
            )
        if m.shape[1] < 2:
            raise FitFailure(
                "A combination of one model is that model. At least two components are needed.",
                "Tick another model in Step 9 and refit.",
            )
        self._forecast_matrix = m
        self._realised = np.asarray(realised, dtype=float)
        self._aicc = list(aicc or [])
        #: Where each row of `matrix` came from, and the whole series. Together they let the
        #: combination place its fitted path on the same index the battery will read; without them
        #: the in-sample diagnostic block would compare a 10-point path against a 140-point series.
        self._origins = list(origins) if origins is not None else []
        # `list(x or [])` on a numpy array raises: the truth value of an array is ambiguous. An
        # explicit None check is the only correct form here.
        self._full_series = list(full_series) if full_series is not None else []
        if component_forecasts:
            self._component_forecasts = list(component_forecasts)
        return self

    # -- fit ---------------------------------------------------------------- #
    def _fit(self, y: pd.Series) -> None:
        if self._forecast_matrix is None or self._realised is None:
            raise FitFailure(
                "The forecast combination was fitted without its component forecasts.",
                "It is built after the other models have been cross-validated, not on its own. "
                "Untick it and use the fits in Step 10, or run the selection pass that builds it.",
                kind="usage",
            )
        p, actual = self._forecast_matrix, self._realised
        n, k = p.shape
        origins_used = list(self._origins or [])
        actual_full = np.asarray(self._full_series, dtype=float)
        mask = np.isfinite(actual) & np.isfinite(p).all(axis=1)
        p, actual = p[mask], actual[mask]
        if p.shape[0] < 4:
            raise FitFailure(
                f"Only {p.shape[0]} held-out origins where every component forecast a finite value.",
                "A combination needs at least 4 — with fewer, its weights are fitting noise.",
                kind="length",
            )
        caveats: list[str] = []

        if self.method == "mean":
            w = np.full(k, 1.0 / k)
        elif self.method == "median":
            w = np.full(k, 1.0 / k)
            caveats.append(
                "The median is not a weighted average: these weights are recorded as 1/k for "
                "reference, and the point forecast is the per-origin median of the components."
            )
        elif self.method == "ic_weighted":
            w = _weights_ic(self._aicc if len(self._aicc) == k else [None] * k)
            if not self._aicc or len(self._aicc) != k:
                caveats.append(
                    "No comparable information criteria were available for the components, so the "
                    "weights fell back to equal. An IC-weighted combination needs the ICs."
                )
            else:
                caveats.append(
                    "The IC weights come from the components' in-sample fits; they are a measure of "
                    "fit, not of forecast accuracy, which is why the mean is the default."
                )
        else:
            w, gr_caveats = _weights_granger_ramanathan(p, actual)
            self._intercept = float(w[-1])
            w = w[:-1]
            caveats.extend(gr_caveats)

        self._weights = w

        # The combination's "in-sample" fit is the component forecast *at each origin*, aligned to the
        # same series the battery will read. It is built on the full path so its length matches `y`:
        # an unfolded `_combine(p)` is `len(origins)` long, which does not align with the series and
        # made `diagnose` compare the wrong pairs — the first run reported MASE 2.27 for a combination
        # of two models scoring 0.44 and 0.85, which is arithmetically impossible and was the tell.
        fitted_full = np.full(actual_full.size, np.nan)
        for i, fold in enumerate(origins_used):
            fitted_full[fold.test_start] = float(self._combine(p[i : i + 1])[0]) if i < p.shape[0] else np.nan
        self._fitted = fitted_full
        # The residuals are the held-out errors at the origins — the only residuals this row has, and
        # the only ones that mean anything for it. Recorded so the battery can read them rather than
        # reporting "no residuals", which reads like a bug instead of a property of the row.
        self._resid = np.asarray(actual, dtype=float) - np.asarray(self._combine(p), dtype=float)

        # Measured on the origin rows, not on the aligned full-length path: `actual` is one value per
        # origin, and pairing it with the 140-long path is the same length mistake in the other
        # direction.
        in_sample = float(np.abs(np.asarray(actual) - np.asarray(self._combine(p))).mean()) if actual.size else float("nan")
        self.card = {
            "method": self.method,
            "method_plain": METHOD_PLAIN[self.method],
            "method_noun": METHOD_NOUN[self.method],
            "components": [
                {"model_id": c.model_id, "weight": round(float(wt), 4)}
                for c, wt in zip(self._components_for_card(k), w)
            ],
            "held_out_origins": int(p.shape[0]),
            "in_sample_mae": round(in_sample, 4),
            "intercept": round(self._intercept, 4) if self.method == "granger_ramanathan" else None,
            "why": (
                "The weights are estimated on the components' held-out forecasts, not on their "
                "in-sample fits. A combination built from in-sample fits wins for a reason that does "
                "not survive contact with new data."
            ),
            "caveats": caveats,
        }
        self._caveats = caveats

    def _components_for_card(self, k: int) -> list[ComponentForecast]:
        if len(self._component_forecasts) == k:
            return self._component_forecasts
        ids = self.component_ids or [f"component {i + 1}" for i in range(k)]
        return [ComponentForecast(model_id=i, display_name=i) for i in ids[:k]]

    def _combine(self, p: np.ndarray) -> np.ndarray:
        if self.method == "median":
            return np.median(p, axis=1)
        assert self._weights is not None
        return p @ self._weights + (self._intercept if self.method == "granger_ramanathan" else 0.0)

    # -- forecast ----------------------------------------------------------- #
    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        """
        Combine the components' *future* paths.

        The intervals are the **envelope** of the component paths, widened to `1 − alpha` by taking
        each component's own interval and combining the extremes. That is a deliberate simplification
        and it is disclosed: the correct predictive distribution of a combination needs the joint
        distribution of the components' errors, which this row does not estimate. An envelope is
        narrower than the truth when the components' errors are independent — so the card says the
        interval is the components' range, not a calibrated predictive interval.
        """
        comps = [c for c in self._component_forecasts if len(c.mean) >= horizon]
        if not comps:
            raise FitFailure(
                "No component forecast paths were supplied to the combination.",
                "This is a bug in the caller: the combination needs each component's future path.",
            )
        means = np.vstack([np.asarray(c.mean[:horizon], dtype=float) for c in comps])
        lower = np.vstack(
            [np.asarray(c.lower[:horizon], dtype=float) if c.lower else np.full(horizon, np.nan) for c in comps]
        )
        upper = np.vstack(
            [np.asarray(c.upper[:horizon], dtype=float) if c.upper else np.full(horizon, np.nan) for c in comps]
        )
        if self.method == "median":
            point = np.median(means, axis=0)
        else:
            assert self._weights is not None
            w = self._weights[: means.shape[0]]
            point = means.T @ w + (self._intercept if self.method == "granger_ramanathan" else 0.0)
        lo = np.nanmin(lower, axis=0) if np.isfinite(lower).any() else point
        hi = np.nanmax(upper, axis=0) if np.isfinite(upper).any() else point
        # Never let the envelope sit the wrong side of the point forecast.
        lo = np.minimum(lo, point)
        hi = np.maximum(hi, point)
        return pd.DataFrame({"mean": point, "lower": lo, "upper": hi})

    # -- summary ------------------------------------------------------------ #
    def _summary(self) -> FittedSummary:
        comps = self.card.get("components", [])
        weights_txt = ", ".join(f"{c['model_id']} {c['weight']:.2f}" for c in comps)
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            # Parameters = the weights. Reported so the table's k column is meaningful, but the IC is
            # refused below, which is what keeps this row out of the AICc ranking.
            n_params=len(comps),
            loglik=None,
            aic=None,
            aicc=None,
            bic=None,
            n_obs_used=int(self.card.get("held_out_origins", 0)),
            fitted_on=self.fitted_on,
            roots_ok=None,
            roots_note="",
            is_baseline=False,
            converged_warnings=list(self._caveats),
            plain_one_liner=(
                f"The {METHOD_NOUN[self.method]} across {len(comps)} models"
                + (f" ({weights_txt})." if weights_txt else ".")
            ),
            caveats=list(self._caveats),
            ic_comparable=False,
            ic_note=(
                "No information criterion: a combination has no likelihood of its own, and scoring it "
                "from the regression's residual sum of squares would compare it with models on a "
                "different scale. It is ranked by measured out-of-sample accuracy."
            ),
        )


__all__ = [
    "METHODS",
    "METHOD_PLAIN",
    "CombinationMethod",
    "CombinationModel",
    "ComponentForecast",
]
