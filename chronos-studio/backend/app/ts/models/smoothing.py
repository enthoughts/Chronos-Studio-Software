"""
ts/models/smoothing.py — ETS / Holt–Winters and the Theta method (§5.8 B).

**ETS** is fitted as a grid over the legal component combinations, selected by AICc, using
statsmodels' `ExponentialSmoothing`. The grid is filtered before fitting, not after: a multiplicative
component on a series containing zero or a negative value is not a model that merely fits badly — it
is undefined, because the recursion multiplies by a level that must be positive. Offering it and then
apologising is how a user concludes the app is broken.

**Theta** is the M3/M4-winning method and is here because it is the single best answer to "short
series, strong seasonality, no appetite for order selection". It is implemented in the form that won:
seasonal decomposition by ratio-to-moving-average when a period exists, then theta lines at θ = 0
(linear regression, extrapolated) and θ = 2 (the other extreme), averaged with weights chosen by the
standard 50/50 rule — the "optimised θ" variant is deliberately *not* faked, because the optimisation
requires the M3 fitting procedure and a half-copy of it would be its own kind of lie.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper


class ETSModel(ModelWrapper):
    model_id = "ets"
    display_name = "ETS"

    family = ModelFamily.SMOOTHING

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.spec: tuple[str, str, str, bool] | None = None
        self._alternatives: list[dict[str, Any]] = []
        #: Size of the legal grid. Public because "chosen from N specifications" is a sentence the
        #: summary says out loud, and a test should be able to check it against the grid itself.
        self.candidates_tried: int = 0

    @property
    def alternatives(self) -> list[dict[str, Any]]:
        """Every specification the grid tried, with its AICc — the audit trail behind the choice."""
        return list(self._alternatives)

    def _legal(self, y: np.ndarray) -> list[tuple[str, str, str, bool]]:
        positive = bool(np.all(y > 0))
        seasonal = self.period is not None and y.size >= 2 * int(self.period or 0)
        specs: list[tuple[str, str, str, bool]] = []
        for trend in ("add", None):
            for damped in ((False, True) if trend == "add" else (False,)):
                for season in (("add", "mul") if seasonal else (None,)):
                    if season == "mul" and not positive:
                        continue
                    if season is None and y.size < 4:
                        continue
                    specs.append(("add", trend or "none", season or "none", damped))
        return specs

    def _fit(self, y: pd.Series) -> None:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        yv = y.dropna().to_numpy(dtype=float)
        if yv.size < 8:
            raise FitFailure(
                f"ETS needs at least 8 observations and there are {yv.size}.",
                "Use the mean or naive benchmark instead.", kind="length",
            )
        if yv.std() == 0:
            raise FitFailure(
                "The series is constant, so there is no level or trend to smooth.",
                "The mean model already is the answer here.", kind="degenerate",
            )

        best: dict[str, Any] | None = None
        tried = 0
        failures: list[str] = []
        for err, trend, season, damped in self._legal(yv):
            tried += 1
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        yv,
                        trend=trend if trend != "none" else None,
                        damped_trend=damped,
                        seasonal=season if season != "none" else None,
                        seasonal_periods=int(self.period) if season != "none" else None,
                        initialization_method="estimated",
                    )
                    res = model.fit(optimized=True, use_brute=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{trend}/{season}{'/damped' if damped else ''}: {type(exc).__name__}")
                continue
            # statsmodels' HoltWintersResults exposes `aicc`/`bic`/`sse` but not `llf`, so the
            # log-likelihood is recovered from its own AIC (aic = −2ℓ + 2k) rather than invented.
            k = 1 + (1 if trend != "none" else 0) + (1 if damped else 0) \
                + (int(self.period) if season != "none" else 0)
            n = yv.size
            aic_native = getattr(res, "aic", None)
            llf = (
                float(-0.5 * (float(aic_native) - 2 * k))
                if aic_native is not None and np.isfinite(aic_native) else None
            )
            _, aicc, _ = self.ic(n, k + 1, llf)
            score = aicc if aicc is not None else None
            if score is None:
                continue
            entry = {
                "spec": (err, trend, season, damped),
                "res": res, "score": score, "aicc": aicc, "aic": self.ic(n, k + 1, llf)[0],
                "loglik": llf, "k": k + 1,
            }
            self._alternatives.append({"spec": entry["spec"], "aicc": aicc})
            if best is None or score < best["score"]:
                best = entry
        if best is None:
            raise FitFailure(
                f"No legal ETS specification converged ({tried} tried).",
                "This usually means the series is too short for the optimiser; the baselines still "
                "stand. Details: " + "; ".join(failures[:3]),
                kind="convergence",
            )

        res = best["res"]
        self.spec = best["spec"]
        self.result = res
        self._aicc = best["aicc"]
        self._aic = best["aic"]
        self._loglik = best["loglik"]
        self._n_params = best["k"]
        self._fitted = np.asarray(res.fittedvalues, dtype=float)
        self._resid = yv - self._fitted
        raw: dict[str, tuple[float | None, float | None, bool]] = {}
        for name in ("smoothing_level", "smoothing_trend", "smoothing_seasonal", "damping_trend",
                     "initial_level", "initial_trend", "initial_seasons"):
            val = getattr(res.params, name, None)
            if val is None:
                continue
            arr = np.atleast_1d(np.asarray(val, dtype=float))
            # The seasonal initials are a vector; report their spread rather than seven rows.
            raw[name] = (float(arr[0]), None, False) if arr.size == 1 else (
                float(np.mean(arr)), None, False
            )
        self._raw_params = raw
        self.candidates_tried = tried

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        fc = self.result.forecast(int(horizon))
        mean = np.asarray(fc, dtype=float)
        resid_sd = float(np.nanstd(self._resid, ddof=1)) if self._resid is not None else 0.0
        try:
            from scipy import stats

            z = float(stats.norm.ppf(1 - alpha / 2))
        except Exception:  # pragma: no cover
            z = 1.96
        steps = np.arange(1, horizon + 1)
        # ETS predictive intervals are not symmetric in the multiplicative case; without the
        # analytical formulae from Hyndman et al. §6 a symmetric band from the residual scale is the
        # defensible approximation, and the caption says so rather than implying otherwise.
        half = z * resid_sd * np.sqrt(steps)
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=steps)

    def _summary(self) -> FittedSummary:
        err, trend, season, damped = self.spec  # type: ignore[misc]
        label = f"ETS({err[0].upper()},{trend[0].upper() if trend != 'none' else 'N'}," \
                f"{season[0].upper() if season != 'none' else 'N'}" \
                f"{'d' if damped else ''})"
        n = int(self._fitted.size)
        _, aicc, bic = self.ic(n, self._n_params, self._loglik)
        caveats: list[str] = []
        if season != "none":
            caveats.append(
                f"Intervals are symmetric approximations from the residual scale; a "
                f"{season}-season ETS has asymmetric predictive distributions, so the upper band is "
                "slightly too narrow at long horizons."
            )
        alt = sorted(
            (a for a in self._alternatives if a["aicc"] is not None),
            key=lambda a: a["aicc"],
        )
        if len(alt) > 1:
            second = alt[1]
            caveats.append(
                f"The second-best specification ({second['spec'][1]}/{second['spec'][2]}) is within "
                f"{abs(second['aicc'] - alt[0]['aicc']):.1f} AICc units, so the component choice is "
                "not decisive."
            )
        return FittedSummary(
            model_id=self.model_id,
            display_name=label,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=self._aic,
            aicc=self._aicc,
            bic=bic,
            n_obs_used=n,
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"{label}: a weighted average where recent values dominate"
                + (", with a trend" if trend != "none" else "")
                + (f" and a {self.period}-period season" if season != "none" else "")
                + f". Chosen from {self.candidates_tried} legal specifications by AICc."
            ),
            params=self.params_table(self._raw_params),
            caveats=caveats,
            ic_comparable=True,
            ic_note="AIC/AICc on the levels series; comparable with other levels-scale models.",
        )


class ThetaModel(ModelWrapper):
    """
    The classic Theta method (Assimakopoulos–Nikolopoulos), as used in the M3/M4 competitions.

    Seasonal adjustment by ratio-to-moving-average, then two theta lines — θ = 0, which is the
    linear-regression line extrapolated, and θ = 2, which doubles the curvature of the
    seasonally-adjusted series — averaged 50/50. That is the specification that won M3; the
    "optimised θ" variant needs the M3 optimisation harness and is not approximated here.
    """

    model_id = "theta"
    display_name = "Theta"

    family = ModelFamily.SMOOTHING

    def _fit(self, y: pd.Series) -> None:
        yv = y.dropna().to_numpy(dtype=float)
        if yv.size < 10:
            raise FitFailure(
                f"Theta needs at least 10 observations and there are {yv.size}.",
                "Use the mean or naive benchmark.", kind="length",
            )
        if yv.std() == 0:
            raise FitFailure(
                "The series is constant, so every theta line is the same line.",
                "The mean model is the answer.", kind="degenerate",
            )
        self.period_used = int(self.period) if self.period and yv.size >= 2 * self.period else None
        if self.period_used:
            self._deseasonalise(yv)
            z = self.z
        else:
            self.seasonal_idx = None
            self.z = yv.copy()
            z = self.z

        t = np.arange(1, z.size + 1, dtype=float)
        slope, intercept = np.polyfit(t, z, 1)
        self.slope, self.intercept = float(slope), float(intercept)
        # θ = 2 line: the series is doubled and detrended against the fitted line.
        theta2 = 2.0 * z - (self.intercept + self.slope * t)
        self.theta_line_0 = self.intercept + self.slope * t
        self.theta_line_2 = theta2
        combined = 0.5 * self.theta_line_0 + 0.5 * self.theta_line_2
        # The two theta lines average back to the series itself — that is an algebraic identity
        # (0.5·L + 0.5·(2Z − L) = Z), not a perfect fit. Reporting it as fitted values would give
        # Theta an in-sample MASE of exactly 0.0, which is the most convincing wrong number this app
        # could print. So the in-sample fit is declared unavailable: Theta can only be judged out of
        # sample, and the diagnostics block says so by staying empty.
        self._fitted_adj = combined
        self._fitted = None
        self._resid = None

    def _deseasonalise(self, yv: np.ndarray) -> None:
        m = int(self.period_used)  # type: ignore[arg-type]
        n = yv.size
        if np.any(yv <= 0):
            raise FitFailure(
                "Seasonal adjustment for Theta uses ratios to a moving average, which needs "
                f"strictly positive data and the minimum is {yv.min():.3g}.",
                "Use an additive ETS model instead.", kind="positivity",
            )
        # Classic centred moving average, with the two half-weights at the ends of an even window.
        if m % 2 == 0:
            weights = np.concatenate([[0.5], np.ones(m - 1), [0.5]]) / m
        else:
            weights = np.ones(m) / m
        w = weights.size
        ma = np.full(n, np.nan)
        for i in range(n - w + 1):
            ma[i + (w - 1) // 2] = float(np.dot(yv[i: i + w], weights))
        ratio = yv / ma
        idx = np.full(n, np.nan)
        for j in range(m):
            col = ratio[j::m]
            idx[j::m] = np.nanmean(col)
        # Normalise the indices to mean 1 so the level is not inflated by the adjustment.
        mean_idx = np.nanmean(idx)
        if not np.isfinite(mean_idx) or mean_idx <= 0:
            raise FitFailure(
                "The seasonal indices could not be estimated — the series has no usable cycle at "
                f"period {m}.", "Use a non-seasonal method.", kind="seasonality",
            )
        self.seasonal_idx = idx / mean_idx
        self.z = yv / self.seasonal_idx

    def _reseasonalise(self, z: np.ndarray) -> np.ndarray:
        assert self.seasonal_idx is not None
        return z * self.seasonal_idx

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        n = self.z.size
        steps = np.arange(1, horizon + 1)
        t_future = n + steps
        theta0 = self.intercept + self.slope * t_future
        # θ = 2 has zero slope by construction: it is 2·z minus the trend line, so its forecast is
        # the flat continuation of (2·z_n − trend_n).
        theta2_future = np.full(horizon, float(self.theta_line_2[-1]))
        mean_adj = 0.5 * theta0 + 0.5 * theta2_future
        if self.seasonal_idx is not None:
            idx = np.array([self.seasonal_idx[(n + h - 1) % len(self.seasonal_idx)] for h in steps])
            mean = mean_adj * idx
        else:
            mean = mean_adj
        # A dispersion proxy for the interval: the deviation of the series from the regression line,
        # which is the part of the series the two theta lines disagree about.
        resid = (self.z - self.theta_line_0)
        resid_sd = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
        from scipy import stats

        z = float(stats.norm.ppf(1 - alpha / 2))
        half = z * resid_sd * np.sqrt(steps)
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=steps)

    def _summary(self) -> FittedSummary:
        n = int(self.z.size)
        # Theta has two free parameters in effect: the regression line's slope and intercept.
        k = 2
        _, aicc, bic = self.ic(n, k, None)
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"Theta(θ=2, season {self.period_used or 'none'})",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=k,
            loglik=None,
            aic=None,
            aicc=None,
            bic=None,
            n_obs_used=n,
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                "Averages two versions of the series: one extrapolated as a straight line, one "
                "curving more than the data does"
                + (f", after removing a {self.period_used}-period seasonal pattern" if self.period_used else "")
                + f". Trend is {self.slope:,.4f} per step."
            ),
            params=self.params_table({
                "slope": (self.slope, None, False),
                "intercept": (self.intercept, None, False),
            }),
            caveats=[
                "No in-sample residuals are reported: Theta's two lines average back to the series "
                "itself, so its in-sample error is zero by construction and means nothing. Judge it "
                "on out-of-sample accuracy.",
            ] + (
                [f"Seasonal indices were estimated from a centred {self.period_used}-point moving "
                 "average, so the first and last few periods are less reliable."]
                if self.period_used else []
            ),
            ic_comparable=False,
            ic_note="Theta is not a likelihood fit, so it has no AIC. Deciding between it and ETS "
                   "on AIC would compare a number with a blank.",
        )


IMPLEMENTED_SMOOTHING: dict[str, type[ModelWrapper]] = {
    "ets": ETSModel,
    "theta": ThetaModel,
}


__all__ = ["ETSModel", "IMPLEMENTED_SMOOTHING", "ThetaModel"]
