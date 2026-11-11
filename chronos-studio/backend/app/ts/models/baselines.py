"""
ts/models/baselines.py — the benchmarks that are never optional (§5.8 A).

Five models, each of which exists to make a *specific* kind of claim checkable:

* **mean** — the floor. Any model that cannot beat the average is not a model.
* **naive** — the random-walk benchmark, and the MASE denominator when there is no season.
* **seasonal naive** — the same, at the seasonal period.
* **drift** — the cheapest model that extrapolates a trend, so "your model found a trend" can be
  compared against "there was a straight line all along".
* **white noise** — the null hypothesis with a *forecast*. This one is the important one: on a series
  with no residual autocorrelation at any lag, the honest output is "no model needed", and the app
  has to be willing to say it rather than fit an ARIMA to nothing.

The intervals here are analytic where a closed form exists (naive, drift) and simulation-free where
it does not (white noise uses the normal predictive distribution). Each is labelled, because "the
interval came from a formula" and "the interval came from 2000 simulated paths" are different
claims about the same picture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from app.core.formatting import p_phrase

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper
from app.ts.models.diagnostics import ljung_box


class _NoFitNeeded(ModelWrapper):
    """
    Shared machinery for the closed-form baselines.

    These have no likelihood in the usual sense, so "AIC" is reported as `None` rather than as a
    number computed from a pseudo-log-likelihood that would then be ranked against ARIMA's. §5.9's
    comparability rule applies to baselines too.
    """

    family = ModelFamily.BASELINE
    is_baseline = True

    def _fit(self, y: pd.Series) -> None:
        if y.dropna().size < 3:
            raise FitFailure(
                f"{self.display_name} needs at least 3 observations to estimate anything.",
                "Load a longer series.", kind="length",
            )

    def _summary(self) -> FittedSummary:
        y = self.y.dropna().to_numpy()
        n = y.size
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params(),
            loglik=None,
            aic=None, aicc=None, bic=None,
            n_obs_used=n,
            fitted_on=self.fitted_on,
            is_baseline=True,
            plain_one_liner=self._one_liner(),
            params=self.params_table(getattr(self, "_raw_params", {})),
            ic_comparable=False,
            ic_note=(
                "This is a benchmark, not a likelihood fit: it has no information criterion, and "
                "ranking it by AIC against ARIMA would compare nothing."
            ),
        )

    def _n_params(self) -> int:
        return 1

    def _one_liner(self) -> str:
        return ""


class MeanModel(_NoFitNeeded):
    model_id = "mean"
    display_name = "Mean (global average)"

    def _fit(self, y: pd.Series) -> None:
        super()._fit(y)
        yv = y.dropna().to_numpy()
        self.mu = float(yv.mean())
        self.sigma = float(yv.std(ddof=1))
        self._fitted = np.full(yv.size, self.mu)
        self._resid = yv - self.mu
        self._raw_params = {"mean": (self.mu, self.sigma / np.sqrt(yv.size), False)}

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        z = float(stats.norm.ppf(1 - alpha / 2))
        mean = np.full(horizon, self.mu)
        half = z * self.sigma
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=np.arange(1, horizon + 1))

    def _one_liner(self) -> str:
        return f"Every future value predicted as {self.mu:,.1f}, the sample average."


class NaiveModel(_NoFitNeeded):
    model_id = "naive"
    display_name = "Naive (random walk)"

    def _fit(self, y: pd.Series) -> None:
        super()._fit(y)
        yv = y.dropna().to_numpy()
        self.last = float(yv[-1])
        diffs = np.diff(yv)
        self.sigma = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
        self._fitted = np.concatenate([[np.nan], yv[:-1]])
        self._resid = diffs
        self._raw_params = {"last_value": (self.last, None, False),
                            "sigma(diff)": (self.sigma, None, False)}

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        z = float(stats.norm.ppf(1 - alpha / 2))
        steps = np.arange(1, horizon + 1)
        mean = np.full(horizon, self.last)
        # The random-walk forecast error variance grows linearly in h, so the interval width grows
        # as √h. Reporting a constant width here is the single most common way a fan chart lies.
        half = z * self.sigma * np.sqrt(steps)
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=steps)

    def _one_liner(self) -> str:
        return f"Tomorrow equals today ({self.last:,.1f}); uncertainty widens as √h."


class SeasonalNaiveModel(_NoFitNeeded):
    model_id = "seasonal_naive"
    display_name = "Seasonal naive"

    def _fit(self, y: pd.Series) -> None:
        super()._fit(y)
        yv = y.dropna().to_numpy()
        if self.period is None:
            raise FitFailure(
                "No seasonal period was detected, so there is no 'same season last year' to copy.",
                "Run Step 1 to confirm the calendar, or use the plain naive model.",
                kind="seasonality",
            )
        if yv.size < self.period + 2:
            raise FitFailure(
                f"Seasonal naive needs more than one full cycle (period {self.period}) and there "
                f"are only {yv.size} observations.",
                "Use the plain naive model, or load a longer series.", kind="length",
            )
        self.season_vals = yv[-self.period:]
        resid = yv[self.period:] - yv[:-self.period]
        self.sigma = float(resid.std(ddof=1)) if resid.size > 1 else 0.0
        self._fitted = np.concatenate([np.full(self.period, np.nan), yv[:-self.period]])
        self._resid = resid
        self._raw_params = {"period": (float(self.period), None, False),
                            "sigma(seasonal diff)": (self.sigma, None, False)}

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        z = float(stats.norm.ppf(1 - alpha / 2))
        m = int(self.period or 1)
        steps = np.arange(1, horizon + 1)
        mean = np.array([self.season_vals[(h - 1) % m] for h in steps], dtype=float)
        cycles = np.ceil(steps / m)
        half = z * self.sigma * np.sqrt(cycles)
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=steps)

    def _one_liner(self) -> str:
        return f"Each period repeats its value from {self.period} observations earlier."


class DriftModel(_NoFitNeeded):
    model_id = "drift"
    display_name = "Random walk with drift"

    def _fit(self, y: pd.Series) -> None:
        super()._fit(y)
        yv = y.dropna().to_numpy()
        n = yv.size
        self.last = float(yv[-1])
        self.slope = float((yv[-1] - yv[0]) / (n - 1)) if n > 1 else 0.0
        resid = np.diff(yv) - self.slope
        self.sigma = float(resid.std(ddof=1)) if resid.size > 1 else 0.0
        self._fitted = yv[:-1] + self.slope
        self._resid = np.diff(yv)
        self._raw_params = {
            "drift": (self.slope, None, False),
            "last_value": (self.last, None, False),
            "sigma(diff)": (self.sigma, None, False),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        z = float(stats.norm.ppf(1 - alpha / 2))
        steps = np.arange(1, horizon + 1)
        mean = self.last + self.slope * steps
        half = z * self.sigma * np.sqrt(steps)
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=steps)

    def _one_liner(self) -> str:
        return (
            f"Last value {self.last:,.1f} plus {self.slope:,.3f} per step — "
            "algebraically ARIMA(0,1,0) with a constant."
        )


class WhiteNoiseModel(_NoFitNeeded):
    """
    The null model, with a verdict attached.

    Its `verdict` is the sentence the app exists to be able to say: if nothing in the residual
    autocorrelation battery rejects, then every more complex model is fitting noise, and the
    correct recommendation is "no model needed". `evidence` carries the p-values that led there so
    the claim is checkable rather than rhetorical.
    """

    model_id = "white_noise"
    display_name = "White noise (no model)"

    #: Lags tested for the "no model needed" verdict.
    lags: tuple[int, ...] = (5, 10, 15)

    def _fit(self, y: pd.Series) -> None:
        yv = y.dropna().to_numpy()
        if yv.size < 10:
            raise FitFailure(
                f"Testing for white noise needs at least 10 observations and there are {yv.size}.",
                "Load a longer series.", kind="length",
            )
        self.mu = float(yv.mean())
        self.sigma = float(yv.std(ddof=1))
        self._fitted = np.full(yv.size, self.mu)
        self._resid = yv - self.mu
        self._raw_params = {"mean": (self.mu, self.sigma / np.sqrt(yv.size), False),
                            "sigma": (self.sigma, None, False)}

        # The whole model rests on this battery, so it is stored, not summarised away.
        self.evidence: list[dict[str, float | int | None]] = []
        for lag in self.lags:
            p, used = ljung_box(self._resid, lags=min(lag, max(2, yv.size // 5)))
            if p is not None:
                self.evidence.append({"lag": int(used or lag), "p_value": float(p)})
        self.is_white = all(e["p_value"] > 0.05 for e in self.evidence) if self.evidence else False
        self.verdict = (
            "No model needed: nothing in the autocorrelation battery rejects."
            if self.is_white else
            "There is structure here for a model to find."
        )

    def _summary(self) -> FittedSummary:
        s = super()._summary()
        s.plain_one_liner = self.verdict
        if self.evidence:
            s.converged_warnings = [
                f"Ljung–Box at lag {e['lag']}: {p_phrase(e['p_value'])}" for e in self.evidence
            ]
        return s

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        z = float(stats.norm.ppf(1 - alpha / 2))
        mean = np.full(horizon, self.mu)
        half = z * self.sigma
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=np.arange(1, horizon + 1))

    def _one_liner(self) -> str:
        return self.verdict


BASELINE_CLASSES: dict[str, type[ModelWrapper]] = {
    "mean": MeanModel,
    "naive": NaiveModel,
    "seasonal_naive": SeasonalNaiveModel,
    "drift": DriftModel,
    "white_noise": WhiteNoiseModel,
}


__all__ = [
    "BASELINE_CLASSES",
    "DriftModel",
    "MeanModel",
    "NaiveModel",
    "SeasonalNaiveModel",
    "WhiteNoiseModel",
]
