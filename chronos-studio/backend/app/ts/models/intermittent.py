"""
ts/models/intermittent.py — Croston (SBA) and TSB (§5.8 C).

Intermittent demand breaks the assumptions of every method above: the series is mostly zeros, so a
level-and-trend model forecasts a small non-zero number every period and is wrong in a way that
compounds — it never forecasts a zero, so it always carries phantom inventory.

Croston's insight was to smooth two things separately: the **size** of a demand and the **interval**
between demands, then forecast the ratio (a rate per period). The Syntetos–Boylan correction (SBA)
halves the interval estimate to remove Croston's known positive bias. TSB keeps Croston's smoothing
but lets the occurrence probability decay when demand stops, which is what you want for a part that
may have become obsolete.

The intervals here come from the empirical distribution of the simulated rate, not from a normal
approximation: with 72 % zeros the predictive distribution is a spike at zero plus a small tail, and
a symmetric band would be a fiction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper
from app.ts.models.predictive import DiscretePredictive

#: The smoothing constant. 0.1 is Croston's own choice and remains the default in every standard
#: implementation; exposing it as a tunable would invite fitting it to the sample, which is exactly
#: the overfitting intermittent data cannot afford.
ALPHA = 0.1


def _integerise(draws: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Turn non-negative demand draws into counts without biasing their mean."""
    z = np.maximum(np.asarray(draws, dtype=float), 0.0)
    floor = np.floor(z)
    fraction = z - floor
    return (floor + (rng.random(z.shape) < fraction)).astype(np.int64)


def _nonzero_runs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The non-zero demand sizes, the intervals between them, and their positions.

    The interval *before* the first demand is not defined by the series — the record simply starts
    where it starts — so it is excluded rather than imputed.
    """
    idx = np.flatnonzero(y > 0)
    if idx.size == 0:
        return np.array([]), np.array([]), idx
    sizes = y[idx]
    intervals = np.diff(idx)
    return sizes, intervals, idx


class _IntermittentBase(ModelWrapper):
    family = ModelFamily.INTERMITTENT

    def _check(self, y: pd.Series) -> np.ndarray:
        yv = y.dropna().to_numpy(dtype=float)
        if yv.size < 12:
            raise FitFailure(
                f"{self.display_name} needs at least 12 observations and there are {yv.size}.",
                "Use a baseline benchmark; with this little data a rate is all that can be said.",
                kind="length",
            )
        if np.any(yv < 0):
            raise FitFailure(
                f"The series contains {int((yv < 0).sum())} negative values, which intermittent "
                "demand models cannot represent.",
                "This is not a demand series; use the continuous branch.", kind="positivity",
            )
        sizes, _, idx = _nonzero_runs(yv)
        if idx.size < 3:
            raise FitFailure(
                f"Only {idx.size} non-zero observations — there is not enough to estimate a demand "
                "size and an interval.",
                "If the series really is that sparse, the honest forecast is the observed rate with "
                "a very wide interval, and the app will not manufacture one.", kind="length",
            )
        return yv

    def _simulation_draws(self, horizon: int, *, n_paths: int = 4000, seed: int = 20260101) -> np.ndarray:
        """One seeded occurrence/size simulation from the fold's fitted parameters only."""
        rng = np.random.default_rng(int(seed))
        yv = self.y.dropna().to_numpy(dtype=float)
        sizes = yv[yv > 0]
        if self.model_id == "tsb":
            probs = self.prob * (1 - self.beta) ** np.arange(int(horizon))
            return np.where(
                rng.random((int(n_paths), int(horizon))) < probs[None, :],
                rng.choice(sizes, size=(int(n_paths), int(horizon)), replace=True),
                0.0,
            )
        interval = getattr(self, "interval_level", None) or getattr(self, "mean_interval", 1.0)
        p_hit = float(np.clip(1.0 / max(float(interval), 1.0), 1e-6, 1.0))
        hit = rng.random((int(n_paths), int(horizon))) < p_hit
        raw = np.where(hit, rng.choice(sizes, size=(int(n_paths), int(horizon)), replace=True), 0.0)
        correction = 0.5 if self.model_id == "croston" else float(getattr(self, "correction", 1.0))
        return np.asarray(raw, dtype=float) * correction

    def evaluation_predictive(
        self, *, horizon: int, n_paths: int = 2000, seed: int = 20260101
    ) -> DiscretePredictive:
        """
        An empirical integer predictive PMF for S14, kept separate from ``predictive()``.

        Croston/TSB/ADIDA simulate demand occurrences and sizes rather than exposing a closed-form
        likelihood. Evaluation still has to score a discrete PMF (§5.12), so the same seeded
        mechanism is sampled, stochastically integerised to preserve its mean, and turned into an
        explicit per-step PMF. It is empirical, not exact: the construction travels with the scores.
        This method is deliberately *not* named ``predictive()`` so S13's forecast presentation is
        unchanged; S14 asks for it explicitly.
        """
        h, n = int(horizon), int(n_paths)
        draws = self._simulation_draws(h, n_paths=n, seed=int(seed))
        counts = _integerise(draws, np.random.default_rng(int(seed) + 7919))
        top = max(int(counts.max()), 1)
        grid = np.arange(top + 1, dtype=float)
        pmf = np.empty((h, top + 1), dtype=float)
        for j in range(h):
            pmf[j] = np.bincount(counts[:, j], minlength=top + 1) / float(n)
        cdf = np.cumsum(pmf, axis=1)
        lo = np.array([grid[np.searchsorted(cdf[j], 0.025, side="left")] for j in range(h)])
        hi = np.array([grid[np.searchsorted(cdf[j], 0.975, side="left")] for j in range(h)])
        return DiscretePredictive(
            grid=np.tile(grid, (h, 1)), pmf=pmf, mean=pmf @ grid,
            lower=np.minimum(lo, hi), upper=np.maximum(lo, hi),
            truncated=np.zeros(h, dtype=bool),
            family=f"empirical_integer_{self.model_id}",
        )


class CrostonModel(_IntermittentBase):
    model_id = "croston"
    display_name = "Croston (SBA)"

    def _fit(self, y: pd.Series) -> None:
        yv = self._check(y)
        sizes, intervals, idx = _nonzero_runs(yv)
        z = float(sizes[0])
        # `intervals` has one fewer element than `sizes` (the gap *before* the first demand is not a
        # gap the series defines), so the smoothing pairs interval i with size i+1.
        p = float(intervals[0]) if intervals.size else 1.0
        for i in range(1, sizes.size):
            z += ALPHA * (sizes[i] - z)
            if i - 1 < intervals.size:
                p += ALPHA * (intervals[i - 1] - p)
        self.size_level = z
        self.interval_level = p
        # SBA: the ratio is biased high because the interval estimate lags the last demand, so the
        # interval is halved. This is the Syntetos–Boylan correction and it is not optional.
        self.rate = float(z / max(p, 1e-9)) * 0.5 if p > 1 else float(z / max(p, 1e-9))
        self.rate_uncorrected = float(z / max(p, 1e-9))
        self.n_nonzero = int(idx.size)
        self.alpha = ALPHA

        # The one-step fitted value is the rate itself: Croston forecasts a constant rate, which is
        # why it can never forecast a zero and why its intervals matter so much.
        self._fitted = np.full(yv.size, self.rate)
        self._resid = yv - self.rate
        self._raw_params = {
            "demand_size": (self.size_level, None, False),
            "interval": (self.interval_level, None, False),
            "rate": (self.rate, None, False),
            "alpha": (self.alpha, None, True),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        # Simulate: with probability 1/p a demand occurs, with the smoothed size distribution.
        # SBA halves the realised size expectation so the interval and point forecast agree.
        draws = self._simulation_draws(horizon, n_paths=4000, seed=20260101)
        mean = np.full(horizon, self.rate)
        lo = np.quantile(draws, alpha / 2, axis=0)
        hi = np.quantile(draws, 1 - alpha / 2, axis=0)
        return pd.DataFrame({"mean": mean, "lower": lo, "upper": hi},
                            index=np.arange(1, horizon + 1))

    def _summary(self) -> FittedSummary:
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=3,
            loglik=None, aic=None, aicc=None, bic=None,
            n_obs_used=int(self.y.dropna().size),
            fitted_on="levels",
            roots_ok=None,
            plain_one_liner=(
                f"Smooths how big a demand is ({self.size_level:,.1f}) and how often it happens "
                f"(every {self.interval_level:,.1f} periods), then forecasts the rate: "
                f"{self.rate:,.3f} per period."
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                f"Croston's rate is corrected for bias by halving the interval (SBA); the uncorrected "
                f"rate would be {self.rate_uncorrected:,.3f} per period.",
                "The forecast is a constant rate, so this model never predicts a zero — the interval "
                "is where the zeros live.",
                "MAPE is not reported for intermittent series: it is undefined at zero.",
            ],
            ic_comparable=False,
            ic_note="Not a likelihood fit, so no AIC. Compare on MASE or on the interval coverage.",
        )


class TSBModel(_IntermittentBase):
    model_id = "tsb"
    display_name = "TSB"

    def _fit(self, y: pd.Series) -> None:
        yv = self._check(y)
        sizes, _, idx = _nonzero_runs(yv)
        # TSB smooths the occurrence *probability* at every period (including zeros), so obsolescence
        # pulls the forecast down instead of freezing it at the last rate.
        beta = ALPHA
        prob = float(np.mean(yv > 0))
        size_level = float(sizes[0])
        probs: list[float] = [prob]
        for t in range(yv.size):
            occurred = 1.0 if yv[t] > 0 else 0.0
            prob += beta * (occurred - prob)
            if yv[t] > 0:
                size_level += ALPHA * (yv[t] - size_level)
            probs.append(prob)
        self.prob = float(probs[-1])
        self.size_level = size_level
        self.rate = float(self.prob * self.size_level)
        self.n_nonzero = int(idx.size)
        self.beta = beta
        self._fitted = self.prob * size_level * np.ones(yv.size)
        self._resid = yv - self._fitted
        self._raw_params = {
            "occurrence_prob": (self.prob, None, False),
            "demand_size": (self.size_level, None, False),
            "rate": (self.rate, None, False),
            "beta": (self.beta, None, True),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        # The occurrence probability decays geometrically when nothing happens, which is the whole
        # point of TSB: a long run of zeros must lower the forecast, not leave it frozen.
        draws = self._simulation_draws(horizon, n_paths=4000, seed=20260101)
        mean = self.prob * (1 - self.beta) ** np.arange(horizon) * self.size_level
        lo = np.quantile(draws, alpha / 2, axis=0)
        hi = np.quantile(draws, 1 - alpha / 2, axis=0)
        return pd.DataFrame({"mean": mean, "lower": lo, "upper": hi},
                            index=np.arange(1, horizon + 1))

    def _summary(self) -> FittedSummary:
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=3,
            loglik=None, aic=None, aicc=None, bic=None,
            n_obs_used=int(self.y.dropna().size),
            fitted_on="levels",
            roots_ok=None,
            plain_one_liner=(
                f"Smooths the chance that a period has any demand ({self.prob * 100:.1f}%) and how "
                f"large it is when it happens ({self.size_level:,.1f}), giving {self.rate:,.3f} per "
                "period — falling as zeros accumulate."
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                "TSB's forecast decays when demand stops, which is right for a part that may be "
                "obsolete but wrong if the zero run is simply a short dry spell.",
                "MAPE is not reported for intermittent series: it is undefined at zero.",
            ],
            ic_comparable=False,
            ic_note="Not a likelihood fit, so no AIC. Compare on MASE or on interval coverage.",
        )




class ADIDAModel(_IntermittentBase):
    """
    ADIDA — Aggregate–Disaggregate Intermittent Demand Approach.

    Lumpy demand (many zeros, and when demand does come it is large and irregular) is hard for Croston
    because the *sizes* are too variable for a single smoothed average. ADIDA sidesteps that by
    changing the time unit: aggregate the series into blocks whose expected demand is about one per
    block, forecast the aggregated series with a simple exponential smoothing, then disaggregate back
    to the original period by dividing by the block length.

    The aggregation bucket is chosen as ceil(mean interval), which is the standard rule and is also the
    one that makes the disaggregation honest: with a bucket that short, most blocks contain one demand,
    so the aggregate series is smooth enough to forecast and the division by the bucket length is the
    only step that could distort. On AirPassengers-scale data buckets get truncated at 1 and ADIDA
    degenerates to SES — which is correct, and is why it is only offered for intermittent series.
    """

    model_id = "adida"
    display_name = "ADIDA (aggregate–disaggregate)"

    def _fit(self, y: pd.Series) -> None:
        yv = self._check(y)
        sizes, intervals, idx = _nonzero_runs(yv)
        n = yv.size
        mean_interval = float(np.mean(intervals)) if intervals.size else float(n)
        self.bucket = int(max(1, np.ceil(mean_interval)))
        if n // self.bucket < 4:
            raise FitFailure(
                f"A bucket of {self.bucket} periods leaves only {n // self.bucket} aggregate blocks, "
                "which is too few to smooth.",
                "Use Croston or TSB, which work on the original period.",
                kind="length",
            )
        nb = n // self.bucket
        blocks = yv[: nb * self.bucket].reshape(nb, self.bucket).sum(axis=1)
        # SES on the aggregated series. Alpha is chosen by SSE over a small grid rather than fixed,
        # because ADIDA's whole claim is that the aggregate series is smooth enough to forecast well —
        # if a fixed alpha is wrong, the claim is untested.
        best: tuple[float, float] | None = None
        for alpha in np.linspace(0.05, 0.6, 12):
            level = float(blocks[0])
            sse = 0.0
            for v in blocks[1:]:
                sse += (v - level) ** 2
                level += alpha * (v - level)
            if best is None or sse < best[1]:
                best = (float(alpha), float(sse))
        assert best is not None
        self.alpha, self._sse = best
        level = float(blocks[0])
        fitted_blocks = np.empty(nb)
        for i, v in enumerate(blocks):
            fitted_blocks[i] = level
            level += self.alpha * (v - level)
        self.block_level = level
        self.rate = float(level / self.bucket)
        self.mean_interval = mean_interval
        self.n_nonzero = int(idx.size)

        # Disaggregate the fitted blocks back onto the original index so diagnostics work on the series
        # the user actually has, rather than on a smoothed-up series that shares none of its values.
        fitted = np.repeat(fitted_blocks / self.bucket, self.bucket)
        if fitted.size < n:
            fitted = np.r_[fitted, np.full(n - fitted.size, fitted[-1])]
        self._fitted = fitted
        self._resid = yv - fitted
        self._raw_params = {
            "aggregation bucket": (float(self.bucket), None, self.bucket == 1),
            "alpha (on blocks)": (self.alpha, None, False),
            "block level": (self.block_level, None, False),
            "rate per period": (self.rate, None, False),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        # The predictive distribution is the same mechanism Croston uses — sometimes nothing,
        # sometimes a demand size — but with the aggregate level as the rate, so the two models
        # can be compared on their intervals as well as their point forecasts.
        draws = self._simulation_draws(horizon, n_paths=4000, seed=20260101)
        mean = np.full(horizon, self.rate)
        lo = np.quantile(draws, alpha / 2, axis=0)
        hi = np.quantile(draws, 1 - alpha / 2, axis=0)
        return pd.DataFrame({"mean": mean, "lower": lo, "upper": hi},
                            index=np.arange(1, horizon + 1))

    def _summary(self) -> FittedSummary:
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"{self.display_name}, bucket {self.bucket}",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=3,
            loglik=None, aic=None, aicc=None, bic=None,
            n_obs_used=int(self.y.dropna().size),
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"Adds up {self.bucket} periods at a time until demand is roughly one per block, "
                f"smooths that easier series (alpha {self.alpha:.2f}), then divides back down to "
                f"{self.rate:.3f} per period."
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                "The rate is constant over the horizon: ADIDA smooths a level, so it cannot anticipate "
                "a trend in the demand rate.",
                "MAPE is not reported for intermittent series: it is undefined at zero.",
            ] + (
                ["A bucket of 1 means the aggregation changed nothing and ADIDA reduces to plain "
                 "exponential smoothing — the model is reported, but it is not doing what it is for."]
                if self.bucket == 1 else []
            ),
            ic_comparable=False,
            ic_note="Not a likelihood fit, so no AIC. Compare on MASE or interval coverage.",
        )


class CrostonVariantModel(_IntermittentBase):
    """
    Croston with a *selectable* bias correction: classic, SBA, or SY.

        classic: rate = z / p
        SBA    : rate = z / p × (1 − α/2)      (Syntetos–Boylan)
        SY     : rate = z / p × (1 − α/2) but only when p > 1.32; otherwise the correction is skipped

    The three are offered as one model with a parameter rather than three registry rows, because they
    are the same estimator with different small-sample corrections and presenting them as separate
    "models" would inflate the comparison table with three near-identical rows. The SY threshold is
    from the same paper as SBA and exists because the correction over-corrects when demand is frequent:
    at p ≈ 1.05, halving the rate is simply wrong.
    """

    model_id = "croston_sy"
    display_name = "Croston (SY-corrected)"

    #: The Syntetos–Boylan cut-off: below this mean interval the SBA correction is not applied.
    SY_THRESHOLD = 1.32

    def _fit(self, y: pd.Series) -> None:
        yv = self._check(y)
        sizes, intervals, idx = _nonzero_runs(yv)
        z = float(sizes[0])
        p = float(intervals[0]) if intervals.size else 1.0
        for i in range(1, sizes.size):
            z += ALPHA * (sizes[i] - z)
            if i - 1 < intervals.size:
                p += ALPHA * (intervals[i - 1] - p)
        self.size_level, self.interval_level = z, p
        raw_rate = float(z / max(p, 1e-9))
        correction = (1 - ALPHA / 2) if p > self.SY_THRESHOLD else 1.0
        self.correction = correction
        self.rate = raw_rate * correction
        self.rate_uncorrected = raw_rate
        self.n_nonzero = int(idx.size)
        self._fitted = np.full(yv.size, self.rate)
        self._resid = yv - self.rate
        self._raw_params = {
            "demand_size": (self.size_level, None, False),
            "interval": (self.interval_level, None, False),
            "correction applied": (correction, None, correction == 1.0),
            "rate": (self.rate, None, False),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        draws = self._simulation_draws(horizon, n_paths=4000, seed=20260101)
        mean = np.full(horizon, self.rate)
        lo = np.quantile(draws, alpha / 2, axis=0)
        hi = np.quantile(draws, 1 - alpha / 2, axis=0)
        return pd.DataFrame({"mean": mean, "lower": lo, "upper": hi},
                            index=np.arange(1, horizon + 1))

    def _summary(self) -> FittedSummary:
        applied = self.correction < 1.0
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=3,
            loglik=None, aic=None, aicc=None, bic=None,
            n_obs_used=int(self.y.dropna().size),
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"Demand of about {self.size_level:,.1f} every {self.interval_level:,.1f} periods → "
                f"{self.rate:,.3f} per period"
                + (f" after the SY correction, which applies because the interval is above the "
                   f"{self.SY_THRESHOLD} cut-off; uncorrected it would be {self.rate_uncorrected:,.3f}."
                   if applied else
                   f". The mean interval is {self.interval_level:,.2f}, below the {self.SY_THRESHOLD} "
                   "cut-off, so the correction is skipped — demand is too frequent for it.")
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                "The bias correction is only applied when the mean interval exceeds "
                f"{self.SY_THRESHOLD}; below that, halving the rate would understate frequent demand.",
                "MAPE is not reported for intermittent series: it is undefined at zero.",
            ],
            ic_comparable=False,
            ic_note="Not a likelihood fit, so no AIC. Compare on MASE or interval coverage.",
        )



IMPLEMENTED_INTERMITTENT: dict[str, type[ModelWrapper]] = {
    "croston": CrostonModel,
    "tsb": TSBModel,
    "adida": ADIDAModel,
    "croston_sy": CrostonVariantModel,
}


__all__ = [
    "ADIDAModel",
    "ALPHA",
    "CrostonModel",
    "CrostonVariantModel",
    "IMPLEMENTED_INTERMITTENT",
    "TSBModel",
]
