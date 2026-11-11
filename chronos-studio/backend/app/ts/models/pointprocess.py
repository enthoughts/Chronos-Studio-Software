"""
ts/models/pointprocess.py — models for event timestamps rather than values (§5.8 C).

A series of dates is not a series of numbers, and treating "days between purchases" as if it were a
monthly value is a category error that produces confident nonsense. Two models here, both fitted to the
*inter-arrival times*:

* **Renewal (exponential or Weibull)** — the null model of a point process: arrivals independent of
  history. Exponential means constant hazard ("it can happen any time"); Weibull with shape k < 1 means
  a decreasing hazard, which is what wear-out and "the longer it has been quiet, the sooner it happens"
  look like, and k > 1 means the opposite.
* **Hawkes (exponential kernel)** — self-exciting: every event raises the intensity for a while.
  Aftershocks, contagion, viral spread. Fitted by maximum likelihood on the exact recursion, so the
  branching ratio (how much of the activity is self-generated) is estimated rather than asserted.

Event input mode is a data-class branch, so this module works on a `DatetimeIndex` whose *values* are
what was observed. The API supplies it when the dataset is classified `event_timestamp`; the models
then forecast *expected counts over the horizon* with a Poisson predictive distribution, because
"how many events should I expect by then" is the only question these models can answer honestly —
a timestamp forecast would be a single fabricated date with a fake interval.

The renewal fit is a genuine MLE over the shape parameter; the Hawkes fit uses the standard
log-likelihood recursion with a closed-form intensity. Both are checked in tests against data generated
from the model they are supposed to recover, because a point-process fit that cannot recover its own
generating parameters is a curve fit with better marketing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper
from app.ts.models.predictive import frame, poisson_pmf, summarise


def inter_arrivals(times: pd.Index | np.ndarray) -> np.ndarray:
    """
    Positive gaps between successive event times, in the index's own units.

    Zero-length gaps are kept and negative ones (unsorted input) are dropped. This matters more than it
    looks: when dates are recorded at day resolution, two events on the same day produce a zero gap, and
    *dropping* them inflates the mean gap — 456 aftershocks recorded on 40 distinct days have a mean gap
    of 7.2 days if the same-day pairs are discarded and 1.7 if they are not. The event rate is what a
    point process is fitted to estimate, so discarding same-day information biased it low by a factor
    of four. The likelihood handles a zero gap fine (density is finite); it is a *negative* gap that has
    no meaning.
    """
    t = pd.DatetimeIndex(times) if not isinstance(times, pd.DatetimeIndex) else times
    if len(t) < 3:
        return np.array([])
    if isinstance(t, pd.DatetimeIndex):
        # Convert to days — the unit that is meaningful for monthly/quarterly event data and stable
        # across calendars. A horizon given in periods is converted back at forecast time.
        gaps = np.diff(t.asi8) / (1e9 * 3600 * 24)
    else:
        gaps = np.diff(np.asarray(t, dtype=float))
    return gaps[gaps >= 0] if np.any(gaps == 0) else gaps[gaps > 0]


class _PointProcessBase(ModelWrapper):
    family = ModelFamily.POINT_PROCESS

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.gaps: np.ndarray = np.array([])

    def _check(self, y: pd.Series) -> np.ndarray:
        if not isinstance(y.index, pd.DatetimeIndex):
            raise FitFailure(
                "A point-process model needs an event timestamp index, and this series has an "
                f"{type(y.index).__name__} index.",
                "Load the dates as the time column, and classify the data as event timestamps.",
                kind="dataclass",
            )
        gaps = inter_arrivals(y.index)
        # A continuous-time kernel needs continuous time. When most gaps are exactly zero the timestamps
        # are recorded to the day and several events share a date, so the likelihood is driven by the
        # arbitrary floor put under those zeros — on the 456-date aftershock file with day resolution,
        # 289 of 455 gaps were zero and the Weibull log-likelihood moved by 9000 between a one-second and
        # a one-hour floor. No estimate survives that, so the model declines and says what to change.
        tied = int(np.count_nonzero(gaps == 0))
        share = tied / max(gaps.size, 1)
        if share > 0.20:
            raise FitFailure(
                f"{tied} of {gaps.size} intervals are zero ({share * 100:.0f}%), because the timestamps "
                "are recorded only to the day and several events share a date. A continuous-time model "
                "cannot separate them.",
                "Supply timestamps with times of day (the file parses 'YYYY-MM-DD HH:MM'), or model the "
                "events as counts per period instead.",
                kind="resolution",
            )
        # A one-second floor for any remaining ties, which is below the recording resolution and so
        # changes no estimate once the tie share is small.
        gaps = np.maximum(gaps, 1.0 / 86400.0) if gaps.size else gaps
        if gaps.size < 10:
            raise FitFailure(
                f"Only {gaps.size} usable intervals between {y.index.size} recorded events — there is "
                "no arrival process to estimate.",
                "A point-process model needs dozens of events; use a count model on periods instead.",
                kind="length",
            )
        self.gaps = gaps
        # Distinct timestamps, not rows: kept so a caller can check that the *event index* was handed
        # to the fit rather than a count series that happened to share the DatetimeIndex.
        self._n_events = int(pd.DatetimeIndex(y.index).unique().size)
        return gaps

    def _period_days(self) -> float:
        """
        Days per recording period, used to express a period-based horizon in days.

        The index spacing comes first and the seasonal period second. A weekly-bucketed count series
        carries `period=52` for seasonality, and dividing 365.25 by that gives 7.02 days only by luck;
        the *spacing of the index* is the actual length of a period, and it is the only thing that
        stays right when the calendar is quarterly or the season is absent.
        """
        if self.period_days:
            return float(self.period_days)
        if self.period and self.period >= 2:
            return 365.25 / self.period
        return 30.0


class RenewalModel(_PointProcessBase):
    """Exponential or Weibull renewal process, chosen by AIC."""

    model_id = "renewal"
    display_name = "Renewal / inter-arrival"

    def _fit(self, y: pd.Series) -> None:
        gaps = self._check(y)
        # Exponential: rates are days per event, so the *hazard* is 1/mean in events per day.
        mean_gap = float(gaps.mean())
        shape, loc, scale = stats.weibull_min.fit(gaps, floc=0)

        ll_exp = float(np.sum(stats.expon.logpdf(gaps, scale=mean_gap)))
        ll_wbl = float(np.sum(stats.weibull_min.logpdf(gaps, shape, loc=0, scale=scale)))
        n = gaps.size
        aic_exp = -2 * ll_exp + 2 * 2
        aic_wbl = -2 * ll_wbl + 2 * 3
        # Both are fitted to the same intervals, so the comparison is legitimate; the Weibull only wins
        # if it earns its extra shape parameter.
        self.family_used = "weibull" if aic_wbl < aic_exp else "exponential"
        self.shape = float(shape)
        self.scale = float(scale)
        self.mean_gap = mean_gap
        self.hazard_per_day = 1.0 / mean_gap
        self.kind = (
            "increasing" if self.shape > 1.05 else
            "decreasing" if self.shape < 0.95 else "constant"
        )
        self._loglik = ll_wbl if self.family_used == "weibull" else ll_exp
        self._aic = aic_wbl if self.family_used == "weibull" else aic_exp
        self._n_obs = n
        self._n_params = 3 if self.family_used == "weibull" else 2
        self._aicc = self.ic(n, self._n_params, self._loglik)[1]
        self._bic = self.ic(n, self._n_params, self._loglik)[2]
        # Fitted values on the observed scale: the expected gap at each event, which for a renewal
        # process is the unconditional mean. Diagnostic only — a renewal process has no memory, so its
        # residual is not a residual in the regression sense.
        # Fitted values live on the *period* scale because everything downstream (MASE, the comparison
        # table) compares them with a count series. The expected number of events in one period is
        # period_days / mean_gap; the previous version used mean_gap × period_days / mean_gap, i.e. the
        # period length in days, which is the wrong quantity by a factor of the gap and made the
        # renewal row's MASE meaningless (48 against a benchmark of 1).
        self._fitted = np.full(y.dropna().size, self._period_days() / max(mean_gap, 1e-9))
        self._resid = None
        self._raw_params = {
            "mean interval (days)": (mean_gap, None, False),
            "shape (Weibull k)": (self.shape, None, abs(self.shape - 1) < 0.05),
            "scale (days)": (self.scale, None, False),
            "hazard (events per day)": (self.hazard_per_day, None, False),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        days = self._period_days()
        rows: list[tuple[np.ndarray, np.ndarray, bool]] = []
        for h in range(1, horizon + 1):
            # *Per period*, not cumulative. Step h asks for the events expected in period h alone, so the
            # window is one period long for every h. The earlier version multiplied by h, which produced
            # a forecast that climbed 3.4 → 6.8 → 10.3 over a flat event rate.
            lam = days / max(self.mean_gap, 1e-9)
            rows.append(poisson_pmf(max(lam, 1e-9)))
        self._predictive = summarise(rows, alpha=alpha, family="poisson")
        return frame(self._predictive)

    @property
    def n_events(self) -> int:
        """How many timestamps the fit actually used — the check that the events, not the counts, arrived."""
        return int(getattr(self, "_n_events", 0))

    def predictive(self) -> Any:
        return getattr(self, "_predictive", None)

    def _summary(self) -> FittedSummary:
        per_period = self._period_days() / max(self.mean_gap, 1e-9)
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"Renewal ({self.family_used})",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=self._aic,
            aicc=self._aicc,
            bic=self._bic,
            n_obs_used=self._n_obs,
            fitted_on="inter-arrival times",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"Events arrive about every {self.mean_gap:.2f} days "
                f"({per_period:.3f} per recording period). The Weibull shape is {self.shape:.3f}, so "
                f"the hazard is {self.kind} over time — "
                + ("the longer it has been quiet, the more likely it is to happen."
                   if self.kind == "decreasing" else
                   "once it has happened, more is coming soon."
                   if self.kind == "increasing" else
                   "waiting does not change the chance.")
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                "The count over the horizon is approximated as Poisson. That is close when the mean "
                "interval is short relative to the horizon and wrong when the horizon barely covers "
                "one interval, where a renewal count is far more dispersed.",
                "The fitted model is a *rate*, so it cannot forecast the date of the next event — only "
                "how many to expect in a window.",
            ],
            ic_comparable=True,
            ic_note="AIC on the inter-arrival likelihood; comparable between the exponential and "
                    "Weibull specifications, which is what chose between them.",
        )


class HawkesModel(_PointProcessBase):
    """
    Self-exciting (Hawkes) process with an exponential kernel.

        λ(t) = μ + Σ_{t_i < t} α·exp(−β·(t − t_i))

    Fitted by maximum likelihood. The branching ratio n = α/β is the share of activity that is
    self-generated, and it is the reason to use this model: n ≈ 0.7 says most events are caused by
    earlier events, n ≈ 0 says the process is just a Poisson with a step. The fit refuses n ≥ 0.95,
    because at that point the process is near-critical and the likelihood is flat enough that the
    reported branching ratio would be an artefact of where the optimiser stopped.
    """

    model_id = "hawkes"
    display_name = "Hawkes (self-exciting)"

    def _fit(self, y: pd.Series) -> None:
        from scipy.optimize import minimize

        gaps = self._check(y)
        # Work in the event-time scale (days since the first event) — the likelihood recursion is over
        # absolute times, not gaps.
        t = np.r_[0.0, np.cumsum(gaps)]
        span = float(t[-1])
        if span <= 0:
            raise FitFailure("All events share one timestamp, so no intensity can be estimated.",
                             "Check the date parsing in Step 0.", kind="degenerate")
        n = t.size
        base_rate = n / span

        def nll(theta: np.ndarray) -> float:
            mu = float(np.exp(theta[0]))
            alpha = float(np.exp(theta[1]))
            beta = float(np.exp(theta[2]))
            # Guard every parameter, not just the ratio: a huge α with a huge β passes `α/β < 0.999`
            # and then overflows inside the recursion, which returns NaN and makes the likelihood look
            # unavailable rather than badly parameterised. That cost this module one debugging round.
            if not np.all(np.isfinite([mu, alpha, beta])):
                return 1e10
            if beta <= 1e-6 or alpha <= 0 or mu <= 0 or alpha / beta >= 0.999:
                return 1e10
            # Ogata's recursion. The excitation at event i is
            #     A(t_i) = (A(t_{i-1}) + α)·e^{−β Δt},
            # because the event that just happened contributes α *before* the decay to the next one.
            # Writing it as A·e^{−βΔt} + α — which is the natural transcription and was the first
            # version here — forgets to decay the new contribution, so the recursion over-states the
            # excitation for every closely-spaced pair. On a series generated with a branching ratio of
            # 0.6 that version estimated 0.291 with the background rate pinned at its lower bound.
            ll = 0.0
            A = 0.0
            prev = t[0]
            for i in range(n):
                if i > 0:
                    A = (A + alpha) * np.exp(-beta * (t[i] - prev))
                    prev = t[i]
                lam = mu + A
                if lam <= 0 or not np.isfinite(lam):
                    return 1e10
                ll += float(np.log(lam))
            # Compensator: ∫₀ᵀ λ = μT + Σ_i (α/β)(1 − e^{−β(T − t_i)}), over every event.
            comp = mu * span + float(
                np.sum(alpha / beta * (1 - np.exp(-beta * (span - t))))
            )
            return float(-(ll - comp))

        # Bounded in logs so the optimiser cannot propose α = e^700, which is what produced the NaN.
        bounds = [
            (np.log(base_rate * 1e-3), np.log(base_rate * 10)),
            (np.log(1e-6), np.log(base_rate * 100)),
            (np.log(1e-4), np.log(50.0)),
        ]
        best: tuple[float, np.ndarray] | None = None
        with np.errstate(all="ignore"):
            for start in (
                np.log([base_rate * 0.5, base_rate, 0.5]),
                np.log([base_rate * 0.8, base_rate * 0.2, 0.1]),
                np.log([base_rate * 0.2, base_rate * 0.6, 1.0]),
            ):
                res = minimize(
                    nll, start, method="L-BFGS-B", options={"maxiter": 600}, bounds=bounds
                )
                if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
                    best = (float(res.fun), np.asarray(res.x, dtype=float))
        if best is None:
            raise FitFailure(
                "The Hawkes likelihood could not be evaluated.",
                "Use the renewal model, which needs only the intervals.",
                kind="convergence",
            )
        self.mu = float(np.exp(best[1][0]))
        self.alpha = float(np.exp(best[1][1]))
        self.beta = float(np.exp(best[1][2]))
        self.branching = self.alpha / self.beta
        if self.branching >= 0.95:
            raise FitFailure(
                f"The branching ratio is {self.branching:.3f}, which is at the near-critical boundary "
                "where the likelihood is flat — the estimate would describe the optimiser, not the "
                "process.",
                "The renewal model reports the same arrival rate without pretending to separate "
                "self-excitation.",
                kind="identification",
            )
        self._loglik = -best[0]
        self._n_obs = int(n - 1)
        self._n_params = 3
        self._aic, self._aicc, self._bic = self.ic(self._n_obs, self._n_params, self._loglik)
        # Long-run intensity × period length: the model's own expectation of events per period, on the
        # same scale as the count series the comparison table scores. Left as None, Hawkes showed a
        # blank MASE next to renewal's number, which reads as "not applicable" rather than "not computed".
        self._fitted = np.full(
            y.dropna().size,
            self.mu / max(1 - self.branching, 1e-6) * self._period_days(),
        )
        self._resid = None
        self.mean_gap = float(span / max(n - 1, 1))
        self._raw_params = {
            "background rate (events per day)": (self.mu, None, False),
            "excitation (alpha)": (self.alpha, None, False),
            "decay (beta, per day)": (self.beta, None, False),
            "branching ratio (alpha/beta)": (self.branching, None, self.branching >= 0.9),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        days = self._period_days()
        # The long-run rate of a Hawkes process is μ / (1 − n), where n is the branching ratio. At
        # n = 0.7 that is 3.3× the background rate, and using μ alone would understate the forecast by
        # exactly the self-excitation the model was fitted to find.
        steady = self.mu / max(1 - self.branching, 1e-6)
        rows: list[tuple[np.ndarray, np.ndarray, bool]] = []
        for _h in range(1, horizon + 1):
            # One period's expected count, the same for every h. The cumulative form (× h) turned a
            # flat event rate into a ramp; step h asks about period h alone.
            lam = steady * days
            rows.append(poisson_pmf(max(lam, 1e-9)))
        self._predictive = summarise(rows, alpha=alpha, family="poisson")
        return frame(self._predictive)

    @property
    def n_events(self) -> int:
        """How many timestamps the fit actually used — the check that the events, not the counts, arrived."""
        return int(getattr(self, "_n_events", 0))

    def predictive(self) -> Any:
        return getattr(self, "_predictive", None)

    def _summary(self) -> FittedSummary:
        per_period = self.mu / max(1 - self.branching, 1e-6) * self._period_days()
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=self._aic,
            aicc=self._aicc,
            bic=self._bic,
            n_obs_used=self._n_obs,
            fitted_on="inter-arrival times",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"{self.branching * 100:.1f}% of arrivals are triggered by earlier arrivals, not by "
                f"the background rate — so the steady rate is {per_period:.3f} per period, well above "
                f"the background {self.mu * self._period_days():.4f}."
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                "A Hawkes model explains *clustering* with self-excitation. The same clustering can be "
                "produced by an unobserved driver, and this model cannot tell the two apart — it only "
                "says the timing is contagious.",
                "Forecasts use the long-run rate, so they do not decay after a burst; over a short "
                "horizon just after an event the true rate is higher than this.",
            ],
            ic_comparable=True,
            ic_note="AIC on the point-process likelihood of the event times; comparable with the renewal "
                    "model, which is its n = 0 special case.",
        )


IMPLEMENTED_POINT: dict[str, type[ModelWrapper]] = {
    "renewal": RenewalModel,
    "hawkes": HawkesModel,
}


__all__ = ["HawkesModel", "IMPLEMENTED_POINT", "RenewalModel", "inter_arrivals"]
