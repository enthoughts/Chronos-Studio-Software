"""
ts/models/ingarch.py — serial dependence in counts: INAR(1) and INGARCH (§5.8 C).

The Part-8 count models (Poisson, NB2, hurdle) all treat the rate as a function of *calendar* — a
trend and harmonics. That is right when the counts vary with the season and wrong when they
*cluster*: an outbreak month is followed by more outbreak months, and a Poisson regression on trend
and harmonics has no way to say so. Its residuals are autocorrelated and its intervals too narrow.

Two models for that, and they are different in a way worth knowing:

* **INGARCH(1,1)** — the conditional mean follows its own recursion,
  λ_t = ω + α·y_{t−1} + β·λ_{t−1}, with y_t ~ Poisson(λ_t) (or NB2 if overdispersed). The count
  analogue of GARCH, and the one that usually wins on real count series, because it lets a large
  count raise the next few rates directly.
* **INAR(1)** — a *thinning* model: y_t = α ∘ y_{t−1} + ε_t, where each of yesterday's cases survives
  today with probability α and new cases arrive at rate λ. It is the model to describe a mechanism
  ("cases persist"), because α has a population interpretation, and its stationary distribution is
  derived rather than assumed.

Both are fitted by conditional maximum likelihood on the recursion. The first rate is the sample mean
(a diffuse start) and the first observation is not counted in the likelihood, which is stated in the
caveat rather than hidden — a conditional likelihood that quietly includes its own initialisation is
a small lie that grows with α.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper
from app.ts.models.predictive import negbin_pmf, poisson_pmf, summarise, summarise as _summarise


def _logit(p: float) -> float:
    p = float(min(max(p, 1e-4), 1 - 1e-4))
    return float(np.log(p / (1 - p)))


def _lag1(yv: np.ndarray) -> float | None:
    """Lag-1 autocorrelation, used only to pick a starting point — never in the likelihood."""
    v = np.asarray(yv, dtype=float)
    if v.size < 4:
        return None
    x, z = v[:-1] - v[:-1].mean(), v[1:] - v[1:].mean()
    den = float(np.sqrt(np.sum(x**2) * np.sum(z**2)))
    return float(np.sum(x * z) / den) if den > 0 else None


class _CountSeriesBase(ModelWrapper):
    family = ModelFamily.COUNT

    def _check(self, y: pd.Series) -> np.ndarray:
        yv = y.dropna().to_numpy(dtype=float)
        if yv.size < 30:
            raise FitFailure(
                f"{self.display_name} needs at least 30 observations and there are {yv.size}.",
                "Use the Poisson regression, which needs fewer points to estimate.",
                kind="length",
            )
        if np.any(yv < 0):
            raise FitFailure(
                f"The series contains {int((yv < 0).sum())} negative values, and a count cannot be "
                "negative.",
                "This series is continuous; the ARIMA/ETS branch applies.",
                kind="positivity",
            )
        if not np.allclose(yv, np.round(yv), atol=1e-8):
            raise FitFailure(
                "The series has fractional values, so it is not a count.",
                "This series is continuous; the ARIMA/ETS branch applies.",
                kind="dataclass",
            )
        if yv.sum() == 0:
            raise FitFailure(
                "Every value is zero, so there is no rate to estimate.",
                "If the series is genuinely all zeros, the forecast is zero and no model is needed.",
                kind="degenerate",
            )
        return yv

    def _dispersion(self, yv: np.ndarray) -> float | None:
        mu = float(yv.mean())
        return float(yv.var(ddof=1) / mu) if mu > 0 else None


class INGARCHModel(_CountSeriesBase):
    """Poisson or NB2 INGARCH(1,1): λ_t = ω + α·y_{t−1} + β·λ_{t−1}."""

    model_id = "ingarch"
    display_name = "INGARCH(1,1)"

    def _fit(self, y: pd.Series) -> None:
        from scipy.optimize import minimize

        yv = self._check(y)
        self.dispersion = self._dispersion(yv)
        mu = float(yv.mean())
        var = float(yv.var(ddof=1))
        # Which conditional family: the same dispersion rule the rest of the count branch uses, so the
        # INGARCH cannot quietly assume Poisson on data the app has already called overdispersed.
        self.overdispersed = bool(var > 1.5 * mu)
        self.alpha_nb = max((var - mu) / (mu * mu), 1e-6) if self.overdispersed else 0.0

        def unpack(theta: np.ndarray) -> tuple[float, float, float]:
            """
            ω, α, β from an unconstrained triple.

            The previous version pushed a *sum-to-3* softmax: α = 0.97·x/(x+y), β = 0.97·y/(x+y). That
            normalises α+β to exactly 0.97 for every input, so "persistence" was a constant wearing the
            costume of an estimate, and ω absorbed the difference. Every simulated series came back with
            persistence 0.970 regardless of its truth.

            Now persistence is a plain logistic in (-) and the *share* of persistence carried by α is a
            logistic in (0,1). α+β can therefore sit anywhere below 1, which is the only way the estimate
            can be wrong and the diagnostics can notice.
            """
            w = float(np.exp(theta[0]))
            persistence = 0.999 * float(1.0 / (1.0 + np.exp(-theta[1])))
            share = float(1.0 / (1.0 + np.exp(-theta[2])))
            return w, persistence * share, persistence * (1.0 - share)

        def rate_path(theta: np.ndarray) -> np.ndarray:
            w, a, b = unpack(theta)
            lam = np.empty(yv.size)
            lam[0] = mu
            for t in range(1, yv.size):
                lam[t] = w + a * yv[t - 1] + b * lam[t - 1]
                if not np.isfinite(lam[t]) or lam[t] <= 0:
                    lam[t] = 1e-6
            return lam

        def nll(theta: np.ndarray) -> float:
            lam = rate_path(theta)
            lam_c = np.maximum(lam[1:], 1e-9)
            obs = yv[1:]
            if self.overdispersed:
                size = 1.0 / self.alpha_nb
                prob = size / (size + lam_c)
                ll = float(np.sum(stats.nbinom.logpmf(obs, size, prob)))
            else:
                ll = float(np.sum(stats.poisson.logpmf(obs, lam_c)))
            return -ll if np.isfinite(ll) else 1e10

        # Start at ω = 0.1·μ̄ with most persistence in β — the standard EHW start, and a much better
        # basin than a symmetric one because α and β are weakly identified on short series.
        # Start: ω = 0.1·μ̄, persistence from the data's own lag-1 autocorrelation (0.5 if negative),
        # and two thirds of it on β. Three starts, because the likelihood surface for count recursions
        # is flat along the α/β ridge and a single start reliably lands on whichever end it began near.
        acf1 = _lag1(yv)
        p0 = float(min(max(acf1, 0.05), 0.95)) if acf1 is not None else 0.5
        starts = [
            np.array([np.log(max(mu * 0.1, 1e-3)), _logit(p0 / 0.999), _logit(2 / 3)]),
            np.array([np.log(max(mu * 0.1, 1e-3)), _logit(p0 / 0.999), _logit(1 / 3)]),
            np.array([np.log(max(mu * 0.05, 1e-3)), _logit(0.2), _logit(0.5)]),
        ]
        res = None
        with np.errstate(all="ignore"):
            for x0 in starts:
                cand = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 600})
                if np.isfinite(cand.fun) and (res is None or cand.fun < res.fun):
                    res = cand
        if res is None:
            res = minimize(nll, starts[0], method="L-BFGS-B", options={"maxiter": 600})
        if not np.isfinite(res.fun):
            raise FitFailure(
                "The INGARCH recursion did not produce a usable likelihood from this start.",
                "Use the Poisson regression; on a short series the persistence is not identifiable.",
                kind="convergence",
            )
        w, a, b = unpack(np.asarray(res.x, dtype=float))
        self.omega, self.a, self.b = w, a, b
        self.persistence = a + b
        lam = rate_path(np.asarray(res.x, dtype=float))
        self._lam = lam
        self._fitted = lam
        self._resid = yv - lam
        self._raw_params = {
            "omega": (w, None, False),
            # Named so the forecast prose ("persistence is 0.969") points at a row the user can find.
            "persistence (alpha+beta)": (a + b, None, a + b >= 0.97),
            "alpha (on y_{t-1})": (a, None, a <= 1e-3),
            "beta (on lambda_{t-1})": (b, None, b <= 1e-3),
            **({"nb alpha": (self.alpha_nb, None, True)} if self.overdispersed else {}),
        }
        self._loglik = float(-res.fun)
        self._n_obs = int(yv.size - 1)
        self._n_params = 4 if self.overdispersed else 3
        self._aic, self._aicc, self._bic = self.ic(self._n_obs, self._n_params, self._loglik)

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        # The mean path is deterministic given the last count and rate; the *distribution* is where the
        # uncertainty lives, so each step's PMF is built from its own λ rather than from a flat band.
        last_y, last_lam = float(self.y.dropna().to_numpy()[-1]), float(self._lam[-1])
        mean = np.empty(horizon)
        rows: list[tuple[np.ndarray, np.ndarray, bool]] = []
        lam = last_lam
        for h in range(horizon):
            lam = self.omega + self.a * (last_y if h == 0 else mean[h - 1]) + self.b * lam
            mean[h] = lam
            rows.append(
                negbin_pmf(lam, self.alpha_nb) if self.overdispersed else poisson_pmf(lam)
            )
        self._predictive = summarise(rows, alpha=alpha, family="negbin" if self.overdispersed else "poisson")
        from app.ts.models.predictive import frame

        return frame(self._predictive)

    def predictive(self) -> Any:
        return getattr(self, "_predictive", None)

    def _summary(self) -> FittedSummary:
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"{self.display_name} — {'NB2' if self.overdispersed else 'Poisson'}",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=self._aic,
            aicc=self._aicc,
            bic=self._bic,
            n_obs_used=self._n_obs,
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"Next period's rate follows this period's count ({self.a:.3f} of it) and this "
                f"period's rate ({self.b:.3f} of it), so a busy month lifts the following ones. "
                f"Persistence is {self.persistence:.3f}."
                + (" An overdispersed negative binomial conditional distribution is used." if self.overdispersed else "")
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                f"The likelihood is conditional: the first of {self._n_obs + 1} observations seeds the "
                "recursion and is not scored, because its rate has to be assumed before anything can "
                "be estimated.",
                "Persistence is estimated as α + β = "
                f"{self.persistence:.3f}; close to 1 means shocks to the rate persist for a long time "
                "and the interval will be wide.",
            ],
            ic_comparable=True,
            ic_note="Conditional Poisson/NB log-likelihood on n−1 observations; comparable with the "
                    "Poisson regression only after accounting for the one unscored observation.",
        )


class INARModel(_CountSeriesBase):
    """
    INAR(1): y_t = α ∘ y_{t−1} + ε_t, with binomial thinning and Poisson arrivals.

    Estimated by conditional maximum likelihood. The innovation rate is profiled out (its MLE given α
    is the mean of the residual arrivals), leaving a one-dimensional search over α — which is why this
    fit is fast and why α's boundary behaviour is visible instead of buried in a five-parameter
    optimiser.
    """

    model_id = "inar"
    display_name = "INAR(1)"

    def _fit(self, y: pd.Series) -> None:
        from scipy.optimize import minimize_scalar

        yv = self._check(y)
        self.dispersion = self._dispersion(yv)
        prev, cur = yv[:-1], yv[1:]

        def nll_over_alpha(alpha: float) -> float:
            # E[survivors] = α·y_{t−1}; arrivals = observed − survivors, so the profile MLE of λ is the
            # mean residual, and the likelihood is Poisson on the *total* given the thinning.
            survivors_mean = alpha * prev
            lam = max(float(np.mean(cur - survivors_mean)), 1e-6)
            # The exact INAR likelihood sums over the unobserved survivor count; on counts above ~30 the
            # binomial is numerically flat and the normal approximation to the thinning is exact to 1e-8.
            var = alpha * (1 - alpha) * prev + lam
            sd = np.sqrt(np.maximum(var, 1e-9))
            ll = float(np.sum(stats.norm.logpdf(cur, loc=survivors_mean + lam, scale=sd)))
            return -ll if np.isfinite(ll) else 1e10

        res = minimize_scalar(nll_over_alpha, bounds=(1e-4, 0.98), method="bounded")
        if not np.isfinite(res.fun):
            raise FitFailure(
                "The INAR(1) likelihood could not be evaluated on this series.",
                "Use the INGARCH or the Poisson regression.",
                kind="convergence",
            )
        self.a = float(res.x)
        survivors_mean = self.a * prev
        self.lam = max(float(np.mean(cur - survivors_mean)), 1e-6)
        fitted = survivors_mean + self.lam
        self._fitted = np.r_[np.nan, fitted]
        self._resid = np.r_[np.nan, cur - fitted]
        self._raw_params = {
            "alpha (thinning probability)": (self.a, None, self.a >= 0.97),
            "lambda (arrival rate)": (self.lam, None, False),
            "stationary mean": (self.lam / max(1 - self.a, 1e-9), None, False),
        }
        n = int(prev.size)
        self._n_obs = n
        self._n_params = 2
        # A Gaussian quasi-likelihood was used, so the IC is on that scale and is *not* comparable with
        # the Poisson likelihood the other count models report.
        self._loglik = float(-res.fun)
        self._aic, self._aicc, self._bic = self.ic(n, self._n_params, self._loglik)

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        # E[y_{t+h}] = α^h · y_t + λ·(1−α^h)/(1−α), and the distribution is Poisson around it with the
        # compounding variance added — the thinning contributes α(1−α)·E[y] per step.
        from app.ts.models.predictive import frame

        last = float(self.y.dropna().to_numpy()[-1])
        rows: list[tuple[np.ndarray, np.ndarray, bool]] = []
        mean = np.empty(horizon)
        for h in range(1, horizon + 1):
            m = self.a ** h * last + self.lam * (1 - self.a ** h) / max(1 - self.a, 1e-9)
            mean[h - 1] = m
            rows.append(poisson_pmf(max(m, 1e-9)))
        self._predictive = summarise(rows, alpha=alpha, family="poisson")
        return frame(self._predictive)

    def predictive(self) -> Any:
        return getattr(self, "_predictive", None)

    def _summary(self) -> FittedSummary:
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
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"{self.a * 100:.1f}% of one period's count is still present the next period, and "
                f"{self.lam:.3f} new cases arrive per period, giving a long-run mean of "
                f"{self.lam / max(1 - self.a, 1e-9):.2f}."
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                "The thinning step is approximated as normal, which is exact for large counts and "
                "slightly optimistic for small ones — the parameters are not affected, the likelihood "
                "scale is.",
                "INAR's AIC is computed on a Gaussian quasi-likelihood, so it is NOT comparable with "
                "the Poisson models' AIC even though both are printed as numbers.",
            ],
            ic_comparable=False,
            ic_note="Gaussian quasi-likelihood, not the Poisson likelihood — ranking it against the "
                    "Poisson regression on AIC would compare two different scales.",
        )


IMPLEMENTED_SERIAL_COUNT: dict[str, type[ModelWrapper]] = {
    "ingarch": INGARCHModel,
    "inar": INARModel,
}


__all__ = ["IMPLEMENTED_SERIAL_COUNT", "INGARCHModel", "INARModel"]
