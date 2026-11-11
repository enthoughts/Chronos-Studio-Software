"""
ts/models/counts.py — Poisson, negative binomial and hurdle models (§5.8 C).

This is the branch most tools get wrong, and the failure is always the same shape: a count series is
handed to an ARIMA model, MAPE is reported, and the intervals ignore the fact that the outcome is a
non-negative integer. Three things here exist specifically to prevent that:

1. **The dispersion check runs before the model is offered.** Variance/mean ≈ 1 is the Poisson
   assumption. Above ~1.5 the honest model is the negative binomial, because a Poisson interval on
   overdispersed data is too narrow by a factor the user never sees.
2. **The predictive distribution is discrete.** `predict()` returns the mean and a *quantile-based*
   interval computed by inverting the PMF, not mean ± 1.96·σ. For small counts those two answers
   differ visibly (the normal interval can include impossible negative values), and the interval is
   what a planner acts on.
3. **MAPE is refused, not computed.** See `diagnostics.py`: the refusal is in the output.

All three are fitted as GLMs with a log link, a linear trend and harmonic seasonal terms — the
"count + trend + seasonality" row of the registry — which is the form that actually forecasts rather
than merely describing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper
from app.ts.models.predictive import frame, mixture_pmf, negbin_pmf, poisson_pmf, summarise


def _design(n: int, period: int | None, n_harmonics: int = 2) -> np.ndarray:
    """
    Intercept, a scaled time index, and harmonic terms at the seasonal period.

    The time index is scaled to [−1, 1] rather than 1..n: a raw index makes the intercept and slope
    almost perfectly collinear at large n, which turns a stable fit into a converged-looking one with
    a wild slope.
    """
    t = np.linspace(-1.0, 1.0, n) if n > 1 else np.zeros(n)
    cols = [np.ones(n), t]
    if period and period >= 2:
        for k in range(1, n_harmonics + 1):
            ang = 2 * np.pi * k * np.arange(n) / period
            cols.append(np.sin(ang))
            cols.append(np.cos(ang))
    return np.column_stack(cols)


def _future_design(n: int, horizon: int, period: int | None, n_harmonics: int = 2) -> np.ndarray:
    full = _design(n + horizon, period, n_harmonics)
    return full[n:]


class _CountBase(ModelWrapper):
    family = ModelFamily.COUNT

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._harmonics = 2

    def _check(self, y: pd.Series) -> np.ndarray:
        yv = y.dropna().to_numpy(dtype=float)
        if yv.size < 20:
            raise FitFailure(
                f"{self.display_name} needs at least 20 observations and there are {yv.size}.",
                "Use a baseline benchmark.", kind="length",
            )
        if np.any(yv < 0):
            raise FitFailure(
                f"The series contains {int((yv < 0).sum())} negative values, and a count cannot be "
                "negative.",
                "This series is continuous; the ARIMA/ETS branch applies.", kind="positivity",
            )
        if not np.allclose(yv, np.round(yv), atol=1e-8):
            raise FitFailure(
                "The series has fractional values, so it is not a count.",
                "This series is continuous; the ARIMA/ETS branch applies.", kind="dataclass",
            )
        if yv.sum() == 0:
            raise FitFailure(
                "Every value is zero, so there is no rate to estimate.",
                "If the series is genuinely all zeros, the forecast is zero and no model is needed.",
                kind="degenerate",
            )
        return yv

    def _fit_glm(self, yv: np.ndarray, family: Any) -> Any:
        import statsmodels.api as sm

        X = _design(yv.size, self.period, self._harmonics)
        model = sm.GLM(yv, X, family=family)
        try:
            res = model.fit(maxiter=200)
        except Exception as exc:  # noqa: BLE001
            raise FitFailure(
                f"The {self.display_name} fit did not converge ({type(exc).__name__}: {exc}).",
                "Try the other count model, or a shorter seasonal structure.",
                kind="convergence",
            ) from exc
        if not getattr(res, "converged", True):
            raise FitFailure(
                f"The {self.display_name} optimiser stopped without converging.",
                "Report the fit as unreliable or use the other count model.",
                kind="convergence",
            )
        return res

    def _params(self, res: Any) -> dict[str, tuple[float | None, float | None, bool]]:
        names = ["intercept", "trend"]
        if self.period and self.period >= 2:
            for k in range(1, self._harmonics + 1):
                names += [f"sin{k}", f"cos{k}"]
        out: dict[str, tuple[float | None, float | None, bool]] = {}
        try:
            vals = np.asarray(res.params, dtype=float).ravel()
            ses = np.asarray(res.bse, dtype=float).ravel()
        except Exception:  # pragma: no cover
            return out
        for i, name in enumerate(names):
            if i < vals.size:
                out[name] = (float(vals[i]), float(ses[i]) if i < ses.size else None, False)
        return out

    def _ic(self, res: Any, extra: int = 0) -> tuple[float | None, float | None, float | None, float]:
        n = int(res.nobs)
        k = int(np.asarray(res.params).size) + extra
        llf = float(res.llf) if np.isfinite(getattr(res, "llf", np.nan)) else None
        aic, aicc, bic = self.ic(n, k, llf)
        return aic, aicc, bic, llf


class PoissonModel(_CountBase):
    model_id = "poisson"
    display_name = "Poisson regression"

    def _fit(self, y: pd.Series) -> None:
        import statsmodels.api as sm

        yv = self._check(y)
        mu = float(yv.mean())
        ratio = float(yv.var(ddof=1) / mu) if mu > 0 else None
        self.dispersion = ratio
        res = self._fit_glm(yv, sm.families.Poisson())
        self.result = res
        mu_hat = np.asarray(res.fittedvalues, dtype=float)
        self._fitted = mu_hat
        self._resid = yv - mu_hat  # response residuals; the PMF intervals do the rest
        self._raw_params = self._params(res)

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        X = _future_design(self.y.dropna().size, horizon, self.period, self._harmonics)
        eta = X @ np.asarray(self.result.params, dtype=float)
        lam = np.exp(eta)
        # The PMF is built here and *kept*: the interval, the CRPS in Part 15 and the PIT plot all read
        # the same array, so they cannot disagree about what the model predicted.
        self._predictive = summarise(
            [poisson_pmf(float(v)) for v in lam], alpha=alpha, family="poisson"
        )
        return frame(self._predictive)

    def predictive(self):
        return getattr(self, "_predictive", None)

    def _summary(self) -> FittedSummary:
        aic, aicc, bic, llf = self._ic(self.result)
        n = int(self.result.nobs)
        caveats: list[str] = []
        if self.dispersion is not None and self.dispersion > 1.5:
            caveats.append(
                f"Variance is {self.dispersion:.2f}× the mean, so this Poisson fit is overdispersed: "
                "the intervals are too narrow and the negative binomial is the honest model."
            )
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=int(np.asarray(self.result.params).size),
            loglik=llf, aic=aic, aicc=aicc, bic=bic,
            n_obs_used=n,
            fitted_on="levels",
            roots_ok=None,
            plain_one_liner=(
                f"Counts modelled as a rate with a trend"
                + (f" and a {self.period}-period season" if self.period else "")
                + f"; variance is {self.dispersion:.2f}× the mean"
                + (" (overdispersed — use the negative binomial)" if (self.dispersion or 1) > 1.5 else "")
                + "."
            ),
            params=self.params_table(self._raw_params),
            caveats=caveats,
            ic_comparable=True,
            ic_note="Poisson log-likelihood; comparable with the negative binomial (nested).",
        )


class NegativeBinomialModel(_CountBase):
    model_id = "neg_binomial"
    display_name = "Negative binomial (NB2)"

    def _fit(self, y: pd.Series) -> None:
        import statsmodels.api as sm

        yv = self._check(y)
        mu = float(yv.mean())
        self.dispersion = float(yv.var(ddof=1) / mu) if mu > 0 else None
        res = self._fit_glm(yv, sm.families.NegativeBinomial(alpha=1.0))
        # The GLM treats alpha as fixed; estimate it by moment matching (var = μ + αμ²) on the
        # Pearson residuals, which is the standard NB2 two-step and does not pretend to be a joint
        # MLE. The value is reported so the reader can see how much extra variance was needed.
        try:
            alpha_hat = float(res.pearson_chi2 / max(1, res.df_resid - 1)) if res.df_resid > 0 else 1.0
        except Exception:  # pragma: no cover
            alpha_hat = 1.0
        self.alpha = max(alpha_hat, 1e-6)
        self.result = res
        mu_hat = np.asarray(res.fittedvalues, dtype=float)
        self._fitted = mu_hat
        self._resid = yv - mu_hat
        raw = self._params(res)
        raw["alpha"] = (self.alpha, None, True)
        self._raw_params = raw

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        X = _future_design(self.y.dropna().size, horizon, self.period, self._harmonics)
        mu = np.exp(X @ np.asarray(self.result.params, dtype=float))
        # NB2 parameterisation: var = μ + αμ², so p = 1/(1+αμ) in numpy's (n, p) form.
        # The NB2 PMF goes through the same helper the INGARCH model uses, so the two cannot drift into
        # different parameterisations of "the same" distribution.
        self._predictive = summarise(
            [negbin_pmf(float(m), float(self.alpha)) for m in mu], alpha=alpha, family="negbin"
        )
        return frame(self._predictive)

    def predictive(self):
        return getattr(self, "_predictive", None)

    def _summary(self) -> FittedSummary:
        # +1 for α, which is estimated rather than fixed.
        aic, aicc, bic, llf = self._ic(self.result, extra=1)
        n = int(self.result.nobs)
        caveats: list[str] = []
        if self.dispersion is not None and self.dispersion <= 1.2:
            caveats.append(
                f"Variance is only {self.dispersion:.2f}× the mean, so the extra dispersion "
                "parameter is buying almost nothing over a Poisson."
            )
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=int(np.asarray(self.result.params).size) + 1,
            loglik=llf, aic=aic, aicc=aicc, bic=bic,
            n_obs_used=n,
            fitted_on="levels",
            roots_ok=None,
            plain_one_liner=(
                f"Counts modelled as a rate that is allowed to be noisier than its own mean "
                f"(α = {self.alpha:.3f}); variance is {self.dispersion:.2f}× the mean."
            ),
            params=self.params_table(self._raw_params),
            caveats=caveats,
            ic_comparable=True,
            ic_note="NB log-likelihood with α estimated; the Poisson is nested at α → 0.",
        )


class HurdleModel(_CountBase):
    """
    Two-part model: a Bernoulli gate ("did anything happen?") and a zero-truncated count for the
    size. Reported separately, because the two answers are different questions and a business often
    needs only one of them.
    """

    model_id = "zero_inflated"
    display_name = "Hurdle count"

    def _fit(self, y: pd.Series) -> None:
        import statsmodels.api as sm

        yv = self._check(y)
        n_zero = int((yv == 0).sum())
        if n_zero == 0:
            raise FitFailure(
                "There are no zeros, so there is no hurdle to model.",
                "Use the Poisson or negative binomial.", kind="degenerate",
            )
        if n_zero == yv.size:
            raise FitFailure(
                "Every value is zero, so the gate has nothing to contrast with.",
                "The forecast is zero; no model is needed.", kind="degenerate",
            )
        X = _design(yv.size, self.period, self._harmonics)
        gate_y = (yv > 0).astype(float)
        try:
            gate = sm.GLM(gate_y, X, family=sm.families.Binomial()).fit(maxiter=200)
        except Exception as exc:  # noqa: BLE001
            raise FitFailure(
                f"The hurdle gate did not converge ({type(exc).__name__}).",
                "Use the negative binomial, which handles excess zeros less sharply but always fits.",
                kind="convergence",
            ) from exc
        pos = yv > 0
        if pos.sum() < 5:
            raise FitFailure(
                f"Only {int(pos.sum())} non-zero observations — too few to model the positive part.",
                "Use the negative binomial, or treat this as intermittent demand.", kind="length",
            )
        pos_y = yv[pos]
        X_pos = X[pos]
        try:
            size = sm.GLM(pos_y, X_pos, family=sm.families.Poisson()).fit(maxiter=200)
        except Exception as exc:  # noqa: BLE001
            raise FitFailure(
                f"The zero-truncated size component did not converge ({type(exc).__name__}).",
                "Use the negative binomial.", kind="convergence",
            ) from exc

        self.gate = gate
        self.size = size
        # Two different conditional means are in play, and confusing them is the classic hurdle bug:
        #   * the Bernoulli part estimates P(y > 0);
        #   * the Poisson part, fitted only on the positive observations, estimates the mean of the
        #     *positive* counts, which is above the unconditional rate.
        # The hurdle's unconditional mean is P(y>0) × E[y | y>0], and that is what `mean` reports.
        p_hat = np.asarray(gate.fittedvalues, dtype=float)
        self._mu_positive = np.exp(X @ np.asarray(size.params, dtype=float))
        self._fitted = p_hat * self._mu_positive
        self._resid = yv - self._fitted
        self.n_zero = n_zero
        raw = {f"gate_{k}": v for k, v in self._params(gate).items()}
        raw.update({f"size_{k}": v for k, v in self._params(size).items()})
        self._raw_params = raw

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        n = self.y.dropna().size
        Xf = _future_design(n, horizon, self.period, self._harmonics)
        p = np.asarray(1.0 / (1.0 + np.exp(-(Xf @ np.asarray(self.gate.params, dtype=float)))))
        mu = np.exp(Xf @ np.asarray(self.size.params, dtype=float))
        mean = p * mu
        # Uncertainty comes from both parts: a Bernoulli gate times a count drawn from a Poisson
        # *truncated below 1*. Simulating the two-part process is the only honest way to get a
        # quantile, and 4000 draws is cheap. The +1 that a previous version added is not a
        # truncation: it shifts every draw up, which is why that version's interval sat entirely
        # above its own point forecast.
        # The exact two-part PMF: (1 − p) on {0}, plus p × a Poisson *conditioned on being ≥ 1}. The
        # earlier version simulated this, which put a Monte-Carlo wobble into a distribution that has a
        # closed form — and the tail of a truncated Poisson is exactly where simulation is thinnest.
        rows: list[tuple[np.ndarray, np.ndarray, bool]] = []
        for h in range(horizon):
            grid, pmf, trunc = poisson_pmf(float(mu[h]))
            pos_mass = float(pmf[grid >= 1].sum())
            cond = np.zeros_like(pmf)
            if pos_mass > 0:
                cond[grid >= 1] = pmf[grid >= 1] / pos_mass
            zero = np.zeros_like(pmf)
            zero[0] = 1.0
            rows.append(
                mixture_pmf(
                    [(grid, zero, 1 - float(p[h])), (grid, cond, float(p[h]))]
                )
            )
        self._predictive = summarise(rows, alpha=alpha, family="hurdle")
        return frame(self._predictive)

    def predictive(self):
        return getattr(self, "_predictive", None)

    def _summary(self) -> FittedSummary:
        n = int(self.gate.nobs)
        k = int(np.asarray(self.gate.params).size + np.asarray(self.size.params).size)
        llf = float(self.gate.llf + self.size.llf)
        aic, aicc, bic = self.ic(n, k, llf)
        share = self.n_zero / n
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=k,
            loglik=llf, aic=aic, aicc=aicc, bic=bic,
            n_obs_used=n,
            fitted_on="levels",
            roots_ok=None,
            plain_one_liner=(
                f"Two questions modelled separately: whether an event happens "
                f"({share * 100:.0f}% of periods are zero) and how large it is when it does."
            ),
            params=self.params_table(self._raw_params),
            caveats=[
                "The two parts are estimated separately, so the combined AIC is indicative rather "
                "than exact — the likelihood is not jointly maximised."
            ],
            ic_comparable=False,
            ic_note="Component-wise likelihood; not directly comparable with the single-equation "
                   "count models.",
        )


IMPLEMENTED_COUNT: dict[str, type[ModelWrapper]] = {
    "poisson": PoissonModel,
    "neg_binomial": NegativeBinomialModel,
    "zero_inflated": HurdleModel,
}


__all__ = [
    "HurdleModel",
    "IMPLEMENTED_COUNT",
    "NegativeBinomialModel",
    "PoissonModel",
]
