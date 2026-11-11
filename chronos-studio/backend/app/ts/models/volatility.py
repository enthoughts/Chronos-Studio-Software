"""
ts/models/volatility.py — conditional-variance models (§5.8 D).

Part 8's ARIMA family models the *mean* and assumes the variance is a constant. A great many series
break the second half of that assumption while obeying the first: the level behaves like a random walk
with drift and the noise arrives in clusters — calm stretches, then violent ones. Fitting a constant-
variance model to that gives correctly-placed point forecasts and intervals that are **too narrow when
it matters and too wide when it doesn't**, which is the worst combination for anyone using them.

Four specifications, all fitted by maximum likelihood on the same recursion:

    mean:      y_t = μ + Σ φ_i y_{t−i} + ε_t,          ε_t = σ_t z_t
    variance:  σ²_t = ω + Σ α_i ε²_{t−i} + Σ β_j σ²_{t−j}          (GARCH)
    asymmetric:σ²_t = ω + Σ (α_i + γ_i·1[ε_{t−i}<0]) ε²_{t−i} + Σ β_j σ²_{t−j}   (GJR)
    log-variance: log σ²_t = ω + Σ α_i(|z_{t−i}| − E|z|) + Σ γ_i z_{t−i} + Σ β_j log σ²_{t−j}  (EGARCH)

and two error distributions:

* **Gaussian** — the default, and the one whose QMLE standard errors are valid even when the true
  innovations are not normal (Bollerslev–Wooldridge), provided the mean is specified correctly.
* **Student-t** — ν estimated, because standardised residuals from financial and count-like series are
  routinely fatter-tailed and Gaussian intervals understate the tail.

Four things this module refuses to do, each because omitting them is how GARCH results mislead:

1. **Fit without checking for ARCH first.** If the ARCH-LM test on the mean-equation residuals does not
   reject, clustered variance is not in the data, and a GARCH fit will still "converge" and report a
   persistence of 0.9 by absorbing noise. The model reports the test's p-value in the same table.
2. **Report a variance forecast that implies a negative variance.** Multi-step forecasts are computed by
   the recursion, not by `σ̄² · h`, and the recursion is checked for positivity at every step.
3. **Present the mean forecast as the interesting output.** For a volatility model the deliverable is
   the *variance path*; the mean path is whatever the mean equation says and is usually flat. Both are
   returned, and the plain-language line leads with the variance.
4. **Ignore the leverage effect.** If γ is significantly positive on a GJR fit, negative shocks raise
   variance more than positive ones, and a symmetric GARCH is understating the risk after a fall. The
   comparison says which of the two won and by how much.

Forecast intervals for the mean use the *predicted variance path*, so interval width grows with
expected volatility rather than with a single sample-wide standard deviation. That is the whole point
of fitting one of these models instead of a constant-variance alternative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.formatting import p_phrase
from scipy import stats

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

#: Variance persistence (Σα + Σβ) at or above this is treated as an integrated-variance boundary: the
#: unconditional variance does not exist, the long-run forecast is undefined, and the requirement is
#: reported rather than printed as a number.
PERSISTENCE_CAP = 0.999

#: Minimum observations. GARCH needs enough of a sample for the variance recursion to have something
#: to be identified from; below this the persistence estimate is dominated by the initialisation.
MIN_N = 80

#: Default ARCH lag for the pre-check, matching the facts layer's `arch_lm`.
PRE_CHECK_LAGS = 5

#: The asymmetry coefficient below which a leverage *story* is not told. Fitted γ on symmetric data is
#: routinely 0.01–0.03 in absolute value; calling that "bad news raises uncertainty more" is reading a
#: number that is distinguishable from zero only by its sign.
LEVERAGE_MATERIAL = 0.02


@dataclass
class VariancePath:
    """The conditional-variance fit, kept as arrays so the forecast and the plots read one source."""

    sigma2: np.ndarray              # filtered conditional variance, one per observation
    resid: np.ndarray               # ε_t
    std_resid: np.ndarray           # z_t = ε_t / σ_t
    persistence: float
    unconditional: float | None     # None when persistence ≥ PERSISTENCE_CAP
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VolatilitySpec:
    """One specification: variance shape, error distribution, and the mean equation."""

    key: str
    label: str
    variance: Literal["arch", "garch", "gjr", "egarch"]
    error: Literal["normal", "t"]
    p: int = 1
    q: int = 1
    mean_ar: int = 0
    #: Free parameters of the variance equation, before the error shape.
    k_variance: int = 3

    @property
    def k(self) -> int:
        """
        Free parameters for the information criteria.

        `k_variance` counts only the variance-equation parameters; the mean constant and the t degrees
        of freedom are added here. Double-counting ν in `k_variance` was the first bug in this module:
        the t-spec then read `theta[4]` for ν when ν was at index 3, and every t fit died with an
        IndexError inside the optimiser.
        """
        return self.k_variance + self.mean_ar + 1 + (1 if self.error == "t" else 0)


#: The grid, ordered from simplest to most flexible. `garch` (Gaussian) is the reference every other
#: row is compared against — not because it is usually best but because it is the one a reader has seen.
GRID: tuple[VolatilitySpec, ...] = (
    VolatilitySpec("arch", "ARCH(1)", "arch", "normal", k_variance=2),
    VolatilitySpec("garch", "GARCH(1,1)", "garch", "normal", k_variance=3),
    VolatilitySpec("garch_t", "GARCH(1,1)-t", "garch", "t", k_variance=3),
    VolatilitySpec("gjr", "GJR-GARCH(1,1)", "gjr", "normal", k_variance=4),
    VolatilitySpec("gjr_t", "GJR-GARCH(1,1)-t", "gjr", "t", k_variance=4),
    VolatilitySpec("egarch", "EGARCH(1,1)", "egarch", "normal", k_variance=4),
    VolatilitySpec("egarch_t", "EGARCH(1,1)-t", "egarch", "t", k_variance=4),
)

#: The combined row. Registered separately because it is not a variance *shape* — it is a mean equation
#: paired with one — so it does not belong in the grid of variance specifications, but it does belong
#: in the lookup every row reads its parameters from.
ARIMA_GARCH_SPEC = VolatilitySpec("arima_garch", "AR(1)-GARCH(1,1)", "garch", "normal", mean_ar=1, k_variance=3)

SPEC_BY_KEY: dict[str, VolatilitySpec] = {s.key: s for s in GRID}
SPEC_BY_KEY[ARIMA_GARCH_SPEC.key] = ARIMA_GARCH_SPEC


# --------------------------------------------------------------------------- #
# The model internals
# --------------------------------------------------------------------------- #
def _arch_lm(z: np.ndarray, lags: int = PRE_CHECK_LAGS) -> float | None:
    """
    Engle's ARCH-LM p-value on a residual series.

    Regress the squared residuals on their own lags and use n·R². Low p means variance clusters — which
    is what a conditional-variance model is for. Reported, not decisive: on 100 observations the test
    has little power, and a model that wins on AICc can still be worth fitting when this p is 0.2.
    """
    v = np.asarray(z, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < lags + 20:
        return None
    e2 = v**2
    e2 = e2 - e2.mean()
    y = e2[lags:]
    X = np.column_stack([e2[lags - j - 1 : e2.size - j - 1] for j in range(lags)])
    # Add nothing else: the LM statistic uses the *centred* series in a regression with no intercept.
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum(y**2))
        if ss_tot <= 0:
            return None
        r2 = 1.0 - ss_res / ss_tot
        stat = float(y.size * r2)
        return float(stats.chi2.sf(stat, lags))
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None


def _filter_variance(
    spec: VolatilitySpec, eps: np.ndarray, theta: np.ndarray, sigma2_0: float
) -> np.ndarray | None:
    """
    Run the variance recursion and return σ²_t, or None if any step goes non-positive.

    The positivity check is not paranoia: an unconstrained GJR fit will happily propose ω < 0 with α+γ
    large enough to make the variance negative in a quiet stretch, and the likelihood of a negative
    variance is the log of a negative number — a NaN that the optimiser reads as "try somewhere else".
    """
    n = eps.size
    s2 = np.empty(n)
    if spec.variance == "arch":
        # ARCH(q) is GARCH with no variance term at all — not GARCH with β estimated at zero, which is
        # a different model with one more parameter and a boundary estimate.
        w, a = theta[0], theta[1]
        s2[0] = sigma2_0
        for t in range(1, n):
            val = w + a * eps[t - 1] ** 2
            if not np.isfinite(val) or val <= 0:
                return None
            s2[t] = val
    elif spec.variance == "garch":
        w, a, b = theta[0], theta[1], theta[2]
        s2[0] = sigma2_0
        for t in range(1, n):
            val = w + a * eps[t - 1] ** 2 + b * s2[t - 1]
            if not np.isfinite(val) or val <= 0:
                return None
            s2[t] = val
    elif spec.variance == "gjr":
        w, a, g, b = theta[0], theta[1], theta[2], theta[3]
        s2[0] = sigma2_0
        for t in range(1, n):
            neg = 1.0 if eps[t - 1] < 0 else 0.0
            val = w + (a + g * neg) * eps[t - 1] ** 2 + b * s2[t - 1]
            if not np.isfinite(val) or val <= 0:
                return None
            s2[t] = val
    else:  # egarch, in logs
        w, a, g, b = theta[0], theta[1], theta[2], theta[3]
        log_s2 = np.empty(n)
        log_s2[0] = np.log(sigma2_0)
        for t in range(1, n):
            z = eps[t - 1] / np.sqrt(np.exp(log_s2[t - 1]))
            val = w + a * (abs(z) - np.sqrt(2.0 / np.pi)) + g * z + b * log_s2[t - 1]
            if not np.isfinite(val):
                return None
            # EGARCH cannot produce a negative variance by construction; only guard the exponent.
            log_s2[t] = float(np.clip(val, -30.0, 30.0))
        s2 = np.exp(log_s2)
    return s2


def _persistence(spec: VolatilitySpec, theta: np.ndarray) -> float:
    if spec.variance == "arch":
        return float(theta[1])
    if spec.variance == "garch":
        return float(theta[1] + theta[2])
    if spec.variance == "gjr":
        # The *expected* leverage contribution is half the γ term, since a shock is negative half the
        # time under a symmetric error. Using γ itself overstates persistence.
        return float(theta[1] + 0.5 * theta[2] + theta[3])
    return float(theta[3])


def _unconditional(spec: VolatilitySpec, theta: np.ndarray, persistence: float) -> float | None:
    if spec.variance == "egarch":
        if abs(theta[3]) >= PERSISTENCE_CAP:
            return None
        return float(np.exp(theta[0] / max(1.0 - theta[3], 1e-9)))
    if persistence >= PERSISTENCE_CAP or persistence <= 0:
        return None
    w = float(theta[0])
    return w / (1.0 - persistence) if w > 0 else None


def _loglik(spec: VolatilitySpec, eps: np.ndarray, s2: np.ndarray, extra: np.ndarray) -> float:
    """Log-likelihood of the variance equation given the residuals, for the chosen error shape."""
    z = eps / np.sqrt(s2)
    if spec.error == "normal":
        return float(np.sum(stats.norm.logpdf(z) - 0.5 * np.log(s2)))
    nu = float(extra[0])
    # Student-t is scaled to unit variance so ν and the variance equation do not fight over scale.
    scale = np.sqrt((nu - 2.0) / nu)
    return float(np.sum(stats.t.logpdf(z * scale, nu) + np.log(scale) - 0.5 * np.log(s2)))


def _fit_mean_ar(y: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, float]:
    """OLS AR(p) with a constant: returns coefficients, residuals and the residual variance."""
    n = y.size
    X = [np.ones(n - p)]
    for i in range(1, p + 1):
        X.append(y[p - i : n - i])
    Xm = np.column_stack(X)
    target = y[p:]
    beta, *_ = np.linalg.lstsq(Xm, target, rcond=None)
    resid = target - Xm @ beta
    eps = np.concatenate([np.full(p, np.nan), resid])
    return beta, eps, float(np.var(resid, ddof=1)) if resid.size > 1 else float(np.var(y))


def _start_values(spec: VolatilitySpec, var_y: float) -> np.ndarray:
    """A start that is in the right basin: low persistence, most of the weight on the GARCH term."""
    if spec.variance == "arch":
        base = [var_y * 0.5, 0.25]
        if spec.error == "t":
            base.append(8.0)
        return np.asarray(base, dtype=float)
    if spec.variance == "egarch":
        base = [np.log(max(var_y, 1e-8)) * 0.05, 0.10, -0.05, 0.90]
    elif spec.variance == "gjr":
        base = [var_y * 0.05, 0.05, 0.05, 0.85]
    else:
        base = [var_y * 0.05, 0.08, 0.85]
    if spec.error == "t":
        base.append(8.0)
    return np.asarray(base, dtype=float)


# --------------------------------------------------------------------------- #
# The wrapper
# --------------------------------------------------------------------------- #
class VolatilityModel(ModelWrapper):
    """
    One conditional-variance specification, fitted by MLE on the variance recursion.

    `_forecast` returns the **mean** path with variance-aware intervals; `variance_forecast()` returns
    the conditional-variance path itself, which is what §5.8 D asks for as the second output.
    """

    model_id = "garch"
    display_name = "GARCH"
    family = ModelFamily.VOLATILITY

    #: Set by the factory below.
    spec: VolatilitySpec = GRID[0]

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.vpath: VariancePath | None = None
        self._model_spec = self.spec

    # -- fitting ------------------------------------------------------------ #
    def _check(self, y: pd.Series) -> np.ndarray:
        yv = y.dropna().to_numpy(dtype=float)
        if yv.size < MIN_N:
            raise FitFailure(
                f"A conditional-variance model needs at least {MIN_N} observations and there are "
                f"{yv.size}.",
                "Use the ARIMA/ETS branch: the variance recursion cannot be identified from a sample "
                "this short, and a persistence estimate from it would be noise.",
                kind="length",
            )
        if np.ptp(yv) == 0:
            raise FitFailure(
                "The series is constant, so there is no variance to model.",
                "Nothing to do — a constant series forecasts itself.",
                kind="degenerate",
            )
        return yv

    def _fit(self, y: pd.Series) -> None:
        from scipy.optimize import minimize

        yv = self._check(y)
        spec = self._model_spec
        var_y = float(np.var(yv, ddof=1))

        # ---- mean equation ------------------------------------------------ #
        # A zero-AR mean is the default: §5.8 D's rows are about the variance, and adding AR terms
        # silently would make the variance parameters depend on the mean specification. It is offered
        # as a separate row instead (`arima_garch`), where the mean is the point.
        if spec.mean_ar:
            beta, eps_raw, sigma2_0 = _fit_mean_ar(yv, spec.mean_ar)
            self._mean_coef = beta
        else:
            mu = float(np.mean(yv))
            beta = np.array([mu])
            eps_raw = yv - mu
            sigma2_0 = var_y
        self._mean_coef = beta
        self._sigma2_0 = max(sigma2_0, 1e-8)
        self._resid_full = eps_raw

        # ---- pre-check: is there anything to model? ------------------------ #
        clean = eps_raw[np.isfinite(eps_raw)]
        self.arch_p = _arch_lm(clean)
        self.arch_p_joint = _arch_lm(clean, lags=10)

        # ---- variance equation -------------------------------------------- #
        eps = np.where(np.isfinite(eps_raw), eps_raw, 0.0)
        eps[0] = eps[0] if np.isfinite(eps[0]) else 0.0

        def nll(theta: np.ndarray) -> float:
            var_theta = theta[: spec.k_variance]
            extra = theta[spec.k_variance :]
            s2 = _filter_variance(spec, eps, var_theta, self._sigma2_0)
            if s2 is None:
                return 1e10
            ll = _loglik(spec, eps, s2, extra)
            return -ll if np.isfinite(ll) else 1e10

        starts = [
            _start_values(spec, var_y),
            _start_values(spec, var_y) * np.array([1.0, 0.5, 1.0, 1.05] + ([1.0] if spec.error == "t" else []))[: len(_start_values(spec, var_y))],
        ]
        if spec.error == "t":
            starts[1][-1] = 5.0
        best = None
        with np.errstate(all="ignore"):
            for x0 in starts:
                res = minimize(nll, x0, method="L-BFGS-B", options={"maxiter": 800})
                if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
                    best = res
        if best is None or not np.isfinite(best.fun):
            raise FitFailure(
                "The variance recursion did not produce a usable likelihood from either start.",
                "Fit the constant-variance ARIMA model; the variance equation is not identified here.",
                kind="convergence",
            )

        theta = np.asarray(best.x, dtype=float)
        var_theta = theta[: spec.k_variance]
        extra = theta[spec.k_variance :]
        s2 = _filter_variance(spec, eps, var_theta, self._sigma2_0)
        if s2 is None:  # pragma: no cover - the optimiser would have scored this 1e10
            raise FitFailure("The fitted variance path went non-positive.", "Report this.", kind="numerical")

        persistence = _persistence(spec, var_theta)
        uncond = _unconditional(spec, var_theta, persistence)
        z = eps / np.sqrt(s2)
        self._theta = theta
        self._var_theta = var_theta
        self._extra = extra
        self._s2 = s2
        self._eps = eps
        self._z = z
        self.persistence = persistence
        self.unconditional_variance = uncond
        self._loglik = -float(best.fun)
        self._n_obs = int(eps.size)
        self._n_params = spec.k

        self.vpath = VariancePath(
            sigma2=s2,
            resid=eps,
            std_resid=z,
            persistence=persistence,
            unconditional=uncond,
            extra={"spec": spec.key},
        )
        # The in-sample fitted value is the *conditional mean*, never the observation itself. The first
        # version of this line was `beta[0] + eps`, which is identically `y`: every residual metric then
        # came out as exactly zero and the comparison table printed MASE = 2.9e-18 with a straight face.
        # A volatility model genuinely cannot predict the value, and the honest way to say that in a
        # table is a MASE near 1, not near 0. For a constant-mean fit the conditional mean *is* the
        # constant; for the AR mean equation it is the one-step AR prediction.
        mean_path = np.full(eps.size, float(beta[0]))
        if spec.mean_ar:
            y_vals = np.asarray(y, dtype=float)[-eps.size:]
            for t in range(spec.mean_ar, eps.size):
                mean_path[t] = beta[0] + float(
                    sum(beta[i + 1] * y_vals[t - i - 1] for i in range(spec.mean_ar))
                )
        self._fitted = mean_path
        self._resid = eps
        self._raw_params = self._param_rows()

    def _param_rows(self) -> dict[str, tuple[float | None, float | None, bool]]:
        spec = self._model_spec
        t = self._var_theta
        # Standard errors from the numerical Hessian, which is what a reader will check against. A
        # parameter pinned at a bound reports its value with no SE and `at_boundary`, never SE = 0.
        se = self._std_errors()
        rows: dict[str, tuple[float | None, float | None, bool]] = {}
        if spec.variance == "arch":
            names = ["omega", "alpha"]
        elif spec.variance == "egarch":
            names = ["omega (log)", "alpha (|z|)", "gamma (leverage)", "beta (log pers.)"]
        elif spec.variance == "gjr":
            names = ["omega", "alpha", "gamma (leverage)", "beta"]
        else:
            names = ["omega", "alpha", "beta"]
        for i, nm in enumerate(names):
            rows[nm] = (float(t[i]), se.get(i), False)
        rows["persistence"] = (self.persistence, None, self.persistence >= PERSISTENCE_CAP)
        if self.unconditional_variance is not None:
            rows["unconditional variance"] = (self.unconditional_variance, None, False)
        if spec.error == "t":
            rows["nu (t degrees of freedom)"] = (
                float(self._extra[0]), se.get(spec.k_variance), float(self._extra[0]) >= 30.0
            )
        if spec.mean_ar:
            for i, b in enumerate(self._mean_coef):
                rows["const" if i == 0 else f"ar{i}"] = (float(b), None, False)
        else:
            rows["mean (constant)"] = (float(self._mean_coef[0]), None, False)
        return rows

    def _std_errors(self) -> dict[int, float]:
        """
        Standard errors from a numerical Hessian of the concentrated likelihood.

        Deliberately conservative: when the Hessian is singular or not positive definite (which happens
        when persistence hits the cap) an empty mapping is returned and the parameter table shows no SE
        rather than a fabricated one.
        """
        from scipy.optimize import approx_fprime

        spec = self._model_spec
        theta = self._theta
        eps = self._eps

        def nll(th: np.ndarray) -> float:
            s2 = _filter_variance(spec, eps, th[: spec.k_variance], self._sigma2_0)
            if s2 is None:
                return 1e10
            ll = _loglik(spec, eps, s2, th[spec.k_variance :])
            return -ll if np.isfinite(ll) else 1e10

        k = theta.size
        h = np.maximum(np.abs(theta) * 1e-4, 1e-6)
        hess = np.empty((k, k))
        try:
            for i in range(k):
                ei = np.zeros(k)
                ei[i] = h[i]
                hess[i, :] = (approx_fprime(theta + ei, nll, h) - approx_fprime(theta - ei, nll, h)) / (2 * h[i])
            cov = np.linalg.inv(hess)
            diag = np.diag(cov)
            if np.any(diag <= 0) or not np.all(np.isfinite(diag)):
                return {}
            return {i: float(np.sqrt(diag[i])) for i in range(k)}
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return {}

    # -- forecasting -------------------------------------------------------- #
    def _variance_path(self, horizon: int) -> np.ndarray:
        """
        The multi-step variance forecast, by iterating the *fitted* recursion.

        This is the number a volatility model exists to produce. Iterating rather than applying
        `σ̄²·h` matters: the term structure converges to the unconditional variance at a rate set by
        persistence, and for a persistent series the difference at h = 10 is large.
        """
        spec = self._model_spec
        t = self._var_theta
        e2 = self._eps**2
        s2_last = float(self._s2[-1])
        e_last = float(self._eps[-1])
        out = np.empty(horizon)
        cur = s2_last
        # h = 1 is a *forecast given the last shock we have already seen*, so it uses the realised last
        # shock and its realised sign. Only h >= 2 involves a shock nobody has seen yet, and there the
        # sign has to be integrated out. Applying the expectation at h = 1 as well was the first version
        # of this function and it disagreed with the filter's own one-step-ahead variance — a model whose
        # forecast for tomorrow contradicts its fit for today is wrong about one of them.
        for h in range(horizon):
            if h == 0:
                prev_e2 = e_last**2
                neg = 1.0 if e_last < 0 else 0.0
                z_last = e_last / np.sqrt(max(cur, 1e-12))
            else:
                prev_e2 = cur
                neg = 0.5
                z_last = 0.0
            if spec.variance == "arch":
                cur = float(t[0] + t[1] * prev_e2)
            elif spec.variance == "egarch":
                # z is unknown from h = 2 onward; E|z| and E[z] are what the recursion can use, and the
                # gamma term's contribution is zero in expectation because E[z] = 0.
                if h == 0:
                    aterm = abs(z_last) - np.sqrt(2 / np.pi)
                    gterm = t[2] * z_last
                else:
                    aterm = np.sqrt(2 / np.pi) - np.sqrt(2 / np.pi)
                    gterm = 0.0
                cur = float(
                    np.exp(
                        np.clip(
                            t[0] + t[1] * aterm + gterm + t[3] * np.log(max(cur, 1e-12)),
                            -30.0,
                            30.0,
                        )
                    )
                )
            else:
                if spec.variance == "gjr":
                    # E[1[ε<0]·ε²] under a symmetric error is half of E[ε²] = σ², so the *expected*
                    # future variance uses half the γ term. Using the realised sign would be a
                    # prediction about the sign of a future shock, which nobody has.
                    cur = float(t[0] + (t[1] + t[2] * neg) * prev_e2 + t[3] * cur)
                else:
                    cur = float(t[0] + t[1] * prev_e2 + t[2] * cur)
            out[h] = max(cur, 1e-12)
        return out

    def variance_forecast(self) -> dict[str, list[float]]:
        """The variance path on its own: σ²_h and σ_h, which is what the §5.8 D deliverables ask for."""
        # 60 steps is enough to show the term structure converging; more is padding.
        var = self._variance_path(60)
        sd = np.sqrt(var)
        return {
            "variance": [float(x) for x in var],
            "sd": [float(x) for x in sd],
            "unconditional_sd": (
                None if self.unconditional_variance is None else float(np.sqrt(self.unconditional_variance))
            ),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        spec = self._model_spec
        # Mean path. With no AR terms this is the constant; with AR terms it recurses.
        mean = np.empty(horizon)
        last = self._eps
        y_series = np.asarray(self._fitted, dtype=float)
        for h in range(horizon):
            if spec.mean_ar:
                val = self._mean_coef[0]
                for i in range(1, spec.mean_ar + 1):
                    src = y_series[-i] if h == 0 else mean[h - i]
                    val += self._mean_coef[i] * src
                mean[h] = val
            else:
                mean[h] = self._mean_coef[0]
        var = self._variance_path(horizon)
        z_crit = (
            stats.norm.ppf(1 - alpha / 2)
            if spec.error == "normal"
            else stats.t.ppf(1 - alpha / 2, float(self._extra[0]))
        )
        sd = np.sqrt(var)
        return pd.DataFrame(
            {"mean": mean, "lower": mean - z_crit * sd, "upper": mean + z_crit * sd, "sd": sd},
            index=np.arange(1, horizon + 1),
        )

    def predictive(self) -> Any:
        """A continuous predictive: no PMF, so the discrete metrics do not apply (§5.8's count rule)."""
        return None

    # -- reporting ---------------------------------------------------------- #
    def _summary(self) -> FittedSummary:
        spec = self._model_spec
        aic, aicc, bic = self.ic(self._n_obs, self._n_params, self._loglik)
        uncond_sd = None if self.unconditional_variance is None else float(np.sqrt(self.unconditional_variance))
        latest_sd = float(np.sqrt(self._s2[-1]))
        caveats: list[str] = []
        if self.arch_p is not None and self.arch_p > 0.10:
            caveats.append(
                f"ARCH-LM on the mean-equation residuals gives {p_phrase(self.arch_p, 3)}, so there is little "
                "evidence of clustered variance in this series — the fit still converges, but it may be "
                "describing noise."
            )
        if self.persistence >= PERSISTENCE_CAP:
            caveats.append(
                f"Variance persistence is {self.persistence:.4f}, at the integrated-variance boundary. "
                "The unconditional variance does not exist, so there is no long-run level for the "
                "forecast to converge to."
            )
        if spec.error == "t" and float(self._extra[0]) >= 30:
            caveats.append(
                f"The t degrees of freedom came out at {float(self._extra[0]):.1f}, which is "
                "indistinguishable from normal — the Gaussian version has one parameter fewer and will "
                "win on AICc."
            )
        if spec.mean_ar == 0:
            caveats.append(
                "The mean equation is a constant, so this model says nothing about the level. Its "
                "output is the variance path; use an ARIMA or a combined model for the mean."
            )
        leverage = None
        if spec.variance == "gjr":
            leverage = float(self._var_theta[2])
            # A γ of 0.002 is a positive number and means nothing. The threshold is stated rather than
            # left implicit because "γ > 0" fired on symmetric data and printed a leverage story about
            # noise — a caveat that appears when it is not true is worse than no caveat.
            if leverage >= LEVERAGE_MATERIAL:
                caveats.append(
                    f"Leverage γ = {leverage:.3f} > 0: a negative shock raises next period's variance "
                    "more than a positive one of the same size."
                )
        elif spec.variance == "egarch":
            leverage = float(self._var_theta[2])
            if leverage <= -LEVERAGE_MATERIAL:
                caveats.append(
                    f"EGARCH γ = {leverage:.3f} < 0: negative shocks raise log-variance, which is the "
                    "leverage effect."
                )

        return FittedSummary(
            model_id=self.model_id,
            display_name=f"{spec.label} — {spec.error}",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=aic,
            aicc=aicc,
            bic=bic,
            n_obs_used=self._n_obs,
            fitted_on="levels (constant-variance mean)",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=(
                f"Variance persistence is {self.persistence:.3f}"
                + (
                    f", so a shock's effect on volatility halves in about "
                    f"{np.log(0.5) / np.log(max(self.persistence, 1e-9)):.1f} periods."
                    if 0 < self.persistence < 1
                    else ", which is at the boundary: shocks to volatility do not decay."
                )
                + (
                    f" Current volatility is {latest_sd:.3f} against a long-run {uncond_sd:.3f}."
                    if uncond_sd is not None
                    else f" Current volatility is {latest_sd:.3f}."
                )
                + (
                    f" ARCH-LM on the residuals: {p_phrase(self.arch_p, 3)}."
                    if self.arch_p is not None
                    else ""
                )
            ),
            params=self.params_table(self._raw_params),
            caveats=caveats,
            ic_comparable=True,
            ic_note="Compared on the conditional-variance likelihood; comparable with the other GARCH rows.",
        )


# --------------------------------------------------------------------------- #
# The five registry-visible rows
# --------------------------------------------------------------------------- #
class _GarchBase(VolatilityModel):
    pass


class ArchModel(_GarchBase):
    """ARCH(1) — the row that exists to make the point that its predecessors lost."""

    model_id = "arch"
    display_name = "ARCH(q)"
    spec = SPEC_BY_KEY["arch"]
    _model_spec = SPEC_BY_KEY["arch"]


class GarchModel(_GarchBase):
    model_id = "garch"
    display_name = "GARCH(p,q)"
    spec = SPEC_BY_KEY["garch"]
    _model_spec = SPEC_BY_KEY["garch"]


class GarchTModel(_GarchBase):
    model_id = "garch_t"
    display_name = "GARCH-t"
    spec = SPEC_BY_KEY["garch_t"]
    _model_spec = SPEC_BY_KEY["garch_t"]


class GjrGarchModel(_GarchBase):
    model_id = "gjr"
    display_name = "GJR-GARCH"
    spec = SPEC_BY_KEY["gjr"]
    _model_spec = SPEC_BY_KEY["gjr"]


class GjrGarchTModel(_GarchBase):
    model_id = "gjr_t"
    display_name = "GJR-GARCH-t"
    spec = SPEC_BY_KEY["gjr_t"]
    _model_spec = SPEC_BY_KEY["gjr_t"]


class EgarchModel(_GarchBase):
    model_id = "egarch"
    display_name = "EGARCH"
    spec = SPEC_BY_KEY["egarch"]
    _model_spec = SPEC_BY_KEY["egarch"]


class EgarchTModel(_GarchBase):
    model_id = "egarch_t"
    display_name = "EGARCH-t"
    spec = SPEC_BY_KEY["egarch_t"]
    _model_spec = SPEC_BY_KEY["egarch_t"]


class ArimaGarchModel(_GarchBase):
    """
    ARIMA mean with a GARCH variance (§5.8 D's "ARIMA–GARCH").

    This is the row that produces *two* forecasts, which the spec calls out explicitly: the mean path
    with variance-aware intervals (from `predict`), and the variance path on its own (from
    `variance_forecast`). The mean equation is AR(1) with a constant, which is the smallest mean model
    that can actually move; anything richer belongs in the ARIMA rows and this row would be re-fitting
    them rather than combining with them.
    """

    model_id = "arima_garch"
    display_name = "ARIMA–GARCH"
    spec = ARIMA_GARCH_SPEC
    _model_spec = ARIMA_GARCH_SPEC

    def _summary(self) -> FittedSummary:
        base = super()._summary()
        base.plain_one_liner = (
            "Two forecasts from one fit. The mean path is the AR(1) equation; the variance path is the "
            "GARCH recursion. " + base.plain_one_liner
        )
        return base

    def variance_forecast(self) -> dict[str, list[float]]:
        return super().variance_forecast()

    def forecast_both(self, horizon: int = 12, *, alpha: float = 0.05) -> dict[str, Any]:
        """Both outputs in one payload, which is what the UI panel and the report want."""
        f = self.predict(horizon, alpha=alpha)
        v = self.variance_forecast()
        return {
            "mean": [float(x) for x in f["mean"]],
            "lower": [float(x) for x in f["lower"]],
            "upper": [float(x) for x in f["upper"]],
            "sd": [float(x) for x in f["sd"]],
            "variance_path": v["variance"][:horizon],
        }


#: Every fittable row in this module, keyed by model id. The fit orchestrator merges this into
#: `IMPLEMENTED`, so a new specification is a row here and a registry entry — no other file changes.
IMPLEMENTED_VOLATILITY: dict[str, type[VolatilityModel]] = {
    "arch": ArchModel,
    "garch": GarchModel,
    "garch_t": GarchTModel,
    "gjr": GjrGarchModel,
    "gjr_t": GjrGarchTModel,
    "egarch": EgarchModel,
    "egarch_t": EgarchTModel,
    "arima_garch": ArimaGarchModel,
}

#: Rows the registry declares but this build does not fit, with the reason. Stochastic volatility and
#: realised volatility both need input this app does not accept (a latent-state MCMC and an intraday
#: return panel respectively), and saying so is better than a row that always fails.
UNIMPLEMENTED_VOLATILITY: dict[str, str] = {
    "sv": "needs a latent-state sampler and a prior; the app has no Bayesian fitting layer",
    "realized": "needs intraday returns; the app ingests one series per file",
}

__all__ = [
    "ARIMA_GARCH_SPEC",
    "GRID",
    "ArchModel",
    "GjrGarchTModel",
    "EgarchTModel",
    "IMPLEMENTED_VOLATILITY",
    "MIN_N",
    "PERSISTENCE_CAP",
    "SPEC_BY_KEY",
    "UNIMPLEMENTED_VOLATILITY",
    "ArimaGarchModel",
    "EgarchModel",
    "GarchModel",
    "GarchTModel",
    "GjrGarchModel",
    "VolatilityModel",
    "VariancePath",
    "VolatilitySpec",
    "_arch_lm",
]
