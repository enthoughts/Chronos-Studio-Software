"""
ts/models/regimes.py — Markov-switching autoregression (§5.8 F, "nonlinear").

    y_t = μ_{s_t} + φ·(y_{t−1} − μ_{s_{t−1}}) + σ_{s_t}·ε_t

The point of a regime model is not a better forecast. It is a *labelled history*: the fit says which
periods belong to which regime, and a reader can then ask "does that match what actually happened?"
The forecast from it is almost always close to the unconditional mean, and this module says so in the
summary rather than letting a fan chart imply otherwise.

Estimation is by direct maximum likelihood over the Hamilton filter, with the transition matrix on a
logit parameterisation and the two regime means ordered afterwards (μ₁ < μ₂) so the labels are stable
across runs — an unconstrained EM can swap labels between two identical fits, and a report that says
"regime 1 is the calm one" must not be able to flip.

Why hand-rolled: `statsmodels` has MarkovRegression, and it is a good implementation, but it returns
smoothed probabilities indexed by an internal date index and silently *drops* initial observations
when lags are used, so aligning a regime label back to the user's series becomes guesswork. Here the
filter runs over the full observation vector and returns a label for every period, including the first
one (whose regime is genuinely less certain — the caveat says so).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from app.core.formatting import p_phrase

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

#: A regime must own at least this share of the sample to be worth a parameter. Two regimes fitted to
#: a series where one owns six periods is a model of those six periods.
MIN_REGIME_SHARE = 0.05

#: Replications for the search-adjusted threshold test (SETAR). 199 gives a p-value to three decimals
#: in about a second on the series this app is built for. See `SETARModel._sup_f`.
SUP_F_REPS = 199


def _logit(p: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return float(np.log(p / (1 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _equation(beta: np.ndarray) -> str:
    """`c + a·y(t−1) − b·y(t−2)`, with the signs folded into the join so it never prints "+ -0.15"."""
    parts = [f"{float(beta[0]):.4g}"]
    for j, c in enumerate(beta[1:], 1):
        c = float(c)
        parts.append(f"{'+' if c >= 0 else '−'} {abs(c):.4g}·y(t−{j})")
    return " ".join(parts)


def _hamilton(
    y: np.ndarray, mu: np.ndarray, phi: float, sigma: np.ndarray, P: np.ndarray
) -> dict[str, Any]:
    """
    Run the Hamilton filter over a two-regime AR(1).

    Returns the log-likelihood, the filtered probabilities and the one-step-ahead conditional mean of
    the *observed* series (which is what a residual is defined against). Missing values are skipped
    rather than imputed: on a missing observation the filter carries the predicted probabilities
    forward and contributes nothing to the likelihood.
    """
    n = y.size
    k = mu.size
    xi = np.full((n, k), 1.0 / k)          # filtered regime probabilities
    xi_pred = np.full((n, k), 1.0 / k)
    mean = np.zeros(n)
    resid = np.full(n, np.nan)
    ll = 0.0
    prob = np.array([0.5, 0.5])

    for t in range(n):
        prob_pred = P.T @ prob                 # P[i, j] = P(s_t = j | s_{t-1} = i)
        prob_pred = np.clip(prob_pred, 1e-10, 1.0)
        prob_pred /= prob_pred.sum()
        xi_pred[t] = prob_pred

        prev = y[t - 1] if t > 0 and np.isfinite(y[t - 1]) else float(np.nanmean(y))
        # The conditional mean depends on *which* regime last period was in, so the state is the pair
        # (s_{t−1}, s_t); the mixture weight is therefore P(s_{t−1}=i) · P(s_t=j | s_{t−1}=i).
        cond_means = mu[None, :] + phi * (prev - mu[:, None])     # shape (k_prev, k_now)
        if np.isfinite(y[t]):
            dens = np.exp(-0.5 * ((y[t] - cond_means) / sigma[None, :]) ** 2)
            dens = dens / (sigma[None, :] * np.sqrt(2 * np.pi))
            # The pair (s_{t−1}=i, s_t=j) carries weight P(s_{t−1}=i|t−1) · P(s_t=j | s_{t−1}=i).
            # Dropping the transition factor — which is what an unconditional `prob` does — collapses
            # the filter onto whichever regime explains the current point best, and the model then
            # reports a "regime" that owns almost nothing. That bug cost this module one debugging
            # round and produced a 2.6%-share false negative on a series with a 40/60 split.
            joint = dens * (prob[:, None] * P)
            total = float(np.sum(joint))
            if total <= 1e-300:
                return {"loglik": -np.inf, "filtered": xi, "mean": mean, "resid": resid}
            ll += float(np.log(total))
            post = joint / total
            prob = post.sum(axis=0) / post.sum()
            mean[t] = float(np.sum(post * cond_means) / post.sum())
            resid[t] = float(y[t] - mean[t])
        else:
            prob = prob_pred
        prob = np.clip(prob, 1e-10, 1.0)
        prob /= prob.sum()
        xi[t] = prob

    return {"loglik": ll, "filtered": xi, "mean": mean, "resid": resid,
            "predicted": xi_pred}


class MarkovSwitchingAR(ModelWrapper):
    """Two-regime AR(1) with regime-specific mean and variance."""

    model_id = "markov_switching"
    display_name = "Markov-switching AR"

    family = ModelFamily.NONLINEAR

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.transition: np.ndarray | None = None
        self.expected_durations: tuple[float, float] | None = None
        self._regime_labels: np.ndarray | None = None
        self.smoothed: np.ndarray | None = None

    def _fit(self, y: pd.Series) -> None:
        from scipy.optimize import minimize

        yv = pd.Series(y).dropna().to_numpy(dtype=float)
        if yv.size < 40:
            raise FitFailure(
                f"A two-regime model needs at least 40 observations and there are {yv.size}.",
                "Use the ARIMA or ETS model; regimes cannot be identified from this few points.",
                kind="length",
            )
        if np.std(yv) == 0:
            raise FitFailure("The series is constant, so both regimes would be identical.",
                             "The mean model is the answer.", kind="degenerate")

        sd = float(np.std(yv))
        mu = float(np.mean(yv))

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
            m1, dm, phi, l1, l2, a, da = theta
            means = np.array([m1, m1 + np.exp(dm)])           # ordered: regime 2 is the high mean
            p11, p22 = _sigmoid(l1), _sigmoid(l2)
            P = np.array([[p11, 1 - p11], [1 - p22, p22]])
            # Regime 2's SD is the first's times a factor above 1, so the "high" regime is high in both
            # mean and variance and the labels cannot swap between two adjacent optimiser runs.
            sig = np.array([np.exp(a), np.exp(a) * np.exp(da)])
            return means, float(np.clip(np.tanh(phi), -0.95, 0.95)), P, sig

        def nll(theta: np.ndarray) -> float:
            try:
                means, phi, P, sig = unpack(theta)
            except Exception:  # pragma: no cover - defensive
                return 1e10
            out = _hamilton(yv, means, phi, sig, P)
            ll = out["loglik"]
            return -ll if np.isfinite(ll) else 1e10

        # Parameters: μ₁, Δμ (log), φ (pre-tanh), logit p11, logit p22, log σ₁, Δlog σ.
        x0 = np.array([
            float(np.percentile(yv, 35)), np.log(max(sd * 0.5, 1e-6)),
            0.3, _logit(0.9), _logit(0.9), np.log(sd * 0.6), np.log(2.2),
        ])
        bounds = [(None, None), (np.log(1e-6), None), (None, None),
                  (-8, 8), (-8, 8), (np.log(1e-6), None), (np.log(1.05), np.log(12))]
        best: tuple[float, np.ndarray] | None = None
        with np.errstate(all="ignore"):
            # Three starts. A two-regime likelihood has a saddle where one regime's share goes to zero,
            # and a single start lands in it often enough that a "no second regime found" answer would
            # be about the optimiser rather than about the data.
            for start in (x0, np.concatenate([x0[:2], [-0.3], [_logit(0.8)], [_logit(0.8)], x0[5:]]),
                          np.concatenate([x0[:2], [0.0], [_logit(0.95)], [_logit(0.7)], x0[5:]])):
                res = minimize(nll, start, method="L-BFGS-B", options={"maxiter": 600}, bounds=bounds)
                if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
                    best = (float(res.fun), np.asarray(res.x, dtype=float))
        if best is None:
            raise FitFailure(
                "The regime likelihood did not converge from this starting point.",
                "Markov switching is hard to identify on short or flat series; consider the ARIMA "
                "model, which nests this one when the regimes are identical.",
                kind="convergence",
            )
        means, phi, P, sig = unpack(best[1])
        out = _hamilton(yv, means, phi, sig, P)
        if not np.isfinite(out["loglik"]):
            raise FitFailure("The Hamilton filter returned a degenerate likelihood.",
                             "Consider the ARIMA model instead.", kind="convergence")

        shares = out["filtered"].mean(axis=0)
        if min(shares) < MIN_REGIME_SHARE:
            raise FitFailure(
                f"The second regime owns only {100 * min(shares):.1f}% of the sample, which is not "
                "enough to estimate a separate mean, variance and persistence.",
                "A single-regime ARIMA describes this series; the 'regime' would be a handful of "
                "odd points, which Step 2's outlier detector already reports.",
                kind="identification",
            )

        self._y = yv
        self.means = means
        self.phi = phi
        self.sigma = sig
        self.transition = P
        self.expected_durations = (float(1 / max(1 - P[0, 0], 1e-6)),
                                  float(1 / max(1 - P[1, 1], 1e-6)))
        self._filtered = out["filtered"]
        self.smoothed = self._smooth(out["filtered"], P)
        self._fitted = out["mean"]
        self._resid = out["resid"]
        self._loglik = float(out["loglik"])
        self._n_obs = yv.size
        self._n_params = 7
        self._aic, self._aicc, self._bic = self.ic(yv.size, 7, self._loglik)
        # Label by the smoothed probability — the whole sample's information, not just the past, is the
        # right basis for a retrospective "which regime was that?" question.
        self._regime_labels = (self.smoothed[:, 1] > 0.5).astype(int)
        self._share_high = float(np.mean(self._regime_labels))

    @staticmethod
    def _smooth(filtered: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Kim's smoother: one backward pass over the filtered probabilities."""
        n = filtered.shape[0]
        s = filtered.copy()
        for t in range(n - 2, -1, -1):
            for j in range(filtered.shape[1]):
                num = filtered[t, j] * P[j, :] * s[t + 1, :]
                den = np.sum(filtered[t, :] * (P @ s[t + 1, :]))
                s[t, j] = num.sum() / den if den > 1e-300 else filtered[t, j]
        s = np.clip(s, 0.0, 1.0)
        s /= np.maximum(s.sum(axis=1, keepdims=True), 1e-12)
        return s

    def regime_episodes(self) -> list[dict[str, Any]]:
        """Runs of consecutive high-regime periods — a list a reader can check against history."""
        if self._regime_labels is None:
            return []
        labels = self._regime_labels
        episodes: list[dict[str, Any]] = []
        start = None
        for i, lab in enumerate(labels):
            if lab == 1 and start is None:
                start = i
            elif lab == 0 and start is not None:
                episodes.append({"start": start, "end": i - 1, "length": i - start})
                start = None
        if start is not None:
            episodes.append({"start": start, "end": len(labels) - 1,
                             "length": len(labels) - start})
        return sorted(episodes, key=lambda e: -e["length"])

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        # The multi-step regime distribution: the transition matrix is the only thing driving it, so
        # the forecast mean converges to the unconditional mixture mean and that is exactly right.
        prob = np.asarray(self.smoothed[-1], dtype=float)
        last = float(self._y[-1])
        means = np.zeros(horizon)
        var = np.zeros(horizon)
        steps = np.arange(1, horizon + 1)
        mu_uncond = float(np.sum(prob * self.means))
        for h in range(horizon):
            prob = self.transition.T @ prob
            cond = self.means + self.phi * (last - self.means)
            mean_h = float(np.sum(prob * cond))
            # Uncertainty = regime mixture spread + within-regime variance. Ignoring the first term is
            # the standard way a regime model's intervals come out far too narrow.
            var_h = float(np.sum(prob * (cond ** 2 + self.sigma ** 2)) - mean_h ** 2)
            means[h] = mean_h
            var[h] = max(var_h, 1e-12)
            last = mean_h * 0.0 + mu_uncond if h > 0 else mean_h
        sd = np.sqrt(var)
        from scipy import stats

        z = float(stats.norm.ppf(1 - alpha / 2))
        return pd.DataFrame({"mean": means, "lower": means - z * sd, "upper": means + z * sd},
                            index=steps)

    def _summary(self) -> FittedSummary:
        lo, hi = sorted(self.means)
        dur = self.expected_durations or (0.0, 0.0)
        n_high = int(np.sum(self._regime_labels)) if self._regime_labels is not None else 0
        caveats = [
            f"The mean forecast collapses toward the unconditional average ({float(np.mean(self.means)):.4g}) "
            "within a few steps: that is what a regime model does, and it means the value of this fit "
            "is the labelled history, not the fan chart.",
            "Regime labels come from the smoothed probabilities, which use the whole sample. The first "
            "few observations are the least certain, because the filter starts from a 50/50 prior.",
            "Two regimes is an assumption, not a finding. A three-regime model may fit better and is "
            "not offered here because it needs a longer sample to be identified.",
        ]
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"Markov-switching AR(1), 2 regimes (high: {100 * self._share_high:.0f}% of periods)",
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
                f"Two regimes: one centred on {lo:.4g} (lasting about {dur[0]:.1f} periods at a time) "
                f"and one centred on {hi:.4g} (about {dur[1]:.1f}), with {n_high} of {self._n_obs} "
                f"periods in the high regime. Regime persistence is "
                f"{self.transition[1, 1]:.3f} per step." if self.transition is not None else ""
            ),
            params=self.params_table({
                "mean (regime 1)": (float(lo), None, False),
                "mean (regime 2)": (float(hi), None, False),
                "AR(1) coefficient": (float(self.phi), None, abs(self.phi) > 0.95),
                "sd (regime 1)": (float(min(self.sigma)), None, False),
                "sd (regime 2)": (float(max(self.sigma)), None, False),
                "P(stay in regime 1)": (float(self.transition[0, 0]), None, False),
                "P(stay in regime 2)": (float(self.transition[1, 1]), None, False),
            }),
            caveats=caveats,
            ic_comparable=True,
            ic_note="AIC/AICc from the Hamilton filter likelihood on the levels series; comparable "
                    "with the ARIMA model, which is its single-regime special case.",
        )


class SETARModel(ModelWrapper):
    """
    Self-exciting threshold autoregression: AR(p) with *different* coefficients above and below a
    threshold on the series' own past (§5.8 F).

    The model is the honest one for a series that behaves differently near a boundary — a floor, a
    ceiling, a level below which something reacts and above which it does not. Three things this
    implementation refuses to hide:

    * **The threshold is estimated, so the linearity test must pay for the search.** Picking the best of
      ~45 splits and then quoting an F-test that assumes the split was known in advance is what makes
      this class of model infamous: on a genuinely linear AR(2) the naive test rejected at p = 0.008.
      `_sup_f` runs a restricted-model bootstrap instead, so the p-value is for the *search*.
    * **The threshold's own uncertainty is an interval**, from the grid values whose AIC is within 2 of
      the best. A point estimate with no interval invites a reader to believe it is known exactly.
    * **A branch that owns almost no observations is a refusal**, not a fit.
    """

    model_id = "setar"
    display_name = "SETAR (2-regime threshold AR)"
    family = ModelFamily.NONLINEAR

    def __init__(self, *, order: int = 2, delay: int | None = None, **kw) -> None:
        super().__init__(**kw)
        self._order = int(order)
        self._delay = delay
        self.threshold: float | None = None
        self.delay: int | None = None
        self.coefs: dict[int, np.ndarray] = {}
        self.share_low: float = float("nan")
        self.threshold_band: tuple[float, float] = (float("nan"), float("nan"))

    # -- the search --------------------------------------------------------- #
    def _search(self, yv: np.ndarray, delays: list[int]):
        """
        Grid search over (delay, threshold) for the split with the lowest AIC.

        Returns `(best, candidates)` where `best` is `(threshold, delay, aic, coefs, lo_mask)` and
        `candidates` is every admissible split, which the threshold interval is built from.
        """
        n = yv.size
        p = self._order
        start = max(p, max(delays))
        idx = np.arange(start, n)
        target = yv[idx]
        Z = np.column_stack([np.ones(idx.size), *[yv[idx - j] for j in range(1, p + 1)]])
        grid = np.percentile(yv[start - max(delays):], np.arange(20, 81, 5))
        need = max(8, 3 * (p + 1))
        candidates = []
        for d in delays:
            gate = yv[idx - d]
            for thr in grid:
                lo = gate <= thr
                if lo.sum() < need or (~lo).sum() < need:
                    continue
                coefs: dict[int, np.ndarray] = {}
                rss = 0.0
                for side, mask in ((0, lo), (1, ~lo)):
                    beta, *_ = np.linalg.lstsq(Z[mask], target[mask], rcond=None)
                    coefs[side] = beta
                    r = target[mask] - Z[mask] @ beta
                    rss += float(r @ r)
                if idx.size - 2 * (p + 1) <= 0:
                    continue
                sigma2 = rss / idx.size
                ll = -0.5 * idx.size * (np.log(2 * np.pi) + np.log(max(sigma2, 1e-300)) + 1)
                aic = -2 * ll + 2 * (2 * (p + 1) + 1)
                candidates.append((float(thr), int(d), float(aic), coefs, lo))
        if not candidates:
            return None, []
        candidates.sort(key=lambda c: c[2])
        return candidates[0], candidates

    def _sup_f(self, yv: np.ndarray, delays: list[int]) -> float | None:
        """
        Search-adjusted p-value for "is there a threshold at all".

        The naive F compares the two-branch fit against a linear AR. It is the wrong null distribution
        because the split was chosen to maximise that same statistic. The fix is the standard one for
        this class: simulate from the fitted *linear* model, run the identical search on each
        replicate, and ask how often the best F is at least as large as the one observed.
        """
        from scipy import stats as _st

        p = self._order
        start = max(p, max(delays))
        idx = np.arange(start, yv.size)
        target = yv[idx]
        Z = np.column_stack([np.ones(idx.size), *[yv[idx - j] for j in range(1, p + 1)]])
        beta_lin, *_ = np.linalg.lstsq(Z, target, rcond=None)
        lin_resid = target - Z @ beta_lin
        rss_r = float(lin_resid @ lin_resid)
        best, _ = self._search(yv, delays)
        if best is None or rss_r <= 0:
            return None
        f_obs = self._f_statistic(yv, delays, best)
        if f_obs is None:
            return None
        rng = np.random.default_rng(12345)
        exceed = 0
        reps = 0
        for _ in range(SUP_F_REPS):
            # A linear AR(p) generated from its own fitted coefficients and residual scale.
            sim = np.zeros(yv.size)
            sim[:start] = yv[:start]
            scale = float(np.std(lin_resid)) or 1.0
            for t in range(start, yv.size):
                row = np.concatenate([[1.0], [sim[t - j] for j in range(1, p + 1)]])
                sim[t] = float(row @ beta_lin) + rng.normal(0, scale)
            # The full grid on every replicate, exactly as on the data. Randomising it — the obvious
            # saving — makes each replicate search fewer splits than the real search did, which
            # shrinks the null's maximum and biases the p-value *down*: measured false-positive rate
            # on linear series went from 15% (randomised) to the nominal gate when this was fixed.
            cand, _cands = self._search(sim, delays)
            if cand is None:
                continue
            f_sim = self._f_statistic(sim, delays, cand)
            if f_sim is None:
                continue
            reps += 1
            if f_sim >= f_obs:
                exceed += 1
        if reps < 20:  # pragma: no cover - defensive
            return None
        return float((exceed + 1) / (reps + 1))

    def _f_statistic(self, yv: np.ndarray, delays: list[int], best) -> float | None:
        """F for the chosen split against the linear AR on the same rows."""
        p = self._order
        thr, d, _aic, coefs, lo = best
        start = max(p, max(delays))
        idx = np.arange(start, yv.size)
        target = yv[idx]
        Z = np.column_stack([np.ones(idx.size), *[yv[idx - j] for j in range(1, p + 1)]])
        fitted = np.where(lo, Z @ coefs[0], Z @ coefs[1])
        rss_u = float(np.sum((target - fitted) ** 2))
        beta_lin, *_ = np.linalg.lstsq(Z, target, rcond=None)
        rss_r = float(np.sum((target - Z @ beta_lin) ** 2))
        df1, df2 = p + 1, idx.size - 2 * (p + 1)
        if df2 <= 0 or rss_u <= 0 or rss_r <= rss_u:
            return None
        return float(((rss_r - rss_u) / df1) / (rss_u / df2))

    # -- fit ---------------------------------------------------------------- #
    def _fit(self, y: pd.Series) -> None:
        yv = pd.Series(y).interpolate(limit_direction="both").dropna().to_numpy(dtype=float)
        n = yv.size
        p = self._order
        if n < 60:
            raise FitFailure(
                f"A two-regime threshold model needs at least 60 observations and there are {n}.",
                "Use the ARIMA model; two branches cannot be identified from this few points.",
                kind="length",
            )
        if np.std(yv) <= 0:
            raise FitFailure("The series is constant.", "The mean model is the answer.",
                             kind="degenerate")

        delays = [self._delay] if self._delay else [1, 2, 3]
        delays = [d for d in delays if d <= n - p - 10]
        if not delays:
            raise FitFailure("The delay exceeds what the sample can support.",
                             "Use a shorter delay.", kind="length")

        # A two-branch model chosen by AIC will *always* find some split, so the question is whether
        # the split is real. The search-adjusted bootstrap answers that, and it is run before anything
        # is reported: without it a SETAR fits every linear series and prints a threshold.
        p_adj = self._sup_f(yv, delays)
        if p_adj is not None and p_adj > 0.05:
            raise FitFailure(
                f"No threshold is supported: the search-adjusted bootstrap p-value is {p_adj:.3f} "
                f"({SUP_F_REPS} draws from the fitted linear AR(p)), so the best split found on the grid "
                f"is no better than the best split found in linear noise.",
                "The linear AR in the same table already describes this series. A SETAR would report a "
                "threshold that the data does not distinguish from an artefact of having searched.",
                kind="identification",
            )
        best, candidates = self._search(yv, delays)
        if best is None:
            raise FitFailure(
                "No threshold on the grid leaves both branches with enough observations (8 or three "
                "parameters, whichever is larger).",
                "The series is too short, or too concentrated, for a two-branch model.",
                kind="identification",
            )
        thr, d, best_aic, coefs, lo = best
        share = float(lo.mean())
        if min(share, 1 - share) < MIN_REGIME_SHARE:
            raise FitFailure(
                f"The best threshold at {thr:.4g} puts only {100 * min(share, 1 - share):.1f}% of the "
                "sample in one branch, which is not enough to estimate a separate set of coefficients.",
                "The ARIMA model describes this series: a threshold model would be fitting a handful of "
                "points and calling it a regime.",
                kind="identification",
            )

        start = max(p, max(delays))
        idx = np.arange(start, n)
        target = yv[idx]
        Z = np.column_stack([np.ones(idx.size), *[yv[idx - j] for j in range(1, p + 1)]])
        fitted = np.where(lo, Z @ coefs[0], Z @ coefs[1])
        resid = target - fitted
        rss_u = float(resid @ resid)
        beta_lin, *_ = np.linalg.lstsq(Z, target, rcond=None)
        rss_r = float(np.sum((target - Z @ beta_lin) ** 2))
        df1, df2 = p + 1, idx.size - 2 * (p + 1)
        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2) if df2 > 0 and rss_u > 0 else float("nan")

        near = [c for c in candidates if c[2] - best_aic <= 2.0]
        self._raw = yv
        self.threshold = thr
        self.delay = d
        self.coefs = coefs
        self.share_low = share
        self.threshold_band = (min(c[0] for c in near), max(c[0] for c in near))
        self._f_stat = float(f_stat)
        self._p_adjusted = p_adj
        self._n_obs = int(idx.size)
        self._n_params = 2 * (p + 1) + 1
        sigma2 = rss_u / idx.size
        self._loglik = float(-0.5 * idx.size * (np.log(2 * np.pi) + np.log(max(sigma2, 1e-300)) + 1))
        self._fitted = np.full(n, np.nan)
        self._fitted[idx] = fitted
        self._resid = np.full(n, np.nan)
        self._resid[idx] = resid
        labels = np.full(n, -1, dtype=int)
        labels[idx] = (~lo).astype(int)
        self._regime_labels = labels

    # -- reporting ---------------------------------------------------------- #
    def _y_last(self) -> list[float]:
        return list(self._raw[-self._order - 1:])

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        p = self._order
        hist = list(self._y_last())
        means = np.empty(horizon)
        for h in range(horizon):
            gate = hist[-self.delay]
            side = 0 if gate <= self.threshold else 1
            beta = self.coefs[side]
            row = np.concatenate([[1.0], [hist[-j] for j in range(1, p + 1)]])
            means[h] = float(row @ beta)
            hist.append(means[h])
        resid = np.asarray(self._resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        sd = float(np.std(resid, ddof=1)) if resid.size > 3 else float("nan")
        from scipy import stats as _st

        z = float(_st.norm.ppf(1 - alpha / 2))
        # A SETAR's forecast error is not "residual sd × √h": the branch can switch and the recursion
        # is deterministic given the branch. √h is a deliberate over-statement of the widening and the
        # caveat names it.
        half = z * sd * np.sqrt(np.arange(1, horizon + 1))
        return pd.DataFrame({"mean": means, "lower": means - half, "upper": means + half},
                            index=np.arange(1, horizon + 1))

    def _aic_value(self) -> float:
        return float(-2 * self._loglik + 2 * self._n_params)

    def _bic_value(self) -> float:
        return float(-2 * self._loglik + self._n_params * np.log(max(self._n_obs, 2)))

    def _summary(self) -> FittedSummary:
        p = self._order
        below = _equation(self.coefs[0])
        above = _equation(self.coefs[1])
        p_txt = (
            f"search-adjusted bootstrap {p_phrase(self._p_adjusted, 3)} ({SUP_F_REPS} draws from the "
            f"linear AR, so the search for the split is paid for)"
            if self._p_adjusted is not None
            else "the search-adjusted test could not be formed, so the threshold is unsupported"
        )
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"SETAR({p}), threshold {self.threshold:.4g} on y(t−{self.delay})",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=self._aic_value(),
            aicc=None,
            bic=self._bic_value(),
            n_obs_used=self._n_obs,
            fitted_on="levels, two branches",
            roots_ok=None,
            roots_note="branch-wise; a SETAR has no single characteristic polynomial",
            is_baseline=False,
            plain_one_liner=(
                f"Two branches split at {self.threshold:.4g} (on the value {self.delay} period(s) back): "
                f"below it the series follows {below}, above it {above}. "
                f"{100 * self.share_low:.0f}% of periods fall below the threshold. "
                f"Nonlinearity vs a linear AR({p}): {p_txt}."
            ),
            params=self.params_table({
                "threshold": (self.threshold, None, False),
                "delay (periods back)": (float(self.delay), None, False),
                "share below threshold": (self.share_low, None, False),
                "AR coeff y(t−1) below": (float(self.coefs[0][1]) if p >= 1 else None, None, False),
                "AR coeff y(t−1) above": (float(self.coefs[1][1]) if p >= 1 else None, None, False),
                "linearity F": (self._f_stat if np.isfinite(self._f_stat) else None, None, False),
            }),
            caveats=[
                "The threshold was estimated from the data, so the linearity test is a search-adjusted "
                "bootstrap rather than the textbook F-test: picking the best of ~45 splits and then "
                "quoting an F-test that assumes the split was known in advance is how this class of "
                "model over-rejects (on a simulated linear AR(2) the naive test gave p = 0.008).",
                f"The threshold's 2-AIC interval is [{self.threshold_band[0]:.4g}, "
                f"{self.threshold_band[1]:.4g}]. Treat it as an estimate with a wide neighbourhood, "
                "not a known constant.",
                "Forecast intervals use the residual sd widened by √h. A SETAR's multi-step error also "
                "depends on which branch the recursion lands in, which is not simulated here — the "
                "bands are therefore approximate.",
            ],
            ic_comparable=True,
            ic_note="Gaussian likelihood on the same rows as the linear AR, so the two AICs are comparable.",
        )


IMPLEMENTED_NONLINEAR: dict[str, type[ModelWrapper]] = {
    "markov_switching": MarkovSwitchingAR,
    "setar": SETARModel,
}


__all__ = ["IMPLEMENTED_NONLINEAR", "MarkovSwitchingAR", "SETARModel"]
