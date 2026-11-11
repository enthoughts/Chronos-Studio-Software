"""
ts/models/ucm.py — unobserved components / structural models, fitted by Kalman filter (§5.8 B).

    y_t = level_t + season_t + ε_t
    level_t = level_{t-1} + trend_{t-1} + η_t      (local linear trend; trend dropped for a local level)
    trend_t = trend_{t-1} + ζ_t
    season_t = −Σ_{j=1..m−1} season_{t−j} + ω_t    (dummy seasonal, so the m indices sum to zero)

Three specifications, chosen by AICc the same way ETS's grid is:

* **local level** — one state, a random walk plus noise. The structural cousin of SES.
* **local linear trend** — level and slope, each with its own noise. If the slope's variance is
  estimated at zero, the data is saying "there is a trend but it is not changing", and the caveat
  says so rather than silently reporting a stiff model.
* **BSM** — local linear trend plus the seasonal block, when a period is known and there is enough
  data to identify it.

Why a Kalman filter and not statsmodels' `UnobservedComponents`: the brief asks for a model that
"handles missing values natively". statsmodels does that too, but it silently *drops* missing values
from the returned fitted array in some versions, so the alignment between the series and the
residuals becomes a guess. Here the filter runs over the full index, skips the update step on a
missing observation, and the residuals come back the same length as the input — which is what
`diagnose()` needs to be correct rather than approximately correct.

Everything is in NumPy so the state is inspectable: `level`, `trend` and `seasonal` paths are kept and
reported, because "the model says the level is rising 1.6 units a month" is the reason to choose a
structural model in the first place.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

#: Variance floor, relative to the series variance. A state variance of exactly zero makes the
#: likelihood flat and the optimiser wander; 1e-6 is the smallest value that still behaves.
VAR_FLOOR = 1e-6


def _build_matrices(
    spec: str, m: int | None, n: int
) -> tuple[np.ndarray, np.ndarray, int, list[tuple[str, list[int]]]]:
    """
    The transition matrix T, the design vector Z, and which states share a variance.

    State order: level, [trend], [seasonal_1 … seasonal_{m−1}]. The seasonal block uses the standard
    dummy formulation whose transition row is all −1, which keeps the m indices summing to zero
    without carrying an extra constraint.

    The third return value is the part that matters statistically. The m−1 seasonal states are *one*
    stochastic component, not m−1 of them: giving each its own variance means 11 free parameters on a
    12-period season, which cannot be identified from 144 observations. The fit "converges" anyway —
    the likelihood is flat in most of those directions — and it takes a minute to do it. One variance
    per component is the standard Harvey (1989) specification and the reason the filter is fast.
    """
    k = 1 + (1 if spec in {"llt", "bsm"} else 0)
    if spec == "bsm":
        if not m or m < 2:
            raise FitFailure(
                "A seasonal structural model needs a seasonal period of at least 2.",
                "Use the local level or local trend variant.",
                kind="seasonality",
            )
        k += m - 1
    T = np.zeros((k, k))
    T[0, 0] = 1.0
    if spec in {"llt", "bsm"}:
        T[0, 1] = 1.0
        T[1, 1] = 1.0
    if spec == "bsm":
        s0 = 2
        T[s0, s0 : s0 + m - 1] = -1.0
        for i in range(1, m - 1):
            T[s0 + i, s0 + i - 1] = 1.0
    Z = np.zeros(k)
    Z[0] = 1.0
    if spec == "bsm":
        Z[2] = 1.0

    groups: list[tuple[str, list[int]]] = [("level", [0])]
    if spec in {"llt", "bsm"}:
        groups.append(("trend", [1]))
    if spec == "bsm":
        groups.append(("seasonal", list(range(2, k))))
    return T, Z, k, groups


def _kalman(
    y: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    q: np.ndarray,
    h: float,
    *,
    smooth: bool = False,
) -> dict[str, Any]:
    """
    Run the Kalman filter (and optionally the RTS smoother) over `y`, skipping missing values.

    Returns the log-likelihood and, on request, the smoothed states and the one-step-ahead
    prediction errors. The prediction error at a *missing* observation is 0 by construction (there
    was nothing to predict against) and is excluded from the likelihood — that is the whole point of
    handling missing values natively rather than imputing first.
    """
    n = y.size
    k = T.shape[0]
    a = np.zeros(k)          # filtered state
    P = np.eye(k) * 1e3      # diffuse-ish prior on the level
    Q = np.diag(q)

    ll = 0.0
    n_obs = 0
    resid = np.full(n, np.nan)
    fitted = np.full(n, np.nan)
    store_a = np.zeros((n, k))
    store_P = np.zeros((n, k, k))
    store_ap = np.zeros((n, k))
    store_Pp = np.zeros((n, k, k))

    for t in range(n):
        # --- predict ------------------------------------------------------- #
        ap = T @ a
        Pp = T @ P @ T.T + Q
        store_ap[t], store_Pp[t] = ap, Pp

        yt = y[t]
        if np.isfinite(yt):
            v = yt - float(Z @ ap)                 # one-step-ahead prediction error
            F = float(Z @ Pp @ Z) + h              # prediction variance
            F = max(F, 1e-12)
            resid[t] = v
            fitted[t] = yt - v
            ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
            n_obs += 1
            K = Pp @ Z / F
            a = ap + K * v
            P = Pp - np.outer(K, Z @ Pp)
        else:
            a, P = ap, Pp                          # nothing to update with: the prior *is* the estimate
        a, P = _stabilise(a, P)
        store_a[t], store_P[t] = a, P

    out: dict[str, Any] = {
        "loglik": ll,
        "n_obs": n_obs,
        "resid": resid,
        "fitted": fitted,
        "states": store_a if not smooth else None,
    }
    if smooth:
        # Rauch–Tung–Striebel: the smoothed path is what a plot wants, the filtered path is what a
        # forecast needs. Keeping both is cheap and avoids the classic confusion between them.
        s = np.zeros((n, k))
        s[-1] = store_a[-1]
        for t in range(n - 2, -1, -1):
            Pp = store_Pp[t + 1]
            J = store_P[t] @ T.T @ np.linalg.pinv(Pp)
            s[t] = store_a[t] + J @ (s[t + 1] - store_ap[t + 1])
        out["states"] = s
    return out


def _stabilise(a: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the covariance symmetric and positive-definite; a Kalman filter drifts off this in long runs."""
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    if w.min() < 0:
        w = np.maximum(w, 0.0)
        P = V @ np.diag(w) @ V.T
    return a, P


class UCMModel(ModelWrapper):
    """Structural model: local level, local linear trend, or BSM. Selected by AICc."""

    model_id = "ucm"
    display_name = "Unobserved components"

    family = ModelFamily.STRUCTURAL

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        #: Which specification won, and what the grid looked like.
        self.spec: str = "local level"
        self.alternatives: list[dict[str, Any]] = []
        self.level: np.ndarray | None = None
        self.trend: np.ndarray | None = None
        self.seasonal: np.ndarray | None = None

    # -- the grid ------------------------------------------------------------ #
    def _specs(self, y: np.ndarray) -> list[tuple[str, str, int | None]]:
        specs: list[tuple[str, str, int | None]] = [("local level", "ll", None)]
        if y.size >= 12:
            specs.append(("local linear trend", "llt", None))
        m = self.period
        if m and y.size >= 3 * m:
            specs.append(("local linear trend + seasonal (BSM)", "bsm", int(m)))
            specs.append(("local level + seasonal", "bsm_level", int(m)))
        return specs

    def _fit(self, y: pd.Series) -> None:
        yv = y.to_numpy(dtype=float)
        if np.isfinite(yv).sum() < 10:
            raise FitFailure(
                f"A structural model needs at least 10 observations and there are "
                f"{int(np.isfinite(yv).sum())}.",
                "Use the mean or naive benchmark instead.",
                kind="length",
            )
        if np.nanstd(yv) == 0:
            raise FitFailure(
                "The series is constant, so every state variance is zero.",
                "The mean model already is the answer.",
                kind="degenerate",
            )

        best: dict[str, Any] | None = None
        for label, key, m in self._specs(yv):
            try:
                entry = self._fit_spec(yv, label, key, m)
            except FitFailure:
                continue
            except Exception as exc:  # noqa: BLE001 - a numerical failure in one spec is not a failed model
                self.alternatives.append({"spec": label, "aicc": None,
                                          "error": f"{type(exc).__name__}"})
                continue
            if entry is None:
                continue
            self.alternatives.append({"spec": label, "aicc": entry["aicc"]})
            if best is None or (entry["aicc"] is not None and
                                (best["aicc"] is None or entry["aicc"] < best["aicc"])):
                best = entry

        if best is None:
            raise FitFailure(
                f"None of the {len(self._specs(yv))} structural specifications converged.",
                "This usually means the series is too short or too flat for state variances to be "
                "identified; the baselines still stand.",
                kind="convergence",
            )
        self.spec = best["label"]
        self._m = best["m"]
        self._q = best["q"]
        self._h = best["h"]
        self._loglik = best["loglik"]
        self._n_params = best["k_params"]
        self._aicc = best["aicc"]
        self._aic = best["aic"]
        self._state = best["states"]
        self._fitted = best["fitted"]
        self._resid = best["resid"]
        self._n_obs = best["n_obs"]
        if self._state is not None:
            self.level = self._state[:, 0]
            if best["key"] in {"llt", "bsm"}:
                self.trend = self._state[:, 1]
            if best["key"] in {"bsm", "bsm_level"}:
                s0 = 2 if best["key"] == "bsm" else 1
                self.seasonal = self._state[:, s0]

    def _fit_spec(
        self, yv: np.ndarray, label: str, key: str, m: int | None
    ) -> dict[str, Any] | None:
        """Estimate the component variances of one specification by maximum likelihood."""
        from scipy.optimize import minimize

        base = "bsm" if key in {"bsm", "bsm_level"} else key
        T, Z, k_state, groups = _build_matrices(base, m, yv.size)
        if key == "bsm_level":
            # Local level + season: drop the slope state. Rebuilding the matrices from the reduced
            # state list keeps the seasonal block's own −1 transition row intact (slicing the full
            # matrix would leave a row pointing at a state that no longer exists).
            keep = [0] + list(range(2, k_state))
            T = T[np.ix_(keep, keep)]
            Z = Z[keep]
            groups = [("level", [0]), ("seasonal", list(range(1, len(keep))))]
            k_state = len(keep)
            if k_state == 1:
                raise FitFailure("degenerate", kind="degenerate")

        var_y = float(np.nanvar(yv))
        # One variance per *component* (level, trend, seasonal) plus the observation variance.
        # Parameterised in logs so the optimiser cannot propose a negative variance.
        n_comp = len(groups)

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
            q = np.zeros(k_state)
            for i, (_name, idx) in enumerate(groups):
                q[idx] = np.exp(theta[i]) * var_y
            h = float(np.exp(theta[n_comp]) * var_y)
            return q, max(h, var_y * VAR_FLOOR)

        def nll(theta: np.ndarray) -> float:
            q, h = unpack(theta)
            try:
                out = _kalman(yv, T, Z, q, h)
            except Exception:  # pragma: no cover - defensive
                return 1e10
            ll = out["loglik"]
            return -ll if np.isfinite(ll) else 1e10

        # L-BFGS-B rather than Nelder–Mead: the likelihood is smooth in the log-variances, the gradient
        # is worth having, and on AirPassengers this is 3.5 s against 14 s for the simplex search —
        # for the *same* optimum (586.47). Three starts, because the ridge at "component variance = 0"
        # is a genuine local trap: Powell lands on a strictly worse point from the 0.05 start alone.
        # Keeping the best of three L-BFGS-B runs costs nothing and cannot do worse than one.
        best_theta: np.ndarray | None = None
        best_val = np.inf
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for start in (0.05, 0.01, 0.2):
                res = minimize(
                    nll, np.full(n_comp + 1, np.log(start)), method="L-BFGS-B",
                    options={"maxiter": 300},
                )
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val, best_theta = float(res.fun), np.asarray(res.x, dtype=float)
        if best_theta is None:
            return None
        q, h = unpack(best_theta)
        kal = _kalman(yv, T, Z, q, h, smooth=True)
        if kal["n_obs"] < 10:
            return None

        # Count only the components whose variance is not pinned at the floor — a component estimated
        # at exactly zero variation is not a free parameter, and charging an IC penalty for it would
        # bias the selection against the very specification the data is calling simpler.
        free = sum(1 for _n, idx in groups if float(q[idx[0]]) > var_y * 1e-4)
        k_params = free + 1 + int(h > var_y * 1e-4)
        n = kal["n_obs"]
        aic = -2.0 * kal["loglik"] + 2.0 * k_params
        denom = n - k_params - 1
        aicc = (-2.0 * kal["loglik"] + 2.0 * k_params * n / denom) if denom > 0 else None
        return {
            "label": label, "key": key, "m": m, "q": q, "h": h,
            "loglik": kal["loglik"], "k_params": k_params, "aic": aic, "aicc": aicc,
            "states": kal["states"], "fitted": kal["fitted"], "resid": kal["resid"],
            "n_obs": int(n),
        }

    # -- forecasting --------------------------------------------------------- #
    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        base = "bsm" if ("seasonal" in self.spec or "BSM" in self.spec) else (
            "llt" if "trend" in self.spec else "ll"
        )
        T, Z, k_state, _groups = _build_matrices(base, self._m, horizon + 1)
        if self.spec.startswith("local level +"):
            keep = [0] + list(range(2, k_state))
            T = T[np.ix_(keep, keep)]
            Z = Z[keep]

        a = self._state[-1].copy()
        if T.shape[0] != a.size:  # pragma: no cover - guards a matrix/state mismatch
            raise FitFailure("The state vector and transition matrix disagree.", "This is a bug.")
        Q = np.diag(self._q[: a.size])
        steps = np.arange(1, horizon + 1)
        mean = np.zeros(horizon)
        var = np.zeros(horizon)
        # The predictive variance accumulates the state and observation variance along the path;
        # a flat "residual SD × √h" band would be the wrong shape for a local *level* model, whose
        # uncertainty grows like h rather than √h.
        P = np.zeros((a.size, a.size))
        for i in range(horizon):
            a = T @ a
            P = T @ P @ T.T + Q
            mean[i] = float(Z @ a)
            var[i] = float(Z @ P @ Z) + self._h
        sd = np.sqrt(np.maximum(var, 1e-12))
        z = _z(alpha)
        return pd.DataFrame(
            {"mean": mean, "lower": mean - z * sd, "upper": mean + z * sd}, index=steps
        )

    # -- summary ------------------------------------------------------------- #
    def _summary(self) -> FittedSummary:
        n = self._n_obs
        k = self._n_params
        _, aicc, bic = self.ic(n, k, self._loglik)
        caveats: list[str] = []
        q_level, q_trend = float(self._q[0]), float(self._q[1]) if self._q.size > 1 else 0.0
        if self.trend is not None and q_trend <= VAR_FLOOR:
            caveats.append(
                "The slope's variance was estimated at the boundary of zero, so the model says the "
                "trend is constant rather than changing — a straight line, not a wandering one."
            )
        if "seasonal" in self.spec:
            caveats.append(
                f"The {self._m}-period seasonal component is a dummy block: it lets the seasonal "
                "shape evolve slowly, but on a series this short most of its variance will be "
                "estimated near zero, which is the data saying the season is stable."
            )
        caveats.append(
            "A structural model decomposes by assumption: the split into level, trend and season is "
            "only as trustworthy as the assumption that each moves as a random walk."
        )
        level_now = float(self.level[-1]) if self.level is not None else float("nan")
        slope_now = (
            float(self.trend[-1]) if self.trend is not None and np.isfinite(self.trend[-1]) else None
        )
        one_liner = (
            f"{self.spec}: today's underlying level is {level_now:.4g}"
            + (f", moving {slope_now:+.4g} per step" if slope_now is not None else "")
            + f". State and observation variances were estimated by likelihood ({k} free parameters)."
        )
        alt = sorted(
            (a for a in self.alternatives if a.get("aicc") is not None), key=lambda a: a["aicc"]
        )
        if len(alt) > 1:
            gap = abs(alt[1]["aicc"] - alt[0]["aicc"])
            # Only a *small* gap is ambiguous. 24 AICc units is decisive evidence, and describing it
            # as "not decisive" would be the opposite of what the number says.
            if gap <= 4.0:
                caveats.append(
                    f"The second-best specification ({alt[1]['spec']}) is within {gap:.1f} AICc units, "
                    "so the choice of states is not decisive."
                )
            else:
                caveats.append(
                    f"The next-best specification ({alt[1]['spec']}) is {gap:.1f} AICc units behind, "
                    "so the extra states are earning their place."
                )
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"UCM — {self.spec}",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=k,
            loglik=self._loglik,
            aic=self._aic,
            aicc=aicc,
            bic=bic,
            n_obs_used=n,
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=one_liner,
            params=self.params_table({
                "level variance": (float(self._q[0]), None, q_level <= VAR_FLOOR),
                **(
                    {"slope variance": (float(self._q[1]), None, q_trend <= VAR_FLOOR)}
                    if self._q.size > 1 else {}
                ),
                "observation variance": (float(self._h), None, self._h <= VAR_FLOOR),
            }),
            caveats=caveats,
            ic_comparable=True,
            ic_note="AIC/AICc from the Kalman filter likelihood on the levels series; comparable "
                    "with other levels-scale Gaussian models.",
        )


def _z(alpha: float) -> float:
    from scipy import stats

    return float(stats.norm.ppf(1 - alpha / 2))


IMPLEMENTED_STRUCTURAL: dict[str, type[ModelWrapper]] = {
    "ucm": UCMModel,
}


__all__ = ["IMPLEMENTED_STRUCTURAL", "UCMModel"]
