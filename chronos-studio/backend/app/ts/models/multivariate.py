"""
ts/models/multivariate.py — VAR, VECM and the machinery that makes them readable (§5.8 B).

The single-series models in this build all answer "where is *this* series going". A multivariate model
answers a different question: **how do these series move each other?** That question has three parts,
and a VAR is only the first:

1. **Does x help predict y?** — Granger causality, which is a test on the VAR's coefficients and
   nothing more. It is not causation and the module says so in the sentence it prints.
2. **How does a shock to one travel through the others?** — the impulse response, with bootstrap bands
   because a point IRF with no band is a line somebody drew.
3. **How much of each series' forecast error is each shock responsible for?** — the FEVD, which is the
   question most people actually mean when they say "how important is x".

Three things this module refuses to do, each because omitting it is how a VAR result misleads:

* **Difference by default.** Two independent random walks regressed on each other give significant
  coefficients almost every time. The lag order is chosen on a *differenced* series unless the
  Johansen trace test says a stationary linear combination exists, and the choice is reported.
* **Report FEVD from an unidentified system.** Cholesky ordering is an assumption about which shock
  acts first; it changes the answer, so the ordering is stated in the output rather than buried.
* **Report Granger causality as causation.** The sentence says "predicts", and the caveat says what
  the test cannot rule out — a common driver that the model does not include.

Estimation is by OLS equation-by-equation, which is the maximum-likelihood estimator for a VAR with
the same regressors in every equation (Zellner's result). No `statsmodels` VAR here for the same reason
as the regime model: its fitted object indexes by its own date grid and drops the first p rows without
saying so, which makes aligning a residual or an IRF back to the user's calendar guesswork.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.core.formatting import p_phrase
from scipy import stats

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

#: Minimum observations per equation. With k series and p lags a VAR estimates k(kp + 1) parameters, so
#: a 3-series VAR at p = 4 needs 39; below this the user is fitting their own noise.
MIN_OBS_PER_PARAM = 3

#: Lagrange-multiplier lag-order test is evaluated up to this many lags. Beyond it the test has more
#: parameters than the sample can distinguish on the series this app is built for.
MAX_LAGS = 8


# --------------------------------------------------------------------------- #
# Small linear-algebra helpers
# --------------------------------------------------------------------------- #
def companion_roots(A: np.ndarray) -> np.ndarray:
    """Eigenvalues of the companion matrix: inside the unit circle ⇔ the VAR is stable."""
    k, _ = A.shape[0], A.shape[1] // A.shape[0]
    p = A.shape[1] // A.shape[0]
    top = A
    bottom = np.hstack([np.eye(k * (p - 1)), np.zeros((k * (p - 1), k))])
    comp = np.vstack([top, bottom])
    return np.linalg.eigvals(comp)


#: Roots this close to the unit circle are treated as exactly on it. Not a fudge: the companion roots
#: of a cointegrated system are *exactly* 1 in theory and 0.99999… after floating point, and a strict
#: "< 1" test would refuse every VECM ever fitted.
UNIT_ROOT_TOL = 1e-6


def var_stability_ok(A: np.ndarray) -> tuple[bool, float]:
    """
    Is the system explosive? Returns `(ok, largest modulus)`.

    **A root of exactly 1 is not instability.** A system over k series with r cointegrating relations
    has k − r unit roots by construction — that is what cointegration *means* — so the question is never
    "are all roots inside the circle" but "is any root outside it". Getting this backwards refuses every
    correctly specified VECM, which is exactly what this check did before it was rewritten.
    """
    roots = companion_roots(A)
    if not roots.size:
        return True, 0.0
    max_mod = float(np.max(np.abs(roots)))
    return max_mod <= 1.0 + UNIT_ROOT_TOL, max_mod


def unit_root_count(A: np.ndarray) -> int:
    """How many companion roots sit on the unit circle — should be exactly k − rank."""
    roots = companion_roots(A)
    return int(np.sum(np.abs(np.abs(roots) - 1.0) <= UNIT_ROOT_TOL))


def _lag_matrix(data: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Build (Y, X) for a VAR(p) in levels: X is [1, y_{t−1}, …, y_{t−p}]."""
    n, k = data.shape
    rows = n - p
    Y = data[p:]
    cols = [np.ones(rows)]
    for lag in range(1, p + 1):
        cols.append(data[p - lag: n - lag])
    X = np.column_stack(cols)
    return Y, X


def fit_var_ols(data: np.ndarray, p: int) -> dict[str, Any]:
    """
    Estimate a VAR(p) by equation-wise OLS and return everything the tests need.

    `B` is (1 + k·p, k) with the constant first, followed by the lag blocks in lag order — the layout
    every consumer here reads, and the one `companion_roots` expects when reshaped.
    """
    Y, X = _lag_matrix(data, p)
    n_used, k = Y.shape
    n_params_1 = X.shape[1]
    if n_used <= n_params_1:
        raise FitFailure(
            f"A VAR({p}) over {k} series has {n_params_1} coefficients per equation and only "
            f"{n_used} usable rows.",
            "Use fewer lags or fewer series — this is more parameters than observations.",
            kind="length",
        )
    # lstsq rather than inv: a near-collinear pair of series makes X'X singular, and pinv degrades to a
    # minimum-norm solution instead of raising.
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ B
    dof = n_used - n_params_1
    sigma_u = (resid.T @ resid) / dof
    loglik = -0.5 * n_used * (k * np.log(2 * np.pi) + np.log(np.linalg.det(sigma_u)) + k)
    return {
        "B": B,
        "resid": resid,
        "sigma_u": sigma_u,
        "n_used": n_used,
        "k": k,
        "p": p,
        "n_params_per_eq": n_params_1,
        "loglik": float(loglik),
        "A": _companion_from_B(B, k, p),
    }


def _companion_from_B(B: np.ndarray, k: int, p: int) -> np.ndarray:
    """The (k×kp) coefficient matrix of the level VAR, in companion form ordering."""
    blocks = [B[1 + i * k: 1 + (i + 1) * k].T for i in range(p)]
    return np.hstack(blocks)


def aic_bic(loglik: float, n: int, k: int, p: int) -> tuple[float, float]:
    n_params = k * (k * p + 1)
    return (-2 * loglik + 2 * n_params) / n, (-2 * loglik + n_params * np.log(n)) / n


def select_order(data: np.ndarray, max_lags: int = MAX_LAGS) -> dict[str, Any]:
    """Choose p by AIC and BIC over a small grid, and report both so the disagreement is visible."""
    rows = []
    for p in range(1, max_lags + 1):
        if data.shape[0] - p < max(20, data.shape[1] * (data.shape[1] * p + 1) * MIN_OBS_PER_PARAM):
            break
        try:
            r = fit_var_ols(data, p)
        except FitFailure:
            break
        a, b = aic_bic(r["loglik"], r["n_used"], r["k"], p)
        rows.append({"p": p, "aic": a, "bic": b, "loglik": r["loglik"]})
    if not rows:
        raise FitFailure(
            "The sample is too short to estimate even a VAR(1) over this many series.",
            "Choose fewer series, or a longer history.",
            kind="length",
        )
    best_aic = min(rows, key=lambda r: r["aic"])
    best_bic = min(rows, key=lambda r: r["bic"])
    return {
        "rows": rows,
        "p_aic": best_aic["p"],
        "p_bic": best_bic["p"],
        # AIC over-fits by construction; BIC under-fits on short samples. Where they disagree the
        # module uses BIC and says so, because an over-fitted VAR produces confident IRFs from noise.
        "p": best_bic["p"] if best_bic["p"] != best_aic["p"] else best_aic["p"],
        "disagreement": best_aic["p"] != best_bic["p"],
    }


def granger_f(y: np.ndarray, x: np.ndarray, p: int) -> dict[str, Any]:
    """
    Does x Granger-cause y? An F-test of the lagged x block, the standard operational definition.

    Restricted model: y on its own lags. Unrestricted: plus x's lags. The question the F-test answers is
    exactly "do x's lags add explanatory power", which is a *predictability* statement.
    """
    n = y.size
    if n - p < p * 2 + 4:
        return {"f": None, "p": None, "note": "too short for the test"}
    Y = y[p:]
    own = [np.ones(Y.size)]
    for lag in range(1, p + 1):
        own.append(y[p - lag: n - lag])
    Xr = np.column_stack(own)
    Xu = np.column_stack(own + [x[p - lag: n - lag] for lag in range(1, p + 1)])

    def rss(X: np.ndarray) -> float:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        r = Y - X @ beta
        return float(r @ r)

    rss_r, rss_u = rss(Xr), rss(Xu)
    df1, df2 = p, Y.size - Xu.shape[1]
    if df2 <= 0 or rss_u <= 0:
        return {"f": None, "p": None, "note": "degenerate residual variance"}
    f = ((rss_r - rss_u) / df1) / (rss_u / df2)
    return {"f": float(f), "p": float(stats.f.sf(f, df1, df2)), "df1": df1, "df2": df2, "note": ""}


def orthonormalise(beta: np.ndarray) -> np.ndarray:
    """Turn Johansen's β into orthonormal columns: only the space matters, not the normalisation."""
    if beta.size == 0:
        return beta
    q, r = np.linalg.qr(beta)
    return q * np.sign(np.diag(r)) if r.size else q


def johansen_trace(data: np.ndarray, det_order: int = 0, k_ar_diff: int = 1) -> dict[str, Any]:
    """
    Johansen's trace statistic, and the eigenvalues it is built from.

    Two-stage least squares on the residuals of Δy and y_{t−1} after removing the short-run dynamics;
    the canonical correlations between those residual sets give the eigenvalues λ_i, and
    trace(r) = −n Σ_{i>r} ln(1 − λ_i) tests "at most r cointegrating relations".
    """
    n, k = data.shape
    dy = np.diff(data, axis=0)
    n_eff = dy.shape[0]
    if n_eff < k * k_ar_diff + k + 8:
        raise FitFailure(
            "The sample is too short for the Johansen test at this lag order.",
            "Use a longer history or fewer series.",
            kind="length",
        )
    # Short-run regressors: a constant plus Δy lags. All three blocks must have the same number of
    # rows (n_eff − k_ar_diff): the first row with a full set of lags is row k_ar_diff of Δy.
    rows = n_eff - k_ar_diff
    Z = [np.ones(rows)]
    for lag in range(1, k_ar_diff + 1):
        Z.append(dy[k_ar_diff - lag: n_eff - lag])
    Z = np.column_stack(Z)
    y_lag = data[k_ar_diff: n - 1]

    def resid(v: np.ndarray) -> np.ndarray:
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    R0 = resid(dy[k_ar_diff:])
    R1 = resid(y_lag)
    assert R0.shape[0] == R1.shape[0] == rows, (R0.shape, R1.shape, rows)
    S00 = (R0.T @ R0) / rows
    S11 = (R1.T @ R1) / rows
    S01 = (R0.T @ R1) / rows
    S10 = S01.T
    try:
        m = np.linalg.solve(S11, S10) @ np.linalg.solve(S00, S01)
        eig = np.real(np.linalg.eigvals(m))
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
        raise FitFailure(f"The Johansen eigenvalue problem is singular: {exc}", "Check for a constant or perfectly collinear series.", kind="numerical")
    eig = np.sort(np.clip(eig, 0.0, 0.999999))[::-1]

    trace = [float(-rows * np.sum(np.log(1 - eig[i:]))) for i in range(len(eig) + 1)]
    crit = _trace_critical_values(k)
    rank = 0
    for r in range(k):
        if r < len(crit) and trace[r] > crit[r][1]:     # 5%
            rank = r + 1
    return {"eigenvalues": eig.tolist(), "trace": trace, "crit_5pct": [c[1] for c in crit],
            "rank": int(rank), "n_eff": int(rows)}


def _trace_critical_values(k: int) -> list[tuple[float, float]]:
    """
    Asymptotic 5% critical values for the trace statistic (no deterministic trend, Osterwald-Lenum).

    Small table, stated openly: interpolation beyond k = 5 would be invention, so the caller is told
    the test is not available rather than handed a number from a different problem.
    """
    table = {
        2: [(3.84, 12.53), (0.0, 3.84)],
        3: [(15.41, 29.68), (3.76, 15.41), (0.0, 3.76)],
        4: [(29.68, 47.21), (15.41, 29.68), (3.76, 15.41), (0.0, 3.76)],
        5: [(47.21, 68.52), (29.68, 47.21), (15.41, 29.68), (3.76, 15.41)],
    }
    return table.get(k, [])


def adf_stat(z: np.ndarray, *, lags: int = 1) -> dict[str, Any]:
    """
    Augmented Dickey–Fuller t statistic on a single series: Δz on z_{t−1} and lagged Δz, with a constant.

    Returns the statistic and the lag order used. The statistic is compared against Engle–Granger
    critical values by the caller, not the standard Dickey–Fuller ones: the series being tested is an
    *estimated* residual, and using the ordinary table there is the classic way to over-reject.
    """
    z = np.asarray(z, dtype=float)
    z = z[np.isfinite(z)]
    n = z.size
    lags = max(0, min(lags, (n - 10) // 3))
    if n - lags - 2 < 10:
        return {"stat": None, "lags": lags, "note": "too short for the test"}
    dz = np.diff(z)
    target = dz[lags:]
    cols = [np.ones(target.size), z[lags: n - 1]]
    for j in range(1, lags + 1):
        cols.append(dz[lags - j: dz.size - j])
    X = np.column_stack(cols)
    if X.shape[0] <= X.shape[1] + 2:
        return {"stat": None, "lags": lags, "note": "too short for the test"}
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    resid = target - X @ beta
    dof = X.shape[0] - X.shape[1]
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return {"stat": None, "lags": lags, "note": "singular design"}
    se = float(np.sqrt(max(cov[1, 1], 1e-300)))
    t = float(beta[1] / se)
    # MacKinnon's response surface for the residual-based cointegration test, constant, no trend:
    # -3.34, -2.85, -2.57 at the 5%, 10% and 20% levels for one regressor. Only the 5% value is used,
    # because that is the level the rank decision is made at.
    return {"stat": t, "lags": lags, "crit_5pct": -3.34, "note": ""}


# --------------------------------------------------------------------------- #
# The fitted model
# --------------------------------------------------------------------------- #
@dataclass
class VarResult:
    """Everything the UI and the report read, kept as plain arrays and dicts."""

    names: list[str]
    p: int
    B: np.ndarray
    resid: np.ndarray
    sigma_u: np.ndarray
    a: np.ndarray                    # companion coefficient matrix
    stable: bool
    max_root: float
    loglik: float
    n_used: int
    granger: dict[str, Any] = field(default_factory=dict)
    irf: dict[str, Any] = field(default_factory=dict)
    fevd: dict[str, Any] = field(default_factory=dict)
    johansen: dict[str, Any] = field(default_factory=dict)
    differenced: bool = False
    caviats: list[str] = field(default_factory=list)


class _MultiBase(ModelWrapper):
    """Shared plumbing for the multivariate family: it needs more than one series."""

    family = ModelFamily.MULTIVARIATE
    supports_multi = True

    def __init__(self, *, multi: pd.DataFrame | None = None, **kw) -> None:
        super().__init__(**kw)
        self._multi = multi
        self._names: list[str] = []
        self._data: np.ndarray | None = None
        self.var_result: VarResult | None = None

    # -- data preparation --------------------------------------------------- #
    def _prepare(self) -> tuple[list[str], np.ndarray, pd.DataFrame]:
        frame = self._multi
        if frame is None or frame.shape[1] < 2:
            # Count what the model actually has: the target, plus any companion columns. Reporting "0"
            # when the only problem is that the driver column was not mapped sends the reader looking
            # for the wrong thing.
            have = 1 if frame is None else int(frame.shape[1])
            extra = "" if have > 1 else " — the target alone"
            raise FitFailure(
                f"A multivariate model needs at least two series and it was given {have}{extra}.",
                "Map a second numeric column as exogenous in Column mapping, or use a single-series "
                "model — the VAR branch exists to describe how series move each other.",
                kind="multivariate",
            )
        frame = pd.DataFrame(frame).astype(float)
        # Interior gaps are interpolated, not dropped: dropping a row shortens every series and shifts
        # the alignment, which silently breaks the lag structure the whole model is built on.
        n_missing = int(frame.isna().sum().sum())
        if n_missing:
            frame = frame.interpolate(limit_direction="both")
        frame = frame.dropna()
        if frame.shape[0] < 30:
            raise FitFailure(
                f"Only {frame.shape[0]} complete rows are available across the selected series.",
                "A VAR needs a longer, gap-free overlap between the series.",
                kind="length",
            )
        names = [str(c) for c in frame.columns]
        self._names = names
        self._data = frame.to_numpy()
        return names, self._data, frame

    def multi_series(self) -> pd.DataFrame | None:
        return self._multi

    # -- IRF and FEVD ------------------------------------------------------- #
    def _irf(self, horizon: int = 12, *, n_boot: int = 200, seed: int = 20260101) -> dict[str, Any]:
        """
        Orthogonalised impulse responses with bootstrap percentile bands.

        The band is the point. A single IRF path computed from one fit cannot distinguish "a shock to x
        raises y by 0.4" from "this fit happened to produce a positive coefficient", and the sign of an
        IRF is exactly where a reader will draw a conclusion.
        """
        r = self.var_result
        if r is None:
            raise FitFailure("The model has not been fitted.", "Fit it first.", kind="internal")
        point = self._irf_paths(r, horizon)
        rng = np.random.default_rng(seed)
        n = r.resid.shape[0]
        boot = {k: [] for k in self._names}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                resample = self._refit_from_resampled(r, idx)
                path = self._irf_paths(resample, horizon)
            except Exception:  # pragma: no cover - a failed bootstrap draw is skipped, not fatal
                continue
            for j, name in enumerate(self._names):
                boot[name].append(path[name])
        bands: dict[str, dict[str, list[float]]] = {}
        for name, mats in boot.items():
            if not mats:
                continue
            arr = np.stack(mats)
            bands[name] = {
                "lower": np.percentile(arr, 5, axis=0).tolist(),
                "upper": np.percentile(arr, 95, axis=0).tolist(),
            }
        return {
            "horizon": horizon,
            "response_to": self._names[0],
            "ordering": self._names,
            "paths": {k: v.tolist() for k, v in point.items()},
            "bands_90": bands,
            "n_boot": len(boot[self._names[0]]),
        }

    def _refit_from_resampled(self, r: VarResult, idx: np.ndarray) -> VarResult:
        """Recursive (case) bootstrap of the residuals, keeping the fitted structure."""
        data = self._data
        p = r.p
        # Rebuild a synthetic history of the same length by feeding resampled residuals through the
        # fitted recursion. Bootstrapping the *residuals* rather than the rows preserves the dependence
        # the model describes, which row-bootstrap does not.
        n, k = data.shape
        sim = np.zeros_like(data)
        sim[:p] = data[:p]
        for t in range(p, n):
            x = np.ones(1 + k * p)
            for lag in range(1, p + 1):
                x[1 + (lag - 1) * k: 1 + lag * k] = sim[t - lag]
            sim[t] = x @ r.B + r.resid[idx[t - p]]
        fit = fit_var_ols(sim, p)
        return VarResult(
            names=self._names, p=p, B=fit["B"], resid=fit["resid"], sigma_u=fit["sigma_u"],
            a=fit["A"], stable=True, max_root=0.0, loglik=fit["loglik"], n_used=fit["n_used"],
        )

    def _chol_factor(self, sigma: np.ndarray) -> np.ndarray:
        try:
            return np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            # A near-singular Σ happens when two series are almost the same series. Pivoting the
            # Cholesky and undoing the permutation keeps the ordering honest instead of quietly
            # switching to a different identification.
            vals, vecs = np.linalg.eigh(sigma)
            vals = np.clip(vals, 1e-12, None)
            return vecs @ np.diag(np.sqrt(vals))

    def _irf_paths(self, r: VarResult, horizon: int) -> dict[str, np.ndarray]:
        k, p = len(r.names), r.p
        chol = self._chol_factor(r.sigma_u)
        out = {name: np.zeros((horizon, k)) for name in r.names}
        # MA(∞) recursion: Θ_h = Σ_{i=1..min(h,p)} Φ_i Θ_{h−i}, with Θ_0 = chol.
        theta = [chol]
        phis = [r.B[1 + i * k: 1 + (i + 1) * k].T for i in range(p)]
        for h in range(1, horizon):
            acc = np.zeros((k, k))
            for i in range(1, min(h, p) + 1):
                acc += phis[i - 1] @ theta[h - i]
            theta.append(acc)
        for j, name in enumerate(r.names):
            resp = np.stack([t[:, j] for t in theta])
            out[name] = resp
        return out

    def _fevd(self, horizon: int = 12) -> dict[str, Any]:
        """
        Forecast-error variance decomposition from the orthogonalised IRFs.

        Cholesky ordering decides which shock is allowed to move the others contemporaneously, so the
        ordering is *returned with the numbers*. Reporting a FEVD without it is reporting one of k!
        decompositions as if it were the decomposition.
        """
        r = self.var_result
        if r is None:
            raise FitFailure("The model has not been fitted.", "Fit it first.", kind="internal")
        paths = self._irf_paths(r, horizon)
        k = len(r.names)
        # contribution[h, target, shock]
        acc = np.zeros((horizon, k, k))
        for shock_idx, name in enumerate(r.names):
            arr = paths[name]
            for h in range(horizon):
                acc[h, :, shock_idx] = np.sum(arr[: h + 1] ** 2, axis=0)
        share = acc / np.maximum(acc.sum(axis=2, keepdims=True), 1e-300)
        return {
            "horizon": horizon,
            "ordering": r.names,
            "shares": {
                target: {
                    shock: share[horizon - 1, i, j]
                    for j, shock in enumerate(r.names)
                }
                for i, target in enumerate(r.names)
            },
            "shares_full": share.tolist(),
        }

    # -- forecasting -------------------------------------------------------- #
    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        """
        The forecast for the **first** series, with intervals from the VAR's own error covariance.

        A VAR forecasts every series; this interface returns one column, so the first is the target and
        the rest are its predictors. The forecast-error variance of the target at step h is the (0,0)
        element of Σ_{i<h} Θ_i Σ_u Θ_i', which is why the intervals widen with horizon rather than
        staying at the residual standard deviation.
        """
        r = self.var_result
        if r is None:
            raise FitFailure("The model has not been fitted.", "Fit it first.", kind="internal")
        k, p = len(r.names), r.p
        hist = [self._data[-lag] for lag in range(1, p + 1)]
        point = np.zeros((horizon, k))
        cur = list(hist)
        for h in range(horizon):
            x = np.ones(1 + k * p)
            for lag in range(1, p + 1):
                x[1 + (lag - 1) * k: 1 + lag * k] = cur[lag - 1]
            yhat = x @ r.B
            point[h] = yhat
            cur = [yhat] + cur[:-1]
        chol = self._chol_factor(r.sigma_u)
        phis = [r.B[1 + i * k: 1 + (i + 1) * k].T for i in range(p)]
        theta = [np.eye(k)]
        for h in range(1, horizon):
            acc = np.zeros((k, k))
            for i in range(1, min(h, p) + 1):
                acc += phis[i - 1] @ theta[h - i]
            theta.append(acc)
        z = float(stats.norm.ppf(1 - alpha / 2))
        mean = point[:, 0]
        half = np.empty(horizon)
        for h in range(horizon):
            fev = sum(theta[i] @ r.sigma_u @ theta[i].T for i in range(h + 1))
            half[h] = z * np.sqrt(max(float(fev[0, 0]), 1e-300))
        self._multi_forecast = point
        return pd.DataFrame(
            {"mean": mean, "lower": mean - half, "upper": mean + half},
            index=np.arange(1, horizon + 1),
        )


#: A one-liner for the diagnostics: how much of the target's one-step forecast error the other series
#: account for *in this sample*, as an R². This is the `_fitted` the diagnostics block expects, and it
#: is honest about being in-sample — it is never shown as a forecast.
def _in_sample_fit(data: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    One-step-ahead in-sample predictions for column 0, aligned to the rows the model used.

    The lag count is read off `B` rather than passed in: a differenced fit reports *levels* coefficients
    with one more lag than it estimated, and passing the estimated order here would mis-align every
    fitted value by a row.
    """
    k = data.shape[1]
    s = (B.shape[0] - 1) // k
    _, X = _lag_matrix(data, s)
    return X @ B[:, 0]


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _add_companion_shift(B: np.ndarray, k: int, s: int) -> np.ndarray:
    """
    Add a unit root to *every* series, so the system is a VAR in differences rather than in levels.

    If y ~ I(1), Δy = c + π y_{t−1} + Γ₁ Δy_{t−1} + … is the same model as
    y = c' + A₁ y_{t−1} + … , with A₁ = I + π and Aⱼ = −Γ_{j−1} + Γⱼ. The VAR reported for a differenced
    system must be the *levels* coefficients, because that is the model a forecast and an IRF are
    computed from; leaving the differenced coefficients in place makes both answer a question about a
    different series (the change) while the panel says "the level".
    """
    A = np.zeros((k, k * (s + 1)))
    A[:, :k] = np.eye(k) + B[1: 1 + k].T
    for j in range(1, s + 1):
        cur = B[1 + j * k: 1 + (j + 1) * k].T if 1 + (j + 1) * k <= B.shape[0] else np.zeros((k, k))
        prev = B[1 + (j - 1) * k: 1 + j * k].T
        A[:, j * k: (j + 1) * k] = cur - prev
    return A


# --------------------------------------------------------------------------- #
# VAR
# --------------------------------------------------------------------------- #
class VARModel(_MultiBase):
    """
    Vector autoregression over the target and its mapped companions (§5.8 B).

    Order is chosen by BIC with AIC reported alongside, and the stability check is not decoration: an
    unstable VAR's impulse responses grow without bound, and a reader shown one without the root check
    would conclude that a shock permanently raises everything.
    """

    model_id = "var"
    display_name = "VAR"
    family = ModelFamily.MULTIVARIATE

    def __init__(self, *, d_hint: int = 0, horizon: int = 12, terms: str | None = None, **kw) -> None:
        super().__init__(**kw)
        self._d_hint = int(d_hint)
        # The spec lists no deterministic terms as the minimum, a constant as the common choice and a
        # trend as the one to avoid. A trend is available but always carries its caveat.
        self._terms = terms or "const"
        self._horizon = int(horizon)

    def _fit(self, y: pd.Series) -> None:
        names, data, frame = self._prepare()
        # The target is column 0 by construction (the caller puts it there). Checked rather than
        # assumed: a mislabelled target makes every IRF be read as a response of the wrong series.
        target = pd.Series(y).reindex(frame.index)
        if target.notna().sum() > 5 and not np.allclose(
            frame.iloc[:, 0].to_numpy(), target.to_numpy(), equal_nan=True, rtol=1e-6, atol=1e-8
        ):
            raise FitFailure(
                "The target series and the first column of the multivariate frame disagree.",
                "This is an internal wiring problem in the fit step — please report it.",
                kind="internal",
            )

        caveats: list[str] = []
        if self._terms == "trend":
            caveats.append(
                "A deterministic trend was requested. Two trending series regressed on each other "
                "produce significant coefficients almost regardless of any real relationship (the "
                "spurious-regression problem); the Johansen result is what justifies the choice."
            )

        # --- cointegration decides whether levels are admissible ------------------------ #
        differenced = False
        work = data
        joh: dict[str, Any] = {}
        if self._d_hint == 0:
            try:
                joh = johansen_trace(data, k_ar_diff=1)
            except FitFailure as exc:
                joh = {"error": exc.plain}
            if joh.get("error"):
                differenced = True
                caveats.append(
                    "The cointegration test could not run, so the VAR was fitted to first differences. "
                    "Fitting levels without that test risks a spurious fit between trending series."
                )
            elif int(joh["rank"]) == 0:
                differenced = True
                caveats.append(
                    f"Johansen finds no stationary linear combination at the 5% level (trace "
                    f"{joh['trace'][0]:.2f} against a critical value of {joh['crit_5pct'][0]:.2f}), so "
                    "nothing ties the levels together and a level VAR would be a spurious regression. "
                    "Fitted to first differences; the responses are responses to shocks to *changes*."
                )
            else:
                caveats.append(
                    f"Johansen finds {int(joh['rank'])} cointegrating relation(s), which is what makes a "
                    "level VAR defensible. The long-run relationship itself is the VECM row's job."
                )
        else:
            differenced = True
            caveats.append(
                f"The series was already differenced {self._d_hint}× at Step 4, so the VAR is fitted to "
                "the stationary transform the user chose."
            )

        if differenced:
            work = np.diff(data, axis=0)

        order = select_order(work)
        p = order["p"]
        if order["disagreement"]:
            caveats.append(
                f"AIC prefers {order['p_aic']} lag(s) and BIC prefers {order['p_bic']}; BIC was used "
                "because an over-fitted VAR produces confident impulse responses from noise."
            )
        fit = fit_var_ols(work, p)
        stable, max_root = var_stability_ok(fit["A"])
        if not stable:
            raise FitFailure(
                f"The VAR is explosive: the largest companion root has modulus {max_root:.4f} > 1, so a "
                "shock would grow without bound and the impulse responses would not decay.",
                "Reduce the lag order or difference the series — an explosive VAR's impulse responses "
                "are not interpretable.",
                kind="stability",
            )

        k = len(names)
        # For a differenced fit the reported system is in levels: add the unit root back so the
        # forecast and the IRF describe the series the user is looking at, not its change.
        B_levels = fit["B"] if not differenced else self._reparameterise_levels(fit["B"], k, p)
        self.var_result = VarResult(
            names=names,
            p=p,
            B=B_levels,
            resid=fit["resid"],
            sigma_u=fit["sigma_u"],
            a=_companion_from_B(B_levels, k, p),
            stable=True,
            max_root=max_root,
            loglik=fit["loglik"],
            n_used=fit["n_used"],
            caviats=caveats,
            johansen=joh,
            differenced=differenced,
        )

        # --- Granger, every direction, on levels *and* on differences -------------------------- #
        # The level test is the one the model implies; the differenced test is the one that survives a
        # trend. They answer the same question and disagree exactly where a level test is untrustworthy:
        # two unrelated random walks reject on levels almost every time (the spurious-regression
        # artifact) and stop rejecting once differenced. Reporting only the level version — which is
        # what every tutorial does — produces a confident "x predicts y" for two independent walks.
        gr: dict[str, Any] = {}
        for i, nm in enumerate(names):
            for j, other in enumerate(names):
                if i == j:
                    continue
                # The system is estimated on `work`; the test that can be spurious is the one on
                # levels, so that is the version paired with a differenced second opinion.
                level_g = granger_f(work[:, i], work[:, j], p)
                if differenced:
                    # Already differenced: report the levels version as the second opinion, since the
                    # failure mode being guarded against is a trend in the levels.
                    level_g["on_differences"] = level_g
                    level_g["tested_on"] = "differences"
                    diff_g = level_g
                else:
                    diff_g = granger_f(np.diff(data[:, i]), np.diff(data[:, j]), p)
                    level_g["tested_on"] = "levels"
                level_g["on_differences"] = diff_g
                gr[f"{other} -> {nm}"] = level_g
        self.var_result.granger = gr

        # --- IRF and FEVD, computed once so the panel and the report read the same numbers ------- #
        h = max(4, min(self._horizon, 40))
        self.var_result.irf = self._irf(h)
        self.var_result.fevd = self._fevd(h)

        # --- bookkeeping the base class reads --------------------------------------------------- #
        self._n_obs_used = int(fit["n_used"])
        self._n_params = int(k * (k * p + 1))
        self._loglik = float(fit["loglik"])
        self._differenced = differenced
        self._p = p
        self._order_rows = order["rows"]
        self._johansen = joh
        self.fitted_on = "first differences" if differenced else "levels"
        self._resid = fit["resid"][:, 0]
        # In-sample one-step predictions for the target, over the rows the model used. This is what the
        # accuracy block compares against — a constant would give MASE ≈ 0 and claim perfection.
        self._fitted = _in_sample_fit(data, B_levels)

    def _reparameterise_levels(self, B_diff: np.ndarray, k: int, p: int) -> np.ndarray:
        """Map a VAR in differences (of order p) to levels coefficients (of order p + 1)."""
        s = p + 1
        B = np.zeros((1 + k * s, k))
        B[0] = B_diff[0]
        A = _add_companion_shift(B_diff, k, p)
        for j in range(1, s + 1):
            B[1 + (j - 1) * k: 1 + j * k] = A[:, (j - 1) * k: j * k].T
        return B

    # -- tables ------------------------------------------------------------- #
    def granger_table(self) -> list[dict[str, Any]]:
        r = self.var_result
        assert r is not None
        rows = []
        # Only a level fit can be spurious-in-levels; on a differenced fit the second opinion is the
        # same test twice and flagging it would invent a caveat out of nothing.
        levels_fit = not bool(r.differenced)
        for pair, g in r.granger.items():
            cause, effect = [x.strip() for x in pair.split("->")]
            p_val = g.get("p")
            if p_val is None:
                sentence = f"{cause} → {effect}: {g.get('note') or 'not testable'}"
            else:
                # "predicts", never "causes": the F-test has no causal content and a sentence that says
                # otherwise is the single most common misreading of this table.
                sentence = (
                    f"{cause} helps predict {effect} (F = {g['f']:.2f}, {p_phrase(p_val, 3)})"
                    if p_val < 0.05
                    else f"{cause} does not help predict {effect} (F = {g['f']:.2f}, {p_phrase(p_val, 3)})"
                )
            # A level rejection that disappears once the trend is removed is the spurious-regression
            # signature, and the row says so instead of letting the headline sentence stand alone.
            d = g.get("on_differences") or {}
            dp = d.get("p")
            spurious = bool(
                levels_fit and p_val is not None and p_val < 0.05 and dp is not None and dp >= 0.05
            )
            if spurious:
                sentence += (
                    f" — but not once both series are differenced ({p_phrase(dp, 3)}), so this is most "
                    f"likely the trend rather than a relationship"
                )
            rows.append({
                "cause": cause, "effect": effect, "f": g.get("f"), "p": p_val,
                "p_on_differences": dp,
                "significant": bool(p_val is not None and p_val < 0.05),
                "spurious_risk": spurious,
                "sentence": sentence,
            })
        return rows

    def fevd_table(self) -> dict[str, Any]:
        assert self.var_result is not None
        return self.var_result.fevd

    def irf_table(self) -> dict[str, Any]:
        assert self.var_result is not None
        return self.var_result.irf

    def multivariate_payload(self) -> dict[str, Any]:
        """Everything the multivariate panel needs, in one object."""
        r = self.var_result
        assert r is not None
        return {
            "model_id": self.model_id,
            "series": r.names,
            "lag_order": r.p,
            "fitted_on": self.fitted_on,
            "stability": {"stable": r.stable, "max_root": r.max_root},
            "order_selection": self._order_rows,
            "johansen": {k: v for k, v in (self._johansen or {}).items() if k in ("rank", "trace", "crit_5pct")},
            "granger": self.granger_table(),
            "irf": r.irf,
            "fevd": r.fevd,
        }

    # -- summary ------------------------------------------------------------ #
    def _summary(self) -> FittedSummary:
        r = self.var_result
        assert r is not None
        aic, aicc, bic = self.ic(self._n_obs_used, self._n_params, self._loglik)
        sig = {nm: float(self._data[:, i].std()) for i, nm in enumerate(self._names)}
        sig_txt = ", ".join(f"{nm} {v:.3g}" for nm, v in sig.items())
        wins = [row for row in self.granger_table() if row["significant"] and not row["spurious_risk"]]
        one = (
            f"A {r.p}-lag VAR over {len(r.names)} series ({sig_txt}), fitted to "
            f"{self.fitted_on}, stable (largest companion root {r.max_root:.3f})."
        )
        spurious = [row for row in self.granger_table() if row.get("spurious_risk")]
        if wins:
            # Capitalise the first letter only: `str.capitalize()` would lower-case "F = 98.03".
            one += " " + " ".join(_cap(row["sentence"]) + "." for row in wins[:2])
        elif spurious:
            one += (
                f" {len(spurious)} directional test(s) reject on levels but not once the trend is "
                "removed — that is the spurious-regression signature, so no relationship is claimed. "
                "This is a result, not a failure."
            )
        else:
            one += (
                " No directional test finds predictive content — as far as this model can see the "
                "series are unrelated, which is a result rather than a failure."
            )
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=aic,
            aicc=aicc,
            bic=bic,
            n_obs_used=self._n_obs_used,
            fitted_on=self.fitted_on,
            roots_ok=bool(r.stable),
            roots_note=f"largest companion root {r.max_root:.3f} < 1" if r.stable else "unstable",
            is_baseline=False,
            plain_one_liner=one,
            params=self.params_table(self._param_rows()),
            caveats=[*r.caviats, *self._caveats],
            ic_comparable=True,
            ic_note=(
                "Compared on the system likelihood. Comparable with other multivariate rows, not with a "
                "single-equation model whose likelihood covers only one series."
            ),
        )

    def _param_rows(self) -> dict[str, tuple[float | None, float | None, bool]]:
        r = self.var_result
        assert r is not None
        k, target = len(r.names), r.names[0]
        rows: dict[str, tuple[float | None, float | None, bool]] = {"const": (float(r.B[0, 0]), None, False)}
        for lag in range(1, r.p + 1):
            for j, nm in enumerate(r.names):
                rows[f"{nm}(t−{lag}) → {target}"] = (
                    float(r.B[1 + (lag - 1) * k + j, 0]), None, False,
                )
        rows["largest companion root"] = (r.max_root, None, r.max_root >= 0.98)
        return rows


# --------------------------------------------------------------------------- #
# VECM
# --------------------------------------------------------------------------- #
class VECMModel(_MultiBase):
    """
    Vector error correction: short-run dynamics *plus* the long-run relationship (§5.8 B).

    This is the model the spec calls for when the series are cointegrated — they wander but stay tied
    together — and the one place a "long-run relationship" is estimated and then testable rather than
    asserted. The adjustment speed α says how fast a deviation is pulled back; a speed that is not
    significantly negative means the series drift apart instead of correcting.
    """

    model_id = "vecm"
    display_name = "VECM"
    family = ModelFamily.MULTIVARIATE

    def __init__(self, *, rank: int | None = None, **kw) -> None:
        super().__init__(**kw)
        self._rank = rank

    def _fit(self, y: pd.Series) -> None:
        names, data, _ = self._prepare()
        k = len(names)
        joh = johansen_trace(data, k_ar_diff=1)
        rank = int(self._rank or joh["rank"])
        if rank < 1:
            raise FitFailure(
                "A VECM needs at least one cointegrating relation and Johansen finds none "
                f"(trace {joh['trace'][0]:.2f} against a 5% critical value of {joh['crit_5pct'][0]:.2f}).",
                "There is no long-run relationship for the model to correct toward — use the VAR on "
                "differences instead, which makes no such claim.",
                kind="cointegration",
            )
        rank = min(rank, k - 1)

        n = data.shape[0]
        dy = np.diff(data, axis=0)
        # Alignment: Δy_t for t = 2..n−1 against y_{t−1} and Δy_{t−1}. Written out because an
        # off-by-one here shifts the error-correction term by one period and silently weakens α.
        rows = n - 2
        dy_t = dy[1:]                       # Δy_2 .. Δy_{n−1}  (n−2 rows)
        dy_prev = dy[:-1]                   # Δy_1 .. Δy_{n−2}
        y_prev = data[1: n - 1]             # y_1 .. y_{n−2}
        assert dy_t.shape[0] == rows and dy_prev.shape[0] == rows and y_prev.shape[0] == rows
        Z = np.column_stack([np.ones(rows), dy_prev])

        def resid(v: np.ndarray) -> np.ndarray:
            beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
            return v - Z @ beta

        R0, R1 = resid(dy_t), resid(y_prev)
        S00 = (R0.T @ R0) / rows
        S11 = (R1.T @ R1) / rows
        S01 = (R0.T @ R1) / rows
        m = np.linalg.solve(S11, S01.T) @ np.linalg.solve(S00, S01)
        vals, vecs = np.linalg.eig(m)
        pick = np.argsort(np.real(vals))[::-1][:rank]
        beta = orthonormalise(np.real(vecs[:, pick]))
        # Normalise each relation on the first series ("one unit of the target against …"), the only
        # scaling a reader can interpret; the space is what is identified, so this is a choice, not a
        # finding, and it is stated.
        for j in range(beta.shape[1]):
            if abs(beta[0, j]) > 1e-8:
                beta[:, j] = beta[:, j] / beta[0, j]
        alpha = (S01 @ beta) @ np.linalg.inv(beta.T @ S11 @ beta)

        ect = y_prev @ beta
        Xfull = np.column_stack([Z, ect])
        # Column layout: [const | Γ on Δy_{t−1} | α on the error-correction terms]. Read with the same
        # offsets the fit used — an off-by-one here silently loads α into Γ, and the implied level
        # system then has the wrong roots and looks explosive when it is not.
        assert Xfull.shape[1] == 1 + k + rank
        coef, *_ = np.linalg.lstsq(Xfull, dy_t, rcond=None)
        fitted = Xfull @ coef
        resid_m = dy_t - fitted
        dof = max(1, rows - Xfull.shape[1])
        sigma_u = (resid_m.T @ resid_m) / dof
        # Standard errors for α come from the regression's own covariance, not from σ/√n: the error-
        # correction term has its own spread, and dividing by √n instead makes a −0.91 speed with a
        # t of −6 into an "insignificant" −0.07. That mistake is easy to make and it inverts the
        # module's headline sentence.
        try:
            XtX_inv = np.linalg.inv(Xfull.T @ Xfull)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            XtX_inv = np.linalg.pinv(Xfull.T @ Xfull)
        loglik = -0.5 * rows * (
            k * np.log(2 * np.pi) + np.log(max(float(np.linalg.det(sigma_u)), 1e-300)) + k
        )

        # The error-correction form Δy = c + αβ'y_{t−1} + Γ Δy_{t−1} is a VAR(2) in levels with
        # A₁ = I + αβ' + Γ₁, A₂ = −Γ₁. Built explicitly so the base forecast and the IRF machinery —
        # which know only about `B` — describe the level series correctly.
        gamma_cols = coef[1: 1 + k].T            # Γ₁ (k×k)
        alpha = coef[1 + k: 1 + k + rank].T      # α (k×r), re-estimated jointly with Γ
        A1 = np.eye(k) + alpha @ beta.T + gamma_cols
        A2 = -gamma_cols
        # `B` is the *level* VAR(2): y_t = c + A₁ y_{t−1} + A₂ y_{t−2} + u_t, which is exactly the
        # error-correction form rearranged. The base forecast and the IRF work off this, so they
        # describe the series the user sees rather than its first difference.
        B = np.vstack([coef[0:1], A1.T, A2.T])

        self._alpha, self._beta = alpha, beta
        self._gamma = gamma_cols
        self._alpha_se = np.array([
            float(np.sqrt(max(sigma_u[i, i], 1e-300) * XtX_inv[1 + k + j, 1 + k + j]))
            for i in range(k) for j in range(rank)
        ]).reshape(k, rank) if rank else np.zeros((k, 0))
        self._johansen = joh
        self._rank = rank
        self._names = names
        self._p = 2
        self._n_obs_used = int(rows)
        self._n_params = int(k * rank * 2 + 1 * k + k * k)
        self._loglik = float(loglik)
        self._resid = resid_m[:, 0]
        self._differenced = True
        # --- does the estimated relation actually stationarise? ------------------------------ #
        # Johansen's trace test has a real false-positive rate (two independent random walks reject at
        # the level it is run at a few per cent of the time), and a spurious rank produces a VECM whose
        # "long-run relationship" is an artefact. The residual-based check is the standard second
        # opinion: if β'y is not stationary by its own ADF test against Engle–Granger critical values,
        # there is no relationship to correct toward whatever the trace test said.
        eg = adf_stat(ect[:, 0])
        self._eg = eg
        if eg.get("stat") is not None and eg["stat"] > eg["crit_5pct"]:
            raise FitFailure(
                f"The estimated relation is not stationary: an ADF test on β'y gives "
                f"{eg['stat']:.2f}, against an Engle–Granger 5% critical value of {eg['crit_5pct']:.2f}.",
                "Johansen found a rank at the level it was run at, but the relation it produced does not "
                "hold together — on unrelated trending series this is the spurious-cointegration case. "
                "Use the VAR on differences, or a rank of 0, which is what this test effectively says.",
                kind="cointegration",
            )

        self.fitted_on = "differences with an error-correction term"
        self.var_result = VarResult(
            names=names, p=2, B=B, resid=resid_m, sigma_u=sigma_u, a=_companion_from_B(B, k, 2),
            stable=True, max_root=0.0, loglik=float(loglik), n_used=int(rows), johansen=joh,
            differenced=True,
            caviats=[
                f"Johansen finds {rank} cointegrating relation(s) at the 5% level, and an "
                f"Engle–Granger ADF test on β'y gives {eg.get('stat'):.2f} against a 5% critical value "
                f"of {eg.get('crit_5pct'):.2f}, so the relation does stationarise — the extra check "
                f"exists because the trace test alone has a real false-positive rate on independent "
                f"trending series. The long-run "
                f"coefficients are normalised on {names[0]}.",
                "The IRF and FEVD below describe the implied level VAR(2), not the differenced form, "
                "so they can be read against the series the user is actually looking at.",
            ],
        )
        # The implied level system must have exactly k − rank unit roots: those are the trends the
        # relations do not cancel. More than that means the data has a trend nothing in the model
        # explains; fewer means a relation was counted twice. Either way it is explosive and refused.
        stable, max_root = var_stability_ok(self.var_result.a)
        n_unit = unit_root_count(self.var_result.a)
        expected = k - rank
        self.var_result.stable, self.var_result.max_root = stable, max_root
        if not stable or n_unit != expected:
            raise FitFailure(
                f"The VECM's implied level system is "
                + ("explosive (largest companion root " f"{max_root:.4f} > 1)" if not stable
                   else f"mis-specified: it has {n_unit} unit root(s) where {expected} is implied by "
                        f"rank {rank} over {k} series"),
                "Lower the rank by one, or use the VAR on differences — a system with the wrong number "
                "of unit roots produces forecasts that wander off.",
                kind="stability",
            )
        self.var_result.irf = self._irf(12)
        self.var_result.fevd = self._fevd(12)
        # The adjustment speeds are reported with their t statistics in the summary; the Granger block
        # is the VAR row's job and is left empty here rather than borrowed from a different model.
        self.var_result.granger = {}
        # In-sample one-step predictions for the target, from the implied level system — same
        # computation the VAR uses, so the accuracy block compares like with like instead of leaving
        # the VECM row blank in the MASE column.
        self._fitted = _in_sample_fit(data, B)

    def adjustment_table(self) -> list[dict[str, Any]]:
        """α per series with a t statistic, because "the speed is −0.02" alone says nothing."""
        r = self.var_result
        assert r is not None
        out = []
        for i, nm in enumerate(r.names):
            a = float(self._alpha[i, 0])
            se = float(self._alpha_se[i, 0]) if self._alpha_se.size else float("nan")
            t = a / se if se > 0 else None
            out.append({
                "series": nm, "alpha": a, "se": se, "t": t,
                "correcting": bool(a < 0 and t is not None and t < -2),
                "sentence": (
                    f"{nm} adjusts toward the long-run relation at {a:+.3f} per step"
                    if a < 0 and t is not None and t < -2
                    else f"{nm} does not adjust toward the relation ({a:+.3f}, t = "
                         f"{'n/a' if t is None else f'{t:.2f}'})"
                ),
            })
        return out

    def multivariate_payload(self) -> dict[str, Any]:
        r = self.var_result
        assert r is not None
        return {
            "model_id": self.model_id,
            "series": r.names,
            "lag_order": r.p,
            "fitted_on": self.fitted_on,
            "stability": {"stable": r.stable, "max_root": r.max_root},
            "johansen": {
                "rank": self._rank,
                "trace": (self._johansen or {}).get("trace"),
                "crit_5pct": (self._johansen or {}).get("crit_5pct"),
            },
            "beta": self._beta.tolist(),
            "adjustment": self.adjustment_table(),
            "granger": [],
            "irf": r.irf,
            "fevd": r.fevd,
        }

    def _summary(self) -> FittedSummary:
        aic, aicc, bic = self.ic(self._n_obs_used, self._n_params, self._loglik)
        adj = self.adjustment_table()
        corr = [a for a in adj if a["correcting"]]
        beta_txt = ", ".join(f"{nm} {self._beta[i, 0]:+.3f}" for i, nm in enumerate(self._names))
        one = (
            f"{self._rank} long-run relation(s) among {len(self._names)} series: {beta_txt} = 0 "
            f"(normalised on {self._names[0]}). "
        )
        one += (
            f"{len(corr)} series adjust{'' if len(corr) != 1 else 's'} toward it "
            f"({', '.join(a['series'] for a in corr)}); the rest do not."
            if corr
            else "No series adjusts toward it at a significant speed, so it is not acting as an "
                 "attractor — the rank is being carried by the eigenvectors, not by any restoring force."
        )
        rows: dict[str, tuple[float | None, float | None, bool]] = {}
        for i, nm in enumerate(self._names):
            rows[f"β {nm} (normalised)"] = (float(self._beta[i, 0]), None, False)
        for a in adj:
            rows[f"α speed {a['series']}"] = (a["alpha"], None, False)
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=aic,
            aicc=aicc,
            bic=bic,
            n_obs_used=self._n_obs_used,
            fitted_on=self.fitted_on,
            roots_ok=bool(self.var_result and self.var_result.stable),
            roots_note=(
                f"implied level system, largest root {self.var_result.max_root:.3f}, "
                f"{len(self._names) - self._rank} unit root(s) as cointegration requires"
                if self.var_result else ""
            ),
            is_baseline=False,
            plain_one_liner=one,
            params=self.params_table(rows),
            caveats=[
                f"Johansen rank {self._rank} was chosen at the 5% level. A rank chosen at 1% would "
                "change the number of relations and every coefficient with it.",
                *self._caveats,
            ],
            ic_comparable=True,
            ic_note=(
                "System likelihood on the differenced form. Comparable with other multivariate rows on "
                "the same series set; not with a single-equation model."
            ),
        )


#: Rows this module fits. Merged into `fit.IMPLEMENTED`.
IMPLEMENTED_MULTIVARIATE: dict[str, type[ModelWrapper]] = {
    "var": VARModel,
    "vecm": VECMModel,
}

#: Declared but not fitted, with the reason the registry displays.
UNIMPLEMENTED_MULTIVARIATE: dict[str, str] = {}

__all__ = [
    "IMPLEMENTED_MULTIVARIATE",
    "UNIMPLEMENTED_MULTIVARIATE",
    "VARModel",
    "VECMModel",
    "VarResult",
    "companion_roots",
    "fit_var_ols",
    "granger_f",
    "johansen_trace",
    "orthonormalise",
    "select_order",
    "var_stability_ok",
]
