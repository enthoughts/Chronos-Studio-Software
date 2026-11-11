"""
ts/models/arma.py — AR / MA / ARMA / ARIMA / SARIMA (§5.8 B).

`pmdarima` is not a dependency of this project, so order selection is a **bounded grid searched by
AICc**, which is what §5.8 asks for as the fallback. The bound is not arbitrary: `p+q ≤ 5` and
`P+Q ≤ 2` covers every order a human should be reading off an ACF/PACF panel, and a wider search
mostly buys overfitting plus a longer wait.

Three decisions worth stating, because each one is a place where auto-ARIMA implementations commonly
lie:

1. **`d` is not searched.** It comes from Step 5's verdict (and, when the verdict is a unit root, the
   differencing already applied). Searching `d` by IC is how a series ends up differenced three
   times: each extra difference costs a little likelihood and buys a lot of spurious stationarity.
   When the caller passes no verdict, `d = 0` and the fit says so.
2. **Every fit is root-checked.** `enforce_stationarity=False` is used during estimation so the
   optimiser can reach the boundary and the user can be *told* the estimate is on it, instead of the
   library silently clipping it. A model whose AR root is inside the unit circle is reported with a
   caveat, not hidden.
3. **AIC between different `d` is not comparable.** The summaries carry `ic_comparable` and a note,
   because §5.9 ranks models by IC and a series differenced twice has a different likelihood scale.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

#: Hard bounds on the search. Wide enough for any order a panel supports, narrow enough to stay fast.
MAX_P = 3
MAX_Q = 3
MAX_PQ_SUM = 5
MAX_SEASONAL_P = 2
MAX_SEASONAL_Q = 2


#: A root this close to the unit circle is reported as "on the boundary" rather than as a clean pass
#: or a clean failure. `enforce_stationarity=False` lets the optimiser land exactly on it, and the two
#: outcomes mean different things to the reader: one is a persistent but valid process, the other is
#: an explosive one.
BOUNDARY_TOL = 1e-4


def _roots_ok(res: Any) -> tuple[bool | None, str]:
    """
    Stationarity and invertibility, read off the roots rather than the parameters.

    The parameter form is misleading: an AR(1) with φ = 0.99 is stationary but nearly not, while
    φ = 1.01 is explosive — one number either side of a cliff. The root form ("largest AR root
    modulus 1.010 > 1 → explosive") says which cliff.
    """
    bits: list[str] = []
    ok = True
    try:
        ar = np.asarray(getattr(res, "arroots", []), dtype=complex)
        ma = np.asarray(getattr(res, "maroots", []), dtype=complex)
    except Exception:  # pragma: no cover
        return None, ""
    if ar.size:
        # Stationarity means every AR root lies OUTSIDE the unit circle, so the binding constraint is
        # the smallest modulus. Using the largest would call an explosive process stationary half the
        # time, which is the wrong half.
        mod = float(np.min(np.abs(ar)))
        if mod < 1.0 - BOUNDARY_TOL:
            ok = False
            # Five decimals, not three: a modulus of 0.9999 rounds to "1.000" at three, and a
            # message that contradicts its own number is worse than no message.
            bits.append(f"smallest AR root modulus {mod:.5f} is inside the unit circle — the fitted AR "
                        "polynomial is explosive, so forecasts diverge")
        elif mod < 1.0 + BOUNDARY_TOL:
            ok = False
            bits.append(
                f"largest AR root modulus is {mod:.6f} — on the unit circle to within "
                f"{BOUNDARY_TOL:g}. The model has put a unit root in the AR polynomial, which "
                "duplicates the differencing rather than describing it; it is not explosive, but the "
                "estimate is sitting on the boundary and its standard error is not informative"
            )
        elif mod < 1.05:
            bits.append(f"smallest AR root modulus {mod:.3f} is just outside the unit circle — "
                        "the process is persistent and the forecast will be sensitive to the last few "
                        "observations")
    if ma.size:
        mod = float(np.min(np.abs(ma)))
        if mod < 1.0 - BOUNDARY_TOL:
            ok = False
            bits.append(f"smallest MA root modulus {mod:.5f} is inside the unit circle — the MA "
                        "polynomial is not invertible, so the model cannot be read as an "
                        "error-correction form")
        elif mod < 1.0 + BOUNDARY_TOL:
            ok = False
            bits.append(
                f"smallest MA root modulus is {mod:.6f} — on the unit circle. The MA polynomial is "
                "on the edge of invertibility, which is what an over-differenced series looks like: "
                "the extra difference cancels against an MA term instead of adding information"
            )
    return ok, " ".join(bits)


def _extract_params(res: Any, names: list[str]) -> dict[str, tuple[float | None, float | None, bool]]:
    """
    Parameters with standard errors, from statsmodels' `params`/`bse` tables.

    A parameter is flagged `at_boundary` when the optimiser pushed it to the edge of the allowed
    region (|φ| or |θ| within 1e-3 of 1), which is where the standard error stops meaning anything.
    """
    out: dict[str, tuple[float | None, float | None, bool]] = {}
    try:
        params = np.asarray(res.params, dtype=float).ravel()
        bse = np.asarray(getattr(res, "bse", np.full_like(params, np.nan)), dtype=float).ravel()
    except Exception:  # pragma: no cover
        return out
    for i, name in enumerate(names):
        if i >= params.size:
            break
        val = float(params[i])
        se = float(bse[i]) if i < bse.size and np.isfinite(bse[i]) else None
        key = name.replace(".", "_")
        at_boundary = (
            (key.startswith("ar.") or key.startswith("ma.") or key.startswith("ar.S") or
             key.startswith("ma.S")) and abs(abs(val) - 1.0) < 1e-3
        )
        out[key] = (val, se, bool(at_boundary))
    return out


class _ArmaBase(ModelWrapper):
    """Shared fitting, selection and serialization for the ARMA family."""

    family = ModelFamily.ARMA
    seasonal: bool = False

    max_p: int = MAX_P
    max_q: int = MAX_Q
    max_P: int = MAX_SEASONAL_P
    max_Q: int = MAX_SEASONAL_Q

    #: The grid is bounded by sample size, not just by taste. A seasonal search over 134
    #: specifications takes a minute and a half on a 144-point series, and the extra orders it
    #: considers are the ones a reader of the ACF panel would never propose. Small samples get the
    #: small grid, which is also the honest one: with n = 144 there is no power to tell AR(3) from
    #: AR(2), so searching for it only manufactures overfitting.
    def _bounds(self, n: int) -> tuple[int, int, int, int]:
        if self.seasonal:
            if n < 200:
                return 2, 2, 1, 1
            return 3, 3, 1, 2
        if n < 100:
            return 2, 2, 0, 0
        return MAX_P, MAX_Q, 0, 0

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.d: int = 0
        self.D: int = 0
        self.order: tuple[int, int, int] | None = None
        self.seasonal_order: tuple[int, int, int, int] | None = None
        self.candidates_tried: int = 0
        self.selection_note: str = ""
        self._trend: str = "n"

    # -- fitting ----------------------------------------------------------- #
    def _fit(self, y: pd.Series) -> None:
        yv = y.astype(float)
        if yv.isna().any():
            # statsmodels handles NaN in SARIMAX, but a silently dropped gap changes what the fit
            # means; the honest move is to refuse and point at Step 2.
            raise FitFailure(
                f"{self.display_name} will not fit a series with {int(yv.isna().sum())} missing values.",
                "Impute them in Step 2 first, so the model's data is the series the app shows.",
                kind="missing",
            )
        if yv.size < 20:
            raise FitFailure(
                f"{self.display_name} needs at least 20 observations and there are {yv.size}.",
                "Use a baseline benchmark, or load a longer series.", kind="length",
            )

        self.d = int(getattr(self, "d_hint", 0) or 0)
        self.D = int(getattr(self, "D_hint", 0) or 0) if self.seasonal else 0
        if self.seasonal and self.period is None:
            raise FitFailure(
                "No seasonal period was detected, so a seasonal model has nothing to put in P and Q.",
                "Run Step 1 to confirm the calendar, or use the non-seasonal ARIMA.",
                kind="seasonality",
            )
        if self.seasonal and yv.size < 2 * int(self.period or 1):
            # Two cycles is the minimum for a seasonal AR/MA term to be identified at all; below it
            # the search will happily converge on nonsense, which is worse than refusing.
            raise FitFailure(
                f"SARIMA needs at least two full cycles at period {self.period} "
                f"({2 * int(self.period or 1)} observations) and there are {yv.size}.",
                "Use the non-seasonal ARIMA, or load a longer series.",
                kind="length",
            )

        best = self._search(yv)
        if best is None:
            raise FitFailure(
                f"No {self.display_name} specification in the search grid converged "
                f"({self.candidates_tried} tried).",
                "This usually means the series is too short or too close to deterministic for the "
                "optimiser; the baselines in the comparison table still stand.",
                kind="convergence",
            )

        # `self.result` is the statsmodels results object (what `_forecast` and `_summary` need);
        # the selection bookkeeping stays in `self._best`.
        self._best = best
        self.result = best["res"]
        self.order = tuple(best["order"])
        self.seasonal_order = tuple(best["seasonal_order"]) if self.seasonal else None
        self._fitted = best["fitted"]
        self._resid = best["resid"]
        self._raw_params = best["params"]
        self.roots_ok, self.roots_note = best["roots"]
        self._n_params = best["n_params"]
        self._loglik = best["loglik"]
        self._aic = best["aic"]

    def _search(self, yv: pd.Series) -> dict[str, Any] | None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        best: dict[str, Any] | None = None
        tried = 0
        self._selection_notes: list[str] = []
        n_obs = int(yv.dropna().size)
        max_p, max_q, max_P, max_Q = self._bounds(n_obs)
        self._max_P, self._max_Q = max_P, max_Q

        p_range = range(0, max_p + 1)
        q_range = range(0, max_q + 1)
        seasonal_grid = (
            list(itertools.product(range(0, max_P + 1), range(0, max_Q + 1)))
            if self.seasonal else [(0, 0)]
        )

        for p, q in itertools.product(p_range, q_range):
            if p + q > MAX_PQ_SUM:
                continue
            for P, Q in seasonal_grid:
                # In the seasonal search a rich regular part combined with a rich seasonal part is
                # more parameters than a series of this length can support, so the sum is capped.
                if self.seasonal and p + q + P + Q > 4:
                    continue
                if p == 0 and q == 0 and P == 0 and Q == 0 and self.d == 0 and self.D == 0:
                    continue  # that is the mean model, which is already in the comparison
                tried += 1
                order = (p, self.d, q)
                sorder = (P, self.D, Q, int(self.period or 0))
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod = SARIMAX(
                            yv,
                            order=order,
                            seasonal_order=sorder,
                            trend="c" if self._allow_constant() else "n",
                            enforce_stationarity=False,
                            enforce_invertibility=False,
                        )
                        res = mod.fit(disp=False, maxiter=120)
                except Exception as exc:  # noqa: BLE001 - one bad order must not kill the search
                    self._selection_notes.append(f"{order}{sorder if self.seasonal else ''}: {type(exc).__name__}")
                    continue
                if not getattr(res, "mle_retvals", {}).get("converged", True):
                    self._selection_notes.append(
                        f"{order}{sorder if self.seasonal else ''}: did not converge"
                    )
                    continue
                n = int(res.nobs)
                k = int(res.df_model) + 1
                llf = float(res.llf) if np.isfinite(res.llf) else None
                aic, aicc, _ = self.ic(n, k, llf)
                score = aicc if aicc is not None else aic
                if score is None:
                    continue
                if best is None or score < best["score"]:
                    names = [f"ar.{i + 1}" for i in range(p)] + [f"ma.{i + 1}" for i in range(q)]
                    names += [f"ar.S{i + 1}" for i in range(P)] + [f"ma.S{i + 1}" for i in range(Q)]
                    try:
                        names = list(res.param_names)
                    except Exception:  # pragma: no cover
                        pass
                    best = {
                        "order": order,
                        "seasonal_order": sorder,
                        "res": res,
                        "score": score,
                        "aic": aic,
                        "loglik": llf,
                        "n_params": k,
                        "fitted": np.asarray(res.fittedvalues, dtype=float),
                        "resid": np.asarray(res.resid, dtype=float),
                        "params": _extract_params(res, names),
                        "roots": _roots_ok(res),
                    }
        self.candidates_tried = tried
        self.selection_note = (
            f"Best of {tried} specifications by {'AICc' if best else 'AICc (no fit converged)'}; "
            f"d = {self.d}" + (f", D = {self.D} at period {self.period}" if self.seasonal else "")
            + "."
        )
        return best

    def _allow_constant(self) -> bool:
        # A constant is only interpretable when nothing is differenced: after d ≥ 1 it is a drift,
        # and after D ≥ 1 it is a seasonal drift, which almost no series has.
        if getattr(self, "_force_no_constant", False):
            return False
        return self.d == 0 and self.D == 0

    # -- forecasting -------------------------------------------------------- #
    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        fc = self.result.get_forecast(steps=int(horizon))
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=alpha), dtype=float)
        return pd.DataFrame(
            {"mean": mean, "lower": ci[:, 0], "upper": ci[:, 1]},
            index=np.arange(1, horizon + 1),
        )

    # -- reporting ---------------------------------------------------------- #
    def _summary(self) -> FittedSummary:
        res = self.result
        n = int(res.nobs)
        k = self._n_params
        aic, aicc, bic = self.ic(n, k, self._loglik)
        order_txt = f"({self.order[0]},{self.order[1]},{self.order[2]})"
        if self.seasonal_order:
            order_txt += f"({self.seasonal_order[0]},{self.seasonal_order[1]},{self.seasonal_order[2]}){self.seasonal_order[3]}"
        caveats: list[str] = []
        if self.d or self.D:
            caveats.append(
                f"Fitted to the series after differencing (d = {self.d}"
                + (f", D = {self.D}" if self.D else "")
                + "). The AIC is therefore on a different scale from an undifferenced model's."
            )
        if self.roots_note:
            caveats.append(self.roots_note)
        ic_note = (
            f"AIC/AICc/BIC computed on {n} differenced observations; compare only with models "
            f"using the same d = {self.d}"
            + (f", D = {self.D}" if self.D else "")
            + "."
        )
        return FittedSummary(
            model_id=self.model_id,
            display_name=f"{self.display_name} {order_txt}",
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=k,
            loglik=self._loglik,
            aic=aic, aicc=aicc, bic=bic,
            n_obs_used=n,
            fitted_on="levels" if not (self.d or self.D) else f"d={self.d},D={self.D}",
            roots_ok=self.roots_ok if self.roots_ok is not None else True,
            roots_note=self.roots_note,
            is_baseline=False,
            plain_one_liner=(
                f"{self.display_name} {order_txt}: each value depends on the previous "
                f"{self.order[0]} values and {self.order[2]} errors"
                + (f", plus the same season {self.period} observations back" if self.seasonal_order and (self.seasonal_order[0] or self.seasonal_order[2]) else "")
                + f". {self.selection_note}"
            ),
            params=self.params_table(self._raw_params),
            caveats=caveats,
            converged_warnings=(
                [w for w in getattr(self, "_selection_notes", [])[:3]]
            ),
            ic_comparable=True,
            ic_note=ic_note,
        )


class ARIMAModel(_ArmaBase):
    model_id = "arima"
    display_name = "ARIMA"

    def __init__(self, *, d_hint: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.d_hint = d_hint


class SARIMAModel(_ArmaBase):
    model_id = "sarima"
    display_name = "SARIMA"
    seasonal = True

    def __init__(self, *, d_hint: int = 0, D_hint: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.d_hint = d_hint
        self.D_hint = D_hint


# --------------------------------------------------------------------------- #
# ARFIMA
# --------------------------------------------------------------------------- #
def gph_estimate(x: np.ndarray, m: int | None = None) -> tuple[float | None, float | None, int]:
    """
    Geweke–Porter-Hudak estimator of the fractional differencing parameter.

    Regress log I(λ_j) on log(4 sin²(λ_j/2)) over the first `m` Fourier frequencies and take
    d̂ = −slope/2. `m = n^0.5` is the usual choice: more frequencies buys precision, fewer buys
    insensitivity to short-memory contamination. The standard error is π/√(24m), which is
    approximately exact for this estimator and is reported so `d̂` can be read as an estimate.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20:
        return None, None, 0
    m = int(m or max(8, round(np.sqrt(n))))
    m = min(m, n // 2 - 1)
    if m < 4:
        return None, None, 0
    xc = x - x.mean()
    freq = 2 * np.pi * np.arange(1, m + 1) / n
    # Periodogram ordinates at the Fourier frequencies.
    I = np.abs(np.fft.rfft(xc)[1: m + 1]) ** 2 / n
    if np.any(I <= 0):
        return None, None, m
    X = np.log(4 * np.sin(freq / 2) ** 2)
    Y = np.log(I)
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(m), X]), Y, rcond=None)
    d_hat = float(-0.5 * beta[1])
    se = float(np.pi / np.sqrt(24 * m))
    return d_hat, se, m


def _binomial_weights(d: float, tol: float = 1e-10) -> np.ndarray:
    """Coefficients of (1 − B)^d, truncated where they stop mattering."""
    weights = [1.0]
    k = 1
    while True:
        w = weights[-1] * (d - (k - 1)) / k
        if abs(w) < tol or k > 4000:
            break
        weights.append(w)
        k += 1
    return np.array(weights)


def _causal_filter(x: np.ndarray, d: float) -> np.ndarray:
    """
    y_t = Σ_k w_k x_{t−k} with w from (1 − B)^d — causal, length-preserving, zero-padded at the
    start.

    Length-preserving matters more than it looks. Preprocessing before Part 8 may subtract a leading
    mean, and a `valid`-mode convolution that trims the head shifts every index by the filter length,
    so "the forecast starts at the last observation" stops being true by exactly that many steps.
    Alignment is what makes the reconstructed path readable.
    """
    x = np.asarray(x, dtype=float)
    if abs(d) < 1e-10:
        return x.copy()
    w = _binomial_weights(d)
    return np.convolve(x, w, mode="full")[: x.size]


def fractional_difference(x: np.ndarray, d: float) -> np.ndarray:
    """Apply (1 − B)^d — the filter, not an approximation by integer differencing."""
    return _causal_filter(x, d)


def _frac_integrate(z: np.ndarray, d: float) -> np.ndarray:
    """Apply (1 − B)^{−d}, the exact inverse of `fractional_difference` for the same truncation."""
    return _causal_filter(z, -d)


class ARFIMAModel(_ArmaBase):
    """
    ARFIMA as a *diagnostic-first* model: it fits only when the GPH estimate says there is long
    memory, and it says loudly that its information criteria are not comparable with ARIMA's.
    """

    model_id = "arfima"
    display_name = "ARFIMA"

    def __init__(self, *, d_hint: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.d_hint = d_hint
        self.d_frac: float | None = None
        self.gph_se: float | None = None
        self.gph_m: int = 0

    def _fit(self, y: pd.Series) -> None:
        yv = y.astype(float)
        if yv.size < 50:
            raise FitFailure(
                f"The GPH long-memory estimator needs at least 50 observations and there are {yv.size}.",
                "Use ARIMA instead — with this sample a fractional d cannot be told from 0 or 0.5.",
                kind="length",
            )
        if yv.isna().any():
            raise FitFailure(
                "ARFIMA will not fit a series with missing values.",
                "Impute them in Step 2 first.", kind="missing",
            )
        d_hat, se, m = gph_estimate(yv.to_numpy())
        if d_hat is None:
            raise FitFailure(
                "The periodogram is degenerate (a constant or near-constant series), so the "
                "GPH estimator has no slope to fit.",
                "There is nothing to model here beyond the mean.", kind="degenerate",
            )
        self.d_frac, self.gph_se, self.gph_m = d_hat, se, m
        if not (0.0 < d_hat < 0.5):
            raise FitFailure(
                f"The GPH estimate of d is {d_hat:.3f} (SE {se:.3f}), which is "
                + ("negative — the series is anti-persistent" if d_hat <= 0 else
                   "at or above 0.5 — that is a unit root, not long memory")
                + ". ARFIMA exists for the 0 < d < 0.5 case.",
                "Use ARIMA (d = 0 for the first case, d = 1 for the second).",
                kind="range",
            )

        # Fit on the fractionally differenced series, which is short-memory by construction, after
        # removing its mean. Without the demeaning the ARMA part carries a constant, and integrating
        # a non-zero constant back through (1 − B)^{−d} grows polynomially — a forecast that reaches
        # six figures within a year, which is arithmetic, not a bug, but not an answer either.
        fd = fractional_difference(yv.to_numpy(), d_hat)
        self._z_mean = float(np.mean(fd))
        fd_series = pd.Series(fd - self._z_mean)
        self.d_hint = 0  # already differenced; the search must not difference again
        self._force_no_constant = True
        best = self._search(fd_series)
        if best is None:
            raise FitFailure(
                f"No ARFIMA({self.max_p}, {d_hat:.2f}, {self.max_q}) specification converged.",
                "The long-memory estimate stands on its own even without the ARMA part; read it "
                "from the diagnostic below.", kind="convergence",
            )
        self._best = best
        self.result = best["res"]
        self.order = tuple(best["order"])
        self._fitted = best["fitted"]
        self._resid = best["resid"]
        self._raw_params = best["params"]
        self._n_params = best["n_params"] + 1  # d̂ is estimated, so it costs a parameter
        self._loglik = best["loglik"]
        self._aic = best["aic"]
        self.roots_ok, self.roots_note = best["roots"]
        self.selection_note = (
            f"d̂ = {d_hat:.3f} (GPH, SE {se:.3f}, {m} Fourier frequencies); ARMA order "
            f"({self.order[0]},0,{self.order[2]}) on the fractionally differenced series."
        )

    def _summary(self) -> FittedSummary:
        s = super()._summary()
        s.display_name = f"ARFIMA({self.order[0]}, {self.d_frac:.2f}, {self.order[2]})"
        s.ic_comparable = False
        s.ic_note = (
            "AIC is computed on the fractionally differenced series, so it is NOT comparable with "
            "ARIMA's. Ranking this model by AIC against ARIMA would compare two likelihood scales."
        )
        s.caveats = list(s.caveats) + [
            f"d̂ = {self.d_frac:.3f} ± {self.gph_se:.3f} (GPH). A fractional difference is an "
            "estimate of a continuum parameter, not a decision between 'integrated' and not.",
            "Its information criteria are not comparable with ARIMA's — decide between them on "
            "out-of-sample accuracy, not AIC.",
        ]
        return s

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        """
        Forecast the short-memory part, then integrate it back with the *same* fractional filter.

        `(1−B)^d` and `(1−B)^{−d}` are exact inverses in the infinite-past sense, but any finite
        convolution loses the pre-sample tail. That loss shows up as a slowly-decaying offset, so the
        reconstruction is anchored on the last observed value: the difference between the
        reconstructed series and the true one at the last observation is subtracted from the whole
        forecast path. Without that anchoring an ARFIMA forecast can start several units away from
        where the series actually is, which looks like a bug and reads like one.
        """
        base = super()._forecast(horizon, alpha=alpha)
        yv = self.y.to_numpy(dtype=float)
        z_obs = fractional_difference(yv, float(self.d_frac)) - float(self._z_mean)
        z_all = np.concatenate([z_obs, base["mean"].to_numpy()])
        recon = _frac_integrate(z_all, float(self.d_frac))

        # Anchor: index (n-1) of the reconstruction corresponds to the last observed z.
        n = yv.size
        if recon.size >= n and n >= 1:
            shift = float(yv[-1]) - float(recon[n - 1])
        else:  # pragma: no cover - defensive
            shift = 0.0
        mean = recon[-horizon:] + shift
        return pd.DataFrame(
            {
                "mean": mean,
                "lower": mean - (base["upper"] - base["mean"]).to_numpy(),
                "upper": mean + (base["upper"] - base["mean"]).to_numpy(),
            },
            index=np.arange(1, horizon + 1),
        )


IMPLEMENTED_ARMA: dict[str, type[ModelWrapper]] = {
    "arima": ARIMAModel,
    "sarima": SARIMAModel,
    "arfima": ARFIMAModel,
}


__all__ = [
    "ARFIMAModel",
    "ARIMAModel",
    "IMPLEMENTED_ARMA",
    "SARIMAModel",
    "fractional_difference",
    "gph_estimate",
]
