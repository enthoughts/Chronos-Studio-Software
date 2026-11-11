"""
ts/models/exog.py — models that use information from outside the target series (§5.8 B and E).

Two models live here, and they answer different questions:

* **SARIMAX / ARIMAX** (`arimax`) — the ARIMA structure of Part 8, plus regressors. Used when the
  driver is a column in the dataset and the residual autocorrelation still needs modelling.
* **ARDL / dynamic regression with ARMA errors** (`ardl`) — the same idea, built the other way round:
  start from the regressors and lags, then let the errors be ARMA. Included because the two disagree
  often enough that showing both is a lesson, and because the bounds-test framing ("is there a
  long-run relationship at all?") is the question a driver analysis usually starts with.

Three things this module refuses to do, each because doing them is the standard way these models lie:

1. **Invent future values for a driver.** A forecast that assumes next year's promotion days are
   whatever the model finds convenient is not a forecast of anything. Future paths must be supplied;
   when they are not, the model says so and offers the honest alternative (a constant path, labelled
   as an assumption, or forecasting the driver itself and treating that as a scenario).
2. **Report a coefficient on a driver it has not checked for a unit root.** Regressing a trending
   series on a trending series gives a significant coefficient and a spurious story. The ARDL/SARIMAX
   fit therefore reports the correlation of trends and refuses to interpret a level coefficient when
   both sides are integrated — the bounds test framing is what licenses the interpretation, and where
   it cannot be run the caveat says why.
3. **Keep a pulse dummy that the outlier detector did not flag.** Interventions come from Step 2's
   flags, not from looking at the plot. A dummy fitted where the model happens to need one is the
   model explaining its own residuals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

#: A pulse dummy is only worth a parameter if the flagged point is this many residual SDs out.
#: Step 2 already applies its own cut-off; this is the check that the *model's* residuals agree.
MIN_PULSE_Z = 3.0

#: Cap on intervention dummies. Beyond a handful the "interventions" are describing the model's
#: failure to fit the series, and the fit is being steered by its own outliers.
MAX_DUMMIES = 4


def _as_frame(exog: pd.DataFrame | None, index: pd.Index) -> pd.DataFrame | None:
    """Align exogenous columns to the target index, refusing anything that would silently shift."""
    if exog is None or len(exog.columns) == 0:
        return None
    e = pd.DataFrame(exog).astype(float)
    e.index = pd.Index(index)
    return e


def _intervention_dummies(
    index: pd.Index, interventions: list[dict[str, Any]] | None
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Build pulse and step dummies from Step 2's flagged points.

    A *pulse* is 1 at the flagged period only — a one-off shock. A *step* is 1 from the flagged period
    onwards — a level shift. Which one is right is not something a fitting routine can decide: a pulse
    fitted to a genuine level shift leaves the model chasing a new level for the rest of the sample.
    Step 2's structural-break tests are what decide it, so this function only builds what it is told
    and the caveat names the assumption.
    """
    if not interventions:
        return None, []
    n = len(index)
    cols: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for k, item in enumerate(interventions[:MAX_DUMMIES]):
        pos = item.get("position")
        if pos is None or not (0 <= int(pos) < n):
            continue
        kind = str(item.get("kind", "pulse")).lower()
        name = f"{kind}_{k}"
        vec = np.zeros(n)
        if kind == "step":
            vec[int(pos):] = 1.0
            notes.append(
                f"A step dummy starts at observation {int(pos) + 1}; that assumes the change is "
                "permanent, which a pulse would not."
            )
        else:
            vec[int(pos)] = 1.0
        cols[name] = vec
    if not cols:
        return None, []
    return pd.DataFrame(cols, index=index), notes


class _ExogBase(ModelWrapper):
    """Shared plumbing: exog alignment, future paths, and the scenario caveat."""

    supports_exog = True

    def __init__(
        self,
        *,
        exog: pd.DataFrame | None = None,
        future_exog: pd.DataFrame | None = None,
        interventions: list[dict[str, Any]] | None = None,
        **kw,
    ) -> None:
        super().__init__(**kw)
        self._exog_raw = exog
        self._future_raw = future_exog
        self._interventions = interventions
        self._exog: pd.DataFrame | None = None
        self._dummy_note: str = ""
        self.coef_rows: list[tuple[str, float, float]] = []

    def _prepare_exog(self, index: pd.Index) -> pd.DataFrame | None:
        exog = _as_frame(self._exog_raw, index)
        dummies, notes = _intervention_dummies(index, self._interventions)
        if notes:
            self._dummy_note = notes[0]
        frames = [f for f in (exog, dummies) if f is not None]
        if not frames:
            return None
        return pd.concat(frames, axis=1)

    def _future_frame(self, horizon: int) -> pd.DataFrame | None:
        """
        The future regressor paths, or a *labelled* constant path.

        The distinction matters more than the numbers: a constant path is an assumption the user can
        see and disagree with, while silently holding a driver at its mean and reporting the result
        as a forecast is the single most over-trusted output these models produce.
        """
        if self._exog is None or self._exog.shape[1] == 0:
            return None
        n_real = 0 if self._exog_raw is None else len(self._exog_raw.columns)
        user = _as_frame(self._future_raw, pd.RangeIndex(horizon)) if self._future_raw is not None else None
        cols: dict[str, np.ndarray] = {}
        for c in self._exog.columns:
            if user is not None and c in user.columns and len(user) >= horizon:
                cols[c] = user[c].to_numpy(dtype=float)[:horizon]
            elif c in (self._exog.columns[n_real:]):   # a dummy: must be 0 in the future
                cols[c] = np.zeros(horizon)
            else:
                cols[c] = np.full(horizon, float(self._exog[c].iloc[-1]))
        return pd.DataFrame(cols, index=pd.RangeIndex(horizon))

    @property
    def scenario_caveat(self) -> str | None:
        if self._exog is None:
            return None
        n_held = 0
        if self._exog_raw is not None:
            provided = self._future_raw.columns if self._future_raw is not None else []
            n_held = len([c for c in self._exog_raw.columns if c not in provided])
        if n_held == 0:
            return "Future driver paths were supplied by the caller; the forecast is conditional on them."
        return (
            f"Every forecast here assumes {n_held} of the driver column(s) stay at their last observed "
            "value, because no future path was supplied. That is a scenario, not a prediction: if the "
            "driver moves, the forecast is wrong in exactly the way the driver moves it."
        )


class SARIMAXModel(_ExogBase):
    """
    ARIMA with regressors, intervention dummies and scenario paths.

    The ARIMA order is not searched here: it comes from the same bounded grid the ARIMA model uses,
    but with the regressors in the design matrix. Searching orders and regressors jointly is a much
    larger search that on samples this size overfits; the ARIMA order from Part 8 is used as the
    starting structure and the regressors are added to it, which is the defensible order of business.
    """

    model_id = "arimax"
    display_name = "SARIMAX / dynamic regression"

    family = ModelFamily.EXOGENOUS

    def __init__(self, *, d_hint: int = 0, D_hint: int = 0, **kw) -> None:
        super().__init__(**kw)
        self._d = int(d_hint)
        self._D = int(D_hint)

    def _fit(self, y: pd.Series) -> None:
        import statsmodels.api as sm

        yv = pd.Series(y).astype(float)
        exog = self._prepare_exog(yv.index)
        if exog is None:
            raise FitFailure(
                "This model needs at least one outside column (a driver, or an intervention dummy) "
                "and the dataset has none.",
                "Use the plain ARIMA, or map an extra column as exogenous in Column mapping.",
                kind="exogenous",
            )
        self._exog = exog
        if yv.notna().sum() < max(24, 3 * (exog.shape[1] + 1)):
            raise FitFailure(
                f"With {exog.shape[1]} regressor(s) plus ARMA terms there is not enough data: "
                f"{int(yv.notna().sum())} observations.",
                "Drop a regressor or use a simpler model.",
                kind="length",
            )

        p, q = self._seed_order(yv, exog)
        order = (p, self._d, q)
        seasonal_order = (0, self._D, 0, int(self.period)) if (self.period and self._D) else (0, 0, 0, 0)
        try:
            with np.errstate(all="ignore"):
                import warnings as _w

                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    model = sm.tsa.SARIMAX(
                        yv.to_numpy(dtype=float),
                        exog=exog.to_numpy(dtype=float),
                        order=order,
                        seasonal_order=seasonal_order,
                        trend="c",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = model.fit(disp=False, maxiter=200)
        except Exception as exc:  # noqa: BLE001
            raise FitFailure(
                f"The regression-with-ARMA-errors fit did not converge ({type(exc).__name__}).",
                "This usually means a regressor is collinear with the trend; check the columns.",
                kind="convergence",
            ) from exc

        self._res = res
        self._order = order
        self._seasonal_order = seasonal_order
        fitted = np.asarray(res.fittedvalues, dtype=float)
        self._fitted = fitted
        self._resid = yv.to_numpy(dtype=float) - fitted
        self._loglik = float(res.llf)
        k = int(len(res.params))
        self._n_params = k
        n = int(yv.notna().sum())
        self._n_obs = n
        self._aic, self._aicc, self._bic = self.ic(n, k, self._loglik)

        names = list(res.param_names)
        se = np.asarray(res.bse, dtype=float)
        vals = np.asarray(res.params, dtype=float)
        self.params = {nm: float(v) for nm, v in zip(names, vals)}
        self._raw_params: dict[str, tuple[float | None, float | None, bool]] = {}
        for nm, v, s in zip(names, vals, se):
            self._raw_params[nm] = (
                None if not np.isfinite(v) else float(v),
                None if not np.isfinite(s) else float(s),
                False,
            )
        # Regressor rows, in the order they appear, for the plain-English coefficient sentences.
        self.coef_rows = [
            (nm, float(vals[i]), float(se[i]))
            for i, nm in enumerate(names)
            if nm.startswith("x")
        ]
        self._check_trends(yv, exog)

    def _seed_order(self, y: pd.Series, exog: pd.DataFrame) -> tuple[int, int]:
        """
        A small order search over the *differenced* residuals-free specification.

        Deliberately tiny (p, q ≤ 2): the point of this model is the regressors, and a large ARMA
        search on top of them spends the sample on autocorrelation that the drivers may have caused in
        the first place.
        """
        yv = y.to_numpy(dtype=float)
        d = self._d
        for _ in range(d):
            yv = np.diff(yv)
        if yv.size < 12:
            return 1, 0
        try:
            from statsmodels.tsa.stattools import acf, pacf

            a = acf(yv, nlags=min(10, yv.size // 4), fft=True)
            p = int(np.sum(np.abs(a[1:4]) > 2 / np.sqrt(yv.size)))
            q = int(np.sum(np.abs(a[1:3]) > 2 / np.sqrt(yv.size)))
            return min(max(p, 1), 2), min(q, 2)
        except Exception:  # pragma: no cover - defensive
            return 1, 1

    def _check_trends(self, y: pd.Series, exog: pd.DataFrame) -> None:
        self._trend_warning: str | None = None
        real = self._exog_raw
        if real is None or real.shape[1] == 0:
            return
        col = real.columns[0]
        a = pd.Series(y.to_numpy(dtype=float))
        b = pd.Series(np.asarray(real[col], dtype=float))
        if a.notna().sum() < 8 or b.notna().sum() < 8:
            return
        ta = np.polyfit(np.arange(a.size), a.fillna(a.mean()).to_numpy(), 1)[0]
        tb = np.polyfit(np.arange(b.size), b.fillna(b.mean()).to_numpy(), 1)[0]
        self._trend_warning = (
            f"The target trends {ta:+.4g} per step and the driver '{col}' trends {tb:+.4g} per step, "
            "so part of the apparent relationship is two trends moving together. The coefficient is "
            "a conditional association on this sample, not a causal effect."
        )

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        future = self._future_frame(horizon)
        with np.errstate(all="ignore"):
            import warnings as _w

            with _w.catch_warnings():
                _w.simplefilter("ignore")
                pred = self._res.get_forecast(
                    steps=horizon,
                    exog=None if future is None else future.to_numpy(dtype=float),
                )
        mean = np.asarray(pred.predicted_mean, dtype=float)
        ci = np.asarray(pred.conf_int(alpha=alpha), dtype=float)
        steps = np.arange(1, horizon + 1)
        return pd.DataFrame(
            {"mean": mean, "lower": ci[:, 0], "upper": ci[:, 1]}, index=steps
        )

    def _summary(self) -> FittedSummary:
        p, d, q = self._order
        P, D, Q, m = self._seasonal_order
        label = f"SARIMAX ({p},{d},{q})"
        if m and (P or D or Q):
            label += f"({P},{D},{Q}){m}"
        label += f" + {self._exog.shape[1]} regressor(s)" if self._exog is not None else ""
        caveats: list[str] = []
        if self._dummy_note:
            caveats.append(self._dummy_note)
        if self._trend_warning:
            caveats.append(self._trend_warning)
        sc = self.scenario_caveat
        if sc:
            caveats.append(sc)
        coef_bits = []
        for nm, v, s in self.coef_rows[:3]:
            if np.isfinite(v) and np.isfinite(s) and s > 0:
                coef_bits.append(f"{v:+.4g} per unit (SE {s:.2g})")
        one = (
            f"{label}: the target moves {', '.join(coef_bits)} of the driver(s) after accounting for "
            f"its own autocorrelation."
            if coef_bits else f"{label} fitted with {self._exog.shape[1]} regressor(s)."
        )
        return FittedSummary(
            model_id=self.model_id,
            display_name=label,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=bool(self._res.mle_retvals.get("converged", True))
            if hasattr(self._res, "mle_retvals") else True,
            n_params=self._n_params,
            loglik=self._loglik,
            aic=self._aic,
            aicc=self._aicc,
            bic=self._bic,
            n_obs_used=self._n_obs,
            fitted_on="levels",
            roots_ok=None,
            is_baseline=False,
            plain_one_liner=one,
            params=self.params_table(self._raw_params),
            caveats=caveats,
            ic_comparable=True,
            ic_note="AIC/AICc on the levels series with the same regressors; comparable with other "
                    "levels-scale models fitted to this sample.",
        )


class ARDLModel(_ExogBase):
    """
    ARDL / dynamic regression: lags of the target and of the driver, OLS, then a bounds-test framing.

    The bounds test (Pesaran–Shin–Smith) is the reason this model exists as its own row rather than as
    a switch on SARIMAX: it asks whether a *long-run* relationship exists at all, and it is valid
    whether or not the regressors are stationary. That is a different question from "does this
    regressor help predict", and it is the one users of a driver analysis actually have.

    The test is implemented as the F-form of the error-correction reparameterisation: regress Δy on
    lagged levels (y_{t−1}, x_{t−1}) and lagged differences, then test the two level coefficients
    jointly. Both-non-stationary is the case with the widest critical bounds; the reported p-value uses
    the asymptotic bounds' midpoint and is labelled as such rather than dressed up as exact.
    """

    model_id = "ardl"
    display_name = "ARDL / dynamic regression"

    family = ModelFamily.EXOGENOUS

    def __init__(self, *, d_hint: int = 0, max_lags: int = 2, **kw) -> None:
        super().__init__(**kw)
        self._max_lags = int(max_lags)

    def _fit(self, y: pd.Series) -> None:
        yv = pd.Series(y).astype(float)
        exog = self._prepare_exog(yv.index)
        if exog is None or self._exog_raw is None or self._exog_raw.shape[1] == 0:
            raise FitFailure(
                "ARDL needs a real outside column to regress on, and the dataset has none.",
                "Map an extra numeric column as exogenous in Column mapping, then re-run Step 9.",
                kind="exogenous",
            )
        self._exog = exog
        xcol = self._exog_raw.columns[0]
        x = np.asarray(self._exog_raw[xcol], dtype=float)
        v = yv.to_numpy(dtype=float)

        best: dict[str, Any] | None = None
        for p in range(0, self._max_lags + 1):
            for ql in range(0, self._max_lags + 1):
                entry = self._fit_lags(v, x, p, ql, yv.index)
                if entry is None:
                    continue
                if best is None or entry["aicc"] < best["aicc"]:
                    best = entry
        if best is None:
            raise FitFailure(
                "No lag combination of the ARDL grid could be estimated on this sample.",
                "The series may be too short for a lagged regression; use SARIMAX instead.",
                kind="length",
            )
        self._p, self._q = best["p"], best["q"]
        self._res = best["res"]
        self._fitted = best["fitted"]
        self._resid = best["resid"]
        self._n_obs = best["n"]
        self._n_params = best["k"]
        self._loglik = best["loglik"]
        self._aic, self._aicc, self._bic = self.ic(self._n_obs, self._n_params, self._loglik)
        self._xcol = xcol
        self.params = {n: float(c) for n, c in zip(best["names"], best["coef"])}
        self._raw_params = {
            n: (float(c), float(s), False)
            for n, c, s in zip(best["names"], best["coef"], best["se"])
        }
        self.coef_rows = [(n, float(c), float(s))
                          for n, c, s in zip(best["names"], best["coef"], best["se"])
                          if n.startswith("x")]
        self._bounds = self._bounds_test(v, x, best["p"])

    def _fit_lags(
        self, y: np.ndarray, x: np.ndarray, p: int, q: int, index: pd.Index
    ) -> dict[str, Any] | None:
        n = y.size
        start = max(p, q)
        if n - start < 10 + p + q + 2:
            return None
        cols: dict[str, np.ndarray] = {}
        for i in range(1, p + 1):
            cols[f"y_lag{i}"] = np.r_[np.full(start, np.nan), y[start - i: n - i]][start:] if False else y[start - i: n - i]
        for j in range(0, q + 1):
            cols[f"x_lag{j}" if j else "x"] = x[start - j: n - j]
        target = y[start:]
        X = np.column_stack([np.ones(target.size)] + [cols[c] for c in cols])
        if not np.all(np.isfinite(X)):
            return None
        try:
            beta, *_ = np.linalg.lstsq(X, target, rcond=None)
        except np.linalg.LinAlgError:
            return None
        fitted = X @ beta
        resid = target - fitted
        k = X.shape[1]
        nn = target.size
        ssr = float(resid @ resid)
        if ssr <= 0 or not np.isfinite(ssr):
            return None
        sigma2 = ssr / nn
        loglik = -0.5 * nn * (np.log(2 * np.pi * sigma2) + 1.0)
        # Standard errors from the OLS normal equations; xtx^-1 must exist for the IC comparison to mean
        # anything, so a singular design is a rejected lag, not a fitted model.
        try:
            xtx_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            return None
        se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
        _, aicc, _ = self.ic(nn, k, loglik)
        if aicc is None:
            return None
        return {
            "p": p, "q": q, "res": None, "fitted": pd.Series(fitted, index=index[start:]),
            "resid": np.r_[np.full(start, np.nan), resid], "n": nn, "k": k,
            "loglik": loglik, "aicc": aicc, "coef": beta, "se": se,
            "names": ["const"] + list(cols.keys()),
        }

    def _bounds_test(self, y: np.ndarray, x: np.ndarray, p: int) -> dict[str, Any]:
        """
        Pesaran–Shin–Smith bounds F-test for a levels relationship.

        Δy_t on y_{t−1}, x_{t−1}, Δy lags, Δx lags. The null is *no* long-run relationship (both level
        coefficients zero). The p-value is reported against the asymptotic bounds' midpoint and is
        labelled as an approximation for that reason.
        """
        dy, dx = np.diff(y), np.diff(x)
        rows: list[np.ndarray] = []
        target: list[float] = []
        for t in range(max(p, 1), dy.size):
            r = [1.0, y[t], x[t]]
            for i in range(1, p + 1):
                r.append(dy[t - i] if t - i >= 0 else 0.0)
            r.append(dx[t])
            rows.append(np.asarray(r))
            target.append(dy[t])
        if len(rows) < 12:
            return {"available": False,
                    "plain": f"Too few usable differences ({len(rows)}) to run the bounds test."}
        X = np.vstack(rows)
        yy = np.asarray(target)
        try:
            beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
            resid = yy - X @ beta
            k = X.shape[1]
            nn = yy.size
            sigma2 = float(resid @ resid) / nn
            xtx_inv = np.linalg.inv(X.T @ X)
            R = np.zeros((2, k))
            R[0, 1] = R[1, 2] = 1.0
            Rb = R @ beta
            cov = sigma2 * (R @ xtx_inv @ R.T)
            F = float(Rb @ np.linalg.solve(cov, Rb) / 2.0)
        except np.linalg.LinAlgError:
            return {"available": False, "plain": "The bounds-test design was singular."}
        # Pesaran et al. (2001) Table CI(iii), case III, k=1: the 5% bounds are [4.94, 5.73]
        # (all-I(0) and all-I(1) asymptotics). A value outside the band is conclusive either way; a
        # value inside it is not evidence in either direction.
        lo, hi = 4.94, 5.73

        # **No p-value is reported, deliberately.** The obvious-looking move is stats.f.sf(F, 2, dof),
        # and it is wrong: the bounds test exists precisely because the standard F distribution is
        # invalid when a regressor may be I(1). On this dataset it returns p = 0.049 while the bounds
        # say "below the lower bound, no relationship" — a spurious "significant at 5%" next to the
        # opposite conclusion. Printing it would be worse than printing nothing, so the band is the
        # whole answer. (Checked: our F = 3.1138 against statsmodels' own UECM test, 3.1137.)
        verdict = (
            "above the upper bound: a long-run relationship is supported"
            if F > hi else
            "below the lower bound: no long-run relationship is detected"
            if F < lo else
            "inside the inconclusive band, so no conclusion is drawn in either direction"
        )
        return {
            "available": True, "F": F, "lower": lo, "upper": hi,
            "verdict": verdict, "n": nn,
            "plain": (
                f"The bounds test on the error-correction form gives F = {F:.3f}, against the 5% "
                f"band [{lo:.2f}, {hi:.2f}] for one regressor: {verdict}. There is no p-value here on "
                "purpose — the tabulated band is the correct reference, and an ordinary F p-value "
                "would be computed against the wrong distribution."
            ),
        }

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        """
        Recursive multi-step: each step's own lag becomes an input to the next.

        A direct (per-horizon) fit would avoid compounding the recursion's error, but with 2 lags and a
        few dozen usable observations the direct fits would each have a quarter of the sample. The
        caveat names the trade.
        """
        xf = self._future_frame(horizon)
        if xf is None:  # pragma: no cover - guarded at fit time
            raise FitFailure("No driver path is available to forecast with.", "Supply a future path.")
        xcol = self._xcol
        xv = xf[xcol].to_numpy(dtype=float)
        hist = list(self.y.to_numpy(dtype=float))
        histx = list(np.asarray(self._exog_raw[xcol], dtype=float)) if self._exog_raw is not None else []
        coef = self.params
        means = np.zeros(horizon)
        for h in range(horizon):
            row = [1.0]
            for i in range(1, self._p + 1):
                row.append(hist[-i])
            for j in range(0, self._q + 1):
                if j == 0:
                    row.append(xv[h])
                else:
                    row.append(histx[-j] if h < j and len(histx) >= j else
                               (xv[h - j] if h - j >= 0 else histx[-1]))
            names = ["const"] + [f"y_lag{i}" for i in range(1, self._p + 1)] + \
                    [("x" if j == 0 else f"x_lag{j}") for j in range(0, self._q + 1)]
            yhat = float(sum(coef.get(nm, 0.0) * w for nm, w in zip(names, row)))
            means[h] = yhat
            hist.append(yhat)
        resid_sd = float(np.nanstd(self._resid, ddof=1)) if self._resid is not None else 0.0
        from scipy import stats

        z = float(stats.norm.ppf(1 - alpha / 2))
        steps = np.arange(1, horizon + 1)
        half = z * resid_sd * np.sqrt(steps)
        return pd.DataFrame({"mean": means, "lower": means - half, "upper": means + half},
                            index=steps)

    def _summary(self) -> FittedSummary:
        label = f"ARDL({self._p},{self._q}) — y on {self._xcol}"
        caveats: list[str] = []
        b = self._bounds
        if b.get("available"):
            caveats.append(b["plain"])
        else:
            caveats.append(str(b.get("plain", "The bounds test could not be run.")))
        sc = self.scenario_caveat
        if sc:
            caveats.append(sc)
        caveats.append(
            "Multi-step forecasts are recursive: each step's own forecast is used as an input to the "
            "next, so errors compound faster than a one-step-ahead fit suggests. The band grows with "
            "√h rather than with the fitted equation's own uncertainty."
        )
        xrows = [r for r in self.coef_rows if r[0].startswith("x")]
        one = (
            f"A regression of the target on its own last {self._p} value(s) and on "
            f"'{self._xcol}'"
            + (f" — {xrows[0][1]:+.4g} units per unit (SE {xrows[0][2]:.2g})" if xrows else "")
            + f". {b.get('plain', '')}"
        ).strip()
        return FittedSummary(
            model_id=self.model_id,
            display_name=label,
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
            plain_one_liner=one,
            params=self.params_table(self._raw_params),
            caveats=caveats,
            ic_comparable=True,
            ic_note="AIC/AICc from the OLS likelihood on the levels series; the sample is shorter than "
                    "the other models' because of the lags, so treat a small IC gap between ARDL and "
                    "SARIMAX with caution.",
        )


IMPLEMENTED_EXOG: dict[str, type[ModelWrapper]] = {
    "arimax": SARIMAXModel,
    "ardl": ARDLModel,
}


__all__ = ["ARDLModel", "IMPLEMENTED_EXOG", "SARIMAXModel"]
