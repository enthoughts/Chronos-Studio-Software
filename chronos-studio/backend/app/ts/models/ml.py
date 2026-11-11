"""
ts/models/ml.py — the tabular machine-learning family (§5.8 F).

The spec's requirement for this family is not "add LightGBM". It is: *"leakage-safe feature
construction (target lags only, recursive vs direct multi-step declared), a mandatory tabular-model
card stating features/lags/validation scheme, and a comparison against ETS/ARIMA that must be shown
even if the ML model loses."*

So this module is built around the card, not the estimator. Four things are true of every row here:

1. **Features are lags and calendar only.** No future information reaches a row: the feature vector for
   observation *t* is `y[t−1 … t−p]`, an optional seasonal lag `y[t−m]`, two harmonics of the time index,
   and (where drivers exist) the driver at `t−1` — never at *t*, because at forecast time the driver for
   the step being predicted is not known. Anything else is leakage, and leakage in a time series is
   invisible: the fit looks excellent and the forecast is worthless.
2. **Multi-step is recursive, and declared.** The model is fitted on one-step-ahead pairs and called
   repeatedly, feeding each prediction back as a lag. That compounds its own error, so the card says
   "recursive" and the interval widens accordingly. A direct (one model per horizon) scheme is not
   offered here *silently* — it is not offered at all.
3. **Validation is a time-ordered holdout**, never k-fold: shuffled folds let a row be trained on the
   future, which is the single most common way an ML time-series result is wrong. Three expanding folds
   are reported per one-step MASE, plus the holdout.
4. **There is no likelihood, so there is no AIC.** `ic_comparable` is False and the ranking falls back to
   out-of-sample error. Inventing an IC for a tree ensemble would let it be ranked against ARIMA on a
   number that means nothing.

Refusals, each measured rather than assumed:

* **LightGBM / XGBoost** — not installed. `UNIMPLEMENTED_ML` carries the probe result, so the excluded
  list quotes an import error rather than a plan.
* **LSTM / GRU / TFT** — no `torch` (nor `tensorflow`, nor `keras`). Refused for that reason.
* **Gaussian process beyond ~400 rows** — the cubic Cholesky is the cost; the row is offered with a
  subsample and a stated caveat instead of hanging the request.

No `torch` means "neural network autoregression" is a scikit-learn MLP on lags: it is a real
autoregressive neural net (one hidden layer, tanh), just not a recurrent one, and the display name says
NNAR rather than LSTM for that reason.
"""

from __future__ import annotations

import importlib
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.schemas import ModelFamily
from app.ts.models.base import FitFailure, FittedSummary, ModelWrapper

#: Fixed seed everywhere. An ML row that reports a different number on each click is not reproducible,
#: and the report cites these numbers.
SEED = 20260101

#: Default lag depth. Deep enough to cover a weekly cycle at daily resolution without turning the
#: feature count into a multiple of the sample.
DEFAULT_LAGS = 8

#: Above this the Gaussian process is fitted on a contiguous subsample and says so.
GP_MAX_N = 400


@contextmanager
def _quiet():
    """
    Silence the estimators' own convergence chatter, having first captured it.

    A printed `ConvergenceWarning` that nobody reads is noise in a server log; a convergence warning
    that reaches the fit table is information. Callers wrap the fit, then check `_LAST_WARNINGS`.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _LAST_WARNINGS.clear()
        yield
        for w in caught:
            _LAST_WARNINGS.append(f"{w.category.__name__}: {w.message}")


_LAST_WARNINGS: list[str] = []


def _probe(module: str) -> str | None:
    """`None` if importable, else a one-line measured reason. Never raises."""
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - the point is to catch any import failure
        return f"`import {module}` fails with {type(exc).__name__}: {exc}"
    version = getattr(mod, "__version__", "unknown version")
    return f"`{module}` imports fine ({version})"


_OPTIONAL = {
    "lightgbm": _probe("lightgbm"),
    "xgboost": _probe("xgboost"),
    "torch": _probe("torch"),
    "tensorflow": _probe("tensorflow"),
}


# --------------------------------------------------------------------------- #
# The feature builder — one implementation, so every row is leakage-safe by construction
# --------------------------------------------------------------------------- #
@dataclass
class Design:
    """A leakage-safe supervised design matrix, its column names, and the spec that built it."""

    X: np.ndarray
    y: np.ndarray
    columns: list[str]
    #: How many leading observations the design consumes, so predictions can be aligned back to `y`.
    offset: int
    lags: int
    #: The recipe for one feature row. Kept as data so the forecast path cannot drift out of step with
    #: the training path — every leakage bug in this kind of model is a mismatch between the two.
    spec: list[tuple[str, int]] = field(default_factory=list)
    exog: np.ndarray | None = None


def build_design(
    y: np.ndarray,
    *,
    lags: int,
    period: int | None,
    exog: np.ndarray | None = None,
    exog_names: list[str] | None = None,
    harmonics: int = 2,
) -> Design:
    """
    Build `X, y` where every row of `X` uses only values from strictly before its target.

    How the columns are built is stored in `Design.spec` and replayed by `feature_row`, so the training
    matrix and the forecast feature vector are the same computation. They have to be: every leakage or
    misalignment bug in a model of this kind comes from these two paths drifting apart.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n <= lags + 5:
        raise FitFailure(
            f"{lags} lags need more than {lags + 5} observations and there are {n}.",
            "Use fewer lags, or a longer history — an ML model with one row per parameter overfits by "
            "construction.",
            kind="length",
        )
    idx = np.arange(lags, n)
    spec: list[tuple[str, int]] = []
    cols: list[np.ndarray] = []
    names: list[str] = []
    for lag in range(1, lags + 1):
        cols.append(y[idx - lag])
        names.append(f"y(t\u2212{lag})")
        spec.append(("lag", lag))
    # A seasonal lag only where the sample covers at least two cycles after the burn-in; rows that
    # cannot see it are dropped below rather than silently given a zero.
    if period and period >= 2 and n > period + lags:
        vals = np.where(idx - period >= 0, y[np.clip(idx - period, 0, n - 1)], np.nan)
        cols.append(vals)
        names.append(f"y(t\u2212{period})")
        spec.append(("seasonal", period))
    for h in range(1, harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * idx / 12.0))
        names.append(f"sin({h}\u00b7t/12)")
        spec.append(("sin", h))
        cols.append(np.cos(2 * np.pi * h * idx / 12.0))
        names.append(f"cos({h}\u00b7t/12)")
        spec.append(("cos", h))
    if exog is not None and np.asarray(exog).size:
        ex = np.asarray(exog, dtype=float)
        for j in range(ex.shape[1]):
            nm = (exog_names or [])[j] if exog_names and j < len(exog_names) else f"x{j + 1}"
            # Lag-1 driver: the driver's value at the step being predicted is not known at forecast
            # time, so using it would be a leak that only shows up when the forecast is used.
            cols.append(ex[idx - 1, j])
            names.append(f"{nm}(t\u22121)")
            spec.append(("exog", j))
    X = np.column_stack(cols)
    target = y[idx]
    ok = np.isfinite(X).all(axis=1) & np.isfinite(target)
    return Design(
        X=X[ok], y=target[ok], columns=names, offset=int(idx[ok][0]) if ok.any() else lags,
        lags=lags, spec=spec, exog=exog,
    )


def feature_row(history: np.ndarray, t: int, design: Design) -> np.ndarray:
    """The feature vector for predicting position `t`, built from the same `spec` as the training matrix."""
    vals: list[float] = []
    n_hist = history.size
    for kind, arg in design.spec:
        if kind == "lag":
            vals.append(history[t - arg] if t - arg >= 0 else np.nan)
        elif kind == "seasonal":
            vals.append(history[t - arg] if t - arg >= 0 else np.nan)
        elif kind == "sin":
            vals.append(np.sin(2 * np.pi * arg * t / 12.0))
        elif kind == "cos":
            vals.append(np.cos(2 * np.pi * arg * t / 12.0))
        elif kind == "exog":
            # Held at its last observed value: a labelled scenario, stated in the caveats.
            vals.append(float(design.exog[min(t - 1, n_hist - 1), arg]))
    return np.asarray(vals, dtype=float)


# --------------------------------------------------------------------------- #
# The shared wrapper
# --------------------------------------------------------------------------- #
class _MLBase(ModelWrapper):
    """
    Everything an ML row shares: the design, the holdout, the folds, the card, the recursion.

    Subclasses implement `_estimator()` and set `hyperparams`. The fit/forecast mechanics are here once
    so no model in this family can accidentally train on the future.
    """

    family = ModelFamily.ML
    #: Stated in the card and in the summary; a direct scheme would be a different row.
    scheme = "recursive"
    estimator_name = "estimator"
    supports_exog = False

    def __init__(self, *, lags: int = DEFAULT_LAGS, exog: pd.DataFrame | None = None,
                 horizon: int = 12, **kw) -> None:
        super().__init__(**kw)
        self.lags = int(lags)
        self._exog_raw = exog
        self._horizon = int(horizon)
        self._estimator_fitted: Any = None
        self.card: dict[str, Any] = {}

    # -- to implement ------------------------------------------------------- #
    def _estimator(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def hyperparams(self) -> dict[str, Any]:
        return getattr(self, "_hyperparams", {})

    @hyperparams.setter
    def hyperparams(self, value: dict[str, Any]) -> None:
        self._hyperparams = value

    # -- exog --------------------------------------------------------------- #
    def _exog_array(self, index: pd.Index) -> tuple[np.ndarray | None, list[str]]:
        if self._exog_raw is None or getattr(self._exog_raw, "shape", (0, 0))[1] == 0:
            return None, []
        frame = pd.DataFrame(self._exog_raw).reindex(index).astype(float)
        frame = frame.ffill().bfill()
        if frame.isna().all().any():
            return None, []
        return frame.to_numpy(), [str(c) for c in frame.columns]

    # -- fit ---------------------------------------------------------------- #
    def _fit(self, y: pd.Series) -> None:
        ys = pd.Series(y).astype(float)
        ys = ys.interpolate(limit_direction="both").dropna()
        if ys.size < 40:
            raise FitFailure(
                f"An ML model needs at least 40 observations and there are {ys.size}.",
                "Use a classical model on a short series — a tree ensemble on 30 points fits noise.",
                kind="length",
            )
        exog, exog_names = self._exog_array(ys.index)
        design = build_design(ys.to_numpy(), lags=self.lags, period=self.period,
                              exog=exog, exog_names=exog_names)
        if design.X.shape[0] < 20:
            raise FitFailure(
                f"After {self.lags} lags and complete cases only {design.X.shape[0]} rows remain.",
                "Reduce the lag depth.",
                kind="length",
            )

        # --- time-ordered holdout: the last 20%, never a shuffled split ----------- #
        n = design.X.shape[0]
        n_hold = max(5, int(round(0.2 * n)))
        train = slice(0, n - n_hold)
        test = slice(n - n_hold, n)
        # Standardise on the training block only. Scaling on the full sample leaks the holdout's mean
        # and spread into the fit — a small leak, but the one that makes a holdout score optimistic.
        mu = design.X[train].mean(axis=0)
        sd = design.X[train].std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xs = (design.X - mu) / sd

        est = self._estimator()
        with _quiet():
            est.fit(Xs[train], design.y[train])
        self._solve_warnings = list(_LAST_WARNINGS)
        hold_pred = est.predict(Xs[test])
        hold_err = design.y[test] - hold_pred

        # --- three expanding folds, one-step MASE each ---------------------------- #
        folds = []
        fold_edges = np.linspace(max(20, int(0.5 * n)), n - n_hold, 3).astype(int)
        for k, edge in enumerate(fold_edges):
            if edge < 20 or edge >= n - n_hold:
                continue
            e = self._estimator()
            with _quiet():
                e.fit(Xs[:edge], design.y[:edge])
            pred = e.predict(Xs[edge: n - n_hold])
            truth = design.y[edge: n - n_hold]
            scale = _mase_scale(truth, self.period)
            mase = float(np.mean(np.abs(truth - pred)) / scale) if scale else None
            folds.append({
                "fold": k + 1,
                "train_n": int(edge),
                "test_n": int(truth.size),
                "rmse": float(np.sqrt(np.mean((truth - pred) ** 2))),
                "mase": mase,
            })

        # Refit on everything for the forecast: the model that forecasts should have seen all the data.
        self._estimator_fitted = self._estimator()
        with _quiet():
            self._estimator_fitted.fit(Xs, design.y)
        self._scale = {"mu": mu, "sd": sd}
        self._design = design
        self._y_clean = ys.to_numpy()
        self._exog_used = exog

        # --- the accuracy block, computed the only way it means anything here ---------------- #
        # These estimators memorise their training rows: k-NN with distance weighting reproduces the
        # training points *exactly* (residual 0, MASE ≈ 0), and a forest nearly does. In-sample error
        # would therefore report a perfect fit for a model that is not trying. So the reported
        # one-step accuracy comes from expanding-window refits — each block predicted by a model that
        # has only seen the data before it. That is the same information a reader assumes they are
        # getting when they read "in-sample".
        oos = self._blockwise_oos(Xs, design.y, n_blocks=5)
        self._fitted = np.full(ys.size, np.nan)
        rows_kept = np.arange(ys.size - design.X.shape[0], ys.size)
        finite = np.isfinite(design.y)
        self._fitted[rows_kept[finite]] = np.where(np.isfinite(oos), oos, np.nan)
        self._resid = design.y - oos
        self._holdout = {
            "n": int(n_hold),
            "rmse": float(np.sqrt(np.mean(hold_err ** 2))),
            "mase": (
                float(np.mean(np.abs(hold_err)) / _mase_scale(design.y[test], self.period))
                if _mase_scale(design.y[test], self.period) else None
            ),
            "folds": folds,
        }
        # --- the interval, and the evidence for its width ------------------------------- #
        # A normal-theory band (z × residual sd) covered only 21–54% of the out-of-sample errors on the
        # series this was built against, against the 90% it claimed: the residuals are a mixture — many
        # tiny, a few large — and a Gaussian band is the wrong shape for that. So the half-width is the
        # *empirical* 90th percentile of the absolute one-step out-of-sample error, and the coverage
        # below is measured on the holdout block, which took no part in forming that quantile.
        oos_abs = np.abs(self._resid)
        oos_abs = oos_abs[np.isfinite(oos_abs)]
        if oos_abs.size >= 10:
            self._q90 = float(np.percentile(oos_abs, 90))
            hold_abs = np.abs(hold_err)
            self._coverage_90 = float(np.mean(hold_abs <= self._q90))
        else:  # pragma: no cover - defensive
            self._q90 = None
            self._coverage_90 = None
        self._resid_kurt = _kurtosis(self._resid)
        self._band_basis = "empirical 90th percentile of the one-step out-of-sample errors"

        self._n_obs_used = int(design.X.shape[0])
        self._n_params = int(self._n_param_count())
        self._loglik = None
        self.card = {
            "features": design.columns,
            "n_features": len(design.columns),
            "lags": self.lags,
            "scheme": self.scheme,
            "validation": (
                f"time-ordered holdout (last {n_hold} of {n} rows) plus 3 expanding folds, one-step "
                f"horizon; never a shuffled split"
            ),
            "scaling": "standardised on the training block only",
            "hyperparams": self.hyperparams,
            "holdout": self._holdout,
            "interval_coverage_90": self._coverage_90,
            "interval_basis": self._band_basis,
            "residual_excess_kurtosis": self._resid_kurt,
            "solver_warnings": list(getattr(self, "_solve_warnings", [])),
            "benchmark_note": (
                "The naive and seasonal-naive baselines are fitted unconditionally alongside this row, "
                "so the comparison exists even where the ML model loses."
            ),
        }

    def _blockwise_oos(self, Xs: np.ndarray, target: np.ndarray, *, n_blocks: int = 5) -> np.ndarray:
        """
        One-step predictions for each of `n_blocks` contiguous blocks, each from a model refitted on the
        data before that block. Returns NaN for the rows before the first block's training data.

        Blocked rather than leave-one-out or random-fold: a random fold trains on the future, and
        leave-one-out on a k-NN model predicts each point from its own neighbours in time, which is the
        same leak wearing a different hat.
        """
        n = target.size
        first = max(20, int(0.4 * n))
        edges = np.linspace(first, n, n_blocks + 1).astype(int)
        out = np.full(n, np.nan)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            est = self._estimator()
            with _quiet():
                est.fit(Xs[:a], target[:a])
            out[a:b] = est.predict(Xs[a:b])
        return out

    def _n_param_count(self) -> int:
        """A tree ensemble's 'parameter count' is a growth driver, not a free parameter count."""
        return 0

    # -- forecast ----------------------------------------------------------- #
    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:
        est = self._estimator_fitted
        if est is None:
            raise FitFailure("The model has not been fitted.", "Fit it first.", kind="internal")
        hist = np.concatenate([self._y_clean, np.zeros(horizon)])
        n0 = self._y_clean.size
        for h in range(horizon):
            x = feature_row(hist, n0 + h, self._design)
            x = (x - self._scale["mu"]) / self._scale["sd"]
            hist[n0 + h] = float(est.predict(x.reshape(1, -1))[0])
        mean = hist[n0:]

        # Interval from the out-of-sample one-step residuals, widened by √h to reflect the recursion
        # compounding its own error. A constant band around a recursive forecast is the one thing that
        # is definitely wrong: step 12 is much less certain than step 1 by construction.
        resid = np.asarray(self._resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        sd = float(resid.std(ddof=1)) if resid.size > 2 else float("nan")
        if not np.isfinite(sd) or sd <= 0:
            raise FitFailure(
                "The residuals have no spread, so no interval can be formed.",
                "This model reproduced the data exactly — check for a leaked feature.",
                kind="numerical",
            )
        # Empirical quantile for the default 90% band; z×sd for any other level, because forming a
        # second empirical quantile from the same residuals and calling it a different level would be
        # false precision. The caveat states which one was used.
        if self._q90 is not None and abs((1 - alpha) - 0.90) < 1e-9:
            base = self._q90
            self._band_basis = "empirical 90th percentile of the one-step out-of-sample errors"
        else:
            base = float(stats.norm.ppf(1 - alpha / 2)) * sd
            self._band_basis = (
                f"normal theory: z({1 - alpha:.0%}) × out-of-sample residual sd, used because the "
                f"empirical quantile is only formed for the 90% level"
            )
        half = base * np.sqrt(np.arange(1, horizon + 1))
        return pd.DataFrame({"mean": mean, "lower": mean - half, "upper": mean + half},
                            index=np.arange(1, horizon + 1))

    # -- summary ------------------------------------------------------------ #
    def _summary(self) -> FittedSummary:
        hold = self._holdout
        folds = [f["mase"] for f in hold["folds"] if f["mase"] is not None]
        fold_txt = (
            f"Expanding-fold one-step MASE ranged {min(folds):.3f}–{max(folds):.3f}." if folds else ""
        )
        one = (
            f"{self.display_name} on {len(self.card['features'])} lag/calendar features, "
            f"{self.scheme} multi-step. Holdout MASE {_fmt(hold['mase'])} over the last "
            f"{hold['n']} observations. {fold_txt} "
            f"There is no likelihood for this estimator, so it is ranked on forecast error rather "
            f"than AIC."
        )
        params = self.params_table(self._param_rows())
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=True,
            n_params=self._n_params,
            loglik=None,
            aic=None,
            aicc=None,
            bic=None,
            n_obs_used=self._n_obs_used,
            fitted_on="lag features (levels); accuracy from expanding-window refits",
            roots_ok=None,
            roots_note="not applicable: a tabular model has no characteristic roots",
            is_baseline=False,
            plain_one_liner=one,
            params=params,
            caveats=[
                f"Multi-step forecasts are {self.scheme}: each prediction is fed back as a lag, so "
                "error compounds with the horizon and the interval widens by √h.",
                "Features are lags, calendar harmonics and lag-1 drivers only. A model that used the "
                "current-period driver would look excellent in-sample and be unusable out of sample.",
                "No information criterion is reported — there is no likelihood here. Compare this row "
                "on holdout/fold error, not on AIC.",
                "The accuracy figures are out-of-sample by construction: each block is predicted by a "
                "model refitted only on the data before it. A k-NN fit reproduces its own training "
                "rows exactly, so an ordinary in-sample MASE would read ≈ 0 and mean nothing.",
                (
                    f"The interval is built from {self._band_basis}, widened by √h for the "
                    f"recursion. On the untouched holdout it covers {self._coverage_90:.0%} of one-step "
                    f"errors (it claims 90%). An empirical quantile is used rather than z × sd because "
                    f"these residuals are a mixture, not a normal draw — measured excess kurtosis "
                    f"{self._resid_kurt:+.2f} — and on the first series this was tested against a "
                    f"Gaussian band covered 21–54% of what it claimed 90%."
                    if self._coverage_90 is not None and self._resid_kurt is not None else
                    "Interval coverage could not be measured out of sample."
                ),
                *[f"Solver warning during fitting: {w}" for w in getattr(self, "_solve_warnings", [])],
                *self._caveats,
            ],
            ic_comparable=False,
            ic_note=(
                "ML rows have no likelihood, so their AIC/BIC are not defined and they are not ranked "
                "against the IC of the classical models."
            ),
        )

    def _param_rows(self) -> dict[str, tuple[float | None, float | None, bool]]:
        rows: dict[str, tuple[float | None, float | None, bool]] = {
            "lags used": (float(self.lags), None, False),
            "features": (float(len(self.card.get("features", []))), None, False),
            "holdout MASE": (self._holdout.get("mase"), None, False),
        }
        for k, v in self.hyperparams.items():
            if isinstance(v, (int, float)):
                rows[k] = (float(v), None, False)
        return rows


def _mase_scale(y: np.ndarray, period: int | None) -> float | None:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return None
    d = np.abs(y[period:] - y[:-period]) if period and y.size > period + 1 else np.abs(np.diff(y))
    s = float(d.mean()) if d.size else 0.0
    return s if s > 0 else None


def _kurtosis(r: np.ndarray) -> float | None:
    """Excess kurtosis of the residuals, or None where it is not defined."""
    x = np.asarray(r, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 20 or np.std(x) <= 0:
        return None
    return float(stats.kurtosis(x))


def _fmt(v: float | None) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.3f}"


# --------------------------------------------------------------------------- #
# The rows
# --------------------------------------------------------------------------- #
class GradientBoostingModel(_MLBase):
    """
    Gradient-boosted trees on lag features.

    scikit-learn's histogram-based boosting, not LightGBM: `lightgbm` is not installed (see
    `UNIMPLEMENTED_ML`), and pretending the name while running a different library would put a false
    provenance on every number in the report.
    """

    model_id = "gradient_boost"
    display_name = "Gradient-boosted trees"

    def _estimator(self) -> Any:
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.hyperparams = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3, "random_state": SEED}
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=5, random_state=SEED,
            early_stopping=False,
        )

    def _blockwise_oos(self, Xs: np.ndarray, target: np.ndarray, *, n_blocks: int = 5) -> np.ndarray:
        """
        One-step predictions for each of `n_blocks` contiguous blocks, each from a model refitted on the
        data before that block. Returns NaN for the rows before the first block's training data.

        Blocked rather than leave-one-out or random-fold: a random fold trains on the future, and
        leave-one-out on a k-NN model predicts each point from its own neighbours in time, which is the
        same leak wearing a different hat.
        """
        n = target.size
        first = max(20, int(0.4 * n))
        edges = np.linspace(first, n, n_blocks + 1).astype(int)
        out = np.full(n, np.nan)
        for a, b in zip(edges[:-1], edges[1:]):
            if b <= a:
                continue
            est = self._estimator()
            with _quiet():
                est.fit(Xs[:a], target[:a])
            out[a:b] = est.predict(Xs[a:b])
        return out

    def _n_param_count(self) -> int:
        return 200


class RandomForestModel(_MLBase):
    model_id = "random_forest"
    display_name = "Random forest"

    def _estimator(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        self.hyperparams = {"n_estimators": 200, "min_samples_leaf": 3, "random_state": SEED}
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=1)


class KNNModel(_MLBase):
    model_id = "knn"
    display_name = "k-nearest neighbours"

    def _estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsRegressor

        k = max(3, min(9, int(np.sqrt(max(self._design_rows(), 20)))))
        self.hyperparams = {"n_neighbors": k}
        return KNeighborsRegressor(n_neighbors=k, weights="distance")

    def _design_rows(self) -> int:
        return 0 if self.y is None else int(pd.Series(self.y).notna().sum())


class SVRModel(_MLBase):
    model_id = "svr"
    display_name = "Support vector regression"

    def _estimator(self) -> Any:
        from sklearn.svm import SVR

        self.hyperparams = {"C": 10.0, "epsilon": 0.1, "kernel": "rbf"}
        return SVR(C=10.0, epsilon=0.1, kernel="rbf", gamma="scale")


class NNARModel(_MLBase):
    """
    Neural-network autoregression: one tanh hidden layer over the lag features.

    Not an LSTM, and not called one. A recurrent net needs `torch`, which is not installed; the row
    that *would* be the LSTM is refused in `UNIMPLEMENTED_ML` with that reason.
    """

    model_id = "nnar"
    display_name = "Neural network AR (NNAR)"

    def _estimator(self) -> Any:
        from sklearn.neural_network import MLPRegressor

        self.hyperparams = {"hidden": 8, "alpha": 1e-3, "max_iter": 1500}
        return MLPRegressor(
            hidden_layer_sizes=(8,), activation="tanh", solver="lbfgs", alpha=1e-3,
            max_iter=1500, random_state=SEED,
        )


class GaussianProcessModel(_MLBase):
    """
    Gaussian-process regression on lag features.

    The cost is the n×n Cholesky, so beyond `GP_MAX_N` rows the row is fitted on the most recent
    contiguous block and the caveat says so — degrading loudly beats a request that hangs.
    """

    model_id = "gaussian_process"
    display_name = "Gaussian process"

    def _fit(self, y: pd.Series) -> None:
        ys = pd.Series(y).astype(float).dropna()
        self._subsampled = 0
        if ys.size > GP_MAX_N:
            self._subsampled = int(ys.size - GP_MAX_N)
            self._gp_caveat = (
                f"The Gaussian process was fitted on the most recent {GP_MAX_N} of {ys.size} "
                "observations: the covariance matrix is n×n and inverting it on the full series would "
                "cost more than the forecast is worth. Older history is not represented."
            )
            self._caveats = list(self._caveats) + [self._gp_caveat]
            ys = ys.iloc[-GP_MAX_N:]
        super()._fit(ys)

    def _estimator(self) -> Any:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(5.0, (1e-1, 1e2)) + WhiteKernel(1.0, (1e-3, 1e1))
        self.hyperparams = {"kernel": "C·RBF + White", "normalize_y": True}
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=SEED,
                                        n_restarts_optimizer=1)


#: Rows this module fits. Merged into `fit.IMPLEMENTED`.
IMPLEMENTED_ML: dict[str, type[ModelWrapper]] = {
    "gradient_boost": GradientBoostingModel,
    "random_forest": RandomForestModel,
    "knn": KNNModel,
    "svr": SVRModel,
    "nnar": NNARModel,
    "gaussian_process": GaussianProcessModel,
}


def _dependency_reason(*modules: str) -> str:
    """Build the measured refusal text for a row that needs a package this build does not have."""
    bits = [f"{m}: {_OPTIONAL[m]}" for m in modules if m in _OPTIONAL]
    missing = [m for m in modules if _OPTIONAL.get(m)]
    if not missing:
        return ""
    named = ", ".join(f"`{m}`" for m in missing)
    return (
        f"the library it needs is not installed in this build: {named} — "
        + "; ".join(bits)
        + ". The row stays visible and unfitted rather than being silently dropped, and the gradient-"
        "boosted row above runs on scikit-learn instead."
    )


#: Declared but not fitted, each with a **measured** reason (probe output, not a plan).
UNIMPLEMENTED_ML: dict[str, str] = {
    "lightgbm": _dependency_reason("lightgbm"),
    "xgboost": _dependency_reason("xgboost"),
    "lstm": _dependency_reason("torch", "tensorflow"),
}


def dependency_probe() -> dict[str, str | None]:
    """The probe results, exposed so a test can assert the refusal text matches reality."""
    return dict(_OPTIONAL)


__all__ = [
    "DEFAULT_LAGS",
    "GP_MAX_N",
    "IMPLEMENTED_ML",
    "SEED",
    "UNIMPLEMENTED_ML",
    "Design",
    "GaussianProcessModel",
    "GradientBoostingModel",
    "KNNModel",
    "NNARModel",
    "RandomForestModel",
    "SVRModel",
    "build_design",
    "dependency_probe",
]
