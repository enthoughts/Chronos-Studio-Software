"""
ts/models/base.py — the `ModelWrapper` contract (§5.8).

Every model in the app, from `mean` to `SARIMAX` to `GARCH`, is fitted through this one interface.
The point is not tidiness: the evaluation stage (§5.12), the selection stage (§5.9) and the report
(§5.14) must treat models they have never heard of identically, and that is only true if the
interface is narrow and every member is total — including the failures.

Three rules this file exists to enforce:

1. **A fit that fails is still a fit.** `fit()` never raises for a modelling reason. It returns a
   `ModelWrapper` whose `summary.failed` is True and whose `failure_reason` says why, because the
   caller must be able to fall back to the next candidate *and* show the user what happened. A
   missing candidate that nobody can see is worse than a visible failure.
2. **A number here is traceable.** `params_table()` reports standard errors and a boundary flag, so
   "AR(1) coefficient 0.98" can be read as "0.98, SE 0.01, at the boundary" rather than as a fact.
3. **Comparability is declared, not assumed.** `ic_comparability` states whether this fit's AIC can
   be compared with others; an ARFIMA AIC is not comparable with an ARIMA AIC and the report must
   know that before it ranks anything.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from app.core.schemas import ModelFamily, ParamRow


class FitFailure(RuntimeError):
    """Raised internally when a model cannot be fitted. Caught by `ModelWrapper.fit`."""

    def __init__(self, plain: str, fix: str = "", *, kind: str = "modelling") -> None:
        super().__init__(plain)
        self.plain = plain
        self.fix = fix
        self.kind = kind


@dataclass
class FitDiagnostics:
    """
    The diagnostics every model can produce, so the comparison table has no holes.

    Fields that do not apply to a given model are `None` rather than omitted — a blank cell that
    means "not applicable" and a blank cell that means "not computed" must not look the same.
    """

    #: Ljung–Box on the residuals: high p = no remaining autocorrelation for the model to explain.
    ljung_box_p: float | None = None
    ljung_box_lag: int | None = None
    #: Jarque–Bera p: low p = residuals are not normal, so symmetric intervals understate tails.
    jarque_bera_p: float | None = None
    #: ARCH-LM p on residuals: low p = variance left on the table.
    arch_lm_p: float | None = None
    residual_sd: float | None = None
    residual_mean: float | None = None
    #: Fraction of residual autocorrelation still present at lag 1 — a blunt but readable number.
    residual_acf1: float | None = None
    #: In-sample accuracy, always against the same series the fit saw.
    rmse: float | None = None
    mae: float | None = None
    mape: float | None = None            # banned for count data; the API says so
    mase: float | None = None            # against the naive/seasonal-naive benchmark
    #: Free-text warnings the fit itself raised (parameter at a bound, near-unit-root, ...).
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FittedSummary:
    """The serializable half of a fit. Mirrors `schemas.FitSummary` minus the pydantic wrapper."""

    model_id: str
    display_name: str
    family: ModelFamily
    fit_id: str
    converged: bool = True
    n_params: int = 0
    loglik: float | None = None
    aic: float | None = None
    aicc: float | None = None
    bic: float | None = None
    n_obs_used: int = 0
    fitted_on: str = "levels"
    roots_ok: bool | None = None
    roots_note: str = ""
    is_baseline: bool = False
    failed: bool = False
    failure_reason: str = ""
    plain_one_liner: str = ""
    params: list[ParamRow] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    converged_warnings: list[str] = field(default_factory=list)
    #: §5.9's guard: ICs from different likelihood scales are not comparable.
    ic_comparable: bool = True
    ic_note: str = ""


class ModelWrapper:
    """
    The uniform handle on one fitted model.

    Subclasses implement `_fit`, `_predict` and `_forecast`. Everything else — failure handling,
    timing, the naive-benchmark comparisons, the serialization — is here once.
    """

    #: Filled in by the subclass before `fit()` is called.
    model_id: str = "unknown"
    display_name: str = "Unknown model"
    family: ModelFamily = ModelFamily.BASELINE
    is_baseline: bool = False
    #: What the model was fitted to, for the report ("levels", "diff(1)", "log", ...).
    fitted_on: str = "levels"

    def __init__(
        self,
        *,
        series_id: str = "",
        caveats: list[str] | None = None,
        period: int | None = None,
        period_days: float | None = None,
    ) -> None:
        self.series_id = series_id
        #: How many days one recording period covers (7 for a weekly calendar, 30.44 for monthly).
        #: A point process measures intervals in days and has to be told how long "one period ahead"
        #: is; inferring it from the event spacing gives the gap between events, not the calendar.
        self.period_days = float(period_days) if period_days else None
        #: Seasonal period, when the wrapper cares about it. Carried here rather than passed
        #: separately so every constructor in the registry has the same signature.
        self.period = int(period) if period and period >= 2 else None
        self._caveats = list(caveats or [])
        self.y: pd.Series | None = None
        self.params: dict[str, Any] = {}
        self.result: Any = None
        self.diagnostics = FitDiagnostics()
        self.summary: FittedSummary | None = None
        self.fit_seconds: float = 0.0

    # -- the contract ------------------------------------------------------- #
    def fit(self, y: pd.Series) -> "ModelWrapper":
        """
        Fit, or record why not.

        Never raises for a modelling failure: callers get a wrapper either way, and the failure path
        is the same code path as success so it cannot rot from disuse.
        """
        self.y = pd.Series(y).astype(float)
        t0 = time.perf_counter()
        try:
            self._fit(self.y)
            self.summary = self._summary()
        except FitFailure as exc:
            self.summary = self._failure_summary(f"{exc.plain}" + (f" {exc.fix}" if exc.fix else ""))
        except Exception as exc:  # noqa: BLE001 - a library crash is a fit failure, not a 500
            self.summary = self._failure_summary(
                f"{type(exc).__name__}: {exc}. This is a failure inside the fitting library, "
                "reported rather than hidden."
            )
        self.fit_seconds = time.perf_counter() - t0
        return self

    @property
    def failed(self) -> bool:
        return bool(self.summary and self.summary.failed)

    @property
    def ok(self) -> bool:
        return bool(self.summary and not self.summary.failed)

    def predict(self, horizon: int = 1, *, alpha: float = 0.01) -> pd.DataFrame:
        """
        H-step-ahead forecast as a DataFrame indexed by step, with columns
        `mean`, `lower`, `upper` (the last two at `1 - alpha`).

        Refuses on a failed fit rather than returning NaNs — a silent NaN forecast is the one output
        that can reach a report without anybody noticing.
        """
        if self.failed or self.summary is None:
            raise FitFailure(
                f"{self.display_name} was not fitted successfully, so it cannot forecast.",
                "Fix the fit failure shown in the fitting table, or choose another model.",
            )
        out = self._forecast(int(horizon), alpha=alpha)
        if not isinstance(out, pd.DataFrame):
            out = pd.DataFrame(out)
        for col in ("mean", "lower", "upper"):
            if col not in out.columns:
                raise FitFailure(
                    f"{self.display_name} produced no '{col}' column.",
                    "This is a bug in the model wrapper, not in the data.",
                )
        return out

    def fitted_values(self) -> np.ndarray | None:
        """In-sample one-step-ahead predictions aligned with `y`, where the model defines them."""
        return getattr(self, "_fitted", None)

    # -- to be implemented by subclasses ------------------------------------ #
    def _fit(self, y: pd.Series) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _forecast(self, horizon: int, *, alpha: float) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    # -- helpers ------------------------------------------------------------ #
    def _summary(self) -> FittedSummary:
        raise NotImplementedError

    def _failure_summary(self, reason: str) -> FittedSummary:
        return FittedSummary(
            model_id=self.model_id,
            display_name=self.display_name,
            family=self.family,
            fit_id=f"fit-{self.model_id}",
            converged=False,
            failed=True,
            failure_reason=reason,
            is_baseline=self.is_baseline,
            fitted_on=self.fitted_on,
            caveats=list(self._caveats),
            ic_comparable=False,
            ic_note="A failed fit has no information criterion.",
        )

    #: Residuals, for diagnostics. Subclasses set `self._resid`.
    _resid: np.ndarray | None = None

    def residuals(self) -> np.ndarray | None:
        r = getattr(self, "_resid", None)
        return None if r is None else np.asarray(r, dtype=float)

    def compute_diagnostics(self, y: pd.Series, *, ban_mape: bool = False) -> FitDiagnostics:
        """
        Fill the standard diagnostic block from the residuals and the fitted values.

        Kept on the base class so every model is judged by the same code. If a model needs a
        different test, that is a *new field*, not a locally reimplemented Ljung–Box.
        """
        from app.ts.models.diagnostics import diagnose

        self.diagnostics = diagnose(
            y,
            fitted=self.fitted_values(),
            resid=self.residuals(),
            period=getattr(self, "period", None),
            ban_mape=ban_mape,
        )
        return self.diagnostics

    def ic(self, n: int, k: int, loglik: float | None) -> tuple[float | None, float | None, float | None]:
        """AIC, AICc and BIC from a log-likelihood, with the small-sample correction applied only
        where it is defined (k + 1 < n)."""
        if loglik is None or not np.isfinite(loglik):
            return None, None, None
        aic = -2.0 * float(loglik) + 2.0 * k
        denom = n - k - 1
        aicc = (-2.0 * float(loglik) + 2.0 * k * n / denom) if denom > 0 else None
        bic = -2.0 * float(loglik) + k * np.log(max(n, 2))
        return aic, aicc, float(bic)

    def regime_of(self) -> np.ndarray | None:
        """
        A per-observation regime label, for models fitted with more than one regime.

        Deliberately on the base class rather than only on the regime model: a plot that shades
        regimes, a table that splits accuracy by regime, and a report sentence that says "in the
        high-volatility regime" all need to ask *any* model the same question. `None` means "this model
        has one regime", which is an answer, not a missing value.
        """
        return getattr(self, "_regime_labels", None)

    def params_table(self, raw: dict[str, tuple[float | None, float | None, bool]]) -> list[ParamRow]:
        """
        Build the parameter table from `{name: (value, std_err, at_boundary)}`.

        A parameter whose standard error is effectively zero is *at a boundary*, not infinitely
        precise: that is what ETS hitting the edge of its parameter space looks like, and reporting
        SE = 0 as certainty is exactly the wrong reading.
        """
        rows: list[ParamRow] = []
        for name, (val, se, at_boundary) in raw.items():
            z = p = None
            if val is not None and se not in (None, 0.0) and np.isfinite(se):
                z = float(val) / float(se)
                from scipy import stats as _st

                p = float(2 * _st.norm.sf(abs(z)))
            elif at_boundary:
                p = None
            rows.append(
                ParamRow(
                    name=name,
                    value=None if val is None else float(val),
                    std_err=None if se is None else float(se),
                    z=z,
                    p_value=p,
                    at_boundary=bool(at_boundary),
                    plain=(
                        "at the boundary of the parameter space — treat as an estimate, not a "
                        "measurement" if at_boundary else ""
                    ),
                )
            )
        return rows


__all__ = ["FitDiagnostics", "FitFailure", "FittedSummary", "ModelWrapper"]
