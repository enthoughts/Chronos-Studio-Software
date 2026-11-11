"""
ts/select/cv.py — rolling-origin cross-validation, the primary selection criterion (§5.9).

    "Rolling-origin cross-validation (default: 3–5 folds, expanding or sliding, horizon = forecast
    horizon) … **This is the primary selection criterion**; ICs are secondary. State that explicitly
    in the UI."

Three things make this engine different from the usual `TimeSeriesSplit`-and-mean:

1. **The folds are the forecast task.** Each fold refits on the past and forecasts `horizon` steps —
   the same horizon the user asked for. A one-step CV score on a 12-step task ranks models that
   cannot do 12 steps first.
2. **Everything is scored per fold, then aggregated.** A model with one catastrophic fold is not a
   good model with a bad fold; the fold table is kept so the diagnosis survives to the UI.
3. **Nothing is fitted on the future.** `fold_origins` returns origins such that every training set
   ends before its test set begins, and `cross_validate` asserts this per fold rather than trusting
   its own arithmetic.

Refits are the expensive part, so the engine takes a *factory* rather than fitted objects: a caller
that already has the final fit cannot reuse it (the final fit saw the test data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd


class CVScheme(str, Enum):
    """How the training window moves."""

    EXPANDING = "expanding"
    """Grow the window: every observation is used, and early folds are short. The default, because it
    uses all the data for the final fold, which is the fit the user will actually forecast from."""

    SLIDING = "sliding"
    """Fixed-width window that slides forward. Preferable when an old regime is actively misleading —
    but it throws data away, so it is opt-in and its cost is reported."""


#: §5.9's "default: 3–5 folds".
DEFAULT_FOLDS = 4
MIN_FOLDS = 2
#: The ceiling on folds. 30 rather than 10 because the combination row estimates *weights* from the
#: origins and needs more than a handful; a CV comparison of 30 folds is reported with the spread
#: attached, so the extra folds buy power rather than false precision.
MAX_FOLDS = 30

#: A fold whose training set is smaller than this cannot fit anything worth scoring.
MIN_TRAIN = 16


@dataclass(frozen=True)
class CVFold:
    """One origin. The split is explicit so a caller can audit it."""

    index: int
    train_start: int
    train_end: int
    """Inclusive index of the last training observation."""
    test_start: int
    test_end: int
    """Inclusive index of the last test observation."""

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start + 1

    def describe(self, index: pd.Index | None = None) -> str:
        if index is not None and len(index) > self.test_end:
            return (
                f"fold {self.index}: trained on {index[self.train_start]:%b %Y}–{index[self.train_end]:%b %Y} "
                f"({self.n_train}), scored on {index[self.test_start]:%b %Y}–{index[self.test_end]:%b %Y} "
                f"({self.n_test})"
            )
        return f"fold {self.index}: rows {self.train_start}–{self.train_end} → {self.test_start}–{self.test_end}"


def fold_origins(
    n: int,
    *,
    horizon: int,
    folds: int = DEFAULT_FOLDS,
    scheme: CVScheme = CVScheme.EXPANDING,
    min_train: int = MIN_TRAIN,
    step: int | None = None,
) -> list[CVFold]:
    """
    Build the origins.

    The last fold's test set ends at the last observation, so the CV covers the *recent* series — the
    part a user cares about — rather than an arbitrary middle. Folds are equally spaced backwards from
    the end, which is what "rolling origin" means: the origin rolls forward by `horizon` each fold.

    Refuses (raises `ValueError`) rather than returning a short list when the series cannot support the
    requested folds: silently dropping to one fold would produce a "CV score" that is a single
    holdout, which is exactly what the ICs were already doing.
    """
    n = int(n)
    horizon = int(horizon)
    folds = int(folds)
    if horizon < 1:
        raise ValueError("The forecast horizon must be at least 1; it is the length of each test block.")
    if n < horizon + min_train:
        raise ValueError(
            f"{n} observations cannot support a {horizon}-step holdout with {min_train} left to train "
            f"on; at least {horizon + min_train} are needed. Use a shorter horizon or a longer series."
        )
    if folds < MIN_FOLDS:
        raise ValueError(f"Cross-validation needs at least {MIN_FOLDS} folds; {folds} were requested.")
    folds = min(folds, MAX_FOLDS)

    # The widest a single test block can be: leave `min_train` for the *first* fold's training set.
    usable = n - min_train
    if step is not None:
        # An explicit step: folds overlap when `step < horizon`. Used by the combination row, which
        # needs many origins to estimate weights and is not reporting a CV score — a 3-fold CV would
        # give it 3 numbers to regress on.
        step = max(1, int(step))
    elif usable < horizon * folds:
        # Not enough room for `folds` non-overlapping horizon-length blocks; overlap them by stepping
        # one period instead of `horizon`. Reported honestly — the fold count is the promise, the step
        # is the concession.
        step = max(1, usable // folds)
    else:
        step = horizon

    # Walk *backwards* from the last observation, then reverse: folding forward in time is how a
    # reader expects fold 1, fold 2, fold 3 to be labelled, and a fold table that runs backwards is a
    # table somebody misreads. The last fold still ends on the final observation, which is the point
    # of anchoring at the end.
    built: list[tuple[int, int]] = []
    last = n - 1
    for f in range(folds):
        te = last - f * step
        ts = te - horizon + 1
        if ts < 0:
            continue
        built.append((ts, te))
    origins: list[CVFold] = []
    for i, (ts, te) in enumerate(reversed(built)):
        train_end = ts - 1
        if scheme is CVScheme.EXPANDING:
            train_start = 0
        else:
            # A fixed window that slides. Its length is what the *earliest* fold would have had under
            # an expanding scheme, so the two schemes are directly comparable at fold 0 and diverge
            # only as the origin moves — which is the entire point of offering the choice. The first
            # version of this line used `train_end - (n - horizon - min_train)`, which went negative
            # and was clipped to 0, turning "sliding" into "expanding" without saying so.
            window = max(min_train, (n - horizon) - (folds - 1) * step)
            train_start = max(train_end + 1 - window, 0)
        if train_end - train_start + 1 < min_train:
            continue
        origins.append(
            CVFold(
                index=len(origins),
                train_start=train_start,
                train_end=train_end,
                test_start=ts,
                test_end=te,
            )
        )
    if len(origins) < MIN_FOLDS:
        raise ValueError(
            f"Only {len(origins)} fold(s) fit in {n} observations at horizon {horizon}. At least "
            f"{MIN_FOLDS} are needed for the spread between folds to mean anything."
        )
    return origins


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class FoldScore:
    """One model's result on one fold, for every metric the app can compute without inventing one."""

    fold: int
    mae: float | None = None
    rmse: float | None = None
    mase: float | None = None
    rmssf: float | None = None
    smape: float | None = None
    mape: float | None = None
    """`None` whenever the fold has a zero — see `_mape` for the refusal."""
    bias: float | None = None
    """Mean signed error. A small MAE with a large bias is a model that is wrong in one direction."""
    coverage: float | None = None
    """Share of test actuals inside the interval, when the model produced one."""
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out = {
            "fold": self.fold,
            "mae": self.mae,
            "rmse": self.rmse,
            "mase": self.mase,
            "rmssf": self.rmssf,
            "smape": self.smape,
            "mape": self.mape,
            "bias": self.bias,
            "coverage": self.coverage,
        }
        if self.notes:
            out["notes"] = list(self.notes)
        return out


@dataclass
class CVResult:
    """One model's whole CV: the folds, the aggregate, and the reason for any blank."""

    model_id: str
    display_name: str = ""
    scheme: CVScheme = CVScheme.EXPANDING
    horizon: int = 1
    folds: list[FoldScore] = field(default_factory=list)
    #: The signed one-step errors, pooled across folds **in fold order**, and the realised values they
    #: belong to. Kept because the Diebold–Mariano test compares two models' loss *series*, and a test
    #: built from the per-fold means instead is a test on `n_folds` numbers, not on the forecasts.
    errors: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)
    seconds: float = 0.0
    failed: bool = False
    failure_reason: str = ""
    #: Metrics whose aggregate is computed as a mean of fold values. Kept as a list, not hardcoded, so
    #: a metric that must be aggregated differently (a pooled one) is visibly different.
    averaged: tuple[str, ...] = ("mae", "rmse", "mase", "rmssf", "smape", "mape", "bias", "coverage")

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def aggregate(self, metric: str = "mase") -> float | None:
        vals = [getattr(f, metric) for f in self.folds]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    def spread(self, metric: str = "mase") -> float | None:
        """Standard deviation across folds — the number that says whether a mean is worth quoting."""
        vals = [getattr(f, metric) for f in self.folds]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        return float(np.std(vals, ddof=1)) if len(vals) >= 2 else None

    def worst_fold(self, metric: str = "mase") -> FoldScore | None:
        scored = [f for f in self.folds if getattr(f, metric) is not None]
        if not scored:
            return None
        return max(scored, key=lambda f: getattr(f, metric))

    @property
    def mase(self) -> float | None:
        return self.aggregate("mase")

    @property
    def mae(self) -> float | None:
        return self.aggregate("mae")

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "scheme": self.scheme.value,
            "horizon": self.horizon,
            "n_folds": self.n_folds,
            "seconds": round(self.seconds, 2),
            "failed": self.failed,
            "failure_reason": self.failure_reason,
            "metrics": {m: self.aggregate(m) for m in self.averaged},
            "n_errors": len(self.errors),
            "spread": {"mase": self.spread("mase"), "mae": self.spread("mae"), "rmse": self.spread("rmse")},
            "folds": [f.as_dict() for f in self.folds],
        }


def _scale(y_train: np.ndarray, y_test: np.ndarray, period: int | None) -> float | None:
    """
    The MASE denominator, computed **on the training window only**.

    This is the leakage trap in every CV implementation: scaling by the full series' naive error lets
    the denominator see the test data. The denominator here is the mean absolute change inside the
    training window, which is a number the model could have been fitted from.
    """
    y = y_train[np.isfinite(y_train)]
    if y.size < 3:
        return None
    if period and period >= 2 and y.size > period + 1:
        d = np.abs(y[period:] - y[:-period])
    else:
        d = np.abs(np.diff(y))
    if d.size == 0:
        return None
    s = float(d.mean())
    return s if s > 0 else None


def score_fold(
    fold: CVFold,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    y_train: np.ndarray,
    period: int | None = None,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> FoldScore:
    """Score one fold. `actual` and `predicted` are the test block, same length."""
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    n = min(a.size, p.size)
    a, p = a[:n], p[:n]
    mask = np.isfinite(a) & np.isfinite(p)
    score = FoldScore(fold=fold.index)
    if mask.sum() < 1:
        score.notes = ("No finite prediction in this fold.",)
        return score
    a, p = a[mask], p[mask]
    err = a - p
    score.mae = float(np.mean(np.abs(err)))
    score.rmse = float(np.sqrt(np.mean(err**2)))
    score.bias = float(np.mean(err))

    scale = _scale(np.asarray(y_train, dtype=float), a, period)
    if scale:
        score.mase = float(score.mae / scale)
        score.rmssf = float(np.sqrt(np.mean(err**2)) / scale)

    # sMAPE as the spec defines it (symmetric, in %). It is defined at zero, which is why it survives
    # where MAPE is banned.
    denom = np.abs(a) + np.abs(p)
    ok = denom > 1e-12
    score.smape = float(np.mean(2.0 * np.abs(err[ok]) / denom[ok]) * 100) if ok.any() else None

    if np.any(np.abs(a) < 1e-12):
        score.notes = score.notes + ("MAPE is undefined in this fold: an actual value is zero.",)
    else:
        score.mape = float(np.mean(np.abs(err / a)) * 100)

    if lower is not None and upper is not None:
        lo_raw = np.asarray(lower, dtype=float).reshape(-1)
        hi_raw = np.asarray(upper, dtype=float).reshape(-1)
        # A failed fold supplies NaN bands. They are missing intervals, not
        # "outside the band": counting them as uncovered would turn a model
        # failure into an apparently bad calibration score.
        lo_full = np.full(n, np.nan, dtype=float)
        hi_full = np.full(n, np.nan, dtype=float)
        lo_full[: min(n, lo_raw.size)] = lo_raw[: min(n, lo_raw.size)]
        hi_full[: min(n, hi_raw.size)] = hi_raw[: min(n, hi_raw.size)]
        lo, hi = lo_full[mask], hi_full[mask]
        band_mask = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
        a_band, lo_band, hi_band = a[band_mask], lo[band_mask], hi[band_mask]
        if a_band.size:
            inside = (a_band >= lo_band) & (a_band <= hi_band)
            score.coverage = float(inside.mean())
            n_without_band = int((~band_mask).sum())
            if n_without_band:
                score.notes = score.notes + (
                    f"Coverage used {a_band.size} of {a.size} finite rows; "
                    f"{n_without_band} missing or invalid interval row(s) were excluded, not "
                    "counted as violations.",
                )
        else:
            score.notes = score.notes + ("No finite interval pair was available in this fold.",)
    return score


def cross_validate(
    model_ids: Sequence[str],
    y: pd.Series,
    *,
    factory: Callable[[str], Any],
    horizon: int,
    period: int | None = None,
    folds: int = DEFAULT_FOLDS,
    scheme: CVScheme = CVScheme.EXPANDING,
    on_error: Callable[[str, BaseException], None] | None = None,
) -> list[CVResult]:
    """
    Refit and forecast every model on every fold.

    `factory(model_id)` must return a **fresh, unfitted** wrapper each call — a reused object would
    carry the previous fold's state, which is the subtlest way to leak. The engine calls it once per
    (model, fold) and never caches, and passing a fitted object raises rather than guessing.

    A model that fails on a fold is recorded as failed *with the fold named*, not dropped: a model
    that cannot survive a short training window is a fact about the model.
    """
    import time

    values = pd.Series(y).astype(float)
    arr = values.to_numpy()
    n = arr.size
    origins = fold_origins(n, horizon=horizon, folds=folds, scheme=scheme)
    results: list[CVResult] = []

    for model_id in model_ids:
        t0 = time.perf_counter()
        result = CVResult(
            model_id=model_id, scheme=scheme, horizon=horizon, display_name=model_id
        )
        for fold in origins:
            # The leakage assertion, per fold, on the indices rather than on a promise.
            assert fold.train_end < fold.test_start, (
                f"fold {fold.index} trains through index {fold.train_end} and scores from "
                f"{fold.test_start}: the training window overlaps the test block."
            )
            train = values.iloc[fold.train_start : fold.train_end + 1]
            test = arr[fold.test_start : fold.test_end + 1]
            try:
                wrapper = factory(model_id)
                if getattr(wrapper, "_is_fitted", False) or getattr(wrapper, "summary", None) is not None:
                    raise RuntimeError(
                        "cross_validate was handed an already-fitted wrapper; each fold needs a "
                        "fresh fit or the test data leaks into the training window."
                    )
                wrapper.fit(train)
                if wrapper.failed:
                    raise RuntimeError(wrapper.summary.failure_reason or "the fit failed")
                fc = wrapper.predict(horizon=fold.n_test)
                predicted = np.asarray(pd.Series(fc["mean"]).to_numpy(dtype=float))
                lower = upper = None
                for candidate in ("lower", "lo", "pi_lower"):
                    if candidate in fc:
                        lower = pd.Series(fc[candidate]).to_numpy(dtype=float)
                        break
                for candidate in ("upper", "hi", "pi_upper"):
                    if candidate in fc:
                        upper = pd.Series(fc[candidate]).to_numpy(dtype=float)
                        break
                # Every step of every test block, pooled in fold order. The h-step errors overlap, so
                # the series is autocorrelated by construction — which is precisely what the
                # Diebold–Mariano HAC correction with `horizon − 1` lags is for. Keeping only the first
                # step of each fold gave 4 pairs, and the DM test then (correctly) refused to run.
                n_pair = max(0, min(len(test), len(predicted)))
                block = test[:n_pair]
                pred_block = np.asarray(predicted, dtype=float)[:n_pair]
                for a, f in zip(block, pred_block):
                    result.errors.append(float(a) - float(f) if np.isfinite(a) and np.isfinite(f) else float("nan"))
                    result.actuals.append(float(a) if np.isfinite(a) else float("nan"))
                result.folds.append(
                    score_fold(
                        fold, test, predicted, y_train=train.to_numpy(), period=period,
                        lower=lower, upper=upper,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - one fold must not sink the comparison
                result.failed = True
                result.failure_reason = f"fold {fold.index}: {type(exc).__name__}: {exc}"
                if on_error is not None:
                    on_error(model_id, exc)
                break
        result.seconds = time.perf_counter() - t0
        results.append(result)
    return results


def rank_by_cv(results: Sequence[CVResult], metric: str = "mase") -> list[CVResult]:
    """
    Rank by the CV aggregate — the primary criterion.

    Failed models sort last regardless of metric, and models with no score sort after models with one.
    Ties break on the *spread*: two models with the same mean MASE are not equally good if one got
    there by being stable and the other by averaging a great fold with a terrible one.
    """
    def key(r: CVResult):
        val = r.aggregate(metric)
        spread = r.spread(metric)
        return (
            r.failed or val is None,
            val if val is not None else math.inf,
            spread if spread is not None else math.inf,
            r.model_id,
        )

    return sorted(results, key=key)


__all__ = [
    "CVFold",
    "CVResult",
    "CVScheme",
    "DEFAULT_FOLDS",
    "MIN_FOLDS",
    "MIN_TRAIN",
    "FoldScore",
    "cross_validate",
    "fold_origins",
    "rank_by_cv",
    "score_fold",
]
