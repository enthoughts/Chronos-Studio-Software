"""
ts/select/tests_compare.py — is the difference between two forecasts real? (§5.9)

    "Diebold–Mariano (and Clark–West / MGN for nested models) between the top two."

Three tests, each answering a different question, and each with a precondition that decides whether it
may be quoted at all:

* **Diebold–Mariano** — are two *non-nested* forecasts equally accurate in expectation? Uses the loss
  differential series and a HAC (Newey–West) variance so autocorrelated multi-step errors do not
  inflate the statistic. Requires a loss differential that is not itself serially correlated beyond
  what the HAC correction handles.
* **Clark–West** — the *nested* case, where the DM test is invalid: the small model is a restricted
  version of the large one, so under the null the extra parameters are unidentified and the DM
  statistic does not have its nominal distribution. Clark–West adds the squared forecast-gap
  adjustment to the MSPE differential and uses a one-sided normal test on its HAC mean.
* **Mincer–Zarnowitz** — a regression of the actual on the forecast. A slope of 1 and intercept of 0
  means the forecast is *unbiased and efficient*. This is not a comparison between two models; it is
  a test of one, and it catches the failure DM cannot see: two forecasts that are equally wrong in the
  same way.

A deliberate design decision: **all three return a refusal rather than a p-value they cannot justify.**
Too few folds, a degenerate loss differential (identical forecasts), or a nested pair without the
Clark–West variant get `ok=False` and a sentence, not a number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.core.formatting import p_phrase
from scipy import stats


@dataclass
class DMResult:
    """The outcome of a Diebold–Mariano test, or its refusal."""

    ok: bool
    statistic: float | None = None
    p_value: float | None = None
    mean_loss_diff: float | None = None
    """Mean(loss_A − loss_B). Negative means A is more accurate."""
    n: int = 0
    lags: int = 0
    better: str = ""
    """The model with the lower mean loss, named — not "A" or "B"."""
    note: str = ""
    assumptions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "test": "diebold-mariano",
            "statistic": self.statistic,
            "p_value": self.p_value,
            "mean_loss_diff": self.mean_loss_diff,
            "n": self.n,
            "lags": self.lags,
            "better": self.better,
            "note": self.note,
            "assumptions": self.assumptions,
        }


@dataclass
class ClarkWestResult(DMResult):
    """Clark–West: the nested-model variant, with its adjustment disclosed."""
    adjusted_mean: float | None = None
    adjustment: float | None = None
    tail: str = "one-sided (large model better)"

    def as_dict(self) -> dict[str, Any]:
        out = super().as_dict()
        out.update({
            "test": "clark-west",
            "adjusted_mean": self.adjusted_mean,
            "adjustment": self.adjustment,
            "tail": self.tail,
        })
        return out


@dataclass
class MZResult:
    """Mincer–Zarnowitz: is this forecast unbiased and efficient?"""

    ok: bool
    intercept: float | None = None
    slope: float | None = None
    intercept_se: float | None = None
    slope_se: float | None = None
    #: Joint Wald test of (a=0, b=1) — the actual hypothesis. Testing the two separately with two
    #: t-tests doubles the type-I rate for no reason.
    joint_chi2: float | None = None
    joint_p: float | None = None
    n: int = 0
    verdict: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "test": "mincer-zarnowitz",
            "intercept": self.intercept,
            "slope": self.slope,
            "intercept_se": self.intercept_se,
            "slope_se": self.slope_se,
            "joint_chi2": self.joint_chi2,
            "joint_p": self.joint_p,
            "n": self.n,
            "verdict": self.verdict,
            "note": self.note,
        }


#: DM needs enough pairs for the HAC variance to mean anything. Below this the test is a coin flip
#: with a p-value attached.
MIN_PAIRS = 8


def _loss(errors: np.ndarray, kind: str) -> np.ndarray:
    e = np.asarray(errors, dtype=float)
    if kind == "absolute":
        return np.abs(e)
    if kind == "squared":
        return e * e
    raise ValueError(f"Unknown loss '{kind}'; use 'absolute' or 'squared'.")


def _hac_variance(d: np.ndarray, lags: int) -> float:
    """
    Newey–West long-run variance of the mean loss differential.

    Multi-step forecast errors overlap, so the differential is serially correlated by construction;
    using the sample variance would overstate the precision and manufacture significance.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    dc = d - d.mean()
    gamma0 = float((dc * dc).sum() / n)
    total = gamma0
    for k in range(1, lags + 1):
        if k >= n:
            break
        gk = float((dc[k:] * dc[:-k]).sum() / n)
        weight = 1.0 - k / (lags + 1.0)  # Bartlett
        total += 2.0 * weight * gk
    # A negative Bartlett estimate is possible with a bad lag choice. Returning a
    # tiny positive number would manufacture an enormous t-statistic and a
    # significant p-value from a failed variance estimate; refuse instead.
    return max(float(total), 0.0)


def diebold_mariano(
    errors_a: Sequence[float],
    errors_b: Sequence[float],
    *,
    name_a: str = "A",
    name_b: str = "B",
    horizon: int = 1,
    loss: str = "absolute",
) -> DMResult:
    """
    Diebold–Mariano on a paired loss differential.

    `errors_a` and `errors_b` must be the *same* forecast origin and the same target values — a DM
    test on unpaired errors is a comparison of two unrelated samples with a time-series accent.

    The HAC lag is `horizon − 1`, which is the number of overlaps an h-step-ahead forecast creates.
    That is the minimum that removes the induced correlation; more lags trades power for robustness
    and is not warranted here.
    """
    ea = np.asarray(errors_a, dtype=float)
    eb = np.asarray(errors_b, dtype=float)
    n = min(ea.size, eb.size)
    if n < MIN_PAIRS:
        return DMResult(
            ok=False,
            n=n,
            note=(
                f"Only {n} paired errors: the Diebold–Mariano test needs at least {MIN_PAIRS} before "
                "its asymptotic approximation is anything but a formality."
            ),
        )
    ea, eb = ea[:n], eb[:n]
    mask = np.isfinite(ea) & np.isfinite(eb)
    if mask.sum() < MIN_PAIRS:
        return DMResult(
            ok=False, n=int(mask.sum()),
            note=f"Only {int(mask.sum())} of {n} pairs are finite; the rest cannot be compared.",
        )
    ea, eb = ea[mask], eb[mask]
    n = ea.size

    la, lb = _loss(ea, loss), _loss(eb, loss)
    d = la - lb
    if np.allclose(d, 0.0):
        return DMResult(
            ok=False, n=n, mean_loss_diff=0.0,
            note=(
                "The two forecasts have identical loss on every point, so there is no difference for "
                "the test to detect. That is a finding, not a p-value: pick between them on something "
                "else."
            ),
            better="neither",
        )

    lags = max(0, int(horizon) - 1)
    lrv = _hac_variance(d, lags)
    se = math.sqrt(lrv / n)
    if not math.isfinite(lrv) or lrv <= 0 or not math.isfinite(se) or se <= 0:
        return DMResult(
            ok=False, n=n,
            note="The HAC long-run variance is zero or undefined; the statistic is refused "
                 "rather than being divided by a manufactured epsilon.",
        )

    stat = float(d.mean() / se)
    p = float(2.0 * stats.norm.sf(abs(stat)))
    better = name_a if d.mean() < 0 else name_b

    note = (
        f"{better} has the lower mean {loss} loss by {abs(d.mean()):.4g} per point. "
        + (
            f"{p_phrase(p, 3)} — not significant at 5%, so the ranking between these two is within noise."
            if p >= 0.05
            else f"{p_phrase(p, 4)} — the difference is larger than forecast noise explains."
        )
    )
    return DMResult(
        ok=True,
        statistic=round(stat, 4),
        p_value=round(p, 4),
        mean_loss_diff=round(float(d.mean()), 6),
        n=n,
        lags=lags,
        better=better,
        note=note,
        assumptions=[
            "The loss differential is covariance-stationary; a series with a break in forecast "
            "performance breaks this.",
            f"The HAC variance uses {lags} lag(s), the overlap a {horizon}-step horizon creates.",
            "Non-nested forecasts. For nested models use Clark–West, where the DM statistic is invalid.",
        ],
    )


def clark_west(
    errors_small: Sequence[float],
    errors_large: Sequence[float],
    *,
    name_small: str = "restricted",
    name_large: str = "unrestricted",
    horizon: int = 1,
    loss: str = "squared",
    forecast_gap: Sequence[float] | None = None,
) -> ClarkWestResult:
    """
    Clark & West (2007) MSPE-adjusted test for a nested pair.

    With squared loss the adjusted differential is

        f_t = e_small² − e_large² + (f_small − f_large)²
            = e_small² − e_large² + (e_large − e_small)² .

    The last term is the estimation-noise handicap: under the null the larger
    model's extra parameters are zero and its out-of-sample MSPE is biased
    upward. The test is one-sided (the unrestricted model is better when the
    mean adjusted differential is positive) and uses a HAC variance of f_t.
    Clark–West is only defined for squared loss; other losses are refused.

    `forecast_gap`, when supplied, is `f_small − f_large` and is preferred to
    reconstructing it from errors, because it makes the caller's pairing
    explicit. It is derived from `benchmark_forecast − forecast` in S14.
    """
    if loss != "squared":
        return ClarkWestResult(
            ok=False,
            note="Clark–West is derived for squared loss; use Diebold–Mariano for another loss.",
        )
    es = np.asarray(errors_small, dtype=float)
    el = np.asarray(errors_large, dtype=float)
    n = min(es.size, el.size)
    if n < MIN_PAIRS:
        return ClarkWestResult(
            ok=False, n=n,
            note=f"Only {n} paired errors; Clark–West needs at least {MIN_PAIRS}.",
        )
    es, el = es[:n], el[:n]
    if forecast_gap is not None:
        gap_raw = np.asarray(forecast_gap, dtype=float).reshape(-1)
        gap_full = np.full(n, np.nan, dtype=float)
        gap_full[: min(n, gap_raw.size)] = gap_raw[: min(n, gap_raw.size)]
    else:
        gap_full = el - es
    mask = np.isfinite(es) & np.isfinite(el) & np.isfinite(gap_full)
    es, el, gap = es[mask], el[mask], gap_full[mask]
    n = es.size
    if n < MIN_PAIRS:
        return ClarkWestResult(ok=False, n=n, note="Not enough finite pairs.")

    ls, ll = _loss(es, "squared"), _loss(el, "squared")
    raw = ls - ll
    adjustment = gap * gap
    adjusted = raw + adjustment
    mean_raw = float(raw.mean())
    mean_adj = float(adjusted.mean())
    if np.allclose(adjusted, 0.0):
        return ClarkWestResult(
            ok=False, n=n, mean_loss_diff=mean_raw,
            adjusted_mean=mean_adj, adjustment=float(adjustment.mean()),
            better="neither",
            note="The adjusted loss differential is identically zero; there is nothing to compare.",
        )

    lags = max(0, int(horizon) - 1)
    lrv = _hac_variance(adjusted, lags)
    if not math.isfinite(lrv) or lrv <= 0:
        return ClarkWestResult(
            ok=False, n=n, mean_loss_diff=mean_raw,
            adjusted_mean=mean_adj, adjustment=float(adjustment.mean()),
            note="The HAC long-run variance of the adjusted differential is zero or undefined; "
                 "the statistic is refused rather than divided by a manufactured epsilon.",
        )
    se = math.sqrt(lrv / n)
    stat = float(mean_adj / se)
    p = float(stats.norm.sf(stat))  # one-sided: H1 = unrestricted model is better
    better = name_small if mean_raw < 0 else name_large

    verdict_bits = [
        f"The restricted model ({name_small}) has the lower raw mean squared loss."
        if mean_raw < 0
        else f"The unrestricted model ({name_large}) has the lower raw mean squared loss.",
        f"Clark–West adjustment adds mean (f_small − f_large)² = {float(adjustment.mean()):.4g} "
        "to the raw MSPE differential; the adjusted mean is "
        f"{mean_adj:.4g}.",
        f"One-sided {p_phrase(p, 4)}."
        + ("" if p < 0.05 else " Not significant: the nesting gain is not established."),
    ]
    return ClarkWestResult(
        ok=True,
        statistic=round(stat, 4),
        p_value=round(p, 4),
        mean_loss_diff=round(mean_raw, 6),
        adjusted_mean=round(mean_adj, 6),
        adjustment=round(float(adjustment.mean()), 6),
        n=n,
        lags=lags,
        better=better,
        note=" ".join(verdict_bits),
        assumptions=[
            "The models are genuinely nested — the small model's parameter space is a restriction of "
            "the large one's. If they are not, this test is wrong and Diebold–Mariano is right.",
            "Squared loss, which is the loss the Clark–West derivation assumes.",
            "The large model's extra parameters are unidentified under the null; the "
            "(f_small − f_large)² adjustment accounts for that and cannot be skipped.",
            "The p-value is one-sided: positive adjusted differential favours the unrestricted model.",
            "Multi-step errors overlap; the HAC lag is horizon − 1.",
        ],
    )


def mincer_zarnowitz(
    actual: Sequence[float],
    forecast: Sequence[float],
    *,
    name: str = "the model",
    horizon: int = 1,
) -> MZResult:
    """
    Regress actual on forecast: `y = a + b·f + ε`.

    Unbiased and efficient means a = 0 and b = 1 **jointly**. Three readings matter:

    * b significantly < 1 → the forecast is too variable (it overshoots); b > 1 → too flat.
    * a ≠ 0 with b ≈ 1 → a constant bias, which a user can correct by subtraction.
    * a ≠ 0 and b ≠ 1 together → the forecast is a biased *and* mis-scaled transformation of the truth.

    With `horizon > 1` the errors overlap and the standard errors are understated; the note says so
    rather than silently quoting a too-small p.
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    n = min(a.size, f.size)
    if n < MIN_PAIRS:
        return MZResult(ok=False, n=n, note=f"Only {n} pairs; the regression needs at least {MIN_PAIRS}.")
    a, f = a[:n], f[:n]
    mask = np.isfinite(a) & np.isfinite(f)
    a, f = a[mask], f[mask]
    n = a.size
    if n < MIN_PAIRS:
        return MZResult(ok=False, n=n, note="Not enough finite pairs.")
    if np.allclose(f, f[0]):
        return MZResult(
            ok=False, n=n,
            note=(
                "The forecast is constant, so the slope is unidentified — the regression has no "
                "variation to work with. Report the bias alone instead."
            ),
        )

    X = np.column_stack([np.ones(n), f])
    try:
        beta, *_ = np.linalg.lstsq(X, a, rcond=None)
        resid = a - X @ beta
        dof = n - 2
        if dof < 1:
            return MZResult(ok=False, n=n, note="Too few observations to estimate a two-parameter regression.")
        sigma2 = float((resid**2).sum() / dof)
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
    except np.linalg.LinAlgError:
        return MZResult(ok=False, n=n, note="The regression design is singular.")

    # Joint Wald test of (a, b) = (0, 1), which is the hypothesis anyone actually cares about.
    diff = beta - np.array([0.0, 1.0])
    try:
        chi2 = float(diff @ np.linalg.inv(XtX_inv * sigma2) @ diff)
    except np.linalg.LinAlgError:
        chi2 = float("nan")
    p = float(stats.chi2.sf(chi2, 2)) if math.isfinite(chi2) else None

    if p is None:
        verdict = "The joint test could not be computed."
    elif p >= 0.05:
        verdict = (
            f"{name} passes: the intercept is not distinguishable from 0 and the slope not from 1 "
            f"(joint test, {p_phrase(p, 3)}). The forecast is unbiased and efficient."
        )
    elif beta[1] < 1:
        verdict = (
            f"{name} fails the joint test ({p_phrase(p, 4)}): the slope is {beta[1]:.3f}, below 1, so the "
            "forecast varies more than the outcome does — it overshoots the movements."
        )
    else:
        verdict = (
            f"{name} fails the joint test ({p_phrase(p, 4)}): the slope is {beta[1]:.3f}, above 1, so the "
            "forecast is flatter than the outcome — it under-reacts."
        )

    note = ""
    if horizon > 1:
        note = (
            f"The test horizon is {horizon}, so the errors overlap and the standard errors here are "
            "understated; treat the p-value as indicative."
        )
    return MZResult(
        ok=True,
        intercept=round(float(beta[0]), 6),
        slope=round(float(beta[1]), 6),
        intercept_se=round(float(se[0]), 6),
        slope_se=round(float(se[1]), 6),
        joint_chi2=round(chi2, 4) if math.isfinite(chi2) else None,
        joint_p=round(p, 4) if p is not None else None,
        n=n,
        verdict=verdict,
        note=note,
    )


__all__ = [
    "MIN_PAIRS",
    "ClarkWestResult",
    "DMResult",
    "MZResult",
    "clark_west",
    "diebold_mariano",
    "mincer_zarnowitz",
]
