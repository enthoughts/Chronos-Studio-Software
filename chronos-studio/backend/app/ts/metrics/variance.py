"""
ts/metrics/variance.py — scoring a volatility / variance target (§5.12).

A variance forecast is not a mean forecast with the sign removed. QLIKE is the
proper scoring rule under a Gaussian return model, MSE of variance is the blunt
companion, and VaR backtests ask whether the tail violations occur at the right
*rate* (Kupiec) and independently through time (Christoffersen). All four have
low power in short samples, so the results carry that warning instead of reading
a non-rejection as a pass.

The realised variance proxy here is the squared return r_t². That is noisy — one
observation estimates one variance with no degrees of freedom — and is stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

_Z975 = float(stats.norm.ppf(0.975))


@dataclass
class VaRBacktests:
    n: int = 0
    violations: int = 0
    alpha: float = 0.05
    kupiec_lr: float | None = None
    kupiec_p: float | None = None
    christoffersen_lr: float | None = None
    christoffersen_p: float | None = None
    conditional_lr: float | None = None
    conditional_p: float | None = None
    independence_ok: bool | None = None
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "violations": self.violations, "alpha": self.alpha,
            "kupiec_lr": self.kupiec_lr, "kupiec_p": self.kupiec_p,
            "christoffersen_lr": self.christoffersen_lr,
            "christoffersen_p": self.christoffersen_p,
            "conditional_lr": self.conditional_lr, "conditional_p": self.conditional_p,
            "independence_ok": self.independence_ok,
            "note": self.note, "warnings": list(self.warnings),
        }


@dataclass
class VarianceScores:
    n: int = 0
    qlike: float | None = None
    mse_variance: float | None = None
    var: VaRBacktests = field(default_factory=VaRBacktests)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n, "qlike": self.qlike, "mse_variance": self.mse_variance,
            "var_backtests": self.var.as_dict(),
            "warnings": list(self.warnings), "notes": list(self.notes),
        }


def qlike(realized_var: Sequence[float], forecast_var: Sequence[float]) -> float | None:
    """QLIKE = mean(log(v_hat) + v_real / v_hat). Lower is better."""
    r = np.asarray(realized_var, dtype=float).reshape(-1)
    v = np.asarray(forecast_var, dtype=float).reshape(-1)
    n = min(r.size, v.size)
    mask = np.isfinite(r[:n]) & np.isfinite(v[:n]) & (v[:n] > 0)
    if not mask.any():
        return None
    rv, fv = r[:n][mask], np.maximum(v[:n][mask], 1e-12)
    return float(np.mean(np.log(fv) + rv / fv))


def mse_variance(realized_var: Sequence[float], forecast_var: Sequence[float]) -> float | None:
    r = np.asarray(realized_var, dtype=float).reshape(-1)
    v = np.asarray(forecast_var, dtype=float).reshape(-1)
    n = min(r.size, v.size)
    mask = np.isfinite(r[:n]) & np.isfinite(v[:n])
    return float(np.mean((r[:n][mask] - v[:n][mask]) ** 2)) if mask.any() else None


def _loglik(p: float, k: int, n: int) -> float:
    p = float(min(max(p, 1e-12), 1 - 1e-12))
    return float(k * math.log(p) + (n - k) * math.log(1 - p))


def kupiec(violations: Sequence[bool] | Sequence[int], alpha: float = 0.05) -> tuple[float, float]:
    """Unconditional-coverage likelihood-ratio statistic and chi-square(1) p."""
    x = np.asarray(violations, dtype=int).reshape(-1)
    n, k = int(x.size), int(x.sum())
    if n < 10:
        return float("nan"), float("nan")
    p_hat = k / n
    ll_null = _loglik(alpha, k, n)
    ll_alt = _loglik(p_hat, k, n)
    lr = max(0.0, -2.0 * (ll_null - ll_alt))
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen(violations: Sequence[bool] | Sequence[int]) -> tuple[float, float, bool | None]:
    """Independence LR, its p-value, and whether clustered violations were detected."""
    x = np.asarray(violations, dtype=int).reshape(-1)
    n = int(x.size)
    if n < 11:
        return float("nan"), float("nan"), None
    prev, cur = x[:-1], x[1:]
    n00 = int(((prev == 0) & (cur == 0)).sum())
    n01 = int(((prev == 0) & (cur == 1)).sum())
    n10 = int(((prev == 1) & (cur == 0)).sum())
    n11 = int(((prev == 1) & (cur == 1)).sum())
    if (n00 + n01) == 0 or (n10 + n11) == 0:
        return float("nan"), float("nan"), None
    p01 = n01 / (n00 + n01)
    p11 = n11 / (n10 + n11)
    p = (n01 + n11) / max(n - 1, 1)
    ll_iid = _loglik(min(max(p, 1e-12), 1 - 1e-12), n01 + n11, n - 1)
    ll_markov = (
        n00 * math.log(1 - min(max(p01, 1e-12), 1 - 1e-12))
        + n01 * math.log(min(max(p01, 1e-12), 1 - 1e-12))
        + n10 * math.log(1 - min(max(p11, 1e-12), 1 - 1e-12))
        + n11 * math.log(min(max(p11, 1e-12), 1 - 1e-12))
    )
    lr = max(0.0, -2.0 * (ll_iid - ll_markov))
    p_value = float(stats.chi2.sf(lr, 1))
    return float(lr), p_value, bool(p11 > p01 + 0.10)


def var_backtests(
    returns: Sequence[float],
    forecast_sd: Sequence[float],
    *,
    alpha: float = 0.05,
) -> VaRBacktests:
    """Lower-tail VaR violations and the two coverage tests."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    s = np.asarray(forecast_sd, dtype=float).reshape(-1)
    n = min(r.size, s.size)
    mask = np.isfinite(r[:n]) & np.isfinite(s[:n]) & (s[:n] > 0)
    r, s = r[:n][mask], s[:n][mask]
    out = VaRBacktests(n=int(r.size), alpha=float(alpha))
    if r.size == 0:
        out.note = "No finite return / forecast-sd pairs were available."
        return out
    z = float(stats.norm.ppf(1.0 - alpha))
    hit = (r < -z * s).astype(int)
    out.violations = int(hit.sum())
    lr_uc, p_uc = kupiec(hit, alpha)
    lr_ind, p_ind, clustered = christoffersen(hit)
    out.kupiec_lr, out.kupiec_p = lr_uc, p_uc
    out.christoffersen_lr, out.christoffersen_p = lr_ind, p_ind
    out.independence_ok = None if clustered is None else not clustered
    if math.isfinite(lr_uc) and math.isfinite(lr_ind):
        cc = lr_uc + lr_ind
        out.conditional_lr = float(cc)
        out.conditional_p = float(stats.chi2.sf(cc, 2))
    out.note = (
        f"{out.violations} VaR violation(s) in {out.n} returns at a nominal {alpha:.0%} tail. "
        "Kupiec tests the rate; Christoffersen tests whether violations cluster."
    )
    if out.n < 60:
        out.warnings.append(
            "Fewer than 60 returns: both VaR backtests have low power, so a non-rejection is not "
            "evidence of a correct tail."
        )
    if clustered:
        out.warnings.append("Violations cluster in time; conditional coverage is questionable.")
    return out


def variance_scores(
    *,
    realized_var: Sequence[float],
    forecast_var: Sequence[float],
    returns: Sequence[float],
    forecast_sd: Sequence[float],
    alpha: float = 0.05,
) -> VarianceScores:
    out = VarianceScores(
        n=int(min(np.asarray(realized_var).size, np.asarray(forecast_var).size)),
        qlike=qlike(realized_var, forecast_var),
        mse_variance=mse_variance(realized_var, forecast_var),
        var=var_backtests(returns, forecast_sd, alpha=alpha),
    )
    out.notes.extend([
        "QLIKE = mean(log(v_hat) + v_real / v_hat); lower is better.",
        "Realised variance is proxied by squared returns, which is a noisy one-observation "
        "estimate; the scores are comparative, not a calibration test on their own.",
        "VaR uses the normal lower-tail quantile of the model's forecast sd; the Gaussian "
        "assumption is reported rather than hidden.",
    ])
    return out


__all__ = [
    "VaRBacktests", "VarianceScores", "christoffersen", "kupiec",
    "mse_variance", "qlike", "var_backtests", "variance_scores",
]
