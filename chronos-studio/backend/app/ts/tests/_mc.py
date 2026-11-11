"""
ts/tests/_mc.py — Monte-Carlo critical values, honestly labelled.

Several tests in S5 have finite-sample distributions that statsmodels does not ship
(seasonal unit roots, sup-Wald for an unknown break). The alternatives were: hard-code
asymptotic tables that are wrong for a 72-point series, or simulate.

Simulation is the right answer *provided the user is told*. Every result that uses one
carries a caveat saying so, the replication count, and the seed — so the number can be
reproduced exactly, which is what makes it evidence rather than magic.

Critical values are cached per (test, n, params) for the session: the same series asked
about twice must not produce two different quantiles.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any, Callable

import numpy as np

#: How many replications. Enough for a stable 5% quantile, fast enough for a web request.
N_REPS = 400
#: Below this many observations a simulated quantile is noise; the caller decides instead.
MIN_N_FOR_SIMULATION = 24


def _key(test: str, n: int, params: dict[str, Any], seed: int, tail: str) -> str:
    payload = json.dumps({"t": test, "n": n, "p": params, "s": seed, "tail": tail}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


@lru_cache(maxsize=256)
def _cached(key: str, _builder: Any = None) -> tuple[float, float, float]:
    """The cache exists only to make repeated requests identical, not to hide the work."""
    raise RuntimeError("call through critical_values()")     # pragma: no cover


_QUANTILE_CACHE: dict[str, tuple[float, float, float]] = {}


def critical_values(
    test: str,
    statistic_fn: Callable[[np.random.Generator], float],
    n: int,
    *,
    params: dict[str, Any] | None = None,
    seed: int = 20260101,
    tail: str = "lower",
) -> tuple[float, float, float]:
    """
    Simulate the null distribution of `statistic_fn` and return the 1%, 5% and 10%
    critical values. Cached, so a repeat call is exact.

    `tail` matters more than it looks. Unit-root tests reject in the *lower* tail (a very
    negative t); break tests reject in the *upper* tail (a very large F). Returning the
    wrong end of the distribution produces a "critical value" that ordinary noise clears
    almost every time — which is exactly how a sup-Wald test came to report a break in
    white noise.
    """
    if tail not in ("lower", "upper"):                     # pragma: no cover - guard
        raise ValueError("tail must be 'lower' or 'upper'")
    params = params or {}
    key = _key(test, n, params, seed, tail)
    hit = _QUANTILE_CACHE.get(key)
    if hit is not None:
        return hit

    rng = np.random.default_rng(_seed_from(key, seed))
    stats = np.empty(N_REPS, dtype=float)
    for i in range(N_REPS):
        try:
            stats[i] = statistic_fn(rng)
        except Exception:                                  # pragma: no cover - degenerate draw
            stats[i] = np.nan
    stats = stats[np.isfinite(stats)]
    if stats.size < 50:                                    # pragma: no cover
        return (float("nan"), float("nan"), float("nan"))

    probs = (0.01, 0.05, 0.10) if tail == "lower" else (0.99, 0.95, 0.90)
    out = tuple(float(np.quantile(stats, q)) for q in probs)
    _QUANTILE_CACHE[key] = out
    return out


def p_value_from_simulation(samples_fn: Callable[[np.random.Generator], float], observed: float,
                            seed: int = 20260101, n_reps: int = N_REPS) -> float:
    """
    Share of simulated null statistics at least as extreme as `observed` (lower tail).

    Returned as a share, never as "0.000": with a few hundred replications the honest
    floor is 1/reps, and the caller prints "< 0.003" rather than a confident zero.
    Uses the upper tail, which is where break and heteroskedasticity tests reject.
    """
    rng = np.random.default_rng(seed)
    draws = np.array([samples_fn(rng) for _ in range(n_reps)], dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:                                    # pragma: no cover
        return float("nan")
    return float(max(1, int(np.sum(draws >= observed))) / draws.size)


def _seed_from(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{key}:{seed}".encode()).hexdigest()[:8]
    return int(digest, 16)


def caveat(reps: int = N_REPS, seed: int = 20260101) -> str:
    return (
        f"Critical values for this test were simulated here ({reps} replications, seed "
        f"{seed}), because no published table covers this exact sample size and period. "
        "They are reproducible: the same data gives the same numbers."
    )


__all__ = ["critical_values", "p_value_from_simulation", "caveat", "N_REPS", "MIN_N_FOR_SIMULATION"]
