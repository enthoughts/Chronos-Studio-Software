"""
ts/eda/stl_cache.py — one robust STL fit shared by the EDA seasonality and
outlier passes (Part 19 performance hardening).

`seasonal.stl_strength` and `outliers._decompose` both fit the same robust STL on
the same series for a weekly period (seasonal smoother 7). On 10,000 rows that
duplicate fit cost ~0.23 s. An LRU cache keyed by the exact float64 values,
period and seasonal window makes the second call a lookup. The cache is small,
immutable, and never keyed on a mutable object:

  * keys are the raw bytes of the interpolated values plus the STL settings;
  * values are copies of the fitted numpy component arrays;
  * callers wrap them in a Series with their own index, so no cached object is
    mutated;
  * series containing unrecoverable NaNs bypass the cache and raise as before.

This changes no statistic. It only avoids computing an identical decomposition
twice in one request.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


@lru_cache(maxsize=8)
def _fit_arrays(
    values: bytes,
    period: int,
    seasonal: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.frombuffer(values, dtype=np.float64).copy()
    fit = STL(pd.Series(y), period=period, robust=True, seasonal=seasonal).fit()
    return (
        np.asarray(fit.trend, dtype=float).copy(),
        np.asarray(fit.seasonal, dtype=float).copy(),
        np.asarray(fit.resid, dtype=float).copy(),
    )


def robust_stl_components(
    values: pd.Series,
    period: int,
    seasonal: int = 7,
) -> dict[str, pd.Series]:
    """Return trend / seasonal / residual Series on the caller's index.

    Missing values are interpolated for the fit exactly as both legacy callers
    did; the returned residual therefore covers every position, not just the
    observed ones. Any failure is propagated so the caller can use its existing
    rolling-median fallback.
    """
    s = values.astype(float).interpolate(limit_direction="both")
    if s.isna().any() or len(s) < 2 * period + 2:
        raise ValueError("not enough finite observations for a robust STL fit")
    key = np.ascontiguousarray(s.to_numpy(dtype=np.float64)).tobytes()
    trend, seas, resid = _fit_arrays(key, period, seasonal)
    return {
        "trend": pd.Series(trend.copy(), index=s.index),
        "seasonal": pd.Series(seas.copy(), index=s.index),
        "resid": pd.Series(resid.copy(), index=s.index),
    }


def stl_cache_info():
    """Exposed for the exactness / reuse regression test."""
    return _fit_arrays.cache_info()


def stl_cache_clear():
    """Exposed for tests that need a deterministic cache-hit count."""
    _fit_arrays.cache_clear()


__all__ = ["robust_stl_components", "stl_cache_info", "stl_cache_clear"]
