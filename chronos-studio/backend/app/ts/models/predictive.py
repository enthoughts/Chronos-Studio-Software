"""
ts/models/predictive.py — the predictive distribution as a first-class object (§5.8 C).

A count, intermittent or event model does not produce "a number and a band". It produces a
*probability mass function* for each future period, and the band is only one thing you can read off
it. CRPS, the log score, PIT, the median, the probability of a stockout, the probability of no demand
at all — every one of those is a different question asked of the same PMF, and Part 15's metric
policy engine will switch to exactly these questions the moment a model declares `discrete_predictive`.

So the PMF is computed once, here, and everything downstream reads it:

* point forecast = the **mean** of the PMF (for a count model that is λ; for a hurdle it is
  P(y>0)·E[y|y>0], which is *not* the mean of either part),
* interval = the **quantiles** of the PMF, snapped to integers,
* CRPS/log score/PIT = functions of the same array.

The one design rule that matters: **a quantile is never interpolated between two integers.** Linear
interpolation of a count quantile returns 3.4, which is not a count, and an interval endpoint that
cannot occur is the kind of small wrongness that makes a planner distrust the whole panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: A PMF is truncated at the smallest support point whose cumulative mass reaches this. 1 − 1e-10
#: keeps the quantiles exact while stopping a heavy tail from allocating a million-cell array.
TAIL_MASS = 1e-10

#: Ceiling on support size, so a pathological rate cannot allocate gigabytes. Reaching it is recorded
#: in `truncated`, because a truncated PMF understates the tail and CRPS would inherit that.
MAX_SUPPORT = 20_000


@dataclass
class DiscretePredictive:
    """
    A per-step PMF summary that is small on the wire and complete for the metrics.

    `grid` and `pmf` are the full distribution (both length `k`), shipped so a probabilistic score can
    be recomputed without re-running the model. `quantiles` are cached for the interval, but they are
    derived — there is exactly one place that inverts the CDF.
    """

    #: (horizon, k) support values, identical on every row unless a distribution is unbounded above.
    grid: np.ndarray
    #: (horizon, k) probabilities, each row summing to 1.
    pmf: np.ndarray
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    #: True where the support had to be cut at MAX_SUPPORT, so the tail mass is not represented.
    truncated: np.ndarray
    family: str

    def step(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        """The (support, probabilities) pair for one forecast step, with the grid trimmed to live mass."""
        p = self.pmf[h]
        live = p > 0
        return self.grid[h][live], p[live]

    def log_score(self, h: int, observed: float) -> float | None:
        """
        −log P(Y = y) at the observed count.

        Outside the stored support the true penalty is −log(0) = +∞, which would poison an average. The
        honest finite stand-in is the penalty the model *would* have paid at the edge of its own grid
        plus the distance beyond it: it keeps the ordering between near-misses and absurd misses without
        turning one outlier into an infinite mean score. `None` is returned only for a non-integer
        observation, where "probability of exactly 3.4" is not a meaningful question.
        """
        if not np.isfinite(observed) or abs(observed - round(observed)) > 1e-9:
            return None
        grid, p = self.step(h)
        hit = np.flatnonzero(grid == observed)
        if hit.size:
            return float(-np.log(max(p[hit[0]], 1e-300)))
        # The comparison is with the mass at the nearest *edge* of the grid, not with the mode: an
        # observation past the tail is a miss by the same standard as one just outside it, only worse.
        edge = float(p[0]) if observed < float(grid[0]) else float(p[-1])
        beyond = abs(observed - (float(grid[0]) if observed < float(grid[0]) else float(grid[-1])))
        return float(-np.log(max(edge, 1e-300)) + beyond)

    def crps(self, h: int, observed: float) -> float:
        """
        CRPS for a discrete predictive distribution (the sum form, exact — not the Gaussian shortcut).

            CRPS(F, y) = Σ_k (F_k − 1{y ≤ x_k})²  over the support grid

        This is the *ranked probability score*, which is CRPS's discrete cousin and is what the
        literature means by CRPS on counts. Using the continuous formula on a count PMF silently
        reports a different quantity.
        """
        grid, p = self.step(h)
        lo, hi = float(grid[0]), float(grid[-1])
        if observed < lo or observed > hi:
            # Outside the stored support the score must still be finite and correctly ordered:
            # the CDF is 0 below the grid and 1 above it, so the sum has a closed form. Returning a
            # NaN here would quietly drop the worst observations from an average CRPS, which is the
            # opposite of what an evaluation battery is for.
            if observed < lo:
                return float(np.sum(np.cumsum(p) ** 2) + (lo - observed))
            return float(np.sum((np.cumsum(p) - 1.0) ** 2) + (observed - hi))
        cdf = np.cumsum(p)
        indicator = (observed <= grid).astype(float)
        return float(np.sum((cdf - indicator) ** 2))

    def pit(self, h: int, observed: float) -> float:
        """
        Probability integral transform, with the randomised convention for atoms.

        F(y−1) + U·P(Y=y) with U ~ U(0,1) — without the randomisation the PIT of a discrete model is a
        staircase and a uniformity test on it rejects for a *correct* model. The RNG is seeded from the
        step index so the value is reproducible.
        """
        grid, p = self.step(h)
        at_or_below = float(np.sum(p[grid <= observed]))
        at = float(np.sum(p[grid == observed]))
        # The draw is a deterministic function of (h, observed): a PIT that changes every time it is
        # computed cannot be plotted, cached, or compared between two runs of the same model.
        u = np.random.default_rng([20260101, int(h), int(np.floor(observed))]).random()
        if observed < float(grid[0]):
            return 0.0 + u * 0.0
        if observed > float(grid[-1]):
            return 1.0
        return float(min(max(at_or_below - at + u * at, 0.0), 1.0))

    def probability_of(self, h: int, *, at_least: float = 1.0) -> float:
        """P(Y ≥ at_least) — the stockout/at-least-one-demand question, asked directly of the PMF."""
        grid, p = self.step(h)
        return float(np.sum(p[grid >= at_least]))

    def as_dict(self, *, max_steps: int = 40) -> dict[str, Any]:
        """Wire form. The PMF can be large, so it is trimmed to `max_steps` rows and live mass only."""
        k = min(self.mean.size, max_steps)
        return {
            "family": self.family,
            "index": list(range(1, k + 1)),
            "mean": [float(x) for x in self.mean[:k]],
            "lower": [float(x) for x in self.lower[:k]],
            "upper": [float(x) for x in self.upper[:k]],
            "support": [[float(v) for v in self.grid[i]] for i in range(k)],
            "pmf": [[float(x) for x in self.pmf[i]] for i in range(k)],
            "truncated": [bool(x) for x in self.truncated[:k]],
            "p_at_least_one": [self.probability_of(i, at_least=1.0) for i in range(k)],
            # What Part 15's metric policy reads to decide "cluster" vs "density": the PMF is only
            # ever compared with other PMFs, never with a Gaussian band.
            "discrete": True,
            "support_kind": "integer",
        }


def poisson_pmf(lam: float, *, max_support: int | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
    """Support, PMF and truncation flag for a Poisson rate, sized to the requested tail mass."""
    lam = float(max(lam, 0.0))
    cap = int(max_support or MAX_SUPPORT)
    hi = int(stats.poisson.ppf(1 - TAIL_MASS, lam)) if lam > 0 else 0
    if hi + 1 > cap:
        hi = cap - 1
        truncated = True
    else:
        truncated = False
    grid = np.arange(hi + 1, dtype=float)
    pmf = stats.poisson.pmf(grid, lam)
    total = pmf.sum()
    return grid, (pmf / total if total > 0 else pmf), truncated


def negbin_pmf(mu: float, alpha: float) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    NB2 with variance μ + αμ², parameterised the way the model estimates it.

    statsmodels' `NegativeBinomial` uses the same α, so the PMF here lines up with the GLM's fitted
    values rather than introducing a second convention.
    """
    mu = float(max(mu, 0.0))
    alpha = float(max(alpha, 1e-9))
    if mu <= 0:
        return np.array([0.0]), np.array([1.0]), False
    size = 1.0 / alpha
    prob = size / (size + mu)
    hi = int(stats.nbinom.ppf(1 - TAIL_MASS, size, prob))
    if hi + 1 > MAX_SUPPORT:
        hi = MAX_SUPPORT - 1
        truncated = True
    else:
        truncated = False
    grid = np.arange(hi + 1, dtype=float)
    pmf = stats.nbinom.pmf(grid, size, prob)
    total = pmf.sum()
    return grid, (pmf / total if total > 0 else pmf), truncated


def mixture_pmf(components: list[tuple[np.ndarray, np.ndarray, float]]) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    A weighted mixture of discrete PMFs on a common grid (used by the hurdle and by INGARCH).

    Each component is `(grid, pmf, weight)`. The union grid runs to the largest component's support;
    a component with no mass beyond that point contributes nothing, so the truncation flag is the OR
    of its parts.
    """
    total_w = sum(w for _g, _p, w in components)
    if total_w <= 0:
        return np.array([0.0]), np.array([1.0]), False
    top = int(max(g[-1] for g, _p, _w in components))
    grid = np.arange(top + 1, dtype=float)
    out = np.zeros(grid.size)
    for g, p, w in components:
        out[: g.size] += (w / total_w) * p
    s = out.sum()
    if s <= 0:
        return np.array([0.0]), np.array([1.0]), False
    return grid, out / s, False


def summarise(
    rows: list[tuple[np.ndarray, np.ndarray, bool]], *, alpha: float, family: str
) -> DiscretePredictive:
    """
    Stack per-step PMFs into one object, inverting the CDF exactly once.

    `lower`/`upper` are the smallest support points whose CDF reaches α/2 and 1−α/2. That is a genuine
    quantile of a discrete distribution — an interval that *contains* at least (1−α) of the mass, in
    the conservative sense a planner wants — rather than an interpolated position between two
    impossible values.
    """
    horizon = len(rows)
    width = max(g.size for g, _p, _t in rows)
    grid = np.zeros((horizon, width))
    pmf = np.zeros((horizon, width))
    trunc = np.zeros(horizon, dtype=bool)
    mean = np.zeros(horizon)
    lower = np.zeros(horizon)
    upper = np.zeros(horizon)
    for i, (g, p, t) in enumerate(rows):
        grid[i, : g.size] = g
        pmf[i, : p.size] = p
        # Pad the grid past the support with values that cannot occur, so a padded cell is never
        # chosen as a quantile.
        if g.size < width:
            grid[i, g.size :] = g[-1] + np.arange(1, width - g.size + 1)
        trunc[i] = t
        mean[i] = float(np.sum(g * p))
        cdf = np.cumsum(p)
        lower[i] = float(g[int(np.searchsorted(cdf, alpha / 2))])
        upper[i] = float(g[int(np.searchsorted(cdf, 1 - alpha / 2))])
        lower[i] = min(lower[i], np.floor(mean[i]))
        upper[i] = max(upper[i], np.ceil(mean[i]))
    return DiscretePredictive(
        grid=grid, pmf=pmf, mean=mean, lower=lower, upper=upper, truncated=trunc, family=family
    )


def frame(pred: DiscretePredictive) -> pd.DataFrame:
    """
    The `mean`/`lower`/`upper` frame every `ModelWrapper._forecast` must return.

    Kept as a function so the three columns are built in one place: the *only* alternative is each
    model assembling them itself, and that is how one of them ends up returning a normal interval.
    """
    steps = np.arange(1, pred.mean.size + 1)
    return pd.DataFrame(
        {"mean": pred.mean, "lower": pred.lower, "upper": pred.upper}, index=steps
    )


__all__ = [
    "DiscretePredictive",
    "MAX_SUPPORT",
    "TAIL_MASS",
    "frame",
    "mixture_pmf",
    "negbin_pmf",
    "poisson_pmf",
    "summarise",
]
