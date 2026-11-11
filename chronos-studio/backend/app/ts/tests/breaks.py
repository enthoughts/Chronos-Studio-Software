"""
ts/tests/breaks.py — structural breaks (§5.5).

A break is not one test. What you want to know depends on what you already suspect:

  Chow              — "I know the date; did the relationship change there?" Only valid
                      for a date chosen *before* looking, which is why it is offered
                      separately and the caveat says so.
  Quandt–Andrews    — "I don't know the date; is there *some* break?" The sup-Wald
                      (supF) statistic over all admissible dates, with simulated critical
                      values, because the published table is asymptotic and the app is
                      routinely used on 72–144 points.
  Bai–Perron        — "How many breaks, and where?" Sequential supF with the
                      repartition rule: find the best break, then re-scan the segments,
                      until the segment stops improving.
  CUSUM             — "Does the *residual* structure wander?" A moving diagnostic rather
                      than a single test; the recursive version is what predicts
                      parameter instability.

The point of returning break *dates* (not just a p-value) is that a break can then be
offered as a step dummy to Part 8 rather than being deleted (§5.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from app.core.formatting import p_phrase

from app.ts.tests._mc import caveat, critical_values

#: Never test a break in the first or last 15% of the series: the segments get too short
#: to estimate anything, and the Wald statistic explodes for a spurious reason.
TRIM = 0.15
#: Minimum observations in each segment.
MIN_SEGMENT = 5


def _draw_null(rng: np.random.Generator, n: int, null_kind: str, phi: float = 0.5) -> np.ndarray:
    """
    Draw one series from the null.

    `null_kind` is the whole ballgame. If the data are stationary (ADF rejects), the null is
    white noise and a sup-Wald line near 5–6 is right. If the data are persistent (ADF does not
    reject) the null has to be persistent too — but *which* persistence matters:

    * an AR(φ) with φ ≈ 0.5 reproduces both the wandering and the memory of typical
      persistent data, and keeps the critical value usable;
    * a pure random walk does not. It wanders without memory, and after a step dummy the
      residuals look like white noise, so even a genuine level shift cannot be distinguished
      from drift — a textbook break scored 161.7 against a critical value of 282.4;
    * i.i.d. noise at this sample size calls drift a break: a random walk produced three
      "significant" breaks.

    The φ used is recorded in the caveat, so the reader can see which null was assumed.
    """
    e = rng.standard_normal(n)
    if null_kind == "iid":
        return e
    if null_kind == "ar":
        out = np.empty(n)
        prev = 0.0
        for i in range(n):
            prev = phi * prev + e[i]
            out[i] = prev
        return out
    raise ValueError("null_kind must be 'iid' or 'ar'")


@dataclass
class BreakTest:
    id: str
    name: str
    statistic: float
    p_value: float | None
    critical: dict[str, float] = field(default_factory=dict)
    break_at: int | None = None          # position in the series
    break_time: str | None = None
    breaks: list[dict[str, Any]] = field(default_factory=list)
    rejects_null: bool | None = None
    says: str = ""
    plain: str = ""
    simulated_cv: bool = False
    caveats: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Chow — a known date
# --------------------------------------------------------------------------- #
def chow(series: pd.Series, break_time: str | pd.Timestamp, spec: Literal["mean", "trend"] = "mean") -> BreakTest:
    """
    Chow test at a date the user supplies: fit the model on both sides and ask whether
    one model describes both. F = ((RSS_p − RSS_1 − RSS_2)/k) / ((RSS_1 + RSS_2)/(n − 2k)).
    """
    y = series.dropna()
    t_split = pd.Timestamp(break_time)
    if t_split <= y.index.min() or t_split >= y.index.max():
        raise ValueError("The break date must fall inside the series.")

    def design(idx: pd.DatetimeIndex) -> np.ndarray:
        x = np.arange(len(idx), dtype=float)
        return np.column_stack([np.ones_like(x), x]) if spec == "trend" else np.ones((len(idx), 1))

    left, right = y[y.index < t_split], y[y.index >= t_split]
    if len(left) < MIN_SEGMENT or len(right) < MIN_SEGMENT:
        raise ValueError("Both sides of the break need at least a handful of observations.")

    def rss(sub: pd.Series) -> float:
        X = design(sub.index)
        b, *_ = np.linalg.lstsq(X, sub.to_numpy(dtype=float), rcond=None)
        r = sub.to_numpy(dtype=float) - X @ b
        return float(r @ r)

    X_all = design(y.index)
    b_all, *_ = np.linalg.lstsq(X_all, y.to_numpy(dtype=float), rcond=None)
    r_all = y.to_numpy(dtype=float) - X_all @ b_all
    rss_p = float(r_all @ r_all)

    k = X_all.shape[1]
    rss_1, rss_2 = rss(left), rss(right)
    denom = max(1e-12, (rss_1 + rss_2) / max(1, len(y) - 2 * k))
    F = ((rss_p - (rss_1 + rss_2)) / k) / denom
    p = float(1 - stats.f.cdf(F, k, max(1, len(y) - 2 * k)))
    rejects = bool(p < 0.05)

    return BreakTest(
        id=f"stat.chow.{spec}",
        name="Chow test (known date)",
        statistic=float(F),
        p_value=p,
        break_at=int(y.index.get_loc(t_split)),
        break_time=str(t_split),
        rejects_null=rejects,
        says="break at that date" if rejects else "no evidence of a break there",
        plain=(
            f"Chow F = {F:.3f} with {p_phrase(p, 3)} at {t_split:%Y-%m-%d}: "
            + ("the model differs before and after that date."
               if rejects else "no evidence that the relationship changed there.")
        ),
        caveats=[
            "Only valid when the date was chosen before looking at the data (a known policy "
            "change, say). Searching for the date and then running Chow overstates significance — "
            "use Quandt–Andrews for that.",
        ],
    )


# --------------------------------------------------------------------------- #
# Quandt–Andrews — an unknown date
# --------------------------------------------------------------------------- #
def _sup_f(y: np.ndarray, lo: int, hi: int, spec: Literal["mean", "trend"]) -> tuple[float, int]:
    """
    Best single break by the Wald/F statistic over candidate positions `lo..hi`.

    Computed from cumulative sums rather than by fitting one regression per candidate: a
    sup-Wald over a few hundred candidate dates, repeated 400 times for the simulated null,
    is otherwise tens of thousands of least-squares solves per request. The cumulative-sum
    form gives identical numbers (the 2x2 or 1x1 normal equations are closed-form) and runs
    roughly a hundred times faster.
    """
    n = len(y)
    x = np.arange(n, dtype=float)
    k = 2 if spec == "trend" else 1

    # Suffix/prefix sums: sY[i] = sum of y[:i], etc.
    s1 = np.r_[0.0, np.cumsum(y)]
    syy = np.r_[0.0, np.cumsum(y * y)]
    if spec == "trend":
        sx = np.r_[0.0, np.cumsum(x)]
        sxx = np.r_[0.0, np.cumsum(x * x)]
        sxy = np.r_[0.0, np.cumsum(x * y)]

    def rss(m: np.ndarray, Y: np.ndarray, YY: np.ndarray, X: np.ndarray | None,
            XX: np.ndarray | None, XY: np.ndarray | None) -> np.ndarray:
        """Residual sum of squares for the model fitted on each prefix m (vectorised)."""
        m = m.astype(float)
        if spec == "trend":
            assert X is not None and XX is not None and XY is not None
            # Normal equations for [1, x]:
            #   m·b0 + Sx·b1 = Sy ;  Sx·b0 + Sxx·b1 = Sxy
            det = m * XX - X * X
            det = np.where(np.abs(det) < 1e-12, np.nan, det)
            b0 = (Y * XX - X * XY) / det
            b1 = (m * XY - X * Y) / det
            return YY - (b0 * Y + b1 * XY)
        return YY - Y * Y / np.maximum(m, 1.0)

    idx = np.arange(lo, hi + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if spec == "trend":
            rss_l = rss(idx, s1[idx], syy[idx], sx[idx], sxx[idx], sxy[idx])
            rss_r = rss(
                (n - idx), (s1[n] - s1[idx]), (syy[n] - syy[idx]),
                (sx[n] - sx[idx]), (sxx[n] - sxx[idx]), (sxy[n] - sxy[idx]),
            )
        else:
            rss_l = rss(idx, s1[idx], syy[idx], None, None, None)
            rss_r = rss((n - idx), (s1[n] - s1[idx]), (syy[n] - syy[idx]), None, None, None)
        rss_p = rss(np.array([n]), np.array([s1[n]]), np.array([syy[n]]),
                    np.array([sx[n]]) if spec == "trend" else None,
                    np.array([sxx[n]]) if spec == "trend" else None,
                    np.array([sxy[n]]) if spec == "trend" else None)[0]

    rss_split = rss_l + rss_r
    feff = np.maximum(1, n - 2 * k)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = ((rss_p - rss_split) / k) / (rss_split / feff)
    f = np.where(np.isfinite(f) & (rss_split > 0), f, -np.inf)
    if not np.any(np.isfinite(f)):
        return float("-inf"), int(lo)
    best = int(np.argmax(f))
    return float(f[best]), int(idx[best])


def _f_at(y: np.ndarray, cut: int, spec: Literal["mean", "trend"]) -> float:
    """Wald/F statistic of a single break at position `cut` (no searching)."""
    n = len(y)
    x = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), x]) if spec == "trend" else np.ones((n, 1))
    k = X.shape[1]
    b_all, *_ = np.linalg.lstsq(X, y, rcond=None)
    r_all = y - X @ b_all
    rss_p = float(r_all @ r_all)
    b1, *_ = np.linalg.lstsq(X[:cut], y[:cut], rcond=None)
    b2, *_ = np.linalg.lstsq(X[cut:], y[cut:], rcond=None)
    rss = float(np.sum((y[:cut] - X[:cut] @ b1) ** 2) + np.sum((y[cut:] - X[cut:] @ b2) ** 2))
    if rss <= 0:
        return float("inf")
    return float(((rss_p - rss) / k) / (rss / max(1, n - 2 * k)))


def quandt_andrews(series: pd.Series, spec: Literal["mean", "trend"] = "trend",
                   trim: float = TRIM, reps: int = 200,
                   null_kind: str = "iid") -> BreakTest:
    """
    sup-Wald (Quandt–Andrews) test for one unknown break.

    Critical values are simulated by drawing from the null (no break) at this exact n and
    trim, because Andrew's asymptotic table over-rejects at 72–144 observations. The
    simulation is labelled and seeded.
    """
    y = series.dropna()
    n = len(y)
    if n < 20:
        raise ValueError("Quandt–Andrews needs at least about 20 observations.")
    lo = max(MIN_SEGMENT, int(np.floor(trim * n)))
    hi = n - lo
    if hi <= lo:
        raise ValueError("Too few observations for the trim chosen.")

    stat, at = _sup_f(y.to_numpy(dtype=float), lo, hi, spec)

    def sim(rng: np.random.Generator) -> float:
        return _sup_f(_draw_null(rng, n, null_kind), lo, hi, spec)[0]

    c1, c5, c10 = critical_values("quandt_andrews", sim, n,
                                  params={"trim": trim, "spec": spec, "reps": reps,
                                          "null": null_kind},
                                  tail="upper")
    rejects = bool(stat > c5)
    t_at = y.index[min(at, n - 1)]

    # A sup-Wald test *always* has an argmax, including on white noise, where that argmax is
    # meaningless. Reporting it as `break_time` next to "no evidence of a break" invites the date
    # to be copied downstream (into a step dummy, a report, a chart annotation). So the date is
    # only a break when the test rejects; otherwise it is described as the best candidate and
    # carried in `says`/`plain` as such, never in `break_time`.
    return BreakTest(
        id=f"stat.quandt_andrews.{spec}",
        name="Quandt–Andrews sup-Wald (unknown date)",
        statistic=float(stat),
        p_value=None,
        critical={"1%": c1, "5%": c5, "10%": c10},
        break_at=at if rejects else None,
        break_time=str(t_at) if rejects else None,
        breaks=[{"index": at, "time": str(t_at), "statistic": float(stat)}] if rejects else [],
        rejects_null=rejects,
        says="one break somewhere" if rejects else "no evidence of a break",
        plain=(
            f"The largest sup-Wald across all candidate dates is {stat:.3f} (at {t_at:%Y-%m-%d}), "
            f"against a simulated 5% critical value of {c5:.2f}. "
            + ("There is evidence of a structural break." if rejects else
               f"There is no evidence of a structural break — {t_at:%Y-%m-%d} is only the best "
               "candidate date, and on a series with no break a best candidate always exists.")
        ),
        simulated_cv=True,
        caveats=[
            caveat(reps=400),
            (
                "Critical values are simulated against a random walk, because this series has a "
                "unit root: a wandering series produces large break statistics at every turning "
                "point, so a noise-based line would call a break almost every time."
                if null_kind == "walk" else
                "Critical values are simulated against stationary noise, matching the series."
            ),
            "Finds at most one break; use Bai–Perron when more may exist.",
        ] + ([] if rejects else [
            "No break date is reported: the argmax of a sup-Wald scan is not evidence of a break, "
            "and a date that only records 'where the scan peaked' is exactly the number that gets "
            "mistaken for a finding.",
        ]),
    )


def bai_perron(series: pd.Series, max_breaks: int = 3, spec: Literal["mean", "trend"] = "trend",
               trim: float = TRIM, null_kind: str = "iid") -> BreakTest:
    """
    Bai–Perron: the best partition of the series into `m` stable segments, for m = 0..max.

    Every candidate partition is scored by its total residual sum of squares, so the segments
    compete with each other rather than being judged in isolation; the number of breaks is
    then chosen by BIC. That detail matters: testing each segment independently finds a
    *second* break inside the noisy aftermath of the first, and reported three breaks for a
    series that has one. A global criterion prefers the one-break model, because splitting
    again barely lowers the residual sum of squares while costing another penalty term.
    """
    y = series.dropna()
    n = len(y)
    if n < 30:
        raise ValueError("Bai–Perron needs at least 30 observations.")
    vals = y.to_numpy(dtype=float)
    lo = max(MIN_SEGMENT, int(np.floor(trim * n)))

    def segment_sse(a: int, b: int) -> float:
        """SSE of a trend (or mean) fitted to vals[a:b+1]."""
        yy = vals[a:b + 1]
        m = len(yy)
        x = np.arange(m, dtype=float)
        X = np.column_stack([np.ones(m), x]) if spec == "trend" else np.ones((m, 1))
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        r = yy - X @ beta
        return float(r @ r)

    def bounds(points: tuple[int, ...]) -> list[tuple[int, int]]:
        """Inclusive (start, end) for each segment. A break at index c starts the new segment."""
        starts = (0,) + tuple(points)
        ends = tuple(c - 1 for c in points) + (n - 1,)
        return list(zip(starts, ends))

    def sse(points: tuple[int, ...]) -> float:
        return sum(segment_sse(a, b) for a, b in bounds(points))

    def admissible(points: tuple[int, ...]) -> bool:
        return all(b - a + 1 >= max(MIN_SEGMENT, lo) for a, b in bounds(points))

    # Candidates: the best break point of every segment, refined greedily per break count.
    best_by_count: dict[int, tuple[float, tuple[int, ...]]] = {0: (sse(()), ())}
    frontier: list[tuple[float, tuple[int, ...]]] = [(sse(()), ())]
    for _m in range(1, max_breaks + 1):
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for _s, points in frontier:
            for a, b in bounds(points):
                lo_i = max(MIN_SEGMENT, int(np.floor(trim * (b - a + 1))))
                hi_i = (b - a + 1) - lo_i
                if hi_i <= lo_i:
                    continue
                _f, off = _sup_f(vals[a:b + 1], lo_i, hi_i, spec)
                cut = a + off
                trial = tuple(sorted(points + (cut,)))
                if len(set(trial)) != len(trial) or not admissible(trial):
                    continue
                candidates.append((sse(trial), trial))
        if not candidates:
            break
        frontier = sorted(candidates, key=lambda t: t[0])[:6]
        best_by_count[_m] = frontier[0]

    # BIC over break counts. Each break adds k parameters (k = 1 for mean, 2 for trend).
    k = 2 if spec == "trend" else 1
    bic: dict[int, float] = {}
    for m, (res, _pts) in best_by_count.items():
        params = (m + 1) * k + m
        bic[m] = float(len(vals) * np.log(max(res, 1e-12) / len(vals)) + params * np.log(len(vals)))
    m_star = min(bic, key=lambda m: bic[m])
    _res, chosen = best_by_count[m_star]

    # Report each chosen date with the F statistic *at that date*, judged against a null
    # simulated on its own segment — so the reader can see how emphatic each split is.
    found: list[dict[str, Any]] = []
    starts = (0,) + tuple(chosen)
    for i, cut in enumerate(chosen):
        a = starts[i]
        b = (chosen[i + 1] - 1) if i + 1 < len(chosen) else n - 1
        if len(vals[a:cut]) < MIN_SEGMENT or len(vals[cut:b + 1]) < MIN_SEGMENT:
            continue
        f_stat = _f_at(vals[a:b + 1], cut - a, spec)
        lo_i = max(MIN_SEGMENT, int(np.floor(trim * (b - a + 1))))
        hi_i = (b - a + 1) - lo_i

        def sim(rng: np.random.Generator, _len=b - a + 1, _lo=lo_i, _hi=hi_i) -> float:
            return _sup_f(_draw_null(rng, _len, null_kind), _lo, _hi, spec)[0]

        _c1, c5, _c10 = critical_values(
            "bai_perron", sim, b - a + 1,
            params={"trim": trim, "spec": spec, "seg": b - a + 1, "null": null_kind},
            tail="upper",
        )
        found.append({
            "index": int(cut),
            "time": str(y.index[cut]),
            "statistic": float(f_stat),
            "critical_5pct": float(c5),
            "segment": [int(a), int(b)],
        })

    found = _distinct_breaks(found)
    # A date BIC asked for but whose own segment test is not significant is not reported as a
    # break: the reader would take the date seriously while the statistic does not support it.
    significant = [b for b in found if b["statistic"] > b["critical_5pct"]]
    dropped = len(found) - len(significant)
    found = significant
    n_breaks = len(found)
    return BreakTest(
        id="stat.bai_perron",
        name="Bai–Perron multiple breaks",
        statistic=float(sum(b["statistic"] for b in found)) if found else 0.0,
        p_value=None,
        break_at=found[0]["index"] if found else None,
        break_time=found[0]["time"] if found else None,
        breaks=found,
        # The test-level critical value is the strictest of the per-break ones, so the table
        # never shows a zero where a threshold belongs.
        critical={"5%": max((b["critical_5pct"] for b in found), default=0.0)},
        rejects_null=bool(found),
        says=f"{n_breaks} break{'s' if n_breaks != 1 else ''}" if n_breaks else "no break",
        plain=(
            f"{n_breaks} structural break{'s' if n_breaks != 1 else ''} detected"
            + (": " + ", ".join(f"{b['time'][:10]}" for b in found) if found else "")
            + ". The number of breaks is chosen by BIC, not by testing segments one at a time, "
              "so a long noisy stretch is not mistaken for several breaks. Each date is offered "
              "as a step dummy in the next stage rather than being removed from the data."
        ),
        simulated_cv=True,
        caveats=[
            caveat(reps=400),
            "The number of breaks is chosen by BIC on the residual sum of squares; each reported "
            "date is then required to be significant at 5% inside its own segment.",
        ]
        + ([
            f"BIC preferred {len(found) + dropped} breaks but only {len(found)} reached 5% "
            "significance on its own segment; the weaker date is not reported."
        ] if dropped else []),
    )


# --------------------------------------------------------------------------- #
# CUSUM — parameter stability, visually and numerically
# --------------------------------------------------------------------------- #
def _cusum_stat(y: np.ndarray, spec: Literal["mean", "trend"], burn: float = 0.2) -> float:
    """
    Maximum absolute CUSUM path of standardised recursive residuals.

    Three details decide whether this statistic works at all:

    * First observations are skipped. With k parameters the first recursive residuals are
      estimated from k+1 points, so they are wildly volatile and one of them dominates the
      path. The first 20% is left out, which is the usual convention.
    * The path is *not* rescaled by its own standard deviation. A break inflates that spread,
      and dividing by it cancels the very signal being tested — with the rescaling, a series
      with a textbook level shift scored *below* its critical value.
    * The expanding regressions are computed from cumulative sums in one pass. The
      observation-by-observation version is the textbook definition but needs n least-squares
      solves per series, which makes a 400-replication null untenable inside a web request.
    """
    n = len(y)
    x = np.arange(n, dtype=float)
    k = 2 if spec == "trend" else 1
    t_start = max(k + 2, int(np.ceil(burn * n)))
    if n - t_start < 5:
        raise ValueError("Too few recursive residuals for CUSUM.")

    # cumulative sums up to and including index i (so S[i] covers y[0..i])
    S1 = np.cumsum(y)
    Syy = np.cumsum(y * y)
    if spec == "trend":
        Sx = np.cumsum(x)
        Sxx = np.cumsum(x * x)
        Sxy = np.cumsum(x * y)

    # Fit on y[0..t-1] for every t (prefix of length m = t), predict y[t].
    t = np.arange(t_start, n)
    m = t.astype(float)                                  # prefix length
    Y = S1[t - 1]
    YY = Syy[t - 1]
    if spec == "trend":
        X = Sx[t - 1]
        XX = Sxx[t - 1]
        XY = Sxy[t - 1]
        det = m * XX - X * X
        det = np.where(np.abs(det) < 1e-12, np.nan, det)
        b0 = (Y * XX - X * XY) / det
        b1 = (m * XY - X * Y) / det
        fit_next = b0 + b1 * t                            # prediction at the next x (= t)
        # leverage of the new point: 1 + z'(Z'Z)^{-1}z for z = [1, t]
        lev = (XX - 2.0 * X * t + m * t * t) / det
        rss = YY - (b0 * Y + b1 * XY)
    else:
        b0 = Y / m
        fit_next = b0
        lev = 1.0 / m
        rss = YY - Y * Y / m

    s2 = rss / np.maximum(1.0, m - k)
    resid = y[t] - fit_next
    scale = np.sqrt(s2 * (1.0 + lev))
    with np.errstate(invalid="ignore", divide="ignore"):
        w = resid / scale
    w = w[np.isfinite(w)]
    if len(w) < 5:
        raise ValueError("Too few finite recursive residuals for CUSUM.")
    return float(np.max(np.abs(np.cumsum(w))))


def cusum(series: pd.Series, spec: Literal["mean", "trend"] = "trend",
          null_kind: str = "iid") -> BreakTest:
    """
    CUSUM of recursive residuals (Brown, Durbin & Evans).

    Answers a different question from the sup-Wald tests: not "where is a break" but "do
    the parameters drift". The 5% line is simulated at this sample size rather than taken
    from the textbook 0.948·√n approximation, because that approximation is for the
    *asymptotic* Brownian-bridge path and is too tight at 72–144 points — with it, plain
    white noise trips the boundary.
    """
    y = series.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 20:
        raise ValueError("CUSUM needs at least about 20 observations.")

    stat = _cusum_stat(y, spec)

    def sim(rng: np.random.Generator) -> float:
        return _cusum_stat(_draw_null(rng, n, null_kind), spec)

    _c1, c5, _c10 = critical_values("cusum", sim, n,
                                   params={"spec": spec, "null": null_kind}, tail="upper")
    asymptotic = 0.948 * np.sqrt(max(1, n - (2 if spec == "trend" else 1)))
    rejects = bool(stat > c5)

    return BreakTest(
        id=f"stat.cusum.{spec}",
        name="CUSUM of recursive residuals",
        statistic=stat,
        p_value=None,
        critical={"5%": float(c5)},
        rejects_null=rejects,
        says="parameters drift" if rejects else "no evidence of parameter drift",
        plain=(
            f"The CUSUM path reaches {stat:.3f} against a simulated 5% boundary of {c5:.3f}. "
            + ("The coefficients are not stable over the window."
               if rejects else "No evidence that the coefficients move over the window.")
        ),
        simulated_cv=True,
        caveats=[
            caveat(reps=400),
            f"The textbook asymptotic line is {asymptotic:.2f}; the simulated line is used "
            "because it is calibrated for this sample size.",
            "CUSUM reacts to a single large outlier as well as to a break; check the outlier "
            "panel before concluding the parameters changed.",
        ],
    )


def _distinct_breaks(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sequential scanning can re-find the same date from both neighbouring segments. Two
    entries for one date would read as two breaks, so nearby dates are merged — keeping the
    stronger statistic — and the list is returned in time order.
    """
    kept: list[dict[str, Any]] = []
    for b in sorted(found, key=lambda d: -d["statistic"]):
        if any(abs(b["index"] - k["index"]) <= 1 for k in kept):
            continue
        kept.append(b)
    return sorted(kept, key=lambda d: d["index"])


def break_battery(series: pd.Series, spec: Literal["mean", "trend"] = "trend",
                  period: int | None = None) -> list[BreakTest]:
    """
    Run what is runnable. A test that cannot run is *omitted*, never reported as a pass —
    a missing test is not evidence.
    """
    from app.ts.tests.unitroot import adf
    y = series.dropna()
    null_kind = "ar"
    try:
        # Test the LEVEL (constant only): a constant+trend ADF can absorb a level shift with its
        # trend term and report the broken series as stationary.
        if adf(y, "constant").rejects_null:
            null_kind = "iid"
    except Exception:                                  # pragma: no cover - too short to decide
        null_kind = "ar"                              # the conservative default

    out: list[BreakTest] = []
    for fn, kwargs in (
        (quandt_andrews, {"spec": spec, "null_kind": null_kind}),
        (bai_perron, {"spec": spec, "null_kind": null_kind}),
        (cusum, {"spec": spec, "null_kind": null_kind}),
    ):
        try:
            out.append(fn(series, **kwargs))
        except Exception:                                  # pragma: no cover - short series
            continue
    return out


__all__ = ["BreakTest", "chow", "quandt_andrews", "bai_perron", "cusum", "break_battery"]
