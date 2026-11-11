"""
ts/eda/outliers.py — find anomalies and, crucially, *label their type* (§5.2).

"Outlier" is not one thing, and the distinction matters for what you do next:

    spike            — one observation away from its neighbours; usually a typo or a
                       one-off shock. A pulse dummy or a careful correction.
    level shift      — the series steps to a new level and stays there. Never delete
                       these: they are information. Model with a step dummy.
    temporary change — an excursion that decays back over a few periods.
    isolated_run     — several consecutive extreme values.

We do **not** claim to identify "seasonal outliers" as a separate type. A single
period that is extreme is a spike; what we add instead is the separate flag
`unusual_for_season`, computed by comparing it with the same calendar position in
other years. That is honest about what one series can tell you, and it still gives
Part 3 what it needs to offer a season-aware treatment.

We detect on the **STL remainder** where possible (so seasonality and trend do not
masquerade as anomalies), falling back to a robust global MAD rule otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from app.ts.eda.stl_cache import robust_stl_components

OutlierType = Literal["spike", "level_shift", "temporary_change", "isolated_run"]


# How far a value must sit from its calendar peers (in units of that season's own SD)
# before we call it a *seasonal* anomaly rather than a plain spike. A spike whose
# seasonal z is exactly this large is still a spike; a value that is wild for its
# month while its neighbours are normal is what this label is for.
SEASONAL_Z = 3.0

# Never present more than this share of a series as anomalies. A tool that flags 1
# in 10 observations has taught the user nothing except to ignore the panel.
MAX_FLAG_RATE = 0.05

# The remainder's scale may not fall below this share of the series' own robust SD,
# however much a robust STL fit would like it to.
FLOOR_SCALE_SHARE = 0.75

# An anomaly must span at least this many recording steps (see `_resolution`) — a
# series stored to two decimals should not have its second decimal called an outlier.
MIN_EFFECT_STEPS = 3.0


@dataclass
class Outlier:
    time: str
    value: float
    expected: float
    remainder: float
    robust_z: float
    direction: Literal["above", "below"]
    type: OutlierType
    unusual_for_season: bool
    plain: str


def _calibrate_scale(remainder_scale: float, values: pd.Series) -> float:
    """
    Do not let a robust fit's own shrinkage become the outlier threshold.

    The rule of thumb: the spread of the remainder should not be much smaller than the
    spread of the series it came from (differencing/seasonal adjustment can *add*
    noise, and the robust STL fit can remove some, but a ratio far below the usual
    ~0.75 for a seasonal series means the remainder has been over-trimmed).
    """
    v = values.dropna().to_numpy(dtype=float)
    if v.size < 12:
        return remainder_scale
    spread = float(np.subtract(*np.percentile(v, [75, 25]))) / 1.349  # IQR → SD
    if not np.isfinite(spread) or spread <= 0:
        return remainder_scale
    ratio = remainder_scale / spread
    if ratio >= FLOOR_SCALE_SHARE:
        return remainder_scale
    # a noisy series legitimately has a wide remainder relative to its IQR-based spread
    # (weekly seasonality inside a monthly calendar, for instance), so the floor is a
    # gentle correction, never a replacement
    return min(spread, remainder_scale * 1.35)


def _tail_outliers(sorted_abs_z: pd.Series, alpha: float = 0.01) -> list[Any]:
    """
    Which of the most extreme points are far out *for their rank*.

    For n roughly-Gaussian points the expected magnitude of the k-th largest |z| grows
    with k; comparing each candidate with that expectation (with a generous factor)
    stops the ordinary tail of a short series from being reported as anomalies, while
    a genuine 10-sigma spike is never in doubt.
    """
    n = len(sorted_abs_z)
    if n == 0:
        return []
    # Part 19 performance pass: one vectorised `norm.isf` instead of one scalar
    # scipy call per observation (10k scalar calls cost ~1.1 s and made S1 the
    # slowest part of the 10k-row path). The arithmetic is unchanged.
    k = np.arange(1, n + 1, dtype=float)
    q = alpha * k / n
    valid = (q > 0) & (q < 1)
    expected = np.zeros(n, dtype=float)
    expected[valid] = stats.norm.isf(0.5 * q[valid])
    vals = sorted_abs_z.to_numpy(dtype=float)
    mask = (expected > 0) & np.isfinite(expected) & (vals >= 1.15 * expected)
    return [sorted_abs_z.index[i] for i in np.flatnonzero(mask)]


def _resolution(values: pd.Series) -> float:
    """
    The smallest step the data is actually *recorded* at (0 when it is continuous).

    Estimated from the gaps between distinct sorted values rather than from magic
    constants: if the 10th percentile of those gaps is small relative to the spread,
    the data is continuous and there is no resolution floor to apply.
    """
    v = np.unique(np.round(values.dropna().to_numpy(dtype=float), 10))
    if len(v) < 5:
        return 0.0
    gaps = np.diff(v)
    gaps = gaps[gaps > 0]
    if len(gaps) < 3:
        return 0.0
    step = float(np.percentile(gaps, 10))
    spread = float(np.nanpercentile(v, 95) - np.nanpercentile(v, 5))
    if spread <= 0 or step * 100 > spread:
        return 0.0
    # snap to the nearest recognisable resolution so 0.0999999 reads as 0.1
    for candidate in (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0, 5.0, 10.0, 100.0):
        if abs(step - candidate) <= 0.25 * candidate:
            return candidate
    return step


def _decompose(values: pd.Series, period: int | None) -> tuple[pd.Series, pd.Series, str]:
    """
    Return (remainder, deseasonalised_level, basis).

    The remainder is what the outlier rule tests. The deseasonalised level (series
    minus its seasonal component) is what the *type* classifier needs: a level shift
    is invisible in an STL remainder, because STL absorbs the step into its trend, but
    it is the most obvious thing in the world in the deseasonalised series.
    """
    s = values.astype(float)
    n = int(s.notna().sum())
    if period and period > 1 and n >= 2 * period + 2:
        try:
            # seasonal smoother must be odd and at least 7; 7 on a 12-period cycle
            # tracks the seasonal shape so closely that it absorbs genuine extremes,
            # which then look ordinary in their own remainder. Use the larger of 7 and
            # the shortest sensible odd window for the period.
            win = max(7, (period // 2) * 2 + 1)
            # Part 19: for period 7 this is byte-for-byte the fit used by
            # seasonal.stl_strength, so the second request is a cache lookup.
            res = robust_stl_components(s, period, seasonal=win)
            seas = res["seasonal"]
            return (
                res["resid"],
                s - seas,
                f"robust STL remainder (period {period})",
            )
        except Exception:  # pragma: no cover - very short or degenerate series
            pass
    window = max(3, min(13, (period or 7) | 1))
    trend = s.rolling(window, center=True, min_periods=2).median()
    return (s - trend), (s - trend), f"series minus a centred {window}-point moving median"


def _step_onsets(deseason: pd.Series, max_steps: int = 3) -> list[tuple[int, float]]:
    """
    Positions where the deseasonalised level jumps and stays jumped.

    A step is detected from the *first differences*: the jump itself is one large
    difference, while the ordinary period-to-period spread is estimated robustly
    (MAD), so a single jump cannot inflate its own threshold. Returns (position, size).
    """
    d = deseason.dropna()
    if len(d) < 12:
        return []
    diffs = np.diff(d.to_numpy(dtype=float))
    mad = float(np.median(np.abs(diffs - np.median(diffs))))
    scale = 1.4826 * mad if mad > 0 else float(np.std(diffs, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return []
    # The floor is expressed in units of the *differences*, not the levels: a step must
    # be large compared with how the series ordinarily moves, not compared with how big
    # its values happen to be.
    tolerance = max(6.0 * scale, 0.5 * float(np.nanpercentile(np.abs(diffs), 95)))
    values = d.to_numpy(dtype=float)
    pos = np.arange(1, len(values))
    jumps = np.abs(diffs)
    # a step's size must also be material against the *series* — 0.3 in a series whose
    # values run to thousands is a rounding artefact, not a regime change
    material = 0.15 * float(np.nanpercentile(np.abs(values), 95) or 1.0)
    hits = [
        (int(i), float(values[i] - values[i - 1]))
        for i in pos
        if jumps[i - 1] >= tolerance and abs(float(values[i] - values[i - 1])) >= material
    ]
    # A jump must *stay* jumped. A single spike also produces a huge first difference,
    # but the level comes straight back, so compare the level three points after the
    # candidate with the level three points before it.
    persistent = []
    for i, size in hits:
        before = values[max(0, i - 3): i]
        after = values[i + 3: i + 6]
        if len(before) < 3 or len(after) < 3:
            continue
        gap_before = float(values[i] - np.median(before))
        gap_after = float(np.median(after) - np.median(before))
        if abs(gap_after) >= 0.6 * abs(gap_before) and abs(gap_after) >= tolerance:
            persistent.append((i, gap_after))
    hits = persistent
    if not hits:
        return []
    # keep the largest jump in each neighbourhood, biggest first
    hits.sort(key=lambda x: -abs(x[1]))
    kept: list[tuple[int, float]] = []
    for idx, size in hits:
        if all(abs(idx - k) > 6 for k, _ in kept):
            kept.append((idx, size))
    return sorted(kept)[:max_steps]


def detect_outliers(
    values: pd.Series,
    period: int | None = None,
    z_cutoff: float = 4.0,
    max_report: int = 25,
) -> tuple[list[Outlier], dict[str, Any]]:
    """
    Return the labelled outliers plus a summary the UI and advisor can both use.

    The score is a standardised deviation from the remainder's median, in a scale that
    is the SD when it is not distorted and the MAD-equivalent when it is (see below).
    `z_cutoff` is 4.0: the classic Iglewicz–Hoaglin suggestion is 3.5 on a MAD scale,
    but because we switch to the less tail-robust SD whenever it looks clean, the
    cut-off has to be stricter to keep the false-positive rate where 3.5-on-MAD has it.
    """
    s = values.astype(float)
    resid, deseason, basis = _decompose(s, period)
    # A flag must also be material in the units the user reads. For a series recorded
    # to the nearest unit (counts, whole pesos) every ordinary wobble is a large
    # multiple of MAD, and the panel would otherwise report the last decimal place as
    # an anomaly. So require the deviation to span at least a few recording steps.
    resolution = _resolution(s)

    r = resid.dropna()
    if len(r) < 4:
        return [], {"available": False, "basis": basis,
                    "plain": "Too few observations to look for outliers reliably."}

    # Work in robust standardised units. An intermittent series can have a remainder
    # spanning six orders of magnitude, where raw variances are meaningless.
    med = float(np.median(r))
    mad = float(np.median(np.abs(r - med)))
    base = mad / 0.6745 if mad > 0 else float(np.std(r.to_numpy(), ddof=1)) or 1e-12
    mad_scale = 1.0
    sd_scale = float(np.std(r.to_numpy(), ddof=1)) / base or 1.0
    # MAD is unbiased for clean data but very inefficient under heavy tails; the SD is the
    # reverse. Use the SD when it is consistent with MAD (ratio ≈ 1 for Gaussian data,
    # well below the 2–3 an outlier would produce), and fall back to MAD exactly when
    # something is inflating the spread — which is the situation MAD exists for. The
    # band is ±25% because on a few hundred points the ratio itself carries ~10% noise,
    # and a rejection boundary sitting on that noise flickers between the two scales.
    scale = sd_scale if sd_scale <= 1.25 * mad_scale else mad_scale
    z = (r - med) / (base * scale)

    # A z-score alone flags trivial wobbles when the remainder is very tight, so we
    # also require the deviation to be material relative to the remainder's spread.
    sd_resid = sd_scale                       # in units of the robust base scale
    min_effect = (min(MIN_EFFECT_STEPS * resolution / base, 0.75 * sd_resid * base)
                  if resolution > 0 else 0.0)
    material = (np.abs(z) > 0.75 * sd_resid) & (
        (np.abs(r - med) >= min_effect) if min_effect > 0 else True
    )
    flagged = z[(np.abs(z) > z_cutoff) & material.reindex(z.index, fill_value=False)]

    # A 4-sigma rule on the *estimated* spread still over-fires, for a reason specific
    # to this pipeline: STL's robust (bisquare) fit deliberately shrinks its own
    # remainder — on pure noise it returns about 0.88 of the series' SD, so honest
    # extremes are scored against a scale that has already deleted them. Calibrate the
    # scale to the series instead of the data to the scale.
    scale = _calibrate_scale(scale * base, s) / base
    z = (r - med) / (base * scale)

    # Second guard: ask whether each flagged point is far out *for its rank* — the same
    # idea as a studentised extreme-value test, in one line.
    if len(z) >= 20:
        order = z.abs().sort_values(ascending=False)
        flagged = flagged[flagged.index.isin(_tail_outliers(order))]

    # Cap the flag rate. When a model-free remainder is very tight, a fixed z cut-off
    # flags a large share of perfectly ordinary observations; showing the user 10% of
    # their series as "outliers" is worse than showing none. We keep the most extreme
    # MAX_FLAG_RATE share and say in the summary that we did.
    max_flags = max(2, int(np.ceil(MAX_FLAG_RATE * len(r))))
    capped = False
    if len(flagged) > max_flags:
        flagged = flagged.loc[flagged.abs().nlargest(max_flags).index].sort_index()
        capped = True
    # Level shifts are found separately: a step is not a property of one observation,
    # and the z-rule above sees only its onset. Reporting the onset is what makes the
    # advice actionable ("add a step dummy at 2015-01").
    steps = _step_onsets(deseason)
    step_times = {s.index[i] for i, _ in steps}

    # Consecutive flags describe one event, not five: keep the most extreme point of
    # each cluster so a level shift is labelled once, at its onset.
    kept: list[pd.Timestamp] = []
    positions = [s.index.get_loc(t) for t in flagged.index]
    last_pos = -10
    for t, pos in zip(flagged.index, positions):
        if pos - last_pos > 2:
            best_in_window = max(
                (p for p in positions if pos <= p <= pos + 2),
                key=lambda p: abs(float(z.iloc[p])),
            )
            kept.append(s.index[best_in_window])
        last_pos = pos

    # a step absorbs the z-flags within a few observations of its onset
    def _near_step(i: int) -> bool:
        return any(abs(i - k) <= 4 for k, _ in steps)

    kept = [t for t in kept if not _near_step(int(s.index.get_loc(t)))]

    out: list[Outlier] = []
    for i, size in steps:
        t = s.index[i]
        val = float(s.loc[t])
        expected = val - size
        try:
            zz = float(z.loc[t])
        except KeyError:  # pragma: no cover - onset outside the tested remainder
            zz = float("nan")
        direction = "above" if size > 0 else "below"
        out.append(Outlier(
            time=str(t), value=val, expected=expected, remainder=size,
            robust_z=zz, direction=direction, type="level_shift",
            unusual_for_season=False,
            plain=_plain_for("level_shift", t, val, expected, period),
        ))

    for t in kept[:max_report]:
        val = float(s.loc[t])
        rem = float(r.loc[t])
        zz = float(z.loc[t])
        expected = val - rem
        kind, unusual = _classify(s, resid, t, period, deseason)
        out.append(Outlier(
            time=str(t), value=val, expected=expected, remainder=rem, robust_z=zz,
            direction="above" if rem > 0 else "below", type=kind,
            unusual_for_season=unusual,
            plain=_plain_for(kind, t, val, expected, period, unusual),
        ))

    counts: dict[str, int] = {}
    for o in out:
        counts[o.type] = counts.get(o.type, 0) + 1

    n_flagged = len(out)
    summary = {
        "available": True,
        "basis": basis,
        "z_cutoff": z_cutoff,
        "scale_used": "standard deviation" if scale is sd_scale else "MAD",
        "n_flagged": n_flagged,
        "capped": capped,
        "max_flag_rate": MAX_FLAG_RATE,
        "counts_by_type": counts,
        "largest_abs_z": float(np.max(np.abs(z))) if len(z) else 0.0,
        "largest_times": [str(t) for t in z.abs().nlargest(5).index],
        "plain": _summary_plain(n_flagged, counts, len(s), capped),
    }
    return out, summary


def _classify(
    values: pd.Series,
    resid: pd.Series,
    t: pd.Timestamp,
    period: int | None,
    deseason: pd.Series | None = None,
) -> tuple[OutlierType, bool]:
    """
    Return (type, unusual_for_season).

    Order of tests: persistence beats transience — a displaced level is a
    structural feature, not an anomaly to be deleted.
    """
    pos = values.index.get_loc(t)
    here = float(resid.loc[t])
    # Persistence is judged against *this episode*, not against the whole series, and
    # in the units the episode actually moves in. Comparing with a local baseline
    # median is what lets a one-off spike on a wandering series stay a spike.
    level = deseason if deseason is not None else resid
    peak = float(level.loc[t])
    before = level.iloc[max(0, pos - 3): pos].dropna()
    baseline = float(np.median(before)) if len(before) else peak
    after = (level.iloc[pos + 1: pos + 4].dropna() - baseline)

    if len(after) >= 2:
        same_side = np.sign(after.to_numpy()) == np.sign(here)
        if same_side.mean() >= 0.67 and abs(float(after.mean())) > 0.6 * abs(here):
            return "level_shift", False

    if len(after) >= 3 and abs(here) > 0:
        magnitudes = np.abs(after.to_numpy() - float(level.iloc[max(0, pos - 3): pos].mean()))
        if magnitudes[0] > 0.5 * abs(here) and magnitudes[-1] < 0.3 * abs(here):
            return "temporary_change", False

    if len(after) >= 2 and abs(here) > 0:
        window = resid.iloc[max(0, pos - 1): pos + 2].dropna()
        if len(window) >= 3 and np.all(np.abs(window.to_numpy()) > 0.6 * abs(here)):
            return "isolated_run", False

    unusual = False
    if period and period > 1 and hasattr(t, "month"):
        peers = values[values.index.month == t.month].drop(labels=[t], errors="ignore")
        if len(peers) >= 4:
            sd = float(peers.std(ddof=1)) or 1e-12
            unusual = bool(abs(float(values.loc[t]) - float(peers.mean())) / sd > 3.0)
    return "spike", unusual


def _plain_for(
    kind: OutlierType, t: Any, val: float, expected: float, period: int | None,
    unusual_for_season: bool = False,
) -> str:
    gap = val - expected
    direction = "above" if gap > 0 else "below"
    if kind == "level_shift":
        return (
            f"{t:%Y-%m-%d}: the series steps to a new level ({val:,.4g}, about "
            f"{abs(gap):,.4g} {direction} what the pattern predicts) and stays there. "
            "Do not delete this point — model it with a step dummy."
        )
    if kind == "temporary_change":
        return (
            f"{t:%Y-%m-%d}: a temporary excursion {direction} trend and season expect, "
            "decaying back over the next few observations."
        )
    if kind == "isolated_run":
        return (
            f"{t:%Y-%m-%d}: part of a short run of unusual values rather than a single point."
        )
    base = (
        f"{t:%Y-%m-%d}: a single point {abs(gap):,.4g} {direction} what trend and season "
        f"predict ({expected:,.4g}). Often a typo or a one-off event."
    )
    if unusual_for_season:
        base += (
            " It is also far from the same month/quarter in other years, so correcting the "
            "seasonal pattern rather than deleting the point may be the better fix."
        )
    return base


def _summary_plain(n: int, counts: dict[str, int], total: int, capped: bool = False) -> str:
    if n == 0:
        return "No observations are extreme once trend and seasonality are removed."
    parts = ", ".join(f"{v} {k.replace('_', ' ')}{'s' if v > 1 else ''}" for k, v in counts.items())
    text = (
        f"{n} of {total} observations are unusual after removing trend and seasonality "
        f"({parts})."
    )
    if capped:
        text += (
            " More points passed the threshold than that, so only the "
            f"{n} most extreme ({MAX_FLAG_RATE:.0%} of the series) are shown."
        )
    return text


__all__ = ["Outlier", "OutlierType", "detect_outliers"]
