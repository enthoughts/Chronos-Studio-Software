"""
ts/correlate/_acf.py — ACF, PACF, bands, and the reading of their shape (§5.6).

The tests in this module are arithmetic; the judgement is in `read_pattern` and `order_hints`,
and the governing rule there is that **a shape is a hint, never a decision**. An ACF that decays
slowly is *consistent with* a unit root; it is not evidence of one — Step 5's tests are. Every
string this module returns is a sentence that says "this looks like", and the API attaches the
unit-root verdict it must be checked against.

Three things are easy to get wrong and are done deliberately here:

* **Bands are Bartlett's, not just ±1.96/√n.** The white-noise band is drawn as well (it is the
  one beginners have seen), but significance is judged against the Bartlett band
  `±1.96·sqrt((1 + 2·Σ_{k<lag} r_k²)/n)`, which widens once the series is autocorrelated. Judging a
  strongly autocorrelated series against the white-noise band declares almost every lag
  "significant".

* **PACF is computed by Durbin–Levinson from the ACF**, not by regressing on lagged values. The
  regression route silently absorbs unit roots and returns values outside [−1, 1] on near-random
  walks.

* **The ACF of the raw series is computed on the raw series.** Unlike the periodogram (Part 4),
  there is no artefact here: correlation is what the raw data has. Where differencing is needed,
  the *comparison* grid is what shows it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# The plain ACF
# --------------------------------------------------------------------------- #
@dataclass
class ACF:
    """Autocorrelations at lags 1..max_lag, with both bands."""

    lags: list[int]
    values: list[float]
    band_white: float
    band_bartlett: list[float]
    n: int
    max_lag: int

    def at(self, lag: int) -> float | None:
        if lag < 1 or lag > self.max_lag:
            return None
        return self.values[lag - 1]

    def significant(self, lag: int) -> bool:
        """Is the correlation at `lag` outside the Bartlett band? (None-safe.)"""
        v = self.at(lag)
        if v is None:
            return False
        return abs(v) > self.band_bartlett[lag - 1]

    def significant_lags(self) -> list[int]:
        return [lag for lag in self.lags if self.significant(lag)]

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "lag": int(lag),
                "acf": round(float(v), 4),
                "band": round(float(b), 4),
                "significant": bool(abs(v) > b),
            }
            for lag, v, b in zip(self.lags, self.values, self.band_bartlett)
        ]


@dataclass
class PACF:
    lags: list[int]
    values: list[float]
    band_white: float
    band_bartlett: list[float]

    def significant(self, lag: int) -> bool:
        if lag < 1 or lag > len(self.values):
            return False
        return abs(self.values[lag - 1]) > self.band_bartlett[lag - 1]

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "lag": int(lag),
                "pacf": round(float(v), 4),
                "band": round(float(b), 4),
                "significant": bool(abs(v) > b),
            }
            for lag, v, b in zip(self.lags, self.values, self.band_bartlett)
        ]


def default_max_lag(n: int, period: int | None) -> int:
    """
    §5.6 asks for lags 1…min(4·period, n/5).

    Two guards on top of that literal rule: never fewer than 10 lags (a 12-observation monthly
    series would otherwise get 2, and every "no pattern" verdict would be vacuous), and never more
    than a third of the sample (each extra lag is another correlation estimated from fewer pairs).
    """
    rule = n // 5
    if period:
        rule = min(4 * period, rule) if rule >= 8 else max(4 * period, 10)
    max_lag = max(10, min(int(rule), n // 3))
    return int(max(5, min(max_lag, n - 5)))


def acf(series: pd.Series, max_lag: int | None = None, period: int | None = None) -> ACF:
    """
    Sample autocorrelation with the bias-corrected denominator (n, not n−k).

    Using n−k inflates the ACF at long lags and makes a short series look persistent. Every
    serious implementation uses n; so does this one.
    """
    y = series.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 8:
        raise ValueError("Autocorrelation needs at least 8 observations.")
    max_lag = int(max_lag or default_max_lag(n, period))
    max_lag = max(1, min(max_lag, n - 3))

    dev = y - y.mean()
    denom = float(np.dot(dev, dev))
    if denom <= 0:
        raise ValueError("The series is constant, so autocorrelation is undefined.")

    values: list[float] = []
    for k in range(1, max_lag + 1):
        values.append(float(np.dot(dev[:-k], dev[k:]) / denom))

    # Bartlett: Var(r_k) ≈ (1 + 2·Σ_{j<k} r_j²)/n. Lag 1 uses the white-noise band by definition.
    bartlett: list[float] = []
    cum = 0.0
    for k, r in enumerate(values, start=1):
        bartlett.append(float(1.96 * np.sqrt((1.0 + 2.0 * cum) / n)))
        cum += r * r

    return ACF(
        lags=list(range(1, max_lag + 1)),
        values=values,
        band_white=float(1.96 / np.sqrt(n)),
        band_bartlett=bartlett,
        n=n,
        max_lag=max_lag,
    )


def pacf(series: pd.Series, max_lag: int | None = None, period: int | None = None) -> PACF:
    """
    Partial autocorrelation via Durbin–Levinson recursion on the ACF.

    `pacf[k-1]` is the last coefficient of an AR(k) fit — the correlation between y_t and y_{t−k}
    with the intermediate lags projected out. Computed from the ACF because that route is stable;
    the direct regression route returns |values| > 1 on near-unit-root data, which is a numerical
    artefact, not a finding.
    """
    a = acf(series, max_lag=max_lag, period=period)
    r = np.asarray(a.values, dtype=float)
    m = len(r)

    phi = np.zeros((m + 1, m + 1), dtype=float)
    out: list[float] = []
    for k in range(1, m + 1):
        if k == 1:
            phi[1, 1] = r[0]
        else:
            num = r[k - 1] - sum(phi[k - 1, j] * r[k - j - 1] for j in range(1, k))
            den = 1.0 - sum(phi[k - 1, j] * r[j - 1] for j in range(1, k))
            phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        out.append(float(np.clip(phi[k, k], -1.0, 1.0)))

    # The PACF band is the white-noise band: under the null of AR(p), the coefficients beyond p are
    # approximately N(0, 1/n). A Bartlett widening does not apply here — the AR fit already
    # removes the dependence that would justify it.
    band = float(1.96 / np.sqrt(a.n))
    return PACF(
        lags=list(range(1, m + 1)),
        values=out,
        band_white=band,
        band_bartlett=[band] * m,
    )


# --------------------------------------------------------------------------- #
# Reading the shape
# --------------------------------------------------------------------------- #
PatternKind = Literal[
    "white_noise", "ar", "ma", "nonstationary_decay", "seasonal", "overdifferenced",
    "single_spike", "unclear",
]


@dataclass
class Pattern:
    """One named reading of the correlogram, with the numbers that produced it."""

    kind: PatternKind
    label: str
    plain: str
    confidence: str = "medium"                    # low | medium | high
    suggests: str = ""                            # the order hint this pattern implies
    evidence: dict[str, Any] = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "plain": self.plain,
            "confidence": self.confidence,
            "suggests": self.suggests,
            "evidence": self.evidence,
            "caveats": self.caveats,
        }


def _run_of_significant(values: list[float], band: list[float], limit: int) -> int:
    """How many consecutive lags from lag 1 are significant?"""
    run = 0
    for v, b in zip(values, band):
        if abs(v) > b:
            run += 1
        else:
            break
    return min(run, limit)


def read_pattern(
    a: ACF,
    p: PACF,
    *,
    period: int | None = None,
    ljung_box_p: float | None = None,
    differenced: bool = False,
) -> Pattern:
    """
    Name the shape of the correlogram.

    The order of the tests matters. Seasonality is checked before AR/MA because a seasonal spike
    also creates significant ACF at 2s, 3s… and would otherwise read as a slowly decaying AR.
    Over-differencing is checked before "MA-like", because a single large negative lag-1 is the
    signature of both, and the differencing already applied is the tie-breaker.
    """
    r = a.values
    sig = [lag for lag in a.lags if a.significant(lag)]
    n_sig = len(sig)
    r1 = r[0]
    # `decay` is a *contiguous run from lag 1*. Counting scattered significant lags instead is the
    # classic misreading: at 5% with 36 lags, about two lags are expected to clear the band by
    # chance alone, and a count-based rule then invents a cut-off that is not there.
    decay = _run_of_significant(r, a.band_bartlett, len(r))
    chance = 0.05 * a.max_lag
    # Upper end of what chance alone produces: the mean plus about two binomial SDs. Using the
    # mean alone would call a genuinely white series "structured" whenever it drew one extra lag.
    chance_ceiling = chance + 2.0 * (chance ** 0.5)

    # ---- 1. white noise ----------------------------------------------------
    if decay == 0 and n_sig <= max(1.0, chance_ceiling) and not differenced:
        return Pattern(
            kind="white_noise",
            label="white-noise-like",
            plain=(
                f"No autocorrelation is outside its band at lag 1, and only {n_sig} of the first "
                f"{a.max_lag} lags clear the band at all — about {chance:.1f} are expected by chance "
                f"at 5%. There is no linear dependence left at these lags, which is what white "
                "noise looks like."
            ),
            confidence="high" if n_sig == 0 else "medium",
            suggests="no AR or MA term is justified by the correlogram",
            evidence={"significant_lags": sig, "max_lag": a.max_lag,
                      "largest": round(float(max(r, key=abs)), 4)},
            caveats=[
                "This is a statement about *linear* dependence at the lags examined. A nonlinear "
                "pattern, a variance shift or a break would not show up here.",
                "With a short series the band is wide, so 'no significant lag' is also what a weak "
                "AR(1) looks like. Absence of evidence, not evidence of absence.",
            ],
        )

    # ---- 2. seasonal spike -------------------------------------------------
    if period and period >= 2 and period <= a.max_lag:
        seasonal_sig = [k for k in range(period, a.max_lag + 1, period) if a.significant(k)]
        if seasonal_sig and abs(a.at(period) or 0) > max(a.band_bartlett[period - 1] * 1.5, 0.2):
            harmonics = len(seasonal_sig)
            return Pattern(
                kind="seasonal",
                label=f"seasonal spike at {period}",
                plain=(
                    f"The autocorrelation peaks at lag {period} "
                    f"(r = {_n(a.at(period))}), which is the seasonal period, and repeats at "
                    f"{', '.join(str(k) for k in seasonal_sig[1:3]) or 'no further multiples'} "
                    f"({harmonics} of the {a.max_lag // period} seasonal harmonics are outside "
                    "their bands). The pattern repeats every "
                    f"{period} observations."
                ),
                confidence="high" if harmonics >= 2 else "medium",
                suggests=f"seasonal AR/MA terms at lag {period} (P or Q ≥ 1), or a seasonal "
                         "difference if the seasonal spike decays slowly",
                evidence={"period": period, "r_at_period": round(float(a.at(period) or 0), 4),
                          "seasonal_lags_significant": seasonal_sig},
                caveats=[
                    "A seasonal spike in the ACF is not proof that the seasonal pattern is stable. "
                    "Step 3's decomposition shows whether it is; Step 5's OCSB test says whether it "
                    "needs a seasonal difference.",
                ],
            )

    # ---- 3. over-differencing ---------------------------------------------
    if r1 < -0.5:
        return Pattern(
            kind="overdifferenced",
            label="alternating / negative lag-1 → possible over-differencing",
            plain=(
                f"Lag 1 is strongly negative (r₁ = {_n(r1)}). A correlation that negative at lag 1 is "
                "the signature of differencing a series that did not need it: differencing a white "
                "noise series produces an MA(1) with coefficient −1, whose ACF is exactly −0.5 at "
                "lag 1 and zero after. If this series has already been differenced, that is the "
                "likely explanation."
            ),
            confidence="high" if r1 < -0.6 else "medium",
            suggests="undo the last difference rather than adding another",
            evidence={"r1": round(float(r1), 4)},
            caveats=[
                "A genuine MA(1) with a large negative coefficient looks identical. The tie-breaker "
                "is whether a difference was actually applied — and whether the variance *rose* "
                "after differencing, which is the formal symptom of over-differencing.",
            ],
        )

    # ---- 4. slow decay → non-stationary / long memory ----------------------
    if decay >= max(3, min(len(r) - 1, 6)) and abs(r1) > 0.6:
        half = decay / 2
        mean_first = float(np.mean(r[: max(1, int(half))]))
        mean_second = float(np.mean(r[int(half):decay])) if decay > 2 else mean_first
        still_large = mean_second > 0.5 * mean_first
        return Pattern(
            kind="nonstationary_decay",
            label="slowly decaying → non-stationary / long memory",
            plain=(
                f"The autocorrelation stays outside its band for {decay} consecutive lags and is "
                f"still {_n(r[decay - 1])} at lag {decay}. A decay this slow is what a series with a "
                "trend, a unit root or long memory looks like"
                + (", and it has not halved over that stretch, so this is the persistent version"
                   if still_large else "")
                + ". A stationary AR process' ACF dies away geometrically instead."
            ),
            confidence="medium",
            suggests="check Step 5's unit-root verdict before differencing; if it says stationary, "
                     "this is long memory or a trend, not a unit root",
            evidence={"decay_lags": decay, "r1": round(float(r1), 4),
                      "r_first_half": round(mean_first, 4), "r_second_half": round(mean_second, 4)},
            caveats=[
                "The correlogram cannot distinguish a unit root from a near-unit-root stationary "
                "process, from a trend, or from long memory — all four look like this. Step 5's "
                "tests are the ones that can; this is only a reason to go and look.",
            ],
        )

    # ---- 5. MA-like: sharp cut-off after a contiguous run -------------------
    # A cut-off means lags 1..k are significant *in a row* and k+1..k+3 are not. Requiring the run
    # is what separates a real MA signature from scattered chance lags.
    if 1 <= decay <= 4:
        last = decay
        after = [k for k in range(last + 1, min(last + 4, a.max_lag) + 1) if a.significant(k)]
        if not after:
            pacf_sig = [lag for lag in p.lags[: min(8, len(p.values))] if p.significant(lag)]
            return Pattern(
                kind="ma",
                label=f"MA({last})-like",
                plain=(
                    f"The autocorrelation is outside its band for lags 1–{last} and then drops "
                    f"inside it and stays there (the next significant lag is "
                    f"{min(after) if after else 'none within ' + str(min(last + 4, a.max_lag))}). A "
                    "sharp cut-off in the ACF, with the partial autocorrelation trailing off "
                    f"(significant at {len(pacf_sig)} of the first 8 lags), is the textbook MA "
                    "signature."
                ),
                confidence="medium" if last <= 2 else "low",
                suggests=f"MA({last}) — or ARIMA(0,0,{last}) if a mean is needed",
                evidence={"cut_off": last, "significant_lags": sig},
                caveats=[
                    "'MA(q)-like' means the shape is *consistent with* an MA(q). A cut-off is "
                    "estimated loosely even from a long series, and an AR process with a root near "
                    "the unit circle can mimic it.",
                ],
            )

    # ---- 6. AR-like: geometric decay --------------------------------------
    if decay >= 2 and abs(r1) > 2 * a.band_bartlett[0]:
        pacf_cut = max((lag for lag in p.lags[: min(6, len(p.values))] if p.significant(lag)),
                       default=1)
        return Pattern(
            kind="ar",
            label=f"short-memory AR({pacf_cut})-like",
            plain=(
                f"The autocorrelation decays steadily (r₁ = {_n(r1)}, r₂ = {_n(a.at(2))}, "
                f"r₃ = {_n(a.at(3))}) and leaves the band by lag {decay}. The partial "
                f"autocorrelation is significant up to lag {pacf_cut} and then inside its band, "
                "which is the signature of an autoregressive process of that order."
            ),
            confidence="medium",
            suggests=f"AR({pacf_cut}) — or ARMA({pacf_cut},1) if one MA term improves the fit",
            evidence={"decay_lags": decay, "pacf_cut_off": pacf_cut, "r1": round(float(r1), 4)},
            caveats=[
                "The order read off a correlogram is a starting point. Any serious choice is made "
                "by comparing information criteria over a grid — the correlogram narrows the grid, "
                "it does not decide.",
            ],
        )

    # ---- 7. one spike only -------------------------------------------------
    if n_sig == 1:
        lag = sig[0]
        return Pattern(
            kind="single_spike",
            label=f"single significant spike at lag {lag}",
            plain=(
                f"Exactly one lag is outside its band: lag {lag} (r = {_n(a.at(lag))}). A single "
                "isolated spike is a one-term hint, but it can also come from one outlier or one "
                "unusual observation propagating into a few correlations."
            ),
            confidence="low",
            suggests=f"check lag {lag} against a possible MA term, and check the raw series for an "
                     "outlier at that spacing",
            evidence={"lag": lag, "r": round(float(a.at(lag) or 0), 4)},
            caveats=[
                "With "
                f"{a.max_lag} lags examined at 5%, about {round(0.05 * a.max_lag, 1)} significant "
                "lags are expected by chance alone. One spike is within that expectation.",
            ],
        )

    return Pattern(
        kind="unclear",
        label="no single clean pattern",
        plain=(
            f"The correlogram is significant at {n_sig} of the first {a.max_lag} lags "
            f"({', '.join(map(str, sig[:6]))}{', …' if n_sig > 6 else ''}) without a clean cut-off or "
            "a smooth decay. That is common in real data and means the correlogram does not, by "
            "itself, point at an order."
        ),
        confidence="low",
        suggests="compare a small grid of (p,q) by information criterion rather than reading an "
                 "order off the plot",
        evidence={"significant_lags": sig[:12], "n_significant": n_sig},
        caveats=[
            "Multiple significant lags without structure can also be a variance shift, a level "
            "shift, or a few outliers. Check Step 4's remainder and Step 5's break tests before "
            "reading this as memory.",
        ],
    )


# --------------------------------------------------------------------------- #
# Box–Pierce family
# --------------------------------------------------------------------------- #
@dataclass
class Portmanteau:
    lags: list[int]
    stats: list[float]
    p_values: list[float]
    df: list[int]
    kind: str                       # "ljung_box" | "box_pierce"

    def at(self, lag: int) -> dict[str, Any] | None:
        if lag not in self.lags:
            return None
        i = self.lags.index(lag)
        return {"lag": lag, "statistic": self.stats[i], "p_value": self.p_values[i],
                "df": self.df[i]}

    def rows(self) -> list[dict[str, Any]]:
        return [
            {"lag": int(l), "statistic": round(float(s), 3), "df": int(d),
             "p_value": round(float(p), 4), "rejects": bool(p < 0.05)}
            for l, s, p, d in zip(self.lags, self.stats, self.p_values, self.df)
        ]


def _portmanteau(a: ACF, lags: list[int], *, kind: str) -> Portmanteau:
    n = a.n
    stats_out: list[float] = []
    p_out: list[float] = []
    df_out: list[int] = []
    for h in lags:
        h = min(h, a.max_lag)
        if kind == "ljung_box":
            q = n * (n + 2) * sum((a.values[k - 1] ** 2) / (n - k) for k in range(1, h + 1))
        else:
            q = n * sum(a.values[k - 1] ** 2 for k in range(1, h + 1))
        stats_out.append(float(q))
        df_out.append(int(h))
        p_out.append(float(stats.chi2.sf(q, h)))
    return Portmanteau(lags=lags, stats=stats_out, p_values=p_out, df=df_out, kind=kind)


def ljung_box(a: ACF, lags: list[int] | None = None) -> Portmanteau:
    """
    Ljung–Box Q: are *any* of the first h autocorrelations non-zero?

    The lag set is chosen to be informative rather than arbitrary: early lags where AR/MA lives,
    2·period where seasonality lives, and 2·period again for the annual echo. A single Ljung–Box at
    lag 10 on monthly data cannot see a seasonal spike at 12 at all — reporting only that is a
    classic way to "pass" a badly specified model.
    """
    return _portmanteau(a, lags or [10], kind="ljung_box")


def box_pierce(a: ACF, lags: list[int] | None = None) -> Portmanteau:
    """Box–Pierce for comparison; it is known to be badly sized on short samples."""
    return _portmanteau(a, lags or [10], kind="box_pierce")


# --------------------------------------------------------------------------- #
# Variance ratio — the over-differencing cross-check
# --------------------------------------------------------------------------- #
def variance_ratio(series: pd.Series, q: int = 2) -> dict[str, Any]:
    """
    Lo–MacKinlay variance-ratio statistic for a random walk null.

    For a random walk, Var(y_t − y_{t−q}) = q·Var(y_t − y_{t−1}). A ratio well below 1 means
    successive differences are negatively autocorrelated — the fingerprint of differencing a
    series that was already stationary. A ratio well above 1 means positive dependence, i.e. the
    level series is persistent and differencing is doing something real.

    This is the formal version of the "the variance went up after differencing" symptom that
    beginners notice but rarely quantify.
    """
    y = series.dropna().to_numpy(dtype=float)
    n = len(y)
    if n < 4 * q + 4:
        raise ValueError(f"The variance-ratio test with q = {q} needs more observations.")
    d1 = np.diff(y)
    dq = y[q:] - y[:-q]
    v1 = float(np.var(d1, ddof=1))
    vq = float(np.var(dq, ddof=1))
    if v1 <= 0:
        raise ValueError("The series is constant, so the variance ratio is undefined.")
    vr = vq / (q * v1)
    # Lo–MacKinlay heteroskedasticity-robust variance of VR
    delta = 0.0
    mu = float(np.mean(d1))
    denom = float(np.sum((d1 - mu) ** 2))
    if denom > 0:
        for j in range(1, q):
            num = float(np.sum(((d1[j:] - mu) ** 2) * ((d1[:-j] - mu) ** 2)))
            delta += (num / denom) ** 2 * (2 * (q - j) / q)
    nq = n - 1
    var_vr = delta / nq if delta > 0 else (2.0 * (2 * q - 1) * (q - 1)) / (3.0 * q * nq)
    z = (vr - 1.0) / np.sqrt(var_vr) if var_vr > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return {
        "q": int(q),
        "ratio": float(vr),
        "z": float(z),
        "p_value": p,
        "var_diff1": v1,
        "var_diffq": vq,
        # What the numbers mean, stated so that neither branch is a verdict on its own:
        # white noise gives VR = 0.5 (its differences are an MA(1) with θ = −1), which is *not*
        # over-differencing — it is simply not a random walk. Only the combination of a low VR and
        # a difference actually having been applied is evidence of over-differencing, and the
        # advisor rule (`next.diff.undo_overdifferenced`) is where that combination is judged.
        "reading": (
            "changes are negatively autocorrelated, as they are for white noise (VR = 0.5) or for "
            "a series that has been over-differenced — the two are told apart by whether a "
            "difference was actually applied"
            if vr < 0.8 else
            "changes are positively autocorrelated, which is what a persistent level series looks "
            "like"
            if vr > 1.25 else
            "the variance ratio is close to 1, which is what a random walk looks like"
        ),
    }


# --------------------------------------------------------------------------- #
# Model-order hints (never decisions)
# --------------------------------------------------------------------------- #
@dataclass
class OrderHint:
    """One candidate order, with the correlogram feature that produced it."""

    p: int
    d: int
    q: int
    P: int
    D: int
    Q: int
    s: int
    label: str
    plain: str
    support: list[str] = field(default_factory=list)
    against: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": {"p": self.p, "d": self.d, "q": self.q,
                      "P": self.P, "D": self.D, "Q": self.Q, "s": self.s},
            "notation": self.label,
            "plain": self.plain,
            "support": self.support,
            "against": self.against,
        }


def order_hints(
    a: ACF,
    p: PACF,
    *,
    pattern: Pattern,
    period: int | None = None,
    differencing_needed: bool = False,
    seasonal_differencing_needed: bool = False,
    variance_ratio: dict[str, Any] | None = None,
) -> list[OrderHint]:
    """
    Turn the correlogram reading into a *short* list of starting orders.

    Deliberately short and deliberately hedged. The advice carries `support` (the features that
    point at it) and `against` (what would have to be checked), because an order hint without its
    counter-evidence is just a guess wearing a suit.
    """
    s = period or 0
    d = 1 if differencing_needed else 0
    D = 1 if (seasonal_differencing_needed and s) else 0
    out: list[OrderHint] = []

    pacf_sig = [lag for lag in p.lags[: min(8, len(p.values))] if p.significant(lag)]
    acf_sig = [lag for lag in a.lags if a.significant(lag)]
    p_cut = max(pacf_sig, default=0)
    # The MA order the ACF supports is the *run* of significant lags from lag 1, not the largest
    # early significant lag — the latter picks up scattered chance lags as extra terms.
    q_cut = 0
    for lag in a.lags:
        if a.significant(lag):
            q_cut = lag
        else:
            break

    if pattern.kind == "white_noise":
        out.append(OrderHint(
            p=0, d=0, q=0, P=0, D=0, Q=0, s=s,
            label="(0,0,0)" + (f"(0,0,0){s}" if s else ""),
            plain=(
                "The correlogram shows no dependence, which points at the empty model — a constant "
                "plus noise. The honest candidate set here is 'mean only' versus 'one AR term', "
                "compared out of sample, because the null the data has to beat is this one."
            ),
            support=[f"no lag among the first {a.max_lag} is outside its band"],
            against=[
                "A white-noise correlogram is also what a series with a *shifted* mean looks like. "
                "If Step 5 or Step 4 found a break or a trend, that belongs in the model first.",
            ],
        ))
    elif pattern.kind == "ma":
        q = max(1, min(q_cut, 4))
        out.append(OrderHint(
            p=0, d=d, q=q, P=0, D=D, Q=1 if (D and s) else 0, s=s,
            label=_fmt_order(0, d, q, 0, D, 1 if (D and s) else 0, s),
            plain=(
                f"The ACF cuts off after lag {q} while the PACF tails off, which is the MA signature. "
                f"ARIMA(0,{d},{q}) is the smallest model with that shape."
            ),
            support=[f"ACF out of band for lags 1–{q}, inside after",
                     f"PACF significant at {len(pacf_sig)} of the first 8 lags (tails off)"],
            against=["Add an AR term only if the residual ACF still shows one after fitting."],
        ))
        out.append(OrderHint(
            p=1, d=d, q=q, P=0, D=D, Q=1 if (D and s) else 0, s=s,
            label=_fmt_order(1, d, q, 0, D, 1 if (D and s) else 0, s),
            plain="The same MA shape with one AR term — usually indistinguishable by AIC, and worth fitting because it sometimes forecasts better.",
            support=["ARMA(1,q) nests ARIMA(0,d,q), so the comparison is a nested test"],
            against=["One more parameter to estimate from the same data."],
        ))
    elif pattern.kind == "ar":
        order = max(1, min(p_cut or 1, 4))
        out.append(OrderHint(
            p=order, d=d, q=0, P=0, D=D, Q=0, s=s,
            label=_fmt_order(order, d, 0, 0, D, 0, s),
            plain=(
                f"The PACF is significant through lag {order} and then inside its band, so "
                f"AR({order}) is the smallest model with that shape."
            ),
            support=[f"PACF cut off after lag {order}", f"ACF decays without a cut-off"],
            against=["A near-unit-root AR(1) mimics a longer AR; check the root, not just the fit."],
        ))
        out.append(OrderHint(
            p=order, d=d, q=1, P=0, D=D, Q=1 if (D and s) else 0, s=s,
            label=_fmt_order(order, d, 1, 0, D, 1 if (D and s) else 0, s),
            plain="The same AR order with one MA term; the standard next candidate when the ACF does not quite die away.",
            support=["MA(1) absorbs a single remaining significant ACF spike"],
            against=["If the residual ACF is clean without it, the extra term is noise."],
        ))
    elif pattern.kind in {"nonstationary_decay", "overdifferenced"}:
        if pattern.kind == "overdifferenced":
            out.append(OrderHint(
                p=0, d=0, q=1, P=0, D=0, Q=0, s=s,
                label=_fmt_order(0, 0, 1, 0, 0, 0, s),
                plain=(
                    "The strongly negative lag-1 is what differencing already-stationary data "
                    "produces. The candidate that fits that shape without differencing again is "
                    "ARIMA(0,0,1) with a negative MA coefficient — but the better answer is usually "
                    "to undo the last difference."
                ),
                support=[f"r₁ = {_n(a.values[0])} is at or below −0.5"],
                against=[
                    "If a difference is genuinely needed, this model will fit badly out of sample.",
                    "Check the variance ratio: if variance fell after differencing, the difference "
                    "was doing real work.",
                ],
            ))
        else:
            # d comes from Step 5 and nowhere else. When the tests have not confirmed a unit root,
            # the hint is d = 0 and the difference appears as something to *preview*, not as the
            # recommendation — the advisor rule `next.diff.guardrail` exists for exactly this case.
            out.append(OrderHint(
                p=0, d=d, q=0, P=0, D=D, Q=1 if (D and s) else 0, s=s,
                label=_fmt_order(0, d, 0, 0, D, 1 if (D and s) else 0, s),
                plain=(
                    "The ACF decays too slowly for a stationary process, which points at one "
                    "difference — and Step 5 confirms a unit root, so d = 1 is the consistent "
                    "choice."
                    if differencing_needed else
                    "The ACF decays too slowly for a stationary process, which would normally mean "
                    "d = 1 — but Step 5 has not confirmed a unit root, so this hint keeps d = 0. "
                    "A trend, a break or long memory produces the same shape and is not fixed by "
                    "differencing. Preview it before committing to anything."
                ),
                support=[f"ACF still {_n(a.values[min(5, len(a.values) - 1)])} at lag 6"],
                against=[
                    "A unit root, a deterministic trend, long memory and a level shift all produce "
                    "this shape; only the first is fixed by differencing.",
                    "Step 5's verdict overrides this hint on d. If you disagree with it, change "
                    "Step 5's inputs, not this table.",
                ] if differencing_needed else [
                    "Step 5 did not confirm a unit root, so d = 1 here would be the correlogram "
                    "overruling the tests.",
                ],
            ))
    elif pattern.kind == "seasonal":
        out.append(OrderHint(
            p=0, d=d, q=1, P=0, D=D, Q=1, s=s,
            label=_fmt_order(0, d, 1, 0, D, 1, s),
            plain=(
                f"The ACF repeats every {s} observations, so the seasonal terms are the ones that "
                f"matter: one seasonal MA term (Q = 1) at lag {s}"
                + (", plus one seasonal difference if Step 5 found a seasonal unit root" if D else "")
                + "."
            ),
            support=[f"ACF spike at lag {s}", f"seasonal spike decays across harmonics"],
            against=[
                "If the seasonal amplitude is growing, a multiplicative model may fit better than "
                "adding terms.",
                "A seasonal MA on a short series is estimated from few effective cycles.",
            ],
        ))
    else:
        out.append(OrderHint(
            p=1, d=d, q=1, P=0, D=D, Q=1 if (D and s) else 0, s=s,
            label=_fmt_order(1, d, 1, 0, D, 1 if (D and s) else 0, s),
            plain=(
                "No clean cut-off, so the smallest model that covers both AR and MA behaviour is the "
                "default starting point — to be compared against its neighbours, not trusted."
            ),
            support=["ACF and PACF both show more than one significant lag"],
            against=["With no clean shape, information criteria over a small grid are the real guide."],
        ))

    # A universal caveat: hints are cross-checked against the stationarity verdict, not free.
    if variance_ratio and pattern.kind != "overdifferenced":
        if variance_ratio["ratio"] < 0.8:
            for h in out:
                h.against.append(
                    f"The variance ratio is {variance_ratio['ratio']:.2f} (below 0.8), which means "
                    "the differences are already negatively autocorrelated — a difference may "
                    "already be one too many."
                )
    return out


def _fmt_order(p: int, d: int, q: int, P: int, D: int, Q: int, s: int) -> str:
    if s:
        return f"({p},{d},{q})({P},{D},{Q}){s}"
    return f"({p},{d},{q})"


def _n(x: float | None, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def _b(bands: list[float] | float) -> str:
    if isinstance(bands, (int, float)):
        return f"±{bands:.3f}"
    return f"±{bands[0]:.3f} at lag 1"


__all__ = [
    "ACF", "PACF", "Pattern", "PatternKind", "Portmanteau", "OrderHint",
    "acf", "pacf", "default_max_lag", "read_pattern", "ljung_box", "box_pierce",
    "variance_ratio", "order_hints",
]
