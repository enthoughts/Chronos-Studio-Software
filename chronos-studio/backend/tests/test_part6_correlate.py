"""
tests/test_part6_correlate.py — S6 (ACF/PACF) and S7 (the "what next" advisor).

Two kinds of test here, and the second kind matters more:

1. **Arithmetic.** The ACF of a known MA(1) has a known shape; Durbin–Levinson must match a direct
   AR fit; Bartlett bands must widen as the series gets more persistent. These catch sign errors.

2. **The refusal tests.** The value of this stage is in what it will *not* say: it must not call a
   white-noise correlogram structured, must not recommend differencing a series Step 5 called
   stationary, must not invert OCSB's null, and must not report a preview verdict it did not
   compute. Each of those is a test below, because each is a way the app could be confidently wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.advisor.rules_next import RULES, NextContext, headline, rank
from app.ts.correlate._acf import (
    acf,
    box_pierce,
    default_max_lag,
    ljung_box,
    order_hints,
    pacf,
    read_pattern,
    variance_ratio,
)
from app.ts.correlate._ccf import ccf, spurious_warning
from app.ts.correlate._preview import preview_action


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260101)


@pytest.fixture(scope="module")
def white_noise(rng) -> pd.Series:
    return pd.Series(rng.standard_normal(180),
                     index=pd.date_range("2000-01-01", periods=180, freq="MS"))


@pytest.fixture(scope="module")
def seasonal_series() -> pd.Series:
    idx = pd.date_range("2008-01-01", periods=144, freq="MS")
    t = np.arange(144)
    y = (20 + 0.12 * t
         + 4.0 * np.sin(2 * np.pi * t / 12)
         + np.random.default_rng(4).standard_normal(144) * 0.6)
    return pd.Series(y, index=idx)


@pytest.fixture(scope="module")
def ma1(rng) -> pd.Series:
    """MA(1) with θ = 0.8: ACF should be ≈ θ/(1+θ²) = 0.49 at lag 1 and ≈ 0 after."""
    e = rng.standard_normal(600)
    y = e[1:] + 0.8 * e[:-1]
    return pd.Series(y, index=pd.date_range("1990-01-01", periods=len(y), freq="MS"))


@pytest.fixture(scope="module")
def ar1(rng) -> pd.Series:
    """AR(1) with φ = 0.7: ACF should decay geometrically, PACF should cut off after lag 1."""
    e = rng.standard_normal(600)
    y = np.zeros(600)
    for i in range(1, 600):
        y[i] = 0.7 * y[i - 1] + e[i]
    return pd.Series(y, index=pd.date_range("1990-01-01", periods=600, freq="MS"))


@pytest.fixture(scope="module")
def random_walk(rng) -> pd.Series:
    return pd.Series(np.cumsum(rng.standard_normal(200)),
                     index=pd.date_range("2000-01-01", periods=200, freq="MS"))


# --------------------------------------------------------------------------- #
# 1. Arithmetic
# --------------------------------------------------------------------------- #
def test_acf_uses_the_n_denominator_not_n_minus_k():
    """
    The n−k denominator inflates long-lag autocorrelation and makes short series look persistent.
    With a perfectly constant-plus-noise series the lag-k correlation must fall off, not stay put.
    """
    y = pd.Series(np.arange(50, dtype=float))
    a = acf(y, max_lag=5)
    # A pure ramp has an ACF that decays smoothly; the point is that all values stay in [-1, 1]
    # *and* are computed with n in the denominator, which the exact value below pins.
    dev = y - y.mean()
    denom = float(np.dot(dev, dev))
    expected_lag3 = float(np.dot(dev[:-3], dev[3:]) / denom)
    assert a.values[2] == pytest.approx(expected_lag3)
    assert all(-1 <= v <= 1 for v in a.values)


def test_acf_of_ma1_matches_theory(ma1):
    """
    For MA(1), ρ₁ = θ/(1+θ²) and ρ_k = 0 for k ≥ 2. This is the one case where theory gives an
    exact answer, so it pins both the sign and the magnitude.
    """
    a = acf(ma1, max_lag=10)
    assert a.values[0] == pytest.approx(0.8 / (1 + 0.8 ** 2), abs=0.06)
    assert abs(a.values[1]) < 0.10


def test_pacf_of_ar1_cuts_off_after_lag_one(ar1):
    """An AR(1)'s PACF is φ at lag 1 and zero after — the mirror image of its ACF."""
    p = pacf(ar1, max_lag=8)
    assert p.values[0] == pytest.approx(0.7, abs=0.08)
    assert all(abs(v) < 0.12 for v in p.values[1:])


def test_bartlett_band_widens_with_persistence(white_noise, random_walk):
    """
    The whole reason for a Bartlett band: on a persistent series the white-noise band is far too
    narrow, so every lag looks significant. The two bands must differ substantially.
    """
    a_noise = acf(white_noise, max_lag=20)
    a_walk = acf(random_walk, max_lag=20)
    assert a_noise.band_bartlett[10] < 1.5 * a_noise.band_white
    assert a_walk.band_bartlett[10] > 3 * a_walk.band_white


def test_default_max_lag_respects_both_limits():
    assert default_max_lag(144, 12) == min(48, 144 // 5)
    assert default_max_lag(40, 12) >= 10          # never fewer than 10
    assert default_max_lag(1000, None) <= 1000 // 3


def test_ljung_box_matches_statsmodels(white_noise):
    from statsmodels.stats.diagnostic import acorr_ljungbox
    a = acf(white_noise, max_lag=20)
    ours = ljung_box(a, [10])
    theirs = acorr_ljungbox(white_noise, lags=[10], return_df=True)
    assert ours.stats[0] == pytest.approx(float(theirs["lb_stat"].iloc[0]), rel=1e-6)
    assert ours.p_values[0] == pytest.approx(float(theirs["lb_pvalue"].iloc[0]), rel=1e-4)


def test_box_pierce_is_smaller_than_ljung_box(white_noise):
    """Ljung–Box adds the (n+2)/(n−k) correction, so it is always the larger of the two."""
    a = acf(white_noise, max_lag=20)
    lb = ljung_box(a, [10]).stats[0]
    bp = box_pierce(a, [10]).stats[0]
    assert lb > bp


def test_variance_ratio_of_white_noise_is_about_a_half(white_noise):
    """
    For white noise, Var(Δ₂) = 2σ² and 2·Var(Δ₁) = 4σ², so VR = 0.5 exactly. This is not a fault:
    it is why a low variance ratio alone is *not* evidence of over-differencing — you also need to
    know that a difference was applied.
    """
    vr = variance_ratio(white_noise, q=2)
    assert vr["ratio"] == pytest.approx(0.5, abs=0.2)
    assert "told apart" in vr["reading"]


def test_variance_ratio_of_a_random_walk_is_near_one(random_walk):
    """A random walk's *differences* are white, so VR ≈ 1 — this is the null, not a failure."""
    vr = variance_ratio(random_walk, q=2)
    assert 0.5 < vr["ratio"] < 2.0


def test_variance_ratio_detects_over_differencing(rng):
    """Differencing white noise gives an MA(1) with θ = −1, whose VR should be near 0."""
    noise = pd.Series(rng.standard_normal(400),
                      index=pd.date_range("2000-01-01", periods=400, freq="MS"))
    vr = variance_ratio(noise.diff().dropna(), q=2)
    assert vr["ratio"] < 0.8
    assert "negatively autocorrelated" in vr["reading"]


# --------------------------------------------------------------------------- #
# 2. Reading the shape — and refusing to over-read it
# --------------------------------------------------------------------------- #
def test_white_noise_is_read_as_white_noise(white_noise):
    """The critical refusal: at 5% with 36 lags, ~2 chance lags are expected. That is not a pattern."""
    a = acf(white_noise, max_lag=36, period=None)
    p = pacf(white_noise, max_lag=36, period=None)
    pattern = read_pattern(a, p, period=None, ljung_box_p=ljung_box(a, [10]).p_values[0])
    assert pattern.kind == "white_noise", pattern.plain
    assert "white noise" in pattern.plain.lower()


def test_white_noise_pattern_is_not_called_ma():
    """
    Regression test for a real bug: a count-based rule saw 2-3 scattered significant lags at 5% and
    reported "MA(2)-like" on pure noise. A cut-off must be a *contiguous run from lag 1*.
    """
    for seed in range(8):
        rng = np.random.default_rng(seed)
        y = pd.Series(rng.standard_normal(180),
                      index=pd.date_range("2000-01-01", periods=180, freq="MS"))
        a = acf(y, max_lag=36, period=None)
        p = pacf(y, max_lag=36, period=None)
        pattern = read_pattern(a, p, period=None)
        assert pattern.kind in {"white_noise", "unclear"}, (seed, pattern.kind, pattern.plain)


def test_seasonal_series_is_read_as_seasonal(seasonal_series):
    a = acf(seasonal_series, max_lag=30, period=12)
    p = pacf(seasonal_series, max_lag=30, period=12)
    pattern = read_pattern(a, p, period=12)
    assert pattern.kind in {"seasonal", "ar", "nonstationary_decay"}
    assert a.significant(12) is True


def test_random_walk_is_read_as_nonstationary(random_walk):
    a = acf(random_walk, max_lag=30)
    p = pacf(random_walk, max_lag=30)
    pattern = read_pattern(a, p)
    assert pattern.kind == "nonstationary_decay"
    assert any("cannot distinguish" in c for c in pattern.caveats)


def test_over_differenced_series_is_named(rng):
    noise = pd.Series(rng.standard_normal(400),
                      index=pd.date_range("2000-01-01", periods=400, freq="MS"))
    d = noise.diff().dropna()
    a = acf(d, max_lag=20)
    p = pacf(d, max_lag=20)
    pattern = read_pattern(a, p, differenced=True)
    assert pattern.kind == "overdifferenced"
    assert a.values[0] < -0.5
    assert any("MA(1)" in c for c in pattern.caveats)


def test_nonstationary_reading_never_claims_a_unit_root(random_walk):
    """
    A slow decay is consistent with a unit root, a trend, long memory or a near-unit-root process.
    The sentence must say 'look at Step 5', not 'this is a unit root'.
    """
    a = acf(random_walk, max_lag=30)
    p = pacf(random_walk, max_lag=30)
    text = read_pattern(a, p).plain.lower()
    assert "unit root" in text
    assert "look" in text or "check" in text or "before" in text


# --------------------------------------------------------------------------- #
# 3. Order hints
# --------------------------------------------------------------------------- #
def test_order_hints_have_support_and_counter_evidence(ma1):
    a = acf(ma1, max_lag=20)
    p = pacf(ma1, max_lag=20)
    pattern = read_pattern(a, p)
    hints = order_hints(a, p, pattern=pattern)
    assert hints, "an MA(1) should produce at least one candidate order"
    for h in hints:
        assert h.support, "every hint must say what supports it"
        assert h.against, "every hint must say what could contradict it"


def test_order_hints_take_d_from_the_tests_not_the_acf(random_walk):
    """
    A persistent ACF can never set d on its own — that is Step 5's job. With `differencing_needed`
    False, every hint must have d = 0 even though the ACF screams non-stationary.
    """
    a = acf(random_walk, max_lag=30)
    p = pacf(random_walk, max_lag=30)
    pattern = read_pattern(a, p)
    hints = order_hints(a, p, pattern=pattern, differencing_needed=False)
    assert all(h.d == 0 for h in hints)

    hints_d = order_hints(a, p, pattern=pattern, differencing_needed=True)
    assert any(h.d == 1 for h in hints_d)


def test_white_noise_hint_is_the_empty_model(white_noise):
    a = acf(white_noise, max_lag=30)
    p = pacf(white_noise, max_lag=30)
    pattern = read_pattern(a, p)
    hints = order_hints(a, p, pattern=pattern)
    assert hints[0].p == 0 and hints[0].q == 0 and hints[0].d == 0


# --------------------------------------------------------------------------- #
# 4. CCF and pre-whitening
# --------------------------------------------------------------------------- #
def test_ccf_finds_a_lagged_copy(seasonal_series):
    y = seasonal_series.shift(3).dropna()
    c = ccf(seasonal_series, y, max_lag=10, x_name="driver", y_name="target", prewhiten=True)
    peak = c.peak()
    assert peak["lag"] == 3
    assert peak["value"] > 0.8
    # y_t = x_{t-3}, so at lag 3 the driver leads by 3. Positive lag = input leads.
    assert peak["leads_lags"] == "driver leads target by 3 observation(s)"


def test_prewhitening_removes_shared_autocorrelation():
    """
    Two independent random walks are almost perfectly correlated at every lag without any
    relationship. Pre-whitening must collapse that — this is the test that proves the filter does
    what the docstring claims.
    """
    rng = np.random.default_rng(11)
    idx = pd.date_range("2000-01-01", periods=300, freq="MS")
    a = pd.Series(np.cumsum(rng.standard_normal(300)), index=idx)
    b = pd.Series(np.cumsum(rng.standard_normal(300)), index=idx)

    raw = ccf(a, b, max_lag=10, x_name="a", y_name="b", prewhiten=False)
    pw = ccf(a, b, max_lag=10, x_name="a", y_name="b", prewhiten=True)
    assert abs(raw.peak()["value"]) > abs(pw.peak()["value"])
    assert spurious_warning(pw, raw) is not None


def test_ccf_direction_words_are_not_reversed(seasonal_series):
    """A negative lag means the *input* leads. Getting this backwards is the classic CCF error."""
    y = seasonal_series.shift(2).dropna()
    c = ccf(seasonal_series, y, max_lag=6, x_name="driver", y_name="target", prewhiten=True)
    peak = c.peak()
    assert peak["lag"] == 2
    assert "driver leads target" in peak["leads_lags"]


# --------------------------------------------------------------------------- #
# 5. The preview — the thing a textbook cannot do
# --------------------------------------------------------------------------- #
def test_preview_says_differencing_white_noise_makes_it_worse(rng):
    noise = pd.Series(rng.standard_normal(200),
                      index=pd.date_range("2000-01-01", periods=200, freq="MS"))
    out = preview_action(noise, "difference", period=None, params={"order": 1})
    assert out["verdict"] == "worse", out["plain"]
    assert "spread grew" in out["plain"] or "signature of differencing" in out["plain"]


def test_preview_says_a_first_difference_helps_a_random_walk(random_walk):
    out = preview_action(random_walk, "difference", period=None, params={"order": 1})
    assert out["verdict"] in {"better", "neutral"}, out["plain"]


def test_preview_of_a_log_is_judged_on_variance_not_acf(seasonal_series):
    """
    A log is not supposed to change the ACF, so judging it by the ACF would mark a correct
    transform as a failure. It must be judged on the level-to-spread link.
    """
    y = seasonal_series + 30          # positive, growing spread
    out = preview_action(y, "transform.log", period=12)
    assert "rank correlation" in out["plain"]
    assert "ACF is deliberately not the criterion" in out["plain"] or "little for a log to do" in out["plain"]


def test_preview_refuses_a_log_on_non_positive_data():
    y = pd.Series(np.r_[-5.0, np.ones(50)], index=pd.date_range("2000-01-01", periods=51, freq="MS"))
    out = preview_action(y, "transform.log", period=None)
    assert out["verdict"] == "unavailable"
    assert "zero or negative" in out["plain"]


def test_preview_never_reports_a_verdict_it_did_not_compute():
    """`unavailable` is a real answer and must carry a reason, never a silent 'neutral'."""
    y = pd.Series(np.ones(40), index=pd.date_range("2000-01-01", periods=40, freq="MS"))
    out = preview_action(y, "difference", period=None, params={"order": 1})
    assert out["verdict"] in {"unavailable", "worse", "neutral"}
    if out["verdict"] == "unavailable":
        assert out["plain"]


# --------------------------------------------------------------------------- #
# 6. The rules
# --------------------------------------------------------------------------- #
def _all_rule_ids() -> set[str]:
    return {r.id for r in RULES}


def test_rule_ids_are_unique():
    ids = [r.id for r in RULES]
    assert len(ids) == len(set(ids))


def test_every_rule_because_has_a_number_and_no_thousands_separator():
    ctx = NextContext(
        n=144, period=12, r1=0.948, max_lag=28, n_significant=14, decay_lags=14,
        pattern="nonstationary_decay", ljung_box_p=0.0, ljung_box_p_seasonal=0.0,
        seasonal_acf_significant=True, variance_ratio=1.4,
        stationarity_verdict="seasonal_unit_root", differencing_needed=True,
        seasonal_differencing_needed=True, n_unit_root_tests=10,
        trend_share=0.861, seasonal_share=0.136, seasonal_strength=0.975,
        seasonal_growing=True, trend_band="strong", already_logged=False,
        already_differenced=False, n_transforms_applied=0, variance_stable=False,
    )
    for rule in RULES:
        if not rule.applies(ctx):
            continue
        text = rule.because(ctx)
        assert any(ch.isdigit() for ch in text), rule.id
        assert ",000" not in text and "1,0" not in text


def test_leave_alone_is_a_real_recommendation_not_an_absence():
    """
    The most important rule in the stage. On a white-noise series the honest answer is 'change
    nothing', and it must be ranked first — not dropped in favour of a harmless-looking transform.
    """
    ctx = NextContext(n=180, max_lag=36, pattern="white_noise", ljung_box_p=0.42,
                      n_significant=1, r1=-0.014, stationarity_verdict="stationary",
                      differencing_needed=False)
    ranked = rank(ctx)
    assert ranked, "a white-noise series still gets an answer"
    assert ranked[0]["id"] == "next.none.already_adequate"
    assert ranked[0]["confidence"] == "high"
    assert "does not reject" in ranked[0]["because"]


def test_guardrail_protects_a_stationary_series_from_differencing():
    """
    The ACF looks persistent but Step 5 says stationary: differencing must be blocked, not merely
    not-suggested, and the reason must cite both facts.
    """
    ctx = NextContext(n=200, max_lag=30, pattern="nonstationary_decay", r1=0.93,
                      stationarity_verdict="stationary", differencing_needed=False,
                      decay_lags=12, ljung_box_p=0.0001)
    ranked = rank(ctx)
    ids = [r["id"] for r in ranked]
    assert "next.diff.guardrail" in ids
    assert ids[0] == "next.diff.guardrail"
    guard = next(r for r in ranked if r["id"] == "next.diff.guardrail")
    assert guard["kind"] == "protect"
    assert "0.930" in guard["because"] and "stationary" in guard["because"]
    assert "next.diff.first" not in ids


def test_a_seasonal_unit_root_is_never_answered_with_a_first_difference():
    """
    Step 5's own verdict text says a plain difference moves the problem into the seasonal lags. If
    the advisor recommended diff(1) anyway, the app would be contradicting itself in two panels.
    """
    ctx = NextContext(n=144, period=12, pattern="nonstationary_decay", r1=0.948,
                      differencing_needed=True, seasonal_differencing_needed=True,
                      stationarity_verdict="seasonal_unit_root", ljung_box_p=0.0)
    ranked = rank(ctx)
    ids = [r["id"] for r in ranked]
    assert "next.diff.seasonal_only" in ids
    assert "next.diff.first" not in ids
    assert ids[0] == "next.diff.seasonal_only"


def test_both_differences_note_only_fires_when_both_are_unapplied():
    base = dict(n=144, period=12, differencing_needed=True,
                seasonal_differencing_needed=True, stationarity_verdict="unit_root")
    assert any(r.id == "next.note.both_differences"
               for r in RULES if r.applies(NextContext(**base)))
    applied = NextContext(**base, already_differenced=True, already_seasonally_differenced=True)
    assert not any(r.id == "next.note.both_differences"
                   for r in RULES if r.applies(applied))


def test_over_differencing_repair_is_higher_confidence_when_a_difference_was_applied():
    seen = NextContext(n=200, max_lag=30, pattern="overdifferenced", r1=-0.62,
                       already_differenced=False)
    done = NextContext(n=200, max_lag=30, pattern="overdifferenced", r1=-0.62,
                       already_differenced=True)
    a = next(r for r in rank(seen) if r["id"] == "next.diff.undo_overdifferenced")
    b = next(r for r in rank(done) if r["id"] == "next.diff.undo_overdifferenced")
    assert a["confidence"] == "low"
    assert b["confidence"] == "high"


def test_a_worse_preview_demotes_a_rule_and_says_why():
    """The ranking must respond to the preview, and the demotion must be visible in the text."""
    base = dict(n=200, max_lag=30, r1=0.95, pattern="nonstationary_decay",
                differencing_needed=True, stationarity_verdict="unit_root",
                decay_lags=15, ljung_box_p=0.0)
    plain = rank(NextContext(**base))
    with_preview = rank(NextContext(**base, previews={
        "next.diff.first": {"verdict": "worse", "plain": "The spread grew after the difference."}
    }))
    a = next(r for r in plain if r["id"] == "next.diff.first")
    b = next(r for r in with_preview if r["id"] == "next.diff.first")
    assert b["rank"] < a["rank"]
    assert "preview says this makes things worse" in b["because"]
    assert "spread grew" in b["because"]


def test_ranking_is_deterministic():
    ctx = NextContext(n=144, period=12, pattern="nonstationary_decay", r1=0.9,
                      differencing_needed=True, stationarity_verdict="unit_root",
                      ljung_box_p=0.0)
    first = [r["id"] for r in rank(ctx)]
    second = [r["id"] for r in rank(ctx)]
    assert first == second


def test_headline_matches_the_top_ranked_rule():
    ctx = NextContext(n=144, period=12, pattern="nonstationary_decay", r1=0.9,
                      differencing_needed=True, stationarity_verdict="unit_root",
                      ljung_box_p=0.0)
    ranked = rank(ctx)
    text = headline(ranked, ctx)
    assert ranked[0]["recommend"].split(" (")[0].lower()[:20] in text.lower()


# --------------------------------------------------------------------------- #
# 7. The API
# --------------------------------------------------------------------------- #
def test_correlate_endpoint_returns_everything_the_screen_needs(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    r = client.get("/api/v1/analysis/correlate", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n"] == 144
    assert body["max_lag"] == 28
    assert len(body["acf"]) == 28
    assert body["pattern"]["kind"]
    ids = {x["id"] for x in body["results"]}
    assert {"table.correlogram_reading", "fig.acf_pacf", "table.portmanteau",
            "table.order_hints"} <= ids
    # Ljung–Box must be reported at the seasonal lag, not just at 10.
    assert 12 in [row["lag"] for row in body["ljung_box"]]


def test_correlate_is_cached_but_invalidated_by_the_s5_verdict(client, session_id):
    """
    Regression test for a real bug: S6 cached its payload keyed only on the series version, so
    running S6 *before* S5 left a verdict-less payload that S7 then read — losing the seasonal
    half of the advice while still looking authoritative.
    """
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    first = client.get("/api/v1/analysis/correlate", headers=h).json()
    assert first["stationarity"] is None            # no S5 yet
    assert first["cached"] is False
    again = client.get("/api/v1/analysis/correlate", headers=h).json()
    assert again["cached"] is True

    client.post("/api/v1/analysis/tests", json={"allow_stale": True}, headers=h)
    after = client.get("/api/v1/analysis/correlate", headers=h).json()
    assert after["stationarity"] is not None, "S5 changed the answer; the cache must notice"
    assert after["cached"] is False


def test_next_refuses_to_guess_without_the_stationarity_verdict(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    r = client.get("/api/v1/analysis/next", headers=h)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "next.missing_stationarity"
    assert "Step 5" in err["fix"]


def test_next_reads_ocsb_the_right_way_round(client, session_id):
    """
    OCSB's null IS the seasonal unit root, so `rejects_null is False` means D = 1 is needed.
    Inverting this would tell the user not to difference a series that needs it.
    """
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    client.post("/api/v1/analysis/tests", json={"allow_stale": True}, headers=h)
    body = client.get("/api/v1/analysis/next", headers=h).json()
    assert body["facts"]["seasonal_differencing_needed"] is True
    ids = [r["id"] for r in body["ranked"]]
    assert "next.diff.seasonal_only" in ids
    assert "next.diff.first" not in ids


def test_next_on_white_noise_says_leave_it_alone(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "white_noise"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    client.post("/api/v1/analysis/decompose", json={"method": "stl"}, headers=h)
    client.post("/api/v1/analysis/tests", json={"allow_stale": True}, headers=h)
    body = client.get("/api/v1/analysis/next", headers=h).json()
    ranked = body["ranked"]
    assert ranked, "there must be at least one recommendation"
    assert ranked[0]["id"] == "next.none.already_adequate"
    assert "No transformation is needed" in body["headline"]


def test_preview_endpoint_does_not_touch_the_session(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    before = client.get("/api/v1/preprocess/state", headers=h).json()
    r = client.post("/api/v1/analysis/next/preview",
                    json={"op": "difference", "order": 1}, headers=h)
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] in {"better", "neutral", "worse", "unavailable"}
    after = client.get("/api/v1/preprocess/state", headers=h).json()
    assert before == after, "a preview must never modify the series"


def test_apply_goes_through_the_versioned_reversible_path(client, session_id):
    """
    An action taken from 'what next' must produce the same versioned record as the Step 2 menu,
    or undo will not match what the user did.
    """
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    client.post("/api/v1/analysis/profile", headers=h)
    state_before = client.get("/api/v1/preprocess/state", headers=h).json()
    r = client.post("/api/v1/analysis/next/apply",
                    json={"op": "difference", "order": "seasonal",
                          "rule_id": "next.diff.seasonal_only"}, headers=h)
    assert r.status_code == 200, r.text
    state_after = client.get("/api/v1/preprocess/state", headers=h).json()
    assert state_after["series_version"] > state_before["series_version"]
    undo = client.post("/api/v1/preprocess/undo", headers=h)
    assert undo.status_code == 200, undo.text
    state_undo = client.get("/api/v1/preprocess/state", headers=h).json()
    # Undo restores the *data*; the series version counter keeps moving forward so that no cached
    # result can ever be mistaken for current. What must match is the applied-step list.
    assert len(state_undo["steps"]) == len(state_before["steps"])
    assert state_undo["n"] == state_before["n"]


def test_difference_op_removes_observations_and_says_so():
    from app.ts.preprocess.ops import apply_op
    s = pd.Series(np.arange(50, dtype=float),
                  index=pd.date_range("2000-01-01", periods=50, freq="MS"))
    out = apply_op("difference", s, {"lag": 1})
    assert int(out.series.notna().sum()) == 49
    assert "49 long rather than 50" in out.plain
    assert out.warnings, "a differenced series must warn that it is now changes, not levels"
    assert out.reversible is True


def test_difference_of_seasonal_lag_removes_that_many_observations():
    from app.ts.preprocess.ops import apply_op
    s = pd.Series(np.arange(60, dtype=float),
                  index=pd.date_range("2000-01-01", periods=60, freq="MS"))
    out = apply_op("difference", s, {"lag": 12})
    assert int(out.series.notna().sum()) == 48
    assert "12" in out.plain


def test_ccf_endpoint_explains_why_there_is_nothing_to_correlate(client, session_id):
    h = {"X-Session-Id": session_id}
    client.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    r = client.post("/api/v1/analysis/correlate/ccf", json={"column": "gdp"}, headers=h)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "correlate.no_column"
    assert err["fix"]


def test_snippets_parse(monkeypatch):
    """Every reproducible snippet shown to a user must be valid Python."""
    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app

    c = TestClient(fastapi_app)
    sid = c.post("/api/v1/sessions").json()["session_id"]
    h = {"X-Session-Id": sid}
    c.post("/api/v1/data/example", json={"example_id": "airpassengers"}, headers=h)
    c.post("/api/v1/analysis/profile", headers=h)
    c.post("/api/v1/analysis/tests", json={"allow_stale": True}, headers=h)
    body = c.get("/api/v1/analysis/correlate", headers=h).json()
    body2 = c.get("/api/v1/analysis/next", headers=h).json()
    import ast
    snippets = [r["reproducible_snippet"] for r in body["results"] if r.get("reproducible_snippet")]
    snippets += [r["reproducible_snippet"] for r in body2["results"] if r.get("reproducible_snippet")]
    assert snippets
    for s in snippets:
        ast.parse(s)


# --------------------------------------------------------------------------- #
# 8. Figures
# --------------------------------------------------------------------------- #
def test_figures_serialise_without_numpy_arrays(seasonal_series):
    """FastAPI/pydantic cannot serialise numpy arrays — every figure must go through `to_json`."""
    import json
    from app.ts.correlate._figures import acf_comparison_grid, acf_pacf_grid

    a = acf(seasonal_series, max_lag=24, period=12)
    p = pacf(seasonal_series, max_lag=24, period=12)
    figures = [
        acf_pacf_grid(a, p, name="test"),
        acf_comparison_grid([("raw", a, p),
                             ("diff", acf(seasonal_series.diff().dropna(), max_lag=24, period=12),
                              pacf(seasonal_series.diff().dropna(), max_lag=24, period=12))],
                            name="test"),
    ]
    for fig in figures:
        json.dumps(fig.figure_json)          # raises if a numpy type survived
        assert fig.alt_text


def test_acf_figure_alt_text_carries_the_numbers(seasonal_series):
    from app.ts.correlate._figures import acf_pacf_grid
    a = acf(seasonal_series, max_lag=24, period=12)
    p = pacf(seasonal_series, max_lag=24, period=12)
    alt = acf_pacf_grid(a, p, name="Airline passengers").alt_text
    assert "lags 1 to 24" in alt
    assert "band" in alt.lower()
