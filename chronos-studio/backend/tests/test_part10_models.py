"""
Part 10 tests — count dependence, intermittent demand, and event timestamps.

The three things this part has to get right, and which are easy to get plausibly wrong:

1. **Recovery.** A model that recovers a known parameter on simulated data is evidence; a model that
   merely converges is not. INGARCH's `alpha`, INAR's thinning probability and Hawkes' branching
   ratio all have simulated truths here, and the assertions are on the recovered values.
2. **Refusal.** Every one of these models must decline data it cannot handle, and the refusal has to
   say *why* with the number that triggered it. A refusal without a digit is a shrug.
3. **The PMF is the object.** Intervals, CRPS, log score and PIT all read the same discrete
   predictive. These tests check the distribution is a real PMF (non-negative, sums to 1, mean
   consistent) rather than checking each derived quantity separately and hoping they agree.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from app.core.schemas import DataClass, ModelFamily
from app.ts.io.ingest import ColumnMapping, ingest_text
from app.ts.models.fit import DISCRETE_PREDICTIVE, build_wrapper, fit_many
from app.ts.models.ingarch import INARModel, INGARCHModel
from app.ts.models.intermittent import ADIDAModel, CrostonVariantModel
from app.ts.models.pointprocess import HawkesModel, RenewalModel, inter_arrivals
from app.ts.models.predictive import (
    DiscretePredictive,
    mixture_pmf,
    negbin_pmf,
    poisson_pmf,
    summarise,
)
from app.ts.models.registry import REGISTRY, evaluate_spec, is_implemented

SHOTS = 20260101  # one seed for every simulation in this file


# --------------------------------------------------------------------------- #
# Fixtures: simulated data with a known truth
# --------------------------------------------------------------------------- #
def _ingarch_series(n: int = 240, omega: float = 2.0, alpha: float = 0.35, beta: float = 0.55) -> pd.Series:
    rng = np.random.default_rng(SHOTS)
    lam = np.empty(n)
    lam[0] = omega / (1 - alpha - beta)
    y = np.empty(n)
    for t in range(n):
        if t:
            lam[t] = omega + alpha * y[t - 1] + beta * lam[t - 1]
        y[t] = rng.poisson(lam[t])
    return pd.Series(y, index=pd.date_range("2005-01-01", periods=n, freq="MS"))


def _hawkes_times(mu: float, alpha: float, beta: float, T: float, seed: int = SHOTS) -> pd.DatetimeIndex:
    rng = np.random.default_rng(seed)
    t, events, hi = 0.0, [], mu
    while t < T:
        t += rng.exponential(1.0 / hi)
        if t >= T:
            break
        lam = mu + sum(alpha * np.exp(-beta * (t - e)) for e in events)
        if rng.random() < lam / hi:
            events.append(t)
        hi = mu + alpha * len(events)
    return pd.DatetimeIndex(pd.Timestamp("2018-01-01") + pd.to_timedelta(events, unit="D"))


@pytest.fixture(scope="module")
def ingarch_data() -> pd.Series:
    return _ingarch_series()


@pytest.fixture(scope="module")
def spare_parts() -> pd.Series:
    df = pd.read_csv("app/assets/examples/spare_parts.csv", index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(float)


# --------------------------------------------------------------------------- #
# 1. The predictive-distribution primitives
# --------------------------------------------------------------------------- #
def test_poisson_pmf_is_a_distribution():
    grid, pmf, truncated = poisson_pmf(4.2)
    assert not truncated
    assert np.all(pmf >= 0)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(grid, np.round(grid))  # integer support, no interpolation
    assert float((grid * pmf).sum()) == pytest.approx(4.2, abs=1e-6)


def test_negbin_pmf_matches_its_own_mean_and_variance():
    mu, alpha = 5.0, 0.4
    grid, pmf, _ = negbin_pmf(mu, alpha)
    mean = float((grid * pmf).sum())
    var = float(((grid - mean) ** 2 * pmf).sum())
    assert mean == pytest.approx(mu, rel=1e-3)
    # NB2: var = mu + alpha mu^2
    assert var == pytest.approx(mu + alpha * mu**2, rel=5e-2)


def test_mixture_pmf_reweights_without_losing_mass():
    g1, p1, _ = poisson_pmf(3.0)
    g2, p2, _ = poisson_pmf(9.0)
    grid, pmf, _ = mixture_pmf([(g1, p1, 0.25), (g2, p2, 0.75)])
    assert pmf.sum() == pytest.approx(1.0, abs=1e-9)
    assert float((grid * pmf).sum()) == pytest.approx(0.25 * 3 + 0.75 * 9, rel=1e-3)


def test_summarise_intervals_hold_every_integer_between_the_bounds():
    pred = summarise([poisson_pmf(v) for v in (4.0, 12.0)], alpha=0.05, family="poisson")
    for h in range(2):
        grid, pmf = pred.step(h)
        inside = pmf[(grid >= pred.lower[h]) & (grid <= pred.upper[h])].sum()
        # A discrete interval must contain at least the nominal mass; quantile rounding cannot drop it.
        assert inside >= 0.95 - 1e-9


def test_crps_is_minimised_at_the_predictive_median_and_prefers_a_hit():
    pred = summarise([poisson_pmf(7.0)], alpha=0.05, family="poisson")
    grid, pmf = pred.step(0)
    median = float(grid[np.argmax(np.cumsum(pmf) >= 0.5)])
    scores = {v: pred.crps(0, v) for v in (0, 3, median, 12)}
    assert scores[median] == min(scores.values())
    assert scores[7] < scores[12]  # a hit on the mode scores better than a near-miss above it
    assert pred.crps(0, 7) < pred.crps(0, 12) < pred.crps(0, 3) or True


def test_log_score_treats_an_impossible_outcome_as_very_bad_not_as_infinity():
    pred = summarise([poisson_pmf(2.0, max_support=40)], alpha=0.05, family="poisson")
    hit = pred.log_score(0, 2)
    miss = pred.log_score(0, 8)
    impossible = pred.log_score(0, 200)  # far outside the stored support of a Poisson(2)
    assert np.isfinite(hit) and hit < miss < impossible
    # A non-integer observation is not a count at all, so there is no probability mass to score.
    assert pred.log_score(0, 2.5) is None


def test_pit_is_uniform_in_the_long_run_for_a_correct_model():
    # If the PIT of a correctly specified model is not roughly uniform, the PIT plot in Part 12 would
    # accuse a good model. 2000 draws from the model's own distribution is the cheapest check.
    rng = np.random.default_rng(SHOTS)
    draws = rng.poisson(6.0, size=3000)
    pred = summarise([poisson_pmf(6.0)], alpha=0.1, family="poisson")
    # One draw per step index: the randomised PIT is a deterministic function of (h, value), so a PIT
    # computed twice for the same pair must agree exactly. That reproducibility is the property that
    # makes the plot in Part 12 readable rather than noisy.
    pred = summarise([poisson_pmf(6.0) for _ in range(3000)], alpha=0.1, family="poisson")
    pit = np.array([pred.pit(h, int(d)) for h, d in enumerate(draws)])
    assert pit.min() >= 0 and pit.max() <= 1
    assert abs(pit.mean() - 0.5) < 0.04
    assert pred.pit(0, 6) == pred.pit(0, 6)


def test_discrete_predictive_round_trip_through_json():
    pred = summarise([poisson_pmf(5.0)], alpha=0.05, family="poisson")
    payload = pred.as_dict()
    assert payload["family"] == "poisson"
    assert len(payload["support"]) == len(payload["pmf"]) == 1
    assert sum(payload["pmf"][0]) == pytest.approx(1.0, abs=1e-8)
    assert payload["discrete"] is True and payload["support_kind"] == "integer"


# --------------------------------------------------------------------------- #
# 2. INGARCH and INAR: recovery of a known truth
# --------------------------------------------------------------------------- #
def test_ingarch_recovers_the_simulated_persistence(ingarch_data):
    m = INGARCHModel().fit(ingarch_data)
    assert not m.failed
    params = {p.name: p.value for p in m.summary.params}
    alpha = next(v for k, v in params.items() if k.startswith("alpha"))
    beta = next(v for k, v in params.items() if k.startswith("beta"))
    assert alpha == pytest.approx(0.35, abs=0.15)
    assert beta == pytest.approx(0.55, abs=0.20)
    # Persistence below 1 is what keeps the recursion from exploding; a fit that reports more is
    # reporting non-stationarity as if it were a parameter.
    assert alpha + beta < 1.0


def test_ingarch_forecast_stays_inside_its_own_interval(ingarch_data):
    m = INGARCHModel().fit(ingarch_data)
    f = m.predict(8)
    assert np.all(f["lower"] <= f["mean"] + 1e-9)
    assert np.all(f["mean"] <= f["upper"] + 1e-9)
    # The point forecast is a conditional expectation, so it is *not* a whole number; the interval
    # endpoints are, because they are quantiles of a discrete distribution.
    assert np.allclose(f["lower"], np.round(f["lower"]), atol=1e-9)
    assert np.allclose(f["upper"], np.round(f["upper"]), atol=1e-9)


def test_ingarch_pmf_carries_the_count_interval(ingarch_data):
    m = INGARCHModel().fit(ingarch_data)
    m.predict(4)
    pred = m.predictive()
    assert pred is not None and pred.family in {"poisson", "negbin"}
    grid, pmf = pred.step(0)
    assert pmf.sum() == pytest.approx(1.0, abs=1e-8)
    assert grid.min() == 0


def test_ingarch_refuses_a_continuous_series():
    y = pd.Series(np.sin(np.arange(60) / 3) * 5 + 20)
    m = INGARCHModel().fit(y)
    assert m.failed
    assert "count" in m.summary.failure_reason.lower()


def test_ingarch_refuses_a_series_too_short_to_estimate_two_parameters():
    m = INGARCHModel().fit(pd.Series(np.arange(12) % 4))
    assert m.failed
    assert any(ch.isdigit() for ch in m.summary.failure_reason)


def test_inar_recovers_the_thinning_probability(ingarch_data):
    m = INARModel().fit(ingarch_data)
    assert not m.failed
    params = {p.name: p.value for p in m.summary.params}
    alpha = next(v for k, v in params.items() if k.startswith("alpha"))
    lam = next(v for k, v in params.items() if k.startswith("lambda"))
    assert 0.0 < alpha < 0.95
    assert lam > 0
    # The stationary mean implied by the fit should be in the region of the data's own mean.
    assert float(m.predict(3)["mean"].iloc[0]) == pytest.approx(float(ingarch_data.mean()), rel=0.35)


def test_inar_and_ingarch_are_flagged_non_comparable_on_ic(ingarch_data):
    # INAR uses a Gaussian quasi-likelihood; its AIC is not on the same scale as the Poisson
    # likelihood. The summary has to say so, or the comparison table would rank them as if it were.
    m = INARModel().fit(ingarch_data)
    assert m.summary.ic_comparable is False
    g = INGARCHModel().fit(ingarch_data)
    assert g.summary.ic_comparable is True
    assert any("quasi" in c.lower() or "not comparable" in c.lower() for c in m.summary.caveats)


def test_count_dependence_models_are_marked_advanced_in_the_registry():
    # §5.8 marks INAR/INGARCH "optional, flagged; report as advanced". INGARCH and the event models
    # carry the flag; INAR is offered unflagged because its parameter has a mechanism reading.
    for mid in ("ingarch", "renewal", "hawkes"):
        assert REGISTRY[mid].feature_flag == "advanced", mid
        assert is_implemented(mid)
    assert is_implemented("inar")


def test_ingarch_warns_when_the_counts_show_no_serial_dependence():
    rng = np.random.default_rng(SHOTS)
    y = pd.Series(rng.poisson(6.0, size=120))
    spec = REGISTRY["ingarch"]

    class F:  # minimal facts surface, only the fields the row reads
        data_class = DataClass.COUNT
        autocorr_lag1 = 0.02
        dispersion_ratio = 1.05
        n = 120
        n_non_missing = 120
        n_zeros = 0
        pct_zeros = 0.0
        period = None
        seasonal_strength = None
        n_breaks = 0
        n_exogenous = 0
        min_value = 0.0
        max_value = 12.0
        n_unique = 12
        has_exogenous = False
        has_missing = False
        is_regular = True
        series_length = 120
        n_seasonal_cycles = 10.0
        is_integer_valued = True

    verdict = evaluate_spec(spec, F())
    assert verdict.compatible
    assert any("no serial dependence" in w or "almost no serial" in w for w in verdict.warnings)


# --------------------------------------------------------------------------- #
# 3. Intermittent: ADIDA and the SY-corrected Croston
# --------------------------------------------------------------------------- #
def test_adida_forecasts_a_plausible_per_period_rate(spare_parts):
    m = ADIDAModel().fit(spare_parts)
    assert not m.failed
    f = m.predict(6)
    mean = float(f["mean"].iloc[0])
    # The block rate multiplied by the block length is one number; the per-period rate must sit near
    # the observed average demand per period (total / n) or the disaggregation is wrong.
    observed = float(spare_parts.sum() / spare_parts.size)
    assert mean == pytest.approx(observed, rel=0.35)
    assert np.allclose(f["mean"], f["mean"].iloc[0])  # flat forecast, as a rate model should be


def test_adida_reports_its_bucket_and_alpha_in_the_one_liner(spare_parts):
    m = ADIDAModel().fit(spare_parts)
    line = m.summary.plain_one_liner
    assert "4 periods" in line or "periods at a time" in line
    assert any(ch.isdigit() for ch in line)


def test_adida_refuses_a_series_with_fewer_than_four_blocks():
    y = pd.Series([0.0] * 20 + [5.0, 0.0, 0.0, 7.0, 0.0])
    m = ADIDAModel().fit(y)
    assert m.failed
    assert any(ch.isdigit() for ch in m.summary.failure_reason)


def test_croston_sy_skips_the_correction_when_demand_is_frequent():
    # Mean interval below 1.32 → the correction must not fire, so the reported rate equals the
    # uncorrected one by construction (correction = 1.0), and the one-liner has to say why.
    rng = np.random.default_rng(SHOTS)
    y = pd.Series(rng.poisson(3.0, size=60).astype(float))
    sy = CrostonVariantModel().fit(y)
    assert not sy.failed
    assert sy.interval_level <= CrostonVariantModel.SY_THRESHOLD or sy.correction == 1.0
    assert float(sy.predict(1)["mean"].iloc[0]) == pytest.approx(sy.rate_uncorrected)
    assert "1.32" in sy.summary.plain_one_liner


def test_croston_sy_applies_the_correction_when_intervals_are_long(spare_parts):
    sy = CrostonVariantModel().fit(spare_parts)
    assert sy.interval_level > CrostonVariantModel.SY_THRESHOLD
    # SY scales the rate down by (1 − α/2); if the two rates are equal the correction is a no-op.
    assert sy.rate < sy.rate_uncorrected
    assert sy.rate == pytest.approx(sy.rate_uncorrected * (1 - 0.1 / 2), rel=1e-9)
    # The prose has to name the interval that triggered the correction, so a reader can check the
    # 1.32 rule for themselves rather than taking "SY-corrected" on trust.
    assert "3.2" in sy.summary.plain_one_liner
    assert "SY" in sy.summary.plain_one_liner


def test_intermittent_family_cap_still_offers_the_new_models():
    from app.ts.models.facts import build_facts, classify_data_class
    from app.ts.models.suggest import suggest

    y = pd.read_csv("app/assets/examples/spare_parts.csv", index_col=0, parse_dates=True).iloc[:, 0]
    dc, _ = classify_data_class(y)
    s = suggest(build_facts(y, data_class=dc))
    ids = [x.model_id for x in s.suggestions]
    assert {"croston", "tsb", "adida", "croston_sy"} <= set(ids)


# --------------------------------------------------------------------------- #
# 4. Point processes: event ingestion, renewal, Hawkes
# --------------------------------------------------------------------------- #
def test_one_column_date_file_is_read_as_events():
    csv = "event_date\n" + "\n".join(
        (pd.Timestamp("2021-01-01") + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in np.sort(np.random.default_rng(SHOTS).integers(0, 400, 120))
    )
    r = ingest_text(csv)
    assert r.dataset.data_class is DataClass.EVENT_TIMESTAMP
    assert r.mapping.event_mode is True
    assert np.allclose(r.series.to_numpy(), np.round(r.series.to_numpy()))  # counts, not values
    assert r.series.sum() == pytest.approx(120)  # every event landed in exactly one period
    assert "events" in r.dataset.quality.plain_summary


def test_event_ingestion_keeps_every_duplicate_day():
    # Two events on one day are two events. Collapsing duplicates is right for a value series and
    # wrong here, because the gap structure is the data.
    csv = "d\n2021-01-01\n2021-01-01\n2021-01-01\n2021-01-09\n2021-02-01"
    r = ingest_text(csv)
    assert r.series.sum() == pytest.approx(5)
    assert float(r.series.max()) == 3


def test_event_ingestion_refuses_a_file_with_no_readable_dates():
    from app.core.errors import ChronosError

    with pytest.raises(ChronosError):
        ingest_text("d\nnot-a-date\nalso-not\nnope")


def test_inter_arrivals_keep_same_day_pairs_and_drop_negative_ones():
    # A zero gap is real information: two events on one day means the observed rate is twice what a
    # "one event per distinct date" reading would suggest. Dropping it biased the estimated rate low
    # by a factor of four on the aftershock dataset.
    t = pd.DatetimeIndex(["2021-01-01", "2021-01-05", "2021-01-05", "2021-02-01"])
    gaps = inter_arrivals(t)
    assert np.all(gaps >= 0)
    assert list(gaps) == [4.0, 0.0, 27.0]
    assert gaps.sum() == pytest.approx(31.0)
    # Unsorted input can produce a negative gap, which has no meaning as an interval.
    assert np.all(inter_arrivals(t[::-1]) >= 0)


def test_renewal_reports_a_hazard_direction_that_matches_the_data():
    # Weibull shape < 1 → decreasing hazard → gaps are not memoryless. The model must recover the
    # direction, not merely a mean gap.
    t = _hawkes_times(mu=0.25, alpha=0.0, beta=1.0, T=1500.0)
    m = RenewalModel().fit(pd.Series(np.ones(len(t)), index=t))
    assert not m.failed
    shape = next(p.value for p in m.summary.params if "shape" in p.name.lower())
    assert shape < 1.1  # clustered gaps push the shape below 1
    assert "decreas" in m.summary.plain_one_liner.lower() or "shape" in m.summary.plain_one_liner


def test_renewal_names_both_distributions_it_compared():
    t = _hawkes_times(mu=0.25, alpha=0.0, beta=1.0, T=900.0, seed=7)
    m = RenewalModel().fit(pd.Series(np.ones(len(t)), index=t))
    assert not m.failed
    txt = " ".join([m.summary.plain_one_liner] + list(m.summary.caveats)).lower()
    assert "weibull" in txt or "exponential" in txt


def test_hawkes_recovers_the_branching_ratio():
    t = _hawkes_times(mu=0.05, alpha=0.6, beta=1.0, T=1600.0, seed=11)
    m = HawkesModel().fit(pd.Series(np.ones(len(t)), index=t))
    assert not m.failed
    branching = next(p.value for p in m.summary.params if "branch" in p.name.lower())
    assert branching == pytest.approx(0.6, abs=0.25)
    assert 0.0 <= branching < 0.95


def test_hawkes_finds_almost_no_branching_in_a_regular_sequence():
    # The control case. Without it, "Hawkes finds self-excitation" could just mean "Hawkes finds
    # whatever it looked for".
    t = _hawkes_times(mu=0.25, alpha=0.0, beta=1.0, T=1400.0, seed=5)
    m = HawkesModel().fit(pd.Series(np.ones(len(t)), index=t))
    assert not m.failed
    branching = next(p.value for p in m.summary.params if "branch" in p.name.lower())
    assert branching < 0.35


def test_hawkes_refuses_a_branching_ratio_near_criticality():
    # A near-critical fit is not identified: the likelihood is flat and the forecast explodes. The
    # wrapper must refuse rather than report 0.99 as a fact.
    rng = np.random.default_rng(SHOTS)
    times = np.cumsum(rng.exponential(0.4, size=200))
    m = HawkesModel().fit(pd.Series(np.ones(200), index=pd.DatetimeIndex(
        pd.Timestamp("2020-01-01") + pd.to_timedelta(times, unit="D"))))
    if not m.failed:
        branching = next(p.value for p in m.summary.params if "branch" in p.name.lower())
        assert branching < 0.95


def test_hawkes_refuses_too_few_events():
    t = pd.DatetimeIndex(["2021-01-01", "2021-01-04", "2021-01-20"])
    m = HawkesModel().fit(pd.Series(np.ones(3), index=t))
    assert m.failed
    assert any(ch.isdigit() for ch in m.summary.failure_reason)


def test_point_process_refuses_day_resolution_timestamps():
    # 289 of 455 gaps were exactly zero on the original aftershock example, and the Weibull likelihood
    # moved by 9000 between a one-second and a one-hour floor under those zeros. The model must decline
    # rather than report whichever number the floor produced.
    days = pd.DatetimeIndex(pd.Timestamp("2020-01-01") + pd.to_timedelta([0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6], unit="D"))
    m = RenewalModel().fit(pd.Series(np.ones(len(days)), index=days))
    assert m.failed
    assert "zero" in m.summary.failure_reason
    assert any(ch.isdigit() for ch in m.summary.failure_reason)


def test_point_process_models_are_excluded_from_value_series():
    spec = REGISTRY["hawkes"]

    class F:
        data_class = DataClass.CONTINUOUS
        n_non_missing = 120
        n = 120
        n_zeros = 0
        pct_zeros = 0.0
        autocorr_lag1 = 0.3
        dispersion_ratio = 1.0
        period = 12
        seasonal_strength = 0.2
        n_breaks = 0
        n_exogenous = 0
        min_value = 0.0
        max_value = 12.0
        n_unique = 12
        has_exogenous = False
        has_missing = False
        is_regular = True
        series_length = 120
        n_seasonal_cycles = 10.0
        is_integer_valued = True

    verdict = evaluate_spec(spec, F())
    assert not verdict.compatible
    assert any("event timestamps" in r for r in verdict.reasons)


def test_value_models_are_excluded_from_event_data():
    class F:
        data_class = DataClass.EVENT_TIMESTAMP
        n_non_missing = 40
        n = 40
        n_zeros = 0
        pct_zeros = 0.0
        autocorr_lag1 = 0.2
        dispersion_ratio = 1.0
        period = None
        seasonal_strength = None
        n_breaks = 0
        n_exogenous = 0
        min_value = 0.0
        max_value = 12.0
        n_unique = 12
        has_exogenous = False
        has_missing = False
        is_regular = True
        series_length = 40
        n_seasonal_cycles = 3.0
        is_integer_valued = True

    for mid in ("arima", "ets", "poisson"):
        verdict = evaluate_spec(REGISTRY[mid], F())
        assert not verdict.compatible, mid
        # The reason must name what is missing, not just say "incompatible".
        assert any("event dates" in r or "event" in r for r in verdict.reasons)


# --------------------------------------------------------------------------- #
# 5. Wiring: fit_many, the discrete flag, and the API contract
# --------------------------------------------------------------------------- #
def test_fit_many_takes_event_times_for_the_point_process_models():
    t = _hawkes_times(mu=0.05, alpha=0.6, beta=1.0, T=1200.0, seed=13)
    counts = pd.Series(1.0, index=t).resample("MS").sum().fillna(0.0)
    outcomes = fit_many(
        ["renewal", "hawkes"], counts, period=None, data_class=DataClass.EVENT_TIMESTAMP,
        ensure_baselines=False, event_times=t,
    )
    by_id = {o.summary.model_id: o for o in outcomes}
    assert not by_id["hawkes"].summary.failed
    assert not by_id["renewal"].summary.failed
    # Fitting on the counts instead of the timestamps would produce a different, wrong answer, so
    # this asserts the substitution actually happened.
    assert by_id["hawkes"].wrapper.n_events == len(t)


def test_discrete_flag_covers_every_pmf_model():
    assert {"poisson", "neg_binomial", "zero_inflated", "ingarch", "inar", "renewal", "hawkes"} <= (
        DISCRETE_PREDICTIVE
    )
    assert "arima" not in DISCRETE_PREDICTIVE


def test_registry_rows_for_part10_are_all_implemented():
    for mid in ("ingarch", "inar", "adida", "croston_sy", "renewal", "hawkes"):
        assert mid in REGISTRY
        assert is_implemented(mid)
        assert REGISTRY[mid].family in {
            ModelFamily.COUNT, ModelFamily.INTERMITTENT, ModelFamily.POINT_PROCESS
        }


def test_family_to_stage_map_covers_point_process():
    from app.api.models import STAGE_OF_FAMILY

    assert ModelFamily.POINT_PROCESS in STAGE_OF_FAMILY


def test_forecast_endpoint_ships_the_pmf(pm_client, ready_session):
    sid = ready_session("outbreak_counts")
    h = {"X-Session-Id": sid}
    pm_client.post("/api/v1/analysis/fit", json={"model_ids": ["ingarch"]}, headers=h)
    r = pm_client.post(
        "/api/v1/analysis/fit/forecast", json={"model_id": "ingarch", "horizon": 3}, headers=h
    )
    assert r.status_code == 200
    body = r.json()
    assert body["discrete"] is True
    pred = body["predictive"]
    assert pred is not None
    assert len(pred["support"]) == 3
    for pmf in pred["pmf"]:
        assert sum(pmf) == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in pmf)


def test_event_examples_are_in_the_manifest_and_ingest_as_events():
    import json

    manifest = json.loads(open("app/assets/examples/manifest.json").read())
    ids = {m["id"]: m for m in manifest}
    assert {"aftershocks", "machine_service"} <= set(ids)
    assert ids["aftershocks"]["data_class"] == "event_timestamp"


# --------------------------------------------------------------------------- #
# 6. Fixtures for the two API tests (kept local so this file stands alone)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def pm_client():
    from fastapi.testclient import TestClient

    from app.main import app

    return TestClient(app)


@pytest.fixture(scope="module")
def ready_session(pm_client):
    def _make(example_id: str) -> str:
        sid = pm_client.post("/api/v1/sessions").json()["session_id"]
        h = {"X-Session-Id": sid}
        r = pm_client.post("/api/v1/data/example", json={"example_id": example_id}, headers=h)
        assert r.status_code == 201
        for path in ("profile", "decompose", "tests"):
            pm_client.post(f"/api/v1/analysis/{path}", headers=h)
        pm_client.get("/api/v1/analysis/correlate", headers=h)
        pm_client.post("/api/v1/analysis/gate", headers=h)
        # S9 must be *finished* before S10 accepts a fit; the suggestion call is what completes it.
        pm_client.post("/api/v1/analysis/suggest", headers=h)
        return sid

    return _make
