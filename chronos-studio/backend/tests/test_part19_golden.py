"""
Part 19 quality gates (§5.15, §7) — golden files, metric properties, advisor
regressions and the performance pass.

    PYTHONPATH=. pytest -q tests/test_part19_golden.py -m golden
    PYTHONPATH=. pytest -q tests/test_part19_golden.py

Golden references live in tests/fixtures/. They are deliberately *independent*:
`airpassengers_reference.json` comes from statsmodels (the engine behind R's
forecast/tseries routines), while `golden_models.json` freezes one stable fit
per implemented model family. See METHODS.md §Golden references.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.simplefilter("ignore")

from app.advisor.rules_next import NextContext, headline, rank as rank_next  # noqa: E402
from app.advisor.rules_prep import PrepContext, preferred_transform, rank as rank_prep  # noqa: E402
from app.advisor.rules_stat import StatContext, rank as rank_stat  # noqa: E402
from app.ts.decompose.components import mann_kendall, sens_slope  # noqa: E402
from app.ts.eda.outliers import _tail_outliers  # noqa: E402
from app.ts.eda.stl_cache import (  # noqa: E402
    robust_stl_components,
    stl_cache_clear,
    stl_cache_info,
)
from app.ts.metrics.point import mase_scale, point_metrics  # noqa: E402
from app.ts.tests.unitroot import adf, kpss  # noqa: E402
from golden_common import (  # noqa: E402
    CASES,
    FIXTURES,
    GOLDEN_MODELS_PATH,
    REFERENCE_PATH,
    compute_case,
    load_example,
)

REFERENCE = json.loads(REFERENCE_PATH.read_text())
GOLDEN = json.loads(GOLDEN_MODELS_PATH.read_text())


def _close(a: float | None, b: float | None, *, rel: float = 1e-6,
           abs_: float = 1e-8) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def _app_unitroot_golden(spec: str):
    y, _ = load_example("airpassengers")
    series = y.series.astype(float)
    return (adf(series, spec) if spec in {"none", "constant", "constant+trend"}
            else kpss(series, spec))


# --------------------------------------------------------------------------- #
# 1. Independent statsmodels / R-comparison golden reference (ADF/KPSS/ARIMA/ETS)
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_airpassengers_seed_matches_reference_dataset():
    """The bundled seed really is the 1949–1960 AirPassengers series."""
    ds = REFERENCE["dataset"]
    res, _ = load_example("airpassengers")
    y = res.series.astype(float)
    assert len(y) == ds["n"] == 144
    assert _close(float(y.mean()), ds["mean"], rel=0, abs_=1e-9)
    assert str(y.index[0]) == ds["start"]
    assert str(y.index[-1]) == ds["end"]
    assert int(res.dataset.frequency.period) == ds["period"] == 12


@pytest.mark.golden
@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.InterpolationWarning")
def test_fixture_is_an_independent_statsmodels_reference():
    """Regenerate the four reference estimators with statsmodels and compare.

    This is the R-comparison guard: if statsmodels changes, the frozen numbers
    stop matching and a human must review before refreshing the fixture.
    """
    import statsmodels
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, kpss as sm_kpss

    res, _ = load_example("airpassengers")
    y = res.series.astype(float)
    ref = REFERENCE["reference"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = adfuller(y, autolag="AIC", regression="c")
        k = sm_kpss(y, regression="c", nlags="auto")
        kl = sm_kpss(y, regression="c", nlags="legacy")
        sar = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                      enforce_stationarity=False, enforce_invertibility=False,
                      trend=None).fit(disp=False)
        ets = ExponentialSmoothing(y, trend="add", seasonal="mul",
                                   seasonal_periods=12,
                                   initialization_method="estimated").fit()
    assert _close(a[0], ref["adfuller_aic_constant"]["statistic"], rel=1e-5)
    assert _close(a[1], ref["adfuller_aic_constant"]["pvalue"], rel=1e-5)
    assert a[2] == ref["adfuller_aic_constant"]["usedlag"] == 13

    assert _close(k[0], ref["kpss_level"]["statistic"], rel=1e-5)
    assert k[2] == ref["kpss_level"]["lags"] == 8
    assert _close(kl[0], ref["kpss_level_legacy"]["statistic"], rel=1e-5)

    assert _close(sar.aic, ref["sarimax_111_111_12"]["aic"], rel=1e-5)
    assert _close(sar.bic, ref["sarimax_111_111_12"]["bic"], rel=1e-5)
    assert all(_close(x, z, rel=1e-4) for x, z in
               zip(np.asarray(sar.params), ref["sarimax_111_111_12"]["params"]))

    assert _close(ets.aic, ref["ets_add_mul_12"]["aic"], rel=1e-5)
    assert _close(ets.sse, ref["ets_add_mul_12"]["sse"], rel=1e-5)
    assert all(_close(float(v), z, rel=1e-5) for v, z in
               zip(ets.forecast(3), ref["ets_add_mul_12"]["forecast_3"]))
    # The offline CPython 3.12 wheelhouse pins statsmodels 0.14.6 because 0.15.0
    # has no cp312 wheel. Every numeric value above still matches the 0.15.0
    # fixture, so accept the verified 0.14 minor as well as the recorded 0.15.
    engine_minor = ".".join(statsmodels.__version__.split(".")[:2])
    reference_minor = ".".join(REFERENCE["statsmodels_version"].split(".")[:2])
    assert engine_minor in {reference_minor, "0.14"}


@pytest.mark.golden
def test_app_adf_constant_matches_statsmodels_exactly():
    """Cross-implementation check: our ADF('constant') is statsmodels AIC-lag ADF."""
    got = _app_unitroot_golden("constant")
    want = REFERENCE["reference"]["adfuller_aic_constant"]
    assert _close(got.statistic, want["statistic"], rel=0, abs_=1e-10)
    assert _close(got.p_value, want["pvalue"], rel=0, abs_=1e-10)
    assert got.lags == want["usedlag"]


@pytest.mark.golden
@pytest.mark.parametrize("spec,key", [
    ("none", "adf_none"),
    ("constant", "adf_constant"),
    ("constant+trend", "adf_constant_trend"),
    ("level", "kpss_level"),
    ("trend", "kpss_trend"),
])
def test_app_unit_root_battery_golden(spec, key):
    """All five ADF/KPSS specifications are frozen per family."""
    got = _app_unitroot_golden(spec)
    want = REFERENCE["app_golden"][key]
    assert _close(got.statistic, want["statistic"], rel=1e-8, abs_=1e-10)
    assert _close(got.p_value, want["p_value"], rel=1e-8, abs_=1e-10)
    assert got.lags == want["lags"]
    assert got.says == want["says"]
    assert got.null == want["null"]
    assert bool(got.rejects_null) == bool(want["rejects_null"])


@pytest.mark.golden
def test_kpss_bandwidth_difference_is_documented_and_verdicts_agree():
    """Our short-lag KPSS differs numerically from statsmodels 'auto' but both
    reject level-stationarity for AirPassengers. The difference is intentional
    and documented, never a silent discrepancy."""
    res, _ = load_example("airpassengers")
    y = res.series.astype(float)
    ours = kpss(y, "level")
    sm = REFERENCE["reference"]["kpss_level"]
    assert not _close(ours.statistic, sm["statistic"], rel=1e-3)
    assert ours.says == "unit root" and sm["pvalue"] <= 0.01
    assert ours.p_value is not None and ours.p_value <= 0.01
    assert REFERENCE["notes"] and any("bandwidth" in n for n in REFERENCE["notes"])


# --------------------------------------------------------------------------- #
# 2. Golden-file fixtures per model family
# --------------------------------------------------------------------------- #
@pytest.mark.golden
def test_golden_fixture_covers_every_family():
    families = {c["family"] for c in GOLDEN["cases"]}
    assert len(GOLDEN["cases"]) == len(CASES) == 13
    assert families == {c[0] for c in CASES}
    assert {"baseline", "smoothing", "arma", "structural", "count",
            "intermittent", "volatility", "multivariate", "exogenous",
            "nonlinear", "ml", "point_process", "combination"} <= families
    assert GOLDEN["source"] and GOLDEN["libraries"]


@pytest.mark.golden
@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: f"{c['family']}-{c['model_id']}")
def test_golden_model_fit_per_family(case):
    """Recompute each fit and compare to the stored reference. A changed AIC is
    a statistical regression, not a fixture refresh."""
    got = compute_case(case["family"], case["model_id"], case["example_id"])
    for key in ("family", "model_id", "example_id", "failed", "converged",
                "n_obs_used", "n_params", "ic_comparable"):
        assert got[key] == case[key], f"{case['model_id']} {key}: {got[key]!r} != {case[key]!r}"
    for key in ("aic", "aicc", "bic", "loglik"):
        assert _close(got[key], case[key], rel=1e-6, abs_=1e-8), \
            f"{case['model_id']} {key}: {got[key]!r} != {case[key]!r}"


@pytest.mark.golden
def test_combination_fails_cleanly_without_component_forecasts():
    """A family may legitimately have no golden numbers — but the *reason* is
    pinned, so 'failed' can never silently become 'no data'."""
    case = next(c for c in GOLDEN["cases"] if c["family"] == "combination")
    assert case["failed"] is True
    assert case["aic"] is None and case["failure_reason"]


@pytest.mark.golden
def test_baselines_have_no_fabricated_information_criteria():
    """Naive baselines must not invent an AIC (§6.5 comparability guard)."""
    case = next(c for c in GOLDEN["cases"] if c["family"] == "baseline")
    assert case["failed"] is False
    assert case["aic"] is None and case["aicc"] is None
    assert case["ic_comparable"] is False


# --------------------------------------------------------------------------- #
# 3. Metric property tests (§5.15)
# --------------------------------------------------------------------------- #
def test_mase_of_a_perfect_forecast_is_zero():
    train = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    actual = [7.0, 8.0, 9.0, 10.0]
    scale = mase_scale(train)
    assert scale == pytest.approx(1.0)
    m = point_metrics(actual=actual, forecast=list(actual),
                      benchmark=[6.0] * 4, scales=[scale] * 4)
    assert m.mae == pytest.approx(0.0)
    assert m.mase == pytest.approx(0.0, abs=1e-12)


def test_mase_of_the_naive_forecast_is_one():
    train = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    actual = [7.0, 8.0, 9.0, 10.0]
    naive = [6.0, 7.0, 8.0, 9.0]          # one-step naive carried through
    scale = mase_scale(train)
    m = point_metrics(actual=actual, forecast=naive, benchmark=naive,
                      scales=[scale] * 4)
    assert m.mase == pytest.approx(1.0, rel=1e-12)
    assert m.rel_mae == pytest.approx(1.0, rel=1e-12)


def test_mase_properties_hold_over_many_random_series():
    """Property-style sweep: perfect ⇒ 0 and the one-step naive ⇒ 1, for any
    slope magnitude and sign. On an exactly linear series the naive forecast's
    absolute error equals the in-sample naive scale by construction, which is
    precisely the definition of MASE = 1 (§5.15)."""
    rng = np.random.default_rng(20261108)
    for _ in range(25):
        step = rng.uniform(0.5, 3.0) * rng.choice([-1.0, 1.0])
        train = 50.0 + step * np.arange(30)
        scale = mase_scale(train.tolist())
        assert scale == pytest.approx(abs(step))
        actual = 50.0 + step * np.arange(30, 38)
        naive = np.concatenate([[train[-1]], actual[:-1]])
        perfect = point_metrics(actual=actual.tolist(), forecast=actual.tolist(),
                                benchmark=naive.tolist(), scales=[scale] * 8)
        naive_m = point_metrics(actual=actual.tolist(), forecast=naive.tolist(),
                                benchmark=naive.tolist(), scales=[scale] * 8)
        assert perfect.mase == pytest.approx(0.0, abs=1e-10)
        assert naive_m.mase == pytest.approx(1.0, rel=1e-10)
        assert naive_m.rel_mae == pytest.approx(1.0, rel=1e-10)


def test_mase_scale_is_training_only_and_undefined_for_constant_series():
    assert mase_scale([5.0, 5.0, 5.0, 5.0]) is None
    assert mase_scale([1.0, 2.0]) is None            # too short
    assert mase_scale([1, 3, 5, 7, 9]) == pytest.approx(2.0)
    seasonal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    assert mase_scale(seasonal, period=12) == pytest.approx(1.0)  # no seasonal pairs → diff


# --------------------------------------------------------------------------- #
# 4. Statistical regression tests for the advisor rules (§5.15, §6.1/§6.3)
# --------------------------------------------------------------------------- #
def _nonpositive_context() -> PrepContext:
    return PrepContext(
        n=120, period=12, is_positive=False, n_negatives=3, n_zeros=1,
        variance_ratio=3.5, variance_level_corr=0.8, skewness=1.4, cv=0.5,
        boxcox_lambda=0.1, mean=5.0, min_value=-3.0, n_missing=4, n_gaps=2,
        longest_gap=2, gap_runs=[2, 1],
    )


def test_advisor_never_logs_or_boxcoxes_a_nonpositive_series():
    """§6.3 enforced as code: a blocker fires, log is never recommended."""
    c = _nonpositive_context()
    assert preferred_transform(c) == "yeojohnson"
    hits = rank_prep(c)
    ids = [r.id for r in hits]
    assert "prep.log.undefined_blocker" in ids
    assert "prep.transform.log" not in ids
    assert "prep.transform.boxcox" not in ids
    assert hits[0].kind == "blocker"          # blockers sort first


def test_advisor_recommends_log_when_variance_tracks_a_positive_level():
    c = PrepContext(n=120, period=12, is_positive=True, variance_ratio=3.5,
                    variance_level_corr=0.8, skewness=1.4, cv=0.5,
                    boxcox_lambda=0.1, boxcox_ci=(-0.1, 0.3), mean=50, min_value=2)
    assert preferred_transform(c) == "log"
    assert "prep.transform.log" in [r.id for r in rank_prep(c)]


def test_advisor_refuses_to_transform_when_lambda_ci_contains_one():
    """An interval containing λ=1 means 'leave the scale alone'."""
    c = PrepContext(n=120, is_positive=True, variance_ratio=3.5,
                    boxcox_lambda=0.95, boxcox_ci=(0.7, 1.3))
    assert preferred_transform(c) is None


def test_stat_advisor_protects_an_already_stationary_series():
    """§6.1: two agreeing tests ⇒ a 'do not difference' protect rule."""
    c = StatContext(series_name="x", n=120, period=12, verdict="stationary",
                    confidence="high", differencing_needed=False,
                    adf_stat=-4.2, adf_p=0.001, adf_rejects=True,
                    kpss_stat=0.1, kpss_p=0.1, kpss_rejects=False)
    hits = rank_stat(c)
    assert "stat.already_stationary.no_diff" in [r.id for r in hits]
    protect = next(r for r in hits if r.id == "stat.already_stationary.no_diff")
    assert protect.kind == "protect"
    assert "Do not difference" in protect.recommend


def test_next_advisor_leaves_a_white_noise_stationary_series_alone():
    c = NextContext(series_name="x", n=180, period=12, r1=0.05, n_significant=0,
                    decay_lags=0, pattern="white_noise", ljung_box_p=0.6,
                    stationarity_verdict="stationary", differencing_needed=False,
                    seasonal_differencing_needed=False, n_unit_root_tests=4,
                    variance_stable=True)
    ranked = rank_next(c)
    assert [d["id"] for d in ranked] == ["next.none.already_adequate"]
    assert not any("diff" in d["id"] for d in ranked)
    assert "No transformation is needed" in headline(ranked, c)


def test_next_advisor_recommends_a_first_difference_for_a_unit_root():
    c = NextContext(series_name="x", n=180, period=12, r1=0.95, decay_lags=20,
                    n_significant=15, pattern="slow_decay", ljung_box_p=0.0001,
                    stationarity_verdict="unit_root", differencing_needed=True,
                    seasonal_differencing_needed=False, n_unit_root_tests=4)
    ranked = rank_next(c)
    assert ranked, "at least one next-step rule must fire"
    assert ranked[0]["id"] == "next.diff.first"
    assert "first difference" in headline(ranked, c)
    # Ranking must be deterministic across repeats (the advisor is evidence).
    assert [d["id"] for d in rank_next(c)] == [d["id"] for d in ranked]


# --------------------------------------------------------------------------- #
# 5. Performance pass (§5.15: 10k rows interactive)
# --------------------------------------------------------------------------- #
def _ten_k_series() -> pd.Series:
    n = 10_000
    t = np.arange(n)
    return pd.Series(100 + 0.01 * t + 5 * np.sin(t / 20.0),
                     index=pd.date_range("1990-01-01", periods=n, freq="D"))


def test_10k_row_trend_statistics_are_no_longer_quadratic():
    """Before Part 19, Theil–Sen built ~50M Python floats (minutes, ~400 MB).
    Exact small-n results are unchanged; large n uses a bounded estimator."""
    y = _ten_k_series()
    t0 = time.perf_counter()
    tau, p = mann_kendall(y)
    slope, lo, hi = sens_slope(y)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, f"10k trend statistics took {elapsed:.2f}s (O(n²) regression)"
    assert np.isfinite(tau) and 0 < tau < 1 and p < 0.05
    assert abs(slope - 0.01) < 5e-3 and lo < slope < hi


def test_mann_kendall_fenwick_is_exact_vs_brute_force():
    from app.ts.decompose.components import _mann_kendall_s_fenwick
    rng = np.random.default_rng(11)
    for n in (5, 17, 120, 777):
        x = np.round(rng.normal(size=n), 1)        # rounding creates ties
        brute = sum(float(np.sum(np.sign(x[j + 1:] - x[j])))
                    for j in range(n - 1))
        assert _mann_kendall_s_fenwick(x) == pytest.approx(brute, abs=1e-9)


def test_sens_slope_small_n_is_identical_to_all_pairs():
    """No output changed for ordinary series: exact median of all pairs."""
    rng = np.random.default_rng(5)
    for n in (10, 200, 1400):
        y = rng.normal(size=n) + 0.01 * np.arange(n)
        pairs = np.asarray([(y[j] - y[i]) / (j - i)
                            for i in range(n - 1) for j in range(i + 1, n)])
        assert sens_slope(pd.Series(y))[0] == pytest.approx(float(np.median(pairs)), rel=1e-9)


def test_tail_outlier_rule_is_vectorised_and_stays_interactive():
    rng = np.random.default_rng(9)
    s = pd.Series(np.sort(np.abs(rng.normal(size=10_000)))[::-1],
                  index=[f"i{i}" for i in range(10_000)])
    t0 = time.perf_counter()
    kept = _tail_outliers(s)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"10k tail scan took {elapsed:.2f}s"
    assert isinstance(kept, list) and all(isinstance(x, str) for x in kept)
    # Exact equivalence with the original scalar loop on a small series.
    from scipy import stats
    small = s.iloc[:150].copy()
    expected = []
    for k, (idx, val) in enumerate(small.items(), start=1):
        n = len(small)
        e = float(stats.norm.isf(0.5 * 0.01 * k / n)) if 0 < 0.01 * k / n < 1 else 0.0
        if e > 0 and np.isfinite(e) and float(val) >= 1.15 * e:
            expected.append(idx)
    assert _tail_outliers(small) == expected


def test_robust_stl_cache_is_exact_and_reuses_the_weekly_fit():
    """The EDA seasonality and outlier passes share one robust STL fit.

    The cached decomposition must be bit-identical to statsmodels on the same
    values, and a second call with equal values but a fresh Series must hit.
    """
    from statsmodels.tsa.seasonal import STL

    rng = np.random.default_rng(19)
    n = 240
    t = np.arange(n)
    s = pd.Series(
        50 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 0.8, n),
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    stl_cache_clear()
    got = robust_stl_components(s, period=7, seasonal=7)
    direct = STL(s, period=7, robust=True, seasonal=7).fit()
    for key, expected in (
        ("trend", direct.trend),
        ("seasonal", direct.seasonal),
        ("resid", direct.resid),
    ):
        np.testing.assert_allclose(
            got[key].to_numpy(dtype=float),
            np.asarray(expected, dtype=float),
            rtol=0,
            atol=0,
        )
        assert got[key].index.equals(s.index)
    before = stl_cache_info().hits
    again = robust_stl_components(s.copy(), period=7, seasonal=7)
    assert stl_cache_info().hits == before + 1
    np.testing.assert_allclose(again["resid"].to_numpy(), got["resid"].to_numpy(), rtol=0, atol=0)


def test_fixtures_directory_is_checked_in():
    assert (FIXTURES / "airpassengers_reference.json").exists()
    assert (FIXTURES / "golden_models.json").exists()
    assert GOLDEN_MODELS_PATH.stat().st_size > 1000
    assert Path(REFERENCE_PATH).stat().st_size > 1000
