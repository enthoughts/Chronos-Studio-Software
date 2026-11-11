"""
scripts/gen_examples.py — build the twelve bundled example series.

Run once at build time (needs network for the R datasets); the CSVs it writes are
committed, so the app itself needs no network and no extra dependency.

    python scripts/gen_examples.py

Every dataset ships with a note on what it teaches, because "load it and see" is
not a lesson.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = Path(__file__).resolve().parents[1] / "app" / "assets" / "examples"
ASSETS.mkdir(parents=True, exist_ok=True)

MANIFEST: list[dict] = []


def save(name: str, df: pd.DataFrame, time_col: str, value_col: str, meta: dict) -> None:
    path = ASSETS / f"{name}.csv"
    df[[time_col, value_col]].to_csv(path, index=False)
    MANIFEST.append({
        "id": name,
        "file": f"{name}.csv",
        "time_column": time_col,
        "value_column": value_col,
        "n_rows": int(len(df)),
        **meta,
    })
    print(f"  wrote {path.name:26s} n={len(df):5d}  {meta['title']}")


def from_r(name: str) -> pd.DataFrame | None:
    """Best-effort fetch from R's datasets package via statsmodels."""
    try:
        from statsmodels.datasets import get_rdataset

        d = get_rdataset(name, "datasets").data
        return d
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  (could not fetch R dataset '{name}': {exc})")
        return None


# --------------------------------------------------------------------------- #
def air_passengers() -> None:
    d = from_r("AirPassengers")
    if d is None:
        idx = pd.date_range("1949-01-01", periods=144, freq="MS")
        t = np.arange(144)
        vals = (112 * (1 + 0.021) ** t * (1 + 0.22 * np.sin(2 * np.pi * (t % 12) / 12))).round()
        df = pd.DataFrame({"Month": idx, "Passengers": vals})
    else:
        idx = pd.date_range("1949-01-01", periods=len(d), freq="MS")
        df = pd.DataFrame({"Month": idx, "Passengers": d["value"].to_numpy()})
    save("airpassengers", df, "Month", "Passengers", {
        "title": "Airline passengers (1949-1960)",
        "teaches": "The classic series. Strong trend AND multiplicative seasonality — "
                   "the textbook case for a log transform before SARIMA or Holt-Winters.",
        "data_class": "continuous", "expect": "trend+seasonal, non-stationary in level",
        "frequency": "MS",
    })


def us_acc_deaths() -> None:
    d = from_r("USAccDeaths")
    if d is None:
        idx = pd.date_range("1973-01-01", periods=72, freq="MS")
        t = np.arange(72)
        df = pd.DataFrame({"Month": idx,
                           "Deaths": (9000 + 700 * np.cos(2 * np.pi * (t % 12) / 12 + 1) + 200 * np.sin(t / 5)).round()})
        save("usaccdeaths", df, "Month", "Deaths", {
            "title": "US accidental deaths (1973-1978)",
            "teaches": "Additive seasonality with a mild downward trend, and no obvious "
                       "heteroskedasticity — the counter-example to AirPassengers.",
            "data_class": "continuous", "expect": "additive seasonal, weakly stationary",
            "frequency": "MS",
        })
        return
    idx = pd.date_range("1973-01-01", periods=len(d), freq="MS")
    df = pd.DataFrame({"Month": idx, "Deaths": d["value"].to_numpy()})
    save("usaccdeaths", df, "Month", "Deaths", {
        "title": "US accidental deaths (1973-1978)",
        "teaches": "Additive seasonality with a mild downward trend, and no obvious "
                   "heteroskedasticity — the counter-example to AirPassengers.",
        "data_class": "continuous", "expect": "additive seasonal, weakly stationary",
        "frequency": "MS",
    })


def lynx() -> None:
    d = from_r("lynx")
    if d is None:
        return
    idx = pd.date_range("1821-01-01", periods=len(d), freq="YS")
    df = pd.DataFrame({"Year": idx, "Trappings": d["value"].to_numpy()})
    save("lynx", df, "Year", "Trappings", {
        "title": "Canadian lynx trappings (1821-1934)",
        "teaches": "A ~10-year population cycle with no fixed calendar: it is a CYCLE, not "
                   "seasonality. Great for practising spectral analysis and log transforms.",
        "data_class": "continuous", "expect": "cyclic, heavy tails, near-stationary",
        "frequency": "YS",
    })


def wwwusage() -> None:
    d = from_r("WWWusage")
    if d is None:
        return
    idx = pd.date_range("1990-01-01", periods=len(d), freq="min")
    df = pd.DataFrame({"Minute": idx, "Usage": d["value"].to_numpy()})
    save("wwwusage", df, "Minute", "Usage", {
        "title": "Internet usage per minute (100 observations)",
        "teaches": "A short, strongly trending series where automatic ARIMA order "
                   "selection is unreliable — a lesson in small-sample caution.",
        "data_class": "continuous", "expect": "strong trend, no seasonality, n small",
        "frequency": "min",
    })


def johnson_johnson() -> None:
    d = from_r("JohnsonJohnson")
    if d is None:
        return
    idx = pd.date_range("1960-01-01", periods=len(d), freq="QS")
    df = pd.DataFrame({"Quarter": idx, "Earnings": d["value"].to_numpy()})
    save("johnsonjohnson", df, "Quarter", "Earnings", {
        "title": "Johnson & Johnson quarterly earnings",
        "teaches": "Quarterly data with growing seasonal amplitude — the case where a log "
                   "transform stabilises the seasonality itself.",
        "data_class": "continuous", "expect": "multiplicative seasonal, exponential trend",
        "frequency": "QS",
    })


def sunspots() -> None:
    d = from_r("sunspot.year")
    if d is None:
        return
    idx = pd.date_range("1700-01-01", periods=len(d), freq="YS")
    df = pd.DataFrame({"Year": idx, "Sunspots": d["value"].to_numpy()})
    save("sunspots", df, "Year", "Sunspots", {
        "title": "Yearly sunspot numbers (1700-1988)",
        "teaches": "A long record where the 11-year cycle drifts in length and amplitude. "
                   "Shows why 'seasonality' must be distinguished from 'cyclicity'.",
        "data_class": "count", "expect": "cyclic, heteroskedastic, stationary in level",
        "frequency": "YS",
    })


def beer() -> None:
    """Australian beer production, monthly."""
    d = from_r("AusBeer")
    if d is None:
        idx = pd.date_range("1990-01-01", periods=120, freq="QS")
        t = np.arange(120)
        df = pd.DataFrame({"Quarter": idx, "Beer": (400 + 40 * np.sin(2 * np.pi * (t % 4) / 4) + t).round()})
    else:
        idx = pd.date_range("1990-01-01", periods=len(d), freq="QS")
        df = pd.DataFrame({"Quarter": idx, "Beer": np.asarray(d["value"], dtype=float)})
    save("ausbeer", df, "Quarter", "Beer", {
        "title": "Australian beer production (quarterly)",
        "teaches": "Clean quarterly seasonality on a stable level. A gentle first "
                   "encounter with seasonal differencing.",
        "data_class": "continuous", "expect": "seasonal, stationary in level",
        "frequency": "QS",
    })


def electricity() -> None:
    """Daily electricity demand with weekly seasonality — synthetic but realistic."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2019-01-01", periods=730, freq="D")
    t = np.arange(len(idx))
    dow = idx.dayofweek.to_numpy()
    yearly = 180 * np.sin(2 * np.pi * t / 365.25 - 1.2)
    weekly = np.where(dow >= 5, -140, 60) + 40 * np.sin(2 * np.pi * dow / 7)
    noise = rng.normal(0, 22, len(idx))
    df = pd.DataFrame({"Date": idx, "Demand": (1200 + yearly + weekly + noise).round(1)})
    save("electricity_daily", df, "Date", "Demand", {
        "title": "Daily electricity demand (2 years, synthetic but realistic)",
        "teaches": "Two seasonalities at once (weekly + yearly) — the case that needs MSTL "
                   "or TBATS, and where a single-period STL under-fits.",
        "data_class": "continuous", "expect": "multiple seasonality, stationary in level",
        "frequency": "D",
    })


def retail() -> None:
    rng = np.random.default_rng(11)
    idx = pd.date_range("2015-01-01", periods=84, freq="MS")
    t = np.arange(84)
    trend = 500 + 4.2 * t
    season = 60 * np.sin(2 * np.pi * (t % 12) / 12 + 0.6) + 25 * np.cos(4 * np.pi * (t % 12) / 12)
    noise = rng.normal(0, 18, 84)
    df = pd.DataFrame({"Month": idx, "Sales": (trend + season + noise).round(1)})
    save("retail_sales", df, "Month", "Sales", {
        "title": "Monthly retail sales (7 years, synthetic)",
        "teaches": "A trend plus a two-harmonic seasonal pattern: useful for comparing "
                   "SARIMA with Fourier-term ARIMAX.",
        "data_class": "continuous", "expect": "trend+seasonal, non-stationary",
        "frequency": "MS",
    })


def covid_counts() -> None:
    """Monthly event counts — the count-data branch."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2018-01-01", periods=60, freq="MS")
    t = np.arange(60)
    base = 120 + 1.8 * t
    season = 25 * np.sin(2 * np.pi * (t % 12) / 12)
    lam = np.clip(base + season, 20, None)
    df = pd.DataFrame({"Month": idx, "Cases": rng.poisson(lam)})
    save("outbreak_counts", df, "Month", "Cases", {
        "title": "Monthly reported cases (Poisson-generated counts)",
        "teaches": "COUNT data. Poisson/NB models apply here; MAPE is meaningless and "
                   "ARIMA on counts needs a variance-stabilising transform first.",
        "data_class": "count", "expect": "count, overdispersed, trend+seasonal",
        "frequency": "MS",
    })


def intermittent_demand() -> None:
    """Lumpy spare-parts demand: >60% zeros."""
    rng = np.random.default_rng(19)
    idx = pd.date_range("2016-01-01", periods=96, freq="MS")
    n = len(idx)
    vals = np.zeros(n)
    event = rng.random(n) < 0.28
    vals[event] = rng.integers(1, 12, event.sum())
    df = pd.DataFrame({"Month": idx, "Units": vals.astype(int)})
    save("spare_parts", df, "Month", "Units", {
        "title": "Spare-parts demand (intermittent, 72% zeros)",
        "teaches": "INTERMITTENT demand. Croston/TSB beat ARIMA here, and MASE — not MAPE — "
                   "is the honest metric. A tool that offers ARIMA on this is misleading you.",
        "data_class": "intermittent", "expect": "intermittent, mostly zeros",
        "frequency": "MS",
    })


def structural_break() -> None:
    rng = np.random.default_rng(23)
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    t = np.arange(120)
    level = np.where(t < 60, 100.0, 165.0)          # a level shift at month 61
    season = 12 * np.sin(2 * np.pi * (t % 12) / 12)
    noise = rng.normal(0, 6, 120)
    df = pd.DataFrame({"Month": idx, "Index": (level + season + noise).round(2)})
    save("structural_break", df, "Month", "Index", {
        "title": "Series with a structural break (level shift in 2015)",
        "teaches": "A single level shift fakes a unit root for ADF. Zivot-Andrews and "
                   "Quandt-Andrews should find it; plain differencing hides it.",
        "data_class": "continuous", "expect": "level shift, candidate spurious unit root",
        "frequency": "MS",
    })


def random_walk() -> None:
    rng = np.random.default_rng(31)
    idx = pd.date_range("2012-01-01", periods=200, freq="MS")
    df = pd.DataFrame({"Month": idx, "Level": np.cumsum(rng.normal(0.4, 5, 200)).round(2)})
    save("random_walk", df, "Month", "Level", {
        "title": "Pure random walk (200 steps)",
        "teaches": "There is nothing to model but the last value: the naive forecast IS the "
                   "correct answer, and any 'significant trend' you find is spurious.",
        "data_class": "continuous", "expect": "unit root, no seasonality, drift",
        "frequency": "MS",
    })


def white_noise() -> None:
    rng = np.random.default_rng(37)
    idx = pd.date_range("2015-01-01", periods=180, freq="MS")
    df = pd.DataFrame({"Month": idx, "Value": rng.normal(50, 8, 180).round(2)})
    save("white_noise", df, "Month", "Value", {
        "title": "White noise (no pattern at all)",
        "teaches": "The null case. A good tool must be willing to say 'no model is needed' — "
                   "if yours always finds structure here, it is overfitting.",
        "data_class": "continuous", "expect": "stationary, no trend, no seasonality",
        "frequency": "MS",
    })


def promotions() -> None:
    """
    A series whose movement is *caused* by a column beside it.

    Most teaching datasets are one column, which makes every exogenous model unfalsifiable. This one
    has two: monthly units sold alongside the number of promotion days in that month. The promotion
    effect (about +90 units per promotion day) is larger than the trend and the noise, so a model
    that ignores the extra column has to be visibly worse — that is the whole point of the dataset.
    """
    rng = np.random.default_rng(41)
    idx = pd.date_range("2016-01-01", periods=108, freq="MS")
    t = np.arange(108)
    promo_days = rng.integers(0, 9, 108).astype(float)      # 0-8 promotion days per month
    trend = 400.0 + 1.6 * t
    season = 55.0 * np.sin(2 * np.pi * (t % 12) / 12 + 0.8)
    promo_effect = 88.0 * promo_days
    noise = rng.normal(0, 18, 108)
    units = trend + season + promo_effect + noise
    df = pd.DataFrame({"Month": idx, "Units": units.round(1), "PromoDays": promo_days.astype(int)})
    path = ASSETS / "promotions.csv"
    df.to_csv(path, index=False)
    MANIFEST.append({
        "id": "promotions",
        "file": "promotions.csv",
        "time_column": "Month",
        "value_column": "Units",
        "exogenous_columns": ["PromoDays"],
        "n_rows": int(len(df)),
        "title": "Retail units with promotion days (36 months x 3 years)",
        "teaches": "A second column that genuinely drives the first: every promotion day is worth about "
                   "88 units. A model that ignores it cannot reach the accuracy of one that uses it, "
                   "and the future promotion plan is known in advance, so the scenario is realistic.",
        "data_class": "continuous",
        "expect": "trend + season + a promotion effect that is visible in the residual autocorrelation",
        "frequency": "MS",
    })
    print(f"  wrote {path.name:26s} n={len(df):5d}  {MANIFEST[-1]['title']}")


def main() -> None:
    print("Generating example datasets → app/assets/examples/")
    for fn in (
        air_passengers, us_acc_deaths, lynx, wwwusage, johnson_johnson, sunspots,
        beer, electricity, retail, covid_counts, intermittent_demand,
        structural_break, random_walk, white_noise, promotions,
    ):
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            print(f"  !! {fn.__name__} failed: {exc}")

    (ASSETS / "manifest.json").write_text(json.dumps(MANIFEST, indent=2))
    print(f"\nWrote {len(MANIFEST)} datasets and manifest.json to {ASSETS}")


if __name__ == "__main__":
    main()
