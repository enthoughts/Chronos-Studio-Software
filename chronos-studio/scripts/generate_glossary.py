#!/usr/bin/env python3
"""
Generate the single-source Chronos Studio glossary (Part 19, §5.15: 120+ terms).

    python scripts/generate_glossary.py

Writes:
    GLOSSARY.md                        — committee-facing alphabetical reference
    frontend/src/copy/glossary.ts      — in-app tooltips (same definitions)

Keep this file as the ONLY place definitions are edited. Both outputs are
regenerated from TERMS, so the docs and the UI can never disagree. Entries are
(key, category, definition); the key is a stable slug used by <GlossaryTip>.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Acronyms and proper names whose display label is not a simple capitalisation.
DISPLAY_OVERRIDES: dict[str, str] = {
    "adf": "ADF", "kpss": "KPSS", "df_gls": "DF-GLS",
    "zivot_andrews": "Zivot–Andrews", "canova_hansen": "Canova–Hansen",
    "ocsb": "OCSB / CH", "hegy": "HEGY", "phillips_perron": "Phillips–Perron",
    "arch_lm": "ARCH-LM", "mackinnon_p": "MacKinnon p-values",
    "stl": "STL", "mstl": "MSTL", "ucm": "UCM", "hp_filter": "HP filter",
    "christiano_fitzgerald": "Christiano–Fitzgerald filter", "x13": "X-13ARIMA-SEATS",
    "f_t": "F_T (trend strength)", "f_s": "F_S (seasonal strength)",
    "fisher_g": "Fisher's g-test", "kruskal_wallis": "Kruskal–Wallis",
    "acf": "ACF", "pacf": "PACF", "ccf": "CCF",
    "bartlett_bands": "Bartlett bands", "ljung_box": "Ljung–Box",
    "arma": "ARMA", "arima": "ARIMA", "sarimax": "SARIMAX / ARIMAX",
    "auto_arima": "auto-ARIMA", "arfima": "ARFIMA", "ets": "ETS",
    "holt_winters": "Holt–Winters", "theta": "Theta method",
    "inar": "INAR(1)", "ingarch": "INGARCH", "croston": "Croston",
    "sba": "SBA", "tsb": "TSB", "adida": "ADIDA", "hawkes": "Hawkes process",
    "arch": "ARCH", "garch": "GARCH", "egarch": "EGARCH",
    "gjr_garch": "GJR-GARCH", "var": "VAR", "vecm": "VECM",
    "johansen": "Johansen test", "irf": "IRF", "fevd": "FEVD",
    "setar": "SETAR", "nna": "NNAR", "svr": "SVR",
    "mae": "MAE", "rmse": "RMSE", "mape": "MAPE", "smape": "sMAPE",
    "mase": "MASE", "rmsse": "RMSSE", "crps": "CRPS", "pit": "PIT",
    "rqr": "RQR", "dunn_smyth": "Dunn–Smyth residuals", "ks_test": "KS test",
    "diebold_mariano": "Diebold–Mariano", "clark_west": "Clark–West",
    "pi_coverage": "PI coverage", "dpi": "DPI", "wcag": "WCAG 2.1 AA",
    "aria_live": "ARIA live region", "ic_comparable": "IC comparability guard",
    "one_file_html": "One-file HTML", "chronos_archive": ".chronos archive",
}


def display_label(slug: str) -> str:
    return DISPLAY_OVERRIDES.get(slug, slug.replace("_", " ").capitalize())


# (slug, category, definition)
TERMS: list[tuple[str, str, str]] = [
    # ---------------------------------------------------------------- Basics
    ("time_series", "Basics", "A sequence of observations indexed in time order, usually at a fixed frequency. Order is the information: shuffling the rows destroys the analysis."),
    ("observation", "Basics", "One recorded value at one point in time. The count of observations is written n and is always shown beside any accuracy number."),
    ("frequency", "Basics", "How often observations are recorded — daily, monthly, quarterly. The app infers it and never compares models fitted at different effective frequencies."),
    ("period", "Basics", "The number of observations in one full seasonal cycle, e.g. 12 for monthly data with a yearly pattern."),
    ("season", "Basics", "The position within a cycle (January, Monday, hour 09). Distinct from seasonality, which is the repeating pattern itself."),
    ("trend", "Basics", "The slow, long-run movement in the level of a series. It need not be linear and is estimated, not assumed away."),
    ("level", "Basics", "The current underlying height of the series once short-term noise is smoothed out."),
    ("noise", "Basics", "The irregular remainder left after trend, season and cycle are removed. It should look unstructured; structure here is a finding for the modelling stage."),
    ("stationarity", "Basics", "A series is stationary when its level, spread and dependence structure do not drift over time. Most classical models assume it, so we usually stabilise the series first."),
    ("ergodicity", "Basics", "The property that time averages converge to the ensemble average. It is what makes a single historical path enough to estimate a model."),
    ("stochastic_process", "Basics", "A collection of random variables indexed by time; a time series is one realised path from such a process."),
    ("white_noise", "Basics", "Uncorrelated, zero-mean, constant-variance shocks. A white-noise series has nothing to forecast; the app can return a 'no model needed' verdict."),
    ("random_walk", "Basics", "A series whose next value is the last value plus a shock. It has a unit root and its naive forecast is optimal."),
    ("innovation", "Basics", "The one-step-ahead surprise, i.e. the residual after the model's conditional prediction. Residual diagnostics examine innovations, not fitted values."),
    ("lag", "Basics", "A past value or a past error; lag k means k time steps back."),
    ("lead", "Basics", "A future value. Forecasts are leads; leakage occurs when a lead influences a modelling choice."),
    ("sample", "Basics", "The observed window used to estimate a model. It is split into training and test blocks, never interleaved."),
    ("population", "Basics", "The hypothetical full data-generating process. Estimates describe the sample; intervals express uncertainty about the population."),
    ("parameter", "Basics", "A number estimated from data inside a model, e.g. an AR coefficient. Its count enters AIC, BIC and AICc."),
    ("estimate", "Basics", "A computed value of a parameter or quantity. Every number displayed is an estimate with a stated method, not a typed-in fact."),
    ("hypothesis_test", "Basics", "A procedure that compares a statistic against a null distribution and returns a p-value. It never proves the null; it can only fail to reject it."),
    ("p_value", "Basics", "The probability, assuming the null hypothesis is true, of seeing a statistic at least as extreme as the observed one. Small values are evidence against the null."),
    ("confidence_level", "Basics", "The nominal long-run coverage of an interval procedure, e.g. 95%. It describes the method, not a probability for this one interval."),
    ("significance", "Basics", "A pre-agreed threshold (usually 5%) at which a test is said to reject. The app prints the exact p-value so a reader can use their own threshold."),
    ("effect_size", "Basics", "The magnitude of an estimated effect, independent of sample size. Recommending a transform requires an effect, not just a small p-value."),
    ("robust", "Basics", "Insensitive to a few unusual observations. Robust STL and median-based slopes are preferred where spikes are common."),
    ("parametric", "Basics", "Assuming a finite, named distribution (e.g. Gaussian errors). Non-parametric or distribution-free methods avoid that assumption."),
    ("deterministic", "Basics", "Containing no random component: the same inputs always give the same output. Narratives and advisor rules in this app are deterministic."),
    ("seed", "Basics", "An integer that fixes every random draw — simulation paths, bootstrap bands, ML splits — so a study can be reproduced exactly."),

    # ------------------------------------------------------- Data & quality
    ("ingestion", "Data & quality", "Reading CSV, Excel, JSON, Parquet, Stata or SPSS into a dated series with an explicit column mapping."),
    ("column_mapping", "Data & quality", "The user's declaration of which column is time, which is the value, and which are exogenous. Nothing is guessed silently."),
    ("data_class", "Data & quality", "The branch chosen from the values themselves: continuous, count, intermittent, or event-timestamp. It decides which model families and metrics are legal."),
    ("missing_value", "Data & quality", "A gap in the recorded series. Gaps are counted, reported and imputed only with an explicit, previewable operation."),
    ("gap", "Data & quality", "A run of consecutive missing values. Short, medium and long gaps get different advice rather than one blanket imputation."),
    ("duplicate_timestamp", "Data & quality", "Two rows with the same time index. The app reports them and never silently drops a row."),
    ("constant_run", "Data & quality", "A stretch of identical values. Long runs can mean a stalled sensor rather than genuine stability."),
    ("impossible_value", "Data & quality", "A value that cannot occur (negative counts, a date outside the file's range). Flagged for review, never quietly deleted."),
    ("outlier", "Data & quality", "An observation far from what its neighbours and season predict. The app flags spikes, level shifts and temporary changes separately."),
    ("level_shift", "Data & quality", "A permanent step change in level. It is invisible in an STL remainder because STL absorbs it into the trend, so the app tests the deseasonalised series."),
    ("temporary_change", "Data & quality", "A spike that decays back to the previous level. Unlike a level shift, it should not be modelled as a step dummy."),
    ("zero_inflation", "Data & quality", "More zeros than a standard count distribution predicts. It triggers zero-inflated or hurdle models and a metric-policy switch."),
    ("overdispersion", "Data & quality", "Variance larger than the mean in count data. A mandatory pre-check before fitting Poisson, because it changes the whole likelihood."),
    ("row_order_is_time", "Data & quality", "The convention that physical row order defines time when no date column exists. It is recorded explicitly so it can be audited."),
    ("dayfirst", "Data & quality", "Whether an ambiguous date like 03/04/2020 means 3 April (true) or 4 March (false). Stored in the column mapping."),
    ("content_hash", "Data & quality", "A fingerprint of the loaded values. Reports and archives carry it so a reader can prove which data produced a number."),
    ("resolution", "Data & quality", "The smallest step at which values are recorded, e.g. 0.1. It prevents an outlier rule from flagging rounding artefacts."),
    ("calendar_heatmap", "Data & quality", "A plot of values by calendar position (day × week/month) that makes missing days and weekly patterns visible."),
    ("provenance", "Data & quality", "The record of which stage, method and data version produced a result. Every recommendation is traceable to a displayed statistic."),

    # ------------------------------------------------- Stationarity & tests
    ("unit_root", "Stationarity & tests", "A stochastic trend: shocks never die out and the variance grows with time. Differencing is the usual remedy, but only when the tests agree."),
    ("adf", "Stationarity & tests", "Augmented Dickey–Fuller test. Null: a unit root. Lags are chosen by AIC and MacKinnon p-values are used; three deterministic specifications are reported."),
    ("kpss", "Stationarity & tests", "Kwiatkowski–Phillips–Schmidt–Shin test. Null: stationarity. It is the mirror image of ADF, which is why the app reconciles the two rather than quoting one."),
    ("phillips_perron", "Stationarity & tests", "A unit-root test that corrects the ADF statistic non-parametrically for serial correlation and heteroskedasticity instead of adding lags."),
    ("df_gls", "Stationarity & tests", "DF-GLS: a GLS-detrended Dickey–Fuller test with better small-sample power than ordinary ADF."),
    ("zivot_andrews", "Stationarity & tests", "A unit-root test that allows one endogenous structural break in intercept, trend or both."),
    ("canova_hansen", "Stationarity & tests", "A test of seasonal stability; rejection indicates seasonal unit roots that need seasonal differencing."),
    ("ocsb", "Stationarity & tests", "Osborn–Chui–Smith–Birchenhall test for seasonal unit roots, used with HEGY and Canova–Hansen in the seasonal branch."),
    ("hegy", "Stationarity & tests", "Hylleberg–Engle–Granger–Yoo test that separates unit roots at the zero frequency from those at each seasonal frequency."),
    ("reconciliation", "Stationarity & tests", "The 2×2 rule combining a unit-root test and a stationarity test into one verdict: both stationary, both non-stationary, or a contradiction needing judgement."),
    ("difference", "Stationarity & tests", "Replacing each value with the change from the previous one. Removes a trend; applying it to an already stable series adds noise and breaks the model."),
    ("seasonal_difference", "Stationarity & tests", "Subtracting the value one full season ago. It removes a seasonal unit root while preserving the annual shape."),
    ("integration_order", "Stationarity & tests", "The number of differences needed for stationarity, written d in ARIMA(p,d,q). The app refuses more than two without a recorded justification."),
    ("structural_break", "Stationarity & tests", "A date at which the data-generating process changes. Break tests include Chow, Quandt–Andrews, Bai–Perron and CUSUM."),
    ("chow_test", "Stationarity & tests", "An F-test for a break at a known, user-specified date."),
    ("quandt_andrews", "Stationarity & tests", "A Chow test maximised over all candidate break dates, with a supremum statistic rather than a known breakpoint."),
    ("bai_perron", "Stationarity & tests", "A procedure for estimating several unknown break dates simultaneously with confidence intervals."),
    ("cusum", "Stationarity & tests", "A cumulative-sum test on recursive residuals for slow parameter drift."),
    ("mann_kendall", "Stationarity & tests", "A distribution-free rank test for monotone trend. The app runs it on the deseasonalised series, not on a loess-smoothed trend."),
    ("sens_slope", "Stationarity & tests", "Theil–Sen slope: the median of all pairwise slopes, with a distribution-free interval. Robust to a handful of extreme values."),
    ("cox_stuart", "Stationarity & tests", "A sign test comparing the second half of a series with the first; catches slow drift without reacting to one jump."),
    ("heteroskedasticity", "Stationarity & tests", "Variance that changes over time. ARCH-LM on the remainder detects it and points to a log/Box–Cox transform or a GARCH model."),
    ("arch_lm", "Stationarity & tests", "Engle's Lagrange-multiplier test for autoregressive conditional heteroskedasticity in residuals."),
    ("normality_battery", "Stationarity & tests", "Shapiro–Wilk, Jarque–Bera and D'Agostino–Pearson applied together, because each is sensitive to a different departure."),
    ("mackinnon_p", "Stationarity & tests", "Response-surface p-values for Dickey–Fuller statistics, used because the null distribution is non-standard."),

    # ------------------------------------------ Decomposition & components
    ("decomposition", "Decomposition & components", "Splitting a series into trend, seasonal, cycle and remainder so each can be measured and described separately."),
    ("classical_decomposition", "Decomposition & components", "Moving-average based additive or multiplicative decomposition. Simple, but it cannot handle a changing seasonal shape."),
    ("stl", "Decomposition & components", "Seasonal-Trend decomposition using Loess: robust, handles evolving seasonality, and supplies the remainder that trend and adequacy tests use."),
    ("mstl", "Decomposition & components", "Multiple STL: decomposes several seasonal periods at once, e.g. daily and weekly cycles in hourly data."),
    ("ucm", "Decomposition & components", "Unobserved-components (structural) model estimated by Kalman filtering; handles missing values natively."),
    ("hp_filter", "Decomposition & components", "Hodrick–Prescott filter separating a smooth trend from a cycle. Its second component is a cycle, not a season."),
    ("christiano_fitzgerald", "Decomposition & components", "A band-pass filter isolating cyclical frequencies; like HP it produces a cycle rather than a seasonal component."),
    ("x13", "Decomposition & components", "X-13ARIMA-SEATS, the official seasonal-adjustment program. Used only when its external binary is installed; otherwise flagged."),
    ("f_t", "Decomposition & components", "Hyndman's trend-strength measure, 1 minus the variance ratio of remainder to trend-plus-remainder. Capped by the Mann–Kendall verdict."),
    ("f_s", "Decomposition & components", "Hyndman's seasonal-strength measure, computed from an STL fit; reported only when the seasonal model is valid."),
    ("variance_share", "Decomposition & components", "The fraction of total variation attributed to one component. The scorecard ranks components by it."),
    ("seasonal_index", "Decomposition & components", "The average effect of each season position, in units and as a percentage of the mean."),
    ("periodogram", "Decomposition & components", "An estimate of the spectral density. Its peaks reveal cycles, but a tallest peak always exists even in noise."),
    ("fisher_g", "Decomposition & components", "Fisher's exact test for whether the largest periodogram peak is more than spectral noise would produce."),
    ("kruskal_wallis", "Decomposition & components", "A distribution-free test of whether seasonal groups differ, used as evidence that a seasonal pattern is real."),
    ("amplitude_growing", "Decomposition & components", "Whether the peak-to-trough seasonal swing widens with the level. It motivates a multiplicative model or a variance-stabilising transform."),
    ("cycle", "Decomposition & components", "A rise and fall with no fixed calendar length — unlike seasonality, you cannot predict when the next turn comes."),
    ("multiplicative", "Decomposition & components", "A structure where the seasonal swing scales with the level. It is blocked on non-positive data, exactly like a log transform."),

    # ---------------------------------- Correlation & model identification
    ("autocorrelation", "Correlation & model identification", "Correlation between a series and a lagged copy of itself; the basic measure of memory."),
    ("acf", "Correlation & model identification", "Autocorrelation function across lags. Slow decay suggests non-stationarity; a sharp cutoff suggests an MA order."),
    ("pacf", "Correlation & model identification", "Partial autocorrelation: correlation at a lag after removing the effect of all shorter lags. A cutoff suggests an AR order."),
    ("ccf", "Correlation & model identification", "Cross-correlation between two series at each lag, used to time-align an exogenous regressor."),
    ("bartlett_bands", "Correlation & model identification", "Approximate ±1.96/√n confidence bands for a correlogram under white noise."),
    ("ljung_box", "Correlation & model identification", "A portmanteau test that a whole block of residual autocorrelations is jointly zero, with a lag-adjusted null distribution."),
    ("order_hint", "Correlation & model identification", "A suggested p or q from the ACF/PACF pattern. It is a starting point, always checked by information criteria and CV."),
    ("correlogram", "Correlation & model identification", "The plotted ACF/PACF with confidence bands and a plain-language pattern classification."),
    ("leakage", "Correlation & model identification", "Any use of test-period information in a modelling choice. Rolling-origin CV and training-only scaling exist to prevent it."),
    ("benchmark", "Correlation & model identification", "A deliberately simple reference model (naive, seasonal naive, mean, drift) that every candidate must beat out of sample."),
    ("information_criterion", "Correlation & model identification", "AIC, AICc or BIC: fit penalised for parameter count. Comparable only across models with the same differencing and effective sample."),
    ("model_registry", "Correlation & model identification", "The single table of ModelSpec rows describing every model, its data class, assumptions and required sample size."),

    # --------------------------------------------------------------- Models
    ("arma", "Models", "Autoregressive moving-average model for a stationary series, ARMA(p,q)."),
    ("arima", "Models", "ARMA applied after d differences, ARIMA(p,d,q); seasonal terms give SARIMA(p,d,q)(P,D,Q)m."),
    ("sarimax", "Models", "SARIMA with exogenous regressors, used for interventions (pulse/step dummies) and future scenario paths."),
    ("auto_arima", "Models", "A bounded order search by AICc with stationarity and invertibility root checks; a deterministic grid is the fallback when pmdarima is absent."),
    ("arfima", "Models", "Fractionally integrated ARMA for long memory, where the differencing parameter d is non-integer."),
    ("ets", "Models", "Error-Trend-Seasonal exponential smoothing with an automatic additive/multiplicative, damped grid and a non-positive-data guard."),
    ("holt_winters", "Models", "Classical exponential smoothing with explicit level, trend and seasonal updating equations."),
    ("theta", "Models", "The Theta method, equivalent to a drifted SES decomposition; θ controls how much the curvature is retained."),
    ("damped_trend", "Models", "A trend whose slope decays with horizon, preventing absurdly long-run linear extrapolation."),
    ("naive", "Models", "Forecast equals the last observed value. The mandatory baseline; MASE is scaled by its in-sample error."),
    ("seasonal_naive", "Models", "Forecast equals the value one season ago. The correct baseline for strongly seasonal series."),
    ("drift", "Models", "A random walk plus an average drift estimated from first differences."),
    ("mean_model", "Models", "Forecast equals the historical mean; the baseline for a stationary, structureless series."),
    ("poisson_model", "Models", "A count model with a log link and time-varying intensity; requires an overdispersion pre-check."),
    ("negative_binomial", "Models", "NB2 count model allowing variance to exceed the mean, compared to Poisson by a likelihood-ratio test and AIC."),
    ("zero_inflated", "Models", "A two-part model for excess zeros: one process generates structural zeros, another the counts."),
    ("hurdle", "Models", "A two-part model where a binary process decides whether any event occurs, then a truncated count model gives the size."),
    ("inar", "Models", "Integer-valued autoregression using binomial thinning; flagged as advanced in the registry."),
    ("ingarch", "Models", "Integer-valued GARCH: a count model whose conditional mean follows a GARCH-like recursion."),
    ("croston", "Models", "Intermittent-demand method that separately smooths non-zero sizes and inter-arrival intervals."),
    ("sba", "Models", "Syntetos–Boylan approximation, a bias-corrected variant of Croston."),
    ("tsb", "Models", "Teunter–Syntetos–Babai method that updates the occurrence probability every period, including zero-demand periods."),
    ("adida", "Models", "Aggregate-Disaggregate Intermittent Demand Approach: aggregate to a non-intermittent scale, forecast, then disaggregate."),
    ("renewal_process", "Models", "A model of event timestamps based on the distribution of inter-arrival times."),
    ("hawkes", "Models", "A self-exciting point process where each event raises the short-term intensity of further events."),
    ("arch", "Models", "Autoregressive conditional heteroskedasticity: today's variance depends on past squared shocks."),
    ("garch", "Models", "Generalised ARCH with variance depending on past variances as well as squared shocks; produces both a mean and a variance forecast."),
    ("egarch", "Models", "Exponential GARCH modelling log-variance, so it captures leverage without parameter positivity constraints."),
    ("gjr_garch", "Models", "Glosten–Jagannathan–Runkle GARCH with an asymmetric response to negative shocks."),
    ("leverage_effect", "Models", "The tendency of negative returns to raise future volatility more than positive returns of the same size."),
    ("var", "Models", "Vector autoregression: every variable is regressed on lags of itself and of the others."),
    ("granger_causality", "Models", "A predictive, not causal, test: X Granger-causes Y if lags of X improve a forecast of Y."),
    ("irf", "Models", "Impulse-response function: the path of a system after a one-unit shock, with bootstrap bands."),
    ("fevd", "Models", "Forecast-error variance decomposition: the share of each variable's forecast uncertainty attributed to each shock."),
    ("vecm", "Models", "Vector error-correction model for cointegrated series, with a spurious-regression guard."),
    ("johansen", "Models", "A likelihood-based test for the number of cointegrating relationships in a system."),
    ("setar", "Models", "Self-exciting threshold autoregression: different AR regimes depending on a lagged value crossing a threshold."),
    ("markov_switching", "Models", "A model whose parameters switch between regimes according to a latent Markov chain."),
    ("nna", "Models", "Neural-network autoregression using lagged values as features."),
    ("gaussian_process", "Models", "A non-parametric Bayesian regression over functions, giving a predictive distribution rather than just a point."),
    ("random_forest", "Models", "An ensemble of decision trees on leakage-safe lag features, always benchmarked against ETS/ARIMA even when it loses."),
    ("gradient_boosting", "Models", "LightGBM/XGBoost boosted trees with a mandatory model card stating features, tuning and benchmark comparison."),
    ("svr", "Models", "Support-vector regression on lag features with an explicit kernel and scaling."),
    ("forecast_combination", "Models", "Mean, median, information-criterion-weighted or Granger–Ramanathan pooling of component forecasts; requires component forecasts first."),
    ("model_card", "Models", "A short record of a fitted model's features, assumptions, training window and out-of-sample benchmark, required for every ML entrant."),
    ("fit_failure", "Models", "A fit that did not converge is reported as a failure with a reason; it never silently becomes an empty result or a fabricated number."),
    ("root_check", "Models", "Verification that AR and MA polynomials have roots outside the unit circle, i.e. the fit is stationary and invertible."),

    # ---------------------------------------------------------- Forecasting
    ("forecast", "Forecasting", "A predictive distribution for future observations, not just a single number. The app attaches coverage and a residual-adequacy verdict."),
    ("horizon", "Forecasting", "How many steps ahead are predicted. The app flags horizons beyond roughly one fifth of the training length."),
    ("point_forecast", "Forecasting", "A single central value (mean or median) for each future step."),
    ("prediction_interval", "Forecasting", "An interval intended to contain the future observation with a stated nominal coverage and a stated construction method."),
    ("analytic_interval", "Forecasting", "An interval from a closed-form forecast variance under a Gaussian assumption."),
    ("simulation_interval", "Forecasting", "An interval from quantiles of seeded simulated paths, used when no analytic variance exists."),
    ("fan_chart", "Forecasting", "A plot of several nested prediction intervals with deepening shades, so uncertainty widening with horizon is visible."),
    ("back_transformation", "Forecasting", "Returning a forecast to the original scale after a log/Box–Cox transform, with a bias adjustment because E[exp X] ≠ exp(E[X])."),
    ("scenario", "Forecasting", "A user-supplied future path for an exogenous regressor, e.g. a marketing-spend plan."),
    ("hierarchical_reconciliation", "Forecasting", "Adjusting forecasts at different aggregation levels so they add up consistently."),
    ("sharpness", "Forecasting", "How narrow a predictive distribution is. Valued only subject to calibration — a narrow but dishonest interval is worse."),
    ("calibration", "Forecasting", "Whether empirical coverage matches nominal coverage across interval levels."),
    ("discrete_predictive_distribution", "Forecasting", "A full probability mass function over future counts, needed for CRPS and PIT on count or intermittent data."),
    ("horizon_advisor", "Forecasting", "Guidance on a defensible horizon given sample length, frequency and seasonal period."),

    # ----------------------------------------------------------- Evaluation
    ("mae", "Evaluation", "Mean absolute error. Robust, in the units of the series, and the primary point error on zero-heavy data."),
    ("rmse", "Evaluation", "Root mean squared error; penalises large misses more than MAE."),
    ("mape", "Evaluation", "Mean absolute percentage error. Banned on intermittent and zero-heavy data because it is undefined or explodes near zero."),
    ("smape", "Evaluation", "Symmetric absolute percentage error, capped at 200%; shown only for comparability, never as a genuine percentage error."),
    ("mase", "Evaluation", "Mean Absolute Scaled Error: MAE divided by the in-sample naive error. Below 1 beats 'tomorrow equals today'; 0 for a perfect forecast, 1 for the naive one."),
    ("rmsse", "Evaluation", "Root Mean Squared Scaled Error, the RMSE analogue of MASE, using the same training-only denominator."),
    ("mase_by_horizon", "Evaluation", "MASE computed separately at each forecast step, revealing whether a model degrades gracefully with horizon."),
    ("crps", "Evaluation", "Continuous Ranked Probability Score: a proper score for a whole predictive distribution, rewarding accuracy and honest uncertainty (Gneiting & Raftery, 2007)."),
    ("pinball_loss", "Evaluation", "Quantile (check) loss averaged over quantile levels; the proper score behind quantile forecasts."),
    ("log_score", "Evaluation", "Minus the log predictive density at the observation. Proper but sensitive to tails; the continuous-case caveat is stated."),
    ("winkler_score", "Evaluation", "Interval score penalising both width and misses outside a prediction interval."),
    ("pi_coverage", "Evaluation", "Empirical fraction of observations inside an interval, compared with its nominal coverage."),
    ("pit", "Evaluation", "Probability Integral Transform: if the forecast distribution is honest these values are uniform; systematic shapes reveal over- or under-confidence."),
    ("rqr", "Evaluation", "Randomised Quantile Residuals: normal-like residuals for discrete or non-normal models, obtained by randomising within predictive probabilities."),
    ("dunn_smyth", "Evaluation", "Dunn–Smyth residuals for count models, a randomised normal residual used in the discrete diagnostic battery."),
    ("ks_test", "Evaluation", "Kolmogorov–Smirnov test of PIT uniformity against the uniform distribution."),
    ("diebold_mariano", "Evaluation", "A test of whether two forecasts have equal expected loss, with HAC-corrected variance for multi-step errors."),
    ("clark_west", "Evaluation", "A small-sample correction to the Diebold–Mariano comparison for nested models."),
    ("rolling_origin_cv", "Evaluation", "Time-ordered cross-validation that expands the training window and always keeps the test block in the future."),
    ("fold", "Evaluation", "One train/test split in rolling-origin CV; the app reports fold-level accuracy, not just the average."),
    ("relative_metric", "Evaluation", "Any error divided by a benchmark error or a training-only scale. §6.4 forbids computing one before the naive benchmark is fitted."),
    ("metric_policy", "Evaluation", "The rule engine that bans MAPE on zero-heavy data, switches to CRPS when a probabilistic forecast exists, and always prints a reason."),
    ("out_of_sample", "Evaluation", "Computed only on observations held out of fitting. In-sample fit is never reported as forecast accuracy."),
    ("ic_comparable", "Evaluation", "The guard that refuses AIC/BIC comparisons across different differencing orders or effective sample sizes."),

    # -------------------------------------------------------- Preprocessing
    ("imputation", "Preprocessing", "Filling missing values with an explicit, previewable rule (seasonal, Kalman) rather than a silent default."),
    ("transform", "Preprocessing", "A monotone rescaling of the values, e.g. log, square root, Box–Cox or Yeo–Johnson."),
    ("box_cox", "Preprocessing", "A power transform indexed by λ; λ=0 is a log, λ=1 means no transform. Its confidence interval is printed with the recommendation."),
    ("log_transform", "Preprocessing", "Taking logarithms to stabilise a variance that grows with the level. Blocked on non-positive values; Yeo–Johnson is used instead."),
    ("yeojohnson", "Preprocessing", "A power transform defined for zero and negative values too, behaving like a log on positive values."),
    ("smoothing", "Preprocessing", "Averaging neighbouring values to reduce measurement noise; it changes the series and therefore invalidates downstream results."),
    ("aggregation", "Preprocessing", "Summing or averaging to a coarser frequency, e.g. daily to weekly, to remove an intermittent zero problem."),
    ("outlier_treatment", "Preprocessing", "Flagging or replacing unusual observations. The app prefers flagging (and intervention dummies) to deleting rows."),
    ("preview", "Preprocessing", "Running the exact same operation on a copy of the series before committing, so what the user sees is what happens."),
    ("undo", "Preprocessing", "Stepping back one treatment by replaying the remaining chain from the untouched original series."),
    ("redo", "Preprocessing", "Re-applying the most recently undone treatment; a new treatment clears the redo history, exactly like an editor."),
    ("series_version", "Preprocessing", "A counter bumped whenever the series changes. Results record the version they describe so stale panels can be marked."),
    ("stale", "Preprocessing", "A completed stage whose inputs have since changed. The UI says 'Needs re-running' instead of reusing an obsolete number."),
    ("advisor", "Preprocessing", "A rule engine that ranks recommendations from computed statistics and attaches the evidence, confidence and a 'what if ignored' note."),
    ("readiness_gate", "Preprocessing", "The S8 checklist of blockers and caveats that must be resolved before model suggestion is allowed."),
    ("caveat", "Preprocessing", "A qualification that travels with downstream results so a forecast or report never hides a known limitation."),

    # ------------------------------- Reporting, reproducibility & privacy
    ("report_bundle", "Reporting & reproducibility", "A ZIP containing every requested format plus a manifest; builder failures are listed in errors.txt rather than hidden."),
    ("one_file_html", "Reporting & reproducibility", "A self-contained report with embedded figures and no external CSS, scripts or fonts, so it works offline."),
    ("reproducible_snippet", "Reporting & reproducibility", "A Python or R script exported with the analysis that regenerates the numbers from the raw data; it must parse as valid source."),
    ("chronos_archive", "Reporting & reproducibility", "A .chronos session export containing the pipeline state and results, so a study can be reloaded exactly."),
    ("scorecard", "Reporting & reproducibility", "Ten weighted completeness checks with exact result links and the panel to visit for each outstanding item."),
    ("verdict", "Reporting & reproducibility", "The exact manuscript-readiness sentence: ready, nearly ready with two outstanding items, or not yet with blockers."),
    ("data_behind", "Reporting & reproducibility", "The exact numbers behind any plot or table, downloadable as XLSX/CSV so a figure is never the only record."),
    ("caption", "Reporting & reproducibility", "A one-sentence description of a figure or table, generated from the computed values, not hand-written."),
    ("alt_text", "Reporting & reproducibility", "A screen-reader description of a figure or table that summarises the actual values, not just 'chart'."),
    ("dpi", "Reporting & reproducibility", "Dots per inch; publication figures are exported at 300 dpi plus vector SVG and PDF."),
    ("session_ttl", "Reporting & reproducibility", "The in-memory lifetime of a session. There is no server-side database and no persistence of user data beyond this window."),
    ("delete_session", "Reporting & reproducibility", "A button that hard-deletes the session from server memory immediately."),
    ("privacy_note", "Reporting & reproducibility", "A plain-English statement of what is stored, what is sent to an AI service, and what is exported only on request."),
    ("payload_preview", "Reporting & reproducibility", "A review of exactly what would be sent to the AI service before the first live request; data values are off by default."),
    ("analysis_digest", "Reporting & reproducibility", "The compact, number-only summary of computed results that grounds every AI answer; no raw values are included by default."),
    ("offline_mode", "Reporting & reproducibility", "The deterministic narrative assistant that needs no API key; removing the AI layer leaves a fully functional app."),
    ("library_versions", "Reporting & reproducibility", "The recorded versions of numpy/pandas/statsmodels/scikit-learn used for a study, so a result can be reproduced later."),
    ("golden_file", "Reporting & reproducibility", "A stored, reviewed reference output that a test compares against, so an unintended statistical change fails the build."),
    ("coverage", "Reporting & reproducibility", "The share of source lines executed by the test suite. §5.15 requires at least 80% on ts/."),
    ("demo_mode", "Reporting & reproducibility", "One click that replays a complete seed study through S0–S15 with progress and cancel, producing a real explorable analysis."),
    ("seed_data", "Reporting & reproducibility", "The bundled example datasets shipped with the image so a new user needs no upload and the build needs no network."),
    ("background_job", "Reporting & reproducibility", "Heavy work (CV grids, ML fits, demo replays) executed off the request thread with progress, per-step timings and cancel."),
    ("methods_document", "Reporting & reproducibility", "METHODS.md: exactly which test, implementation and assumption is used for every reported number — the document a thesis committee reads."),

    # ---------------------------------------------- Product & accessibility
    ("ribbon", "Product & accessibility", "The JASP-style command bar. It implements the WAI-ARIA toolbar pattern: one tab stop, arrow keys between commands, Home/End to jump."),
    ("stepper", "Product & accessibility", "The pipeline navigation bar; completed steps can be revisited, and gated steps explain which earlier step is missing."),
    ("simplify_view", "Product & accessibility", "A mode that hides all but the eight essential panels so a beginner is not confronted with the full 16-stage ribbon."),
    ("plain_language_mode", "Product & accessibility", "A toggle that suppresses the technical layer of a result card and keeps the plain-English reading."),
    ("guided_tour", "Product & accessibility", "A step-by-step walkthrough of a worked analysis with next, back and skip controls."),
    ("tooltip", "Product & accessibility", "A short definition shown on hover and on keyboard focus; the glossary supplies 120+ of them and never relies on colour alone."),
    ("wcag", "Product & accessibility", "Web Content Accessibility Guidelines 2.1 AA: the target standard for contrast, keyboard operation and screen-reader labels."),
    ("colorblind_safe", "Product & accessibility", "A palette distinguishable under the common colour-vision deficiencies; every status also carries text and an icon."),
    ("no_color_only", "Product & accessibility", "The rule that colour never carries meaning by itself — verdicts add words, icons and shapes."),
    ("reduced_motion", "Product & accessibility", "Respecting the operating-system 'prefers-reduced-motion' setting by suppressing transitions and animations."),
    ("screen_reader", "Product & accessibility", "Assistive technology that reads the accessible name and description of each control, figure and table."),
    ("aria_live", "Product & accessibility", "A region that announces asynchronous changes (progress, errors, toasts) without moving keyboard focus."),
    ("focus_visible", "Product & accessibility", "A clear keyboard focus ring that appears for keyboard users but not on mouse clicks."),
    ("table_caption", "Product & accessibility", "A visible title on every data table, paired with a value summary for screen readers."),
    ("error_library", "Product & accessibility", "Plain-English error messages with a code, a fix and a technical detail, so a failure is never a dead end."),
]


def main() -> None:
    slugs = [t[0] for t in TERMS]
    assert len(slugs) == len(set(slugs)), "duplicate glossary slug"
    assert len(TERMS) >= 120, f"need 120+ terms, got {len(TERMS)}"
    categories: dict[str, list[tuple[str, str]]] = {}
    for slug, cat, definition in TERMS:
        categories.setdefault(cat, []).append((slug, definition))

    # ---- GLOSSARY.md ---------------------------------------------------- #
    lines = [
        "# Chronos Studio — Glossary",
        "",
        f"**{len(TERMS)} terms**, grouped by topic and listed alphabetically within each group. "
        "The same definitions power the in-app tooltips: both are generated by "
        "`scripts/generate_glossary.py` from a single source, so they cannot disagree.",
        "",
        "Definitions are deliberately short and in plain English. For the exact test, "
        "implementation and assumption behind each number, see [METHODS.md](METHODS.md).",
        "",
        "## Contents",
        "",
    ]
    for cat in categories:
        anchor = cat.lower().replace(" ", "-").replace("&", "")
        lines.append(f"- [{cat}](#{anchor}) — {len(categories[cat])} terms")
    lines.append("")
    for cat, entries in categories.items():
        lines += [f"## {cat}", ""]
        for slug, definition in sorted(entries, key=lambda e: e[0]):
            title = display_label(slug)
            lines += [f"**{title}** (`{slug}`) — {definition}", ""]
    lines += [
        "---",
        "",
        f"*Generated by `scripts/generate_glossary.py` · {len(TERMS)} terms · "
        "Part 19 (§5.15 requires 120+).*",
        "",
    ]
    (ROOT / "GLOSSARY.md").write_text("\n".join(lines))

    # ---- frontend/src/copy/glossary.ts ---------------------------------- #
    entries = [
        {"slug": slug, "term": display_label(slug),
         "category": cat, "definition": definition}
        for slug, cat, definition in TERMS
    ]
    ts = [
        "/**",
        " * copy/glossary.ts — the single in-app glossary (Part 19, §5.15).",
        " *",
        " * GENERATED FILE. Do not edit by hand: run",
        " *   python scripts/generate_glossary.py",
        " * from the repository root. GLOSSARY.md is generated from the same source.",
        f" * {len(TERMS)} terms; every one is reachable as a tooltip via <GlossaryTip>.",
        " */",
        "",
        "export interface GlossaryEntry {",
        "  slug: string;",
        "  term: string;",
        "  category: string;",
        "  definition: string;",
        "}",
        "",
        "export const GLOSSARY: Readonly<Record<string, GlossaryEntry>> = Object.freeze({",
    ]
    for e in entries:
        d = json.dumps(e["definition"], ensure_ascii=True)
        t = json.dumps(e["term"], ensure_ascii=True)
        c = json.dumps(e["category"], ensure_ascii=True)
        ts.append(f'  "{e["slug"]}": Object.freeze({{ slug: "{e["slug"]}", term: {t}, category: {c}, definition: {d} }}),')
    ts += [
        "});",
        "",
        f"export const GLOSSARY_TERM_COUNT = {len(TERMS)};",
        "",
        "export const GLOSSARY_CATEGORIES: readonly string[] = Object.freeze([",
    ]
    for cat in categories:
        ts.append(f"  {json.dumps(cat)},")
    ts += [
        "]);",
        "",
        "/** Look up by slug, display term or a case-insensitive loose phrase. */",
        "export function lookupGlossary(input: string): GlossaryEntry | undefined {",
        "  if (!input) return undefined;",
        '  const key = input.trim().toLowerCase().replace(/[\\s-]+/g, "_");',
        "  if (GLOSSARY[key]) return GLOSSARY[key];",
        "  return Object.values(GLOSSARY).find(",
        '    (e) => e.term.toLowerCase().replace(/[\\s-]+/g, "_") === key',
        '      || e.slug.toLowerCase() === key,',
        "  );",
        "}",
        "",
        "/** The eight panels kept visible by \"Simplify view\" (§5.15). */",
        "export const SIMPLIFY_ESSENTIAL_PANELS: readonly string[] = Object.freeze([",
        '  "upload", "examples", "descriptives", "advice", "stationarity",',
        '  "suggest", "forecast", "report",',
        "]);",
        "",
        "export function isEssentialPanel(id: string): boolean {",
        "  return SIMPLIFY_ESSENTIAL_PANELS.includes(id);",
        "}",
        "",
    ]
    (ROOT / "frontend" / "src" / "copy" / "glossary.ts").write_text("\n".join(ts))

    print(f"WROTE GLOSSARY.md and frontend/src/copy/glossary.ts — {len(TERMS)} terms, "
          f"{len(categories)} categories")


if __name__ == "__main__":
    main()
