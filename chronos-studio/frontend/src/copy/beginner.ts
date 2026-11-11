/**
 * copy/beginner.ts — the Part 19 "what is this?" layer.
 *
 * CRAFT §5.15 asks for an explanation next to every panel, not just a tooltip
 * on one term. This module is a plain data table: the drawer component only
 * renders it, so the words can be reviewed and unit-tested without a DOM.
 * All user-facing prose lives here or in strings.ts (§4.6).
 */

export interface PanelHelp {
  /** Short sentence a first-year student can read in one breath. */
  summary: string;
  /** What the numbers or controls on the panel actually mean. */
  reading: string;
  /** The one safe action to try next. */
  tryThis: string;
  /** A common misinterpretation, phrased as a warning. */
  watchOut: string;
  /** Slugs from the generated 242-term glossary. */
  terms: readonly string[];
}

export const BEGINNER_COPY = Object.freeze({
  eyebrow: "Part 19 · beginner help",
  drawerTitle: "What is this?",
  drawerIntro: "A one-minute explanation of this panel. Underlined words open a plain-English definition on hover or keyboard focus.",
  readingLabel: "What you are looking at",
  tryLabel: "Try this",
  watchLabel: "Common misunderstanding",
  termsLabel: "Words you will meet",
  close: "Close explanation",
  open: "What is this?",
  back: "Back",
  next: "Next",
  backLabel: "Go back one pipeline step",
  nextLabel: "Go forward one pipeline step",
  historyTitle: "Recent changes",
  historyHint: "Preprocessing is reversible. Undo and redo replay the chain from the untouched original series; later steps are marked stale.",
  undo: "Undo",
  redo: "Redo",
  undoLabel: "Undo the last preprocessing treatment",
  redoLabel: "Redo the last undone preprocessing treatment",
  noUndo: "Nothing to undo yet.",
  noRedo: "Nothing to redo yet. Undo a treatment first.",
  errorTitle: "Plain-English error library",
  errorIntro: "Every failure has a code, a plain explanation, a fix, and a technical detail. Your analysis is never silently changed.",
  errorSearch: "Search errors, explanations and fixes…",
  errorOpen: "Error help",
  errorOpenLabel: "Open the plain-English error library",
  errorToastLink: "Open error library",
  errorToastMeaning: "What this means",
  errorAll: "All topics",
  errorEmpty: "No error matches that search.",
  selectionHint: "Select any underlined or familiar term to see its glossary definition.",
});

/** Each ribbon command resolves to the stage that explains it. */
export const PANEL_ALIASES: Readonly<Record<string, string>> = Object.freeze({
  upload: "S0", examples: "S0", mapping: "S0",
  descriptives: "S1", plots: "S1", outliers: "S1", quality: "S1",
  advice: "S2", transform: "S3", decompose: "S4",
  stationarity: "S5", breaks: "S5", acf: "S6", next: "S7",
  suggest: "S9", fit: "S10", select: "S11", diagnose: "S12",
  forecast: "S13", evaluate: "S14", report: "S15", catalog: "S15",
  ask: "S15", explain: "S15",
});

export const PANEL_HELP: Readonly<Record<string, PanelHelp>> = Object.freeze({
  S0: {
    summary: "Bring your data in and say which column is time and which is the value.",
    reading: "The data-quality report counts observations, gaps, duplicates and unusable values. The inferred frequency and data class decide which methods are legal later.",
    tryThis: "Open one of the nineteen built-in example series, then read the data-quality note before doing anything else.",
    watchOut: "A spreadsheet row is not automatically a date. A wrong time column makes every later result meaningless.",
    terms: ["ingestion", "column_mapping", "frequency", "data_class", "content_hash"],
  },
  S1: {
    summary: "Look at the series before changing it: shape, gaps, spikes and seasonality.",
    reading: "Descriptives describe the observed window, plots show order and shape, and outlier labels distinguish a one-off spike from a permanent level shift.",
    tryThis: "Read the time plot left to right and check whether a suspicious point is a spike, a step, or a genuine seasonal peak.",
    watchOut: "A large value is not automatically an error, and a missing date is not the same as a missing value.",
    terms: ["outlier", "level_shift", "temporary_change", "missing_value", "calendar_heatmap"],
  },
  S2: {
    summary: "The advisor explains which fixes the data actually needs, with evidence.",
    reading: "Each recommendation names a rule, the statistic behind it, a preview, and a reason. A recommendation is never applied without your confirmation.",
    tryThis: "Open a recommendation's preview and compare the before and after series before committing to it.",
    watchOut: "A small p-value alone does not justify a transform; the advisor also requires a real effect size.",
    terms: ["advisor", "preview", "hypothesis_test", "effect_size", "imputation"],
  },
  S3: {
    summary: "Apply agreed treatments to produce a versioned, reversible prepared series.",
    reading: "The step strip lists every treatment in order, with its method and reason. Undo and redo replay that chain from the untouched original series.",
    tryThis: "Apply one treatment, press Undo, then Redo. Re-run Step 1 afterwards so the profile matches the new version.",
    watchOut: "Changing the series makes downstream results stale; stale results are shown, never quietly reused as current.",
    terms: ["series_version", "undo", "redo", "transform", "stale"],
  },
  S4: {
    summary: "Split the series into trend, season and remainder so each can be inspected.",
    reading: "Strength measures describe how much variation belongs to each component. For long series the bounded Mann–Kendall and Theil–Sen checks keep this interactive.",
    tryThis: "Compare classical and STL decompositions; look for structure left in the remainder rather than just a pretty trend line.",
    watchOut: "A decomposition is a description, not a causal explanation, and an STL trend can absorb a sudden level shift.",
    terms: ["decomposition", "stl", "mstl", "trend", "season"],
  },
  S5: {
    summary: "Check whether the series has a stable level, spread and dependence structure.",
    reading: "ADF, KPSS, Phillips–Perron and break tests disagree by design; the reconciliation matrix explains the combined verdict instead of hiding the conflict.",
    tryThis: "Read the reconciliation row before the individual tests, then inspect any structural break it names.",
    watchOut: "Failing to reject a unit root is not proof that a series has one, and KPSS bandwidth choices can change a statistic.",
    terms: ["stationarity", "unit_root", "adf", "kpss", "structural_break"],
  },
  S6: {
    summary: "See how strongly the series relates to its own past and to other columns.",
    reading: "ACF and PACF bars outside the Bartlett bands point to possible model orders; the CCF does the same across series. These are hints, not final orders.",
    tryThis: "Find the first few significant lags and compare them with the seasonal period before choosing a model.",
    watchOut: "Correlation is not causation, and a significant bar can appear by chance when many lags are inspected.",
    terms: ["acf", "pacf", "ccf", "bartlett_bands", "lag"],
  },
  S7: {
    summary: "Rank possible next actions and preview each one before committing.",
    reading: "Differencing, transforms and 'leave it alone' are ranked from the stored test evidence. A preview shows what would happen without changing the analysis.",
    tryThis: "Preview the top-ranked action, then choose the one whose plot and explanation you can defend in a methods section.",
    watchOut: "More differencing is not more rigorous; overdifferencing injects avoidable structure into the errors.",
    terms: ["difference", "integration_order", "reconciliation", "preview", "white_noise"],
  },
  S8: {
    summary: "One explicit readiness verdict, with every blocker and caveat listed.",
    reading: "The gate checks stored evidence from earlier stages. READY WITH CAVEATS means you may continue, but the caveats travel into every downstream result.",
    tryThis: "Open each numbered blocker from the checklist rather than guessing which earlier stage to re-run.",
    watchOut: "A green gate is not a guarantee that a model will fit well; it says the prerequisites are satisfied.",
    terms: ["readiness_gate", "verdict", "caveat", "reconciliation", "stale"],
  },
  S9: {
    summary: "See which model families your data can support, with reasons for exclusions.",
    reading: "The registry is filtered by data class, sample size, seasonality, exogenous columns and library availability. Excluded families state why.",
    tryThis: "Read one exclusion before fitting; it usually tells you what kind of data that family requires.",
    watchOut: "A long list of models is not a recommendation. Fit a short, defensible candidate set plus the mandatory naive benchmark.",
    terms: ["model_registry", "data_class", "benchmark", "ic_comparable", "fit_failure"],
  },
  S10: {
    summary: "Estimate the candidate models and keep a record of failures as well as fits.",
    reading: "Parameter tables, convergence flags and information criteria are shown only when comparable. A failed fit is reported, never silently replaced.",
    tryThis: "Check convergence and residual warnings before looking at which model has the smallest AIC.",
    watchOut: "AIC cannot compare models fitted to differently transformed data; use out-of-sample accuracy for those.",
    terms: ["information_criterion", "fit_failure", "model_card", "parameter", "root_check"],
  },
  S11: {
    summary: "Compare candidates on data they did not see during fitting.",
    reading: "Rolling-origin cross-validation keeps test blocks in the future. Selection uses out-of-sample accuracy first and information criteria second.",
    tryThis: "Read the 'why this model won' paragraph and check whether the margin over the naive benchmark is meaningful.",
    watchOut: "A model that wins on every displayed metric may still be leaking future information; read the leakage audit.",
    terms: ["rolling_origin_cv", "out_of_sample", "fold", "leakage", "metric_policy"],
  },
  S12: {
    summary: "Ask whether anything explainable is left in the model's errors.",
    reading: "Residual ACF, Q-Q, PIT and randomised quantile residuals test different assumptions. The verdict is an adequacy judgement, not a proof of truth.",
    tryThis: "Start with residual ACF and PIT; if both look unstructured, move on to forecasting.",
    watchOut: "A diagnostic plot can look unusual simply because the sample is short. Read the accompanying test and its caveat.",
    terms: ["innovation", "pit", "rqr", "ljung_box", "normality_battery"],
  },
  S13: {
    summary: "Produce future values with honest uncertainty, not just a single line.",
    reading: "Point forecasts, analytic and simulated intervals, scenario paths and combinations are kept separate. Back-transformation is explicit and reproducible.",
    tryThis: "Compare the 80% and 95% bands: a wider horizon should usually widen uncertainty.",
    watchOut: "A narrow interval is not automatically better; coverage and calibration matter more than sharpness alone.",
    terms: ["forecast", "horizon", "prediction_interval", "fan_chart", "back_transformation"],
  },
  S14: {
    summary: "Score forecasts on held-out data using metrics legal for this data class.",
    reading: "The metric policy explains why MAPE is refused for zero-heavy series and why MASE needs a naive benchmark. Every number is out-of-sample.",
    tryThis: "Find the MASE-by-horizon table, then check interval coverage before declaring a winner.",
    watchOut: "A perfect in-sample fit says nothing about forecast accuracy; only the held-out folds count here.",
    terms: ["mase", "crps", "pi_coverage", "metric_policy", "diebold_mariano"],
  },
  S15: {
    summary: "Turn computed results into a captioned, reproducible write-up.",
    reading: "The scorecard links ten weighted checks to stored result ids. The catalogue supplies plots and tables, and exports keep the numbers behind every item.",
    tryThis: "Open the catalogue, insert one figure and one table, then preview the one-file HTML report.",
    watchOut: "A high completeness score is not peer review. Keep the caveats and verify every AI-drafted sentence.",
    terms: ["scorecard", "report_bundle", "one_file_html", "caption", "alt_text"],
  },
});

/** A ribbon id, a stage code, or an unknown id all resolve to usable help. */
export const DEFAULT_PANEL_HELP: PanelHelp = Object.freeze({
  summary: "This panel shows one part of the analysis, from loading data to writing up.",
  reading: "Read the plain sentence at the top of the panel first; the technical layer is available but never required.",
  tryThis: "Use the Back and Next controls in the pipeline bar, or open the glossary to look up an unfamiliar word.",
  watchOut: "Results computed before an earlier change are marked stale. Re-run the step before quoting them.",
  terms: ["stepper", "guided_tour", "stale", "plain_language_mode", "tooltip"],
});

export function panelHelp(id: string): PanelHelp {
  const stage = PANEL_ALIASES[id] ?? id;
  return PANEL_HELP[stage] ?? DEFAULT_PANEL_HELP;
}

/** Pure stepper helper, so back/next behaviour is testable without a DOM. */
export function stepNavigation(
  codes: readonly string[],
  current: string,
): { previous: string | null; next: string | null; position: number; total: number } {
  const i = Math.max(0, codes.indexOf(current));
  return {
    previous: i > 0 ? codes[i - 1] : null,
    next: i >= 0 && i < codes.length - 1 ? codes[i + 1] : null,
    position: i + 1,
    total: codes.length,
  };
}
