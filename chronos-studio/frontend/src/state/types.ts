/**
 * src/state/types.ts — mirror of backend `core/schemas.py`.
 *
 * Keep this file boring and mechanical: if a field is added to the backend
 * contract, it is added here in the same commit. The UI must never invent a
 * number that did not arrive in one of these payloads.
 */

export type StageCode =
  | "S0" | "S1" | "S2" | "S3" | "S4" | "S5" | "S6" | "S7"
  | "S8" | "S9" | "S10" | "S11" | "S12" | "S13" | "S14" | "S15";

export type StageStatus =
  | "not_started" | "running" | "complete" | "stale" | "blocked" | "failed";

export type Confidence = "high" | "medium" | "low";
export type Verdict = "ready" | "ready_with_caveats" | "not_ready" | "inconclusive";
export type DataClass =
  | "continuous" | "count" | "intermittent" | "binary_or_proportion" | "event_timestamp";

export interface StageState {
  stage: StageCode;
  status: StageStatus;
  completed_at?: string | null;
  duration_ms?: number | null;
  n_results: number;
  message: string;
  blockers: string[];
}

export interface StageDef {
  code: StageCode;
  label: string;
  order: number;
  depends_on: StageCode[];
}

export interface Progress {
  session_id: string;
  current_stage: StageCode;
  stages_complete: number;
  stages_total: number;
  percent: number;
  stages: StageState[];
}

export interface SessionState {
  session_id: string;
  created_at: string;
  seed: number;
  pipeline: {
    session_id: string;
    current_stage: StageCode;
    stages: Record<string, StageState>;
    caveats: string[];
    blockers: string[];
    seed: number;
    library_versions: Record<string, string>;
  };
  progress: Progress;
  dataset: SessionDataset | null;
  preprocessing: PreprocessingRecord[];
  series_version: number;
  optional_features: Record<string, boolean>;
  ai_configured: boolean;
}

export interface FrequencySpec {
  code: string;
  label: string;
  period: number | null;
  offset_step_ns?: number | null;
  confidence: Confidence;
  evidence: string;
}

export interface SessionDataset {
  dataset_id: string;
  name: string;
  source: "upload" | "example" | "url" | "paste";
  time_column: string;
  target_column: string;
  exogenous_columns: string[];
  frequency: FrequencySpec;
  n_observations: number;
  time_start: string;
  time_end: string;
  data_class: DataClass;
  content_hash: string;
  preview_rows: Record<string, unknown>[];
  quality: DataQuality;
}

/* ----------------------------- advisor (S2+) ---------------------------- */

export interface AdviceResponse {
  stage: StageCode;
  headline: string;
  confidence: Confidence;
  next_stage_ready: boolean;
  blockers: RuleOut[];
  recommendations: RuleOut[];
  facts_used: Record<string, number | string | boolean>;
  narrative: string;
  results: AnyResult[];
  caveats: string[];
  duration_ms: number;
}

export interface PreviewResponse {
  op: string;
  step: string;
  method: string;
  plain: string;
  n_before: number;
  n_after: number;
  changed: boolean;
  preview: { time: string; value: number | null }[];
  warnings: string[];
  reversible: boolean;
}

export interface PreprocessState {
  steps: PreprocessingRecord[];
  series_version: number;
  n: number;
  n_missing: number;
  start: string | null;
  end: string | null;
  can_undo: boolean;
  /** Part 19: POST /preprocess/redo is available while an undo stack exists. */
  can_redo?: boolean;
}

export interface PreprocessingRecord {
  version: number;
  step: "impute" | "smooth" | "transform" | "outlier" | "aggregate" | "trim" | "difference";
  method: string;
  params: Record<string, unknown>;
  reason: string;
  advisor_rule_id?: string | null;
  applied_at: string;
  reversible: boolean;
}

/* ----------------------------- results ---------------------------------- */

export interface ResultBase {
  id: string;
  stage: StageCode;
  kind: "table" | "figure" | "stat" | "narrative" | "verdict";
  title: string;
  plain: string;
  why: string;
  technical: string;
  method: string;
  inputs: Record<string, unknown>;
  citations: string[];
  caveats: string[];
  warnings: string[];
  reproducible_snippet: string;
  created_at: string;
  schema_version: string;
}

export interface TableColumn {
  key: string;
  label: string;
  dtype: "string" | "number" | "integer" | "date" | "bool";
  format?: string | null;
  align: "left" | "right" | "center";
}

export interface TableResult extends ResultBase {
  kind: "table";
  columns: TableColumn[];
  rows: Record<string, unknown>[];
  caption: string;
  footnote: string;
}

export interface FigureResult extends ResultBase {
  kind: "figure";
  figure_json: Record<string, unknown>;
  caption: string;
  alt_text: string;
  data_behind?: TableResult | null;
}

export interface StatResult extends ResultBase {
  kind: "stat";
  value: number;
  p_value?: number | null;
  statistic_name: string;
  df?: number | null;
  effect_size?: number | null;
  ci_low?: number | null;
  ci_high?: number | null;
  interpretation: string;
  band: "none" | "weak" | "moderate" | "strong" | "inconclusive";
}

export interface NarrativeResult extends ResultBase {
  kind: "narrative";
  paragraphs: string[];
  audience: "beginner" | "peer" | "reviewer";
  ai_drafted: boolean;
  numbers_cited: Record<string, number>;
}

export interface ChecklistItem {
  key: string;
  label: string;
  passed: boolean | null;
  value: string;
  detail: string;
  jump_to_stage?: StageCode | null;
}

export interface VerdictResult extends ResultBase {
  kind: "verdict";
  verdict: Verdict;
  headline: string;
  checklist: ChecklistItem[];
  blockers: string[];
  caveats_travelling: string[];
}

export type AnyResult =
  | TableResult | FigureResult | StatResult | NarrativeResult | VerdictResult;

/* ------------------------------ advisor --------------------------------- */

export interface RuleOut {
  id: string;
  stage: StageCode;
  recommend: string;
  confidence: Confidence;
  because: string;
  provenance: string[];
  alternative?: string | null;
  risk_if_ignored: string;
  how_to: string;
}

/* ------------------------ Part 4: decomposition ------------------------- */

export interface DecompRow {
  component: string;
  share: number;
  reading: string;
}

export interface DecompFinding {
  id: string;
  name: string;
  statistic: number | null;
  p_value: number | null;
  band: "none" | "weak" | "moderate" | "strong";
  plain: string;
  detail: string;
  caveats: string[];
  table?: { position?: number; effect?: number; period?: number | null; power?: number }[];
}

export interface MethodOption {
  method: string;
  label: string;
  needs: string;
  available: boolean;
  why_not: string;
}

export interface DecompResponse {
  stage: StageCode;
  status: string;
  method?: string;
  model?: "additive" | "multiplicative";
  label?: string;
  period?: number | null;
  scorecard?: { rows: DecompRow[]; plain: string } | null;
  findings: DecompFinding[];
  shares?: Record<string, number>;
  caveats?: string[];
  results: AnyResult[];
  recommendations?: RuleOut[];
  blockers?: RuleOut[];
  headline?: string;
}

export interface CompareResponse {
  stage: StageCode;
  ran: string[];
  skipped: { method: string; why: string }[];
  rows: {
    method: string;
    trend_share: number;
    seasonal_share: number;
    remainder_share: number;
    F_T: number | null;
    F_S: number | null;
  }[];
  trend_spread: number | null;
  notes: string[];
  figure: Record<string, unknown>;
  alt_text: string;
}

/* --------------------- Part 7: the readiness gate ------------------------ */

export type CriterionState = "pass" | "fail" | "unknown" | "not_applicable";

export interface GateCriterion {
  key: string;
  label: string;
  passed: boolean | null;
  state: CriterionState;
  value: string;
  detail: string;
  why_unknown: string;
  jump_to_stage: StageCode | null;
  severity: "blocker" | "caveat";
}

export interface GateResponse {
  stage: StageCode;
  status: "not_started" | "complete";
  verdict?: "ready" | "ready_with_caveats" | "not_ready" | "inconclusive" | null;
  headline?: string;
  plain?: string;
  checklist?: GateCriterion[];
  blockers?: GateCriterion[];
  caveats?: GateCriterion[];
  unknown?: GateCriterion[];
  caveats_travelling?: string[];
  n_pass?: number;
  n_fail?: number;
  n_unknown?: number;
  n_not_applicable?: number;
  n_total?: number;
  facts?: Record<string, unknown>;
  stale?: boolean;
  computed_for_version?: number;
  current_version?: number;
  message?: string;
}

export interface HorizonResponse {
  stage: StageCode;
  horizon: number;
  verdict: string;
  headline: string;
  horizon_check: GateCriterion | null;
  ratio: number;
  caveats_travelling: string[];
}

/* ------------------ Part 8: the registry and fitting --------------------- */

export interface ModelSuggestion {
  model_id: string;
  display_name: string;
  family: string;
  rank: number;
  is_baseline: boolean;
  recommended: boolean;
  advanced?: boolean;
  because: string;
  would_win_if: string;
  plain_description: string;
  compatible: boolean;
  incompatibilities: string[];
  assumption_warnings: string[];
  order_hint: string | null;
  expected_seconds: number | null;
}

export interface SuggestionSet {
  stage: string;
  data_class: string;
  headline: string;
  plain: string;
  suggestions: ModelSuggestion[];
  excluded: ModelSuggestion[];
  default_fit_ids: string[];
  facts_used: Record<string, unknown>;
  n_registry: number;
  caveats: string[];
}

export interface ParamRow {
  name: string;
  value: number | null;
  std_err: number | null;
  z: number | null;
  p_value: number | null;
  at_boundary: boolean;
  plain: string;
}

export interface FitRow {
  model_id: string;
  display_name: string;
  family: string;
  fit_id: string;
  converged: boolean;
  n_params: number;
  aic: number | null;
  aicc: number | null;
  bic: number | null;
  n_obs_used: number;
  fitted_on: string;
  roots_ok: boolean | null;
  roots_note: string;
  is_baseline: boolean;
  failed: boolean;
  failure_reason: string;
  plain_one_liner: string;
  params: ParamRow[];
  caveats: string[];
  converged_warnings: string[];
  ic_comparable?: boolean;
  ic_note?: string;
  seconds?: number;
  diagnostics?: Record<string, number | string | null | string[] | undefined>;
  /** §5.8 F's mandatory tabular-model card. Present for the ML rows, null for everything else. */
  model_card?: ModelCard | null;
}

/** What an ML row was built from. Shipped with the fit so the panel never has to guess. */
export interface ModelCard {
  features: string[];
  n_features: number;
  lags: number;
  scheme: string;
  validation: string;
  scaling: string;
  hyperparams: Record<string, string | number>;
  holdout: {
    n: number;
    rmse: number | null;
    mase: number | null;
    folds: { fold: number; train_n: number; test_n: number; rmse: number; mase: number | null }[];
  };
  interval_coverage_90: number | null;
  interval_basis: string;
  residual_excess_kurtosis: number | null;
  solver_warnings: string[];
  benchmark_note: string;
}

/**
 * One impulse-response band: the 5th and 95th percentile of the bootstrap paths.
 *
 * Shaped like `paths[series]` — one row per STEP, one column per response series — not a flat series,
 * because the band has to be drawn against the same axis as the path it belongs to.
 */
export interface IrfBand {
  lower: number[][];
  upper: number[][];
}

export interface GrangerRow {
  cause: string;
  effect: string;
  f: number | null;
  p: number | null;
  p_on_differences?: number | null;
  significant: boolean;
  spurious_risk: boolean;
  sentence: string;
}

/** The multivariate payload — Granger, IRF, FEVD, and for a VECM the long-run coefficients. */
export interface MultivariateResponse {
  status: string;
  model_id: string;
  series: string[];
  lag_order: number;
  fitted_on: string;
  stability: { stable: boolean; max_root: number };
  johansen: { rank?: number; trace?: number[]; crit_5pct?: number[] };
  beta?: number[][];
  adjustment?: { series: string; alpha: number; se: number; t: number | null; correcting: boolean; sentence: string }[];
  granger: GrangerRow[];
  irf: {
    horizon: number;
    response_to: string;
    ordering: string[];
    paths: Record<string, number[][]>;
    bands_90: Record<string, IrfBand>;
    n_boot: number;
  };
  fevd: { horizon: number; ordering: string[]; shares: Record<string, Record<string, number>>; shares_full: number[][][] };
  caveats: string[];
}

export interface FitsResponse {
  stage: string;
  status: "not_started" | "complete";
  stale?: boolean;
  computed_for_version?: number;
  current_version?: number;
  data_class?: string;
  d?: number;
  D?: number;
  period?: number | null;
  failure_summary?: string;
  fits: FitRow[];
  all?: FitRow[];
  message?: string;
}

/** One step of the conditional-variance forecast (§8 Part 11's second forecast). */
export interface VariancePath {
  variance: number[];
  sd: number[];
  unconditional_sd: number | null;
  persistence: number | null;
  leverage: { gamma: number; active: boolean; reading: string } | null;
  error?: string;
}

export interface FitForecast {
  model_id: string;
  display_name: string;
  horizon: number;
  alpha: number;
  index: string[];
  mean: number[];
  lower: number[];
  upper: number[];
  caveats: string[];
  /** Present only for a conditional-variance fit: the mean path's intervals widen along this path. */
  variance_path?: VariancePath | null;
}

/* ------------------- Part 6: ACF / PACF and "what next" ------------------ */

export interface ACFRow {
  lag: number;
  acf?: number;
  pacf?: number;
  band: number;
  significant: boolean;
}

export interface CorrelogramPattern {
  kind:
    | "white_noise" | "ar" | "ma" | "nonstationary_decay" | "seasonal"
    | "overdifferenced" | "single_spike" | "unclear";
  label: string;
  plain: string;
  confidence: string;
  suggests: string;
  evidence: Record<string, unknown>;
  caveats: string[];
}

export interface PortmanteauRow {
  lag: number;
  statistic: number;
  df: number;
  p_value: number;
  rejects: boolean;
  box_pierce?: number | null;
  box_pierce_p?: number | null;
  reading?: string;
}

export interface OrderHint {
  order: { p: number; d: number; q: number; P: number; D: number; Q: number; s: number };
  notation: string;
  plain: string;
  support: string[];
  against: string[];
}

export interface VarianceRatio {
  q: number;
  ratio: number;
  z: number;
  p_value: number;
  reading: string;
}

export interface CorrelateResponse {
  stage: StageCode;
  status: string;
  cached: boolean;
  series_name: string;
  n: number;
  period: number | null;
  max_lag: number;
  band_white: number;
  band_bartlett: number[];
  acf: ACFRow[];
  pacf: ACFRow[];
  pattern: CorrelogramPattern;
  ljung_box: PortmanteauRow[];
  box_pierce: PortmanteauRow[];
  seasonal_acf: { period: number; r: number; significant: boolean; band: number } | null;
  variance_ratio: VarianceRatio | null;
  order_hints: OrderHint[];
  stationarity: {
    verdict: string; confidence: string; differencing_needed: boolean; plain: string;
  } | null;
  results: AnyResult[];
}

export type PreviewVerdict = "better" | "neutral" | "worse" | "unavailable";

export interface ActionPreview {
  verdict: PreviewVerdict;
  op?: string;
  label?: string;
  plain: string;
  pattern_after?: string;
  pattern_after_label?: string;
  pattern_after_plain?: string;
  before: {
    n: number; sd: number; r1: number; decay: number; n_sig: number;
    lb_p: number; lb_stat: number; vr: number | null; band: number;
    level_spread?: number; seasonal_left?: boolean;
  } | null;
  after: ActionPreview["before"];
  n_lost?: number;
}

export interface NextRecommendation extends RuleOut {
  op: string | null;
  op_params: Record<string, unknown>;
  preview: ActionPreview | null;
  rank: number;
  kind: string;
}

export interface NextResponse {
  stage: StageCode;
  status: string;
  headline: string;
  ranked: NextRecommendation[];
  facts: Record<string, unknown>;
  pattern: CorrelogramPattern;
  stationarity: CorrelateResponse["stationarity"];
  advisor: AdvisorReport;
  results: AnyResult[];
}

/* ------------------------ Part 5: stationarity ------------------------- */

export type StatVerdict =
  | "stationary"
  | "stationary_around_break"
  | "trend_stationary"
  | "unit_root"
  | "unit_root_or_break"
  | "seasonal_unit_root"
  | "inconclusive"
  | "conflicting";

export interface TestRow {
  test: string;
  id: string;
  null?: string;
  statistic: number;
  p_value?: number | null;
  cv_5pct?: number | null;
  rejects_null?: boolean;
  reading?: string;
  breaks?: string;
}

export interface BreakEntry {
  index?: number;
  time: string;
  statistic: number;
  critical_5pct?: number;
  segment?: [number, number];
}

export interface BreakTestSummary {
  test: string;
  id: string;
  statistic: number;
  cv_5pct?: number | null;
  says: string;
  breaks: BreakEntry[];
  caveats: string[];
}

export interface StatTestsResponse {
  stage: StageCode;
  status: string;
  unit_root: TestRow[];
  seasonal: TestRow[];
  breaks: BreakTestSummary[];
  verdict: {
    verdict: StatVerdict;
    headline: string;
    plain: string;
    action: string;
    confidence: string;
    differencing_needed: boolean;
    facts_used: string[];
    contradictions: string[];
    caveats: string[];
  } | null;
  results: AnyResult[];
  recommendations?: RuleOut[];
  blockers?: RuleOut[];
}

export interface ReconcileResponse {
  stage: StageCode;
  status: string;
  verdict: StatVerdict;
  headline: string;
  plain: string;
  action: string;
  confidence: string;
  differencing_needed: boolean;
  facts_used: string[];
  contradictions: string[];
  caveats: string[];
  blockers: RuleOut[];
  recommendations: RuleOut[];
  duration_ms: number;
}

export interface AdvisorReport {
  stage: StageCode;
  headline: string;
  recommendations: RuleOut[];
  blockers: RuleOut[];
  facts_used: Record<string, number | string | boolean>;
  confidence: Confidence;
  next_stage_ready: boolean;
  narrative: string;
  generated_at: string;
}

export interface AnalysisResponse {
  stage: StageCode;
  advisor: AdvisorReport | null;
  results: AnyResult[];
  pipeline: Partial<Progress>;
  caveats: string[];
  duration_ms?: number | null;
}

/* -------------------------------- api ---------------------------------- */

export interface ApiErrorBody {
  error: { code: string; plain: string; fix: string; technical: string; context: Record<string, unknown> };
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  stages: number;
  ai_configured: boolean;
  optional_features: Record<string, boolean>;
}

/* ------------------------------ Part 1: data ---------------------------- */

export interface ParseIssue {
  code: string;
  severity: "info" | "warning" | "error";
  plain: string;
  fix: string;
  rows_affected?: number | null;
}

export interface MissingPattern {
  index: string;
  n_missing: number;
  pct_missing: number;
  longest_run: number;
  longest_run_start?: string | null;
  gaps_over_two: number;
  plain: string;
}

export interface DataQuality {
  n_rows_raw: number;
  n_rows_used: number;
  n_columns: number;
  dropped_rows: number;
  time_column?: string | null;
  target_column?: string | null;
  exogenous_columns: string[];
  time_start?: string | null;
  time_end?: string | null;
  frequency?: FrequencySpec | null;
  n_observations: number;
  n_expected_observations?: number | null;
  n_gaps: number;
  is_regular: boolean;
  data_class: DataClass;
  data_class_evidence: Record<string, number | string | boolean>;
  n_missing_total: number;
  missing_by_column: MissingPattern[];
  n_zeros: number;
  pct_zeros: number;
  n_negatives: number;
  n_constant_runs: number;
  n_duplicate_timestamps: number;
  n_outlier_candidates: number;
  issues: ParseIssue[];
  plain_summary: string;
}

export interface ColumnMapping {
  time_column?: string | null;
  target_column?: string | null;
  exogenous_columns: string[];
  frequency_code?: string | null;
  row_order_is_time: boolean;
  start_date?: string | null;
  dayfirst?: boolean | null;
  year_column?: string | null;
  month_column?: string | null;
  agg_duplicate: "mean" | "first" | "sum" | "last";
  /** Empty = keep the class detected at load time. See "Change mapping" in Step 0. */
  data_class?: string | null;
}

export interface IngestResponse {
  dataset: SessionDataset;
  mapping: ColumnMapping;
  cross_sell: string[];
  profiles: {
    name: string;
    dtype: string;
    n_non_null: number;
    n_missing: number;
    pct_missing: number;
    n_unique: number;
    is_numeric: boolean;
    sample_values: string[];
    role_guess: string;
    role_confidence: Confidence;
  }[];
  candidate_frequencies: FrequencySpec[];
  next: string;
}

export interface ExampleDataset {
  id: string;
  file: string;
  time_column: string;
  value_column: string;
  n_rows: number;
  title: string;
  teaches: string;
  data_class: DataClass;
  expect: string;
  frequency: string;
  available: boolean;
}

export interface CalendarOption {
  code: string;
  label: string;
  period: number | null;
}

export interface CurrentDatasetResponse {
  dataset: SessionDataset | null;
  quality?: DataQuality;
  mapping?: ColumnMapping | null;
  series_head: { time: string; value: number | null }[];
  n_missing: number;
  message?: string;
}

export interface SeriesPoints {
  points: { time: string; value: number }[];
  n: number;
  n_displayed: number;
  decimated: boolean;
  note: string;
  missing: number;
}

/* ------------------------------------------------------------------ *
 * Part 13 — S11 selection and S12 diagnostics (§5.9)
 * ------------------------------------------------------------------ */

/** One row of the ranked shortlist. Everything here was computed, not decided. */
export interface ShortlistEntry {
  rank: number;
  model_id: string;
  display_name: string;
  cv_mase: number | null;
  cv_mae: number | null;
  cv_rmse: number | null;
  /** Fold-to-fold standard deviation of MASE — how much the mean is worth. */
  cv_spread: number | null;
  aicc: number | null;
  delta_aicc: number | null;
  akaike_weight: number | null;
  support: string;
  n_folds: number;
  worst_fold_mase: number | null;
  notes: string[];
  verdict: string;
}

export interface DmTest {
  ok: boolean;
  test: string;
  statistic: number | null;
  p_value: number | null;
  mean_loss_diff: number | null;
  n: number;
  better: string;
  note: string;
  assumptions: string[];
}

export interface MzTest {
  ok: boolean;
  test: string;
  intercept: number | null;
  slope: number | null;
  joint_p: number | null;
  n: number;
  verdict: string;
  note: string;
}

export interface SelectionReport {
  primary_criterion: string;
  caveat: string;
  shortlist: ShortlistEntry[];
  winner: string;
  winner_name: string;
  runner_up: string;
  /** True when the top two cannot be told apart — the report then refuses to crown one. */
  indistinguishable: boolean;
  why: string;
  ic_comparable: boolean;
  ic_reason: string;
  dm: DmTest | null;
  mz: MzTest | null;
  scheme: string;
  horizon: number;
  n_folds: number;
  notes: string[];
}

export interface SelectionResponse {
  stage: string;
  status: string;
  stale?: boolean;
  message?: string;
  report: SelectionReport | null;
  cv?: unknown[];
  request?: Record<string, unknown>;
}

/** One failed diagnostic check: what failed, the number, and the remedy. */
export interface DiagnosticFinding {
  test: string;
  kind: "structural" | "distributional";
  severity: "high" | "medium" | "low";
  detail: string;
  remedy: string;
}

export interface AdequacyVerdict {
  adequacy: "adequate" | "adequate_with_caveats" | "inadequate";
  headline: string;
  reasons: DiagnosticFinding[];
  remedies: string[];
  notes: string[];
  n_tests_run: number;
  n_rejected: number;
  expected_false_positives: number;
}

export interface DiagnosticBattery {
  model_id: string;
  display_name: string;
  n_resid: number;
  ljung_box: { results: Record<string, { q: number; p: number | null; dof: number }>; rejects: number[]; note: string };
  box_pierce: { q: number | null; p: number | null; lags: number };
  acf: { lags: number[]; acf: number[]; bounds: number[]; n_exceed: number; note: string };
  normality: {
    jarque_bera: { statistic: number | null; p: number | null; skew: number | null; kurtosis: number | null };
    shapiro: { w: number | null; p: number | null; note?: string };
    anderson_darling: { statistic: number | null; p: number | null; note?: string };
    note: string;
  };
  arch_lm: { p: number | null; lags?: number; advice: string };
  runs: { runs: number | null; expected: number | null; p: number | null; note?: string };
  signs: { positive: number; n: number; p: number | null };
  cusum: {
    path: number[];
    critical: number | null;
    breach: boolean;
    breach_where?: { start_index: number; end_index: number; since: number } | null;
    note: string;
  };
  outliers: {
    position: number;
    when: string | null;
    residual: number;
    z: number;
    influence: number;
    reading: string;
  }[];
  notes: string[];
}

export interface DiagnosticBlock {
  model_id: string;
  display_name: string;
  status: "ok" | "unavailable";
  message?: string;
  battery: DiagnosticBattery | null;
  verdict: AdequacyVerdict | null;
}

export interface DiagnosticsResponse {
  stage: string;
  status: string;
  stale?: boolean;
  message?: string;
  period: number | null;
  blocks: Record<string, DiagnosticBlock>;
}

/* ------------------------------------------------------------------ *
 * Part 14 — S13 forecasting (§5.11)
 * ------------------------------------------------------------------ */

export interface ForecastBand {
  coverage: number;
  coverage_pct: number;
  alpha: number;
  source: string;
  construction: string;
  lower: number[];
  upper: number[];
  mean_width: number;
  nesting_adjusted: boolean;
}

export interface HorizonAdvicePayload {
  n: number;
  horizon: number;
  period: number | null;
  recommended_max: number;
  long_horizon: boolean;
  season_short: boolean;
  acknowledged: boolean;
  requires_acknowledgement: boolean;
  allowed: boolean;
  warnings: string[];
  advice: string;
  method: string;
  test_observations_used: number;
}

export interface ForecastPayload {
  model_id: string;
  display_name: string;
  horizon: number;
  index: string[];
  point: number[];
  bands: ForecastBand[];
  analytic_bands: ForecastBand[];
  simulation_bands: ForecastBand[];
  interval_method: "analytic" | "simulation";
  coverages: number[];
  simulation: {
    n_paths: number;
    seed: number;
    construction: string;
    distribution: string;
    coupling: string;
    scale_source: string;
    notes: string[];
  } | null;
  discrete: Record<string, unknown> | null;
  transform: {
    applied: boolean;
    method: string;
    transform: Record<string, unknown> | null;
    scale: string;
    formula: string;
    point_definition: string;
    factor: number | null;
    clipped_bounds: number;
    clipped_paths: number;
    notes: string[];
    silent_exponentiation: boolean;
  };
  adequacy: {
    status: string;
    adequacy: string;
    headline: string;
    model_id: string;
    source: string;
    stale?: boolean;
  };
  horizon_advice: HorizonAdvicePayload;
  scenario: {
    applies: boolean;
    model_supports_exog: boolean;
    drivers: string[];
    supplied_paths: string[];
    held_flat: string[];
    note: string;
    horizon_mismatch: boolean;
    scenario_steps: number;
    comparison: {
      baseline: number[];
      scenario: number[];
      change: number[];
      changed: boolean;
      max_abs_change: number;
      construction: string;
    } | null;
  };
  combination: {
    method: string;
    weights: Record<string, number>;
    component_ids: string[];
    n_paths: number;
    construction: string;
    notes: string[];
  } | null;
  inputs: Record<string, unknown>;
  warnings: string[];
  caveats: string[];
  notes: string[];
  construction: string;
  seed: number;
  n_paths: number;
  scale: string;
  intervals_stated: boolean;
  test_observations_used: number;
}

export interface ForecastOptions {
  models: {
    model_id: string;
    display_name: string;
    family: string;
    failed: boolean;
    supports_exog: boolean;
    aicc: number | null;
    ic_comparable: boolean;
  }[];
  winner: string | null;
  coverages: number[];
  interval_methods: string[];
  back_transform_methods: string[];
  combination_methods: string[];
  reconciliation_methods: string[];
  min_paths: number;
  transform_active: Record<string, unknown> | null;
  transform_required: boolean;
  transform_warnings: string[];
  drivers: string[];
  scenario_endpoint: string;
  n: number;
  period: number | null;
  frequency: string;
  recommended_max_h: number;
}

export interface ForecastResponse {
  stage: string;
  status: string;
  message?: string;
  stale?: boolean;
  computed_for_version?: number;
  current_version?: number;
  forecast: ForecastPayload | null;
  request?: Record<string, unknown>;
  options: ForecastOptions;
}

export interface ForecastRequestPayload {
  model_id?: string | null;
  horizon: number;
  levels: number[];
  interval_method: "auto" | "analytic" | "simulation";
  n_paths: number;
  seed?: number | null;
  acknowledge_long_horizon: boolean;
  back_transform?: "naive" | "smearing" | "lognormal" | null;
  combination: "none" | "mean" | "median" | "ic_weighted";
  models: string[];
}

export interface HierarchyResponse {
  stage: string;
  status: string;
  message?: string;
  stale?: boolean;
  reconciliation: (Record<string, unknown> & {
    method?: string;
    coherence_error?: number;
    interval_construction?: string;
  }) | null;
}

/* ------------------- Part 15: S14 evaluation metrics -------------------- */

export interface EvaluationPolicy {
  cluster: string;
  primary: string;
  metrics: string[];
  metric_labels: Record<string, string>;
  banned: string[];
  reasons: string[];
  notes: string[];
  probabilistic: boolean;
  discrete: boolean;
  variance: boolean;
}

export interface EvaluationOptions {
  models: {
    model_id: string;
    display_name: string;
    family: string;
    failed: boolean;
    supports_exog: boolean;
    aicc: number | null;
    ic_comparable: boolean;
  }[];
  winner: string | null;
  shortlist: string[];
  default_models: string[];
  coverages: number[];
  schemes: string[];
  back_transform_methods: string[];
  transform_active: Record<string, unknown> | null;
  transform_required: boolean;
  transform_warnings: string[];
  min_paths: number;
  max_paths: number;
  min_folds: number;
  max_folds: number;
  min_train: number;
  min_horizon: number;
  max_horizon: number;
  reps_min: number;
  reps_max: number;
  default_horizon: number;
  default_folds: number;
  default_scheme: "expanding" | "sliding";
  default_n_paths: number;
  default_reps: number;
  n: number;
  period: number | null;
  frequency: string;
  recommended_max_h: number;
  data_class: string;
  class_note: string;
  zero_share: number;
  target_kind: string;
  discrete: boolean;
  probabilistic: boolean;
  benchmark_id: string;
  policy: EvaluationPolicy;
  metric_labels: Record<string, string>;
  policy_reasons: string[];
  policy_notes: string[];
  banned: string[];
  adequacy: Record<string, unknown>;
  s13_status: string;
  s13_forecast_horizon: number | null;
  s13_stale: boolean;
  stage_dependencies: string[];
  next_stage_requires: string[];
  evaluation_endpoint: string;
}

export interface EvaluationRequestPayload {
  horizon: number;
  folds: number;
  scheme: "expanding" | "sliding";
  models: string[];
  include_baselines: boolean;
  coverages: number[];
  n_paths: number;
  seed?: number | null;
  back_transform?: "naive" | "smearing" | "lognormal" | null;
  reps: number;
}

export interface EvaluationLeakageCheck {
  name: string;
  ok: boolean;
  detail: string;
  rows?: Record<string, unknown>[];
}

export interface EvaluationReport {
  policy: EvaluationPolicy;
  inputs: Record<string, unknown>;
  folds: Record<string, unknown>[];
  models: string[];
  point: Record<string, Record<string, unknown>>;
  probabilistic: Record<string, Record<string, unknown>>;
  discrete: Record<string, Record<string, unknown>>;
  variance: Record<string, Record<string, unknown>>;
  comparative: Record<string, unknown>;
  leakage: {
    ok: boolean;
    checks: EvaluationLeakageCheck[];
    fold_digests: Record<string, unknown>[];
    notes: string[];
    poison_value: number;
  };
  accuracy: {
    primary: string;
    primary_metric: string;
    ranked: string[];
    n_scored: number;
    evaluation_window: string;
  };
  metric_rows: Record<string, unknown>[];
  mase_by_horizon: Record<string, (number | null)[]>;
  fold_scores: Record<string, unknown>[];
  loss_curves: Record<string, Record<string, unknown>>;
  pit_histograms: Record<string, Record<string, unknown>>;
  rqr_by_model: Record<string, Record<string, unknown>>;
  calibration: { rows: Record<string, unknown>[]; philosophy: string };
  coverage_construction: Record<string, unknown>[];
  fold_failures: Record<string, unknown>[];
  warnings: string[];
  notes: string[];
  adequacy: Record<string, unknown>;
  seconds: number;
}

export interface EvaluationResponse {
  stage: string;
  status: string;
  message?: string;
  stale?: boolean;
  computed_for_version?: number;
  current_version?: number;
  evaluation: EvaluationReport | null;
  request?: EvaluationRequestPayload;
  options: EvaluationOptions;
}

/* ------------------- Part 16: plots & tables catalogue ------------------- */

export type CatalogMode = "light" | "dark" | "print";
export type CatalogFormat = "png" | "svg" | "pdf" | "xlsx" | "csv";
export type CatalogKind = "plot" | "table";

export interface CatalogColumn {
  key: string;
  label: string;
  dtype: "string" | "number" | "integer" | "date" | "bool";
  format?: string | null;
  align: "left" | "right" | "center";
}

export interface CatalogItem {
  id: string;
  title: string;
  group: string;
  stage: StageCode;
  kind: CatalogKind;
  source: string;
  caption: string;
  alt_text: string;
  figure_json: Record<string, unknown>;
  columns: CatalogColumn[];
  rows: Record<string, unknown>[];
  formats: CatalogFormat[];
  dpi: number;
  report_insertable: boolean;
}

export interface CatalogGroupCount {
  plots: number;
  tables: number;
}

export interface CatalogScorecard {
  plots: number;
  tables: number;
  min_plots: number;
  min_tables: number;
  plots_ok: boolean;
  tables_ok: boolean;
  captions_ok: boolean;
  alt_text_ok: boolean;
  data_behind_ok: boolean;
  groups: Record<string, CatalogGroupCount>;
  styling: {
    single_source: string;
    mode: CatalogMode;
    colorblind_safe: boolean;
    dark_light_print: CatalogMode[];
    dash_patterns: boolean;
  };
  exports: {
    formats: CatalogFormat[];
    dpi: number;
    captioned: boolean;
    report_insertable: boolean;
    data_behind: boolean;
  };
}

export interface CatalogReport {
  item_ids: string[];
  items: CatalogItem[];
  count: number;
}

export interface CatalogResponse {
  status: string;
  stage: StageCode;
  mode: CatalogMode;
  scorecard: CatalogScorecard;
  groups: string[];
  plots: CatalogItem[];
  tables: CatalogItem[];
  report: CatalogReport;
  exports: {
    formats: CatalogFormat[];
    dpi: number;
    scale: number;
    endpoint: string;
  };
}

export interface CatalogReportAction {
  action: "add" | "remove" | "set" | "clear";
  item_id?: string | null;
  item_ids?: string[] | null;
}

export interface CatalogReportResponse extends CatalogReport {
  status: string;
  action?: string;
}


/* ------------------- Part 18: reports & finalization ------------------- */

export type ReportMode = CatalogMode;
export type ReportScope = "all" | "selected";
export type ReportStatusName = "ready" | "nearly" | "not_ready";
export type ReportKind =
  | "html" | "docx" | "pdf" | "latex" | "xlsx" | "python" | "r"
  | "chronos" | "figures" | "all";

export interface ReportItem {
  id: string;
  label: string;
  weight: number;
  passed: boolean;
  evidence: string;
  result_ids: string[];
  available_result_ids: string[];
  panel_id: StageCode;
  panel_label: string;
  fix: string;
}

export interface ReportNextStep {
  number: number;
  item_id: string;
  text: string;
  panel_id: StageCode;
  panel_label: string;
  result_ids: string[];
}

export interface ReportScorecard {
  schema_version: string;
  stage: StageCode;
  generated_at: string;
  items: ReportItem[];
  score: number;
  max_score: number;
  percent: number;
  outstanding: string[];
  blockers: string[];
  next_steps: ReportNextStep[];
  verdict: string;
  status: ReportStatusName;
  ready: boolean;
}

export interface ReportExportSpec {
  kind: ReportKind;
  label: string;
  description: string;
  filename: string;
  media_type: string;
}

export interface ReportStatus {
  status: string;
  stage: StageCode;
  stage_status: string;
  generated_at: string;
  scorecard: ReportScorecard;
  exports: ReportExportSpec[];
  catalog_counts: Record<string, number>;
  selection_count: number;
  scope: ReportScope;
  mode: ReportMode;
  privacy_note: string;
}

export interface ReportFinalizeResponse {
  status: string;
  stage: StageCode;
  stage_status: string;
  scorecard_verdict: string;
  score: number;
  max_score: number;
  result_ids: string[];
}
