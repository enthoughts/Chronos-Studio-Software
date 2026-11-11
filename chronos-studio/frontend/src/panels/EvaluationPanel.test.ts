import { describe, expect, it } from "vitest";

import {
  buildEvaluationPayload,
  primaryMetricValue,
  validateEvaluationForm,
  type EvaluationForm,
} from "./EvaluationPanel";
import type { EvaluationOptions, EvaluationReport } from "../state/types";

const form: EvaluationForm = {
  horizon: 6,
  folds: 3,
  scheme: "expanding",
  models: ["ets", "drift"],
  includeBaselines: true,
  coverages: [0.5, 0.8, 0.8],
  nPaths: 2000,
  seedText: "",
  backTransform: "",
  reps: 5,
};

function options(overrides: Partial<EvaluationOptions> = {}): EvaluationOptions {
  return {
    models: [], winner: "ets", shortlist: ["ets"], default_models: ["ets", "seasonal_naive"],
    coverages: [0.5, 0.66, 0.8, 0.9, 0.95], schemes: ["expanding", "sliding"],
    back_transform_methods: ["naive", "smearing", "lognormal"], transform_active: null,
    transform_required: false, transform_warnings: [],
    min_paths: 2000, max_paths: 20000, min_folds: 2, max_folds: 12, min_train: 16,
    min_horizon: 1, max_horizon: 60, reps_min: 1, reps_max: 20,
    default_horizon: 6, default_folds: 4, default_scheme: "expanding",
    default_n_paths: 2000, default_reps: 5,
    n: 180, period: 12, frequency: "MS", recommended_max_h: 36,
    data_class: "continuous", class_note: "declared continuous", zero_share: 0,
    target_kind: "level", discrete: false, probabilistic: true,
    benchmark_id: "seasonal_naive",
    policy: {
      cluster: "probabilistic", primary: "crps", metrics: ["crps", "pinball"],
      metric_labels: { crps: "CRPS" }, banned: [], reasons: ["distribution available"],
      notes: [], probabilistic: true, discrete: false, variance: false,
    },
    metric_labels: { crps: "CRPS" }, policy_reasons: ["distribution available"],
    policy_notes: [], banned: [], adequacy: {}, s13_status: "complete",
    s13_forecast_horizon: 6, s13_stale: false, stage_dependencies: ["S13"],
    next_stage_requires: ["S14", "S12"], evaluation_endpoint: "/analysis/evaluate",
    ...overrides,
  } as EvaluationOptions;
}

describe("S14 evaluation request contract", () => {
  it("always includes the 95% interval and never silently back-transforms", () => {
    const payload = buildEvaluationPayload(form, false);
    expect(payload.coverages).toEqual([0.5, 0.8, 0.95]);
    expect(payload.seed).toBeNull();
    expect(payload.back_transform).toBeNull();
    expect(payload.scheme).toBe("expanding");
    expect(payload.include_baselines).toBe(true);
  });

  it("requires an explicit bias adjustment only when a transform is active", () => {
    expect(validateEvaluationForm(form, options())).toBeNull();
    expect(validateEvaluationForm(form, options({ transform_required: true })))
      .toContain("bias-adjustment");
    const transformed = buildEvaluationPayload({ ...form, backTransform: "smearing" }, true);
    expect(transformed.back_transform).toBe("smearing");
  });

  it("refuses a stage-order shortcut, too few paths and a non-integer seed", () => {
    expect(validateEvaluationForm(form, options({ s13_status: "not_started" })))
      .toContain("Step 13");
    expect(validateEvaluationForm({ ...form, nPaths: 100 }, options()))
      .toContain("seeded paths");
    expect(validateEvaluationForm({ ...form, nPaths: 2000.5 }, options()))
      .toContain("whole number");
    expect(validateEvaluationForm({ ...form, seedText: "3.5" }, options()))
      .toContain("whole number");
  });
});

describe("S14 primary metric readback", () => {
  it("reads the backend row and refuses missing or non-finite values", () => {
    const report = {
      accuracy: { primary: "ets", primary_metric: "crps", ranked: ["ets"], n_scored: 18, evaluation_window: "x" },
      metric_rows: [
        { model: "ets", crps: 2.5 },
        { model: "drift", crps: Number.NaN },
      ],
    } as unknown as EvaluationReport;
    expect(primaryMetricValue(report, "ets")).toBe(2.5);
    expect(primaryMetricValue(report, "drift")).toBeNull();
    expect(primaryMetricValue(report, "missing")).toBeNull();
  });
});
