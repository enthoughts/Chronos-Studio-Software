/**
 * copy/demo.ts — Part 19 demo mode, guided tour and background-job helpers.
 *
 * Pure data + pure functions only, so the whole module is unit-testable without
 * a DOM. The API calls themselves live in state/api.ts.
 */

export interface DemoScenario {
  id: string;
  title: string;
  example_id: string;
  blurb: string;
  teaches: string;
  models: string[];
  horizon: number;
  n_steps: number;
  steps: string[];
  stages: string[];
}

export interface DemoStepResult {
  label: string;
  stage: string;
  status: "ok" | "error";
  seconds: number;
  error: string;
}

export interface DemoStatus {
  job_id: string;
  scenario_id: string;
  session_id: string;
  state: "queued" | "running" | "complete" | "cancelled" | "failed";
  done: number;
  total: number;
  current: string;
  percent: number;
  labels: string[];
  finished: DemoStepResult[];
  errors: string[];
  elapsed_s: number;
}

export const DEMO_POLL_MS = 600;

export const DEMO_TERMINAL_STATES: readonly string[] = Object.freeze([
  "complete", "cancelled", "failed",
]);

export function demoIsTerminal(state: string): boolean {
  return DEMO_TERMINAL_STATES.includes(state);
}

export function demoProgressLabel(status: DemoStatus): string {
  if (status.state === "complete") return "Study complete — ready to explore";
  if (status.state === "cancelled") return "Stopped";
  if (status.state === "failed") return "Stopped with errors";
  if (!status.current) return "Starting…";
  return `Step ${status.done + 1} of ${status.total}: ${status.current}`;
}

/** The six cards of the optional guided tour (§5.15 beginner features). */
export interface TourStep {
  id: string;
  title: string;
  body: string;
  glossary: string;
}

export const GUIDED_TOUR: readonly TourStep[] = Object.freeze([
  {
    id: "load",
    title: "1 · Load your data",
    body: "Upload a spreadsheet, paste a table, or open one of the 19 built-in example series. The app tells you which column is time and which is the value.",
    glossary: "ingestion",
  },
  {
    id: "explore",
    title: "2 · Look before you model",
    body: "Descriptives, plots, data quality and outliers come first. Every number here is computed from your file, nothing is typed in by hand.",
    glossary: "outlier",
  },
  {
    id: "test",
    title: "3 · Check the assumptions",
    body: "Stationarity tests are reconciled in a 2×2 verdict matrix. The advisor explains what to do next and shows a preview before you commit.",
    glossary: "stationarity",
  },
  {
    id: "model",
    title: "4 · Compare, do not just fit",
    body: "Simple baselines are always fitted. Candidates are ranked by rolling-origin cross-validation, and a residual-adequacy verdict is attached.",
    glossary: "rolling_origin_cv",
  },
  {
    id: "forecast",
    title: "5 · Forecast with honest intervals",
    body: "Every interval states its coverage and how it was built. Accuracy is scored out of sample against the naive benchmark.",
    glossary: "prediction_interval",
  },
  {
    id: "report",
    title: "6 · Write it up",
    body: "A weighted scorecard says exactly what is missing, then one click exports HTML, DOCX, LaTeX tables, a reproducible script and the data behind every figure.",
    glossary: "scorecard",
  },
]);

export function tourStepLabel(index: number): string {
  const step = GUIDED_TOUR[index];
  return step ? `${step.title} (${index + 1}/${GUIDED_TOUR.length})` : "Tour finished";
}

/** Keyboard roving-tabindex helper for the WAI-ARIA toolbar (§5.15). */
export function ribbonNextIndex(
  key: string,
  current: number,
  count: number,
  enabled: readonly boolean[] = [],
): number {
  if (count <= 0) return -1;
  const isOn = (i: number) => enabled.length === 0 ? true : Boolean(enabled[i]);
  const find = (start: number, direction: 1 | -1): number => {
    for (let k = 1; k <= count; k += 1) {
      const i = (((start + direction * k) % count) + count) % count;
      if (isOn(i)) return i;
    }
    return current;
  };
  if (key === "ArrowRight" || key === "ArrowDown") return find(current, 1);
  if (key === "ArrowLeft" || key === "ArrowUp") return find(current, -1);
  if (key === "Home") {
    for (let i = 0; i < count; i += 1) if (isOn(i)) return i;
  }
  if (key === "End") {
    for (let i = count - 1; i >= 0; i -= 1) if (isOn(i)) return i;
  }
  return current;
}
