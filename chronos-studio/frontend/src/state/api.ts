/**
 * src/state/api.ts — the only place the browser talks to the backend.
 *
 * Rules:
 *  - relative URLs only ("/api/v1/..."), never localhost (the preview proxy
 *    and production both need this to work unchanged)
 *  - the session id travels in the X-Session-Id header
 *  - every failure becomes a typed ApiError carrying the backend's plain/fix
 */

const BASE = "/api/v1";

export class ApiError extends Error {
  readonly code: string;
  readonly fix: string;
  readonly technical: string;
  readonly status: number;

  constructor(status: number, code: string, plain: string, fix: string, technical = "") {
    super(plain);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.fix = fix;
    this.technical = technical;
  }
}

function sessionFromStorage(): string | null {
  try {
    return localStorage.getItem("chronos.session");
  } catch {
    return null;
  }
}

export function rememberSession(id: string | null): void {
  try {
    if (id) localStorage.setItem("chronos.session", id);
    else localStorage.removeItem("chronos.session");
  } catch {
    /* private browsing — sessions stay in memory only */
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  sessionId?: string | null,
): Promise<T> {
  const headers = new Headers(init.headers);
  const sid = sessionId ?? sessionFromStorage();
  if (sid) headers.set("X-Session-Id", sid);
  if (init.body && !(init.body instanceof FormData)) {
    headers.set("Content-Type", "application/json");
  }

  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(
      0,
      "network.offline",
      "The analysis server could not be reached.",
      "Check that the backend is running, then reload the page.",
    );
  }

  if (!res.ok) {
    let body: { error?: { code?: string; plain?: string; fix?: string; technical?: string } } = {};
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    const e = body.error ?? {};
    throw new ApiError(
      res.status,
      e.code ?? "internal.error",
      e.plain ?? `The server returned ${res.status}.`,
      e.fix ?? "Reload the page and try again.",
      e.technical ?? "",
    );
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

/** A binary catalogue download. Errors still become the typed ApiError above. */
async function downloadRequest(
  path: string,
  sessionId?: string | null,
): Promise<{ blob: Blob; filename: string }> {
  const headers = new Headers();
  const sid = sessionId ?? sessionFromStorage();
  if (sid) headers.set("X-Session-Id", sid);
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { headers });
  } catch {
    throw new ApiError(
      0,
      "network.offline",
      "The analysis server could not be reached.",
      "Check that the backend is running, then reload the page.",
    );
  }
  if (!res.ok) {
    let body: { error?: { code?: string; plain?: string; fix?: string; technical?: string } } = {};
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    const e = body.error ?? {};
    throw new ApiError(
      res.status,
      e.code ?? "internal.error",
      e.plain ?? `The server returned ${res.status}.`,
      e.fix ?? "Reload the page and try again.",
      e.technical ?? "",
    );
  }
  const disposition = res.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^";]+)"?/i.exec(disposition);
  return { blob: await res.blob(), filename: match?.[1] ?? "chronos-catalogue" };
}

/* ------------------------------ endpoints -------------------------------- */

export const api = {
  health: () => request<import("./types").HealthResponse>("/health"),

  pipelineMap: () =>
    request<{ stages: import("./types").StageDef[] }>("/pipeline"),

  createSession: (seed?: number) =>
    request<{ session_id: string; seed: number; pipeline: unknown }>("/sessions", {
      method: "POST",
      body: seed ? JSON.stringify({ seed }) : undefined,
    }),

  getSession: (id: string) => request<import("./types").SessionState>(`/sessions/${id}`),

  deleteSession: (id: string) =>
    request<{ deleted: boolean }>(`/sessions/${id}`, { method: "DELETE" }),

  results: (id: string) =>
    request<Record<string, import("./types").AnyResult[]>>(`/sessions/${id}/results`),

  explain: (id: string, resultId: string) =>
    request<Record<string, unknown>>(
      `/sessions/${id}/results/${resultId}/explain`,
      { method: "POST" },
      id,
    ),

  /* ---------------------------- Part 1: data ---------------------------- */

  dataFormats: () =>
    request<{
      extensions: string[];
      max_upload_mb: number;
      calendars: import("./types").CalendarOption[];
    }>("/data/formats"),

  examples: () =>
    request<{ examples: import("./types").ExampleDataset[] }>("/data/examples"),

  upload: (file: File, mapping?: unknown, sheetName?: string) => {
    const form = new FormData();
    form.append("file", file);
    if (mapping) form.append("mapping_json", JSON.stringify(mapping));
    if (sheetName) form.append("sheet_name", sheetName);
    return request<import("./types").IngestResponse>("/data/upload", {
      method: "POST",
      body: form,
    });
  },

  paste: (text: string, mapping?: unknown) =>
    request<import("./types").IngestResponse>("/data/paste", {
      method: "POST",
      body: JSON.stringify({ text, mapping }),
    }),

  loadExample: (exampleId: string) =>
    request<import("./types").IngestResponse>("/data/example", {
      method: "POST",
      body: JSON.stringify({ example_id: exampleId }),
    }),

  inspect: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{
      columns: { name: string; samples: string[] }[];
      suggested_mapping: import("./types").ColumnMapping;
      issues: import("./types").ParseIssue[];
      notes: string[];
      n_rows: number;
      format: string;
      loader_notes: string[];
      profiles: import("./types").IngestResponse["profiles"];
    }>("/data/inspect", { method: "POST", body: form });
  },

  remap: (mapping: unknown) =>
    request<import("./types").IngestResponse>("/data/remap", {
      method: "POST",
      body: JSON.stringify({ mapping }),
    }),

  currentDataset: () =>
    request<import("./types").CurrentDatasetResponse>("/data/current"),

  /* ------------------------- Part 2: exploration ------------------------- */

  runProfile: (window?: number, audience = "beginner") =>
    request<import("./types").AnalysisResponse>("/analysis/profile", {
      method: "POST",
      body: JSON.stringify({ window: window ?? null, audience }),
    }),

  profileResults: () =>
    request<{
      stage: string;
      label: string;
      status: string;
      results: import("./types").AnyResult[];
      facts: Record<string, number | string | boolean>;
      caveats: string[];
    }>("/analysis/profile/results"),

  facts: () =>
    request<{
      eda: Record<string, number | string | boolean>;
      dataset: Record<string, number | string>;
      series_version: number;
      seed: number;
    }>("/analysis/facts"),

  /* ---------------------- Part 7: the readiness gate ---------------------- */

  runGate: () =>
    request<import("./types").AnalysisResponse>("/analysis/gate", { method: "POST" }),

  getGate: () => request<import("./types").GateResponse>("/analysis/gate"),

  setHorizon: (horizon: number) =>
    request<import("./types").HorizonResponse>("/analysis/gate/horizon", {
      method: "POST",
      body: JSON.stringify({ horizon }),
    }),

  /* ------------------ Part 8: the registry and fitting -------------------- */

  getRegistry: () =>
    request<{ n_rows: number; n_implemented: number; rows: Record<string, unknown>[] }>(
      "/analysis/registry",
    ),

  getSuggestions: () => request<import("./types").SuggestionSet>("/analysis/suggest"),

  runSuggest: () =>
    request<import("./types").AnalysisResponse>("/analysis/suggest", { method: "POST" }),

  runFit: (model_ids: string[]) =>
    request<import("./types").AnalysisResponse>("/analysis/fit", {
      method: "POST",
      body: JSON.stringify({ model_ids }),
    }),

  getFits: () => request<import("./types").FitsResponse>("/analysis/fits"),

  fitForecast: (model_id: string, horizon: number) =>
    request<import("./types").FitForecast>("/analysis/fit/forecast", {
      method: "POST",
      body: JSON.stringify({ model_id, horizon }),
    }),

  multivariate: (model_id: string) =>
    request<import("./types").MultivariateResponse>("/analysis/fit/multivariate", {
      method: "POST",
      body: JSON.stringify({ model_id }),
    }),

  /* ------------------- Part 6: ACF / PACF and next steps ------------------ */

  getCorrelate: () => request<import("./types").CorrelateResponse>("/analysis/correlate"),

  crossCorrelate: (body: { column: string; max_lag?: number; prewhiten?: boolean }) =>
    request<{
      column: string; n: number; band: number;
      peak: { lag: number; value: number; significant: boolean; leads_lags: string };
      raw_peak: { lag: number; value: number; significant: boolean; leads_lags: string };
      filter_ar: number[]; rows: { lag: number; ccf: number; band: number; significant: boolean }[];
      raw_rows: { lag: number; ccf: number; band: number; significant: boolean }[];
      warning: string | null;
      figure: Record<string, unknown>; alt_text: string; caveats: string[];
    }>("/analysis/correlate/ccf", { method: "POST", body: JSON.stringify(body) }),

  getNext: () => request<import("./types").NextResponse>("/analysis/next"),

  previewAction: (body: {
    op: "difference" | "transform.log" | "undo";
    order?: number | "seasonal";
    rule_id?: string | null;
  }) =>
    request<import("./types").ActionPreview & { rule_id: string | null }>(
      "/analysis/next/preview",
      { method: "POST", body: JSON.stringify(body) },
    ),

  applyNext: (body: {
    op: "difference" | "transform.log";
    order?: number | "seasonal";
    rule_id?: string | null;
    reason?: string;
  }) =>
    request<import("./types").AnalysisResponse>("/analysis/next/apply", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /* ----------------------- Part 4: decomposition ----------------------- */

  decomposeMethods: () =>
    request<{
      n: number;
      period: number | null;
      methods: import("./types").MethodOption[];
      models: { model: string; available: boolean; why_not: string }[];
    }>("/analysis/decompose/methods"),

  runDecompose: (body: {
    method: string;
    model?: "additive" | "multiplicative";
    period?: number | null;
  }) =>
    request<import("./types").AnalysisResponse & { advisor: import("./types").AdvisorReport | null }>(
      "/analysis/decompose",
      { method: "POST", body: JSON.stringify(body) },
    ),

  getDecompose: () => request<import("./types").DecompResponse>("/analysis/decompose"),

  compareDecompositions: (methods: string[], model: "additive" | "multiplicative" = "additive") =>
    request<import("./types").CompareResponse>("/analysis/decompose/compare", {
      method: "POST",
      body: JSON.stringify({ methods, model }),
    }),

  /* ----------------------- Part 5: stationarity ----------------------- */

  runTests: (audience = "beginner", runBreaks = true) =>
    request<import("./types").AnalysisResponse>("/analysis/tests", {
      method: "POST",
      body: JSON.stringify({ audience, run_breaks: runBreaks }),
    }),

  getTests: () => request<import("./types").StatTestsResponse>("/analysis/tests"),

  reconcileTests: () =>
    request<import("./types").ReconcileResponse>("/analysis/tests/reconcile", {
      method: "POST",
    }),

  runAdvice: (audience = "beginner") =>
    request<import("./types").AdviceResponse>("/analysis/advice", {
      method: "POST",
      body: JSON.stringify({ audience }),
    }),

  advice: () =>
    request<import("./types").AdviceResponse & { status?: string }>("/analysis/advice"),

  previewOp: (op: string, params: Record<string, unknown> = {}) =>
    request<import("./types").PreviewResponse>("/preprocess/preview", {
      method: "POST",
      body: JSON.stringify({ op, params }),
    }),

  applyOp: (op: string, params: Record<string, unknown> = {}, ruleId?: string) =>
    request<import("./types").AnalysisResponse>("/preprocess/apply", {
      method: "POST",
      body: JSON.stringify({ op, params, rule_id: ruleId ?? null }),
    }),

  undoOp: () =>
    request<import("./types").AnalysisResponse>("/preprocess/undo", { method: "POST" }),

  preprocessState: () => request<import("./types").PreprocessState>("/preprocess/state"),

  seriesPoints: (limit = 2000) =>
    request<{
      points: { time: string; value: number }[];
      n: number;
      n_displayed: number;
      decimated: boolean;
      note: string;
      missing: number;
    }>(`/analysis/series?limit=${limit}`),

  select: (body: {
    horizon?: number;
    folds?: number;
    scheme?: string;
    metric?: string;
    models?: string[];
    include_baselines?: boolean;
  }) =>
    request<import("./types").AnalysisResponse>("/analysis/select", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  selection: () => request<import("./types").SelectionResponse>("/analysis/selection"),

  diagnose: (body: { models?: string[]; period?: number | null } = {}) =>
    request<import("./types").AnalysisResponse>("/analysis/diagnose", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  diagnostics: () => request<import("./types").DiagnosticsResponse>("/analysis/diagnostics"),

  /* ------------------- Part 14: S13 forecasting ----------------------- */

  forecast: () => request<import("./types").ForecastResponse>("/analysis/forecast"),

  runForecast: (body: import("./types").ForecastRequestPayload) =>
    request<import("./types").AnalysisResponse>("/analysis/forecast", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  horizonAdvice: (horizon: number, acknowledged = false) =>
    request<{ stage: string } & import("./types").HorizonAdvicePayload>(
      `/analysis/forecast/horizon?horizon=${horizon}&acknowledged=${acknowledged}`,
    ),

  setScenario: (body: { horizon: number; paths: Record<string, number[]> }) =>
    request<import("./types").AnalysisResponse>("/analysis/scenario", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  reconcile: (body: {
    bottom: Record<string, number[]>;
    total?: number[] | null;
    model_id?: string | null;
    horizon?: number;
    method?: "bottom_up" | "top_down" | "ols";
  }) =>
    request<import("./types").AnalysisResponse>("/analysis/forecast/hierarchy", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  hierarchy: () => request<import("./types").HierarchyResponse>("/analysis/forecast/hierarchy"),

  /* ------------------- Part 15: S14 evaluation metrics -------------------- */

  evaluation: () =>
    request<import("./types").EvaluationResponse>("/analysis/evaluate"),

  runEvaluation: (body: import("./types").EvaluationRequestPayload) =>
    request<import("./types").AnalysisResponse>("/analysis/evaluate", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /* ------------------- Part 16: plots & tables catalogue ------------------- */

  catalog: (mode: import("./types").CatalogMode = "light") =>
    request<import("./types").CatalogResponse>(`/analysis/catalog?mode=${mode}`),

  catalogReport: (mode: import("./types").CatalogMode = "light") =>
    request<import("./types").CatalogReportResponse>(`/analysis/catalog/report?mode=${mode}`),

  catalogReportAction: (
    body: import("./types").CatalogReportAction,
    mode: import("./types").CatalogMode = "light",
  ) =>
    request<import("./types").CatalogReportResponse>(`/analysis/catalog/report?mode=${mode}`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  catalogDownload: (
    itemId: string,
    format: import("./types").CatalogFormat,
    mode: import("./types").CatalogMode = "light",
  ) =>
    downloadRequest(
      `/analysis/catalog/${encodeURIComponent(itemId)}/download?format=${format}&mode=${mode}`,
    ),

  /* ------------------- Part 18: reports & finalization ------------------- */

  reportStatus: (
    mode: import("./types").ReportMode = "print",
    scope: import("./types").ReportScope = "all",
  ) =>
    request<import("./types").ReportStatus>(
      `/export/status?mode=${mode}&scope=${scope}`,
    ),

  finalizeReport: (
    mode: import("./types").ReportMode = "print",
    scope: import("./types").ReportScope = "all",
  ) =>
    request<import("./types").ReportFinalizeResponse>(
      `/export/finalize?mode=${mode}&scope=${scope}`,
      { method: "POST", body: "{}" },
    ),

  exportDownload: (
    kind: import("./types").ReportKind,
    mode: import("./types").ReportMode = "print",
    scope: import("./types").ReportScope = "all",
    forceDownload = false,
  ) =>
    downloadRequest(
      `/export/${kind}?mode=${mode}&scope=${scope}&download=${forceDownload ? "1" : "0"}`,
    ),

  reset: (id: string, fromStage?: string) =>
    request<{ reset_from: string; pipeline: unknown }>(
      `/sessions/${id}/reset${fromStage ? `?from_stage=${fromStage}` : ""}`,
      { method: "POST" },
      id,
    ),

  /* ------------------- Part 19: demo mode & background jobs -------------- */

  demoScenarios: () =>
    request<{ scenarios: import("../copy/demo").DemoScenario[] }>("/demo/scenarios"),

  demoStart: (scenarioId: string) =>
    request<import("../copy/demo").DemoStatus>("/demo/start", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId }),
    }),

  demoStatus: (jobId: string) =>
    request<import("../copy/demo").DemoStatus>(
      `/demo/status?job_id=${encodeURIComponent(jobId)}`,
    ),

  demoCancel: (jobId: string) =>
    request<import("../copy/demo").DemoStatus>(
      `/demo/cancel?job_id=${encodeURIComponent(jobId)}`,
      { method: "POST" },
    ),

  /* ------------------- Part 19: preprocessing redo ----------------------- */

  redoOp: () =>
    request<import("./types").AnalysisResponse>("/preprocess/redo", {
      method: "POST",
      body: "{}",
    }),
};
