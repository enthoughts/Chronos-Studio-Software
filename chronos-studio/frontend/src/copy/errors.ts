/**
 * copy/errors.ts — the plain-English error library (Part 19, §5.15).
 *
 * The backend already returns { code, plain, fix, technical } for every
 * expected failure (CRAFT R0.9). This module adds the beginner's missing layer:
 * what happened, what to do, what NOT to do, and the glossary terms involved.
 * It is pure data + pure helpers, so the library can be searched, reviewed and
 * unit-tested without a browser.
 */

export interface ErrorHelp {
  code: string;
  group: string;
  /** One short sentence, no jargon. */
  title: string;
  /** The beginner's explanation, two sentences at most. */
  meaning: string;
  /** The next action, phrased as a command. */
  fix: string;
  /** What not to do — prevents the usual wrong workaround. */
  avoid: string;
  terms: readonly string[];
}

export const ERROR_GROUPS: readonly string[] = Object.freeze([
  "Data & files",
  "Pipeline & sessions",
  "Preparation & tests",
  "Models & forecasts",
  "Evaluation & reports",
  "Assistant & privacy",
  "Connection & unexpected",
]);

const E = (
  code: string,
  group: string,
  title: string,
  meaning: string,
  fix: string,
  avoid: string,
  terms: string[],
): ErrorHelp => Object.freeze({ code, group, title, meaning, fix, avoid, terms });

/** Exact entries mirror backend/app/core/errors.py plus the most common
 *  domain-specific API codes. Prefix rules below cover the long tail. */
export const ERROR_LIBRARY: readonly ErrorHelp[] = Object.freeze([
  E("data.unsupported_format", "Data & files", "That file type cannot be read.",
    "The upload reached the app, but this build does not have a reader for that format.",
    "Save the same table as CSV or Excel (.xlsx), then upload it again.",
    "Do not rename the file extension; the contents must actually be converted.",
    ["ingestion", "resolution", "column_mapping"]),
  E("data.too_large", "Data & files", "The file is too large for this browser app.",
    "A very large upload makes the interactive analysis exceed the free-tier memory budget.",
    "Upload a shorter date range, or split the table into separate series.",
    "Do not paste millions of rows into the browser text box.",
    ["ingestion", "observation", "sample"]),
  E("data.empty", "Data & files", "The file has no data rows.",
    "Headers were found, but no observations could be read underneath them.",
    "Check the sheet, delimiter and header row; confirm the file contains at least two rows.",
    "Do not continue with an empty series; later stages have nothing to test.",
    ["observation", "column_mapping", "data_class"]),
  E("data.time_index", "Data & files", "The date column could not be identified.",
    "A time series needs a real, ordered time index. The app could not infer one from the column names or values.",
    "Open Column mapping and choose the date or time column yourself.",
    "Do not use an arbitrary row number as a substitute for a date.",
    ["time_series", "column_mapping", "row_order_is_time"]),
  E("data.frequency", "Data & files", "The observation spacing is unclear.",
    "Daily, monthly, quarterly and yearly data need different seasonal periods.",
    "Choose the frequency explicitly in Column mapping.",
    "Do not let a guessed frequency determine a seasonal model.",
    ["frequency", "period", "season"]),
  E("data.irregular", "Data & files", "The dates are not evenly spaced.",
    "The gaps between observations vary, so the usual time-series algebra does not apply directly.",
    "Aggregate to a coarser regular frequency, or add the missing dates with blank values.",
    "Do not delete inconvenient dates just to make the spacing look regular.",
    ["gap", "missing_value", "frequency"]),
  E("data.column_mapping", "Data & files", "The chosen columns did not match.",
    "The time and value columns selected in the UI are not present in the parsed table.",
    "Re-open Column mapping and pick the columns again.",
    "Do not assume the first column is always time.",
    ["column_mapping", "ingestion", "time_series"]),
  E("data.non_numeric", "Data & files", "The value column contains text.",
    "Forecasting methods need numbers. Currency symbols, commas and error markers are common causes.",
    "Choose a numeric column or clean the text first (remove symbols, thousands separators and '#N/A').",
    "Do not silently coerce text to zero; that creates fake observations.",
    ["observation", "data_class", "impossible_value"]),
  E("data.example_unknown", "Data & files", "That example dataset is not bundled.",
    "The requested seed dataset is not in this build's manifest.",
    "Choose one of the nineteen examples shown in Example data.",
    "Do not paste a FileNotFound traceback into a report.",
    ["seed_data", "provenance", "content_hash"]),
  E("session.not_found", "Pipeline & sessions", "This analysis session has expired.",
    "Sessions live in server memory and are deleted after their time-to-live or when you press Delete session.",
    "Upload the data again and replay the analysis to start a fresh session; a .chronos file is a report export, not a resumable session.",
    "Do not assume a browser refresh restores an expired server session.",
    ["session_ttl", "chronos_archive", "delete_session"]),
  E("session.stage_order", "Pipeline & sessions", "Finish an earlier step first.",
    "Every stage declares its prerequisites. The app refuses to compute a result from missing evidence.",
    "Follow the stepper from left to right, or use Back to revisit the missing stage.",
    "Do not repeatedly press the same button; the prerequisite will still be missing.",
    ["stepper", "stale", "readiness_gate"]),
  E("network.offline", "Connection & unexpected", "The analysis server is not answering.",
    "The page loaded, but the Python service did not respond. This is a connection problem, not a statistical one.",
    "Check that the backend is running, then reload the page.",
    "Do not delete your data; the upload is still in your browser or report archive.",
    ["payload_preview", "session_ttl", "privacy_note"]),
  E("internal.error", "Connection & unexpected", "Something unexpected went wrong.",
    "An unclassified failure occurred. Your stored data and earlier results were not silently changed.",
    "Reload the page and try the step again; expand Technical detail to see the diagnostic.",
    "Do not quote a partial result; wait until the stage reports Complete.",
    ["caveat", "stale", "error_library"]),
  E("prep.non_positive", "Preparation & tests", "A log transform needs positive values.",
    "Logarithms and multiplicative methods are undefined at zero and for negative numbers.",
    "Use an additive method, or apply a Yeo–Johnson / shifted log transform and record the constant.",
    "Do not drop zero rows just to make a log transform legal.",
    ["log_transform", "yeojohnson", "transform", "zero_inflation"]),
  E("prep.insufficient_data", "Preparation & tests", "There are not enough observations.",
    "The requested method needs more data than this series has, often at least two full seasonal cycles.",
    "Choose a simpler method or extend the series before modelling.",
    "Do not pad the series with made-up observations.",
    ["observation", "period", "sample"]),
  E("preprocess.nothing_to_undo", "Preparation & tests", "There is no treatment to undo.",
    "Undo walks back a preprocessing chain. The chain is currently empty or fully unwound.",
    "Apply a treatment first, or check the step strip for the current series version.",
    "Do not refresh the page expecting a new undo slot to appear.",
    ["undo", "series_version", "stale"]),
  E("preprocess.nothing_to_redo", "Preparation & tests", "There is nothing to redo.",
    "Redo replays a treatment you have just undone. Applying a new treatment clears that redo stack by design.",
    "Press Undo first, then Redo; or apply the treatment again from the advisor.",
    "Do not expect redo to survive a new branch of preprocessing.",
    ["redo", "series_version", "stale"]),
  E("gate.blocked", "Pipeline & sessions", "The readiness gate says not yet.",
    "At least one prerequisite has an unresolved blocker. The app will not fit models on unready data.",
    "Open the readiness verdict, read the numbered blockers, and visit the panel each one names.",
    "Do not override the gate; downstream results would inherit invalid assumptions.",
    ["readiness_gate", "verdict", "caveat"]),
  E("model.not_found", "Models & forecasts", "That model is not in this build.",
    "The requested model id is not registered, or its optional library is not installed.",
    "Pick a model from the registry panel; exclusions explain why some families are unavailable.",
    "Do not assume every named method in a textbook is bundled.",
    ["model_registry", "fit_failure", "library_versions"]),
  E("model.fit_failed", "Models & forecasts", "That model could not be fitted.",
    "Estimation did not converge, or the data cannot identify the requested parameters. Other candidates can still succeed.",
    "Try a simpler order, a different family, or remove exogenous drivers.",
    "Do not interpret a failed fit's placeholder parameters.",
    ["fit_failure", "root_check", "parameter"]),
  E("model.not_comparable", "Models & forecasts", "These AIC values cannot be compared.",
    "Information criteria assume the same response data. Differently transformed series have different likelihood scales.",
    "Compare the models with rolling-origin cross-validation instead.",
    "Do not rank AIC columns that the app marks as not comparable.",
    ["ic_comparable", "rolling_origin_cv", "transform"]),
  E("model.leakage", "Models & forecasts", "A leakage guard stopped the analysis.",
    "An internal check found that future information reached a modelling or evaluation choice.",
    "Report this with the report archive; the app withheld the result rather than printing a misleading score.",
    "Do not reuse the questionable score; it is a software bug, not a data problem.",
    ["leakage", "out_of_sample", "golden_file"]),
  E("forecast.back_transform_required", "Models & forecasts", "A back-transformation is required.",
    "The model was fitted on a transformed series, so forecasts must be returned to the original scale explicitly.",
    "Choose the documented back-transformation or bias adjustment before exporting forecasts.",
    "Do not exponentiate a forecast and call the result an unbiased mean.",
    ["back_transformation", "log_transform", "point_forecast"]),
  E("forecast.combination_needs_two", "Models & forecasts", "A combination needs two forecasts.",
    "Forecast combination averages or weights component forecasts; one component is not a combination.",
    "Fit and forecast at least two models, then build the combination again.",
    "Do not duplicate one model to fake a second opinion.",
    ["forecast_combination", "benchmark", "point_forecast"]),
  E("evaluate.leakage_detected", "Evaluation & reports", "The evaluation split is not honest.",
    "A test-period value influenced a choice made before scoring. Out-of-sample results would be optimistic.",
    "Rebuild the rolling-origin splits and re-run the affected stage.",
    "Do not quote the contaminated accuracy number.",
    ["leakage", "rolling_origin_cv", "out_of_sample"]),
  E("evaluate.no_forecast", "Evaluation & reports", "There are no forecasts to score yet.",
    "Evaluation needs stored forecast results and matching held-out observations.",
    "Run the Forecast step, then return to Evaluate.",
    "Do not compare fitted values with forecasts; they are different quantities.",
    ["forecast", "out_of_sample", "metric_policy"]),
  E("evaluate.protocol_refused", "Evaluation & reports", "That metric is not legal here.",
    "The metric policy refuses measures that are undefined or misleading for this data class, e.g. MAPE on zero-heavy series.",
    "Use the recommended point, interval or discrete-distribution metric shown in the policy card.",
    "Do not replace zero denominators with a tiny number just to make MAPE computable.",
    ["metric_policy", "mape", "mase", "zero_inflation"]),
  E("export.build_failed", "Evaluation & reports", "The export could not be built.",
    "A report bundle catches each exporter's failure separately so one broken format does not discard the rest.",
    "Open the bundle's errors.txt, then retry the specific format after installing its optional dependency.",
    "Do not assume a successful ZIP means every requested file is inside.",
    ["report_bundle", "data_behind", "library_versions"]),
  E("export.chromium_missing", "Evaluation & reports", "PDF / PNG rendering needs Chromium.",
    "This deployment omits the headless browser to stay inside the free-tier image size.",
    "Use the self-contained HTML, DOCX or SVG exports, or install Chromium and rebuild the image.",
    "Do not upload a private dataset to an external conversion service.",
    ["dpi", "one_file_html", "library_versions"]),
  E("ai.key_missing", "Assistant & privacy", "No Gemini key was entered.",
    "Live AI is optional. The deterministic offline assistant works from computed results without a key.",
    "Paste a free Google AI Studio key in Settings, or switch to Offline mode.",
    "Do not paste a key into a shared report or commit it to a repository.",
    ["offline_mode", "payload_preview", "privacy_note"]),
  E("ai.preview_required", "Assistant & privacy", "Review the AI payload first.",
    "Before the first live request, the app requires you to see exactly what would be sent.",
    "Open Payload preview, read it, then confirm; otherwise use Offline mode.",
    "Do not enable live AI for data you have not reviewed.",
    ["payload_preview", "analysis_digest", "privacy_note"]),
  E("ai.ungrounded", "Assistant & privacy", "An answer was withheld.",
    "The assistant quoted a number that is not in the computed result digest, so the app refused to show it.",
    "Ask again, or click 'Explain this' on the specific result card.",
    "Do not copy unverified AI numbers into a thesis.",
    ["analysis_digest", "provenance", "caveat"]),
  E("ai.rate_limited", "Assistant & privacy", "The free AI quota is temporarily used up.",
    "The external service returned a rate limit. It does not affect the statistical analysis.",
    "Wait a minute, or keep working with the offline assistant and computed results.",
    "Do not retry in a tight loop; that extends the cooldown.",
    ["offline_mode", "payload_preview", "analysis_digest"]),
]);

/** Long-tail codes share a family-level explanation, using the longest prefix. */
export const ERROR_PREFIX_HELP: readonly ErrorHelp[] = Object.freeze([
  E("data.", "Data & files", "The data could not be read as a regular series.",
    "An ingestion or data-quality check found a problem before any analysis was computed.",
    "Read the data-quality report and correct the file or column mapping, then reload it.",
    "Do not edit the values silently until the report identifies the problem.",
    ["ingestion", "column_mapping", "data_class"]),
  E("preprocess.", "Preparation & tests", "A preparation step could not run.",
    "The requested treatment needs a valid preview, an original series, or a longer sample.",
    "Return to Prepare data, check the step strip, and preview the treatment again.",
    "Do not assume the series version changed when an operation failed.",
    ["preview", "series_version", "stale"]),
  E("decompose.", "Preparation & tests", "The decomposition could not run.",
    "A decomposition needs a profile and a legal seasonal period; multiplicative forms also need positive values.",
    "Run Step 1, choose a valid period, or use an additive method.",
    "Do not quote trend strength from a failed decomposition.",
    ["decomposition", "multiplicative", "period"]),
  E("next.", "Preparation & tests", "The next-step recommendation is unavailable.",
    "Ranking actions needs stationarity evidence and a valid lag or seasonal period.",
    "Complete Step 5 and Step 6, then revisit Step 7.",
    "Do not difference repeatedly without reading the reconciliation matrix.",
    ["reconciliation", "difference", "integration_order"]),
  E("forecast.", "Models & forecasts", "The forecast could not be produced.",
    "A forecast needs successful fits, a valid horizon and, for transformed data, a declared back-transformation.",
    "Return to Fit models, then check the Forecast controls and the model card.",
    "Do not extend the horizon beyond the data's defensible seasonal coverage.",
    ["forecast", "horizon", "back_transformation"]),
  E("evaluate.", "Evaluation & reports", "The evaluation could not be completed.",
    "Scoring needs stored forecasts, matching original values and a legal metric policy.",
    "Run Forecast, then choose the recommended out-of-sample metrics and folds.",
    "Do not use in-sample residuals as a forecast accuracy score.",
    ["out_of_sample", "metric_policy", "rolling_origin_cv"]),
  E("export.", "Evaluation & reports", "An export is unavailable.",
    "This build reports missing optional exporters rather than producing an empty or fabricated file.",
    "Use the one-file HTML or another listed format, or install the optional dependency described in DEPLOY.md.",
    "Do not assume a ZIP contains every format when errors.txt lists a failure.",
    ["report_bundle", "one_file_html", "library_versions"]),
  E("catalog.", "Evaluation & reports", "A catalogue item could not be rendered.",
    "The item may need an optional plotting dependency or a Chromium binary that this deployment omits.",
    "Use the HTML or SVG export, or install the noted optional dependency.",
    "Do not treat a missing preview as missing underlying data.",
    ["data_behind", "alt_text", "library_versions"]),
  E("ai.", "Assistant & privacy", "The assistant could not complete that request.",
    "Live AI is optional and guarded by payload review, grounding checks and request limits.",
    "Use Offline mode, or read the notification and adjust Settings before retrying.",
    "Do not send raw values or a key without reviewing the payload.",
    ["offline_mode", "payload_preview", "privacy_note"]),
  E("session.", "Pipeline & sessions", "The session state is not available.",
    "Sessions are in-memory and expire by design; the app will not reconstruct one silently.",
    "Start a new analysis by uploading the data again; a .chronos file is a report export, not a resumable session.",
    "Do not rely on a browser refresh after the session TTL.",
    ["session_ttl", "chronos_archive", "delete_session"]),
]);

const EXACT = new Map(ERROR_LIBRARY.map((e) => [e.code, e]));

export const FALLBACK_ERROR_HELP: ErrorHelp = Object.freeze({
  code: "unknown",
  group: "Connection & unexpected",
  title: "Something went wrong, but your analysis is safe.",
  meaning: "The app could not classify this failure. Earlier computed results were not changed.",
  fix: "Read the suggested fix in the notification, then retry the step. Expand Technical detail if it persists.",
  avoid: "Do not quote a partial result until the stage reports Complete.",
  terms: ["caveat", "stale", "error_library"],
});

export function errorHelp(code: string): ErrorHelp {
  if (!code) return FALLBACK_ERROR_HELP;
  const exact = EXACT.get(code);
  if (exact) return exact;
  let best: ErrorHelp | undefined;
  for (const e of ERROR_PREFIX_HELP) {
    if (code.startsWith(e.code) && (!best || e.code.length > best.code.length)) best = e;
  }
  return best ?? { ...FALLBACK_ERROR_HELP, code };
}

export function errorSearch(
  entries: readonly ErrorHelp[],
  query: string,
  group = "",
): ErrorHelp[] {
  const q = query.trim().toLowerCase();
  return entries.filter((e) => {
    if (group && e.group !== group) return false;
    if (!q) return true;
    return [e.code, e.group, e.title, e.meaning, e.fix, e.avoid, ...e.terms]
      .join(" \n ").toLowerCase().includes(q);
  });
}

/** The one shape every toast-producing helper should use. */
export interface ErrorToastPayload {
  kind: "error";
  title: string;
  body: string;
  fix: string;
  errorCode: string;
}

function isApiError(e: unknown): e is Error & {
  code: string;
  fix: string;
  technical: string;
} {
  return Boolean(e) && typeof e === "object" && e instanceof Error &&
    "code" in e && "fix" in e && typeof (e as { code: unknown }).code === "string";
}

/**
 * Convert any thrown value into a friendly toast. The backend's plain sentence
 * remains the title; the library adds a code that the UI can link to the full
 * plain-English article, and the technical trace stays collapsed.
 */
export function errorToastPayload(e: unknown, fallback = "That step could not be completed."): ErrorToastPayload {
  if (isApiError(e)) {
    const help = errorHelp(e.code);
    return {
      kind: "error",
      title: e.message || help.title,
      body: [e.technical || e.code, help.meaning].filter(Boolean).join("\n\n"),
      fix: e.fix || help.fix,
      errorCode: e.code,
    };
  }
  const message = e instanceof Error ? e.message : String(e ?? "");
  return {
    kind: "error",
    title: fallback,
    body: message || FALLBACK_ERROR_HELP.meaning,
    fix: FALLBACK_ERROR_HELP.fix,
    errorCode: "unknown",
  };
}
