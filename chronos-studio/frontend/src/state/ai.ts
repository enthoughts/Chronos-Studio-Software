/**
 * state/ai.ts — Part 17 assistant client.
 *
 * The Gemini key lives in `sessionStorage` for this browser tab only. It is sent
 * in the `X-Gemini-Key` header, never in a URL or JSON body, and the app keeps
 * working in offline mode with no key at all.
 */

import { ApiError } from "./api";

const BASE = "/api/v1";
const PREFS_KEY = "chronos.ai.prefs";

export type AIActionName =
  | "ask" | "explain" | "interpret_plot" | "what_next" | "write_up"
  | "simplify" | "audience" | "eli15";
export type AIAudience = "beginner" | "peer" | "reviewer";
export type AIMode = "offline" | "gemini" | "degraded";
export type AIRequestMode = "auto" | "offline" | "gemini";
export type AIModelChoice = "flash" | "pro";

export interface AIPrefs {
  key: string;
  mode: AIRequestMode;
  model: AIModelChoice;
  audience: AIAudience;
  includeValues: boolean;
  eli15: boolean;
  previewAck: boolean;
}

export const DEFAULT_PREFS: AIPrefs = {
  key: "",
  mode: "auto",
  model: "flash",
  audience: "beginner",
  includeValues: false,
  eli15: false,
  previewAck: false,
};

export interface AssistantSeed {
  action: AIActionName;
  resultId?: string | null;
  panelId?: string;
  question?: string;
}

export interface AIChatBody {
  question: string;
  action: AIActionName;
  result_id?: string | null;
  panel_id: string;
  audience: AIAudience;
  eli15: boolean;
  include_values: boolean;
  mode: AIRequestMode;
  model?: AIModelChoice;
  image_data_url?: string | null;
}

export interface AIPreviewBody {
  question: string;
  action: AIActionName;
  result_id?: string | null;
  panel_id: string;
  audience: AIAudience;
  eli15: boolean;
  include_values: boolean;
  model?: AIModelChoice;
  mode: AIRequestMode;
  image_data_url?: string | null;
  acknowledge: boolean;
}

export interface AIConfig {
  offline_ready: boolean;
  sdk_available: boolean;
  env_key_present: boolean;
  live_ready: boolean;
  default_model: string;
  pro_model: string;
  temperature: number;
  max_requests_per_session: number;
  max_requests_per_minute: number;
  min_interval_seconds: number;
  max_values: number;
  allow_data_values_by_default: boolean;
  input_usd_per_million: number;
  output_usd_per_million: number;
  privacy_note: string;
}

export interface LiveAvailability {
  ready: boolean;
  reason: "checking" | "sdk" | "key" | "ready";
  title: string;
  message: string;
}

export interface AIUsage {
  requests: number;
  remaining: number;
  max_requests: number;
  prompt_tokens: number;
  response_tokens: number;
  estimated_cost_usd: number;
  queued: number;
  degraded: boolean;
  rate_limited: boolean;
}

export interface AIAction {
  label: string;
  stage: string;
  kind: "open_panel" | "explain_result";
  rule_id: string;
  result_id: string | null;
  description: string;
}

export interface AIGrounding {
  grounded: boolean;
  citations: string[];
  ungrounded_numbers: string[];
  missing_citations: boolean;
  fallback_used: boolean;
}

export interface AIHistoryMessage {
  role: "user" | "assistant";
  content: string;
  action: AIActionName;
  panel_id: string;
  citations: string[];
  grounded: boolean;
  ungrounded_numbers: string[];
  mode: AIMode;
  prompt_tokens: number;
  response_tokens: number;
  timestamp: string;
}

export interface AIHistoryResponse {
  messages: AIHistoryMessage[];
  usage: AIUsage;
  preview_acknowledged: boolean;
  session_id: string;
}

export interface AIPayloadPreview {
  mode: AIMode;
  model: string | null;
  system_prompt: string;
  contents: { role: string; text: string }[];
  digest: Record<string, unknown>;
  question: string;
  panel_id: string;
  estimated_prompt_tokens: number;
  api_key_in_payload: boolean;
  data_values_included: boolean;
  image_attached: boolean;
  image_bytes: number;
  image_mime: string | null;
  privacy_note: string;
  preview_acknowledged: boolean;
}

export interface AIMeta {
  mode: AIMode;
  model: string | null;
  action: AIActionName;
  panel_id: string;
  citations: string[];
  usage: AIUsage;
}

export interface AIChunk { text: string }
export interface AIErrorEvent { code: string; plain: string; fix: string }
export interface AIDone {
  text: string;
  grounding: AIGrounding;
  citations: string[];
  actions: AIAction[];
  usage: AIUsage;
  mode: AIMode;
  ai_drafted: boolean;
  draft_label: string;
}

export interface AIStreamEvent {
  event: string;
  data: unknown;
}

const emptyUsage: AIUsage = {
  requests: 0, remaining: 60, max_requests: 60,
  prompt_tokens: 0, response_tokens: 0, estimated_cost_usd: 0,
  queued: 0, degraded: false, rate_limited: false,
};

function storage(): Storage | null {
  try {
    return typeof sessionStorage === "undefined" ? null : sessionStorage;
  } catch {
    return null;
  }
}

function sessionFromStorage(): string | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage.getItem("chronos.session");
  } catch {
    return null;
  }
}

export function loadAIPrefs(): AIPrefs {
  const raw = storage()?.getItem(PREFS_KEY);
  if (!raw) return { ...DEFAULT_PREFS };
  try {
    const parsed = JSON.parse(raw) as Partial<AIPrefs>;
    return {
      key: typeof parsed.key === "string" ? parsed.key : "",
      mode: parsed.mode === "offline" || parsed.mode === "gemini" ? parsed.mode : "auto",
      model: parsed.model === "pro" ? "pro" : "flash",
      audience: parsed.audience === "peer" || parsed.audience === "reviewer"
        ? parsed.audience : "beginner",
      includeValues: parsed.includeValues === true,
      eli15: parsed.eli15 === true,
      previewAck: parsed.previewAck === true,
    };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

export function saveAIPrefs(patch: Partial<AIPrefs>): AIPrefs {
  const next = { ...loadAIPrefs(), ...patch };
  try {
    storage()?.setItem(PREFS_KEY, JSON.stringify(next));
  } catch {
    /* private browsing: key stays in memory for this render only */
  }
  return next;
}

export function clearAIPrefs(): void {
  try {
    storage()?.removeItem(PREFS_KEY);
  } catch {
    /* nothing persisted to clear */
  }
}

export function chatBody(
  question: string,
  action: AIActionName,
  prefs: AIPrefs,
  extra: Partial<AIChatBody> = {},
): AIChatBody {
  return {
    question,
    action,
    panel_id: "S15",
    audience: prefs.audience,
    eli15: prefs.eli15,
    include_values: prefs.includeValues,
    mode: prefs.mode,
    model: prefs.model,
    ...extra,
  };
}

export function previewBody(
  question: string,
  action: AIActionName,
  prefs: AIPrefs,
  extra: Partial<AIPreviewBody> = {},
): AIPreviewBody {
  return {
    question,
    action,
    panel_id: "S15",
    audience: prefs.audience,
    eli15: prefs.eli15,
    include_values: prefs.includeValues,
    model: prefs.model,
    mode: prefs.mode,
    acknowledge: false,
    ...extra,
  };
}

async function asApiError(res: Response): Promise<ApiError> {
  let body: { error?: { code?: string; plain?: string; fix?: string; technical?: string } } = {};
  try {
    body = await res.json() as typeof body;
  } catch {
    /* non-JSON error body */
  }
  const e = body.error ?? {};
  return new ApiError(
    res.status,
    e.code ?? "internal.error",
    e.plain ?? `The server returned ${res.status}.`,
    e.fix ?? "Reload the page and try again.",
    e.technical ?? "",
  );
}

function headersFor(prefs: AIPrefs, sessionId?: string | null): Headers {
  const headers = new Headers({ "Content-Type": "application/json" });
  const sid = sessionId ?? sessionFromStorage();
  if (sid) headers.set("X-Session-Id", sid);
  if (prefs.key.trim()) headers.set("X-Gemini-Key", prefs.key.trim());
  return headers;
}

async function aiFetch<T>(
  path: string,
  init: RequestInit,
  prefs: AIPrefs,
  sessionId?: string | null,
): Promise<T> {
  const headers = new Headers(init.headers);
  headersFor(prefs, sessionId).forEach((value, key) => headers.set(key, value));
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(
      0, "network.offline", "The analysis server could not be reached.",
      "Check that the backend is running, then reload the page.",
    );
  }
  if (!res.ok) throw await asApiError(res);
  if (res.status === 204) return undefined as T;
  return await res.json() as T;
}

export function parseSSEBlock(block: string): AIStreamEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  if (data.length === 0) return null;
  try {
    return { event, data: JSON.parse(data.join("\n")) as unknown };
  } catch {
    return null;
  }
}

export async function aiStream(
  body: AIChatBody,
  prefs: AIPrefs,
  onEvent: (event: AIStreamEvent) => void,
  signal?: AbortSignal,
  sessionId?: string | null,
): Promise<void> {
  const headers = headersFor(prefs, sessionId);
  if (prefs.previewAck) headers.set("X-Chronos-AI-Preview-Ack", "true");
  let res: Response;
  try {
    res = await fetch(`${BASE}/ai/chat/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
  } catch {
    if (signal?.aborted) {
      throw new ApiError(0, "ai.cancelled", "Answer stopped.", "No partial answer was saved.");
    }
    throw new ApiError(
      0, "network.offline", "The analysis server could not be reached.",
      "Check that the backend is running, then reload the page.",
    );
  }
  if (!res.ok) throw await asApiError(res);
  if (!res.body) {
    throw new ApiError(0, "ai.stream_missing", "The answer stream was empty.", "Try again.");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx = buffer.indexOf("\n\n");
    while (idx >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const event = parseSSEBlock(block);
      if (event) onEvent(event);
      idx = buffer.indexOf("\n\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) {
    const event = parseSSEBlock(buffer);
    if (event) onEvent(event);
  }
}

export const aiApi = {
  config: (prefs: AIPrefs) =>
    aiFetch<AIConfig>("/ai/config", { method: "GET" }, prefs),

  preview: (body: AIPreviewBody, prefs: AIPrefs, sessionId?: string | null) =>
    aiFetch<AIPayloadPreview>("/ai/preview", {
      method: "POST", body: JSON.stringify(body),
    }, prefs, sessionId),

  history: (prefs: AIPrefs, sessionId?: string | null) =>
    aiFetch<AIHistoryResponse>("/ai/history", { method: "GET" }, prefs, sessionId),

  clearHistory: (prefs: AIPrefs, sessionId?: string | null) =>
    aiFetch<{ cleared: boolean; usage: AIUsage }>("/ai/history", { method: "DELETE" }, prefs, sessionId),

  deleteSession: (prefs: AIPrefs, sessionId?: string | null) =>
    aiFetch<{ deleted: boolean; session_id: string }>("/ai/session", { method: "DELETE" }, prefs, sessionId),

  stream: aiStream,
};

export function formatUsage(usage: AIUsage = emptyUsage): string {
  const tokens = usage.prompt_tokens + usage.response_tokens;
  const cost = usage.estimated_cost_usd.toFixed(6);
  return `${usage.requests}/${usage.max_requests} requests · ${tokens} tokens · $${cost} est.`;
}

export function assistantModeLabel(mode: AIMode, config: AIConfig | null): string {
  if (mode === "gemini") return "Live Gemini";
  if (mode === "degraded") return "Live AI unavailable — offline answer";
  if (config?.live_ready) return "Offline mode (live AI available)";
  return "Offline mode — no key needed";
}

export function liveAvailability(
  config: AIConfig | null,
  prefs: AIPrefs,
): LiveAvailability {
  if (!config) {
    return {
      ready: false, reason: "checking", title: "Live AI status",
      message: "Checking live AI availability…",
    };
  }
  if (config.live_ready) {
    return {
      ready: true, reason: "ready", title: "Live AI ready",
      message: "The google-genai SDK and a key are available. Review the payload preview before first use.",
    };
  }
  if (!config.sdk_available) {
    return {
      ready: false, reason: "sdk", title: "Live AI not ready",
      message: "The google-genai SDK is missing on the server. Offline mode still works. Install it with pip install -r backend/requirements-deploy.txt, then restart the backend.",
    };
  }
  const typedKey = Boolean(prefs.key.trim());
  return {
    ready: false, reason: "key", title: "Live AI not ready",
    message: typedKey
      ? "The server did not confirm Live AI with the key currently in Settings. Offline mode still works. Re-enter a valid Google AI Studio key in Assistant Settings, or set GEMINI_API_KEY in backend/.env and restart the backend."
      : "No API key is configured. Offline mode still works. Enter your own Google AI Studio key in Assistant Settings, or set GEMINI_API_KEY in backend/.env and restart the backend.",
  };
}

/** The first-use live-AI preview is irrelevant when the user chose Offline only. */
export function shouldAutoOpenPreview(assistantOpen: boolean, prefs: AIPrefs): boolean {
  return assistantOpen && !prefs.previewAck && prefs.mode !== "offline";
}

export function groundingWarnings(grounding: AIGrounding | null): string[] {
  if (!grounding) return [];
  const out: string[] = [];
  if (grounding.ungrounded_numbers.length > 0) {
    out.push(`Flagged numbers not in the digest: ${grounding.ungrounded_numbers.join(", ")}`);
  }
  if (grounding.missing_citations) out.push("Missing result-id citations.");
  return out;
}

export function privacyBanner(prefs: AIPrefs): string | null {
  return prefs.includeValues
    ? "Data values will be sent with this request. Review the payload preview before continuing."
    : null;
}

export { emptyUsage };
