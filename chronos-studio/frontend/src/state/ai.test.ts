import { afterEach, describe, expect, it, vi } from "vitest";

import {
  DEFAULT_PREFS,
  assistantModeLabel,
  chatBody,
  emptyUsage,
  formatUsage,
  groundingWarnings,
  loadAIPrefs,
  liveAvailability,
  parseSSEBlock,
  previewBody,
  privacyBanner,
  saveAIPrefs,
  shouldAutoOpenPreview,
  aiApi,
  type AIConfig,
  type AIGrounding,
  type AIPrefs,
} from "./ai";

function mockSessionStorage(): Record<string, string> {
  const data: Record<string, string> = {};
  const store = {
    getItem: (key: string) => data[key] ?? null,
    setItem: (key: string, value: string) => { data[key] = value; },
    removeItem: (key: string) => { delete data[key]; },
    clear: () => { Object.keys(data).forEach((k) => delete data[k]); },
    key: (i: number) => Object.keys(data)[i] ?? null,
    get length() { return Object.keys(data).length; },
  };
  Object.defineProperty(globalThis, "sessionStorage", { value: store, configurable: true });
  return data;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  Object.defineProperty(globalThis, "sessionStorage", { value: undefined, configurable: true });
});

describe("Part 17 assistant client contracts", () => {
  it("keeps data values OFF by default and the key out of the JSON body", () => {
    const prefs: AIPrefs = { ...DEFAULT_PREFS, key: "SECRET_KEY" };
    const body = chatBody("Explain this", "explain", prefs, { panel_id: "S5" });
    expect(body.include_values).toBe(false);
    expect(body.panel_id).toBe("S5");
    expect(JSON.stringify(body)).not.toContain("SECRET_KEY");
    const preview = previewBody("Preview", "ask", prefs);
    expect(preview.mode).toBe("auto");
    expect(preview.acknowledge).toBe(false);
    expect(JSON.stringify(preview)).not.toContain("SECRET_KEY");
  });

  it("stores the key in sessionStorage only and restores safe defaults", () => {
    const data = mockSessionStorage();
    const saved = saveAIPrefs({ key: "TAB_ONLY_KEY", mode: "gemini", includeValues: true });
    expect(saved.key).toBe("TAB_ONLY_KEY");
    expect(Object.values(data).join(" ")).toContain("TAB_ONLY_KEY");
    const loaded = loadAIPrefs();
    expect(loaded).toMatchObject({
      key: "TAB_ONLY_KEY", mode: "gemini", includeValues: true,
      model: "flash", audience: "beginner", previewAck: false,
    });
    expect(DEFAULT_PREFS.includeValues).toBe(false);
    expect(DEFAULT_PREFS.previewAck).toBe(false);
  });

  it("parses SSE meta, chunks, resets and done events", () => {
    expect(parseSSEBlock("event: meta\ndata: {\"mode\":\"offline\"}")).toEqual({
      event: "meta", data: { mode: "offline" },
    });
    expect(parseSSEBlock("event: chunk\ndata: {\"text\":\"hello\"}")).toEqual({
      event: "chunk", data: { text: "hello" },
    });
    expect(parseSSEBlock("event: reset\ndata: {\"text\":\"\"}")).toEqual({
      event: "reset", data: { text: "" },
    });
    expect(parseSSEBlock("event: done\ndata: {\"text\":\"grounded\"}")).toEqual({
      event: "done", data: { text: "grounded" },
    });
    expect(parseSSEBlock("")).toBeNull();
  });

  it("formats the token/cost/queue counter", () => {
    expect(formatUsage(emptyUsage)).toBe("0/60 requests · 0 tokens · $0.000000 est.");
    expect(formatUsage({
      ...emptyUsage, requests: 2, prompt_tokens: 700, response_tokens: 30,
      estimated_cost_usd: 0.000285, queued: 1,
    })).toBe("2/60 requests · 730 tokens · $0.000285 est.");
  });

  it("labels offline, live and degraded modes", () => {
    const config = {
      live_ready: true, sdk_available: true, env_key_present: true,
    } as AIConfig;
    expect(assistantModeLabel("offline", config)).toContain("Offline mode");
    expect(assistantModeLabel("gemini", config)).toBe("Live Gemini");
    expect(assistantModeLabel("degraded", config)).toContain("offline answer");
  });

  it("shows a visible banner only when data values are sent", () => {
    expect(privacyBanner(DEFAULT_PREFS)).toBeNull();
    expect(privacyBanner({ ...DEFAULT_PREFS, includeValues: true })).toContain("Data values will be sent");
  });

  it("surfaces ungrounded numbers and missing citations", () => {
    const ok: AIGrounding = {
      grounded: true, citations: ["stat.adf"], ungrounded_numbers: [],
      missing_citations: false, fallback_used: false,
    };
    const bad: AIGrounding = {
      grounded: false, citations: [], ungrounded_numbers: ["987654.321"],
      missing_citations: true, fallback_used: false,
    };
    expect(groundingWarnings(ok)).toEqual([]);
    expect(groundingWarnings(bad)).toHaveLength(2);
    expect(groundingWarnings(bad)[0]).toContain("987654.321");
    expect(groundingWarnings(null)).toEqual([]);
  });
});


  it("explains exactly why live AI is or is not available", () => {
    const base = {
      offline_ready: true, sdk_available: true, env_key_present: false,
      live_ready: false, default_model: "gemini-2.5-flash", pro_model: "gemini-2.5-pro",
      temperature: 0.2, max_requests_per_session: 60, max_requests_per_minute: 15,
      min_interval_seconds: 4, max_values: 40, allow_data_values_by_default: false,
      input_usd_per_million: 0.3, output_usd_per_million: 2.5, privacy_note: "",
    } as AIConfig;
    expect(liveAvailability(null, DEFAULT_PREFS).ready).toBe(false);
    const noSdk = liveAvailability({ ...base, sdk_available: false }, DEFAULT_PREFS);
    expect(noSdk.reason).toBe("sdk");
    expect(noSdk.message).toContain("google-genai");
    const noKey = liveAvailability(base, DEFAULT_PREFS);
    expect(noKey.reason).toBe("key");
    expect(noKey.message).toContain("Offline mode still works");
    const ready = liveAvailability({ ...base, env_key_present: true, live_ready: true }, DEFAULT_PREFS);
    expect(ready.ready).toBe(true);
    expect(ready.reason).toBe("ready");
  });

it("loads live-AI capability before a session and never auto-opens live preview in Offline only", () => {
  // Regression: `/ai/config` is available without a session id. The old drawer
  // waited for sessionId, so the first-use preview stayed on "Checking…" forever.
  expect(shouldAutoOpenPreview(false, DEFAULT_PREFS)).toBe(false);
  expect(shouldAutoOpenPreview(true, DEFAULT_PREFS)).toBe(true);
  expect(shouldAutoOpenPreview(true, { ...DEFAULT_PREFS, mode: "offline" })).toBe(false);
  expect(shouldAutoOpenPreview(true, { ...DEFAULT_PREFS, previewAck: true })).toBe(false);
});

it("sends the typed key only as a header when checking live availability", async () => {
  const base = {
    offline_ready: true, sdk_available: true, env_key_present: false,
    live_ready: true, default_model: "gemini-2.5-flash", pro_model: "gemini-2.5-pro",
    temperature: 0.2, max_requests_per_session: 60, max_requests_per_minute: 15,
    min_interval_seconds: 4, max_values: 40, allow_data_values_by_default: false,
    input_usd_per_million: 0.3, output_usd_per_million: 2.5, privacy_note: "",
  } as AIConfig;
  const seen: Headers[] = [];
  vi.stubGlobal("fetch", vi.fn(async (_url: string, init: RequestInit) => {
    seen.push(new Headers(init.headers));
    return new Response(JSON.stringify(base), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }));
  const cfg = await aiApi.config({ ...DEFAULT_PREFS, key: "DUMMY_KEY" });
  expect(cfg.live_ready).toBe(true);
  expect(seen).toHaveLength(1);
  expect(seen[0].get("X-Gemini-Key")).toBe("DUMMY_KEY");
  expect(seen[0].get("Content-Type")).toBe("application/json");
  expect(seen[0].has("X-Session-Id")).toBe(false);
});

it("gives a definitive message instead of an endless checking state once config responds", () => {
  const base = {
    offline_ready: true, sdk_available: true, env_key_present: false,
    live_ready: false, default_model: "gemini-2.5-flash", pro_model: "gemini-2.5-pro",
    temperature: 0.2, max_requests_per_session: 60, max_requests_per_minute: 15,
    min_interval_seconds: 4, max_values: 40, allow_data_values_by_default: false,
    input_usd_per_million: 0.3, output_usd_per_million: 2.5, privacy_note: "",
  } as AIConfig;
  const result = liveAvailability(base, { ...DEFAULT_PREFS, key: "DUMMY_KEY" });
  expect(result.ready).toBe(false);
  expect(result.reason).toBe("key");
  expect(result.message).toContain("Offline mode still works");
  expect(result.message).not.toContain("Checking live AI availability");
});

