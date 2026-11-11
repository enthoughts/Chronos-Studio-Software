import { afterEach, describe, expect, it, vi } from "vitest";

import {
  activeStageAfterRefresh,
  clearSavedStage,
  isStageCode,
  readSavedStage,
  restoreStage,
  stageMemoryKey,
  writeSavedStage,
} from "./stageMemory";
import type { SessionState } from "./types";

function mockLocalStorage(): Record<string, string> {
  const data: Record<string, string> = {};
  const store = {
    getItem: (key: string) => data[key] ?? null,
    setItem: (key: string, value: string) => { data[key] = value; },
    removeItem: (key: string) => { delete data[key]; },
    clear: () => { Object.keys(data).forEach((k) => delete data[k]); },
    key: (i: number) => Object.keys(data)[i] ?? null,
    get length() { return Object.keys(data).length; },
  };
  Object.defineProperty(globalThis, "localStorage", { value: store, configurable: true });
  return data;
}

function session(hash: string, version: number): SessionState {
  return {
    dataset: { content_hash: hash },
    series_version: version,
  } as unknown as SessionState;
}

afterEach(() => {
  vi.restoreAllMocks();
  Object.defineProperty(globalThis, "localStorage", { value: undefined, configurable: true });
});

describe("stage memory", () => {
  it("keys saved stages by session, dataset and series version", () => {
    expect(stageMemoryKey(null, session("abc", 1))).toBeNull();
    expect(stageMemoryKey("s1", session("abc", 1))).toBe(
      "chronos.activeStage.s1:abc:1",
    );
    expect(stageMemoryKey("s1", session("abc", 1))).not.toBe(
      stageMemoryKey("s1", session("abc", 2)),
    );
    expect(stageMemoryKey("s1", session("abc", 2))).not.toBe(
      stageMemoryKey("s1", session("def", 2)),
    );
  });

  it("persists only valid stage codes and clears only one session", () => {
    const data = mockLocalStorage();
    const key = stageMemoryKey("s1", session("abc", 1));
    const otherKey = stageMemoryKey("s2", session("abc", 1));
    expect(key).not.toBeNull();
    expect(otherKey).not.toBeNull();
    writeSavedStage(key, "S6");
    writeSavedStage(otherKey, "S2");
    expect(readSavedStage(key)).toBe("S6");
    expect(isStageCode("S16")).toBe(false);
    expect(isStageCode("S6", ["S0", "S6"])).toBe(true);
    expect(isStageCode("S6", ["S0"])).toBe(false);
    data[key as string] = "not-a-stage";
    expect(readSavedStage(key)).toBeNull();
    clearSavedStage("s1");
    expect(data[key as string]).toBeUndefined();
    expect(data[otherKey as string]).toBe("S2");
  });

  it("restores the selected stage on reload and ignores backend stage on refresh", () => {
    expect(restoreStage("S2", "S3")).toBe("S2");
    expect(restoreStage(null, "S3")).toBe("S3");
    expect(restoreStage("bogus" as never, "S3")).toBe("S3");
    expect(activeStageAfterRefresh("S6", "S7")).toBe("S6");
    expect(activeStageAfterRefresh("S2", "S3")).toBe("S2");
  });
});
