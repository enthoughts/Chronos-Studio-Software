/**
 * stageMemory.ts — keep the stage the user selected across a page reload.
 *
 * The backend's `current_stage` is the last stage *completed*. It is the right
 * fallback for a brand-new session, but it is not the stage the user was
 * reading. Persisting the selection per session/dataset/version prevents a
 * reload from silently moving S2 to S3 or S6 to S7.
 */

import type { SessionState, StageCode } from "./types";

const PREFIX = "chronos.activeStage.";

export const STAGE_CODES: readonly StageCode[] = [
  "S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7",
  "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15",
];

export function isStageCode(
  value: unknown,
  allowed: readonly StageCode[] = STAGE_CODES,
): value is StageCode {
  return typeof value === "string" && (allowed as readonly string[]).includes(value);
}

function getStorage(): Storage | null {
  try {
    return (globalThis as { localStorage?: Storage }).localStorage ?? null;
  } catch {
    return null;
  }
}

export function stageMemoryKey(
  sessionId: string | null | undefined,
  session: SessionState | null | undefined,
): string | null {
  if (!sessionId) return null;
  const dataset = session?.dataset?.content_hash ?? "no-dataset";
  const version = session?.series_version ?? 1;
  return `${PREFIX}${sessionId}:${dataset}:${version}`;
}

export function readSavedStage(
  key: string | null | undefined,
  allowed: readonly StageCode[] = STAGE_CODES,
): StageCode | null {
  if (!key) return null;
  const storage = getStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(key);
    return isStageCode(raw, allowed) ? raw : null;
  } catch {
    return null;
  }
}

export function writeSavedStage(
  key: string | null | undefined,
  stage: StageCode,
): void {
  if (!key || !isStageCode(stage)) return;
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.setItem(key, stage);
  } catch {
    /* private browsing or quota — keep the in-memory stage */
  }
}

export function clearSavedStage(sessionId: string | null | undefined): void {
  if (!sessionId) return;
  const storage = getStorage();
  if (!storage) return;
  const prefix = `${PREFIX}${sessionId}:`;
  try {
    for (let i = storage.length - 1; i >= 0; i -= 1) {
      const key = storage.key(i);
      if (key && key.startsWith(prefix)) storage.removeItem(key);
    }
  } catch {
    /* storage unavailable — nothing to clear */
  }
}

export function restoreStage(
  saved: StageCode | null,
  backendStage: StageCode,
  allowed: readonly StageCode[] = STAGE_CODES,
): StageCode {
  return isStageCode(saved, allowed) ? saved : backendStage;
}

/**
 * A background refresh must never adopt the backend's `current_stage`.
 * `backendStage` is accepted so the policy is explicit and testable.
 */
export function activeStageAfterRefresh(
  current: StageCode,
  backendStage: StageCode,
): StageCode {
  void backendStage;
  return current;
}
