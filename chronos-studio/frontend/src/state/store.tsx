/**
 * src/state/store.tsx — one React context, no state library.
 *
 * Part 0 keeps state minimal on purpose: session + pipeline + health. Later
 * parts add per-stage result caches behind the same context so the ribbon,
 * stepper and results pane never need refactoring.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ApiError, api, rememberSession } from "./api";
import { clearAIPrefs, type AssistantSeed } from "./ai";
import { errorToastPayload } from "../copy/errors";
import type {
  AnyResult,
  HealthResponse,
  SessionState,
  StageCode,
  StageDef,
} from "./types";
import {
  activeStageAfterRefresh,
  clearSavedStage,
  readSavedStage,
  restoreStage,
  stageMemoryKey,
  writeSavedStage,
} from "./stageMemory";

export interface Toast {
  id: number;
  kind: "error" | "info" | "success" | "warn";
  title: string;
  body?: string;
  fix?: string;
  /** Part 19: links an error toast to the plain-English error library. */
  errorCode?: string;
}

export interface HistoryState {
  can_undo: boolean;
  can_redo: boolean;
  series_version: number;
}

export const EMPTY_HISTORY: HistoryState = Object.freeze({
  can_undo: false,
  can_redo: false,
  series_version: 0,
});

interface StoreValue {
  loading: boolean;
  sessionId: string | null;
  session: SessionState | null;
  stages: StageDef[];
  health: HealthResponse | null;
  results: Record<string, AnyResult[]>;
  activeStage: StageCode;
  toasts: Toast[];
  assistantOpen: boolean;
  assistantSeed: AssistantSeed | null;
  aiLiveReady: boolean;
  /** Part 19: undo/redo availability from GET /preprocess/state. */
  history: HistoryState;
  historyBusy: boolean;
  refreshHistory: () => Promise<void>;
  undoHistory: () => Promise<void>;
  redoHistory: () => Promise<void>;
  setActiveStage: (s: StageCode) => void;
  openAssistant: (seed?: AssistantSeed | null) => void;
  closeAssistant: () => void;
  setAiLiveReady: (v: boolean) => void;
  startSession: () => Promise<void>;
  endSession: () => Promise<void>;
  refresh: () => Promise<void>;
  resetFrom: (stage: StageCode) => Promise<void>;
  dismissToast: (id: number) => void;
  pushToast: (t: Omit<Toast, "id">) => void;
}

const Ctx = createContext<StoreValue | null>(null);

export function StoreProvider({ children }: { children: ReactNode }) {
  const [loading, setLoading] = useState(true);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [session, setSession] = useState<SessionState | null>(null);
  const [stages, setStages] = useState<StageDef[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [results, setResults] = useState<Record<string, AnyResult[]>>({});
  const [activeStage, setActiveStageState] = useState<StageCode>("S0");
  const sessionRef = useRef<SessionState | null>(null);
  const activeStageRef = useRef<StageCode>("S0");
  useEffect(() => { sessionRef.current = session; }, [session]);
  useEffect(() => { activeStageRef.current = activeStage; }, [activeStage]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [assistantSeed, setAssistantSeed] = useState<AssistantSeed | null>(null);
  const [aiLiveReady, setAiLiveReady] = useState(false);
  const [history, setHistory] = useState<HistoryState>(EMPTY_HISTORY);
  const [historyBusy, setHistoryBusy] = useState(false);
  const historyBusyRef = useRef(false);
  const toastSeq = useRef(0);
  const bootstrapped = useRef(false);
  // A ref mirror of sessionId: callbacks that must not be re-created (refresh,
  // startSession) read this, so they never act on a stale id.
  const sessionIdRef = useRef<string | null>(null);
  const setId = useCallback((id: string | null) => {
    sessionIdRef.current = id;
    setSessionId(id);
  }, []);

  const setActiveStage = useCallback((s: StageCode) => {
    activeStageRef.current = s;
    setActiveStageState(s);
    writeSavedStage(stageMemoryKey(sessionIdRef.current, sessionRef.current), s);
  }, []);

  const pushToast = useCallback((t: Omit<Toast, "id">) => {
    const id = ++toastSeq.current;
    setToasts((prev) => [...prev, { ...t, id }]);
    // Errors need the user's attention; info/success fade out.
    const ttl = t.kind === "error" ? 12000 : 5000;
    window.setTimeout(() => setToasts((prev) => prev.filter((x) => x.id !== id)), ttl);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const asToast = useCallback(
    (e: unknown) => {
      // Part 19: one plain-English library behind every error channel.
      pushToast(errorToastPayload(e));
    },
    [pushToast],
  );

  const refreshHistory = useCallback(async () => {
    if (!sessionIdRef.current) {
      setHistory(EMPTY_HISTORY);
      return;
    }
    try {
      const s = await api.preprocessState();
      setHistory({
        can_undo: Boolean(s.can_undo),
        can_redo: Boolean(s.can_redo),
        series_version: s.series_version ?? 0,
      });
    } catch {
      // A session with no dataset has no preprocessing state yet.
      setHistory(EMPTY_HISTORY);
    }
  }, []);

  const refresh = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    try {
      const [state, res] = await Promise.all([api.getSession(sid), api.results(sid)]);
      setSession(state);
      setResults(res);
      // A background refresh must not adopt the backend's current_stage.
      // The explicit guard keeps S2/S6 where the user put them.
      setActiveStageState(
        activeStageAfterRefresh(activeStageRef.current, state.pipeline.current_stage),
      );
      void refreshHistory();
    } catch (e) {
      asToast(e);
      if (e instanceof ApiError && e.code === "session.not_found") {
        clearSavedStage(sid);
        rememberSession(null);
        setId(null);
        setSession(null);
        setHistory(EMPTY_HISTORY);
      }
    }
  }, [asToast, setId, refreshHistory]);

  const applyHistoryStep = useCallback(
    async (direction: "undo" | "redo") => {
      if (historyBusyRef.current || !sessionIdRef.current) return;
      historyBusyRef.current = true;
      setHistoryBusy(true);
      try {
        if (direction === "undo") await api.undoOp();
        else await api.redoOp();
        await refresh();
        pushToast({
          kind: "success",
          title: direction === "undo" ? "Treatment undone." : "Treatment redone.",
          body: "The prepared series has a new version; downstream results are marked stale.",
        });
      } catch (e) {
        pushToast(errorToastPayload(e, `That treatment could not be ${direction === "undo" ? "undone" : "redone"}.`));
      } finally {
        historyBusyRef.current = false;
        setHistoryBusy(false);
      }
    },
    [refresh, pushToast],
  );

  const undoHistory = useCallback(() => applyHistoryStep("undo"), [applyHistoryStep]);
  const redoHistory = useCallback(() => applyHistoryStep("redo"), [applyHistoryStep]);

  const startSession = useCallback(async () => {
    setLoading(true);
    try {
      clearSavedStage(sessionIdRef.current);
      const created = await api.createSession();
      rememberSession(created.session_id);
      setId(created.session_id);
      const [state, res] = await Promise.all([
        api.getSession(created.session_id),
        api.results(created.session_id),
      ]);
      setSession(state);
      setResults(res);
      const initial = state.pipeline.current_stage as StageCode;
      activeStageRef.current = initial;
      setActiveStageState(initial);
      writeSavedStage(stageMemoryKey(created.session_id, state), initial);
      void refreshHistory();
      pushToast({ kind: "success", title: "New analysis started." });
    } catch (e) {
      asToast(e);
    } finally {
      setLoading(false);
    }
  }, [asToast, pushToast, setId, refreshHistory]);

  const openAssistant = useCallback((seed: AssistantSeed | null = null) => {
    setAssistantSeed(seed);
    setAssistantOpen(true);
  }, []);

  const closeAssistant = useCallback(() => {
    setAssistantOpen(false);
    setAssistantSeed(null);
  }, []);

  const endSession = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;
    try {
      await api.deleteSession(sid);
    } catch {
      /* deleting a dead session is fine */
    }
    clearSavedStage(sid);
    clearAIPrefs();
    activeStageRef.current = "S0";
    setActiveStageState("S0");
    rememberSession(null);
    setId(null);
    setSession(null);
    setResults({});
    setHistory(EMPTY_HISTORY);
    setAssistantOpen(false);
    setAssistantSeed(null);
    setAiLiveReady(false);
    pushToast({ kind: "info", title: "Session deleted from the server." });
  }, [pushToast, setId]);

  const resetFrom = useCallback(
    async (stage: StageCode) => {
      const sid = sessionIdRef.current;
      if (!sid) return;
      try {
        await api.reset(sid, stage);
        await refresh();
        pushToast({
          kind: "info",
          title: `Pipeline reset from ${stage}.`,
          body: "Downstream results are now marked stale.",
        });
      } catch (e) {
        asToast(e);
      }
    },
    [refresh, pushToast, asToast],
  );

  /* bootstrap once */
  useEffect(() => {
    if (bootstrapped.current) return;
    bootstrapped.current = true;
    (async () => {
      try {
        const [h, map] = await Promise.all([api.health(), api.pipelineMap()]);
        setHealth(h);
        setStages(map.stages);
        const existing = localStorage.getItem("chronos.session");
        if (existing) {
          try {
            const [state, res] = await Promise.all([
              api.getSession(existing),
              api.results(existing),
            ]);
            setId(existing);
            setSession(state);
            setResults(res);
            const allowed = map.stages.map((s) => s.code as StageCode);
            const key = stageMemoryKey(existing, state);
            const restored = restoreStage(
              readSavedStage(key, allowed),
              state.pipeline.current_stage as StageCode,
              allowed,
            );
            activeStageRef.current = restored;
            setActiveStageState(restored);
            writeSavedStage(key, restored);
            void refreshHistory();
          } catch {
            clearSavedStage(existing);
            activeStageRef.current = "S0";
            setActiveStageState("S0");
            rememberSession(null);
          }
        }
      } catch (e) {
        asToast(e);
      } finally {
        setLoading(false);
      }
    })();
  }, [asToast, refreshHistory]);

  const value = useMemo<StoreValue>(
    () => ({
      loading,
      sessionId,
      session,
      stages,
      health,
      results,
      activeStage,
      toasts,
      assistantOpen,
      assistantSeed,
      aiLiveReady,
      history,
      historyBusy,
      refreshHistory,
      undoHistory,
      redoHistory,
      setActiveStage,
      openAssistant,
      closeAssistant,
      setAiLiveReady,
      startSession,
      endSession,
      refresh,
      resetFrom,
      dismissToast,
      pushToast,
    }),
    [
      loading, sessionId, session, stages, health, results, activeStage, toasts,
      assistantOpen, assistantSeed, aiLiveReady, history, historyBusy,
      startSession, endSession, refresh, resetFrom, dismissToast, pushToast,
      openAssistant, closeAssistant,
      refreshHistory, undoHistory, redoHistory,
    ],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useStore(): StoreValue {
  const v = useContext(Ctx);
  if (!v) throw new Error("useStore must be used inside <StoreProvider>");
  return v;
}
