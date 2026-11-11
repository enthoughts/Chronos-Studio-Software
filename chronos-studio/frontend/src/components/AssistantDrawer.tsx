/**
 * AssistantDrawer.tsx — Part 17's grounded, offline-first assistant UI.
 *
 * Privacy rules: the key is kept in sessionStorage and sent only as a request
 * header; the payload preview shows the exact non-secret digest; data values
 * default OFF; every suggested action is a button the user must click.
 */

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import { COPY } from "../copy/strings";
import { ApiError } from "../state/api";
import {
  aiApi,
  chatBody,
  emptyUsage,
  formatUsage,
  groundingWarnings,
  assistantModeLabel,
  liveAvailability,
  loadAIPrefs,
  shouldAutoOpenPreview,
  previewBody,
  privacyBanner,
  saveAIPrefs,
  type AIAction,
  type AIActionName,
  type AIChunk,
  type AIConfig,
  type AIDone,
  type AIErrorEvent,
  type AIGrounding,
  type AIHistoryMessage,
  type AIMeta,
  type LiveAvailability,
  type AIMode,
  type AIPayloadPreview,
  type AIPrefs,
  type AIStreamEvent,
  type AIUsage,
} from "../state/ai";
import { useStore } from "../state/store";
import type { StageCode } from "../state/types";

const DRAFT_LABEL = "AI-drafted, verify before submission";
const IMAGE_MAX = 4 * 1024 * 1024;

const DEFAULT_QUESTION: Record<AIActionName, string> = {
  ask: "What have the computed results shown so far?",
  explain: "Explain the focused result in plain language.",
  interpret_plot: "Interpret this plot from its figure-data payload.",
  what_next: "What should I do next?",
  write_up: "Write it up from the computed results.",
  simplify: "Simplify the focused explanation.",
  audience: "Rewrite that answer for my audience.",
  eli15: "Explain it like I'm 15.",
};

interface SendOptions {
  resultId?: string | null;
  panelId?: string;
  imageDataUrl?: string | null;
}

interface AttachedImage {
  name: string;
  dataUrl: string;
  bytes: number;
}

function modeClass(mode: AIMode): string {
  return mode === "gemini" ? "pill-ok" : mode === "degraded" ? "pill-warn" : "";
}

export function AssistantDrawer() {
  const {
    assistantOpen,
    assistantSeed,
    closeAssistant,
    openAssistant,
    sessionId,
    activeStage,
    setActiveStage,
    setAiLiveReady,
    startSession,
    endSession,
    pushToast,
    loading,
  } = useStore();

  const [prefs, setPrefs] = useState<AIPrefs>(() => loadAIPrefs());
  const [config, setConfig] = useState<AIConfig | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [history, setHistory] = useState<AIHistoryMessage[]>([]);
  const [usage, setUsage] = useState<AIUsage>(emptyUsage);
  const [question, setQuestion] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [currentText, setCurrentText] = useState("");
  const [currentMode, setCurrentMode] = useState<AIMode>("offline");
  const [currentGrounding, setCurrentGrounding] = useState<AIGrounding | null>(null);
  const [actions, setActions] = useState<AIAction[]>([]);
  const [draft, setDraft] = useState(false);
  const [streamError, setStreamError] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  // Let the effect below decide: no modal for Offline only, and no modal before
  // the drawer is actually opened.
  const [showPreview, setShowPreview] = useState(false);
  const [preview, setPreview] = useState<AIPayloadPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [image, setImage] = useState<AttachedImage | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const autoSentRef = useRef<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const updatePrefs = useCallback((patch: Partial<AIPrefs>) => {
    setPrefs(saveAIPrefs(patch));
  }, []);

  const toastError = useCallback(
    (e: unknown) => {
      if (e instanceof ApiError) {
        pushToast({ kind: "error", title: e.message, body: e.code, fix: e.fix });
      } else {
        pushToast({
          kind: "error",
          title: "Something unexpected went wrong.",
          fix: "Reload the page. Your analysis is safe.",
        });
      }
    },
    [pushToast],
  );

  const loadConfig = useCallback(async () => {
    setConfigError(null);
    try {
      const cfg = await aiApi.config(prefs);
      setConfig(cfg);
      setAiLiveReady(cfg.live_ready);
    } catch (e) {
      setAiLiveReady(false);
      setConfigError(
        "Could not check Live AI availability. Offline mode still works. Check that the backend is running, then reload the page.",
      );
      toastError(e);
    }
  }, [prefs, setAiLiveReady, toastError]);

  const loadHistory = useCallback(async () => {
    if (!sessionId) return;
    try {
      const h = await aiApi.history(prefs, sessionId);
      setHistory(h.messages);
      setUsage(h.usage);
      if (h.preview_acknowledged && !prefs.previewAck) {
        setPrefs(saveAIPrefs({ previewAck: true }));
      }
    } catch (e) {
      if (!(e instanceof ApiError && e.code === "session.not_found")) toastError(e);
    }
  }, [sessionId, prefs, toastError]);

  const fetchPreview = useCallback(
    async (acknowledge = false, overridePrefs?: AIPrefs): Promise<AIPayloadPreview | null> => {
      if (!sessionId) return null;
      const p = overridePrefs ?? prefs;
      const body = previewBody(COPY.assistant.previewTitle, assistantSeed?.action ?? "ask", p, {
        result_id: assistantSeed?.resultId ?? null,
        panel_id: assistantSeed?.panelId ?? activeStage,
        image_data_url: image?.dataUrl ?? null,
        acknowledge,
      });
      setPreviewLoading(true);
      try {
        const result = await aiApi.preview(body, p, sessionId);
        setPreview(result);
        return result;
      } catch (e) {
        toastError(e);
        return null;
      } finally {
        setPreviewLoading(false);
      }
    },
    [sessionId, prefs, assistantSeed, activeStage, image, toastError],
  );

  const reviewPayload = useCallback(
    async (patch: Partial<AIPrefs>) => {
      const next = saveAIPrefs({ ...patch, previewAck: true });
      setPrefs(next);
      await fetchPreview(true, next);
      setShowPreview(false);
    },
    [fetchPreview],
  );

  const send = useCallback(
    async (action: AIActionName, rawQuestion: string, opts: SendOptions = {}) => {
      if (!sessionId || streaming) return;
      if (usage.remaining <= 0) {
        pushToast({
          kind: "warn",
          title: "This analysis reached its assistant request cap.",
          fix: "Start a new analysis, or keep using the computed results.",
        });
        return;
      }
      if (!prefs.previewAck && prefs.mode !== "offline") {
        setShowPreview(true);
        return;
      }
      const q = rawQuestion.trim() || DEFAULT_QUESTION[action];
      const panelId = opts.panelId ?? assistantSeed?.panelId ?? activeStage;
      const resultId = opts.resultId ?? assistantSeed?.resultId ?? null;
      const controller = new AbortController();
      abortRef.current = controller;
      setStreaming(true);
      setCurrentText("");
      setCurrentMode("offline");
      setCurrentGrounding(null);
      setActions([]);
      setDraft(false);
      setStreamError(null);

      const optimistic: AIHistoryMessage = {
        role: "user",
        content: q,
        action,
        panel_id: panelId,
        citations: [],
        grounded: true,
        ungrounded_numbers: [],
        mode: "offline",
        prompt_tokens: 0,
        response_tokens: 0,
        timestamp: new Date().toISOString(),
      };
      setHistory((prev) => [...prev, optimistic]);

      const body = chatBody(q, action, prefs, {
        panel_id: panelId,
        result_id: resultId,
        image_data_url: opts.imageDataUrl ?? image?.dataUrl ?? null,
        eli15: action === "eli15" || prefs.eli15,
      });
      const onEvent = (evt: AIStreamEvent) => {
        if (evt.event === "meta") {
          const meta = evt.data as AIMeta;
          setCurrentMode(meta.mode);
          setUsage(meta.usage);
        } else if (evt.event === "chunk") {
          const chunk = evt.data as AIChunk;
          setCurrentText((t) => t + chunk.text);
        } else if (evt.event === "reset") {
          setCurrentText("");
        } else if (evt.event === "error") {
          const err = evt.data as AIErrorEvent;
          setStreamError(err.plain);
        } else if (evt.event === "done") {
          const done = evt.data as AIDone;
          setCurrentText(done.text);
          setCurrentMode(done.mode);
          setCurrentGrounding(done.grounding);
          setActions(done.actions);
          setUsage(done.usage);
          setDraft(done.ai_drafted);
        }
      };
      try {
        await aiApi.stream(body, prefs, onEvent, controller.signal, sessionId);
        await loadHistory();
      } catch (e) {
        if (e instanceof ApiError && e.code === "ai.cancelled") {
          pushToast({ kind: "info", title: COPY.assistant.stopped });
        } else if (e instanceof ApiError && e.code === "ai.preview_required") {
          setShowPreview(true);
        } else {
          toastError(e);
        }
        await loadHistory();
      } finally {
        setStreaming(false);
        setCurrentText("");
        setQuestion("");
        abortRef.current = null;
      }
    },
    [
      sessionId, streaming, usage.remaining, prefs, assistantSeed, activeStage,
      image, loadHistory, pushToast, toastError,
    ],
  );

  // Capability checking must not wait for an analysis session: the first-use
  // preview is shown before any dataset is loaded.
  useEffect(() => {
    if (!assistantOpen) return;
    void loadConfig();
  }, [assistantOpen, loadConfig]);

  useEffect(() => {
    if (!assistantOpen || !sessionId) return;
    void loadHistory();
  }, [assistantOpen, sessionId, loadHistory]);

  useEffect(() => {
    if (shouldAutoOpenPreview(assistantOpen, prefs)) setShowPreview(true);
    if (!assistantOpen) autoSentRef.current = null;
  }, [assistantOpen, prefs.previewAck, prefs.mode]);

  useEffect(() => {
    if (!assistantOpen || !showPreview || !sessionId) return;
    void fetchPreview(false);
  }, [assistantOpen, showPreview, sessionId, fetchPreview]);

  useEffect(() => {
    if (!assistantOpen || !assistantSeed || !sessionId || showPreview || streaming) return;
    const seedQuestion = assistantSeed.question?.trim();
    if (!seedQuestion) return;
    const token = [
      assistantSeed.action, assistantSeed.resultId ?? "",
      assistantSeed.panelId ?? "", seedQuestion,
    ].join("|");
    if (autoSentRef.current === token) return;
    autoSentRef.current = token;
    setQuestion(seedQuestion);
    void send(assistantSeed.action, seedQuestion, {
      resultId: assistantSeed.resultId ?? null,
      panelId: assistantSeed.panelId ?? activeStage,
    });
  }, [assistantOpen, assistantSeed, sessionId, showPreview, streaming, activeStage, send]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ block: "end" });
  }, [history.length, currentText, streaming]);

  const onImage = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (file.size > IMAGE_MAX) {
      pushToast({
        kind: "error",
        title: "That plot image is too large.",
        fix: "Attach a smaller PNG, JPEG or WebP screenshot (4 MB maximum).",
      });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        setImage({ name: file.name, dataUrl: reader.result, bytes: file.size });
      }
    };
    reader.readAsDataURL(file);
  };

  const quick = (action: AIActionName) => {
    void send(action, DEFAULT_QUESTION[action], {
      resultId: assistantSeed?.resultId ?? null,
      panelId: assistantSeed?.panelId ?? activeStage,
      imageDataUrl: image?.dataUrl ?? null,
    });
  };

  const confirmAction = (action: AIAction) => {
    if (action.kind === "explain_result" && action.result_id) {
      openAssistant({
        action: "explain",
        resultId: action.result_id,
        panelId: action.stage,
        question: `Explain ${action.label}.`,
      });
      return;
    }
    setActiveStage(action.stage as StageCode);
    closeAssistant();
  };

  if (!assistantOpen) return null;

  const banner = privacyBanner(prefs);
  const warnings = groundingWarnings(currentGrounding);
  const modeLabel = assistantModeLabel(currentMode, config);
  const live: LiveAvailability = configError
    ? {
        ready: false,
        reason: "key",
        title: "Live AI status unavailable",
        message: configError,
      }
    : liveAvailability(config, prefs);
  const livePossible = live.ready;

  return (
    <>
      <div
        className="assistant-backdrop"
        onClick={closeAssistant}
        aria-hidden="true"
        data-testid="assistant-backdrop"
      />
      <aside
        className="assistant-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={COPY.assistant.title}
        data-testid="assistant-drawer"
      >
        <header className="assistant-header">
          <div>
            <div className="pane-title">{COPY.assistant.eyebrow}</div>
            <h2>{COPY.assistant.title}</h2>
          </div>
          <span className={`pill ${modeClass(currentMode)}`} data-testid="assistant-mode">
            {modeLabel}
          </span>
          <button className="btn btn-sm btn-ghost" onClick={closeAssistant} aria-label={COPY.assistant.close}>
            ✕
          </button>
        </header>

        <div className="assistant-toolbar">
          <button className="btn btn-sm" onClick={() => setShowSettings((v) => !v)} data-testid="assistant-settings">
            ⚙ {COPY.assistant.settings}
          </button>
          <button className="btn btn-sm" onClick={() => setShowPreview(true)} data-testid="assistant-preview">
            ◫ {COPY.assistant.preview}
          </button>
          <button
            className="btn btn-sm"
            onClick={() => void reviewPayload({ mode: "offline" })}
            data-testid="assistant-offline"
          >
            ✦ {COPY.assistant.offline}
          </button>
          <span className="assistant-usage" data-testid="assistant-usage" title={COPY.assistant.usage}>
            {formatUsage(usage)}
            {usage.queued > 0 ? ` · ${usage.queued} ${COPY.assistant.queued}` : ""}
          </span>
        </div>

        {banner && (
          <div className="assistant-banner" role="status" data-testid="assistant-banner">
            <strong>{COPY.assistant.valuesToggle} ON.</strong> {banner}
          </div>
        )}
        {currentMode === "degraded" && (
          <div className="callout callout-warn" role="status">{COPY.assistant.degraded}</div>
        )}
        {streamError && <div className="callout callout-bad" role="status">{streamError}</div>}

        {showSettings && (
          <section className="assistant-settings" aria-label={COPY.assistant.settings}>
            <label className="assistant-field">
              <span>{COPY.assistant.keyLabel}</span>
              <input
                type="password"
                autoComplete="off"
                value={prefs.key}
                placeholder="AIza…"
                data-testid="assistant-key"
                onChange={(e) => updatePrefs({ key: e.target.value })}
              />
              <small>{COPY.assistant.keyHint}</small>
            </label>
            <div className="assistant-field-row">
              <label className="assistant-field">
                <span>{COPY.assistant.modeLabel}</span>
                <select
                  value={prefs.mode}
                  onChange={(e) => updatePrefs({ mode: e.target.value as AIPrefs["mode"] })}
                >
                  <option value="auto">{COPY.assistant.auto}</option>
                  <option value="gemini">{COPY.assistant.gemini}</option>
                  <option value="offline">{COPY.assistant.offlineMode}</option>
                </select>
              </label>
              <label className="assistant-field">
                <span>{COPY.assistant.modelLabel}</span>
                <select
                  value={prefs.model}
                  onChange={(e) => updatePrefs({ model: e.target.value as AIPrefs["model"] })}
                >
                  <option value="flash">{COPY.assistant.flash}</option>
                  <option value="pro">{COPY.assistant.pro}</option>
                </select>
              </label>
              <label className="assistant-field">
                <span>{COPY.assistant.audience}</span>
                <select
                  value={prefs.audience}
                  onChange={(e) => updatePrefs({ audience: e.target.value as AIPrefs["audience"] })}
                >
                  <option value="beginner">{COPY.assistant.beginner}</option>
                  <option value="peer">{COPY.assistant.peer}</option>
                  <option value="reviewer">{COPY.assistant.reviewer}</option>
                </select>
              </label>
            </div>
            <label className="assistant-check">
              <input
                type="checkbox"
                checked={prefs.includeValues}
                data-testid="assistant-values-toggle"
                onChange={(e) => updatePrefs({ includeValues: e.target.checked })}
              />
              {COPY.assistant.valuesToggle}
              <small>{prefs.includeValues ? COPY.assistant.valuesBanner : COPY.assistant.valuesOff}</small>
            </label>
            <label className="assistant-check">
              <input
                type="checkbox"
                checked={prefs.eli15}
                onChange={(e) => updatePrefs({ eli15: e.target.checked })}
              />
              {COPY.assistant.eli15}
            </label>
            <div className="assistant-privacy">{COPY.assistant.privacyNote}</div>
            <div className="assistant-settings-actions">
              <button
                className="btn btn-sm"
                onClick={async () => {
                  try {
                    await aiApi.clearHistory(prefs, sessionId);
                    await loadHistory();
                  } catch (e) {
                    toastError(e);
                  }
                }}
              >
                {COPY.assistant.clear}
              </button>
              <button
                className="btn btn-sm"
                onClick={() => {
                  if (window.confirm(COPY.assistant.deleteHint)) void endSession();
                }}
                data-testid="assistant-delete-session"
              >
                {COPY.assistant.deleteSession}
              </button>
            </div>
            {config && (
              <>
                <div className="hint" role="status" data-testid="assistant-live-status">
                  <strong>{live.title}.</strong> {live.message}
                </div>
                <div className="assistant-config">
                  SDK {config.sdk_available ? "installed" : "not installed"} · env key{" "}
                  {config.env_key_present ? "present" : "absent"} · live{" "}
                  {config.live_ready ? "ready" : "not ready"} · temperature {config.temperature} · cap{" "}
                  {config.max_requests_per_session}
                </div>
              </>
            )}
          </section>
        )}

        <div className="assistant-quick" aria-label={COPY.assistant.quick}>
          <button className="btn btn-sm" onClick={() => quick("explain")}>{COPY.assistant.explain}</button>
          <button className="btn btn-sm" onClick={() => quick("interpret_plot")}>{COPY.assistant.interpret}</button>
          <button className="btn btn-sm" onClick={() => quick("what_next")}>{COPY.assistant.whatNext}</button>
          <button className="btn btn-sm" onClick={() => quick("write_up")}>{COPY.assistant.writeUp}</button>
          <button className="btn btn-sm" onClick={() => quick("simplify")}>{COPY.assistant.simplify}</button>
          <button className="btn btn-sm" onClick={() => quick("audience")}>{COPY.assistant.rewrite}</button>
          <button className="btn btn-sm" onClick={() => quick("eli15")}>{COPY.assistant.eli15}</button>
        </div>

        <div className="assistant-messages" aria-live="polite" data-testid="assistant-messages">
          {!sessionId ? (
            <div className="assistant-empty">
              <p>{COPY.assistant.noSession}</p>
              <button className="btn btn-sm btn-primary" disabled={loading} onClick={() => void startSession()}>
                {COPY.newSession}
              </button>
            </div>
          ) : history.length === 0 && !streaming ? (
            <div className="assistant-empty">{COPY.assistant.empty}</div>
          ) : (
            <>
              {history.map((m, i) => (
                <article
                  key={`${m.timestamp}-${i}`}
                  className={`assistant-msg ${m.role === "user" ? "assistant-msg-user" : "assistant-msg-ai"}`}
                >
                  <div className="assistant-msg-role">{m.role === "user" ? "You" : "Assistant"}</div>
                  <div className="assistant-msg-text">{m.content}</div>
                  {m.role === "assistant" && (
                    <div className="assistant-msg-foot">
                      {m.content.includes(DRAFT_LABEL) && (
                        <span className="pill pill-warn" data-testid="assistant-draft">
                          {COPY.assistant.draftLabel}
                        </span>
                      )}
                      {m.citations.length > 0 && (
                        <span className="assistant-citations" title={COPY.assistant.grounded}>
                          {m.citations.map((c) => <span key={c}>[{c}]</span>)}
                        </span>
                      )}
                      {m.ungrounded_numbers.length > 0 && (
                        <span className="assistant-warning">{COPY.assistant.ungrounded}: {m.ungrounded_numbers.join(", ")}</span>
                      )}
                      {!m.grounded && m.ungrounded_numbers.length === 0 && (
                        <span className="assistant-warning">{COPY.assistant.missingCitations}</span>
                      )}
                    </div>
                  )}
                </article>
              ))}
              {streaming && (
                <article className="assistant-msg assistant-msg-ai assistant-streaming">
                  <div className="assistant-msg-role">Assistant · {modeLabel}</div>
                  <div className="assistant-msg-text" data-testid="assistant-stream">
                    {currentText || "…"}
                  </div>
                  {warnings.length > 0 && (
                    <div className="assistant-warning" data-testid="assistant-grounding-warning">
                      {warnings.join(" · ")}
                    </div>
                  )}
                  {draft && <span className="pill pill-warn">{COPY.assistant.draftLabel}</span>}
                </article>
              )}
              {actions.length > 0 && (
                <section className="assistant-actions" aria-label={COPY.assistant.actions}>
                  <div className="pane-title">{COPY.assistant.actions}</div>
                  {actions.map((a, i) => (
                    <button
                      key={`${a.stage}-${a.rule_id}-${i}`}
                      className="btn btn-sm"
                      data-testid="assistant-action"
                      onClick={() => confirmAction(a)}
                    >
                      {COPY.assistant.confirmAction}: {a.label}
                    </button>
                  ))}
                </section>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        <div className="assistant-composer">
          {image && (
            <div className="assistant-image-chip">
              ◱ {image.name} · {Math.round(image.bytes / 1024)} KB
              <button className="btn btn-sm btn-ghost" onClick={() => setImage(null)}>
                {COPY.assistant.removeImage}
              </button>
            </div>
          )}
          <label className="assistant-attach" title={COPY.assistant.imageHint}>
            <input type="file" accept="image/png,image/jpeg,image/webp" onChange={onImage} data-testid="assistant-image" />
            {COPY.assistant.attachImage}
          </label>
          <textarea
            value={question}
            rows={2}
            placeholder={COPY.assistant.askPlaceholder}
            data-testid="assistant-message-input"
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send("ask", question);
              }
            }}
          />
          {streaming ? (
            <button
              className="btn btn-sm btn-primary"
              data-testid="assistant-stop"
              onClick={() => abortRef.current?.abort()}
            >
              ■ {COPY.assistant.stop}
            </button>
          ) : (
            <button
              className="btn btn-sm btn-primary"
              data-testid="assistant-send"
              disabled={!sessionId || usage.remaining <= 0}
              onClick={() => void send("ask", question)}
            >
              ➤ {COPY.assistant.send}
            </button>
          )}
        </div>
      </aside>

      {showPreview && (
        <div className="assistant-modal" role="dialog" aria-modal="true" aria-label={COPY.assistant.previewTitle}>
          <div className="assistant-modal-card">
            <div className="pane-title">{COPY.assistant.eyebrow}</div>
            <h2>{COPY.assistant.previewTitle}</h2>
            <p>{COPY.assistant.previewIntro}</p>
            <div className="hint" role="status" data-testid="assistant-preview-live-status">{live.message}</div>
            {previewLoading && <div className="hint">Building the compact digest…</div>}
            {preview && (
              <>
                <div className="assistant-preview-facts">
                  <span className="pill">mode {preview.mode}</span>
                  <span className="pill">model {preview.model ?? "offline"}</span>
                  <span className="pill">~{preview.estimated_prompt_tokens} prompt tokens</span>
                  <span className="pill pill-ok">no key in payload</span>
                  {preview.data_values_included && <span className="pill pill-warn">data values included</span>}
                  {preview.image_attached && <span className="pill">image {preview.image_bytes} bytes</span>}
                </div>
                {preview.data_values_included && (
                  <div className="assistant-banner" role="status">{COPY.assistant.valuesBanner}</div>
                )}
                <details open>
                  <summary>AnalysisDigest JSON</summary>
                  <pre data-testid="assistant-preview-digest">{JSON.stringify(preview.digest, null, 2)}</pre>
                </details>
                <details>
                  <summary>System prompt</summary>
                  <pre>{preview.system_prompt}</pre>
                </details>
                <details>
                  <summary>Request contents</summary>
                  <pre>{JSON.stringify(preview.contents, null, 2)}</pre>
                </details>
                <div className="assistant-privacy">{preview.privacy_note}</div>
              </>
            )}
            <div className="assistant-modal-actions">
              <button
                className="btn btn-sm"
                onClick={() => void reviewPayload({ mode: "offline" })}
                data-testid="assistant-preview-offline"
              >
                {COPY.assistant.previewOffline}
              </button>
              <button
                className="btn btn-sm btn-primary"
                disabled={!livePossible}
                title={livePossible ? COPY.assistant.previewConfirm : live.message}
                onClick={() => void reviewPayload({ mode: "auto" })}
                data-testid="assistant-preview-confirm"
              >
                {COPY.assistant.previewConfirm}
              </button>
              <button
                className="btn btn-sm"
                onClick={() => { setShowPreview(false); setShowSettings(true); }}
                data-testid="assistant-preview-settings"
              >
                {COPY.assistant.openSettings}
              </button>
              <button className="btn btn-sm btn-ghost" onClick={closeAssistant}>{COPY.assistant.cancel}</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
