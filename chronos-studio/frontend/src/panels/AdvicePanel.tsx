/**
 * panels/AdvicePanel.tsx — S2/S3: the preprocessing advisor and its apply/undo engine.
 *
 * Design intent (§5.3, "suggest, don't impose"):
 *
 *  - S2 runs automatically on open, like S1. The user is never asked to "ask for advice".
 *  - Every card shows the *evidence sentence* the rule engine produced, its confidence,
 *    the alternative, and what ignoring it costs. A recommendation without those four
 *    things is a command, not advice.
 *  - Nothing is applied by opening the panel. "Preview" runs the real operation on a
 *    copy and shows the result; "Apply" commits a version; "Undo" steps back.
 *  - The applied-steps strip is always visible, because a user who cannot see what has
 *    been done to their data cannot reason about the results.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IconCheck, IconRefresh, IconWarning } from "../components/Icons";
import { ResultCard } from "../components/ResultCard";
import { api } from "../state/api";
import { useStore } from "../state/store";
import { errorToastPayload } from "../copy/errors";
import { BEGINNER_COPY } from "../copy/beginner";
import type {
  AdviceResponse,
  AnyResult,
  PreprocessState,
  PreviewResponse,
  RuleOut,
} from "../state/types";

/** Which registry operation implements each rule id. Kept here, next to the UI that
 *  offers it — the backend validates the op name, so this cannot drift silently. */
const OP_OF_RULE: Record<string, { op: string; params: Record<string, unknown> }> = {
  "prep.transform.log": { op: "transform.log", params: {} },
  "prep.transform.sqrt": { op: "transform.sqrt", params: {} },
  "prep.transform.boxcox": { op: "transform.boxcox", params: {} },
  "prep.log.undefined_blocker": { op: "transform.yeojohnson", params: {} },
  "prep.outlier.flag_not_delete": { op: "outlier.flag", params: {} },
  "prep.impute.seasonal_interpolation": { op: "impute.seasonal", params: {} },
  "prep.impute.state_space": { op: "impute.kalman", params: {} },
  "prep.gap.long_blocker": { op: "impute.none", params: {} },
  "prep.smooth.for_presentation": { op: "smooth.ma", params: { window: 7 } },
  "prep.window.trim": { op: "trim", params: {} },
  "prep.freq.aggregate": { op: "aggregate", params: { freq: "QS", how: "sum" } },
};

export function AdvicePanel({ step = "S2" }: { step?: "S2" | "S3" }) {
  const { sessionId, refresh, pushToast, session } = useStore();
  const [advice, setAdvice] = useState<AdviceResponse | null>(null);
  const [state, setState] = useState<PreprocessState | null>(null);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [previewRule, setPreviewRule] = useState<string | null>(null);
  /** Set when the series changes underneath the advice: Step 1 must re-run first. */
  const [needsReprofile, setNeedsReprofile] = useState(false);

  const hasData = Boolean(session?.dataset);
  const stageStatus = session?.pipeline.stages.S2?.status ?? "not_started";

  const load = useCallback(async () => {
    if (!sessionId) return;
    const [a, s] = await Promise.all([api.advice(), api.preprocessState()]);
    setState(s);
    if (a && "recommendations" in a && a.status !== "not_started" && a.recommendations) {
      setAdvice(a as AdviceResponse);
    }
  }, [sessionId]);

  const run = useCallback(async () => {
    if (!sessionId) {
      pushToast({ kind: "error", title: "Load some data first.", fix: "Step 0 → Example data." });
      return;
    }
    setBusy(true);
    try {
      const res = await api.runAdvice();
      setAdvice(res);
      await refresh();
      setState(await api.preprocessState());
      pushToast({
        kind: res.blockers.length ? "warn" : "success",
        title: res.headline,
        body: `${res.recommendations.length} suggestion(s), ${res.blockers.length} blocker(s) — in ${res.duration_ms} ms.`,
      });
    } catch (e) {
      report(e, pushToast, "The advice could not be computed.");
    } finally {
      setBusy(false);
    }
  }, [sessionId, refresh, pushToast]);

  // S2 is automatic, exactly like S1. Guarded against StrictMode's double effect.
  const autoRan = useRef<string | null>(null);
  useEffect(() => {
    // S3 is the same engine seen from the other side: it shows what has been applied and
    // where to undo it. It must not re-run the advisor on open — that is S2's job.
    if (step === "S3") return;
    if (!sessionId || !hasData || busy) return;
    if (advice || stageStatus === "complete") {
      if (!advice) void load();
      return;
    }
    if (autoRan.current === sessionId) return;
    autoRan.current = sessionId;
    void run();
  }, [sessionId, hasData, busy, advice, stageStatus, load, run, step]);

  const doPreview = useCallback(
    async (rule: RuleOut) => {
      const mapped = OP_OF_RULE[rule.id];
      if (!mapped) {
        pushToast({
          kind: "info",
          title: rule.recommend,
          body: "This one is guidance for later steps rather than a change to the data.",
        });
        return;
      }
      setBusy(true);
      try {
        const p = await api.previewOp(mapped.op, mapped.params);
        setPreview(p);
        setPreviewRule(rule.id);
      } catch (e) {
        report(e, pushToast, "That preview could not be produced.");
      } finally {
        setBusy(false);
      }
    },
    [pushToast],
  );

  const doApply = useCallback(
    async (ruleId: string) => {
      const mapped = OP_OF_RULE[ruleId];
      if (!mapped) return;
      setBusy(true);
      try {
        await api.applyOp(mapped.op, mapped.params, ruleId);
        setPreview(null);
        setPreviewRule(null);
        await refresh();
        setState(await api.preprocessState());
        pushToast({
          kind: "success",
          title: "Applied.",
          body: "The series has a new version. Undo is available, and Step 1 is now marked stale.",
        });
        // Advice is stale by design once the series changes, and the advisor will (rightly)
        // refuse to read a profile of the old data. So refresh the profile first, then the
        // advice — in that order, and only when the user asks for it.
        setNeedsReprofile(true);
      } catch (e) {
        report(e, pushToast, "That treatment could not be applied.");
      } finally {
        setBusy(false);
      }
    },
    [refresh, pushToast],
  );

  const doUndo = useCallback(async () => {
    setBusy(true);
    try {
      await api.undoOp();
      await refresh();
      setState(await api.preprocessState());
      pushToast({ kind: "info", title: "Undone.", body: "The previous version of the series is back." });
      setNeedsReprofile(true);
    } catch (e) {
      report(e, pushToast, "Nothing could be undone.");
    } finally {
      setBusy(false);
    }
  }, [refresh, pushToast]);

  const reprofileAndAdvise = useCallback(async () => {
    setBusy(true);
    try {
      await api.runProfile();
      const res = await api.runAdvice();
      setAdvice(res);
      setNeedsReprofile(false);
      await refresh();
      setState(await api.preprocessState());
      pushToast({ kind: "success", title: "Fresh advice.", body: "Step 1 and Step 2 were both re-run on the current series." });
    } catch (e) {
      report(e, pushToast, "The refresh failed.");
    } finally {
      setBusy(false);
    }
  }, [refresh, pushToast]);

  const results = useMemo<AnyResult[]>(() => advice?.results ?? [], [advice]);

  useEffect(() => {
    if (step === "S3" && sessionId && hasData) {
      void api.preprocessState().then(setState).catch(() => undefined);
    }
  }, [step, sessionId, hasData]);

  if (!hasData) {
    return (
      <div className="callout callout-warn" style={{ marginTop: 0 }}>
        <strong>No data yet.</strong> Load a file or an example in Step 0. The advisor reads the
        numbers computed in Step 1, so it needs a series first.
      </div>
    );
  }

  return (
    <div>
      {busy && !advice && <Skeleton />}

      {step === "S3" && (
        <div className="callout" style={{ marginTop: 0 }}>
          <strong>This is where treatments are applied.</strong> The suggestions themselves are in
          Step 2; here you can see exactly what has been done to the series, undo it, and check the
          version the rest of the pipeline is using.
        </div>
      )}

      {needsReprofile && (
        <div className="callout callout-warn" style={{ marginTop: 0, marginBottom: 12 }}>
          <strong>The series changed, so this advice describes the old data.</strong>
          <p style={{ margin: "4px 0 8px" }}>
            Re-running Step 1 and Step 2 takes about a second and lets the rules see the series
            you actually have now.
          </p>
          <button className="btn btn-sm btn-primary" onClick={() => void reprofileAndAdvise()} disabled={busy}>
            <IconRefresh size={13} /> Re-run Steps 1 and 2
          </button>
        </div>
      )}

      {advice && (
        <>
          <Headline advice={advice} busy={busy} onRerun={() => void run()} />
          {advice.blockers.length > 0 && (
            <div className="callout callout-warn" style={{ marginTop: 12 }}>
              <strong>
                {advice.blockers.length} decision{advice.blockers.length === 1 ? "" : "s"} to make first
              </strong>
              <p style={{ margin: "4px 0 0" }}>
                These are not preferences — the analysis is unreliable until they are settled.
                You can still continue, and the caveat will follow every result downstream.
              </p>
            </div>
          )}

          <StepStrip state={state} busy={busy} onUndo={() => void doUndo()} />

          {preview && previewRule && (
            <PreviewCard
              preview={preview}
              ruleId={previewRule}
              busy={busy}
              onApply={() => void doApply(previewRule)}
              onDiscard={() => {
                setPreview(null);
                setPreviewRule(null);
              }}
            />
          )}

          <div style={{ marginTop: 16 }}>
            {advice.blockers.map((r) => (
              <RuleCard key={r.id} rule={r} kind="blocker" onPreview={() => void doPreview(r)} busy={busy} />
            ))}
            {advice.recommendations.map((r) => (
              <RuleCard
                key={r.id}
                rule={r}
                kind={OP_OF_RULE[r.id] ? "action" : "note"}
                onPreview={() => void doPreview(r)}
                busy={busy}
              />
            ))}
          </div>

          <h2 style={{ margin: "20px 0 10px", fontSize: 14 }}>The evidence</h2>
          {results.map((r) => (
            <ResultCard key={r.id} result={r} />
          ))}

          <p style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 14 }}>
            Every recommendation above was produced by a rule in{" "}
            <code>advisor/rules_prep.py</code> reading the numbers in{" "}
            <code>facts_used</code> ({Object.keys(advice.facts_used).length} values). No text here was
            written by a language model.
          </p>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ parts */

function Headline({
  advice, busy, onRerun,
}: { advice: AdviceResponse; busy: boolean; onRerun: () => void }) {
  const tone = advice.blockers.length ? "blocked" : advice.recommendations.length ? "advice" : "clear";
  return (
    <section
      className="card"
      style={{
        borderLeft: `3px solid ${
          tone === "blocked" ? "var(--warn)" : tone === "clear" ? "var(--ok)" : "var(--brand)"
        }`,
      }}
      aria-labelledby="advice-headline"
    >
      <div className="card-head">
        <span className="card-kind">Advice</span>
        <div>
          <div className="card-title" id="advice-headline">
            {advice.headline}
          </div>
          <div className="card-plain">
            Confidence: <strong>{advice.confidence}</strong> · {advice.recommendations.length} suggestion
            {advice.recommendations.length === 1 ? "" : "s"} · {advice.blockers.length} blocker
            {advice.blockers.length === 1 ? "" : "s"}
          </div>
        </div>
        <div className="card-actions">
          <button className="btn btn-sm" onClick={onRerun} disabled={busy}>
            <IconRefresh size={13} /> Re-run
          </button>
        </div>
      </div>
      <div className="card-body">
        <p style={{ margin: 0 }}>{advice.narrative}</p>
      </div>
    </section>
  );
}

function RuleCard({
  rule, kind, onPreview, busy,
}: {
  rule: RuleOut;
  kind: "blocker" | "action" | "note";
  onPreview: () => void;
  busy: boolean;
}) {
  const [open, setOpen] = useState(kind === "blocker");
  const canAct = kind !== "note";
  return (
    <article className="card" style={{ marginBottom: 10 }}>
      <div className="card-head">
        <span
          className="card-kind"
          style={{
            background:
              kind === "blocker" ? "var(--warn-wash)" : kind === "action" ? "var(--brand-wash)" : "var(--stale-wash)",
          }}
        >
          {kind === "blocker" ? "Blocker" : kind === "action" ? "Suggestion" : "Note"}
        </span>
        <div>
          <div className="card-title">{rule.recommend}</div>
          <div className="card-plain">{rule.because}</div>
        </div>
        <div className="card-actions">
          <span className="pill" style={{ textTransform: "capitalize" }}>{rule.confidence}</span>
          {canAct && (
            <button className="btn btn-sm" onClick={onPreview} disabled={busy}>
              Preview
            </button>
          )}
          <button
            className="btn btn-sm btn-ghost"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
          >
            {open ? "Less" : "Details"}
          </button>
        </div>
      </div>
      {open && (
        <div className="card-body" style={{ paddingTop: 0 }}>
          <dl style={{ display: "grid", gridTemplateColumns: "max-content 1fr", gap: "6px 12px", margin: 0, fontSize: 12.5 }}>
            {rule.alternative && (
              <>
                <dt style={{ color: "var(--ink-faint)" }}>Alternative</dt>
                <dd style={{ margin: 0 }}>{rule.alternative}</dd>
              </>
            )}
            {rule.risk_if_ignored && (
              <>
                <dt style={{ color: "var(--ink-faint)" }}>If ignored</dt>
                <dd style={{ margin: 0 }}>{rule.risk_if_ignored}</dd>
              </>
            )}
            {rule.how_to && (
              <>
                <dt style={{ color: "var(--ink-faint)" }}>Where</dt>
                <dd style={{ margin: 0 }}>{rule.how_to}</dd>
              </>
            )}
            <dt style={{ color: "var(--ink-faint)" }}>Rule</dt>
            <dd style={{ margin: 0 }}>
              <code>{rule.id}</code>
              {rule.provenance.length > 0 && (
                <span style={{ color: "var(--ink-faint)" }}> — from {rule.provenance.join(", ")}</span>
              )}
            </dd>
          </dl>
        </div>
      )}
    </article>
  );
}

function PreviewCard({
  preview, ruleId, busy, onApply, onDiscard,
}: {
  preview: PreviewResponse;
  ruleId: string;
  busy: boolean;
  onApply: () => void;
  onDiscard: () => void;
}) {
  const shown = preview.preview.filter((p) => p.value !== null).slice(0, 6);
  return (
    <section className="card" style={{ marginTop: 12, borderLeft: "3px solid var(--brand)" }}>
      <div className="card-head">
        <span className="card-kind">Preview</span>
        <div>
          <div className="card-title">Before you commit</div>
          <div className="card-plain">{preview.plain}</div>
        </div>
        <div className="card-actions">
          <button className="btn btn-sm btn-primary" onClick={onApply} disabled={busy || !preview.changed}>
            <IconCheck size={13} /> Apply
          </button>
          <button className="btn btn-sm btn-ghost" onClick={onDiscard} disabled={busy}>
            Discard
          </button>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0 }}>
        <div style={{ display: "flex", gap: 18, fontSize: 12.5, color: "var(--ink-soft)" }}>
          <span>
            observations: <strong>{preview.n_before}</strong> → <strong>{preview.n_after}</strong>
          </span>
          <span>
            method: <code>{preview.method}</code>
          </span>
          <span>
            changes the data: <strong>{preview.changed ? "yes" : "no"}</strong>
          </span>
        </div>
        {shown.length > 0 && (
          <table className="data-table" style={{ marginTop: 10 }}>
            <thead>
              <tr>
                <th>time</th>
                <th style={{ textAlign: "right" }}>value after</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((p) => (
                <tr key={p.time}>
                  <td>{p.time.slice(0, 10)}</td>
                  <td style={{ textAlign: "right" }}>{format(p.value)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {preview.warnings.map((w) => (
          <div key={w} className="callout callout-warn" style={{ marginTop: 10 }}>
            <IconWarning size={13} /> {w}
          </div>
        ))}
        <p style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 8 }}>
          The session is untouched until you press Apply. This preview ran the same code Apply
          will run.
          {ruleId && <> Rule: <code>{ruleId}</code>.</>}
        </p>
      </div>
    </section>
  );
}

function StepStrip({
  state, busy, onUndo,
}: { state: PreprocessState | null; busy: boolean; onUndo: () => void }) {
  const { history, historyBusy, redoHistory } = useStore();
  if (!state) return null;
  const steps = state.steps;
  return (
    <section className="card" style={{ marginTop: 12 }}>
      <div className="card-head">
        <span className="card-kind">Applied</span>
        <div>
          <div className="card-title">
            {steps.length === 0
              ? "Nothing has been changed yet"
              : `${steps.length} treatment${steps.length === 1 ? "" : "s"} applied (series version ${state.series_version})`}
          </div>
          <div className="card-plain">
            {state.start && state.end
              ? `${state.n} observations, ${state.start.slice(0, 10)} → ${state.end.slice(0, 10)}, ${state.n_missing} missing`
              : "The series is as it was loaded."}
          </div>
        </div>
        <div className="card-actions">
          <button className="btn btn-sm" onClick={onUndo} disabled={busy || !state.can_undo}
                  aria-label={BEGINNER_COPY.undoLabel} title={state.can_undo ? BEGINNER_COPY.undoLabel : BEGINNER_COPY.noUndo}>
            ↶ {BEGINNER_COPY.undo} last
          </button>
          <button className="btn btn-sm" onClick={() => void redoHistory()}
                  disabled={busy || historyBusy || !history.can_redo}
                  aria-label={BEGINNER_COPY.redoLabel}
                  title={history.can_redo ? BEGINNER_COPY.redoLabel : BEGINNER_COPY.noRedo}>
            ↷ {BEGINNER_COPY.redo} last
          </button>
        </div>
      </div>
      {steps.length > 0 && (
        <ol style={{ margin: 0, paddingLeft: 20, fontSize: 12.5, color: "var(--ink-soft)" }}>
          {steps.map((s) => (
            <li key={s.version} style={{ marginBottom: 4 }}>
              <strong>{s.step}</strong> · <code>{s.method}</code>
              {s.reason && <span style={{ color: "var(--ink-faint)" }}> — {s.reason.slice(0, 160)}</span>}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

function Skeleton() {
  return (
    <div className="card">
      <div className="card-body">
        <div className="skeleton" style={{ width: "45%", height: 16, marginBottom: 12 }} />
        {[90, 76, 84].map((w, i) => (
          <div key={i} className="skeleton" style={{ width: `${w}%`, marginBottom: 7 }} />
        ))}
        <p style={{ fontSize: 12, color: "var(--ink-faint)", marginTop: 10 }}>
          Reading the diagnostics from Step 1 and fitting a Box–Cox estimate…
        </p>
      </div>
    </div>
  );
}

function format(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (a >= 1) return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
  return v.toPrecision(3);
}

function report(e: unknown, pushToast: ReturnType<typeof useStore>["pushToast"], fallback: string) {
  pushToast(errorToastPayload(e, fallback));
}
