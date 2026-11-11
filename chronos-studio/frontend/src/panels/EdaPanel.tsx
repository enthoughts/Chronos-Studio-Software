/**
 * panels/EdaPanel.tsx — S1: explore the data.
 *
 * Design intent (§5.2): this step requires *no clicks to get results*. Opening it
 * runs the analysis if it has not run. The user's only jobs are to read, and
 * optionally to change the rolling window.
 *
 * Layout: a sticky summary strip (the "what this is" narrative + key numbers),
 * a jump-to list of the figures so 12 cards do not become a scroll marathon, and
 * then the result cards themselves.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PlotlyFigure } from "../charts/PlotlyFigure";
import { IconRefresh } from "../components/Icons";
import { ResultCard } from "../components/ResultCard";
import { ApiError, api } from "../state/api";
import { useStore } from "../state/store";
import type { AnyResult, NarrativeResult } from "../state/types";

export function EdaPanel() {
  const { sessionId, results, refresh, pushToast, session } = useStore();
  const [busy, setBusy] = useState(false);
  const [window, setWindow] = useState<number | "">("");
  const [showAll, setShowAll] = useState(false);
  const [pendingJump, setPendingJump] = useState<string | null>(null);

  const list = results["S1"] ?? [];
  const done = (session?.pipeline.stages.S1?.status ?? "not_started") === "complete";
  const hasData = Boolean(session?.dataset);

  const run = useCallback(
    async (w?: number) => {
      if (!sessionId) {
        pushToast({
          kind: "error",
          title: "Load some data first.",
          fix: "Step 0 → Example data, or upload a file.",
        });
        return;
      }
      setBusy(true);
      try {
        const res = await api.runProfile(w);
        await refresh();
        pushToast({
          kind: "success",
          title: `Exploration finished in ${res.duration_ms ?? "?"} ms`,
          body: `${res.results.length} tables and figures produced.`,
        });
      } catch (e) {
        if (e instanceof ApiError) {
          pushToast({ kind: "error", title: e.message, body: e.code, fix: e.fix });
        } else {
          pushToast({
            kind: "error",
            title: "The exploration could not run.",
            fix: "Check the data quality report in Step 0.",
          });
        }
      } finally {
        setBusy(false);
      }
    },
    [sessionId, refresh, pushToast],
  );

  // S1 is automatic: run once when there is data and no results yet.
  // The ref guard matters — React StrictMode double-invokes effects in development,
  // which would otherwise run the whole analysis twice and double the toasts.
  const autoRan = useRef<string | null>(null);
  useEffect(() => {
    if (!sessionId || !hasData || done || busy || list.length > 0) return;
    if (autoRan.current === sessionId) return;
    autoRan.current = sessionId;
    void run();
  }, [sessionId, hasData, done, busy, list.length, run]);

  const narrative = list.find((r) => r.kind === "narrative") as NarrativeResult | undefined;
  const figures = list.filter((r) => r.kind === "figure");
  const tables = list.filter((r) => r.kind === "table");
  const stats = list.filter((r) => r.kind === "stat");
  const rest = list.filter(
    (r) => !["figure", "table", "stat", "narrative"].includes(r.kind),
  );

  const highlight = useCallback((el: HTMLElement) => {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    el.animate(
      [{ boxShadow: "0 0 0 3px var(--brand)" }, { boxShadow: "0 0 0 0 transparent" }],
      { duration: 900 },
    );
  }, []);

  const jumpTo = useCallback((id: string) => {
    // The three inline figures use `quick-<id>`; the full ResultCards use `<id>`.
    const el = document.getElementById(`quick-${id}`) ?? document.getElementById(id);
    if (el) {
      highlight(el);
      return;
    }
    // Collapsed cards are not in the DOM yet. Expand first, then scroll in the effect below.
    setShowAll(true);
    setPendingJump(id);
  }, [highlight]);

  useEffect(() => {
    if (!pendingJump) return;
    const el = document.getElementById(pendingJump);
    if (el) highlight(el);
    setPendingJump(null);
  }, [pendingJump, showAll, highlight]);

  return (
    <div>
      {!hasData && (
        <div className="callout callout-warn" style={{ marginTop: 0 }}>
          <strong>No data yet.</strong> Load a file or an example in Step 0 — this panel then fills
          itself in automatically.
        </div>
      )}

      {busy && list.length === 0 && <LoadingSkeleton />}

      {narrative && (
        <SummaryCard
          narrative={narrative}
          facts={list.length}
          onRerun={() => void run(typeof window === "number" ? window : undefined)}
          busy={busy}
          windowValue={window}
          onWindowChange={setWindow}
        />
      )}

      {!busy && list.length > 0 && (
        <JumpBar
          items={[
            ...figures.map((f) => ({ id: f.id, label: f.title, kind: "figure" as const })),
            ...tables.map((t) => ({ id: t.id, label: t.title, kind: "table" as const })),
            ...stats.map((s) => ({ id: s.id, label: s.title, kind: "stat" as const })),
            ...rest.map((r) => ({ id: r.id, label: r.title, kind: "note" as const })),
          ]}
          onJump={jumpTo}
        />
      )}

      {/* --- the two key figures, inline, for people who never scroll --- */}
      {figures.length > 0 && <QuickFigures figures={figures} />}

      {list.length > 0 && (
        <h2 style={{ margin: "18px 0 10px", fontSize: 14 }}>
          All results{" "}
          <button
            className="btn btn-sm btn-ghost"
            onClick={() => setShowAll((v) => !v)}
            aria-expanded={showAll}
          >
            {showAll ? "collapse" : `show all ${list.length}`}
          </button>
        </h2>
      )}

      {(showAll ? list : []).map((r) => (
        <ResultCard key={r.id} result={r} />
      ))}

      {list.length > 0 && !showAll && (
        <div style={{ display: "grid", gap: 12 }}>
          {tables.slice(0, 2).map((r) => <ResultCard key={r.id} result={r} />)}
          {stats.slice(0, 2).map((r) => <ResultCard key={r.id} result={r} />)}
          {rest.map((r) => <ResultCard key={r.id} result={r} />)}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ parts */

function LoadingSkeleton() {
  return (
    <div className="card">
      <div className="card-body">
        <div className="skeleton" style={{ width: "35%", height: 16, marginBottom: 12 }} />
        {[92, 84, 70, 88].map((w, i) => (
          <div key={i} className="skeleton" style={{ width: `${w}%`, marginBottom: 7 }} />
        ))}
        <div className="skeleton" style={{ height: 180, marginTop: 14 }} />
        <p style={{ fontSize: 12, color: "var(--ink-faint)", marginTop: 10 }}>
          Reading the series, fitting a robust STL to isolate anomalies, and building the figures…
        </p>
      </div>
    </div>
  );
}

function SummaryCard({
  narrative, onRerun, busy, windowValue, onWindowChange,
}: {
  narrative: NarrativeResult;
  facts: number;
  onRerun: () => void;
  busy: boolean;
  windowValue: number | "";
  onWindowChange: (v: number | "") => void;
}) {
  return (
    <section
      className="card"
      style={{ borderLeft: "3px solid var(--brand)" }}
      aria-labelledby="eda-summary"
    >
      <div className="card-head">
        <span className="card-kind">Summary</span>
        <div>
          <div className="card-title" id="eda-summary">
            What this series is
          </div>
          <div className="card-plain">
            Generated from the computed numbers below — not from a language model.
          </div>
        </div>
        <div className="card-actions">
          <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11.5 }}>
            <span style={{ color: "var(--ink-faint)" }}>rolling window</span>
            <input
              type="number"
              min={2}
              max={120}
              value={windowValue}
              onChange={(e) =>
                onWindowChange(e.target.value === "" ? "" : Number(e.target.value))
              }
              style={{
                width: 62, height: 25, borderRadius: 6, padding: "0 6px",
                border: "1px solid var(--rule-strong)", background: "var(--paper)",
                fontFamily: "var(--mono)",
              }}
              aria-label="Rolling window length"
            />
          </label>
          <button className="btn btn-sm" onClick={onRerun} disabled={busy}>
            <IconRefresh size={12} /> {busy ? "Running…" : "Re-run"}
          </button>
        </div>
      </div>
      <div className="card-body">
        {narrative.paragraphs.map((p, i) => (
          <p key={i} style={{ fontSize: 13.5, marginBottom: 9, lineHeight: 1.6 }}>
            {p}
          </p>
        ))}
      </div>
    </section>
  );
}

function JumpBar({
  items, onJump,
}: {
  items: { id: string; label: string; kind: "figure" | "table" | "stat" | "note" }[];
  onJump: (id: string) => void;
}) {
  return (
    <nav
      aria-label="Jump to a result"
      style={{
        display: "flex", flexWrap: "wrap", gap: 6, margin: "12px 0",
        padding: 10, background: "var(--panel)", border: "1px solid var(--rule)",
        borderRadius: "var(--radius-sm)",
      }}
    >
      <span className="pane-title" style={{ margin: 0, alignSelf: "center", marginRight: 4 }}>
        Jump to
      </span>
      <span style={{ fontSize: 11, color: "var(--ink-faint)", alignSelf: "center", marginRight: 4 }}>
        opens the full list when needed
      </span>
      {items.map((it) => (
        <button
          key={it.id}
          className="btn btn-sm btn-ghost"
          onClick={() => onJump(it.id)}
        >
          {it.kind === "figure" ? "◱" : it.kind === "table" ? "▦" : it.kind === "stat" ? "!" : "•"} {it.label}
        </button>
      ))}
    </nav>
  );
}

function QuickFigures({ figures }: { figures: AnyResult[] }) {
  const picked = useMemo(() => {
    const wanted = ["fig.series", "fig.month_year_heatmap", "fig.seasonal_profile"];
    return wanted
      .map((id) => figures.find((f) => f.id === id))
      .filter((f): f is AnyResult => Boolean(f));
  }, [figures]);

  if (picked.length === 0) return null;

  return (
    <div style={{ display: "grid", gap: 14, marginBottom: 6 }}>
      {picked.map((f) => {
        const fig = f as Extract<AnyResult, { kind: "figure" }>;
        const height = (fig.figure_json.layout as { height?: number } | undefined)?.height;
        return (
          <section className="card" key={f.id} id={`quick-${f.id}`}>
            <div className="card-head">
              <span className="card-kind">Figure</span>
              <div>
                <div className="card-title">{f.title}</div>
                <div className="card-plain">{f.plain}</div>
              </div>
            </div>
            <div className="card-body">
              <PlotlyFigure
                figure={fig.figure_json}
                altText={fig.alt_text}
                height={height}
              />
            </div>
          </section>
        );
      })}
    </div>
  );
}
