/**
 * ResultCard.tsx — one renderer for every result kind.
 *
 * This is the payoff of the universal Result contract (§4.3): the UI has no
 * knowledge of any specific statistic. Give it a TableResult, FigureResult,
 * StatResult, NarrativeResult or VerdictResult and it renders the payload plus
 * the mandatory plain/why/technical/caveats layer.
 *
 * Part 2 adds the Plotly figure body; until then a figure shows its alt text and
 * a placeholder, so nothing is ever silently blank.
 */

import { useState } from "react";
import { COPY } from "../copy/strings";
import { useStore } from "../state/store";
import { PlotlyFigure } from "../charts/PlotlyFigure";
import type { AnyResult, ChecklistItem, TableResult } from "../state/types";
import { IconAlert, IconCheck, IconChevron, IconX } from "./Icons";
import { pNumber } from "../state/format";
import { tableValueSummary } from "../state/accessible";

const KIND_LABEL: Record<string, string> = {
  table: "Table",
  figure: "Figure",
  stat: "Test",
  narrative: "Summary",
  verdict: "Verdict",
};

export function ResultCard({ result, stale }: { result: AnyResult; stale?: boolean }) {
  const [showTech, setShowTech] = useState(false);
  const { openAssistant } = useStore();

  return (
    <article className="card" id={result.id} aria-labelledby={`${result.id}-title`}>
      <div className="card-head">
        <span className="card-kind">{KIND_LABEL[result.kind] ?? result.kind}</span>
        <div style={{ minWidth: 0 }}>
          <div className="card-title" id={`${result.id}-title`}>{result.title}</div>
          <div className="card-plain">{result.plain}</div>
        </div>
        <div className="card-actions">
          <button
            className="btn btn-sm btn-ghost"
            data-testid="result-explain"
            onClick={() => openAssistant({
              action: "explain",
              resultId: result.id,
              panelId: result.stage,
              question: `Explain ${result.title}.`,
            })}
            title="Grounded plain-English explanation; works offline"
          >
            ✧ {COPY.assistant.explain}
          </button>
          {result.kind === "figure" && (
            <button
              className="btn btn-sm btn-ghost"
              data-testid="result-interpret-plot"
              onClick={() => openAssistant({
                action: "interpret_plot",
                resultId: result.id,
                panelId: result.stage,
                question: `Interpret ${result.title}.`,
              })}
              title="Interpret the stored figure-data payload or an attached image"
            >
              ◱ {COPY.assistant.interpret}
            </button>
          )}
          <button
            className="btn btn-sm btn-ghost"
            onClick={() => setShowTech((v) => !v)}
            aria-expanded={showTech}
            title="Technical definition and exact method used"
          >
            <IconChevron size={12} />
            {showTech ? "Hide detail" : "Details"}
          </button>
        </div>
      </div>

      <div className="card-body">
        {stale && (
          <div className="callout callout-warn" role="status">
            <strong>Stale.</strong> {COPY.staleWarning}
          </div>
        )}

        {/* ---- payload ---- */}
        {result.kind === "table" && <TableBody t={result} />}
        {result.kind === "stat" && <StatBody r={result} />}
        {result.kind === "narrative" && <NarrativeBody r={result} />}
        {result.kind === "verdict" && <VerdictBody r={result} />}
        {result.kind === "figure" && (
          <FigureBody figureJson={result.figure_json} altText={result.alt_text} caption={result.caption} />
        )}

        {/* ---- warnings & caveats ---- */}
        {result.caveats.length > 0 && (
          <div className="callout callout-warn">
            <strong>{COPY.caveatLabel}:</strong>
            <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
              {result.caveats.map((c, i) => <li key={`caveat-${i}`}>{c}</li>)}
            </ul>
          </div>
        )}
        {result.warnings.length > 0 && (
          <div className="callout callout-bad">
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {result.warnings.map((w, i) => <li key={`warning-${i}`}>{w}</li>)}
            </ul>
          </div>
        )}

        {/* ---- expandable technical layer (hidden in plain-language mode) ---- */}
        {showTech && (
          <div className="result-tech" style={{ marginTop: 10, paddingTop: 10, borderTop: "1px solid var(--rule)" }}>
            <DetailRow label="Why it matters" value={result.why} />
            {result.technical && <DetailRow label="Technical" value={result.technical} mono />}
            <DetailRow label="Method" value={result.method} />
            {result.citations.length > 0 && (
              <DetailRow label="Citations" value={result.citations.join(" · ")} />
            )}
            {result.reproducible_snippet && (
              <div style={{ marginTop: 8 }}>
                <div className="pane-title" style={{ marginBottom: 4 }}>Reproduce this result</div>
                <pre style={{ background: "var(--canvas)", padding: 10, borderRadius: 6, overflowX: "auto", margin: 0 }}>
                  {result.reproducible_snippet}
                </pre>
              </div>
            )}
            <div style={{ fontSize: 11, color: "var(--ink-faint)", marginTop: 8, fontFamily: "var(--mono)" }}>
              {result.id} · {result.stage} · schema {result.schema_version}
            </div>
          </div>
        )}
      </div>
    </article>
  );
}

function FigureBody({
  figureJson, altText, caption,
}: {
  figureJson: Record<string, unknown>;
  altText: string;
  caption: string;
}) {
  const height = (figureJson.layout as { height?: number } | undefined)?.height;
  return (
    <div>
      <PlotlyFigure figure={figureJson} altText={altText} height={height} />
      {caption && (
        <div style={{ fontSize: 11.5, color: "var(--ink-soft)", marginTop: 4 }}>{caption}</div>
      )}
    </div>
  );
}

function DetailRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ marginBottom: 7 }}>
      <div className="pane-title" style={{ marginBottom: 2 }}>{label}</div>
      <div style={{ fontSize: 12.5, color: "var(--ink-soft)", fontFamily: mono ? "var(--mono)" : undefined }}>
        {value}
      </div>
    </div>
  );
}

/* ------------------------------- payloads -------------------------------- */

function TableBody({ t }: { t: TableResult }) {
  if (t.rows.length === 0) {
    return <div className="empty" style={{ padding: 20 }}>This table has no rows.</div>;
  }
  // Part 19: a spoken summary of the real values, not just a caption.
  const summaryId = `${t.id}-values`;
  const summary = tableValueSummary(t);
  return (
    <>
      <p id={summaryId} className="sr-only">{summary}</p>
      <div style={{ overflowX: "auto", maxHeight: 460, overflowY: "auto", border: "1px solid var(--rule)", borderRadius: 6 }}>
        <table className="data" aria-describedby={summaryId}>
          <caption className="sr-only">{t.caption || t.title}</caption>
          <thead>
            <tr>
              {t.columns.map((c) => (
                <th key={c.key} className={c.dtype === "number" || c.dtype === "integer" ? "num" : undefined}
                    scope="col" style={{ textAlign: c.align }}>
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {t.rows.map((row, i) => (
              <tr key={i}>
                {t.columns.map((c) => (
                  <td key={c.key} className={c.dtype === "number" || c.dtype === "integer" ? "num" : undefined}
                      style={{ textAlign: c.align }}>
                    {formatCell(row[c.key], c.dtype, c.format ?? undefined)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(t.caption || t.footnote) && (
        <div style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 6 }}>
          {t.caption} {t.footnote}
        </div>
      )}
    </>
  );
}

function StatBody({ r }: { r: Extract<AnyResult, { kind: "stat" }> }) {
  const p = r.p_value;
  const sig = p !== null && p !== undefined && p < 0.05;
  return (
    <div style={{ display: "flex", gap: 22, flexWrap: "wrap", alignItems: "baseline" }}>
      <Big label={r.statistic_name || "Statistic"} value={fmt(r.value)} />
      {p !== null && p !== undefined && (
        <Big label="p-value" value={pNumber(p)} accent={sig ? "var(--bad)" : "var(--ok)"} />
      )}
      {r.df !== null && r.df !== undefined && <Big label="df" value={fmt(r.df)} />}
      {r.effect_size !== null && r.effect_size !== undefined && <Big label="Effect size" value={fmt(r.effect_size)} />}
      {(r.ci_low !== null && r.ci_high !== null && r.ci_low !== undefined && r.ci_high !== undefined) && (
        <Big label="95% CI" value={`[${fmt(r.ci_low)}, ${fmt(r.ci_high)}]`} />
      )}
      {r.interpretation && (
        <div style={{ flexBasis: "100%", fontSize: 12.5, color: "var(--ink-soft)" }}>{r.interpretation}</div>
      )}
    </div>
  );
}

function NarrativeBody({ r }: { r: Extract<AnyResult, { kind: "narrative" }> }) {
  return (
    <div>
      {r.paragraphs.map((p, i) => (
        <p key={i} style={{ marginBottom: 8, fontSize: 13.5 }}>{p}</p>
      ))}
      {r.ai_drafted && (
        <span className="pill pill-warn">AI-drafted — verify before submission</span>
      )}
    </div>
  );
}

function VerdictBody({ r }: { r: Extract<AnyResult, { kind: "verdict" }> }) {
  const tone =
    r.verdict === "ready" ? "ok" :
    r.verdict === "ready_with_caveats" ? "warn" :
    r.verdict === "not_ready" ? "bad" : "";
  return (
    <div>
      <div className={`callout callout-${tone}`} style={{ marginTop: 0 }}>
        <strong>{r.headline}</strong>
      </div>
      <ul style={{ listStyle: "none", padding: 0, margin: "8px 0 0" }}>
        {r.checklist.map((c) => <ChecklistRow key={c.key} item={c} />)}
      </ul>
      {r.blockers.length > 0 && (
        <div className="callout callout-bad">
          <strong>Blockers</strong>
          <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
            {r.blockers.map((b) => <li key={b}>{b}</li>)}
          </ul>
        </div>
      )}
      {r.caveats_travelling.length > 0 && (
        <div className="callout callout-warn">
          <strong>These caveats travel with every downstream result</strong>
          <ul style={{ margin: "4px 0 0 16px", padding: 0 }}>
            {r.caveats_travelling.map((c, i) => <li key={`travelling-${i}`}>{c}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function ChecklistRow({ item }: { item: ChecklistItem }) {
  return (
    <li style={{ display: "flex", gap: 9, padding: "5px 0", borderBottom: "1px solid var(--rule)", fontSize: 12.5 }}>
      <span style={{ flex: "none", marginTop: 2, color: item.passed === true ? "var(--ok)" : item.passed === false ? "var(--bad)" : "var(--ink-faint)" }}>
        {item.passed === true ? <IconCheck size={14} /> : item.passed === false ? <IconX size={14} /> : <IconAlert size={14} />}
      </span>
      <span style={{ flex: 1 }}>{item.label}</span>
      <span style={{ fontFamily: "var(--mono)", fontSize: 11.5, color: "var(--ink-soft)" }}>{item.value}</span>
    </li>
  );
}

function Big({ label, value, accent }: { label: string; value: string; accent?: string }) {
  return (
    <div>
      <div className="pane-title" style={{ marginBottom: 1 }}>{label}</div>
      <div style={{ fontSize: 21, fontWeight: 620, fontFamily: "var(--mono)", color: accent, letterSpacing: "-0.02em" }}>
        {value}
      </div>
    </div>
  );
}

function fmt(x: number | null | undefined): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && a < 0.001) return x.toExponential(2);
  return x.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function formatCell(v: unknown, dtype: string, format?: string): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number" && (dtype === "number" || dtype === "integer")) {
    if (format === "p") return pNumber(v, 3);
    if (format === "0.000") return v.toFixed(3);
    if (Number.isInteger(v) && dtype === "integer") return v.toLocaleString();
    return fmt(v);
  }
  if (typeof v === "boolean") return v ? "Yes" : "No";
  return String(v);
}
