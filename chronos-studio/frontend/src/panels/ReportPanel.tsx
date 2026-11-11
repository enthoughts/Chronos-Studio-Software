/**
 * panels/ReportPanel.tsx — Step 15's Part 18 finalization workspace.
 *
 * The UI never computes a statistic. It reads the weighted scorecard from
 * /export/status, lets the user confirm finalization, and downloads/builds
 * reports through /export/{kind}. The Part 16 catalogue supplies the basket.
 */

import { useCallback, useEffect, useState } from "react";
import { COPY } from "../copy/strings";
import { ApiError, api } from "../state/api";
import { useStore } from "../state/store";
import type {
  ReportKind,
  ReportMode,
  ReportNextStep,
  ReportScope,
  ReportStatus,
  ReportStatusName,
} from "../state/types";

export function verdictTone(status: ReportStatusName): string {
  if (status === "ready") return "report-tone-ok";
  if (status === "nearly") return "report-tone-warn";
  return "report-tone-bad";
}

export function reportScoreText(score: number, maxScore: number): string {
  return `${score}/${maxScore}`;
}

export function reportStepLabel(step: ReportNextStep): string {
  return `${step.number}. ${step.text}`;
}

export function reportExportPath(
  kind: ReportKind,
  mode: ReportMode,
  scope: ReportScope,
  forceDownload = false,
): string {
  const params = new URLSearchParams({ mode, scope, download: forceDownload ? "1" : "0" });
  return `/api/v1/export/${kind}?${params.toString()}`;
}

const MODES: ReportMode[] = ["print", "light", "dark"];

export function ReportPanel({ onOpenCatalog }: { onOpenCatalog?: () => void } = {}) {
  const { sessionId, pushToast, refresh } = useStore();
  const [mode, setMode] = useState<ReportMode>("print");
  const [scope, setScope] = useState<ReportScope>("all");
  const [data, setData] = useState<ReportStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!sessionId) {
      setData(null);
      return;
    }
    setLoading(true);
    setError("");
    try {
      setData(await api.reportStatus(mode, scope));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : COPY.report.failed);
    } finally {
      setLoading(false);
    }
  }, [mode, scope, sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  const download = async (kind: ReportKind, forceDownload = false) => {
    setBusy(kind);
    try {
      const { blob, filename } = await api.exportDownload(kind, mode, scope, forceDownload);
      const url = URL.createObjectURL(blob);
      if (kind === "html" && !forceDownload) {
        const opened = window.open(url, "_blank", "noopener");
        if (!opened) {
          const link = document.createElement("a");
          link.href = url;
          link.download = filename;
          document.body.appendChild(link);
          link.click();
          link.remove();
        }
        pushToast({
          kind: "success", title: COPY.report.shareToast, body: filename,
        });
      } else {
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        pushToast({
          kind: "success", title: COPY.report.downloadedToast, body: filename,
        });
      }
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (e) {
      pushToast({
        kind: "error",
        title: e instanceof ApiError ? e.message : COPY.report.failed,
        fix: e instanceof ApiError ? e.fix : undefined,
      });
    } finally {
      setBusy(null);
    }
  };

  const finalize = async () => {
    if (!window.confirm(COPY.report.finalizeConfirm)) return;
    setBusy("finalize");
    try {
      const result = await api.finalizeReport(mode, scope);
      await refresh();
      await load();
      pushToast({
        kind: "success", title: COPY.report.finalizeToast, body: result.scorecard_verdict,
      });
    } catch (e) {
      pushToast({
        kind: "error",
        title: e instanceof ApiError ? e.message : COPY.report.failed,
        fix: e instanceof ApiError ? e.fix : undefined,
      });
    } finally {
      setBusy(null);
    }
  };

  const counts = data?.catalog_counts ?? {};
  const selectedEmpty = scope === "selected" && (data?.selection_count ?? 0) === 0;

  return (
    <main className="results-pane" aria-label={COPY.report.title}>
      <header style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
          <div>
            <div className="pane-title">{COPY.report.eyebrow}</div>
            <h1>{COPY.report.title}</h1>
          </div>
          {onOpenCatalog && (
            <button className="btn btn-sm" onClick={onOpenCatalog} title={COPY.report.catalogueHint}>
              {COPY.report.catalogue}
            </button>
          )}
        </div>
        <p style={{ fontSize: 13, color: "var(--ink-soft)", maxWidth: 820 }}>
          {COPY.report.intro}
        </p>
      </header>

      {!sessionId && (
        <div className="empty">
          <h2 style={{ marginTop: 12 }}>{COPY.report.noSession}</h2>
          <p>{COPY.noSession.body}</p>
        </div>
      )}

      {error && <div className="callout callout-bad" role="alert">{error}</div>}
      {loading && (
        <div className="card"><div className="card-body" aria-busy="true">{COPY.report.loading}</div></div>
      )}

      {data && !loading && (
        <>
          <section className={`card report-verdict ${verdictTone(data.scorecard.status)}`} data-testid="report-verdict">
            <div className="card-head">
              <span className="card-kind">S15</span>
              <div>
                <div className="card-title">{COPY.report.verdict}</div>
                <div className="card-plain">{COPY.report.verdictHint}</div>
              </div>
              <div className="card-actions">
                <span className="pill pill-ok">{reportScoreText(data.scorecard.score, data.scorecard.max_score)}</span>
                <span className="catalog-chip">{data.stage_status}</span>
              </div>
            </div>
            <div className="card-body">
              <div className="report-verdict-text">{data.scorecard.verdict}</div>
              <div className="report-verdict-meta">
                {COPY.report.stageStatus}: {data.stage_status} · {COPY.report.catalogCounts}: {counts.plots ?? 0} plots · {counts.tables ?? 0} tables · {COPY.report.selectionCount}: {data.selection_count}
              </div>
            </div>
          </section>

          <section className="card report-controls" data-testid="report-controls">
            <div className="card-body">
              <div className="report-toolbar">
                <label className="report-field">
                  <span>{COPY.report.scope}</span>
                  <select
                    value={scope}
                    aria-label={COPY.report.scope}
                    onChange={(e) => setScope(e.target.value as ReportScope)}
                  >
                    <option value="all">{COPY.report.allItems}</option>
                    <option value="selected" disabled={data.selection_count === 0}>
                      {COPY.report.selectedItems} ({data.selection_count})
                    </option>
                  </select>
                </label>
                <label className="report-field">
                  <span>{COPY.report.mode}</span>
                  <select
                    value={mode}
                    aria-label={COPY.report.mode}
                    onChange={(e) => setMode(e.target.value as ReportMode)}
                  >
                    {MODES.map((m) => <option key={m} value={m}>{m}</option>)}
                  </select>
                </label>
                <button className="btn btn-sm" disabled={busy !== null} onClick={() => void load()}>
                  {COPY.report.refresh}
                </button>
                {onOpenCatalog && (
                  <button className="btn btn-sm" onClick={onOpenCatalog}>
                    {COPY.report.catalogue}
                  </button>
                )}
                <button
                  className="btn btn-sm btn-primary"
                  data-testid="report-finalize"
                  disabled={busy !== null || data.stage_status === "complete"}
                  title={COPY.report.finalizeHint}
                  onClick={() => void finalize()}
                >
                  {data.stage_status === "complete" ? COPY.report.finalized : COPY.report.finalize}
                </button>
              </div>
              {selectedEmpty && <div className="callout report-warn">{COPY.report.selectedEmpty}</div>}
              {scope === "selected" && !selectedEmpty && (
                <div className="callout">{COPY.report.selectedHint}</div>
              )}
            </div>
          </section>

          <section className="card report-scorecard" data-testid="report-scorecard">
            <div className="card-head">
              <span className="card-kind">S15</span>
              <div>
                <div className="card-title">{COPY.report.scorecard}</div>
                <div className="card-plain">{COPY.report.scorecardHint}</div>
              </div>
            </div>
            <div className="card-body">
              <div className="report-checks" role="table" aria-label={COPY.report.scorecard}>
                <div className="report-check report-check-head" role="row">
                  <span>{COPY.report.checks}</span><span>{COPY.report.weight}</span>
                  <span>{COPY.report.status}</span><span>{COPY.report.panel}</span>
                  <span>{COPY.report.results}</span><span>{COPY.report.evidence}</span>
                </div>
                {data.scorecard.items.map((item) => (
                  <div className="report-check" role="row" key={item.id} id={`report.${item.id}`}>
                    <span>{item.label}</span>
                    <span>{item.weight}</span>
                    <span className={item.passed ? "report-pass" : "report-fail"}>
                      {item.passed ? COPY.report.pass : COPY.report.fail}
                    </span>
                    <span>{item.panel_id} — {item.panel_label}</span>
                    <span className="report-links">
                      {(item.available_result_ids.length ? item.available_result_ids : item.result_ids).join(", ")}
                    </span>
                    <span>{item.evidence}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="card report-next" data-testid="report-next-steps">
            <div className="card-head">
              <span className="card-kind">S15</span>
              <div>
                <div className="card-title">{COPY.report.nextSteps}</div>
                <div className="card-plain">{COPY.report.verdictHint}</div>
              </div>
            </div>
            <div className="card-body">
              {data.scorecard.blockers.length > 0 && (
                <div className="callout callout-bad">
                  <strong>{COPY.report.blockers}:</strong> {data.scorecard.blockers.join(" · ")}
                </div>
              )}
              {data.scorecard.next_steps.length ? (
                <ol className="report-next-list">
                  {data.scorecard.next_steps.map((step) => (
                    <li key={`${step.number}-${step.item_id}`}>{reportStepLabel(step)}</li>
                  ))}
                </ol>
              ) : (
                <p>{COPY.report.nextStepsEmpty}</p>
              )}
            </div>
          </section>

          <section className="card report-exports" data-testid="report-exports">
            <div className="card-head">
              <span className="card-kind">S15</span>
              <div>
                <div className="card-title">{COPY.report.exports}</div>
                <div className="card-plain">{COPY.report.exportsHint}</div>
              </div>
            </div>
            <div className="card-body">
              <div className="report-export-grid">
                <button
                  className="btn btn-primary"
                  data-testid="report-export-html-share"
                  title={COPY.report.htmlHint}
                  disabled={busy !== null}
                  onClick={() => void download("html", false)}
                >
                  {COPY.report.shareHtml}
                </button>
                <button
                  className="btn"
                  data-testid="report-export-html-download"
                  disabled={busy !== null}
                  onClick={() => void download("html", true)}
                >
                  {COPY.report.downloadHtml}
                </button>
                {data.exports.filter((spec) => spec.kind !== "html").map((spec) => (
                  <button
                    key={spec.kind}
                    className="btn"
                    data-testid={`report-export-${spec.kind}`}
                    title={reportExportPath(spec.kind, mode, scope, true)}
                    disabled={busy !== null}
                    onClick={() => void download(spec.kind, true)}
                  >
                    {COPY.report.download}: {spec.kind === "all" ? COPY.report.all : spec.label}
                  </button>
                ))}
              </div>
              <div className="callout report-privacy">
                <strong>{COPY.report.privacy}:</strong> {data.privacy_note}
              </div>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
