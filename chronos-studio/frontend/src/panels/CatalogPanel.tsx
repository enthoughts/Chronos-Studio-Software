/**
 * panels/CatalogPanel.tsx — Step 15's Part 16 completeness catalogue.
 *
 * The panel is read-only with respect to the statistics: it asks the backend for
 * the scorecard built from stored results and stored objects, then renders the
 * universal figure/table previews. All downloads go through the per-item API,
 * so the same caption, alt text and data-behind rows are used by the screen and
 * by PNG/SVG/PDF/XLSX/CSV exports.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { COPY } from "../copy/strings";
import { PlotlyFigure } from "../charts/PlotlyFigure";
import { ApiError, api } from "../state/api";
import { useStore } from "../state/store";
import type {
  CatalogFormat,
  CatalogItem,
  CatalogKind,
  CatalogMode,
  CatalogReportAction,
  CatalogReportResponse,
  CatalogResponse,
  CatalogScorecard,
} from "../state/types";

export type CatalogFilterKind = "all" | CatalogKind;

const FORMAT_LABELS: Record<CatalogFormat, string> = {
  png: COPY.catalog.png,
  svg: COPY.catalog.svg,
  pdf: COPY.catalog.pdf,
  xlsx: COPY.catalog.xlsx,
  csv: COPY.catalog.csv,
};

const ALL_FORMATS: CatalogFormat[] = ["png", "svg", "pdf", "xlsx", "csv"];
const ALL_MODES: CatalogMode[] = ["light", "dark", "print"];

export function catalogFormatLabel(format: CatalogFormat): string {
  return FORMAT_LABELS[format];
}

export function filterCatalogItems(
  items: CatalogItem[],
  group: string,
  kind: CatalogFilterKind,
  query: string,
): CatalogItem[] {
  const q = query.trim().toLowerCase();
  return items.filter((item) => {
    if (group !== "all" && item.group !== group) return false;
    if (kind !== "all" && item.kind !== kind) return false;
    if (!q) return true;
    return [item.id, item.title, item.caption, item.alt_text, item.group, item.stage, item.source]
      .join(" ")
      .toLowerCase()
      .includes(q);
  });
}

export function catalogGroupCounts(
  scorecard: CatalogScorecard,
): { name: string; plots: number; tables: number }[] {
  return Object.entries(scorecard.groups).map(([name, count]) => ({
    name, plots: count.plots, tables: count.tables,
  }));
}

/** Apply the report basket returned by an add/remove/set/clear action.
 *
 * The response, not a locally guessed id list, is authoritative: it carries the
 * full item metadata needed by the basket labels and the selected-item chips.
 * Keeping this as a pure function lets the UI state transition be regression
 * tested without a browser or a running API.
 */
export function catalogWithReport(
  data: CatalogResponse,
  response: CatalogReportResponse,
): CatalogResponse {
  return {
    ...data,
    report: {
      item_ids: response.item_ids,
      items: response.items,
      count: response.count,
    },
  };
}

export function catalogCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return "—";
    }
  }
  return String(value);
}

function CatalogMetric({
  label, value, target, ok,
}: { label: string; value: number; target: number; ok: boolean }) {
  return (
    <div className="catalog-metric">
      <div className="pane-title" style={{ marginBottom: 4 }}>{label}</div>
      <div className="catalog-metric-value">
        {value}<span> / {target}</span>
      </div>
      <span className={`pill ${ok ? "pill-ok" : "pill-warn"}`}>
        {ok ? COPY.catalog.pass : COPY.catalog.fail}
      </span>
    </div>
  );
}

function CatalogTablePreview({ item }: { item: CatalogItem }) {
  const rows = item.rows.slice(0, 8);
  if (!item.columns.length || !rows.length) {
    return <div className="empty" style={{ padding: 18 }}>{COPY.catalog.noFigure}</div>;
  }
  return (
    <div style={{ overflowX: "auto", maxHeight: 260, border: "1px solid var(--rule)", borderRadius: 6 }}>
      <table className="data catalog-table">
        <caption className="sr-only">{item.caption}</caption>
        <thead>
          <tr>
            {item.columns.map((c) => (
              <th key={c.key} scope="col" style={{ textAlign: c.align }}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {item.columns.map((c) => (
                <td
                  key={c.key}
                  className={c.dtype === "number" || c.dtype === "integer" ? "num" : undefined}
                  style={{ textAlign: c.align }}
                >
                  {catalogCell(row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function CatalogPanel({ onOpenReport }: { onOpenReport?: () => void } = {}) {
  const { sessionId, results, pushToast } = useStore();
  const [mode, setMode] = useState<CatalogMode>("light");
  const [data, setData] = useState<CatalogResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [group, setGroup] = useState("all");
  const [kind, setKind] = useState<CatalogFilterKind>("all");
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [explainId, setExplainId] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!sessionId) {
      setData(null);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await api.catalog(mode);
      setData(response);
    } catch (e) {
      const message = e instanceof ApiError ? `${e.message} ${e.fix}` : COPY.catalog.empty;
      setError(message);
      pushToast({
        kind: "error", title: message,
        body: e instanceof ApiError ? e.code : undefined,
        fix: e instanceof ApiError ? e.fix : undefined,
      });
    } finally {
      setLoading(false);
    }
  }, [sessionId, mode, pushToast]);

  // A new result changes the catalogue; reload without requiring a manual refresh.
  useEffect(() => {
    void load();
  }, [load, results]);

  const selected = useMemo(
    () => new Set(data?.report.item_ids ?? []),
    [data],
  );
  const allItems = useMemo(
    () => (data ? [...data.plots, ...data.tables] : []),
    [data],
  );
  const filtered = useMemo(
    () => filterCatalogItems(allItems, group, kind, query),
    [allItems, group, kind, query],
  );

  const reportAction = async (action: CatalogReportAction["action"], itemId?: string) => {
    const key = `report:${itemId ?? "all"}`;
    setBusy(key);
    try {
      const body: CatalogReportAction = action === "clear"
        ? { action: "clear" }
        : { action, item_id: itemId ?? null };
      const response = await api.catalogReportAction(body, mode);
      setData((prev) => prev ? catalogWithReport(prev, response) : prev);
      pushToast({
        kind: "success",
        title: action === "remove" || action === "clear"
          ? COPY.catalog.removedToast
          : COPY.catalog.insertedToast,
        body: itemId,
      });
    } catch (e) {
      pushToast({
        kind: "error",
        title: e instanceof ApiError ? e.message : COPY.catalog.empty,
        fix: e instanceof ApiError ? e.fix : undefined,
      });
    } finally {
      setBusy(null);
    }
  };

  const download = async (item: CatalogItem, format: CatalogFormat) => {
    const key = `${item.id}:${format}`;
    setBusy(key);
    try {
      const { blob, filename } = await api.catalogDownload(item.id, format, mode);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      pushToast({
        kind: "success", title: COPY.catalog.downloadedToast,
        body: `${item.title} · ${format.toUpperCase()}`,
      });
    } catch (e) {
      pushToast({
        kind: "error",
        title: e instanceof ApiError ? e.message : COPY.catalog.empty,
        fix: e instanceof ApiError ? e.fix : undefined,
      });
    } finally {
      setBusy(null);
    }
  };

  return (
    <main className="results-pane" aria-label={COPY.catalog.title}>
      <header style={{ marginBottom: 14 }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
          <div>
            <div className="pane-title">{COPY.catalog.eyebrow}</div>
            <h1>{COPY.catalog.title}</h1>
          </div>
          {onOpenReport && (
            <button className="btn btn-sm btn-primary" onClick={onOpenReport}>
              {COPY.report.backToReport}
            </button>
          )}
        </div>
        <p style={{ fontSize: 13, color: "var(--ink-soft)", maxWidth: 820 }}>
          {COPY.catalog.intro}
        </p>
      </header>

      {!sessionId && (
        <div className="empty">
          <h2 style={{ marginTop: 12 }}>{COPY.noSession.title}</h2>
          <p>{COPY.noSession.body}</p>
        </div>
      )}

      {error && <div className="callout callout-bad" role="alert">{error}</div>}

      {loading && (
        <div className="card"><div className="card-body" aria-busy="true">{COPY.catalog.loading}</div></div>
      )}

      {data && !loading && (
        <>
          <section className="card catalog-score" data-testid="catalog-scorecard">
            <div className="card-head">
              <span className="card-kind">S15</span>
              <div>
                <div className="card-title">{COPY.catalog.scorecard}</div>
                <div className="card-plain">{COPY.catalog.scorecardHint}</div>
              </div>
            </div>
            <div className="card-body">
              <div className="catalog-metrics">
                <CatalogMetric
                  label={COPY.catalog.plots} value={data.scorecard.plots}
                  target={data.scorecard.min_plots} ok={data.scorecard.plots_ok}
                />
                <CatalogMetric
                  label={COPY.catalog.tables} value={data.scorecard.tables}
                  target={data.scorecard.min_tables} ok={data.scorecard.tables_ok}
                />
                <div className="catalog-metric">
                  <div className="pane-title" style={{ marginBottom: 4 }}>{COPY.catalog.minPlots}</div>
                  <div className="catalog-metric-value">
                    {data.scorecard.captions_ok && data.scorecard.alt_text_ok && data.scorecard.data_behind_ok
                      ? COPY.catalog.pass : COPY.catalog.fail}
                  </div>
                  <span className="hint">captions · alt · data behind</span>
                </div>
              </div>
              <div className="pane-title" style={{ marginTop: 14 }}>{COPY.catalog.groups}</div>
              <div className="catalog-groups">
                {catalogGroupCounts(data.scorecard).map((g) => (
                  <span className="catalog-chip" key={g.name}>
                    {g.name}: {g.plots} {COPY.catalog.plots} · {g.tables} {COPY.catalog.tables}
                  </span>
                ))}
              </div>
              <div className="callout catalog-style">
                <strong>{COPY.catalog.styling}:</strong> {data.scorecard.styling.single_source} ·{" "}
                {COPY.catalog.colorblindSafe}
                <br />
                <strong>{COPY.catalog.exports}:</strong>{" "}
                {data.scorecard.exports.formats.join(", ")} · {data.scorecard.exports.dpi} dpi ·{" "}
                {COPY.catalog.exportsHint}
              </div>
            </div>
          </section>

          <section className="card catalog-controls" data-testid="catalog-controls">
            <div className="card-body">
              <div className="catalog-filters">
                <select
                  value={group}
                  aria-label={COPY.catalog.allGroups}
                  onChange={(e) => setGroup(e.target.value)}
                >
                  <option value="all">{COPY.catalog.allGroups}</option>
                  {data.groups.map((g) => <option key={g} value={g}>{g}</option>)}
                </select>
                <select
                  value={kind}
                  aria-label={COPY.catalog.allKinds}
                  onChange={(e) => setKind(e.target.value as CatalogFilterKind)}
                >
                  <option value="all">{COPY.catalog.allKinds}</option>
                  <option value="plot">{COPY.catalog.plotsOnly}</option>
                  <option value="table">{COPY.catalog.tablesOnly}</option>
                </select>
                <input
                  type="search"
                  value={query}
                  placeholder={COPY.catalog.search}
                  aria-label={COPY.catalog.search}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <button className="btn btn-sm" disabled={loading} onClick={() => void load()}>
                  {COPY.catalog.refresh}
                </button>
              </div>
              <div className="catalog-mode">
                <span className="pane-title" style={{ margin: 0 }}>{COPY.catalog.styling}</span>
                {ALL_MODES.map((m) => (
                  <button
                    key={m}
                    className={`btn btn-sm ${mode === m ? "btn-primary" : ""}`}
                    aria-pressed={mode === m}
                    onClick={() => setMode(m)}
                  >
                    {COPY.catalog[m]}
                  </button>
                ))}
              </div>
            </div>
          </section>

          <section className="card catalog-basket" data-testid="catalog-basket">
            <div className="card-head">
              <span className="card-kind">S15</span>
              <div>
                <div className="card-title">{COPY.catalog.reportBasket} ({data.report.count})</div>
                <div className="card-plain">{COPY.catalog.reportBasketHint}</div>
              </div>
              {data.report.count > 0 && (
                <div className="card-actions">
                  <button
                    className="btn btn-sm"
                    disabled={busy === "report:all"}
                    onClick={() => void reportAction("clear")}
                  >
                    {COPY.catalog.clearBasket}
                  </button>
                </div>
              )}
            </div>
            <div className="card-body">
              {data.report.count === 0 ? (
                <p style={{ margin: 0, color: "var(--ink-faint)" }}>{COPY.catalog.reportEmpty}</p>
              ) : (
                <ul className="catalog-basket-list">
                  {data.report.items.map((item) => (
                    <li key={item.id}>
                      <span>{item.title}</span>
                      <button
                        className="btn btn-sm btn-ghost"
                        disabled={busy === `report:${item.id}`}
                        onClick={() => void reportAction("remove", item.id)}
                      >
                        {COPY.catalog.remove}
                      </button>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          {filtered.length === 0 && (
            <div className="empty"><p>{COPY.catalog.empty}</p></div>
          )}

          <div className="catalog-grid" data-testid="catalog-items">
            {filtered.map((item) => {
              const isSelected = selected.has(item.id);
              return (
                <article className="card catalog-item" key={item.id} id={`catalog.${item.id}`} data-testid="catalog-item">
                  <div className="card-head">
                    <span className="card-kind">
                      {item.kind === "plot" ? COPY.catalog.plotsOnly : COPY.catalog.tablesOnly}
                    </span>
                    <div style={{ minWidth: 0 }}>
                      <div className="card-title">{item.title}</div>
                      <div className="card-plain">{item.id}</div>
                    </div>
                    <div className="card-actions">
                      <span className="catalog-chip">{item.stage}</span>
                      <span className="catalog-chip">{item.group}</span>
                      {isSelected && <span className="pill pill-ok">{COPY.catalog.inserted}</span>}
                    </div>
                  </div>
                  <div className="card-body">
                    {item.kind === "plot" ? (
                      Object.keys(item.figure_json || {}).length
                        ? <PlotlyFigure figure={item.figure_json} altText={item.alt_text} height={260} />
                        : <div className="empty" style={{ padding: 18 }}>{COPY.catalog.noFigure}</div>
                    ) : (
                      <CatalogTablePreview item={item} />
                    )}
                    <div className="catalog-caption">{item.caption}</div>
                    <div className="catalog-actions">
                      <button
                        className={`btn btn-sm ${isSelected ? "btn-ghost" : "btn-primary"}`}
                        aria-pressed={isSelected}
                        disabled={busy === `report:${item.id}`}
                        onClick={() => void reportAction(isSelected ? "remove" : "add", item.id)}
                      >
                        {isSelected ? COPY.catalog.remove : COPY.catalog.insert}
                      </button>
                      <button
                        className="btn btn-sm btn-ghost"
                        aria-expanded={explainId === item.id}
                        onClick={() => setExplainId(explainId === item.id ? null : item.id)}
                      >
                        {explainId === item.id
                          ? COPY.catalog.hideExplanation
                          : COPY.catalog.explain}
                      </button>
                      <span className="pane-title" style={{ margin: "0 2px" }}>
                        {COPY.catalog.dataBehind}:
                      </span>
                      {ALL_FORMATS.map((format) => (
                        <button
                          key={format}
                          className="btn btn-sm"
                          disabled={busy === `${item.id}:${format}`}
                          onClick={() => void download(item, format)}
                        >
                          {catalogFormatLabel(format)}
                        </button>
                      ))}
                    </div>
                    {explainId === item.id && (
                      <div className="callout catalog-explain">
                        <p><strong>{COPY.catalog.caption}:</strong> {item.caption}</p>
                        <p><strong>{COPY.catalog.alt}:</strong> {item.alt_text}</p>
                        <p><strong>{COPY.catalog.source}:</strong> {item.source}</p>
                        <p><strong>{COPY.catalog.stage}:</strong> {item.stage} · {item.group}</p>
                        <p>{COPY.catalog.explainIntro}</p>
                      </div>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </>
      )}
    </main>
  );
}
