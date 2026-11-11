/**
 * state/accessible.ts — pure helpers that turn a table or figure into a
 * screen-reader value summary (Part 19 accessibility pass, §5.15).
 *
 * WCAG 2.1 AA asks that a non-visual user can get the *values*, not just the
 * label "table" or "chart". The backend already generates descriptive alt text;
 * these helpers add an explicit, deterministic summary of the actual numbers.
 * Pure functions, so they are unit-tested without a DOM.
 */

import type { TableColumn, TableResult } from "./types";

export function formatSummaryValue(value: unknown): string {
  if (value === null || value === undefined) return "missing";
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "not available";
    return Number(value.toPrecision(6)).toString();
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  return String(value);
}

/**
 * A compact spoken summary: caption, row count, column names, then up to
 * `maxCells` real values from the first rows. Never colour- or shape-dependent.
 */
export function tableValueSummary(
  table: Pick<TableResult, "caption" | "columns" | "rows">,
  maxCells = 12,
): string {
  const cols: TableColumn[] = table.columns ?? [];
  const rows = table.rows ?? [];
  const head = `${table.caption || "Data table"}. ${rows.length} rows, ${cols.length} columns: ${cols.map((c) => c.label).join(", ")}.`;
  const cells: string[] = [];
  for (const row of rows) {
    for (const col of cols) {
      if (cells.length >= maxCells) break;
      const raw = row[col.key];
      if (raw === null || raw === undefined || raw === "") continue;
      cells.push(`${col.label} ${formatSummaryValue(raw)}`);
    }
    if (cells.length >= maxCells) break;
  }
  const tail = cells.length
    ? ` First values: ${cells.join("; ")}.`
    : " No values are available.";
  return head + tail;
}

/** Include trace count and the backend-generated alt text with the values. */
export function figureValueSummary(altText: string, traceCount: number): string {
  const base = altText && altText.trim() ? altText.trim() : "Chart with no descriptive text available.";
  return `Chart with ${traceCount} series. ${base}`;
}

/** A short spoken summary of a demo/progress bar (aria-live friendly). */
export function progressValueSummary(done: number, total: number, current: string): string {
  return `${done} of ${total} steps complete. ${current ? `Now: ${current}.` : "Idle."}`;
}
