/**
 * src/state/format.ts — the UI's mirror of backend `core/formatting.py`.
 *
 * A p-value is never exactly zero. `toFixed(3)` used to print "p = 0.000" for a
 * very small p-value, and a naive "< 0.0001" replacement produced the equally
 * wrong "p = < 0.0001" (an equals sign in front of an inequality). `pPhrase`
 * returns the *whole* comparison, so callers must not add their own "p = ".
 */

export const SMALL_P = 0.0001;

const isNumber = (v: number | null | undefined): v is number =>
  v !== null && v !== undefined && Number.isFinite(v);

/** `p < 0.0001`, `p = 0.00037`, or `p = 0.0075` — including the operator. */
export function pPhrase(v: number | null | undefined, digits = 4): string {
  if (!isNumber(v)) return "the p-value is not available";
  if (v < SMALL_P) return "p < 0.0001";
  if (v < 0.001) return `p = ${Number(v.toPrecision(3))}`;
  return `p = ${v.toFixed(digits)}`;
}

/** Just the value, for a table cell that already has a "p-value" header. */
export function pNumber(v: number | null | undefined, digits = 4): string {
  if (!isNumber(v)) return "—";
  if (v < SMALL_P) return "< 0.0001";
  if (v < 0.001) return String(Number(v.toPrecision(3)));
  return v.toFixed(digits);
}
