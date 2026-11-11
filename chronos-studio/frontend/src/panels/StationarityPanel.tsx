/**
 * panels/StationarityPanel.tsx — S5: is this series stable, broken, or wandering?
 *
 * The panel is built around one idea: the *verdict matrix*. Four tests with two opposite nulls
 * are unreadable one at a time, and the reader's real question — "do I difference this?" — is
 * answered by the pattern, not by any single row. So:
 *
 *  - The matrix is the first thing on screen. Every cell is shown, including the two that mean
 *    "the tests disagree" and "not enough evidence", because those are real answers and hiding
 *    them is how a user ends up differencing a stationary series.
 *  - The verdict card names the evidence it used, in numbers, and shows the dissent separately.
 *    A verdict and its dissent are never merged into one confident sentence.
 *  - Recommendations carry the same four fields as S2 (why / alternative / cost / where) and are
 *    produced by `advisor/rules_stat.py`, not by this component.
 *  - Break dates are offered forward as dummies; the panel says so rather than letting the user
 *    assume the data were cleaned.
 *
 * Nothing here computes a statistic. Every number arrives from `/api/v1/analysis/tests`.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { IconCheck, IconRefresh } from "../components/Icons";
import { ResultCard } from "../components/ResultCard";
import { api } from "../state/api";
import { useStore } from "../state/store";
import { errorToastPayload } from "../copy/errors";
import type {
  AnyResult,
  BreakTestSummary,
  ReconcileResponse,
  RuleOut,
  StatTestsResponse,
  StatVerdict,
  TestRow,
} from "../state/types";

/** How each verdict is framed: short label, tone, and the plain question it answers. */
const VERDICT_META: Record<
  StatVerdict,
  { label: string; tone: "ok" | "warn" | "bad" | "neutral"; question: string }
> = {
  stationary: {
    label: "Stationary",
    tone: "ok",
    question: "It settles back to a stable level. Nothing to fix.",
  },
  stationary_around_break: {
    label: "Stationary around a break",
    tone: "warn",
    question: "One interruption, otherwise stable. Add a dummy; do not difference.",
  },
  trend_stationary: {
    label: "Stationary around a trend",
    tone: "warn",
    question: "It returns to a path, and the path slopes. Keep the trend.",
  },
  unit_root: {
    label: "Unit root",
    tone: "bad",
    question: "Shocks accumulate. Model the changes, not the level.",
  },
  unit_root_or_break: {
    label: "Unit root or one-off shift",
    tone: "neutral",
    question: "Two readings fit. Carry both and let the model decide.",
  },
  seasonal_unit_root: {
    label: "Seasonal unit root",
    tone: "bad",
    question: "The seasonal pattern drifts. Difference at the seasonal lag.",
  },
  inconclusive: {
    label: "Not enough evidence",
    tone: "neutral",
    question: "This window cannot separate 'settles' from 'wanders'.",
  },
  conflicting: {
    label: "Tests conflict",
    tone: "neutral",
    question: "The tests pull apart. Treat any verdict as provisional.",
  },
};

export function StationarityPanel() {
  const { sessionId, session, refresh, pushToast } = useStore();
  const [data, setData] = useState<StatTestsResponse | null>(null);
  const [rec, setRec] = useState<ReconcileResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const hasData = Boolean(session?.dataset);
  const status = session?.pipeline.stages.S5?.status ?? "not_started";
  const profiled = session?.pipeline.stages.S1?.status;

  const load = useCallback(async () => {
    if (!sessionId) return;
    const d = await api.getTests();
    setData(d);
    if (d.verdict) {
      setRec((prev) => prev ?? null);
    }
  }, [sessionId]);

  const run = useCallback(async () => {
    if (!sessionId) {
      pushToast({ kind: "error", title: "Load some data first.", fix: "Step 0 → Example data." });
      return;
    }
    setBusy(true);
    setElapsed(0);
    const tick = window.setInterval(() => setElapsed((s) => s + 1), 1000);
    try {
      await api.runTests();
      const d = await api.getTests();
      setData(d);
      setRec(await api.reconcileTests());
      await refresh();
      pushToast({
        kind: "success",
        title: d.verdict?.headline ?? "Tests complete.",
        body: `${d.unit_root.length} unit-root, ${d.seasonal.length} seasonal and ${d.breaks.length} break test(s) ran on your series.`,
      });
    } catch (e) {
      report(e, pushToast, "The tests could not finish.");
    } finally {
      window.clearInterval(tick);
      setBusy(false);
    }
  }, [sessionId, refresh, pushToast]);

  // S5 does not run itself: unlike S1/S2 it costs seconds and writes a verdict, and the user
  // should see the button. But an already-computed battery is read back on open.
  const loaded = useRef<string | null>(null);
  useEffect(() => {
    if (!sessionId || !hasData) return;
    if (loaded.current === sessionId) return;
    loaded.current = sessionId;
    void load();
  }, [sessionId, hasData, load]);

  const results = useMemo<AnyResult[]>(() => data?.results ?? [], [data]);

  if (!hasData) {
    return (
      <div className="callout callout-warn" style={{ marginTop: 0 }}>
        <strong>No data yet.</strong> Load a file or an example in Step 0 — there is nothing to
        test for stability until there is a series.
      </div>
    );
  }

  if (profiled !== "complete" && profiled !== "stale" && !data?.verdict) {
    return (
      <div className="callout" style={{ marginTop: 0 }}>
        <strong>Step 1 has to run first.</strong> The stationarity battery takes its sample size,
        period and caveats from the profile, so run Step 1 (Explore) and come back.
      </div>
    );
  }

  const verdict = rec ?? data?.verdict ?? null;
  const blockers = rec?.blockers ?? data?.blockers ?? [];
  const recs = rec?.recommendations ?? data?.recommendations ?? [];

  return (
    <div>
      <RunBar busy={busy} elapsed={elapsed} status={status} onRun={() => void run()} />

      {!verdict && !busy && (
        <div className="empty" style={{ marginTop: 12 }}>
          <h2>Nothing tested yet</h2>
          <p>
            Six or seven tests take a few seconds on a series this size, because the critical
            values are simulated for your exact sample rather than read off an asymptotic table.
          </p>
        </div>
      )}

      {verdict && <VerdictCard verdict={verdict} />}
      {verdict && <VerdictMatrix verdict={verdict.verdict} />}

      {verdict && <BreakCallout breaks={data?.breaks ?? []} />}

      {blockers.length > 0 && (
        <div className="callout callout-warn" style={{ marginTop: 12 }}>
          <strong>{blockers.length} decision to make before modelling</strong>
          <p style={{ margin: "4px 0 0" }}>
            This one changes what the model is allowed to see. The caveat follows every result
            downstream until it is settled.
          </p>
        </div>
      )}

      {(blockers.length > 0 || recs.length > 0) && (
        <>
          <h2 style={{ margin: "18px 0 10px", fontSize: 14 }}>What to do about it</h2>
          {blockers.map((r) => (
            <RuleCard key={r.id} rule={r} kind="blocker" />
          ))}
          {recs.map((r) => (
            <RuleCard key={r.id} rule={r} kind={PROTECT.has(r.id) ? "protect" : "note"} />
          ))}
        </>
      )}

      {data && (
        <>
          <h2 style={{ margin: "20px 0 10px", fontSize: 14 }}>The evidence</h2>
          <TestTable title="Unit-root tests" rows={data.unit_root} />
          {data.seasonal.length > 0 && <TestTable title="Seasonal unit-root tests" rows={data.seasonal} />}
          {data.breaks.length > 0 && <BreakTable tests={data.breaks} />}
          <details style={{ marginTop: 14 }}>
            <summary style={{ cursor: "pointer", fontSize: 12.5, color: "var(--ink-soft)" }}>
              Every result block for this step ({results.length})
            </summary>
            <div style={{ marginTop: 10 }}>
              {results.map((r) => (
                <ResultCard key={r.id} result={r} />
              ))}
            </div>
          </details>
        </>
      )}

      <p style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 14 }}>
        Every verdict above comes from <code>ts/tests/reconcile.py</code> reading the statistics in{" "}
        <code>facts_used</code>; the recommendations come from <code>advisor/rules_stat.py</code>.
        No sentence here was written by a language model.
      </p>
    </div>
  );
}

/** Rules that are guardrails rather than suggestions — shown differently on purpose. */
const PROTECT = new Set(["stat.already_stationary.no_diff"]);

/* ------------------------------------------------------------------ parts */

function RunBar({
  busy, elapsed, status, onRun,
}: { busy: boolean; elapsed: number; status: string; onRun: () => void }) {
  return (
    <section className="card" aria-labelledby="run-tests">
      <div className="card-head">
        <span className="card-kind">Step 4</span>
        <div>
          <div className="card-title" id="run-tests">
            {status === "complete" ? "Tests are up to date for this series" : "Test this series"}
          </div>
          <div className="card-plain">
            {busy
              ? `Running… ${elapsed}s. Simulated critical values are the slow part, and they are cached afterwards.`
              : "ADF, KPSS, PP, DF–GLS, Zivot–Andrews, seasonal unit roots and structural breaks — then one verdict."}
          </div>
        </div>
        <div className="card-actions">
          <button className="btn btn-sm btn-primary" onClick={onRun} disabled={busy}>
            {busy ? <IconRefresh size={13} /> : <IconCheck size={13} />}
            {busy ? "Running…" : status === "complete" ? "Re-run" : "Run the tests"}
          </button>
        </div>
      </div>
    </section>
  );
}

function VerdictCard({ verdict }: { verdict: NonNullable<ReconcileResponse | StatTestsResponse["verdict"]> }) {
  const meta = VERDICT_META[verdict.verdict];
  const colour =
    meta.tone === "ok" ? "var(--ok)" : meta.tone === "bad" ? "var(--warn)" : meta.tone === "warn" ? "var(--warn)" : "var(--brand)";
  const facts = "facts_used" in verdict ? verdict.facts_used : [];
  return (
    <section className="card" style={{ marginTop: 12, borderLeft: `3px solid ${colour}` }} aria-labelledby="verdict">
      <div className="card-head">
        <span className="card-kind">{meta.label}</span>
        <div>
          <div className="card-title" id="verdict">
            {verdict.headline}
          </div>
          <div className="card-plain">
            Confidence: <strong>{verdict.confidence}</strong> ·{" "}
            {verdict.differencing_needed ? "difference before modelling" : "do not difference"}
          </div>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0 }}>
        <p style={{ marginTop: 0 }}>{verdict.plain}</p>
        <p style={{ marginBottom: 0 }}>
          <strong>What to do: </strong>
          {verdict.action}
        </p>

        {facts.length > 0 && (
          <>
            <h3 style={{ fontSize: 12, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--ink-faint)", margin: "14px 0 6px" }}>
              The numbers this verdict used
            </h3>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5 }}>
              {facts.map((f) => (
                <li key={f} style={{ marginBottom: 3 }}>{f}</li>
              ))}
            </ul>
          </>
        )}

        {verdict.contradictions.length > 0 && (
          <div className="callout callout-warn" style={{ marginTop: 12 }}>
            <strong>Where the tests disagree</strong>
            <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 12.5 }}>
              {verdict.contradictions.map((c) => (
                <li key={c} style={{ marginBottom: 4 }}>{c}</li>
              ))}
            </ul>
          </div>
        )}

        {verdict.caveats.length > 0 && (
          <ul style={{ margin: "10px 0 0", paddingLeft: 18, fontSize: 12, color: "var(--ink-soft)" }}>
            {verdict.caveats.map((c) => (
              <li key={c} style={{ marginBottom: 3 }}>{c}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

/**
 * The 2×2 the whole battery exists for. Two rows (ADF rejects / does not) and two columns
 * (KPSS rejects / does not), with the cell that actually applies highlighted and the other three
 * dimmed but still readable — a user who sees only their own cell never learns the method.
 */
function VerdictMatrix({ verdict }: { verdict: StatVerdict }) {
  const cells: { adf: string; kpss: string; read: string; verdicts: StatVerdict[] }[] = [
    {
      adf: "ADF rejects a unit root",
      kpss: "KPSS does not reject stability",
      read: "Both agree: the series settles down.",
      verdicts: ["stationary"],
    },
    {
      adf: "ADF rejects a unit root",
      kpss: "KPSS rejects stability",
      read: "Stable, but not around a constant level — a trend, a break, or a slow component.",
      verdicts: ["trend_stationary", "stationary_around_break", "conflicting"],
    },
    {
      adf: "ADF does not reject",
      kpss: "KPSS does not reject stability",
      read: "Neither test finds anything: too short, too noisy, or right on the boundary.",
      verdicts: ["inconclusive"],
    },
    {
      adf: "ADF does not reject",
      kpss: "KPSS rejects stability",
      read: "Both agree: a unit root. The level is a running total of past shocks.",
      verdicts: ["unit_root", "unit_root_or_break", "seasonal_unit_root"],
    },
  ];
  return (
    <section className="card" style={{ marginTop: 12 }} aria-labelledby="matrix">
      <div className="card-head">
        <span className="card-kind">How to read it</span>
        <div>
          <div className="card-title" id="matrix">
            Two tests, opposite nulls
          </div>
          <div className="card-plain">
            ADF hopes to reject "unit root"; KPSS hopes <em>not</em> to reject "stable". Their
            agreement is evidence; their disagreement is information.
          </div>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
          {cells.map((c) => {
            const active = c.verdicts.includes(verdict);
            return (
              <div
                key={c.adf + c.kpss}
                style={{
                  border: `1px solid ${active ? "var(--brand)" : "var(--line)"}`,
                  background: active ? "var(--brand-wash)" : "transparent",
                  borderRadius: 8,
                  padding: "10px 12px",
                  opacity: active ? 1 : 0.62,
                }}
              >
                <div style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: ".05em", color: "var(--ink-faint)" }}>
                  {c.adf} · {c.kpss}
                </div>
                <div style={{ fontSize: 12.5, marginTop: 4 }}>{c.read}</div>
                {active && (
                  <div style={{ fontSize: 11.5, marginTop: 6, color: "var(--brand)", fontWeight: 600 }}>
                    ← this series
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <p style={{ fontSize: 12, color: "var(--ink-soft)", margin: "10px 0 0" }}>
          Two cells are not the whole story: a break (Zivot–Andrews) or a seasonal unit root
          (OCSB) can overturn the bottom-right cell, and both are run above.
        </p>
      </div>
    </section>
  );
}

function BreakCallout({ breaks }: { breaks: BreakTestSummary[] }) {
  const dates = breaks
    .filter((b) => b.says.includes("break"))
    .flatMap((b) => b.breaks)
    .map((b) => b.time.slice(0, 10));
  const unique = Array.from(new Set(dates));
  if (unique.length === 0) return null;
  return (
    <div className="callout" style={{ marginTop: 12 }}>
      <strong>
        {unique.length} break date{unique.length === 1 ? "" : "s"} found: {unique.join(", ")}
      </strong>
      <p style={{ margin: "4px 0 0" }}>
        These are handed forward to the model stage as step dummies. They are <em>not</em> removed
        from the data — deleting a shift destroys the information that it happened.
      </p>
    </div>
  );
}

function TestTable({ title, rows }: { title: string; rows: TestRow[] }) {
  if (rows.length === 0) {
    return (
      <div className="callout" style={{ marginTop: 10 }}>
        <strong>{title}: none could run.</strong> A test that cannot run is not a test that
        passed — for a series this short or this degenerate, the app says so rather than guessing.
      </div>
    );
  }
  return (
    <section className="card" style={{ marginTop: 12 }}>
      <div className="card-head">
        <span className="card-kind">Table</span>
        <div>
          <div className="card-title">{title}</div>
          <div className="card-plain">
            The statistic is only meaningful next to its own 5% critical value and its own null —
            the nulls point in opposite directions.
          </div>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0, overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>test</th>
              <th>its null</th>
              <th style={{ textAlign: "right" }}>statistic</th>
              <th style={{ textAlign: "right" }}>5% line</th>
              <th>reading</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.id}>
                <td>{r.test}</td>
                <td style={{ color: "var(--ink-soft)" }}>{r.null ?? "—"}</td>
                <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{fmt(r.statistic)}</td>
                <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {r.cv_5pct == null ? "—" : fmt(r.cv_5pct)}
                </td>
                <td>
                  <span
                    className="pill"
                    style={{
                      background: r.rejects_null ? "var(--brand-wash)" : "transparent",
                      border: "1px solid var(--line)",
                    }}
                  >
                    {r.reading ?? "—"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ fontSize: 11.5, color: "var(--ink-faint)", margin: "8px 0 0" }}>
          Missing rows could not run on this series. Rejecting a null is evidence, not proof.
        </p>
      </div>
    </section>
  );
}

function BreakTable({ tests }: { tests: BreakTestSummary[] }) {
  return (
    <section className="card" style={{ marginTop: 12 }}>
      <div className="card-head">
        <span className="card-kind">Table</span>
        <div>
          <div className="card-title">Structural-break tests</div>
          <div className="card-plain">
            A break is a change in the rules, not noise. The date matters more than the p-value,
            because the date becomes a column in the model.
          </div>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0, overflowX: "auto" }}>
        <table className="data-table">
          <thead>
            <tr>
              <th>test</th>
              <th style={{ textAlign: "right" }}>statistic</th>
              <th style={{ textAlign: "right" }}>5% line</th>
              <th>date(s)</th>
              <th>reading</th>
            </tr>
          </thead>
          <tbody>
            {tests.map((t) => (
              <tr key={t.id}>
                <td>{t.test}</td>
                <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{fmt(t.statistic)}</td>
                <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                  {t.cv_5pct == null ? "—" : fmt(t.cv_5pct)}
                </td>
                <td>{t.breaks.length ? t.breaks.map((b) => b.time.slice(0, 10)).join(", ") : "none found"}</td>
                <td>{t.says}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {tests.flatMap((t) => t.caveats).filter((c, i, a) => a.indexOf(c) === i).length > 0 && (
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 11.5, color: "var(--ink-faint)" }}>
            {Array.from(new Set(tests.flatMap((t) => t.caveats))).map((c) => (
              <li key={c} style={{ marginBottom: 3 }}>{c}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

function RuleCard({ rule, kind }: { rule: RuleOut; kind: "blocker" | "protect" | "note" }) {
  const [open, setOpen] = useState(kind === "blocker");
  const badge = kind === "blocker" ? "Must decide" : kind === "protect" ? "Guardrail" : "Note";
  const wash =
    kind === "blocker" ? "var(--warn-wash)" : kind === "protect" ? "var(--ok-wash, var(--brand-wash))" : "var(--stale-wash)";
  return (
    <article className="card" style={{ marginBottom: 10 }}>
      <div className="card-head">
        <span className="card-kind" style={{ background: wash }}>{badge}</span>
        <div>
          <div className="card-title">{rule.recommend}</div>
          <div className="card-plain">{rule.because}</div>
        </div>
        <div className="card-actions">
          <span className="pill" style={{ textTransform: "capitalize" }}>{rule.confidence}</span>
          <button className="btn btn-sm btn-ghost" onClick={() => setOpen((v) => !v)} aria-expanded={open}>
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

/* ------------------------------------------------------------------ utils */

function fmt(x: number): string {
  if (!Number.isFinite(x)) return "—";
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 10000)) return x.toExponential(2);
  return x.toFixed(3);
}


function report(e: unknown, pushToast: ReturnType<typeof useStore>["pushToast"], fallback: string) {
  pushToast(errorToastPayload(e, fallback));
}

export default StationarityPanel;
