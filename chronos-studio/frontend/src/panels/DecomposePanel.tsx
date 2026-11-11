/**
 * panels/DecomposePanel.tsx — S4: take the series apart, and say what each part means.
 *
 * Layout follows the order a person actually asks the questions in:
 *
 *   1. **The scorecard** — "what is this series mostly made of?" One sentence plus three shares.
 *      This is the answer; everything below it is the evidence.
 *   2. **The four panels** — observed, trend, seasonal, remainder, stacked and sharing a time
 *      axis, because the decomposition is an identity that should be checked by eye.
 *   3. **Component cards** — trend / seasonality / cycle / remainder, each with its own number,
 *      p-value, band and caveats. The remainder card is the one that matters most: a small
 *      remainder is not the same statement as a white remainder.
 *   4. **Method comparison** — the same series under several methods, with the trends overlaid
 *      and the disagreement quantified. Where the lines separate, the trend is a choice.
 *   5. **Advice**, from `advisor/rules_decompose.py` — never written here.
 *
 * Method buttons are never disabled without a reason: `/analysis/decompose/methods` says what can
 * run and why not, and the panel prints that reason on the button's tooltip and under the menu.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconCheck, IconRefresh } from "../components/Icons";
import { ResultCard } from "../components/ResultCard";
import { PlotlyFigure } from "../charts/PlotlyFigure";
import { api } from "../state/api";
import { pPhrase } from "../state/format";
import { useStore } from "../state/store";
import { errorToastPayload } from "../copy/errors";
import type {
  AnyResult,
  DecompResponse,
  DecompFinding,
  MethodOption,
  RuleOut,
} from "../state/types";

/** The five methods, and the plain sentence that says what each one is for. */
const METHOD_BLURB: Record<string, string> = {
  stl: "Robust, keeps the ends, lets the pattern drift. The default for a reason.",
  classical: "The textbook one: a moving average. Loses the first and last half-cycle.",
  mstl: "Several seasonal periods at once (e.g. weekly and yearly).",
  ucm: "A fitted model with a likelihood, so the components can be forecast.",
  hp: "Smooth trend only — no seasonal component, and the trend is defined by λ.",
  cf: "Band-pass: isolates variation in a chosen period band and calls it the cycle.",
};

const BAND_TONE: Record<string, { label: string; colour: string; wash: string }> = {
  "not applicable": { label: "n/a", colour: "var(--ink-faint)", wash: "transparent" },
  strong: { label: "strong", colour: "var(--brand)", wash: "var(--brand-wash)" },
  moderate: { label: "moderate", colour: "var(--warn)", wash: "var(--warn-wash)" },
  weak: { label: "weak", colour: "var(--ink-faint)", wash: "var(--stale-wash)" },
  none: { label: "none", colour: "var(--ink-faint)", wash: "transparent" },
};

export function DecomposePanel() {
  const { sessionId, session, refresh, pushToast } = useStore();
  const [data, setData] = useState<DecompResponse | null>(null);
  const [methods, setMethods] = useState<MethodOption[]>([]);
  const [model, setModel] = useState<"additive" | "multiplicative">("additive");
  const [method, setMethod] = useState<string>("stl");
  const [busy, setBusy] = useState(false);
  const [compare, setCompare] = useState<{
    ran: string[];
    rows: Record<string, unknown>[];
    spread: number | null;
    notes: string[];
  } | null>(null);
  const [showAll, setShowAll] = useState(false);

  const hasData = Boolean(session?.dataset);
  const period = session?.dataset?.frequency?.period ?? null;
  const status = session?.pipeline.stages.S4?.status ?? "not_started";
  const profiled = session?.pipeline.stages.S1?.status;

  const load = useCallback(async () => {
    if (!sessionId) return;
    const [d, m] = await Promise.all([api.getDecompose(), api.decomposeMethods()]);
    setData(d);
    setMethods(m.methods as unknown as MethodOption[]);
  }, [sessionId]);

  const run = useCallback(
    async (m: string, mod: "additive" | "multiplicative") => {
      if (!sessionId) {
        pushToast({ kind: "error", title: "Load some data first.", fix: "Step 0 → Example data." });
        return;
      }
      setBusy(true);
      try {
        const res = await api.runDecompose({ method: m, model: mod });
        setMethod(m);
        setModel(mod);
        await refresh();
        setData(await api.getDecompose());
        setCompare(null);
        pushToast({
          kind: "success",
          title: res.advisor?.headline ?? "Decomposition complete.",
          body: `${res.results.length} result block(s) — the scorecard is at the top.`,
        });
      } catch (e) {
        report(e, pushToast, "The decomposition could not be run.");
      } finally {
        setBusy(false);
      }
    },
    [sessionId, refresh, pushToast],
  );

  const runCompare = useCallback(async () => {
    if (!sessionId) return;
    setBusy(true);
    try {
      const res = await api.compareDecompositions(["stl", "classical", "ucm", "hp"]);
      setCompare({
        ran: res.ran,
        rows: res.rows as unknown as Record<string, unknown>[],
        spread: res.trend_spread,
        notes: res.notes ?? [],
      });
      setData(await api.getDecompose());
      pushToast({
        kind: "info",
        title: `${res.ran.length} methods compared.`,
        body:
          res.trend_spread != null
            ? `Their trend estimates differ by up to ${res.trend_spread.toFixed(3)} at a single date.`
            : "See the overlay for where they separate.",
      });
    } catch (e) {
      report(e, pushToast, "The comparison could not be run.");
    } finally {
      setBusy(false);
    }
  }, [sessionId, pushToast]);

  useEffect(() => {
    if (!sessionId || !hasData) return;
    void load();
  }, [sessionId, hasData, load]);

  const results = useMemo<AnyResult[]>(() => data?.results ?? [], [data]);
  const figures = useMemo(() => results.filter((r) => r.kind === "figure"), [results]);
  const grid = figures.find((f) => f.id === "fig.decomposition");
  const others = figures.filter((f) => f.id !== "fig.decomposition");

  if (!hasData) {
    return (
      <div className="callout callout-warn" style={{ marginTop: 0 }}>
        <strong>No data yet.</strong> Load a file or an example in Step 0 — there is nothing to
        take apart until there is a series.
      </div>
    );
  }

  if (profiled !== "complete" && profiled !== "stale" && !data) {
    return (
      <div className="callout" style={{ marginTop: 0 }}>
        <strong>Step 1 has to run first.</strong> The decomposition takes the seasonal period from
        the profile, so run Step 1 (Explore) and come back.
      </div>
    );
  }

  const findings = (data?.findings ?? []) as DecompFinding[];
  const recs = data?.recommendations ?? [];
  const blockers = data?.blockers ?? [];
  const multi = methods.length === 0 ? null : null;

  return (
    <div>
      <MethodBar
        methods={methods}
        method={data?.method ?? method}
        model={data?.model ?? model}
        period={period}
        busy={busy}
        status={status}
        onRun={(m, mod) => void run(m, mod)}
        extraNote={multi}
      />

      {!data && !busy && (
        <div className="empty" style={{ marginTop: 12 }}>
          <h2>Nothing decomposed yet</h2>
          <p>
            Pick a method above. STL is the default: it keeps every observation and tolerates
            outliers, which the textbook moving-average method does not.
          </p>
        </div>
      )}

      {data?.scorecard && (
        <section className="card" style={{ marginTop: 12, borderLeft: "3px solid var(--brand)" }} aria-labelledby="scorecard">
          <div className="card-head">
            <span className="card-kind">Scorecard</span>
            <div>
              <div className="card-title" id="scorecard">
                What this series is made of
              </div>
              <div className="card-plain">
                {data.label} · {findings.length} components quantified
              </div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            <p style={{ marginTop: 0 }}>{data.scorecard.plain}</p>
            <ShareBar rows={data.scorecard.rows} />
            <table className="data-table" style={{ marginTop: 10 }}>
              <thead>
                <tr>
                  <th>component</th>
                  <th style={{ textAlign: "right" }}>share of variation</th>
                  <th>strength</th>
                </tr>
              </thead>
              <tbody>
                {data.scorecard.rows.map((r) => (
                  <tr key={r.component}>
                    <td>{r.component}</td>
                    <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                      {(r.share * 100).toFixed(1)}%
                    </td>
                    <td>
                      <span className="pill" style={{ background: BAND_TONE[r.reading]?.wash ?? "transparent" }}>
                        {r.reading}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: 11.5, color: "var(--ink-faint)", margin: "8px 0 0" }}>
              Strength is Hyndman's F_T / F_S: how much of the variation a component explains
              relative to the remainder. The remainder is not a signal, so it gets no band — what
              matters there is whether it is <em>white</em>, which the remainder card answers.
            </p>
          </div>
        </section>
      )}

      {grid && (
        <section className="card" style={{ marginTop: 12 }}>
          <div className="card-head">
            <span className="card-kind">Figure</span>
            <div>
              <div className="card-title">{grid.title}</div>
              <div className="card-plain">{grid.plain}</div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            <PlotlyFigure figure={grid.figure_json} altText={grid.alt_text} height={560} />
            <p style={{ fontSize: 11.5, color: "var(--ink-faint)", margin: "8px 0 0" }}>
              The four panels share a time axis: trend + seasonal + remainder adds back up to the
              observed series, which is what makes this a decomposition rather than four unrelated
              lines.
            </p>
          </div>
        </section>
      )}

      {findings.length > 0 && (
        <>
          <h2 style={{ margin: "18px 0 10px", fontSize: 14 }}>What each component says</h2>
          {findings.map((f) => (
            <FindingCard key={f.id} finding={f} />
          ))}
        </>
      )}

      {(blockers.length > 0 || recs.length > 0) && (
        <>
          <h2 style={{ margin: "18px 0 10px", fontSize: 14 }}>What this means for the model</h2>
          {blockers.map((r) => (
            <RuleCard key={r.id} rule={r} kind="blocker" />
          ))}
          {recs.map((r) => (
            <RuleCard key={r.id} rule={r} kind="note" />
          ))}
        </>
      )}

      {others.length > 0 && (
        <>
          <h2 style={{ margin: "18px 0 10px", fontSize: 14 }}>More views</h2>
          {others.map((f) => (
            <ResultCard key={f.id} result={f} />
          ))}
        </>
      )}

      <section className="card" style={{ marginTop: 12 }}>
        <div className="card-head">
          <span className="card-kind">Method</span>
          <div>
            <div className="card-title">Does the method change the answer?</div>
            <div className="card-plain">
              Run the same series under several methods and overlay the trends. Where they separate,
              the trend is a consequence of the method — the single most useful thing to know about
              decomposition.
            </div>
          </div>
          <div className="card-actions">
            <button className="btn btn-sm" onClick={() => void runCompare()} disabled={busy}>
              <IconRefresh size={13} /> Compare methods
            </button>
          </div>
        </div>
        {compare && (
          <div className="card-body" style={{ paddingTop: 0 }}>
            <p style={{ marginTop: 0 }}>
              Ran: <strong>{compare.ran.join(", ")}</strong>
              {compare.spread != null && (
                <>
                  {" "}· their trends differ by up to{" "}
                  <strong>{compare.spread.toFixed(3)}</strong> at a single date
                </>
              )}
              .
            </p>
            <table className="data-table">
              <thead>
                <tr>
                  <th>method</th>
                  <th style={{ textAlign: "right" }}>trend share</th>
                  <th style={{ textAlign: "right" }}>seasonal share</th>
                  <th style={{ textAlign: "right" }}>F_T</th>
                  <th style={{ textAlign: "right" }}>F_S</th>
                </tr>
              </thead>
              <tbody>
                {compare.rows.map((r) => (
                  <tr key={String(r.method)}>
                    <td>{String(r.method)}</td>
                    <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{fmtPct(r.trend_share)}</td>
                    <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{fmtPct(r.seasonal_share)}</td>
                    <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{fmt(r.F_T)}</td>
                    <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{fmt(r.F_S)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {compare.notes.length > 0 && (
              <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 12, color: "var(--ink-soft)" }}>
                {compare.notes.map((n) => <li key={n}>{n}</li>)}
              </ul>
            )}
          </div>
        )}
      </section>

      <p style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 14 }}>
        Every number above comes from <code>ts/decompose/</code>; every recommendation from{" "}
        <code>advisor/rules_decompose.py</code>.{" "}
        <button className="btn btn-sm btn-ghost" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Hide" : "Show"} all result blocks ({results.length})
        </button>
      </p>
      {showAll && results.map((r) => <ResultCard key={r.id} result={r} />)}
    </div>
  );
}

/* ------------------------------------------------------------------ parts */

function MethodBar({
  methods, method, model, period, busy, status, onRun, extraNote,
}: {
  methods: MethodOption[];
  method: string;
  model: "additive" | "multiplicative";
  period: number | null;
  busy: boolean;
  status: string;
  onRun: (m: string, mod: "additive" | "multiplicative") => void;
  extraNote?: unknown;
}) {
  const [chosen, setChosen] = useState(method);
  const [chosenModel, setChosenModel] = useState<"additive" | "multiplicative">(model);
  const chosenMeta = methods.find((m) => m.method === chosen);
  const available = methods.filter((m) => m.available);

  // The default is STL, which cannot run on a series with no usable seasonal period (yearly,
  // minutely, or fewer than two cycles). Without this, Step 4 opened with its primary "Run"
  // button disabled and the user had to notice a row of greyed-out alternatives. Pick the first
  // method the server says can actually run; the reason STL cannot is still printed below.
  useEffect(() => {
    if (methods.length === 0) return;
    const current = methods.find((m) => m.method === chosen);
    if (current?.available) return;
    const fallback = methods.find((m) => m.available);
    if (fallback && fallback.method !== chosen) setChosen(fallback.method);
  }, [methods, chosen]);

  return (
    <section className="card" aria-labelledby="method-bar">
      <div className="card-head">
        <span className="card-kind">Step 3</span>
        <div>
          <div className="card-title" id="method-bar">
            {status === "complete" ? `Decomposed with ${method} (${model})` : "Choose a method"}
          </div>
          <div className="card-plain">
            {period
              ? (methods.find((m) => m.method === "stl")?.available === false
                  ? `The calendar suggests a period of ${period}, but this series is too short for two full cycles, so STL and the other seasonal methods cannot run here. The available methods below do not need a season.`
                  : `Seasonal period ${period} from the calendar detected in Step 0.`)
              : "No seasonal period was detected, so only non-seasonal methods are offered."}
          </div>
        </div>
        <div className="card-actions">
          <button
            className="btn btn-sm btn-primary"
            onClick={() => onRun(chosen, chosenModel)}
            disabled={busy || (chosenMeta ? !chosenMeta.available : false)}
            title={chosenMeta && !chosenMeta.available ? chosenMeta.why_not : undefined}
          >
            <IconCheck size={13} /> {busy ? "Running…" : "Run"}
          </button>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
          {methods.map((m) => {
            const active = m.method === chosen;
            return (
              <button
                key={m.method}
                className={`btn btn-sm${active ? " btn-primary" : ""}`}
                disabled={busy || !m.available}
                title={m.available ? METHOD_BLURB[m.method] : m.why_not}
                onClick={() => setChosen(m.method)}
                style={!m.available ? { opacity: 0.55, cursor: "not-allowed" } : undefined}
              >
                {/* A method that cannot run must not still be labelled "recommended". */}
                {m.available ? m.label : m.label.replace(" (recommended)", "")}
              </button>
            );
          })}
        </div>
        {chosenMeta && (
          <p style={{ fontSize: 12, color: "var(--ink-soft)", margin: "8px 0 0" }}>
            {chosenMeta.available ? METHOD_BLURB[chosenMeta.method] : chosenMeta.why_not}
          </p>
        )}
        {available.length === 0 && methods.length > 0 && extraNote === null && (
          <p style={{ fontSize: 12, color: "var(--warn)", margin: "8px 0 0" }}>
            No method can run on this series yet — the reasons are above each button.
          </p>
        )}

        <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 12 }}>
          <span style={{ fontSize: 12, color: "var(--ink-faint)" }}>Model:</span>
          {(["additive", "multiplicative"] as const).map((m) => (
            <button
              key={m}
              className={`btn btn-sm${chosenModel === m ? " btn-primary" : ""}`}
              onClick={() => setChosenModel(m)}
              disabled={busy}
              title={
                m === "multiplicative"
                  ? "Requires every value to be positive — the app refuses rather than switching silently."
                  : "Components add up to the series."
              }
            >
              {m}
            </button>
          ))}
          {chosenModel === "multiplicative" && (
            <span style={{ fontSize: 11.5, color: "var(--ink-faint)" }}>
              needs all-positive data; refused otherwise, with the count of offending values
            </span>
          )}
        </div>
      </div>
    </section>
  );
}

function ShareBar({ rows }: { rows: { component: string; share: number; reading: string }[] }) {
  const tones = ["var(--brand)", "var(--warn)", "var(--ink-faint)"];
  return (
    <div
      style={{ display: "flex", height: 18, borderRadius: 9, overflow: "hidden", border: "1px solid var(--line)" }}
      role="img"
      aria-label={rows.map((r) => `${r.component} ${(r.share * 100).toFixed(1)}%`).join(", ")}
    >
      {rows.map((r, i) => (
        <div
          key={r.component}
          title={`${r.component}: ${(r.share * 100).toFixed(1)}%`}
          style={{
            width: `${Math.max(0, r.share) * 100}%`,
            background: tones[i % tones.length],
            opacity: 0.85,
          }}
        />
      ))}
    </div>
  );
}

function FindingCard({ finding }: { finding: DecompFinding }) {
  const tone = BAND_TONE[finding.band] ?? BAND_TONE.none;
  return (
    <article className="card" style={{ marginBottom: 10 }}>
      <div className="card-head">
        <span className="card-kind" style={{ background: tone.wash }}>{finding.name}</span>
        <div>
          <div className="card-title">
            {finding.statistic != null ? `${finding.name} — ${fmt(finding.statistic)}` : finding.name}
            {finding.p_value != null && (
              <span style={{ fontWeight: 400, color: "var(--ink-faint)", fontSize: 12 }}>
                {" "}· {pPhrase(finding.p_value)}
              </span>
            )}
          </div>
          <div className="card-plain">{finding.plain}</div>
        </div>
        <div className="card-actions">
          <span className="pill" style={{ background: tone.wash, color: tone.colour }}>{tone.label}</span>
        </div>
      </div>
      {(finding.caveats.length > 0 || finding.detail) && (
        <div className="card-body" style={{ paddingTop: 0 }}>
          {finding.detail && (
            <p style={{ fontSize: 12, color: "var(--ink-faint)", margin: "0 0 8px" }}>{finding.detail}</p>
          )}
          {finding.caveats.length > 0 && (
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: "var(--ink-soft)" }}>
              {finding.caveats.map((c) => (
                <li key={c} style={{ marginBottom: 3 }}>{c}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </article>
  );
}

function RuleCard({ rule, kind }: { rule: RuleOut; kind: "blocker" | "note" }) {
  return (
    <article className="card" style={{ marginBottom: 10 }}>
      <div className="card-head">
        <span className="card-kind" style={{ background: kind === "blocker" ? "var(--warn-wash)" : "var(--brand-wash)" }}>
          {kind === "blocker" ? "Must decide" : "Suggestion"}
        </span>
        <div>
          <div className="card-title">{rule.recommend}</div>
          <div className="card-plain">{rule.because}</div>
        </div>
        <div className="card-actions">
          <span className="pill" style={{ textTransform: "capitalize" }}>{rule.confidence}</span>
        </div>
      </div>
      {(rule.alternative || rule.risk_if_ignored) && (
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
            <dt style={{ color: "var(--ink-faint)" }}>Rule</dt>
            <dd style={{ margin: 0 }}><code>{rule.id}</code></dd>
          </dl>
        </div>
      )}
    </article>
  );
}

/* ------------------------------------------------------------------ utils */

function fmt(x: unknown): string {
  const n = typeof x === "number" ? x : Number(x);
  if (!Number.isFinite(n)) return "—";
  const a = Math.abs(n);
  if (a !== 0 && (a < 0.001 || a >= 10000)) return n.toExponential(2);
  return n.toFixed(3);
}

function fmtPct(x: unknown): string {
  const n = typeof x === "number" ? x : Number(x);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(1)}%`;
}


function report(e: unknown, pushToast: ReturnType<typeof useStore>["pushToast"], fallback: string) {
  pushToast(errorToastPayload(e, fallback));
}

export default DecomposePanel;
