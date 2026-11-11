/**
 * panels/SelectPanel.tsx — S11 (select) and S12 (diagnose) (§5.9, Part 13).
 *
 * Two screens in one file because they answer one question between them: *which model, and can I trust
 * it?* The order of the S11 panel is the argument:
 *
 *   1. **The verdict sentence first.** Not the table. A reader who sees a ranked table and stops has
 *      been told there is a winner; the sentence says whether the winner is separable from the runner-up,
 *      and that is the difference between a finding and a coin flip.
 *   2. **The shortlist**, with the fold spread beside every mean, because a mean without its spread is
 *      the number people quote and the number that misleads.
 *   3. **The information criteria**, in their own block, below the accuracy — that ordering is the
 *      spec's and the panel has to reflect it or it teaches the opposite.
 *   4. **The pairwise tests**, with their refusal conditions attached.
 *
 * S12 opens with the three-word verdict and puts the *remedy* next to every failing test. A diagnostics
 * screen that lists twelve p-values and no remedies is a screen nobody can act on.
 */

import { useCallback, useEffect, useState } from "react";
import { IconChart, IconRefresh } from "../components/Icons";
import { ApiError, api } from "../state/api";
import { pNumber, pPhrase } from "../state/format";
import { useStore } from "../state/store";
import { completeStageRun } from "./stageRun";
import type {
  DiagnosticBlock,
  DiagnosticFinding,
  SelectionReport,
  ShortlistEntry,
} from "../state/types";

const num = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(digits);

const pct = (v: number | null | undefined): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : `${(v * 100).toFixed(1)}%`;

const ADEQUACY_STYLE: Record<string, { label: string; colour: string; wash: string }> = {
  adequate: { label: "ADEQUATE", colour: "var(--ok)", wash: "var(--ok-wash)" },
  adequate_with_caveats: { label: "ADEQUATE WITH CAVEATS", colour: "var(--warn)", wash: "var(--warn-wash)" },
  inadequate: { label: "INADEQUATE — REVISE", colour: "var(--bad)", wash: "var(--bad-wash)" },
};

export function SelectPanel({ step }: { step: "S11" | "S12" }) {
  return step === "S11" ? <SelectionView /> : <DiagnosticsView />;
}

/* ------------------------------------------------------------------ *
 * S11 — selection
 * ------------------------------------------------------------------ */
function SelectionView() {
  const { sessionId, pushToast, setActiveStage, refresh } = useStore();
  const [report, setReport] = useState<SelectionReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [horizon, setHorizon] = useState(12);
  const [folds, setFolds] = useState(4);
  const [scheme, setScheme] = useState("expanding");
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await api.selection();
      if (r.status === "complete" && r.report) {
        setReport(r.report);
        setHorizon(r.report.horizon);
      }
    } catch {
      /* nothing stored yet — the empty state below says so */
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, sessionId]);

  const run = async () => {
    setBusy(true);
    try {
      await api.select({ horizon, folds, scheme });
      await completeStageRun(load, refresh, () =>
        pushToast({
          kind: "success",
          title: "Selection complete.",
          body: "Every model was refitted on each fold and scored on the data that followed it.",
        }),
      );
    } catch (e) {
      const err = e as ApiError;
      pushToast({ kind: "error", title: err.message, body: err.fix ?? err.code });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div className="card" style={{ padding: 14 }}>
        <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap" }}>
          <Field label="Forecast horizon" hint="also the length of each test block">
            <input
              type="number"
              min={1}
              max={120}
              value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))}
              style={{ width: 74 }}
            />
          </Field>
          <Field label="Folds" hint="3–5 is the usual range">
            <input
              type="number"
              min={2}
              max={10}
              value={folds}
              onChange={(e) => setFolds(Number(e.target.value))}
              style={{ width: 62 }}
            />
          </Field>
          <Field label="Window" hint="expanding uses every observation; sliding forgets the past">
            <select value={scheme} onChange={(e) => setScheme(e.target.value)}>
              <option value="expanding">expanding</option>
              <option value="sliding">sliding</option>
            </select>
          </Field>
          <button className="btn btn-primary" onClick={run} disabled={busy || !sessionId}>
            {busy ? "Cross-validating…" : report ? "Re-run selection" : "Run selection"}
          </button>
        </div>
        <p style={{ margin: "10px 0 0", fontSize: 12, color: "var(--ink-faint)", maxWidth: 780 }}>
          Every model is <strong>refitted</strong> on each fold's training window — this is why it takes
          as long as fitting everything again, and why the score means something the fit table's AICc
          does not.
        </p>
      </div>

      {!report && loaded ? (
        <div className="empty">
          <IconChart size={30} />
          <h2 style={{ marginTop: 12 }}>No selection yet</h2>
          <p style={{ maxWidth: 620 }}>
            Run it once at least two models are fitted in Step 10. Selection compares forecasts on data
            each model never saw; it does not create models.
          </p>
          <button className="btn" style={{ marginTop: 12 }} onClick={() => setActiveStage("S10")}>
            Go to the fits
          </button>
        </div>
      ) : null}

      {report ? (
        <>
          <WhyCard report={report} />
          <ShortlistTable report={report} />
          <PairwiseCard report={report} />
        </>
      ) : null}
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--ink-faint)" }}>
        {label}
      </span>
      {children}
      {hint ? <span style={{ fontSize: 11, color: "var(--ink-faint)" }}>{hint}</span> : null}
    </label>
  );
}

/**
 * The paragraph first. This is the panel's headline, not a footnote: the numbers below it are only
 * interpretable once the reader knows whether the winner is separable from the runner-up.
 */
function WhyCard({ report }: { report: SelectionReport }) {
  const top = report.shortlist[0];
  return (
    <section className="card" data-testid="selection-why" style={{ padding: 0 }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 14px",
          borderBottom: "1px solid var(--rule)",
        }}
      >
        <span className="pill pill-brand" style={{ fontSize: 10.5 }}>
          WHY THIS MODEL
        </span>
        <strong style={{ fontSize: 14 }} data-testid="selection-winner">
          {report.winner_name || report.winner}
        </strong>
        <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
          {report.indistinguishable ? (
            <span className="pill" style={{ background: "var(--warn-wash)", color: "var(--warn)", fontSize: 11 }}>
              tied with {report.shortlist[1]?.display_name ?? "the runner-up"}
            </span>
          ) : null}
          <span className="pill" style={{ fontSize: 11 }}>
            CV MASE {num(top?.cv_mase)}
          </span>
          <span className="pill" style={{ fontSize: 11 }}>
            {report.n_folds} folds · h={report.horizon} · {report.scheme}
          </span>
        </span>
      </header>
      <div style={{ padding: "12px 14px" }}>
        <p style={{ margin: 0, fontSize: 13.5, lineHeight: 1.55 }} data-testid="selection-narrative">
          {report.why}
        </p>
        <p
          style={{
            margin: "10px 0 0",
            fontSize: 12,
            color: "var(--ink-soft)",
            background: "var(--brand-wash)",
            padding: "8px 10px",
            borderRadius: 6,
          }}
        >
          {report.caveat}
        </p>
        {report.notes.length ? (
          <ul style={{ margin: "10px 0 0", paddingLeft: 18, fontSize: 12, color: "var(--ink-soft)" }}>
            {report.notes.map((n, i) => (
              <li key={i} style={{ marginBottom: 4 }}>
                {n}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  );
}

function ShortlistTable({ report }: { report: SelectionReport }) {
  const best = report.shortlist[0]?.cv_mase ?? null;
  return (
    <section className="card" data-testid="selection-shortlist" style={{ padding: 0, overflow: "hidden" }}>
      <header style={{ padding: "10px 14px", borderBottom: "1px solid var(--rule)" }}>
        <strong style={{ fontSize: 13.5 }}>Ranked shortlist — held-out accuracy first</strong>
        <span style={{ marginLeft: 8, fontSize: 12, color: "var(--ink-faint)" }}>
          the AICc columns are reference only, and blank where the comparison was refused
        </span>
      </header>
      <table className="data" style={{ width: "100%" }}>
        <thead>
          <tr>
            <th style={{ width: 28 }}>#</th>
            <th>Model</th>
            <th style={{ textAlign: "right" }}>CV MASE</th>
            <th style={{ textAlign: "right" }}>vs best</th>
            <th style={{ textAlign: "right" }}>fold spread</th>
            <th style={{ textAlign: "right" }}>worst fold</th>
            <th style={{ textAlign: "right" }}>CV MAE</th>
            <th style={{ textAlign: "right" }}>AICc</th>
            <th style={{ textAlign: "right" }}>ΔAICc</th>
            <th style={{ textAlign: "right" }}>Akaike w</th>
          </tr>
        </thead>
        <tbody>
          {report.shortlist.map((e: ShortlistEntry) => {
            const rel = e.cv_mase !== null && best ? (e.cv_mase / best - 1) * 100 : null;
            return (
              <tr key={e.model_id} data-testid={`select-row-${e.model_id}`}>
                <td style={{ color: "var(--ink-faint)" }}>{e.rank}</td>
                <td>
                  <strong style={{ fontSize: 12.5 }}>{e.display_name}</strong>
                  {e.notes.length ? (
                    <div style={{ fontSize: 11, color: "var(--ink-faint)", marginTop: 2, maxWidth: 430 }}>
                      {e.notes[0]}
                    </div>
                  ) : null}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)" }}>
                  {e.rank === 1 ? <strong>{num(e.cv_mase)}</strong> : num(e.cv_mase)}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", color: "var(--ink-faint)" }}>
                  {rel === null ? "—" : rel === 0 ? "best" : `+${rel.toFixed(1)}%`}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", color: "var(--ink-soft)" }}>
                  {num(e.cv_spread)}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", color: "var(--ink-soft)" }}>
                  {num(e.worst_fold_mase)}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)" }}>{num(e.cv_mae, 2)}</td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", color: "var(--ink-faint)" }}>
                  {num(e.aicc, 2)}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", color: "var(--ink-faint)" }}>
                  {num(e.delta_aicc, 2)}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", color: "var(--ink-faint)" }}>
                  {e.akaike_weight === null ? "—" : e.akaike_weight.toFixed(1)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p style={{ margin: 0, padding: "8px 14px", fontSize: 11.5, color: "var(--ink-faint)" }}>
        <strong>fold spread</strong> is the standard deviation of MASE across folds — read it before the
        mean. A model whose spread is larger than its mean has not been measured; it has been sampled.
        {report.ic_comparable ? "" : ` ${report.ic_reason}`}
      </p>
    </section>
  );
}

function PairwiseCard({ report }: { report: SelectionReport }) {
  const { dm, mz } = report;
  if (!dm && !mz) return null;
  return (
    <section className="card" data-testid="selection-pairwise" style={{ padding: 0 }}>
      <header style={{ padding: "10px 14px", borderBottom: "1px solid var(--rule)" }}>
        <strong style={{ fontSize: 13.5 }}>Is the difference real?</strong>
      </header>
      <div style={{ padding: "12px 14px", display: "flex", flexDirection: "column", gap: 12 }}>
        {dm ? (
          <div>
            <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 4 }}>
              {dm.test === "diebold-mariano" ? "Diebold–Mariano" : "Clark–West (nested)"}
              {dm.ok ? (
                <span
                  className="pill"
                  style={{
                    marginLeft: 8,
                    fontSize: 10.5,
                    background: (dm.p_value ?? 1) < 0.05 ? "var(--ok-wash)" : "var(--warn-wash)",
                    color: (dm.p_value ?? 1) < 0.05 ? "var(--ok)" : "var(--warn)",
                  }}
                >
                  {pPhrase(dm.p_value)}
                </span>
              ) : (
                <span className="pill" style={{ marginLeft: 8, fontSize: 10.5 }}>
                  not run
                </span>
              )}
            </div>
            <p style={{ margin: 0, fontSize: 12.5, color: "var(--ink-soft)" }}>{dm.note}</p>
            {dm.ok && dm.assumptions.length ? (
              <ul style={{ margin: "6px 0 0", paddingLeft: 18, fontSize: 11.5, color: "var(--ink-faint)" }}>
                {dm.assumptions.map((a, i) => (
                  <li key={i}>{a}</li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
        {mz ? (
          <div style={{ borderTop: "1px solid var(--rule)", paddingTop: 10 }}>
            <div style={{ fontSize: 12.5, fontWeight: 600, marginBottom: 4 }}>
              Mincer–Zarnowitz
              <span style={{ marginLeft: 8, fontFamily: "var(--mono)", fontSize: 11.5, fontWeight: 400 }}>
                intercept {num(mz.intercept)} · slope {num(mz.slope)} · joint p {pNumber(mz.joint_p, 4)}
              </span>
            </div>
            <p style={{ margin: 0, fontSize: 12.5, color: "var(--ink-soft)" }}>{mz.verdict}</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ *
 * S12 — diagnostics
 * ------------------------------------------------------------------ */
function DiagnosticsView() {
  const { sessionId, pushToast, setActiveStage, refresh } = useStore();
  const [blocks, setBlocks] = useState<Record<string, DiagnosticBlock>>({});
  const [busy, setBusy] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    try {
      const r = await api.diagnostics();
      if (r.status === "complete") {
        setBlocks(r.blocks);
        const first = Object.values(r.blocks).find((b) => b.verdict?.adequacy === "inadequate");
        setOpen((cur) => cur ?? first?.model_id ?? Object.keys(r.blocks)[0] ?? null);
      }
    } catch {
      /* not run yet */
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, sessionId]);

  const run = async () => {
    setBusy(true);
    try {
      await api.diagnose({});
      await completeStageRun(load, refresh, () =>
        pushToast({ kind: "success", title: "Diagnostics complete." }),
      );
    } catch (e) {
      const err = e as ApiError;
      pushToast({ kind: "error", title: err.message, body: err.fix ?? err.code });
    } finally {
      setBusy(false);
    }
  };

  const entries = Object.values(blocks);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <div className="card" style={{ padding: 14, display: "flex", alignItems: "center", gap: 12 }}>
        <button className="btn btn-primary" onClick={run} disabled={busy || !sessionId}>
          {busy ? "Testing residuals…" : entries.length ? "Re-run diagnostics" : "Run diagnostics"}
        </button>
        <span style={{ fontSize: 12, color: "var(--ink-faint)", maxWidth: 700 }}>
          Each model's residuals are put through ten checks: autocorrelation at four lags, Box–Pierce,
          three normality tests, ARCH-LM, runs, signs, CUSUM and an outlier scan. The verdict names the
          failing test and what to do about it.
        </span>
      </div>

      {!entries.length && loaded ? (
        <div className="empty">
          <IconRefresh size={30} />
          <h2 style={{ marginTop: 12 }}>No diagnostics yet</h2>
          <p style={{ maxWidth: 620 }}>
            Diagnostics read the residuals of fitted models, so Step 10 has to have run first.
          </p>
          <button className="btn" style={{ marginTop: 12 }} onClick={() => setActiveStage("S10")}>
            Go to the fits
          </button>
        </div>
      ) : null}

      {entries.map((b) => (
        <VerdictCard
          key={b.model_id}
          block={b}
          open={open === b.model_id}
          onToggle={() => setOpen(open === b.model_id ? null : b.model_id)}
        />
      ))}
    </div>
  );
}

function VerdictCard({
  block,
  open,
  onToggle,
}: {
  block: DiagnosticBlock;
  open: boolean;
  onToggle: () => void;
}) {
  const v = block.verdict;
  if (block.status !== "ok" || !v) {
    return (
      <section className="card" data-testid={`diag-${block.model_id}`} style={{ padding: "10px 14px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <strong style={{ fontSize: 13 }}>{block.display_name}</strong>
          <span className="pill" style={{ fontSize: 10.5 }}>
            not applicable
          </span>
        </div>
        <p style={{ margin: "6px 0 0", fontSize: 12.5, color: "var(--ink-soft)" }}>{block.message}</p>
      </section>
    );
  }
  const style = ADEQUACY_STYLE[v.adequacy] ?? ADEQUACY_STYLE.adequate_with_caveats;
  return (
    <section className="card" data-testid={`diag-${block.model_id}`} style={{ padding: 0 }}>
      <header
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          padding: "10px 14px",
          borderBottom: "1px solid var(--rule)",
          background: style.wash,
        }}
      >
        <strong style={{ fontSize: 13.5 }}>{block.display_name}</strong>
        <span
          className="pill"
          data-testid={`diag-verdict-${block.model_id}`}
          style={{ background: style.colour, color: "#fff", fontSize: 10.5, borderColor: "transparent" }}
        >
          {style.label}
        </span>
        <span style={{ fontSize: 11.5, color: "var(--ink-soft)" }}>
          {v.n_tests_run} checks · {v.n_rejected} rejected · {v.expected_false_positives} expected by chance
        </span>
        <button className="btn" style={{ marginLeft: "auto", fontSize: 11.5 }} onClick={onToggle}>
          {open ? "Hide the checks" : "Show the checks"}
        </button>
      </header>
      <div style={{ padding: "12px 14px" }}>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.55 }} data-testid={`diag-headline-${block.model_id}`}>
          {v.headline}
        </p>
        {v.reasons.length ? (
          <ul style={{ margin: "10px 0 0", paddingLeft: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 8 }}>
            {v.reasons.map((f: DiagnosticFinding, i) => (
              <li
                key={i}
                style={{
                  borderLeft: `3px solid ${
                    f.severity === "high" ? "var(--bad)" : f.severity === "medium" ? "var(--warn)" : "var(--rule)"
                  }`,
                  paddingLeft: 10,
                }}
              >
                <div style={{ fontSize: 12.5, fontWeight: 600 }}>
                  {f.test}
                  <span style={{ marginLeft: 8, fontSize: 10.5, fontWeight: 400, color: "var(--ink-faint)" }}>
                    {f.kind === "distributional" ? "affects the intervals, not the point forecast" : "structural"}
                  </span>
                </div>
                <div style={{ fontSize: 12.5, color: "var(--ink-soft)" }}>{f.detail}</div>
                <div style={{ fontSize: 12.5, color: "var(--brand)", marginTop: 2 }}>
                  <strong>Do this:</strong> {f.remedy}
                </div>
              </li>
            ))}
          </ul>
        ) : null}
        {open ? <BatteryDetail block={block} /> : null}
        {v.notes.length ? (
          <ul style={{ margin: "10px 0 0", paddingLeft: 18, fontSize: 11.5, color: "var(--ink-faint)" }}>
            {v.notes.map((n, i) => (
              <li key={i}>{n}</li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  );
}

/** The numbers behind the verdict. Never shown by default — the verdict is the headline. */
function BatteryDetail({ block }: { block: DiagnosticBlock }) {
  const bat = block.battery;
  if (!bat) return null;
  const lb = Object.entries(bat.ljung_box?.results ?? {});
  return (
    <div style={{ marginTop: 12, borderTop: "1px solid var(--rule)", paddingTop: 10 }}>
      <table className="data" style={{ width: "100%", fontSize: 11.5 }}>
        <tbody>
          <tr>
            <td style={{ color: "var(--ink-faint)", width: 190 }}>Ljung–Box p by lag</td>
            <td style={{ fontFamily: "var(--mono)" }}>
              {lb.length
                ? lb.map(([lag, r]) => `${lag}: ${pNumber(r.p, 3)}`).join("  ·  ")
                : "—"}
            </td>
          </tr>
          <tr>
            <td style={{ color: "var(--ink-faint)" }}>Box–Pierce</td>
            <td style={{ fontFamily: "var(--mono)" }}>Q {num(bat.box_pierce?.q, 2)} · p {pNumber(bat.box_pierce?.p, 3)}</td>
          </tr>
          <tr>
            <td style={{ color: "var(--ink-faint)" }}>Jarque–Bera / Shapiro / Anderson–Darling</td>
            <td style={{ fontFamily: "var(--mono)" }}>
              {pNumber(bat.normality?.jarque_bera?.p, 3)} · {pNumber(bat.normality?.shapiro?.p, 3)} ·{" "}
              {pNumber(bat.normality?.anderson_darling?.p, 3)}
            </td>
          </tr>
          <tr>
            <td style={{ color: "var(--ink-faint)" }}>ARCH-LM</td>
            <td style={{ fontFamily: "var(--mono)" }}>p {pNumber(bat.arch_lm?.p, 3)}</td>
          </tr>
          <tr>
            <td style={{ color: "var(--ink-faint)" }}>Runs / signs</td>
            <td style={{ fontFamily: "var(--mono)" }}>
              {bat.runs?.runs ?? "—"} vs {num(bat.runs?.expected, 1)} expected · p {pNumber(bat.runs?.p, 3)} ·{" "}
              {bat.signs?.positive ?? "—"}/{bat.signs?.n ?? "—"} positive · p {pNumber(bat.signs?.p, 3)}
            </td>
          </tr>
          <tr>
            <td style={{ color: "var(--ink-faint)" }}>CUSUM</td>
            <td style={{ fontFamily: "var(--mono)" }}>
              {bat.cusum?.breach ? "crosses the 5% band" : "inside the 5% band"}
            </td>
          </tr>
        </tbody>
      </table>
      <p style={{ margin: "8px 0 0", fontSize: 11.5, color: "var(--ink-faint)" }}>{bat.arch_lm?.advice}</p>
      {bat.outliers?.length ? (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 11.5, fontWeight: 600, marginBottom: 4 }}>
            Residual outliers ({bat.outliers.length})
          </div>
          <table className="data" style={{ width: "100%", fontSize: 11.5 }}>
            <thead>
              <tr>
                <th>When</th>
                <th style={{ textAlign: "right" }}>residual</th>
                <th style={{ textAlign: "right" }}>robust z</th>
                <th style={{ textAlign: "right" }}>share of SSE</th>
                <th>reading</th>
              </tr>
            </thead>
            <tbody>
              {bat.outliers.map((o, i) => (
                <tr key={i}>
                  <td>{o.when ?? `pos ${o.position}`}</td>
                  <td style={{ textAlign: "right", fontFamily: "var(--mono)" }}>{num(o.residual, 2)}</td>
                  <td style={{ textAlign: "right", fontFamily: "var(--mono)" }}>{num(o.z, 1)}</td>
                  <td style={{ textAlign: "right", fontFamily: "var(--mono)" }}>{pct(o.influence)}</td>
                  <td style={{ color: "var(--ink-soft)" }}>{o.reading}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
