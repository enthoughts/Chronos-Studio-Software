/**
 * panels/CorrelatePanel.tsx — S6 (ACF / PACF) and S7 ("what next") in one place.
 *
 * The two stages share a screen because they are one conversation: the correlogram is the evidence
 * and the ranked list is the conclusion, and separating them across two steps makes the user hold
 * the first in their head while reading the second.
 *
 * Layout, in the order the questions get asked:
 *
 *   1. **What shape is this?** — one badge with the automatic reading, plus its caveats. The label
 *      is always "consistent with", never "this is", because a correlogram cannot decide anything.
 *   2. **The pictures** — ACF/PACF with both bands, then the same correlogram after each
 *      transformation. The second is where over-differencing becomes visible.
 *   3. **The numbers** — Ljung–Box at several lags (including the seasonal one), the variance
 *      ratio, and the candidate orders with their counter-evidence.
 *   4. **What to do next** — the ranked list, each item showing its preview verdict. Nothing is
 *      applied without a preview, and the preview says "worse" in as many words when it is.
 *
 * Nothing on this screen writes to the series except the Apply buttons, which go through the same
 * versioned/undoable path as Step 2.
 */

import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { IconCheck, IconRefresh } from "../components/Icons";
import { PlotlyFigure } from "../charts/PlotlyFigure";
import { ResultCard } from "../components/ResultCard";
import { api } from "../state/api";
import { pNumber, pPhrase } from "../state/format";
import { useStore } from "../state/store";
import { errorToastPayload } from "../copy/errors";
import { applyGate, overrideResetKey } from "./nextAction";
import type {
  ActionPreview,
  AnyResult,
  CorrelateResponse,
  FigureResult,
  NextRecommendation,
  NextResponse,
  OrderHint,
} from "../state/types";

const CONFIDENCE_TONE: Record<string, { colour: string; wash: string }> = {
  high: { colour: "var(--brand)", wash: "var(--brand-wash)" },
  medium: { colour: "var(--warn)", wash: "var(--warn-wash)" },
  moderate: { colour: "var(--warn)", wash: "var(--warn-wash)" },
  low: { colour: "var(--ink-faint)", wash: "var(--stale-wash)" },
};

const VERDICT_TONE: Record<string, { label: string; colour: string; wash: string }> = {
  better: { label: "helps", colour: "var(--brand)", wash: "var(--brand-wash)" },
  neutral: { label: "no clear change", colour: "var(--ink-soft)", wash: "var(--stale-wash)" },
  worse: { label: "makes it worse", colour: "var(--bad, #c0392b)", wash: "var(--bad-wash, #fdecea)" },
  unavailable: { label: "cannot preview", colour: "var(--ink-faint)", wash: "transparent" },
};

export function CorrelatePanel() {
  const { sessionId, session, refresh, pushToast } = useStore();
  const [data, setData] = useState<CorrelateResponse | null>(null);
  const [next, setNext] = useState<NextResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [preview, setPreview] = useState<{ ruleId: string; value: ActionPreview } | null>(null);
  const [showAll, setShowAll] = useState(false);

  const hasData = Boolean(session?.dataset);
  const s5done = session?.pipeline.stages.S5?.status === "complete";
  const s6status = session?.pipeline.stages.S6?.status ?? "not_started";

  const load = useCallback(async () => {
    if (!sessionId || !hasData) return;
    const [c, n] = await Promise.all([
      api.getCorrelate(),
      s5done ? api.getNext().catch(() => null) : Promise.resolve(null),
    ]);
    setData(c);
    setNext(n);
    // Computing S6 marks the stage complete *server-side*, so the stepper and the "Status" card
    // are only correct once the pipeline is re-read. Without this the panel shows results while
    // the sidebar still says "Not started", which reads as though the step failed.
    await refresh();
  }, [sessionId, hasData, s5done, refresh]);

  useEffect(() => {
    void load().catch((e) => report(e, pushToast, "The correlogram could not be loaded."));
  }, [load, pushToast]);

  const runPreview = useCallback(
    async (rec: NextRecommendation) => {
      if (!rec.op || rec.op === "undo") {
        setPreview(null);
        return;
      }
      setBusy(true);
      try {
        const p = await api.previewAction({
          op: rec.op as "difference" | "transform.log",
          order: (rec.op_params?.order as number | "seasonal" | undefined) ?? 1,
          rule_id: rec.id,
        });
        setPreview({ ruleId: rec.id, value: p as ActionPreview });
      } catch (e) {
        report(e, pushToast, "The preview could not be run.");
      } finally {
        setBusy(false);
      }
    },
    [pushToast],
  );

  const apply = useCallback(
    async (rec: NextRecommendation) => {
      if (!rec.op || rec.op === "undo") {
        pushToast({
          kind: "info",
          title: rec.op === "undo" ? "Undo lives in Step 2." : "Nothing to apply for this item.",
          body: rec.op === "undo"
            ? "Open Step 2 (Prepare) and press Undo — that path restores the previous version."
            : "This recommendation is advice about the model, not a change to the data.",
        });
        return;
      }
      setBusy(true);
      try {
        await api.applyNext({
          op: rec.op as "difference" | "transform.log",
          order: (rec.op_params?.order as number | "seasonal" | undefined) ?? 1,
          rule_id: rec.id,
          reason: rec.because.slice(0, 200),
        });
        await refresh();
        setPreview(null);
        await load();
        pushToast({
          kind: "success",
          title: "Applied.",
          body: "The series has changed, so Steps 1–6 are now stale — re-run them to see the new diagnostics.",
        });
      } catch (e) {
        report(e, pushToast, "That change could not be applied.");
      } finally {
        setBusy(false);
      }
    },
    [refresh, load, pushToast],
  );

  const results = useMemo<AnyResult[]>(() => data?.results ?? [], [data]);
  // Narrowed to FigureResult so `figure_json` is type-safe: `AnyResult` is a union and a table has
  // no figure, so `find` alone would let a table through at compile time and blow up at runtime.
  const figures = results.filter((r): r is FigureResult => r.kind === "figure");
  const gridFig = figures.find((r) => r.id === "fig.acf_pacf");
  const cmpFig = figures.find((r) => r.id === "fig.acf_comparison");

  if (!hasData) {
    return (
      <div className="callout callout-warn" style={{ marginTop: 0 }}>
        <strong>No data yet.</strong> Load a file or an example in Step 0 — autocorrelation needs a
        series.
      </div>
    );
  }

  const pattern = data?.pattern;

  return (
    <div>
      {/* ---------- 1. the reading ---------- */}
      {pattern && (
        <section
          className="card"
          style={{ borderLeft: "3px solid var(--brand)" }}
          aria-labelledby="pattern-head"
        >
          <div className="card-head">
            <span className="card-kind">Reading</span>
            <div>
              <div className="card-title" id="pattern-head">
                {pattern.label}
              </div>
              <div className="card-plain">{pattern.plain}</div>
            </div>
            <div className="card-actions">
              <span
                className="pill"
                style={{
                  background: CONFIDENCE_TONE[pattern.confidence]?.wash,
                  color: CONFIDENCE_TONE[pattern.confidence]?.colour,
                }}
              >
                {pattern.confidence} confidence
              </span>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            {pattern.suggests && (
              <p style={{ margin: "0 0 8px", fontSize: 12.5 }}>
                <strong>What it suggests:</strong> {pattern.suggests}
              </p>
            )}
            {pattern.caveats.length > 0 && (
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: "var(--ink-soft)" }}>
                {pattern.caveats.map((c) => (
                  <li key={c} style={{ marginBottom: 3 }}>{c}</li>
                ))}
              </ul>
            )}
            {data.stationarity && (
              <p style={{ margin: "10px 0 0", fontSize: 12.5, color: "var(--ink-soft)" }}>
                <strong>Step 5 says:</strong>{" "}
                {data.stationarity.verdict.replace(/_/g, " ")} ({data.stationarity.confidence}). The
                correlogram above is a description; the verdict is what decides{" "}
                <code>d</code> and <code>D</code>.
              </p>
            )}
            {!data.stationarity && (
              <p style={{ margin: "10px 0 0", fontSize: 12.5, color: "var(--warn)" }}>
                Step 5 has not run yet. The correlogram cannot tell a unit root from a trend or a
                break, so the "what next" advice below stays unavailable until it does.
              </p>
            )}
          </div>
        </section>
      )}

      {/* ---------- 2. the pictures ---------- */}
      {gridFig && (
        <section className="card" style={{ marginTop: 12 }}>
          <div className="card-head">
            <span className="card-kind">Figure</span>
            <div>
              <div className="card-title">{gridFig.title}</div>
              <div className="card-plain">{gridFig.plain}</div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            <PlotlyFigure figure={gridFig.figure_json} altText={gridFig.alt_text} height={420} />
          </div>
        </section>
      )}

      {cmpFig && (
        <section className="card" style={{ marginTop: 12 }}>
          <div className="card-head">
            <span className="card-kind">Figure</span>
            <div>
              <div className="card-title">{cmpFig.title}</div>
              <div className="card-plain">{cmpFig.plain}</div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            <PlotlyFigure figure={cmpFig.figure_json} altText={cmpFig.alt_text} height={560} />
            <p style={{ fontSize: 11.5, color: "var(--ink-faint)", margin: "8px 0 0" }}>
              Look for a lag-1 bar swinging strongly negative in the middle or bottom row — that is
              over-differencing, and it is easier to see here than in any table.
            </p>
          </div>
        </section>
      )}

      {/* ---------- 3. the numbers ---------- */}
      {data && (
        <>
          <h2 style={{ margin: "18px 0 10px", fontSize: 14 }}>The tests behind that reading</h2>

          <section className="card">
            <div className="card-head">
              <span className="card-kind">Ljung–Box</span>
              <div>
                <div className="card-title">Is there any dependence left at all?</div>
                <div className="card-plain">
                  Reported at several lags on purpose: a single test at lag 10 cannot see a seasonal
                  spike at lag {data.period ?? "s"}.
                </div>
              </div>
            </div>
            <div className="card-body" style={{ paddingTop: 0 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>up to lag</th>
                    <th style={{ textAlign: "right" }}>Ljung–Box Q</th>
                    <th style={{ textAlign: "right" }}>df</th>
                    <th style={{ textAlign: "right" }}>p-value</th>
                    <th>reading</th>
                  </tr>
                </thead>
                <tbody>
                  {data.ljung_box.map((row) => (
                    <tr key={row.lag}>
                      <td>{row.lag}{row.lag === data.period ? " (seasonal)" : ""}</td>
                      <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                        {row.statistic.toFixed(2)}
                      </td>
                      <td style={{ textAlign: "right" }}>{row.df}</td>
                      <td style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>
                        {pNumber(row.p_value)}
                      </td>
                      <td style={{ fontSize: 12, color: "var(--ink-soft)" }}>
                        {row.rejects
                          ? "autocorrelation remains"
                          : "no evidence of remaining autocorrelation"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {data.variance_ratio && (
            <section className="card" style={{ marginTop: 10 }}>
              <div className="card-head">
                <span className="card-kind">Variance ratio</span>
                <div>
                  <div className="card-title">
                    VR(2) = {data.variance_ratio.ratio.toFixed(3)}
                    <span style={{ fontWeight: 400, fontSize: 12, color: "var(--ink-faint)" }}>
                      {" "}· {pPhrase(data.variance_ratio.p_value)}
                    </span>
                  </div>
                  <div className="card-plain">{data.variance_ratio.reading}</div>
                </div>
              </div>
            </section>
          )}

          {data.order_hints.length > 0 && (
            <section className="card" style={{ marginTop: 10 }}>
              <div className="card-head">
                <span className="card-kind">Orders</span>
                <div>
                  <div className="card-title">Candidate model orders — starting points</div>
                  <div className="card-plain">
                    The smallest models consistent with the shape. Each one carries what supports it
                    and what would have to be checked; a hint without its counter-evidence is a
                    guess.
                  </div>
                </div>
              </div>
              <div className="card-body" style={{ paddingTop: 0 }}>
                {data.order_hints.map((h: OrderHint, i: number) => (
                  <div
                    key={`${h.notation}-${i}`}
                    style={{
                      borderTop: i ? "1px solid var(--line)" : undefined,
                      paddingTop: i ? 10 : 0,
                      marginTop: i ? 10 : 0,
                    }}
                  >
                    <div style={{ fontWeight: 600, fontSize: 13 }}>
                      ARIMA<span style={{ fontFamily: "monospace" }}>{h.notation}</span>
                    </div>
                    <p style={{ margin: "4px 0 6px", fontSize: 12.5 }}>{h.plain}</p>
                    <dl
                      style={{
                        display: "grid",
                        gridTemplateColumns: "max-content 1fr",
                        gap: "4px 12px",
                        margin: 0,
                        fontSize: 12,
                      }}
                    >
                      <dt style={{ color: "var(--brand)" }}>Support</dt>
                      <dd style={{ margin: 0 }}>{h.support.join(" · ")}</dd>
                      <dt style={{ color: "var(--warn)" }}>Check</dt>
                      <dd style={{ margin: 0 }}>{h.against.join(" · ")}</dd>
                    </dl>
                  </div>
                ))}
                <p style={{ fontSize: 11.5, color: "var(--ink-faint)", margin: "10px 0 0" }}>
                  <code>d</code> and <code>D</code> come from Step 5's tests, never from this plot.
                </p>
              </div>
            </section>
          )}
        </>
      )}

      {/* ---------- 4. what next ---------- */}
      <h2 style={{ margin: "18px 0 10px", fontSize: 14 }}>What to do next</h2>

      {!s5done && (
        <div className="callout callout-warn" style={{ marginTop: 0 }}>
          <strong>Run Step 5 first.</strong> The advice is ranked against the stationarity verdict,
          and without it the app would be guessing whether a slow decay is a unit root, a trend or a
          break.
        </div>
      )}

      {next && (
        <>
          <section
            className="card"
            style={{ borderLeft: "3px solid var(--brand)", marginBottom: 12 }}
          >
            <div className="card-head">
              <span className="card-kind">Verdict</span>
              <div>
                <div className="card-title">{next.headline}</div>
                <div className="card-plain">
                  {next.ranked.length} applicable recommendation
                  {next.ranked.length === 1 ? "" : "s"}, best first. Every one is ranked against
                  doing nothing — and "nothing" is allowed to win.
                </div>
              </div>
              <div className="card-actions">
                <button className="btn btn-sm" onClick={() => void load()} disabled={busy}>
                  <IconRefresh size={13} /> Re-run
                </button>
              </div>
            </div>
          </section>

          {next.ranked.map((rec, i) => (
            <RecommendationCard
              key={rec.id}
              rec={rec}
              first={i === 0}
              busy={busy}
              preview={preview?.ruleId === rec.id ? preview.value : rec.preview}
              onPreview={() => void runPreview(rec)}
              onApply={() => void apply(rec)}
            />
          ))}

          {next.ranked.length === 0 && (
            <div className="empty">
              <h2>Nothing forced a change</h2>
              <p>
                No rule applies to this series. The safest next step is the simplest model,
                compared out of sample against the benchmarks.
              </p>
            </div>
          )}
        </>
      )}

      <p style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 14 }}>
        Reading from <code>ts/correlate/</code>; recommendations from{" "}
        <code>advisor/rules_next.py</code>.{" "}
        <button className="btn btn-sm btn-ghost" onClick={() => setShowAll((v) => !v)}>
          {showAll ? "Hide" : "Show"} all result blocks ({results.length})
        </button>
      </p>
      {showAll && results.map((r) => <ResultCard key={r.id} result={r} />)}
      {s6status === "stale" && (
        <p style={{ fontSize: 11.5, color: "var(--warn)", marginTop: 8 }}>
          The series has changed since this correlogram was computed — press Re-run.
        </p>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ parts */

function RecommendationCard({
  rec,
  first,
  busy,
  preview,
  onPreview,
  onApply,
}: {
  rec: NextRecommendation;
  first: boolean;
  busy: boolean;
  preview: ActionPreview | null | undefined;
  onPreview: () => void;
  onApply: () => void;
}) {
  const tone = CONFIDENCE_TONE[rec.confidence] ?? CONFIDENCE_TONE.medium;
  const pv = preview ? VERDICT_TONE[preview.verdict] : null;
  const worse = preview?.verdict === "worse";
  // The server attaches a preview to every ranked action, so a "worse" verdict
  // used to lock Apply before the user could even read the card. Require an
  // explicit acknowledgement instead of a dead end.
  const [overrideWorse, setOverrideWorse] = useState(false);
  const resetKey = overrideResetKey(rec, preview);
  useEffect(() => {
    setOverrideWorse(false);
  }, [resetKey]);
  const gate = applyGate(rec, preview, overrideWorse);
  const lockId = useId();

  return (
    <article className="card" style={{ marginBottom: 10, borderLeft: first ? "2px solid var(--brand)" : undefined }}>
      <div className="card-head">
        <span
          className="card-kind"
          style={{
            background: rec.kind === "protect" ? "var(--warn-wash)" : "var(--brand-wash)",
          }}
        >
          {rec.kind === "protect" ? "Guardrail" : rec.kind === "note" ? "Note" : "Suggestion"}
        </span>
        <div>
          <div className="card-title">
            {rec.recommend}
            {pv && (
              <span
                className="pill"
                style={{ marginLeft: 8, background: pv.wash, color: pv.colour }}
              >
                preview: {pv.label}
              </span>
            )}
          </div>
          <div className="card-plain">{rec.because}</div>
        </div>
        <div className="card-actions">
          <span className="pill" style={{ background: tone.wash, color: tone.colour }}>
            {rec.confidence}
          </span>
        </div>
      </div>

      <div className="card-body" style={{ paddingTop: 0 }}>
        {preview && (
          <div
            className="callout"
            style={{
              margin: "0 0 10px",
              borderLeft: `3px solid ${VERDICT_TONE[preview.verdict].colour}`,
              background: VERDICT_TONE[preview.verdict].wash,
            }}
          >
            <strong>Preview{preview.label ? ` (${preview.label})` : ""}:</strong> {preview.plain}
            {preview.after && preview.before && (
              <div style={{ fontSize: 11.5, marginTop: 6, color: "var(--ink-soft)" }}>
                ACF significant through lag {preview.before.decay} → {preview.after.decay} · r₁{" "}
                {preview.before.r1.toFixed(3)} → {preview.after.r1.toFixed(3)} · {pPhrase(preview.before.lb_p)} → {pPhrase(preview.after.lb_p)}
                {preview.n_lost ? ` · ${preview.n_lost} observation(s) lost` : ""}
              </div>
            )}
            {preview.pattern_after_label && (
              <div style={{ fontSize: 11.5, marginTop: 4, color: "var(--ink-soft)" }}>
                After: <em>{preview.pattern_after_label}</em>
              </div>
            )}
          </div>
        )}

        <dl
          style={{
            display: "grid",
            gridTemplateColumns: "max-content 1fr",
            gap: "6px 12px",
            margin: 0,
            fontSize: 12.5,
          }}
        >
          {rec.alternative && (
            <>
              <dt style={{ color: "var(--ink-faint)" }}>Alternative</dt>
              <dd style={{ margin: 0 }}>{rec.alternative}</dd>
            </>
          )}
          {rec.risk_if_ignored && (
            <>
              <dt style={{ color: "var(--ink-faint)" }}>If ignored</dt>
              <dd style={{ margin: 0 }}>{rec.risk_if_ignored}</dd>
            </>
          )}
          <dt style={{ color: "var(--ink-faint)" }}>Rule</dt>
          <dd style={{ margin: 0 }}>
            <code>{rec.id}</code>
          </dd>
        </dl>

        <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
          {rec.op && rec.op !== "undo" && (
            <>
              <button className="btn btn-sm" onClick={onPreview} disabled={busy}>
                Preview this
              </button>
              <button
                className="btn btn-sm btn-primary"
                data-testid="apply-recommendation"
                onClick={onApply}
                disabled={busy || !gate.canApply}
                title={
                  gate.requiresOverride
                    ? "The preview says this makes the correlogram worse. Read it, then tick the override box to apply anyway."
                    : undefined
                }
              >
                <IconCheck size={13} /> Apply
              </button>
            </>
          )}
          {worse && (
            <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 11.5,
                            color: VERDICT_TONE.worse.colour, alignSelf: "center" }}>
              <input
                type="checkbox"
                data-testid="override-worse"
                checked={overrideWorse}
                onChange={(e) => setOverrideWorse(e.target.checked)}
                aria-describedby={gate.requiresOverride ? lockId : undefined}
              />
              I read the preview — apply anyway
            </label>
          )}
          {gate.requiresOverride && (
            <span
              id={lockId}
              data-testid="apply-locked-message"
              style={{ fontSize: 11.5, color: VERDICT_TONE.worse.colour, alignSelf: "center" }}
            >
              Apply is locked because the auto-preview says "worse". Tick the box to override.
            </span>
          )}
        </div>
      </div>
    </article>
  );
}

/* ------------------------------------------------------------------ utils */


function report(e: unknown, pushToast: ReturnType<typeof useStore>["pushToast"], fallback: string) {
  pushToast(errorToastPayload(e, fallback));
}

export default CorrelatePanel;
