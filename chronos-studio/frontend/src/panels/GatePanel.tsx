/**
 * panels/GatePanel.tsx — S8, the model-readiness gate (§5.7).
 *
 * One screen, answering one question: **should a model be fitted to this series, and what must the
 * user carry forward if one is?** The order is deliberate:
 *
 *   1. **The verdict** — a single card, one of four words, in large type. Never "it depends",
 *      never a bare "stationary: yes". If the answer is qualified, the qualification is the
 *      headline, not a footnote.
 *   2. **What blocks it** — only when something does. Each blocker carries the number that failed
 *      and a button that goes to the step where it is fixed. A blocker with no route is a dead end.
 *   3. **The caveats that travel** — the list that will be attached to every later output and to
 *      the final report. It is shown here, in full, so nobody discovers it at the end.
 *   4. **The full checklist** — every criterion, its number, and pass / fail / unknown / n/a.
 *      Unknown is visually distinct from pass on purpose: a check that did not run is not a pass.
 *   5. **The horizon** — set it here and the too-long-horizon warning (§5.11) arrives while the
 *      plan can still change rather than after a forecast exists.
 *
 * The panel is read-only apart from the horizon box and the jump buttons: the gate never edits data.
 */

import { useCallback, useEffect, useState } from "react";
import { IconCheck, IconRefresh } from "../components/Icons";
import { api } from "../state/api";
import { useStore } from "../state/store";
import { errorToastPayload } from "../copy/errors";
import type { GateCriterion, GateResponse, StageCode } from "../state/types";

const VERDICT_STYLE: Record<
  string,
  { label: string; colour: string; wash: string; rule: string; icon: string }
> = {
  ready: {
    label: "READY",
    colour: "var(--ok)",
    wash: "var(--ok-wash)",
    rule: "var(--ok)",
    icon: "✓",
  },
  ready_with_caveats: {
    label: "READY WITH CAVEATS",
    colour: "var(--warn)",
    wash: "var(--warn-wash)",
    rule: "var(--warn)",
    icon: "!",
  },
  not_ready: {
    label: "NOT READY",
    colour: "var(--bad)",
    wash: "var(--bad-wash)",
    rule: "var(--bad)",
    icon: "×",
  },
  inconclusive: {
    label: "INCONCLUSIVE",
    colour: "var(--ink-soft)",
    wash: "var(--stale-wash)",
    rule: "var(--stale)",
    icon: "?",
  },
};

/** Four states, four different looks. `not_applicable` is quiet; `unknown` is loud, because an
 *  unmeasured check is the one a reader is most likely to mistake for a pass. */
const STATE_LOOK: Record<string, { label: string; colour: string; wash: string }> = {
  pass: { label: "pass", colour: "var(--ok)", wash: "var(--ok-wash)" },
  fail: { label: "fail", colour: "var(--bad)", wash: "var(--bad-wash)" },
  unknown: { label: "not measured", colour: "var(--warn)", wash: "var(--warn-wash)" },
  not_applicable: { label: "n/a", colour: "var(--ink-faint)", wash: "transparent" },
};

function report(e: unknown, pushToast: ReturnType<typeof useStore>["pushToast"], fallback: string) {
  pushToast(errorToastPayload(e, fallback));
}

export function GatePanel() {
  const { sessionId, session, refresh, pushToast, setActiveStage } = useStore();
  const [data, setData] = useState<GateResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [horizon, setHorizon] = useState<string>("");
  const [loaded, setLoaded] = useState(false);

  const hasData = Boolean(session?.dataset);

  const load = useCallback(async () => {
    if (!sessionId || !hasData) return;
    const g = await api.getGate();
    setData(g);
    if (g.status === "complete" && horizon === "") {
      const h = (g.facts as Record<string, unknown> | undefined)?.horizon_requested;
      setHorizon(typeof h === "number" && h > 0 ? String(h) : "");
    }
    setLoaded(true);
  }, [sessionId, hasData, horizon]);

  useEffect(() => {
    void load().catch((e) => report(e, pushToast, "The readiness gate could not be read."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, hasData]);

  const run = useCallback(async () => {
    setBusy(true);
    try {
      await api.runGate();
      await load();
      await refresh(); // S8 becomes "complete" server-side; the stepper must agree
      pushToast({
        kind: "success",
        title: "Gate assessed.",
        body: "The verdict and every number behind it are below.",
      });
    } catch (e) {
      report(e, pushToast, "The gate could not be assessed.");
    } finally {
      setBusy(false);
    }
  }, [load, refresh, pushToast]);

  const setH = useCallback(async () => {
    const n = Number(horizon);
    if (!Number.isFinite(n) || n < 1) {
      pushToast({ kind: "error", title: "That is not a horizon.", body: "Give a whole number of periods, 1 or more." });
      return;
    }
    setBusy(true);
    try {
      const r = await api.setHorizon(Math.round(n));
      await load();
      pushToast({
        kind: r.horizon_check?.state === "pass" ? "success" : "info",
        title:
          r.horizon_check?.state === "pass"
            ? `A ${r.horizon}-period horizon is proportionate.`
            : `A ${r.horizon}-period horizon is long for this sample.`,
        body: r.horizon_check?.detail ?? "",
      });
    } catch (e) {
      report(e, pushToast, "The horizon could not be set.");
    } finally {
      setBusy(false);
    }
  }, [horizon, load, pushToast]);

  const jump = useCallback(
    (stage: StageCode | null | undefined) => {
      if (!stage) return;
      setActiveStage(stage);
    },
    [setActiveStage],
  );

  if (!hasData) {
    return (
      <div className="empty">
        <IconCheck size={30} />
        <h2 style={{ marginTop: 12 }}>Nothing to assess yet</h2>
        <p>Load a series first. The gate reads what earlier steps measured, so it needs a series.</p>
      </div>
    );
  }

  const verdict = data?.verdict ? VERDICT_STYLE[data.verdict] : null;
  const notRun = !data || data.status !== "complete";

  return (
    <div>
      {/* ---------- 0. staleness and the run button ---------- */}
      {data?.stale && (
        <section
          className="card"
          style={{ borderLeft: "3px solid var(--warn)", marginBottom: 12 }}
          role="status"
        >
          <div className="card-body" style={{ paddingTop: 10 }}>
            <strong style={{ fontSize: 13 }}>This verdict describes an older version of the series.</strong>
            <p style={{ margin: "4px 0 0", fontSize: 12.5, color: "var(--ink-soft)" }}>
              It was computed for version {data.computed_for_version}, but the series is now at
              version {data.current_version}. Re-run any step that is now stale, then re-assess — a
              verdict about different data is worse than no verdict.
            </p>
            <button
              className="btn btn-sm btn-primary"
              style={{ marginTop: 8 }}
              onClick={() => void run()}
              disabled={busy}
            >
              <IconRefresh size={13} /> {busy ? "Assessing…" : "Re-assess now"}
            </button>
          </div>
        </section>
      )}

      {/* ---------- 1. the verdict ---------- */}
      {verdict && (
        <section
          className="card"
          style={{ borderLeft: `4px solid ${verdict.rule}`, background: verdict.wash }}
          aria-labelledby="gate-head"
        >
          <div className="card-body" style={{ paddingTop: 12 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <span
                aria-hidden
                style={{
                  width: 42,
                  height: 42,
                  flex: "none",
                  borderRadius: "50%",
                  display: "grid",
                  placeItems: "center",
                  background: verdict.colour,
                  color: "#fff",
                  fontSize: 22,
                  fontWeight: 700,
                }}
              >
                {verdict.icon}
              </span>
              <div>
                <div
                  id="gate-head"
                  data-testid="gate-verdict"
                  style={{
                    fontSize: 20,
                    fontWeight: 700,
                    letterSpacing: "0.02em",
                    color: verdict.colour,
                  }}
                >
                  {verdict.label}
                </div>
                <div style={{ fontSize: 13, color: "var(--ink)", marginTop: 2 }}>
                  {data?.headline}
                </div>
              </div>
              <div style={{ flex: 1 }} />
              <div style={{ textAlign: "right", fontSize: 11.5, color: "var(--ink-soft)" }}>
                <div>
                  {data?.n_pass} pass · {data?.n_fail} fail · {data?.n_unknown} not measured
                </div>
                <div>of {data?.n_total} criteria</div>
              </div>
            </div>
            {data?.plain && (
              <p style={{ margin: "10px 0 0", fontSize: 12.5, color: "var(--ink-soft)", maxWidth: 760 }}>
                {data.plain}
              </p>
            )}
          </div>
        </section>
      )}

      {!notRun && !data?.stale && (
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 10 }}>
          <button className="btn btn-sm" onClick={() => void run()} disabled={busy}>
            <IconRefresh size={13} /> {busy ? "Assessing…" : "Re-assess"}
          </button>
          <span style={{ fontSize: 11.5, color: "var(--ink-faint)" }}>
            Re-run this after any step above completes — the gate reads their numbers, it never
            guesses them.
          </span>
        </div>
      )}

      {notRun && (
        <section className="card">
          <div className="card-body" style={{ paddingTop: 12 }}>
            <h2 style={{ margin: 0, fontSize: 14 }}>No verdict yet</h2>
            <p style={{ margin: "6px 0 12px", fontSize: 12.5, color: "var(--ink-soft)" }}>
              The gate reads what Steps 1–6 already measured. Anything they have not measured shows
              up as <em>not measured</em> — not as a pass.
            </p>
            <button className="btn btn-primary" onClick={() => void run()} disabled={busy}>
              <IconRefresh size={14} /> {busy ? "Assessing…" : "Assess readiness"}
            </button>
          </div>
        </section>
      )}

      {/* ---------- 2. blockers ---------- */}
      {(data?.blockers?.length ?? 0) > 0 && (
        <>
          <h2 style={{ margin: "18px 0 8px", fontSize: 14 }}>
            What stops a model being fitted
          </h2>
          {data!.blockers!.map((b) => (
            <BlockerRow key={b.key} item={b} onJump={() => jump(b.jump_to_stage)} />
          ))}
        </>
      )}

      {/* ---------- 3. travelling caveats ---------- */}
      {(data?.caveats_travelling?.length ?? 0) > 0 && (
        <section className="card" style={{ marginTop: 14 }}>
          <div className="card-head">
            <span className="card-kind">Caveats</span>
            <div>
              <div className="card-title">These travel with every later output</div>
              <div className="card-plain">
                Each line below is attached to the forecast, the metrics and the final report. They
                are not advice — they are part of the result.
              </div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12.5 }}>
              {data!.caveats_travelling!.map((c) => (
                <li key={c} style={{ marginBottom: 4 }}>{c}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {/* ---------- 4. the checklist ---------- */}
      {!notRun && (
        <section className="card" style={{ marginTop: 14 }} aria-labelledby="gate-check-head">
          <div className="card-head">
            <span className="card-kind">Checklist</span>
            <div>
              <div className="card-title" id="gate-check-head">
                Every criterion, with the number it was judged on
              </div>
              <div className="card-plain">
                A check is only "pass" if something measured it. "Not measured" means the step that
                would have measured it has not run — run that step, then assess again.
              </div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
              <thead>
                <tr style={{ textAlign: "left", color: "var(--ink-faint)", fontSize: 11 }}>
                  <th style={{ padding: "4px 8px 4px 0" }}>Criterion</th>
                  <th style={{ padding: "4px 8px" }}>Measured</th>
                  <th style={{ padding: "4px 8px" }}>State</th>
                  <th style={{ padding: "4px 0 4px 8px" }}>Fix in</th>
                </tr>
              </thead>
              <tbody>
                {(data?.checklist ?? []).map((c) => (
                  <ChecklistRow key={c.key} item={c} onJump={() => jump(c.jump_to_stage)} />
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* ---------- 5. the horizon ---------- */}
      <section className="card" style={{ marginTop: 14 }}>
        <div className="card-head">
          <span className="card-kind">Horizon</span>
          <div>
            <div className="card-title">How far ahead do you need to forecast?</div>
            <div className="card-plain">
              A horizon longer than a fifth of the sample cannot be checked against the data. Set it
              now, while the plan can still change.
            </div>
          </div>
        </div>
        <div className="card-body" style={{ paddingTop: 0 }}>
          <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
            <input
              type="number"
              min={1}
              value={horizon}
              onChange={(e) => setHorizon(e.target.value)}
              placeholder="e.g. 12"
              aria-label="Forecast horizon"
              style={{ width: 110 }}
            />
            <span style={{ fontSize: 12, color: "var(--ink-faint)" }}>periods ahead</span>
            <button className="btn" onClick={() => void setH()} disabled={busy || horizon === ""}>
              {busy ? "Checking…" : "Check this horizon"}
            </button>
            <div style={{ flex: 1 }} />
            <button className="btn btn-primary" onClick={() => void run()} disabled={busy}>
              <IconRefresh size={14} /> Re-assess
            </button>
          </div>
          {(() => {
            const h = data?.checklist?.find((c) => c.key === "horizon");
            if (!h) return null;
            const look = STATE_LOOK[h.state];
            return (
              <p style={{ margin: "10px 0 0", fontSize: 12.5, color: "var(--ink-soft)" }}>
                <span
                  className="pill"
                  style={{ background: look.wash, color: look.colour, marginRight: 8 }}
                >
                  {look.label}
                </span>
                {h.value} — {h.detail}
              </p>
            );
          })()}
        </div>
      </section>

      {loaded && !data?.stale && data?.status === "complete" && (
        <p style={{ marginTop: 12, fontSize: 11.5, color: "var(--ink-faint)" }}>
          Readiness is not a score and cannot be averaged: one blocker outweighs every pass. The
          evidence behind each line is the table and figure cards in the steps above.
        </p>
      )}
    </div>
  );
}

function BlockerRow({ item, onJump }: { item: GateCriterion; onJump: () => void }) {
  return (
    <section
      className="card"
      style={{ borderLeft: "3px solid var(--bad)", marginBottom: 8 }}
      data-testid={`gate-blocker-${item.key}`}
    >
      <div className="card-head">
        <span className="card-kind">Blocker</span>
        <div>
          <div className="card-title">{item.label}</div>
          <div className="card-plain">{item.detail}</div>
        </div>
        <div className="card-actions">
          {item.jump_to_stage ? (
            <button className="btn" onClick={onJump}>
              Fix in Step {item.jump_to_stage.replace("S", "")}
            </button>
          ) : (
            <span className="pill" style={{ color: "var(--ink-faint)" }}>no fix route</span>
          )}
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0 }}>
        <p style={{ margin: 0, fontSize: 12.5 }}>
          <strong>{item.value}.</strong>{" "}
          <span style={{ color: "var(--ink-soft)" }}>{item.why_unknown}</span>
        </p>
      </div>
    </section>
  );
}

function ChecklistRow({ item, onJump }: { item: GateCriterion; onJump: () => void }) {
  const look = STATE_LOOK[item.state] ?? STATE_LOOK.unknown;
  const muted = item.state === "not_applicable";
  return (
    <tr style={{ borderTop: "1px solid var(--rule)", opacity: muted ? 0.66 : 1 }}>
      <td style={{ padding: "6px 8px 6px 0", verticalAlign: "top" }}>
        <span style={{ fontWeight: 600 }}>{item.label}</span>
        {item.severity === "blocker" && item.state === "fail" && (
          <span style={{ fontSize: 10, color: "var(--bad)", marginLeft: 6 }}>blocking</span>
        )}
      </td>
      <td style={{ padding: "6px 8px", verticalAlign: "top", color: "var(--ink-soft)" }}>
        {item.value || <span style={{ color: "var(--ink-faint)" }}>—</span>}
      </td>
      <td style={{ padding: "6px 8px", verticalAlign: "top" }}>
        <span className="pill" style={{ background: look.wash, color: look.colour }}>
          {look.label}
        </span>
      </td>
      <td style={{ padding: "6px 0 6px 8px", verticalAlign: "top" }}>
        {item.jump_to_stage ? (
          <button
            className="btn"
            style={{ fontSize: 11, padding: "2px 8px" }}
            onClick={onJump}
          >
            Step {item.jump_to_stage.replace("S", "")}
          </button>
        ) : (
          <span style={{ color: "var(--ink-faint)" }}>—</span>
        )}
      </td>
    </tr>
  );
}
