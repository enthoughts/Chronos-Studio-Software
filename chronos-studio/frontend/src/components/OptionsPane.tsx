/**
 * OptionsPane.tsx — the left panel. JASP-style: what this step is, what it will
 * ask of you, and what you will get. Part 0 shows the contract; later parts
 * replace `PanelBody` per stage with the real controls.
 */

import { BEGINNER_COPY } from "../copy/beginner";
import { STAGE_COPY } from "../copy/strings";
import { useStore } from "../state/store";
import type { StageCode } from "../state/types";
import { HistoryControls } from "./HistoryControls";
import { GlossaryText } from "./GlossaryText";

const STAGE_PROMISE: Record<string, { asks: string[]; gives: string[] }> = {
  S0: { asks: ["a file or pasted table", "which column is time", "which column is the value"],
        gives: ["a validated time index", "a data-quality report", "a detected data class"] },
  S1: { asks: ["nothing — this runs automatically"],
        gives: ["descriptives", "distribution and time plots", "outliers", "a plain-English description"] },
  S2: { asks: ["whether to accept each suggested fix"],
        gives: ["imputation, transform, smoothing and outlier advice, each with its evidence"] },
  S3: { asks: ["which fixes to apply"],
        gives: ["a versioned, reversible prepared series"] },
  S4: { asks: ["the decomposition method"],
        gives: ["trend, season, cycle and noise, with strength measures"] },
  S5: { asks: ["nothing — the full battery runs"],
        gives: ["unit-root, trend and break tests, reconciled into one matrix"] },
  S6: { asks: ["the maximum lag"],
        gives: ["ACF, PACF, CCF and an automatic pattern reading"] },
  S7: { asks: ["which action to preview"],
        gives: ["a ranked action list, each previewable before you commit"] },
  S8: { asks: ["nothing"],
        gives: ["READY / READY WITH CAVEATS / NOT READY, with the checklist"] },
  S9: { asks: ["any models you want included"],
        gives: ["the registry filtered to what your data can support, with reasons"] },
  S10: { asks: ["which candidates to fit"],
        gives: ["fitted models, parameter tables, convergence warnings"] },
  S11: { asks: ["the selection criterion (CV is the default)"],
        gives: ["a ranked shortlist and a 'why this model won' paragraph"] },
  S12: { asks: ["nothing"],
        gives: ["residual battery, Q-Q, ACF, PIT, RQR and an adequacy verdict"] },
  S13: { asks: ["the horizon", "the interval levels", "the construction", "a bias-adjustment choice", "future driver paths"],
        gives: ["fan charts, analytic + seeded simulation intervals, back-transformed points, scenario comparison, combinations, hierarchy reconciliation"] },
  S14: { asks: ["the rolling-origin scheme", "horizon, folds and interval levels", "which fitted models to compare", "a bias-adjustment choice if the series is transformed"],
        gives: ["an out-of-sample-only metric policy with reasons", "leakage audit and split chart", "point, probabilistic, discrete and variance tables", "MASE by horizon, fold boxplots, loss curves, PIT/RQR, calibration and DM plots"] },
  S15: { asks: ["a display mode", "which plots and tables to insert into the report"],
        gives: ["a ≥35-plot / ≥30-table completeness scorecard", "captioned, accessible items with data behind", "300 dpi PNG, SVG, PDF, XLSX and CSV per-item exports"] },
};

export function OptionsPane() {
  const { activeStage, session, stages } = useStore();
  const code = activeStage as StageCode;
  const copy = STAGE_COPY[code];
  const promise = STAGE_PROMISE[code];
  const status = session?.pipeline.stages[code]?.status ?? "not_started";
  const def = stages.find((s) => s.code === code);
  const missing = (def?.depends_on ?? []).filter(
    (d) => (session?.pipeline.stages[d]?.status ?? "not_started") !== "complete",
  );

  return (
    <aside className="options-pane" aria-label="Options for the current step">
      <div className="pane-title">Step {code}</div>
      <h2 style={{ marginBottom: 4 }}>{copy?.label ?? code}</h2>
      <p style={{ fontSize: 12.5, color: "var(--ink-soft)", marginBottom: 12 }}>
        <GlossaryText text={copy?.plain ?? ""} />
      </p>

      <div className="option-block">
        <h4>Status</h4>
        <div className="hint">
          {status === "complete" && "Complete."}
          {status === "not_started" && "Not started."}
          {status === "running" && "Running…"}
          {status === "stale" && "Stale — an earlier step changed since this last ran."}
          {status === "blocked" && "Blocked — see the readiness gate."}
          {status === "failed" && "Failed — open the notification for the reason."}
        </div>
      </div>

      <div className="option-block">
        <h4>{BEGINNER_COPY.historyTitle}</h4>
        <div className="hint"><GlossaryText text={BEGINNER_COPY.historyHint} /></div>
        <HistoryControls />
      </div>

      {missing.length > 0 && (
        <div className="option-block" style={{ borderColor: "var(--warn)", background: "var(--warn-wash)" }}>
          <h4>Finish these first</h4>
          <div className="hint">{missing.map((m) => `${m} — ${STAGE_COPY[m]?.label ?? m}`).join(", ")}</div>
        </div>
      )}

      {promise && (
        <>
          <div className="option-block">
            <h4>This step will ask you for</h4>
            <ul style={{ margin: "4px 0 0 16px", padding: 0, fontSize: 12, color: "var(--ink-soft)" }}>
              {promise.asks.map((a) => <li key={a}>{a}</li>)}
            </ul>
          </div>
          <div className="option-block">
            <h4>You will get</h4>
            <ul style={{ margin: "4px 0 0 16px", padding: 0, fontSize: 12, color: "var(--ink-soft)" }}>
              {promise.gives.map((g) => <li key={g}>{g}</li>)}
            </ul>
          </div>
        </>
      )}

      {session?.pipeline.caveats.length ? (
        <div className="option-block" style={{ borderColor: "var(--warn)" }}>
          <h4>Caveats in play ({session.pipeline.caveats.length})</h4>
          <ul style={{ margin: "4px 0 0 16px", padding: 0, fontSize: 11.5, color: "var(--ink-soft)" }}>
            {session.pipeline.caveats.slice(0, 5).map((c) => <li key={c}>{c}</li>)}
          </ul>
        </div>
      ) : null}

      <div style={{ marginTop: 14, fontSize: 11, color: "var(--ink-faint)" }}>
        Double-click a step in the bar above to reset the pipeline from that point.
      </div>
    </aside>
  );
}
