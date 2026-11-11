/**
 * ResultsPane.tsx — the right-hand results column.
 *
 * Part 0: renders whatever results exist for the active stage, or a helpful
 * empty state that tells the user exactly what to do next. Because it iterates
 * `AnyResult`, Parts 2–18 add results without touching this file.
 */

import { BEGINNER_COPY } from "../copy/beginner";
import { COPY, STAGE_COPY } from "../copy/strings";
import { useStore } from "../state/store";
import type { StageCode } from "../state/types";
import { ResultCard } from "./ResultCard";
import { IconChart } from "./Icons";
import { DataPanel } from "../panels/DataPanel";
import { EdaPanel } from "../panels/EdaPanel";
import { AdvicePanel } from "../panels/AdvicePanel";
import { StationarityPanel } from "../panels/StationarityPanel";
import { DecomposePanel } from "../panels/DecomposePanel";
import { CorrelatePanel } from "../panels/CorrelatePanel";
import { GatePanel } from "../panels/GatePanel";
import { ModelsPanel } from "../panels/ModelsPanel";
import { SelectPanel } from "../panels/SelectPanel";
import { ForecastPanel } from "../panels/ForecastPanel";
import { EvaluationPanel } from "../panels/EvaluationPanel";
import { CatalogPanel } from "../panels/CatalogPanel";
import { ReportPanel } from "../panels/ReportPanel";
import { GlossaryTip } from "./GlossaryTip";
import { GlossaryText } from "./GlossaryText";

export type S15View = "report" | "catalog";

/** Part 19: one jargon term per stage, so a beginner always has a way in. */
const STAGE_TERM: Record<string, string> = {
  S0: "ingestion", S1: "outlier", S2: "advisor", S3: "preview",
  S4: "decomposition", S5: "unit_root", S6: "acf", S7: "reconciliation",
  S8: "readiness_gate", S9: "model_registry", S10: "ets",
  S11: "rolling_origin_cv", S12: "pit", S13: "prediction_interval",
  S14: "mase", S15: "scorecard",
};

export function ResultsPane({
  s15View = "report",
  onSetS15View,
  onOpenHelp,
}: {
  s15View?: S15View;
  onSetS15View?: (view: S15View) => void;
  /** Part 19: open the stage-specific "What is this?" drawer. */
  onOpenHelp?: (panelId: string) => void;
} = {}) {
  const { results, activeStage, session, loading } = useStore();
  const code = activeStage as StageCode;
  const list = results[code] ?? [];
  const copy = STAGE_COPY[code];

  if (loading) {
    return (
      <main className="results-pane" aria-busy="true">
        <div className="card"><div className="card-body">
          <div className="skeleton" style={{ width: "40%", marginBottom: 10 }} />
          <div className="skeleton" style={{ width: "90%", marginBottom: 6 }} />
          <div className="skeleton" style={{ width: "70%" }} />
        </div></div>
      </main>
    );
  }

  // Loading data is the one step that must work before a session exists —
  // the first upload creates the session.
  if (!session && code === "S0") {
    return (
      <main className="results-pane" aria-label="Load data">
        <header className="results-head" style={{ marginBottom: 14 }}>
          <div>
            <h1>{COPY.noSession.title}</h1>
            <p style={{ fontSize: 13, color: "var(--ink-soft)", maxWidth: 720 }}>
              <GlossaryText text={COPY.noSession.body} />
            </p>
          </div>
          <HelpButton onClick={onOpenHelp} panelId="S0" />
        </header>
        <DataPanel />
      </main>
    );
  }

  if (!session) {
    return (
      <main className="results-pane">
        <div className="empty">
          <IconChart size={34} />
          <h2 style={{ marginTop: 12 }}>{COPY.noSession.title}</h2>
          <p><GlossaryText text={COPY.noSession.body} /></p>
          <HelpButton onClick={onOpenHelp} panelId={code} />
        </div>
      </main>
    );
  }

  return (
    <main className="results-pane" aria-label={`Results for ${copy?.label ?? code}`}>
      <header className="results-head" style={{ marginBottom: 14 }}>
        <div>
          <h1>{copy?.label ?? code}</h1>
          <p style={{ fontSize: 13, color: "var(--ink-soft)", maxWidth: 720 }}>
            <GlossaryText text={copy?.plain ?? ""} />
          </p>
          <p className="stage-key-idea">
            Key idea: <GlossaryTip term={STAGE_TERM[code] ?? "time_series"} />
          </p>
        </div>
        <HelpButton onClick={onOpenHelp} panelId={code} />
      </header>

      {code === "S0" ? (
        <DataPanel />
      ) : code === "S1" ? (
        <EdaPanel />
      ) : code === "S2" || code === "S3" ? (
        <AdvicePanel step={code} />
      ) : code === "S4" ? (
        <DecomposePanel />
      ) : code === "S5" ? (
        <StationarityPanel />
      ) : code === "S6" || code === "S7" ? (
        <CorrelatePanel />
      ) : code === "S8" ? (
        <GatePanel />
      ) : code === "S9" || code === "S10" ? (
        <ModelsPanel />
      ) : code === "S11" || code === "S12" ? (
        <SelectPanel step={code} />
      ) : code === "S13" ? (
        <ForecastPanel />
      ) : code === "S14" ? (
        <EvaluationPanel />
      ) : code === "S15" ? (
        s15View === "catalog" ? (
          <CatalogPanel onOpenReport={() => onSetS15View?.("report")} />
        ) : (
          <ReportPanel onOpenCatalog={() => onSetS15View?.("catalog")} />
        )
      ) : list.length === 0 ? (
        <div className="empty">
          <IconChart size={30} />
          <h2 style={{ marginTop: 12 }}>{COPY.resultsEmpty.title}</h2>
          <p>{COPY.resultsEmpty.body}</p>
          <p style={{ marginTop: 10, fontSize: 12, color: "var(--ink-faint)" }}>
            This step is built in Part {PART_OF[code] ?? "?"} of the build plan.
          </p>
        </div>
      ) : (
        list.map((r) => <ResultCard key={r.id} result={r} />)
      )}
    </main>
  );
}

/** Part 19: every panel header has the same one-click explanation entry point. */
function HelpButton({
  onClick,
  panelId = "S0",
}: {
  onClick?: (panelId: string) => void;
  panelId?: string;
}) {
  return (
    <button
      className="btn btn-sm panel-help-btn"
      onClick={() => onClick?.(panelId)}
      aria-label={`${BEGINNER_COPY.open} ${panelId}`}
      title={BEGINNER_COPY.drawerIntro}
      data-testid="panel-help"
    >
      <span aria-hidden>?</span> {BEGINNER_COPY.open}
    </button>
  );
}

/** Which build part supplies each stage — shown in the empty state so the user
 *  is never left wondering whether something is broken or simply not built yet. */
const PART_OF: Record<string, string> = {
  S0: "1", S1: "2", S2: "3", S3: "3", S4: "4", S5: "5", S6: "6",
  S7: "6", S8: "7", S9: "8", S10: "8–12", S11: "13", S12: "13",
  S13: "14", S14: "15", S15: "18",
};
