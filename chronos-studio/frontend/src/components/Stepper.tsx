/**
 * Stepper.tsx — the pipeline as a walkable bar.
 *
 * The stage map comes from the backend (`/pipeline`), so the frontend never
 * hardcodes the dependency graph. A stale stage is visibly stale, not quietly
 * reused (CRAFT §4.2). Part 19 adds explicit Back/Next controls and arrow-key
 * navigation, so a beginner can revisit an earlier step without hunting for a
 * tiny dot or discovering the double-click reset gesture.
 */

import { BEGINNER_COPY, stepNavigation } from "../copy/beginner";
import { STAGE_COPY } from "../copy/strings";
import { useStore } from "../state/store";
import type { StageCode } from "../state/types";

const STATUS_TITLE: Record<string, string> = {
  not_started: "Not started",
  running: "Running…",
  complete: "Complete",
  stale: "Stale — an earlier step changed, re-run this one",
  blocked: "Blocked — clear the listed blockers first",
  failed: "Failed",
};

export function Stepper() {
  const { stages, session, activeStage, setActiveStage, resetFrom } = useStore();
  if (stages.length === 0) return null;

  const codes = stages.map((s) => s.code);
  const nav = stepNavigation(codes, activeStage);
  const go = (code: string | null) => {
    if (code) setActiveStage(code as StageCode);
  };
  const backLabel = nav.previous
    ? `${BEGINNER_COPY.backLabel}: ${STAGE_COPY[nav.previous]?.label ?? nav.previous}`
    : BEGINNER_COPY.backLabel;
  const nextLabel = nav.next
    ? `${BEGINNER_COPY.nextLabel}: ${STAGE_COPY[nav.next]?.label ?? nav.next}`
    : BEGINNER_COPY.nextLabel;

  return (
    <nav
      className="stepper"
      aria-label="Analysis pipeline"
      data-position={`${nav.position}/${nav.total}`}
      onKeyDown={(e) => {
        if (e.key === "ArrowLeft" && nav.previous) {
          e.preventDefault();
          go(nav.previous);
        } else if (e.key === "ArrowRight" && nav.next) {
          e.preventDefault();
          go(nav.next);
        } else if (e.key === "Home" && codes.length > 0) {
          e.preventDefault();
          go(codes[0]);
        } else if (e.key === "End" && codes.length > 0) {
          e.preventDefault();
          go(codes[codes.length - 1]);
        }
      }}
    >
      <button
        className="step-nav"
        onClick={() => go(nav.previous)}
        disabled={!nav.previous}
        aria-label={backLabel}
        title={backLabel}
        data-testid="step-back"
      >
        <span aria-hidden>←</span> {BEGINNER_COPY.back}
      </button>
      {stages.map((s) => {
        const code = s.code as StageCode;
        const status = session?.pipeline.stages[code]?.status ?? "not_started";
        const message = session?.pipeline.stages[code]?.message ?? "";
        const copy = STAGE_COPY[code];
        const active = activeStage === code;
        const done = status === "complete";

        return (
          <button
            key={code}
            className="step"
            data-status={status}
            data-active={active}
            aria-current={active ? "step" : undefined}
            title={`${copy?.label ?? code} — ${STATUS_TITLE[status] ?? status}${message ? `: ${message}` : ""}`}
            onClick={() => setActiveStage(code)}
            onDoubleClick={() => {
              if (done || status === "stale") void resetFrom(code);
            }}
          >
            <span className="step-dot" aria-hidden />
            <span className="step-code">{code}</span>
            <span>{copy?.short ?? code}</span>
            {status === "stale" && <span aria-label="stale" title="Stale">•</span>}
          </button>
        );
      })}
      <button
        className="step-nav step-nav-next"
        onClick={() => go(nav.next)}
        disabled={!nav.next}
        aria-label={nextLabel}
        title={nextLabel}
        data-testid="step-next"
      >
        {BEGINNER_COPY.next} <span aria-hidden>→</span>
      </button>
    </nav>
  );
}
