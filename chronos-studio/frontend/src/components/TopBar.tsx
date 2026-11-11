/** TopBar.tsx — brand, session status, AI status, beginner modes, destructive actions. */

import { useRef } from "react";
import { COPY } from "../copy/strings";
import { GLOSSARY_TERM_COUNT } from "../copy/glossary";
import { BEGINNER_COPY } from "../copy/beginner";
import { useStore } from "../state/store";
import { HistoryControls } from "./HistoryControls";
import { IconRefresh, IconSparkle, IconTrash } from "./Icons";

export interface TopBarProps {
  simplifyView?: boolean;
  plainLanguage?: boolean;
  onToggleSimplify?: () => void;
  onTogglePlainLanguage?: () => void;
  onOpenGlossary?: () => void;
  onOpenDemo?: () => void;
  onOpenHelp?: () => void;
  onOpenErrors?: () => void;
}

export function TopBar({
  simplifyView = false,
  plainLanguage = false,
  onToggleSimplify,
  onTogglePlainLanguage,
  onOpenGlossary,
  onOpenDemo,
  onOpenHelp,
  onOpenErrors,
}: TopBarProps = {}) {
  const {
    session, sessionId, health, aiLiveReady, startSession, endSession,
    refresh, openAssistant,
  } = useStore();
  const moreRef = useRef<HTMLDetailsElement>(null);
  const closeMore = () => {
    if (moreRef.current) moreRef.current.open = false;
  };

  const pct = session?.progress.percent ?? 0;
  const liveAi = aiLiveReady || Boolean(health?.optional_features?.gemini);

  return (
    <header className="topbar">
      <div className="brand">
        <div className="brand-mark" aria-hidden />
        <div>
          <div className="brand-name">{COPY.appName}</div>
          <div className="brand-sub">{COPY.tagline}</div>
        </div>
      </div>

      {/* Part 19 beginner features: worked demo, glossary, simplify, plain language. */}
      <button className="btn btn-sm" onClick={onOpenDemo}
              title="Watch a complete worked analysis, with progress and cancel">
        ▶ Demo
      </button>
      <button className="btn btn-sm btn-secondary" onClick={onOpenGlossary}
              title={`${GLOSSARY_TERM_COUNT} plain-English definitions, searchable`}>
        📖 Glossary
      </button>
      <button className="btn btn-sm btn-secondary" onClick={onToggleSimplify} aria-pressed={simplifyView}
              title="Hide all but the eight essential panels">
        ◱ Simplify view
      </button>
      <button className="btn btn-sm btn-secondary" onClick={onTogglePlainLanguage} aria-pressed={plainLanguage}
              title="Hide the technical layer of each result card">
        Aa Plain language
      </button>
      <button className="btn btn-sm btn-icon" onClick={onOpenHelp}
              title={`${BEGINNER_COPY.open} ${BEGINNER_COPY.drawerIntro}`}
              aria-label={BEGINNER_COPY.open} data-testid="open-help">
        <span aria-hidden>?</span>
      </button>
      <button className="btn btn-sm btn-icon" onClick={onOpenErrors}
              title={BEGINNER_COPY.errorOpenLabel} aria-label={BEGINNER_COPY.errorOpenLabel}
              data-testid="open-error-library">
        <span aria-hidden>!</span> <span className="btn-icon-text">{BEGINNER_COPY.errorOpen}</span>
      </button>
      <HistoryControls compact />

      <div className="spacer" />

      {session && (
        <>
          <span className="pill pill-session" title="Analysis seed — every random draw is reproducible from it">
            seed {session.seed}
          </span>
          <span className="pill pill-brand pill-session" title="Pipeline completion">
            <span className="dot" /> {pct.toFixed(0)}% complete
          </span>
          {session.dataset && (
            <span className="pill pill-session" title={`${session.dataset.n_observations} observations`}>
              {session.dataset.name} · n={session.dataset.n_observations} ·{" "}
              {session.dataset.frequency.label.split(" (")[0]}
            </span>
          )}
        </>
      )}

      {/* Secondary actions live in an overflow menu so the fixed-height topbar
          never ends in a half-clipped control at the right edge. Compact widths
          also get copies of the beginner toggles here, not display:none dead ends. */}
      <details className="topbar-more" ref={moreRef} data-testid="topbar-more">
        <summary className="btn btn-sm" aria-label="More actions" title="More actions">⋯ More</summary>
        <div className="topbar-more-menu" role="menu">
          <button
            className="btn btn-sm only-compact"
            onClick={() => { onOpenGlossary?.(); closeMore(); }}
            title={`${GLOSSARY_TERM_COUNT} plain-English definitions, searchable`}
            role="menuitem" data-testid="topbar-more-glossary"
          >
            📖 Glossary
          </button>
          <button
            className="btn btn-sm only-compact"
            onClick={() => { onToggleSimplify?.(); closeMore(); }}
            aria-pressed={simplifyView} title="Hide all but the eight essential panels"
            role="menuitem" data-testid="topbar-more-simplify"
          >
            ◱ Simplify view
          </button>
          <button
            className="btn btn-sm only-compact"
            onClick={() => { onTogglePlainLanguage?.(); closeMore(); }}
            aria-pressed={plainLanguage} title="Hide the technical layer of each result card"
            role="menuitem" data-testid="topbar-more-plain"
          >
            Aa Plain language
          </button>
          {session && (
            <div className="topbar-more-session only-narrow">
              <span className="pill" title={`Analysis seed ${session.seed}`}>seed {session.seed}</span>
              <span className="pill pill-brand"><span className="dot" /> {pct.toFixed(0)}% complete</span>
              {session.dataset && (
                <span className="pill" title={`${session.dataset.n_observations} observations`}>
                  {session.dataset.name} · n={session.dataset.n_observations}
                </span>
              )}
            </div>
          )}
          <span
            className={`pill ${liveAi ? "pill-ok" : ""}`}
            title={liveAi ? COPY.aiOnHint : COPY.aiOffHint}
          >
            <IconSparkle size={12} />
            {liveAi ? COPY.aiOn : COPY.aiOff}
          </span>
          <button
            className="btn btn-sm btn-primary"
            onClick={() => { openAssistant(); closeMore(); }}
            title="Grounded answers, offline mode and payload preview"
            data-testid="open-assistant"
            role="menuitem"
          >
            <IconSparkle size={13} /> {COPY.assistant.open}
          </button>
          <button className="btn btn-sm" onClick={() => { void refresh(); closeMore(); }}
                  disabled={!sessionId} title="Reload state from the server" role="menuitem">
            <IconRefresh size={13} /> Refresh
          </button>
          {sessionId ? (
            <button className="btn btn-sm" onClick={() => { void endSession(); closeMore(); }}
                    title="Hard-delete this session from server memory" role="menuitem">
              <IconTrash size={13} /> {COPY.endSession}
            </button>
          ) : (
            <button className="btn btn-sm btn-primary" onClick={() => { void startSession(); closeMore(); }}
                    role="menuitem">
              {COPY.newSession}
            </button>
          )}
        </div>
      </details>
    </header>
  );
}
