/**
 * HelpDrawer.tsx — the per-panel "What is this?" drawer (Part 19 §5.15).
 *
 * It is deliberately non-modal: a beginner can read the explanation while
 * still looking at the panel. Escape closes it, focus moves into the drawer on
 * open and returns to the trigger on close. Every explanation comes from
 * copy/beginner.ts, and every named term is a keyboard-reachable tooltip.
 */

import { useEffect, useId, useRef } from "react";
import { BEGINNER_COPY, panelHelp } from "../copy/beginner";
import { PANEL_ALIASES } from "../copy/beginner";
import { STAGE_COPY } from "../copy/strings";
import { GlossaryTip } from "./GlossaryTip";
import { GlossaryText } from "./GlossaryText";

export function HelpDrawer({
  panelId,
  onClose,
}: {
  panelId: string;
  onClose: () => void;
}) {
  const help = panelHelp(panelId);
  const stage = PANEL_ALIASES[panelId] ?? panelId;
  const label = STAGE_COPY[stage]?.label ?? panelId;
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      previous?.focus?.();
    };
  }, [onClose]);

  return (
    <aside
      className="help-drawer"
      role="dialog"
      aria-labelledby={titleId}
      aria-describedby={`${titleId}-summary`}
      data-testid="help-drawer"
    >
      <header className="help-drawer-head">
        <div>
          <div className="pane-title">{BEGINNER_COPY.eyebrow}</div>
          <h2 id={titleId}>{BEGINNER_COPY.drawerTitle} {label}</h2>
        </div>
        <button ref={closeRef} className="btn btn-sm" onClick={onClose} aria-label={BEGINNER_COPY.close}>
          ✕ {BEGINNER_COPY.close}
        </button>
      </header>

      <p className="help-drawer-intro" id={`${titleId}-summary`}>
        <GlossaryText text={BEGINNER_COPY.drawerIntro} />
      </p>

      <section className="help-block help-summary">
        <GlossaryText text={help.summary} maxHits={4} />
      </section>

      <section className="help-block">
        <h3>{BEGINNER_COPY.readingLabel}</h3>
        <p><GlossaryText text={help.reading} maxHits={4} /></p>
      </section>

      <section className="help-block">
        <h3>{BEGINNER_COPY.tryLabel}</h3>
        <p><GlossaryText text={help.tryThis} maxHits={3} /></p>
      </section>

      <section className="help-block help-watch">
        <h3>{BEGINNER_COPY.watchLabel}</h3>
        <p><GlossaryText text={help.watchOut} maxHits={3} /></p>
      </section>

      <section className="help-block">
        <h3>{BEGINNER_COPY.termsLabel}</h3>
        <div className="help-terms">
          {help.terms.map((slug) => (
            <span className="help-term-chip" key={slug}>
              <GlossaryTip term={slug} />
            </span>
          ))}
        </div>
      </section>
    </aside>
  );
}
