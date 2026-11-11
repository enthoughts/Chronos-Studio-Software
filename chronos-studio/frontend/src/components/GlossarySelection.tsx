/**
 * GlossarySelection.tsx — "tooltip everywhere" for text we do not own.
 *
 * Panels, table headers, captions and backend-generated prose are rendered by
 * many components. Rather than wrapping every one of them, selecting any
 * familiar phrase in the app opens the matching 242-term definition. This is
 * keyboard-friendly (select with Shift+Arrow, dismiss with Escape) and never
 * changes the DOM that React owns.
 */

import { useEffect, useState } from "react";
import type { GlossaryEntry } from "../copy/glossary";
import { glossaryEntryForPhrase } from "./GlossaryText";
import { BEGINNER_COPY } from "../copy/beginner";

interface SelectionTip {
  entry: GlossaryEntry;
  left: number;
  top: number;
}

const WIDTH = 340;

export function GlossarySelection() {
  const [tip, setTip] = useState<SelectionTip | null>(null);

  useEffect(() => {
    let timer: number | undefined;
    const clear = () => setTip(null);
    const inspect = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        const sel = window.getSelection();
        const phrase = sel?.toString().trim() ?? "";
        if (!sel || sel.rangeCount === 0 || phrase.length < 3 || phrase.length > 80) {
          clear();
          return;
        }
        const anchor = sel.anchorNode?.parentElement;
        if (!anchor || anchor.closest("input, textarea, select, [contenteditable='true']")) {
          clear();
          return;
        }
        const entry = glossaryEntryForPhrase(phrase);
        if (!entry) {
          clear();
          return;
        }
        const rect = sel.getRangeAt(0).getBoundingClientRect();
        const left = Math.max(8, Math.min(rect.left, window.innerWidth - WIDTH - 12));
        const below = rect.bottom + 9;
        const top = below + 150 > window.innerHeight ? Math.max(8, rect.top - 155) : below;
        setTip({ entry, left, top });
      }, 180);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") clear();
    };
    document.addEventListener("selectionchange", inspect);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", clear, true);
    window.addEventListener("resize", clear);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener("selectionchange", inspect);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", clear, true);
      window.removeEventListener("resize", clear);
    };
  }, []);

  if (!tip) return null;
  return (
    <div
      className="glossary-selection-tip"
      role="tooltip"
      aria-label={`Glossary: ${tip.entry.term}. ${tip.entry.definition}`}
      style={{ left: tip.left, top: tip.top, width: `min(${WIDTH}px, 92vw)` }}
    >
      <div className="glossary-popover-head">
        <strong>{tip.entry.term}</strong>
        <small>{tip.entry.category}</small>
      </div>
      <p>{tip.entry.definition}</p>
      <small className="glossary-selection-hint">{BEGINNER_COPY.selectionHint}</small>
    </div>
  );
}
