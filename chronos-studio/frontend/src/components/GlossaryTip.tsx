/**
 * GlossaryTip.tsx — one 242-term tooltip, reachable by mouse AND keyboard.
 *
 * WCAG 2.1 AA: the trigger is focusable, shows on focus as well as hover,
 * is dismissable with Escape, and is exposed with role="tooltip" +
 * aria-describedby. The definition is also on the native `title` attribute so
 * a screen reader still gets it even when the popover is closed.
 */

import { useId, useState, type ReactNode } from "react";
import { lookupGlossary, type GlossaryEntry } from "../copy/glossary";

export interface GlossaryTipProps {
  /** A slug from GLOSSARY, a display term, or a loose phrase like "MASE". */
  term: string;
  children?: ReactNode;
}

export function glossaryTipProps(term: string): {
  found: boolean;
  label: string;
  definition: string;
  category: string;
  entry?: GlossaryEntry;
} {
  const entry = lookupGlossary(term);
  return {
    found: Boolean(entry),
    label: entry?.term ?? term,
    definition: entry?.definition ?? "",
    category: entry?.category ?? "",
    entry,
  };
}

export function GlossaryTip({ term, children }: GlossaryTipProps) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const info = glossaryTipProps(term);
  if (!info.found) return <>{children ?? term}</>;

  return (
    <span className="glossary-tip">
      <abbr
        className="glossary-term"
        title={info.definition}
        tabIndex={0}
        aria-describedby={open ? id : undefined}
        aria-label={`Glossary: ${info.label}. ${info.definition}`}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
        }}
      >
        {children ?? info.label}
      </abbr>
      {open && (
        <span id={id} role="tooltip" className="glossary-popover">
          <span className="glossary-popover-head">
            <strong>{info.label}</strong>
            <small>{info.category}</small>
          </span>
          <span className="glossary-popover-body">{info.definition}</span>
        </span>
      )}
    </span>
  );
}
