/**
 * ErrorLibraryPanel.tsx — searchable, plain-English error library.
 *
 * The backend's {code, plain, fix, technical} contract is preserved; this
 * dialog adds "what happened", "what to do", "what not to do" and glossary
 * links. A toast can open it pre-filtered to the exact code that fired.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { BEGINNER_COPY } from "../copy/beginner";
import {
  ERROR_GROUPS,
  ERROR_LIBRARY,
  ERROR_PREFIX_HELP,
  errorHelp,
  errorSearch,
  type ErrorHelp,
} from "../copy/errors";
import { GlossaryTip } from "./GlossaryTip";
import { GlossaryText } from "./GlossaryText";

const ALL: readonly ErrorHelp[] = Object.freeze([
  ...ERROR_LIBRARY,
  ...ERROR_PREFIX_HELP,
]);

export function ErrorLibraryPanel({
  initialCode = "",
  onClose,
}: {
  initialCode?: string;
  onClose: () => void;
}) {
  const [query, setQuery] = useState(initialCode);
  const [group, setGroup] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const active = initialCode ? errorHelp(initialCode) : null;

  useEffect(() => setQuery(initialCode), [initialCode]);
  useEffect(() => {
    inputRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const found = useMemo(() => errorSearch(ALL, query, group), [query, group]);

  return (
    <div className="modal-backdrop" onMouseDown={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <section
        className="error-library-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="error-library-title"
        data-testid="error-library"
      >
        <header className="glossary-dialog-head">
          <div>
            <div className="pane-title">{BEGINNER_COPY.eyebrow}</div>
            <h2 id="error-library-title">{BEGINNER_COPY.errorTitle}</h2>
            <p><GlossaryText text={BEGINNER_COPY.errorIntro} /></p>
          </div>
          <button className="btn btn-sm" onClick={onClose} aria-label="Close error library">✕ Close</button>
        </header>

        {active && (
          <article className="error-active-card">
            <div className="pane-title">About the error you just saw · <code>{initialCode}</code></div>
            <h3>{active.title}</h3>
            <p><GlossaryText text={active.meaning} /></p>
            <dl>
              <dt>What to do</dt><dd><GlossaryText text={active.fix} /></dd>
              <dt>What not to do</dt><dd><GlossaryText text={active.avoid} /></dd>
            </dl>
            <div className="help-terms">
              {active.terms.map((slug) => (
                <span className="help-term-chip" key={slug}><GlossaryTip term={slug} /></span>
              ))}
            </div>
          </article>
        )}

        <div className="glossary-controls">
          <label>
            <span className="sr-only">Search the error library</span>
            <input
              ref={inputRef}
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={BEGINNER_COPY.errorSearch}
              aria-label="Search the error library"
            />
          </label>
          <label>
            <span className="sr-only">Filter errors by topic</span>
            <select value={group} onChange={(e) => setGroup(e.target.value)} aria-label="Filter errors by topic">
              <option value="">{BEGINNER_COPY.errorAll}</option>
              {ERROR_GROUPS.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>
        </div>

        <p className="glossary-count" aria-live="polite">{found.length} explanations</p>

        <ul className="error-library-list">
          {found.map((e) => (
            <li key={e.code}>
              <div className="glossary-row">
                <strong>{e.title}</strong>
                <small>{e.group}</small>
              </div>
              <code>{e.code}</code>
              <p><GlossaryText text={e.meaning} /></p>
              <p className="error-fix"><b>Do:</b> <GlossaryText text={e.fix} /></p>
              <p className="error-avoid"><b>Avoid:</b> <GlossaryText text={e.avoid} /></p>
              <div className="help-terms">
                {e.terms.map((slug) => (
                  <span className="help-term-chip" key={slug}><GlossaryTip term={slug} /></span>
                ))}
              </div>
            </li>
          ))}
          {found.length === 0 && <li className="glossary-empty">{BEGINNER_COPY.errorEmpty}</li>}
        </ul>
      </section>
    </div>
  );
}
