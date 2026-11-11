/**
 * GlossaryPanel.tsx — searchable, keyboard-friendly glossary dialog.
 *
 * §5.15 asks for a 120+ term glossary with tooltips "everywhere"; this dialog
 * is the index of all 242 terms, and every term can also be reached inline via
 * <GlossaryTip>. The search helper is pure and unit-tested.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import {
  GLOSSARY,
  GLOSSARY_CATEGORIES,
  GLOSSARY_TERM_COUNT,
  type GlossaryEntry,
} from "../copy/glossary";

export function glossarySearch(
  entries: readonly GlossaryEntry[],
  query: string,
  category = "",
): GlossaryEntry[] {
  const q = query.trim().toLowerCase();
  return entries.filter((e) => {
    if (category && e.category !== category) return false;
    if (!q) return true;
    return (
      e.term.toLowerCase().includes(q) ||
      e.slug.toLowerCase().includes(q) ||
      e.definition.toLowerCase().includes(q)
    );
  });
}

const ALL_ENTRIES: readonly GlossaryEntry[] = Object.freeze(Object.values(GLOSSARY));

export function GlossaryPanel({ onClose }: { onClose: () => void }) {
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const found = useMemo(
    () => glossarySearch(ALL_ENTRIES, query, category),
    [query, category],
  );

  return (
    <div className="modal-backdrop" onMouseDown={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <section
        className="glossary-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="glossary-dialog-title"
      >
        <header className="glossary-dialog-head">
          <div>
            <div className="pane-title">Part 19 · beginner features</div>
            <h2 id="glossary-dialog-title">Glossary — {GLOSSARY_TERM_COUNT} terms</h2>
            <p>Plain-English definitions. Hover or focus an underlined term anywhere in the app to see the same text.</p>
          </div>
          <button className="btn btn-sm" onClick={onClose} aria-label="Close glossary">✕ Close</button>
        </header>

        <div className="glossary-controls">
          <label>
            <span className="sr-only">Search the glossary</span>
            <input
              ref={inputRef}
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search terms and definitions…"
              aria-label="Search the glossary"
            />
          </label>
          <label>
            <span className="sr-only">Filter by topic</span>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              aria-label="Filter by topic"
            >
              <option value="">All topics</option>
              {GLOSSARY_CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        </div>

        <p className="glossary-count" aria-live="polite">
          {found.length} of {GLOSSARY_TERM_COUNT} terms
        </p>

        <ul className="glossary-list">
          {found.map((e) => (
            <li key={e.slug}>
              <div className="glossary-row">
                <strong>{e.term}</strong>
                <small>{e.category}</small>
              </div>
              <p>{e.definition}</p>
            </li>
          ))}
          {found.length === 0 && <li className="glossary-empty">No term matches that search.</li>}
        </ul>
      </section>
    </div>
  );
}
