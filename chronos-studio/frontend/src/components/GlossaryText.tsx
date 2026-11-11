/**
 * GlossaryText.tsx — automatic inline glossary tooltips for arbitrary prose.
 *
 * <GlossaryTip> handles explicitly labelled terms. This component adds the
 * "everywhere" half of §5.15: any user-facing sentence can be passed through
 * <GlossaryText> and its known glossary phrases become focusable tooltips.
 * The matcher is a pure, unit-testable segmenter; no DOM rewriting is involved,
 * so React remains in control of the rendered tree.
 */

import { GLOSSARY, type GlossaryEntry } from "../copy/glossary";
import { GlossaryTip } from "./GlossaryTip";

export interface GlossarySegment {
  text: string;
  slug?: string;
}

/** Normalize loose phrases ("P value", "p-value", "P  value") to one key. */
export function normalizeGlossaryPhrase(input: string): string {
  return input.trim().toLowerCase().replace(/[^a-z0-9]+/g, "");
}

const ENTRY_BY_PHRASE: ReadonlyMap<string, GlossaryEntry> = new Map(
  Object.values(GLOSSARY).map((e) => [normalizeGlossaryPhrase(e.term), e]),
);

export function glossaryEntryForPhrase(input: string): GlossaryEntry | undefined {
  return ENTRY_BY_PHRASE.get(normalizeGlossaryPhrase(input));
}

/* Short, generic English words would underline ordinary prose without adding
   meaning. They remain in the searchable 242-term glossary and in <GlossaryTip>. */
const SKIP_SLUGS = new Set([
  "sample", "population", "parameter", "estimate", "level", "season", "noise",
  "lead", "lag", "trend", "period", "observation", "significance", "robust",
  "deterministic", "coverage", "caption", "verdict", "stale", "preview",
  "undo", "redo", "fold", "mean_model", "cycle", "multiplicative",
]);

const TERMS = Object.values(GLOSSARY)
  .filter((e) => e.term.length >= 4 && !SKIP_SLUGS.has(e.slug))
  .sort((a, b) => b.term.length - a.term.length || a.term.localeCompare(b.term));

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const MATCHER = new RegExp(
  `(?<![\\p{L}\\p{N}])(${TERMS.map((e) => escapeRegExp(e.term)).join("|")})(?![\\p{L}\\p{N}])`,
  "giu",
);

/**
 * Split prose into plain strings and glossary matches. Longest terms win; only
 * `maxHits` phrases per sentence are underlined so a paragraph stays readable.
 */
export function glossarySegments(text: string, maxHits = 3): GlossarySegment[] {
  if (!text) return [];
  const out: GlossarySegment[] = [];
  const re = new RegExp(MATCHER.source, MATCHER.flags);
  let cursor = 0;
  let hits = 0;
  let m: RegExpExecArray | null;
  while (hits < maxHits && (m = re.exec(text)) !== null) {
    const entry = glossaryEntryForPhrase(m[0]);
    if (!entry) continue; // defensive: regex and map come from the same source
    if (m.index > cursor) out.push({ text: text.slice(cursor, m.index) });
    out.push({ text: m[0], slug: entry.slug });
    cursor = m.index + m[0].length;
    hits += 1;
  }
  if (cursor < text.length) out.push({ text: text.slice(cursor) });
  return out.length > 0 ? out : [{ text }];
}

export function GlossaryText({
  text,
  maxHits = 3,
}: {
  text: string;
  maxHits?: number;
}) {
  return (
    <>
      {glossarySegments(text, maxHits).map((seg, i) =>
        seg.slug ? (
          <GlossaryTip key={`${seg.slug}-${i}`} term={seg.slug}>
            {seg.text}
          </GlossaryTip>
        ) : (
          <span key={`plain-${i}`}>{seg.text}</span>
        ),
      )}
    </>
  );
}
