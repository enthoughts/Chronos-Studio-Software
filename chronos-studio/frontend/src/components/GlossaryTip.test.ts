import { describe, expect, it } from "vitest";

import { glossaryTipProps } from "./GlossaryTip";
import { glossarySearch } from "./GlossaryPanel";
import { GLOSSARY, GLOSSARY_TERM_COUNT } from "../copy/glossary";

describe("Part 19 glossary tooltip + search helpers", () => {
  it("returns the definition, category and label for a known term", () => {
    const p = glossaryTipProps("mase");
    expect(p.found).toBe(true);
    expect(p.label).toBe("MASE");
    expect(p.category).toBe("Evaluation");
    expect(p.definition).toContain("naive");
    expect(p.entry?.slug).toBe("mase");
  });

  it("degrades gracefully for an unknown term", () => {
    const p = glossaryTipProps("definitely_not_a_term");
    expect(p.found).toBe(false);
    expect(p.label).toBe("definitely_not_a_term");
    expect(p.definition).toBe("");
  });

  it("search matches term, slug and definition text and respects the category", () => {
    const all = Object.values(GLOSSARY);
    expect(glossarySearch(all, "")).toHaveLength(GLOSSARY_TERM_COUNT);
    expect(glossarySearch(all, "MASE").map((e) => e.slug)).toContain("mase");
    expect(glossarySearch(all, "Gneiting").map((e) => e.slug)).toContain("crps");
    const evalOnly = glossarySearch(all, "", "Evaluation");
    expect(evalOnly.length).toBeGreaterThan(10);
    expect(evalOnly.every((e) => e.category === "Evaluation")).toBe(true);
    expect(glossarySearch(all, "zzzz_no_match")).toHaveLength(0);
  });
});
