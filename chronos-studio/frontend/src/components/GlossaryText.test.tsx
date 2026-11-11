import { describe, expect, it } from "vitest";

import {
  glossaryEntryForPhrase,
  glossarySegments,
  normalizeGlossaryPhrase,
} from "./GlossaryText";

describe("Part 19 automatic glossary tooltips", () => {
  it("normalizes loose phrases back to real glossary entries", () => {
    expect(normalizeGlossaryPhrase("  Prediction   Interval ")).toBe("predictioninterval");
    expect(glossaryEntryForPhrase("Prediction interval")?.slug).toBe("prediction_interval");
    expect(glossaryEntryForPhrase("p-value")?.slug).toBe("p_value");
    expect(glossaryEntryForPhrase("  MASE ")?.term).toBe("MASE");
    expect(glossaryEntryForPhrase("not a glossary phrase")).toBeUndefined();
  });

  it("segments prose without losing any characters", () => {
    const text = "The MASE score and the prediction interval both describe forecast accuracy.";
    const segments = glossarySegments(text, 4);
    expect(segments.map((s) => s.text).join("")).toBe(text);
    const slugs = segments.map((s) => s.slug).filter(Boolean);
    expect(slugs).toContain("mase");
    expect(slugs).toContain("prediction_interval");
    expect(slugs.length).toBeLessThanOrEqual(4);
    expect(segments.some((s) => s.text === "MASE" && s.slug === "mase")).toBe(true);
  });

  it("caps the number of underlined phrases and never matches inside a word", () => {
    const limited = glossarySegments("MASE and CRPS and prediction interval", 1);
    expect(limited.filter((s) => s.slug)).toHaveLength(1);
    expect(limited.map((s) => s.text).join("")).toBe("MASE and CRPS and prediction interval");
    for (const s of glossarySegments("notmasehere")) expect(s.slug).toBeUndefined();
    expect(glossarySegments("")).toEqual([]);
  });
});
