import { describe, expect, it } from "vitest";

import {
  GLOSSARY,
  GLOSSARY_CATEGORIES,
  GLOSSARY_TERM_COUNT,
  SIMPLIFY_ESSENTIAL_PANELS,
  isEssentialPanel,
  lookupGlossary,
} from "./glossary";

describe("Part 19 glossary (§5.15 requires 120+ terms)", () => {
  it("has well over 120 unique, non-empty definitions", () => {
    expect(GLOSSARY_TERM_COUNT).toBeGreaterThanOrEqual(120);
    const entries = Object.values(GLOSSARY);
    expect(entries).toHaveLength(GLOSSARY_TERM_COUNT);
    expect(new Set(entries.map((e) => e.slug)).size).toBe(entries.length);
    for (const e of entries) {
      expect(e.term.length).toBeGreaterThan(0);
      expect(e.definition.length).toBeGreaterThan(20);
      expect(e.category.length).toBeGreaterThan(0);
      expect(e.slug).toBe(e.slug.toLowerCase());
    }
  });

  it("keeps the categories consistent with the entries", () => {
    const used = new Set(Object.values(GLOSSARY).map((e) => e.category));
    expect(new Set(GLOSSARY_CATEGORIES)).toEqual(used);
    expect(GLOSSARY_CATEGORIES.length).toBeGreaterThanOrEqual(8);
  });

  it("looks up by slug, display term, case and loose phrase", () => {
    expect(lookupGlossary("mase")?.term).toBe("MASE");
    expect(lookupGlossary("MASE")?.slug).toBe("mase");
    expect(lookupGlossary("  Time series ")?.slug).toBe("time_series");
    expect(lookupGlossary("Prediction interval")?.slug).toBe("prediction_interval");
    expect(lookupGlossary("prediction_interval")?.category).toBe("Forecasting");
    expect(lookupGlossary("no_such_term_xyz")).toBeUndefined();
    expect(lookupGlossary("")).toBeUndefined();
  });

  it("documents the key statistical terms a beginner meets", () => {
    for (const slug of [
      "stationarity", "unit_root", "adf", "kpss", "acf", "pacf", "mase",
      "crps", "pit", "rqr", "prediction_interval", "rolling_origin_cv",
      "overdispersion", "golden_file", "wcag", "demo_mode",
    ]) {
      expect(GLOSSARY[slug], `missing glossary term ${slug}`).toBeDefined();
    }
  });

  it("Simplify view keeps exactly the eight essential panels", () => {
    expect(SIMPLIFY_ESSENTIAL_PANELS).toHaveLength(8);
    expect(new Set(SIMPLIFY_ESSENTIAL_PANELS).size).toBe(8);
    expect(isEssentialPanel("upload")).toBe(true);
    expect(isEssentialPanel("report")).toBe(true);
    expect(isEssentialPanel("garch")).toBe(false);
    expect(isEssentialPanel("")).toBe(false);
  });
});
