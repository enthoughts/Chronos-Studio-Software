import { describe, expect, it } from "vitest";

import {
  DEFAULT_PANEL_HELP,
  PANEL_ALIASES,
  PANEL_HELP,
  panelHelp,
  stepNavigation,
} from "./beginner";
import { GLOSSARY } from "./glossary";

describe("Part 19 per-panel beginner help", () => {
  it("explains all sixteen pipeline stages with real glossary slugs", () => {
    expect(Object.keys(PANEL_HELP)).toHaveLength(16);
    for (const [stage, help] of Object.entries(PANEL_HELP)) {
      expect(stage).toMatch(/^S(?:[0-9]|1[0-5])$/);
      expect(help.summary.length).toBeGreaterThan(35);
      expect(help.reading.length).toBeGreaterThan(50);
      expect(help.tryThis.length).toBeGreaterThan(35);
      expect(help.watchOut.length).toBeGreaterThan(35);
      expect(help.terms.length).toBeGreaterThanOrEqual(4);
      for (const slug of help.terms) {
        expect(GLOSSARY[slug], `${stage} references missing term ${slug}`).toBeDefined();
      }
    }
    for (const slug of DEFAULT_PANEL_HELP.terms) expect(GLOSSARY[slug]).toBeDefined();
  });

  it("resolves ribbon ids to the stage that explains them", () => {
    expect(panelHelp("upload")).toBe(PANEL_HELP.S0);
    expect(panelHelp("examples")).toBe(PANEL_HELP.S0);
    expect(panelHelp("decompose")).toBe(PANEL_HELP.S4);
    expect(panelHelp("forecast")).toBe(PANEL_HELP.S13);
    expect(panelHelp("report")).toBe(PANEL_HELP.S15);
    expect(panelHelp("S12")).toBe(PANEL_HELP.S12);
    expect(panelHelp("unknown-panel")).toBe(DEFAULT_PANEL_HELP);
    for (const [id, stage] of Object.entries(PANEL_ALIASES)) {
      expect(PANEL_HELP[stage], `${id} points to a missing stage`).toBeDefined();
    }
  });

  it("computes back/next navigation including edge cases", () => {
    const codes = ["S0", "S1", "S2"];
    expect(stepNavigation(codes, "S0")).toEqual({ previous: null, next: "S1", position: 1, total: 3 });
    expect(stepNavigation(codes, "S1")).toEqual({ previous: "S0", next: "S2", position: 2, total: 3 });
    expect(stepNavigation(codes, "S2")).toEqual({ previous: "S1", next: null, position: 3, total: 3 });
    expect(stepNavigation(codes, "unknown").next).toBe("S1");
    expect(stepNavigation([], "S0")).toEqual({ previous: null, next: null, position: 1, total: 0 });
  });
});
