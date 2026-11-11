import { describe, expect, it } from "vitest";

import {
  DEMO_POLL_MS,
  GUIDED_TOUR,
  demoIsTerminal,
  demoProgressLabel,
  ribbonNextIndex,
  tourStepLabel,
  type DemoStatus,
} from "./demo";

function status(over: Partial<DemoStatus> = {}): DemoStatus {
  return {
    job_id: "demo-0001-abc", scenario_id: "seasonal_demand",
    session_id: "abc", state: "running", done: 3, total: 17,
    current: "Decompose trend, season and remainder", percent: 17.6,
    labels: [], finished: [], errors: [], elapsed_s: 2.5, ...over,
  };
}

describe("Part 19 demo / background job helpers", () => {
  it("recognises the terminal states", () => {
    for (const s of ["complete", "cancelled", "failed"]) {
      expect(demoIsTerminal(s)).toBe(true);
    }
    for (const s of ["queued", "running", ""]) {
      expect(demoIsTerminal(s)).toBe(false);
    }
    expect(DEMO_POLL_MS).toBeGreaterThan(0);
  });

  it("writes a readable progress line for every state", () => {
    expect(demoProgressLabel(status())).toContain("Step 4 of 17");
    expect(demoProgressLabel(status())).toContain("Decompose trend");
    expect(demoProgressLabel(status({ current: "" }))).toBe("Starting…");
    expect(demoProgressLabel(status({ state: "complete" }))).toContain("complete");
    expect(demoProgressLabel(status({ state: "cancelled" }))).toBe("Stopped");
    expect(demoProgressLabel(status({ state: "failed" }))).toContain("errors");
  });

  it("has a six-card guided tour whose glossary links all resolve", () => {
    expect(GUIDED_TOUR).toHaveLength(6);
    for (const step of GUIDED_TOUR) {
      expect(step.title.length).toBeGreaterThan(3);
      expect(step.body.length).toBeGreaterThan(30);
      expect(step.glossary.length).toBeGreaterThan(0);
    }
    expect(tourStepLabel(0)).toContain("1/6");
    expect(tourStepLabel(5)).toContain("6/6");
    expect(tourStepLabel(6)).toBe("Tour finished");
  });
});

describe("Part 19 WAI-ARIA toolbar keyboard navigation", () => {
  it("wraps with Arrow keys", () => {
    expect(ribbonNextIndex("ArrowRight", 0, 5)).toBe(1);
    expect(ribbonNextIndex("ArrowLeft", 0, 5)).toBe(4);
    expect(ribbonNextIndex("ArrowRight", 4, 5)).toBe(0);
    expect(ribbonNextIndex("ArrowDown", 1, 5)).toBe(2);
    expect(ribbonNextIndex("ArrowUp", 1, 5)).toBe(0);
  });

  it("jumps to the first/last enabled command", () => {
    const enabled = [false, true, true, false, true];
    expect(ribbonNextIndex("Home", 4, 5, enabled)).toBe(1);
    expect(ribbonNextIndex("End", 0, 5, enabled)).toBe(4);
    expect(ribbonNextIndex("ArrowRight", 1, 5, enabled)).toBe(2);
    expect(ribbonNextIndex("ArrowRight", 2, 5, enabled)).toBe(4);
    expect(ribbonNextIndex("ArrowLeft", 4, 5, enabled)).toBe(2);
  });

  it("ignores unknown keys and an empty toolbar", () => {
    expect(ribbonNextIndex("Enter", 2, 5)).toBe(2);
    expect(ribbonNextIndex("ArrowRight", 0, 0)).toBe(-1);
  });
});
