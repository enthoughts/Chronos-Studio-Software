import { describe, expect, it } from "vitest";

import { applyGate, overrideResetKey } from "./nextAction";
import type { ActionPreview, NextRecommendation } from "../state/types";

function rec(id: string, op: string | null = "difference"): NextRecommendation {
  return { id, op, op_params: { order: "seasonal" } } as unknown as NextRecommendation;
}

function preview(verdict: ActionPreview["verdict"], plain = "worse"): ActionPreview {
  return {
    verdict,
    plain,
    before: { n: 577, sd: 1, r1: 0.902, decay: 4, n_sig: 8, lb_p: 0, lb_stat: 1, vr: null, band: 0.1 },
    after: { n: 565, sd: 1, r1: 0.849, decay: 8, n_sig: 8, lb_p: 0, lb_stat: 1, vr: null, band: 0.1 },
    n_lost: 12,
  } as ActionPreview;
}

describe("S7 apply gate", () => {
  it("locks a worse preview until the user acknowledges it", () => {
    const worse = preview("worse");
    expect(applyGate(rec("next.diff.seasonal_only"), worse, false)).toEqual({
      canApply: false, requiresOverride: true,
    });
    expect(applyGate(rec("next.diff.seasonal_only"), worse, true)).toEqual({
      canApply: true, requiresOverride: false,
    });
  });

  it("keeps non-worse and unavailable previews applyable, but never a note without an op", () => {
    expect(applyGate(rec("next.transform.log"), preview("better"), false).canApply).toBe(true);
    expect(applyGate(rec("next.diff.first"), preview("neutral"), false).canApply).toBe(true);
    expect(applyGate(rec("next.diff.first"), preview("unavailable"), false).canApply).toBe(true);
    expect(applyGate(rec("next.note.both_differences", null), null, false).canApply).toBe(false);
  });

  it("resets the acknowledgement when the recommendation or its preview changes", () => {
    const base = rec("next.diff.seasonal_only");
    const first = preview("worse", "ACF still significant through lag 8");
    const same = preview("worse", "ACF still significant through lag 8");
    expect(overrideResetKey(base, first)).toBe(overrideResetKey(base, same));
    expect(overrideResetKey(base, first)).not.toBe(
      overrideResetKey(rec("next.diff.first"), first),
    );
    expect(overrideResetKey(base, first)).not.toBe(
      overrideResetKey(base, preview("worse", "different preview")),
    );
    const changedNumbers = preview("worse", "ACF still significant through lag 8");
    changedNumbers.after!.r1 = 0.5;
    expect(overrideResetKey(base, first)).not.toBe(overrideResetKey(base, changedNumbers));
  });
});
