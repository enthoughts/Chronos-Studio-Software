import type { ActionPreview, NextRecommendation } from "../state/types";

export interface ApplyGate {
  canApply: boolean;
  requiresOverride: boolean;
}

/**
 * A server-side `worse` preview is a warning, not a dead end. Apply is enabled
 * only after the user explicitly acknowledges that warning for this exact
 * recommendation and preview.
 */
export function applyGate(
  rec: Pick<NextRecommendation, "op">,
  preview: Pick<ActionPreview, "verdict"> | null | undefined,
  acknowledged: boolean,
): ApplyGate {
  const worse = preview?.verdict === "worse";
  const hasOp = Boolean(rec.op);
  return {
    canApply: hasOp && (!worse || acknowledged),
    requiresOverride: hasOp && worse && !acknowledged,
  };
}

/**
 * Reset the acknowledgement whenever the card or its preview changes. Two
 * different `worse` recommendations must not share one ticked checkbox.
 */
export function overrideResetKey(
  rec: Pick<NextRecommendation, "id" | "op" | "op_params">,
  preview: ActionPreview | null | undefined,
): string {
  const before = preview?.before;
  const after = preview?.after;
  return [
    rec.id,
    rec.op ?? "",
    JSON.stringify(rec.op_params ?? {}),
    preview?.verdict ?? "",
    preview?.plain ?? "",
    preview?.n_lost ?? "",
    before?.r1 ?? "", before?.decay ?? "", before?.lb_p ?? "",
    after?.r1 ?? "", after?.decay ?? "", after?.lb_p ?? "",
  ].join("|");
}
