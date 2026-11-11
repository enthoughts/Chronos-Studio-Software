/**
 * Ribbon.tsx — the JASP-style command ribbon.
 *
 * Part 0 renders the structure; Part 19 makes it a real WAI-ARIA toolbar
 * (§5.15 "keyboard-navigable ribbon"): one tab stop, Arrow/Home/End to move
 * between commands, disabled commands skipped, and "Simplify view" that hides
 * all but the eight essential panels. A disabled command always has a title
 * explaining why (never a silent dead button).
 */

import { useRef, useState } from "react";
import { RIBBON_GROUPS, STAGE_COPY } from "../copy/strings";
import { isEssentialPanel } from "../copy/glossary";
import { ribbonNextIndex } from "../copy/demo";
import { useStore } from "../state/store";
import type { StageCode } from "../state/types";

const NAV_KEYS = ["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown", "Home", "End"];

export function Ribbon({
  onCommand,
  simplifyView = false,
}: {
  /** Return true when the command is handled and should not change the active stage. */
  onCommand?: (id: string, stage: StageCode) => boolean | void;
  /** Part 19: hide all but the eight essential panels for a first-time user. */
  simplifyView?: boolean;
}) {
  const { stages, session, setActiveStage } = useStore();
  const [focusIndex, setFocusIndex] = useState(0);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);

  const statusOf = (stage: StageCode): string =>
    session?.pipeline.stages[stage]?.status ?? "not_started";

  const isAvailable = (stage: StageCode): boolean => {
    if (!session) return stage === "S0";
    const def = stages.find((s) => s.code === stage);
    if (!def) return false;
    return def.depends_on.every((d) => statusOf(d) === "complete");
  };

  const groups = RIBBON_GROUPS
    .map((group) => ({
      ...group,
      items: simplifyView ? group.items.filter((i) => isEssentialPanel(i.id)) : group.items,
    }))
    .filter((group) => group.items.length > 0);

  const flat = groups.flatMap((g) => g.items);
  const companionIds = new Set<string>(["report", "catalog"]);
  for (const g of RIBBON_GROUPS) {
    if (g.id === "assistant") for (const i of g.items) companionIds.add(i.id);
  }
  const enabled = flat.map((item) =>
    companionIds.has(item.id)
      ? Boolean(session)
      : isAvailable(item.stage as StageCode),
  );
  const safeFocus = Math.min(focusIndex, Math.max(0, flat.length - 1));

  let index = -1;
  return (
    <div
      className="ribbon"
      role="toolbar"
      aria-label="Analysis commands"
      aria-orientation="horizontal"
      data-simplify={simplifyView ? "true" : "false"}
      onKeyDown={(e) => {
        if (!NAV_KEYS.includes(e.key)) return;
        const next = ribbonNextIndex(e.key, safeFocus, flat.length, enabled);
        e.preventDefault();
        setFocusIndex(next);
        buttons.current[next]?.focus();
      }}
    >
      {groups.map((group) => (
        <div className="ribbon-group" key={group.id}>
          <div className="ribbon-items">
            {group.items.map((item) => {
              index += 1;
              const myIndex = index;
              const stage = item.stage as StageCode;
              // The report, catalogue, examples and assistant are read-only
              // companions: reachable in a partially analysed session, and
              // offline mode works without a Gemini key. Finalization itself
              // still gates on S14 + S12 in the API.
              const ok = enabled[myIndex];
              const disabled = !ok;
              const title = disabled
                ? !session
                  ? "Start a new analysis first."
                  : `Needs an earlier step first: ${STAGE_COPY[stage]?.label ?? stage}`
                : item.hint;
              return (
                <button
                  key={item.id}
                  ref={(el) => { buttons.current[myIndex] = el; }}
                  className="ribbon-btn"
                  disabled={disabled}
                  title={title}
                  tabIndex={myIndex === safeFocus ? 0 : -1}
                  aria-label={`${item.label}. ${title}`}
                  onFocus={() => setFocusIndex(myIndex)}
                  onClick={() => {
                    const handled = onCommand?.(item.id, stage);
                    if (!handled) setActiveStage(stage);
                  }}
                >
                  <span className="ribbon-glyph" aria-hidden>{item.glyph}</span>
                  <span className="ribbon-btn-label">{item.label}</span>
                </button>
              );
            })}
          </div>
          <div className="ribbon-group-label">{group.label}</div>
        </div>
      ))}
    </div>
  );
}
