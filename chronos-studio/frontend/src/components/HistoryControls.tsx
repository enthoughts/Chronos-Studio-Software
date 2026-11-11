/**
 * HistoryControls.tsx — Part 19 undo/redo UI.
 *
 * The API already replays preprocessing from the untouched original series and
 * clears the redo stack when a new branch is applied. These controls surface
 * `can_undo` / `can_redo` globally (top bar and Step 3 options pane), with
 * disabled reasons and familiar Ctrl/Cmd+Z shortcuts.
 */

import { useEffect } from "react";
import { BEGINNER_COPY } from "../copy/beginner";
import { useStore } from "../state/store";

export function HistoryControls({ compact = false }: { compact?: boolean }) {
  const { history, historyBusy, undoHistory, redoHistory, refreshHistory } = useStore();
  const canUndo = history.can_undo && !historyBusy;
  const canRedo = history.can_redo && !historyBusy;

  useEffect(() => {
    void refreshHistory();
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true']")) return;
      if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === "z" && !e.shiftKey && canUndo) {
        e.preventDefault();
        void undoHistory();
      } else if ((key === "z" && e.shiftKey) || key === "y") {
        if (canRedo) {
          e.preventDefault();
          void redoHistory();
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [refreshHistory, undoHistory, redoHistory, canUndo, canRedo]);

  return (
    <div className={`history-controls ${compact ? "is-compact" : ""}`} role="group" aria-label={BEGINNER_COPY.historyTitle}>
      <button
        className="btn btn-sm"
        onClick={() => void undoHistory()}
        disabled={!canUndo}
        aria-keyshortcuts="Control+Z Meta+Z"
        aria-label={BEGINNER_COPY.undoLabel}
        title={canUndo ? BEGINNER_COPY.undoLabel : BEGINNER_COPY.noUndo}
        data-testid="history-undo"
      >
        <span aria-hidden>↶</span>{!compact && <span> {BEGINNER_COPY.undo}</span>}
      </button>
      <button
        className="btn btn-sm"
        onClick={() => void redoHistory()}
        disabled={!canRedo}
        aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z Control+Y"
        aria-label={BEGINNER_COPY.redoLabel}
        title={canRedo ? BEGINNER_COPY.redoLabel : BEGINNER_COPY.noRedo}
        data-testid="history-redo"
      >
        <span aria-hidden>↷</span>{!compact && <span> {BEGINNER_COPY.redo}</span>}
      </button>
      {!compact && (
        <small className="history-version" aria-live="polite">
          series v{history.series_version}
        </small>
      )}
    </div>
  );
}
