/**
 * Toasts.tsx — the only channel for errors.
 * Errors always carry the backend's plain sentence and the suggested fix (R0.9);
 * the technical detail is available but collapsed.
 */

export type ToastKind = "error" | "info" | "success" | "warn";

import { useState } from "react";
import { useStore } from "../state/store";
import { BEGINNER_COPY } from "../copy/beginner";
import { errorHelp } from "../copy/errors";
import { IconChevron, IconX } from "./Icons";

export function Toasts() {
  const { toasts, dismissToast } = useStore();
  if (toasts.length === 0) return null;

  return (
    <div className="toasts" role="region" aria-live="polite" aria-label="Notifications">
      {toasts.map((t) => (
        <ToastCard
          key={t.id}
          id={t.id}
          kind={t.kind}
          title={t.title}
          body={t.body}
          fix={t.fix}
          errorCode={t.errorCode}
          onClose={() => dismissToast(t.id)}
        />
      ))}
    </div>
  );
}

function ToastCard({
  kind, title, body, fix, errorCode, onClose,
}: {
  id: number;
  kind: ToastKind;
  title: string;
  body?: string;
  fix?: string;
  errorCode?: string;
  onClose: () => void;
}) {
  const [open, setOpen] = useState(false);
  const help = kind === "error" && errorCode ? errorHelp(errorCode) : null;
  const openLibrary = () => {
    window.dispatchEvent(new CustomEvent("chronos:open-error-library", {
      detail: { code: errorCode ?? "" },
    }));
  };
  return (
    <div className={`toast toast-${kind}`} role={kind === "error" ? "alert" : "status"}>
      <button className="toast-close" onClick={onClose} aria-label="Dismiss notification">
        <IconX size={13} />
      </button>
      <div className="toast-title">{title}</div>
      {fix && <div className="toast-fix">{fix}</div>}
      {help && (
        <div className="toast-help">
          <strong>{BEGINNER_COPY.errorToastMeaning}:</strong> {help.meaning}{" "}
          <button className="btn btn-sm btn-ghost" onClick={openLibrary}>
            {BEGINNER_COPY.errorToastLink}
          </button>
        </div>
      )}
      {body && (
        <>
          <button
            className="btn btn-sm btn-ghost"
            style={{ marginTop: 6, paddingLeft: 0 }}
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
          >
            <IconChevron size={12} className={open ? "rot180" : undefined} />
            {open ? "Hide detail" : "Technical detail"}
          </button>
          {open && (
            <pre style={{ whiteSpace: "pre-wrap", margin: "4px 0 0", color: "var(--ink-faint)" }}>
              {body}
            </pre>
          )}
        </>
      )}
    </div>
  );
}
