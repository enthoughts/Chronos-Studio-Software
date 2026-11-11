/**
 * DemoPanel.tsx — one-click worked examples, progress + cancel, guided tour.
 *
 * Part 19 / §5.15:
 *   - "Example analyses" that replay a full study step by step;
 *   - heavy models as background jobs with progress and cancel;
 *   - a guided tour for a first-time user.
 *
 * The browser only polls a read-only status endpoint; all computation happens
 * on the backend worker thread. No API host is hardcoded.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api, rememberSession } from "../state/api";
import {
  DEMO_POLL_MS,
  GUIDED_TOUR,
  demoIsTerminal,
  demoProgressLabel,
  type DemoScenario,
  type DemoStatus,
} from "../copy/demo";
import { GlossaryTip } from "./GlossaryTip";

const COPY = {
  eyebrow: "Part 19 · demo mode",
  title: "Watch a complete analysis",
  intro:
    "Choose a worked study. Chronos replays every step — from loading the data to a manuscript-ready report — in the background. You can stop it at any time.",
  choose: "1 · Choose a study",
  run: "2 · Watch it run",
  start: "Run this example",
  cancel: "Cancel",
  exploring: "Explore the analysis",
  close: "Close",
  failed: "The example stopped early. The individual step errors are listed below.",
  tourTitle: "New here? Take the six-step tour",
  tourNext: "Next",
  tourBack: "Back",
  tourDone: "Finish tour",
  privacy: "Privacy: sessions live in server memory for a few hours only. Delete session removes everything immediately.",
};

export function DemoPanel({ onClose }: { onClose: () => void }) {
  const [scenarios, setScenarios] = useState<DemoScenario[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [status, setStatus] = useState<DemoStatus | null>(null);
  const [error, setError] = useState("");
  const [loadError, setLoadError] = useState("");
  const [tourIndex, setTourIndex] = useState<number | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let live = true;
    void api.demoScenarios()
      .then((r) => {
        if (!live) return;
        setScenarios(r.scenarios);
        setSelected((prev) => prev || r.scenarios[0]?.id || "");
      })
      .catch((e) => setLoadError(e instanceof ApiError ? e.message : "Could not load the examples."));
    return () => { live = false; };
  }, []);

  const stopPolling = useCallback(() => {
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  const poll = useCallback((jobId: string, immediate?: DemoStatus) => {
    stopPolling();
    if (immediate) setStatus(immediate);
    const tick = async () => {
      try {
        const s = await api.demoStatus(jobId);
        setStatus(s);
        if (demoIsTerminal(s.state)) stopPolling();
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "The progress check failed.");
        stopPolling();
      }
    };
    timer.current = window.setInterval(() => void tick(), DEMO_POLL_MS);
  }, [stopPolling]);

  const start = async () => {
    setError("");
    setStatus(null);
    try {
      const s = await api.demoStart(selected);
      poll(s.job_id, s);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The example could not be started.");
    }
  };

  const cancel = async () => {
    if (!status) return;
    try {
      setStatus(await api.demoCancel(status.job_id));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not stop the example.");
    }
  };

  const explore = () => {
    if (!status) return;
    rememberSession(status.session_id);
    window.location.reload();
  };

  const running = Boolean(status && !demoIsTerminal(status.state));
  const chosen = scenarios.find((s) => s.id === selected);
  const tour = tourIndex === null ? null : GUIDED_TOUR[tourIndex];

  return (
    <div className="modal-backdrop" onMouseDown={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <section className="demo-dialog" role="dialog" aria-modal="true"
               aria-labelledby="demo-dialog-title">
        <header className="glossary-dialog-head">
          <div>
            <div className="pane-title">{COPY.eyebrow}</div>
            <h2 id="demo-dialog-title">{COPY.title}</h2>
            <p>{COPY.intro}</p>
          </div>
          <button className="btn btn-sm" onClick={onClose} aria-label={COPY.close}>✕ {COPY.close}</button>
        </header>

        {loadError && <div className="callout callout-bad" role="alert">{loadError}</div>}
        {error && <div className="callout callout-bad" role="alert">{error}</div>}

        <h3 className="demo-section-title">{COPY.choose}</h3>
        <div className="demo-scenario-grid">
          {scenarios.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`demo-scenario ${selected === s.id ? "is-selected" : ""}`}
              aria-pressed={selected === s.id}
              onClick={() => setSelected(s.id)}
              disabled={running}
            >
              <strong>{s.title}</strong>
              <small>{s.n_steps} steps · {s.models.length} models · horizon {s.horizon}</small>
              <span>{s.blurb}</span>
              <em>{s.teaches}</em>
            </button>
          ))}
        </div>

        {chosen && (
          <details className="demo-steps">
            <summary>Exactly what will run ({chosen.steps.length} steps)</summary>
            <ol>
              {chosen.steps.map((label, i) => (
                <li key={`${i}-${label}`}><span className="demo-stage-chip">{chosen.stages[i]}</span> {label}</li>
              ))}
            </ol>
          </details>
        )}

        <h3 className="demo-section-title">{COPY.run}</h3>
        {!status && (
          <button className="btn btn-primary" onClick={() => void start()}
                  disabled={!selected || Boolean(loadError)}>
            ▶ {COPY.start}
          </button>
        )}

        {status && (
          <div className="demo-progress" aria-live="polite" aria-atomic="true">
            <div className="demo-progress-head">
              <strong>{demoProgressLabel(status)}</strong>
              <span>{status.done}/{status.total} · {status.percent}% · {status.elapsed_s}s</span>
            </div>
            <div className="demo-bar" role="progressbar" aria-valuemin={0}
                 aria-valuemax={status.total} aria-valuenow={status.done}
                 aria-label="Example analysis progress">
              <div className="demo-bar-fill" style={{ width: `${status.percent}%` }} />
            </div>
            <ol className="demo-step-list">
              {status.finished.map((f, i) => (
                <li key={`${i}-${f.label}`} className={f.status === "ok" ? "ok" : "bad"}>
                  <span aria-hidden>{f.status === "ok" ? "✓" : "!"}</span>
                  <span>{f.stage} · {f.label}</span>
                  <small>{f.seconds.toFixed(2)}s</small>
                </li>
              ))}
            </ol>
            {status.errors.length > 0 && (
              <div className="callout callout-bad" role="alert">
                {COPY.failed}
                <ul>{status.errors.map((x, i) => <li key={i}>{x}</li>)}</ul>
              </div>
            )}
            <div className="demo-actions">
              {running && <button className="btn" onClick={() => void cancel()}>■ {COPY.cancel}</button>}
              {status.state === "complete" && (
                <button className="btn btn-primary" onClick={explore}>↗ {COPY.exploring}</button>
              )}
              {demoIsTerminal(status.state) && (
                <button className="btn btn-sm" onClick={() => { setStatus(null); stopPolling(); }}>
                  Run another
                </button>
              )}
            </div>
          </div>
        )}

        <section className="demo-tour">
          <h3 className="demo-section-title">{COPY.tourTitle}</h3>
          {tour === null ? (
            <button className="btn btn-sm" onClick={() => setTourIndex(0)}>Start tour</button>
          ) : (
            <div className="demo-tour-card">
              <div className="demo-tour-count">{tourIndex! + 1} / {GUIDED_TOUR.length}</div>
              <h4>{tour.title}</h4>
              <p>{tour.body}</p>
              <p className="demo-tour-term">
                <GlossaryTip term={tour.glossary} />
              </p>
              <div className="demo-tour-actions">
                <button className="btn btn-sm" onClick={() => setTourIndex(Math.max(0, tourIndex! - 1))}
                        disabled={tourIndex === 0}>← {COPY.tourBack}</button>
                <button className="btn btn-sm" onClick={() => setTourIndex(null)}>{COPY.close}</button>
                <button className="btn btn-sm btn-primary"
                        onClick={() => setTourIndex(tourIndex! + 1 >= GUIDED_TOUR.length ? null : tourIndex! + 1)}>
                  {tourIndex! + 1 >= GUIDED_TOUR.length ? COPY.tourDone : `${COPY.tourNext} →`}
                </button>
              </div>
            </div>
          )}
        </section>

        <p className="demo-privacy">{COPY.privacy} <GlossaryTip term="session_ttl" /></p>
      </section>
    </div>
  );
}
