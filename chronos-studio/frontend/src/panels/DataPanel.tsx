/**
 * panels/DataPanel.tsx — S0: load a dataset.
 *
 * Three ways in, all equally legitimate: upload a file, paste a table, or open a
 * teaching example. After loading, the user sees what we understood (time column,
 * value column, calendar, data class) and can override every guess.
 *
 * Beginner rule: never show a form before showing a guess. The mapping editor is
 * collapsed until the user asks for it, or until our confidence is low.
 */

import { useEffect, useRef, useState } from "react";
import { ApiError, api } from "../state/api";
import { useStore } from "../state/store";
import type {
  CalendarOption,
  ColumnMapping,
  DataQuality,
  ExampleDataset,
  IngestResponse,
} from "../state/types";
import { IconChart, IconCheck, IconUpload } from "../components/Icons";

type Tab = "upload" | "paste" | "examples";

/**
 * The panel is mounted inside the S0 view, which re-mounts when the first upload
 * creates the session. Caching the most recent ingest result at module level
 * keeps the "what we understood" card and the mapping editor visible across that
 * transition — the alternative is losing the user's context at the worst moment.
 */
let LAST_INGEST: IngestResponse | null = null;

export function DataPanel() {
  const { sessionId, session, refresh, startSession, pushToast } = useStore();
  const [tab, setTab] = useState<Tab>("examples");
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<IngestResponse | null>(LAST_INGEST);
  const remember = (r: IngestResponse | null) => {
    LAST_INGEST = r;
    setLast(r);
    if (r) setForcedLoader(false);
  };
  const [examples, setExamples] = useState<ExampleDataset[]>([]);
  const [calendars, setCalendars] = useState<CalendarOption[]>([]);
  const [extensions, setExtensions] = useState<string[]>([]);
  const [pasteText, setPasteText] = useState("");
  const [showMapping, setShowMapping] = useState(false);
  const [forcedLoader, setForcedLoader] = useState(false);

  useEffect(() => {
    void (async () => {
      try {
        const [ex, fmt] = await Promise.all([api.examples(), api.dataFormats()]);
        setExamples(ex.examples);
        setCalendars(fmt.calendars);
        setExtensions(fmt.extensions);
      } catch {
        /* the panel still works for upload even if metadata fails */
      }
    })();
  }, []);

  const withSession = async (fn: () => Promise<void>) => {
    if (!sessionId) await startSession();
    setBusy(true);
    try {
      await fn();
      await refresh();
    } catch (e) {
      if (e instanceof ApiError) {
        pushToast({ kind: "error", title: e.message, body: e.code, fix: e.fix });
      } else {
        pushToast({
          kind: "error",
          title: "That did not work.",
          fix: "Try a different file, or export it as CSV first.",
        });
      }
    } finally {
      setBusy(false);
    }
  };

  const loadExample = (id: string) =>
    withSession(async () => {
      const res = await api.loadExample(id);
      remember(res);
      pushToast({ kind: "success", title: `Loaded ${res.dataset.name}` });
    });

  const uploadFile = (file: File) =>
    withSession(async () => {
      const res = await api.upload(file);
      remember(res);
      pushToast({
        kind: "success",
        title: `${res.dataset.name} loaded`,
        body: `${res.dataset.n_observations} observations · ${res.dataset.frequency.label}`,
      });
    });

  const pasteTable = () =>
    withSession(async () => {
      const res = await api.paste(pasteText);
      remember(res);
      pushToast({ kind: "success", title: "Pasted table loaded" });
    });

  // `last` is this instance's state; LAST_INGEST survives the remount that happens
  // when the first upload creates the session. Prefer whichever is present.
  const effective: IngestResponse | null = last ?? LAST_INGEST;
  const dataset = forcedLoader ? null : (effective?.dataset ?? session?.dataset ?? null);
  const timeCol =
    effective?.mapping.time_column ??
    (effective?.mapping.year_column
      ? `${effective.mapping.year_column} + ${effective.mapping.month_column ?? "—"}`
      : null) ??
    session?.dataset?.time_column ??
    "—";
  const quality = effective?.dataset.quality ?? session?.dataset?.quality;

  return (
    <div>
      {!dataset && (
        <>
          <div role="tablist" aria-label="How to load data" style={{ display: "flex", gap: 6, marginBottom: 12 }}>
            {(
              [
                ["examples", "Example data"],
                ["upload", "Upload a file"],
                ["paste", "Paste a table"],
              ] as [Tab, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                className={`btn btn-sm ${tab === id ? "btn-primary" : ""}`}
                onClick={() => setTab(id)}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "upload" && (
            <DropZone
              extensions={extensions}
              busy={busy}
              onFile={(f) => void uploadFile(f)}
            />
          )}

          {tab === "paste" && (
            <div className="card">
              <div className="card-head">
                <span className="card-kind">Paste</span>
                <div>
                  <div className="card-title">Paste a table</div>
                  <div className="card-plain">
                    Copy cells straight from Excel, Google Sheets or a PDF table — include the header row.
                  </div>
                </div>
              </div>
              <div className="card-body">
                <textarea
                  value={pasteText}
                  onChange={(e) => setPasteText(e.target.value)}
                  rows={8}
                  placeholder={"Month\tSales\n2020-01\t120\n2020-02\t135"}
                  style={{
                    width: "100%", fontFamily: "var(--mono)", fontSize: 12.5,
                    padding: 10, borderRadius: 6, border: "1px solid var(--rule-strong)",
                    background: "var(--paper)", resize: "vertical",
                  }}
                  aria-label="Pasted table"
                />
                <button
                  className="btn btn-primary"
                  style={{ marginTop: 8 }}
                  disabled={busy || pasteText.trim().length === 0}
                  onClick={() => void pasteTable()}
                >
                  Analyse this table
                </button>
              </div>
            </div>
          )}

          {tab === "examples" && (
            <div className="card">
              <div className="card-head">
                <span className="card-kind">Examples</span>
                <div>
                  <div className="card-title">Teaching datasets</div>
                  <div className="card-plain">
                    Each one is chosen to break a specific habit — or to show why a method exists.
                  </div>
                </div>
              </div>
              <div className="card-body" style={{ padding: 0 }}>
                {examples.length === 0 && <div className="empty" style={{ padding: 24 }}>Loading examples…</div>}
                {examples.map((ex) => (
                  <button
                    key={ex.id}
                    onClick={() => void loadExample(ex.id)}
                    disabled={busy || !ex.available}
                    style={{
                      display: "block", width: "100%", textAlign: "left",
                      padding: "11px 14px", border: "none", cursor: "pointer",
                      borderBottom: "1px solid var(--rule)", background: "transparent",
                    }}
                  >
                    <div style={{ display: "flex", gap: 10, alignItems: "baseline" }}>
                      <IconChart size={14} />
                      <strong style={{ fontSize: 13 }}>{ex.title}</strong>
                      <span className="pill" style={{ marginLeft: "auto" }}>
                        n={ex.n_rows} · {ex.data_class.replace("_", " ")}
                      </span>
                    </div>
                    <div style={{ fontSize: 12, color: "var(--ink-soft)", marginTop: 3, maxWidth: 760 }}>
                      {ex.teaches}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {dataset && (
        <>
          <div className="callout callout-ok" style={{ marginTop: 0 }}>
            <strong>
              <IconCheck size={13} /> {dataset.name} loaded.
            </strong>{" "}
            {dataset.n_observations} observations · {dataset.frequency.label} ·{" "}
            {dataset.time_start?.slice(0, 10)} → {dataset.time_end?.slice(0, 10)} ·{" "}
            hash <code>{dataset.content_hash}</code>
          </div>

          {quality && <QualityCard quality={quality} />}

          <div className="card">
            <div className="card-head">
              <span className="card-kind">Understood</span>
              <div>
                <div className="card-title">What we think your data is</div>
                <div className="card-plain">
                  Every guess below can be changed. Nothing downstream is final until you accept it.
                </div>
              </div>
              <div className="card-actions">
                <button className="btn btn-sm" onClick={() => setShowMapping((v) => !v)} aria-expanded={showMapping}>
                  {showMapping ? "Hide" : "Change mapping"}
                </button>
              </div>
            </div>
            <div className="card-body">
              <FactGrid
                items={[
                  ["Time column", String(timeCol)],
                  ["Value column", dataset.target_column],
                  ["Calendar", `${dataset.frequency.label} (period ${dataset.frequency.period ?? "—"})`],
                  ["Data class", dataset.data_class.replace(/_/g, " ")],
                  ["Exogenous", dataset.exogenous_columns.length ? dataset.exogenous_columns.join(", ") : "none"],
                  ["Rows read", `${quality?.n_rows_raw ?? "—"} → ${quality?.n_rows_used ?? "—"} used`],
                ]}
              />
              {effective && effective.cross_sell.length > 0 && (
                <p style={{ fontSize: 12.5, color: "var(--ink-soft)", marginTop: 10 }}>
                  Other numeric columns in this file: <strong>{effective.cross_sell.join(", ")}</strong> — pick one
                  in <em>Change mapping</em> to analyse it instead.
                </p>
              )}
              {showMapping && effective && (
                <MappingEditor
                  ingest={effective}
                  calendars={calendars}
                  onApplied={(res) => {
                    remember(res);
                    void refresh();
                    pushToast({ kind: "success", title: "Column mapping updated." });
                  }}
                  onError={(e) =>
                    pushToast({ kind: "error", title: e.message, body: e.code, fix: e.fix })
                  }
                />
              )}
            </div>
          </div>

          {dataset.preview_rows.length > 0 && (
            <div className="card">
              <div className="card-head">
                <span className="card-kind">Preview</span>
                <div>
                  <div className="card-title">First rows, as the app sees them</div>
                  <div className="card-plain">
                    Blank values are real gaps on the calendar — they are not shifted rows.
                  </div>
                </div>
              </div>
              <div className="card-body">
                <div style={{ overflowX: "auto" }}>
                  <table className="data">
                    <thead>
                      <tr>
                        <th className="num">time</th>
                        <th className="num">
                          {dataset.data_class === "event_timestamp"
                            ? "events"
                            : dataset.target_column}
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {dataset.preview_rows.map((r, i) => (
                        <tr key={i}>
                          <td className="num">{String(r.time).slice(0, 19)}</td>
                          <td className="num">
                            {r.value === null || r.value === undefined
                              ? "—"
                              : Number(r.value).toLocaleString(undefined, { maximumFractionDigits: 3 })}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
            <button
              className="btn"
              onClick={() => {
                remember(null);
                setForcedLoader(true);
                setShowMapping(false);
              }}
            >
              Load different data
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ parts */

function DropZone({
  extensions, busy, onFile,
}: {
  extensions: string[];
  busy: boolean;
  onFile: (f: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div
      className="card"
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
      style={{ borderStyle: over ? "dashed" : "solid", borderColor: over ? "var(--brand)" : undefined }}
    >
      <div className="card-body" style={{ textAlign: "center", padding: 34 }}>
        <IconUpload size={26} />
        <h2 style={{ marginTop: 10 }}>Drop a file here</h2>
        <p style={{ fontSize: 12.5, color: "var(--ink-soft)", marginTop: 4 }}>
          or{" "}
          <button className="btn btn-sm" onClick={() => inputRef.current?.click()} disabled={busy}>
            choose a file
          </button>
        </p>
        <input
          ref={inputRef}
          type="file"
          hidden
          accept={extensions.map((e) => `.${e.toLowerCase()}`).join(",")}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
          }}
        />
        <p style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 14 }}>
          {extensions.join(" · ")} — up to 64 MB. Nothing is stored on the server after your session.
        </p>
        {busy && <p style={{ fontSize: 12, marginTop: 10 }}>Reading the file…</p>}
      </div>
    </div>
  );
}

function FactGrid({ items }: { items: [string, string][] }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))",
        gap: 10,
      }}
    >
      {items.map(([label, value]) => (
        <div key={label}>
          <div className="pane-title" style={{ marginBottom: 1 }}>{label}</div>
          <div style={{ fontSize: 13, fontFamily: "var(--mono)" }}>{value}</div>
        </div>
      ))}
    </div>
  );
}

function QualityCard({ quality }: { quality: DataQuality }) {
  const errors = quality.issues.filter((i) => i.severity === "error");
  const warns = quality.issues.filter((i) => i.severity === "warning");
  const infos = quality.issues.filter((i) => i.severity === "info");

  return (
    <div className="card">
      <div className="card-head">
        <span className={`card-kind`} style={{ background: errors.length ? "var(--bad-wash)" : warns.length ? "var(--warn-wash)" : "var(--ok-wash)", color: errors.length ? "var(--bad)" : warns.length ? "var(--warn)" : "var(--ok)" }}>
          Quality
        </span>
        <div>
          <div className="card-title">Data quality report</div>
          <div className="card-plain">{quality.plain_summary}</div>
        </div>
      </div>
      <div className="card-body">
        <FactGrid
          items={[
            [
              "Missing",
              `${quality.n_missing_total} (${(
                (quality.missing_by_column[0]?.pct_missing ?? 0) * 100
              ).toFixed(1)}%)`,
            ],
            ["Gaps inserted", String(quality.n_gaps)],
            ["Duplicates merged", String(quality.n_duplicate_timestamps)],
            ["Zeros", `${quality.n_zeros} (${(quality.pct_zeros * 100).toFixed(1)}%)`],
            ["Negative values", String(quality.n_negatives)],
            ["Outlier candidates", String(quality.n_outlier_candidates)],
          ]}
        />

        {quality.data_class !== "continuous" && (
          <div className="callout callout-warn" style={{ marginTop: 12 }}>
            <strong>Data class: {quality.data_class.replace(/_/g, " ")}.</strong>{" "}
            This changes which models the app will offer and which accuracy metrics are valid.
            You can override it in <em>Change mapping</em>.
          </div>
        )}

        {[...errors, ...warns].length > 0 && (
          <ul style={{ listStyle: "none", padding: 0, margin: "10px 0 0" }}>
            {[...errors, ...warns].map((i, n) => (
              <li
                key={`${i.code}-${n}`}
                style={{
                  display: "flex", gap: 8, padding: "6px 0",
                  borderBottom: "1px solid var(--rule)", fontSize: 12.5,
                }}
              >
                <span
                  style={{
                    flex: "none", marginTop: 4, width: 7, height: 7, borderRadius: "50%",
                    background: i.severity === "error" ? "var(--bad)" : "var(--warn)",
                  }}
                  aria-hidden
                />
                <span>
                  {i.plain}
                  {i.fix && (
                    <span style={{ display: "block", color: "var(--ink-faint)", fontSize: 11.5 }}>
                      {i.fix}
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}

        {infos.length > 0 && (
          <details style={{ marginTop: 10 }}>
            <summary style={{ cursor: "pointer", fontSize: 12, color: "var(--ink-soft)" }}>
              {infos.length} informational note(s)
            </summary>
            <ul style={{ margin: "6px 0 0 16px", padding: 0, fontSize: 12, color: "var(--ink-soft)" }}>
              {infos.map((i, n) => (
                <li key={`${i.code}-${n}`}>{i.plain}</li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}

function MappingEditor({
  ingest, calendars, onApplied, onError,
}: {
  ingest: IngestResponse;
  calendars: CalendarOption[];
  onApplied: (r: IngestResponse) => void;
  onError: (e: ApiError) => void;
}) {
  const columns = ingest.profiles.map((p) => p.name);
  const [mapping, setMapping] = useState<ColumnMapping>({
    ...ingest.mapping,
    exogenous_columns: ingest.mapping.exogenous_columns ?? [],
  });
  const [busy, setBusy] = useState(false);

  const apply = async () => {
    setBusy(true);
    try {
      onApplied(await api.remap(mapping));
    } catch (e) {
      onError(
        e instanceof ApiError
          ? e
          : new ApiError(0, "internal.error", "That mapping did not work.", "Pick another column."),
      );
    } finally {
      setBusy(false);
    }
  };

  const field = (label: string, node: React.ReactNode, hint?: string) => (
    <label style={{ display: "block", marginBottom: 10 }}>
      <span className="pane-title" style={{ display: "block", marginBottom: 3 }}>{label}</span>
      {node}
      {hint && <span style={{ fontSize: 11, color: "var(--ink-faint)" }}>{hint}</span>}
    </label>
  );

  const selectStyle = {
    width: "100%", height: 30, borderRadius: 6, padding: "0 8px",
    border: "1px solid var(--rule-strong)", background: "var(--paper)",
  } as const;

  return (
    <div style={{ marginTop: 14, paddingTop: 14, borderTop: "1px solid var(--rule)" }}>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 14 }}>
        {field(
          "Time column",
          <select
            style={selectStyle}
            value={mapping.time_column ?? ""}
            onChange={(e) => setMapping({ ...mapping, time_column: e.target.value, year_column: null, month_column: null })}
          >
            <option value="">— rows are evenly spaced —</option>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>,
          "Leave empty only if the rows have no dates at all.",
        )}
        {field(
          "Value column",
          <select
            style={selectStyle}
            value={mapping.target_column ?? ""}
            onChange={(e) => setMapping({ ...mapping, target_column: e.target.value || null })}
          >
            {/* A file of event dates has no value column. Without this option the picker forces a
                numeric column and an event-timestamp upload becomes un-mappable by hand. */}
            <option value="">— each row is one event —</option>
            {columns.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>,
          "Leave empty for a list of event dates: events are then counted per period.",
        )}
        {field(
          "Calendar",
          <select
            style={selectStyle}
            value={mapping.frequency_code ?? ""}
            onChange={(e) => setMapping({ ...mapping, frequency_code: e.target.value || null })}
          >
            <option value="">— infer from the dates —</option>
            {calendars.map((c) => (
              <option key={c.code} value={c.code}>
                {c.label}{c.period ? ` (period ${c.period})` : ""}
              </option>
            ))}
          </select>,
        )}
        {field(
          "Duplicate timestamps",
          <select
            style={selectStyle}
            value={mapping.agg_duplicate}
            onChange={(e) =>
              setMapping({ ...mapping, agg_duplicate: e.target.value as ColumnMapping["agg_duplicate"] })
            }
          >
            <option value="mean">Average them</option>
            <option value="sum">Add them up</option>
            <option value="first">Keep the first</option>
            <option value="last">Keep the last</option>
          </select>,
        )}
        {field(
          "Data class",
          <select
            style={selectStyle}
            value={mapping.data_class ?? ""}
            onChange={(e) => setMapping({ ...mapping, data_class: e.target.value || null })}
          >
            {/* The app tells the user in three places that the class can be changed here. It is
                the switch between the ARIMA/ETS branch and Poisson/NB/Croston — so it has to
                actually exist. Empty keeps the detection, which is shown in the quality card. */}
            <option value="">— keep the detected class —</option>
            <option value="continuous">Continuous (measurements)</option>
            <option value="count">Count (events per period)</option>
            <option value="intermittent">Intermittent (many zeros)</option>
            <option value="binary_or_proportion">Binary / proportion</option>
            <option value="event_timestamp">Event timestamps</option>
          </select>,
          `Currently detected: ${(ingest.dataset.data_class ?? "continuous").replace(/_/g, " ")}. Changing this changes which models are offered in Step 9.`,
        )}
      </div>

      {ingest.cross_sell.length > 0 && (
        <p style={{ fontSize: 11.5, color: "var(--ink-faint)", marginTop: 2 }}>
          Available numeric columns: {ingest.cross_sell.join(", ")}
        </p>
      )}

      <button className="btn btn-primary btn-sm" onClick={() => void apply()} disabled={busy} style={{ marginTop: 6 }}>
        {busy ? "Applying…" : "Apply mapping"}
      </button>
    </div>
  );
}
