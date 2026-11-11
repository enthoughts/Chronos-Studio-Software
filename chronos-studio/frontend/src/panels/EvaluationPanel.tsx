/**
 * panels/EvaluationPanel.tsx — S14 (evaluation), Part 15 (§§5.10, 5.12).
 *
 * A renderer, never a second statistics engine. The rolling-origin protocol,
 * metric policy, leakage audit, every score, plot and table come from the
 * backend. This panel only collects the evaluation scheme, states the chosen
 * policy and its reasons, and renders the stored ResultBase cards.
 */

import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { IconAlert, IconChart, IconRefresh } from "../components/Icons";
import { ResultCard } from "../components/ResultCard";
import { ApiError, api } from "../state/api";
import { useStore } from "../state/store";
import type {
  AnyResult,
  EvaluationOptions,
  EvaluationReport,
  EvaluationRequestPayload,
  EvaluationResponse,
} from "../state/types";

export interface EvaluationForm {
  horizon: number;
  folds: number;
  scheme: "expanding" | "sliding";
  models: string[];
  includeBaselines: boolean;
  coverages: number[];
  nPaths: number;
  seedText: string;
  backTransform: string;
  reps: number;
}

const fmt = (v: number | null | undefined, digits = 3): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(digits);

const pct = (v: number): string => `${Math.round(v * 100)}%`;

const text = (v: unknown, fallback = "—"): string =>
  v === null || v === undefined || v === "" ? fallback : String(v);

/** Build the one request the S14 API accepts. 95% coverage always travels. */
export function buildEvaluationPayload(
  form: EvaluationForm,
  transformRequired: boolean,
): EvaluationRequestPayload {
  const levels = [...new Set([...form.coverages, 0.95])].sort((a, b) => a - b);
  const rawSeed = form.seedText.trim();
  return {
    horizon: form.horizon,
    folds: form.folds,
    scheme: form.scheme,
    models: [...form.models],
    include_baselines: form.includeBaselines,
    coverages: levels,
    n_paths: form.nPaths,
    seed: rawSeed ? Number(rawSeed) : null,
    back_transform: transformRequired && form.backTransform
      ? (form.backTransform as "naive" | "smearing" | "lognormal")
      : null,
    reps: form.reps,
  };
}

/** Client-side checks mirror the backend's refusals; they never compute a score. */
export function validateEvaluationForm(
  form: EvaluationForm,
  options: EvaluationOptions | undefined,
): string | null {
  if (!options) return "The evaluation options have not loaded yet.";
  if (options.s13_status !== "complete") {
    return "Run Step 13 (Forecast) before evaluating out of sample.";
  }
  if (form.coverages.length === 0) return "Choose at least one interval level.";
  if (!Number.isInteger(form.horizon) || form.horizon < options.min_horizon || form.horizon > options.max_horizon) {
    return `The horizon must be a whole number from ${options.min_horizon} to ${options.max_horizon}.`;
  }
  if (!Number.isInteger(form.folds) || form.folds < options.min_folds || form.folds > options.max_folds) {
    return `The fold count must be from ${options.min_folds} to ${options.max_folds}.`;
  }
  if (!Number.isInteger(form.nPaths) || form.nPaths < options.min_paths || form.nPaths > options.max_paths) {
    return `Simulation needs a whole number from ${options.min_paths.toLocaleString()}–${options.max_paths.toLocaleString()} seeded paths.`;
  }
  if (!Number.isInteger(form.reps) || form.reps < options.reps_min || form.reps > options.reps_max) {
    return `Randomized-residual repetitions must be from ${options.reps_min} to ${options.reps_max}.`;
  }
  if (form.seedText.trim()) {
    const seed = Number(form.seedText.trim());
    if (!Number.isFinite(seed) || !Number.isInteger(seed)) {
      return "The seed must be a whole number, or blank to use the session seed.";
    }
  }
  if (options.transform_required && !form.backTransform) {
    return "Choose a bias-adjustment method before back-transforming — nothing is exponentiated silently.";
  }
  return null;
}

/** Read the primary score from the backend's metric rows; no client ranking. */
export function primaryMetricValue(
  report: EvaluationReport,
  model: string,
): number | null {
  const row = report.metric_rows.find((r) => r.model === model);
  const value = row?.[report.accuracy.primary_metric];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function EvaluationPanel() {
  const { sessionId, pushToast, refresh, results, setActiveStage } = useStore();
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [horizon, setHorizon] = useState(6);
  const [folds, setFolds] = useState(4);
  const [scheme, setScheme] = useState<"expanding" | "sliding">("expanding");
  const [models, setModels] = useState<string[]>([]);
  const [includeBaselines, setIncludeBaselines] = useState(true);
  const [coverages, setCoverages] = useState<number[]>([0.5, 0.66, 0.8, 0.9, 0.95]);
  const [nPaths, setNPaths] = useState(2000);
  const [seedText, setSeedText] = useState("");
  const [backTransform, setBackTransform] = useState("");
  const [reps, setReps] = useState(5);

  const options = data?.options;
  const report = data?.evaluation;
  const cards: AnyResult[] = useMemo(() => results["S14"] ?? [], [results]);

  const load = useCallback(async () => {
    try {
      const r = await api.evaluation();
      setData(r);
      const stored = r.request;
      if (stored) {
        setHorizon(stored.horizon);
        setFolds(stored.folds);
        setScheme(stored.scheme);
        setModels(stored.models ?? []);
        setIncludeBaselines(stored.include_baselines);
        setCoverages(stored.coverages);
        setNPaths(stored.n_paths);
        setSeedText(stored.seed === null || stored.seed === undefined ? "" : String(stored.seed));
        setBackTransform(stored.back_transform ?? "");
        setReps(stored.reps);
      } else {
        setHorizon(r.options.default_horizon);
        setFolds(r.options.default_folds);
        setScheme(r.options.default_scheme);
        setModels(r.options.default_models);
        setCoverages(r.options.coverages);
        setNPaths(r.options.default_n_paths);
        setReps(r.options.default_reps);
      }
    } catch {
      /* no session yet, or the endpoint is not reachable — the empty state says so */
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load, sessionId]);

  const form: EvaluationForm = useMemo(() => ({
    horizon, folds, scheme, models, includeBaselines, coverages,
    nPaths, seedText, backTransform, reps,
  }), [horizon, folds, scheme, models, includeBaselines, coverages,
    nPaths, seedText, backTransform, reps]);

  const run = async () => {
    const problem = validateEvaluationForm(form, options);
    if (problem) {
      pushToast({ kind: "error", title: problem, body: "Nothing was computed." });
      return;
    }
    setBusy(true);
    try {
      await api.runEvaluation(buildEvaluationPayload(form, Boolean(options?.transform_required)));
      await refresh();
      await load();
      setActiveStage("S14");
      pushToast({
        kind: "success",
        title: "Rolling-origin evaluation complete.",
        body: "All scores are out of sample; the leakage audit and metric-selection reasons travel with the report.",
      });
    } catch (e) {
      const err = e as ApiError;
      pushToast({ kind: "error", title: err.message, body: err.fix ?? err.code });
    } finally {
      setBusy(false);
    }
  };

  const toggleLevel = (c: number) => {
    if (c === 0.95) return; // required by the Winkler / log-score path
    setCoverages((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c].sort((a, b) => a - b));
  };
  const toggleModel = (id: string) => {
    setModels((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  };

  const failedLeakage = report?.leakage.checks.filter((c) => !c.ok) ?? [];
  const primaryValue = report ? primaryMetricValue(report, report.accuracy.primary) : null;
  const longHorizon = Boolean(options && horizon > options.recommended_max_h);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }} data-testid="evaluation-panel">
      {/* ------------------------------ controls ------------------------------ */}
      <section className="card" style={{ padding: 14 }} data-testid="evaluation-controls">
        <div style={{ display: "flex", gap: 14, alignItems: "flex-end", flexWrap: "wrap" }}>
          <Field label="Horizon H" hint={`n = ${options?.n ?? "—"}; recommended max ${options?.recommended_max_h ?? "—"}`}>
            <input type="number" min={options?.min_horizon ?? 1} max={options?.max_horizon ?? 60}
              value={horizon} onChange={(e) => setHorizon(Number(e.target.value) || 1)} style={{ width: 82 }} />
          </Field>
          <Field label="Folds" hint={`min train ${options?.min_train ?? 16}`}>
            <input type="number" min={options?.min_folds ?? 2} max={options?.max_folds ?? 12}
              value={folds} onChange={(e) => setFolds(Number(e.target.value) || 2)} style={{ width: 76 }} />
          </Field>
          <Field label="Origin scheme" hint="expanding or sliding training windows">
            <select value={scheme} onChange={(e) => setScheme(e.target.value as typeof scheme)}>
              {(options?.schemes ?? ["expanding", "sliding"]).map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </Field>
          <Field label="Intervals" hint="95% is always included">
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", paddingTop: 3 }}>
              {(options?.coverages ?? [0.5, 0.66, 0.8, 0.9, 0.95]).map((c) => (
                <label key={c} style={{ fontSize: 12, display: "flex", gap: 3, alignItems: "center" }}>
                  <input type="checkbox" checked={coverages.includes(c) || c === 0.95}
                    disabled={c === 0.95} onChange={() => toggleLevel(c)} />
                  {pct(c)}
                </label>
              ))}
            </div>
          </Field>
          <Field label="Paths" hint={`seeded, ≥ ${options?.min_paths ?? 2000}`}>
            <input type="number" min={options?.min_paths ?? 2000} max={options?.max_paths ?? 20000} step={500}
              value={nPaths} onChange={(e) => setNPaths(Number(e.target.value) || 0)} style={{ width: 92 }} />
          </Field>
          <Field label="Seed" hint="blank = session seed">
            <input type="text" value={seedText} onChange={(e) => setSeedText(e.target.value)}
              placeholder="session" style={{ width: 92 }} />
          </Field>
          <Field label="RQR repetitions" hint="seeded randomization">
            <input type="number" min={options?.reps_min ?? 1} max={options?.reps_max ?? 20}
              value={reps} onChange={(e) => setReps(Number(e.target.value) || 1)} style={{ width: 76 }} />
          </Field>
        </div>

        <div style={{ display: "flex", gap: 14, alignItems: "flex-end", flexWrap: "wrap", marginTop: 12 }}>
          {options?.transform_required ? (
            <Field label="Back-transformation" hint="REQUIRED — no silent exponentiation">
              <select value={backTransform} onChange={(e) => setBackTransform(e.target.value)} style={{ minWidth: 250 }}>
                <option value="">Choose a bias adjustment…</option>
                {(options.back_transform_methods).map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
              </select>
            </Field>
          ) : (
            <span style={{ fontSize: 12, color: "var(--ink-soft)", padding: "6px 0" }}>
              Original units — no bias adjustment needed.
            </span>
          )}
          <label style={{ display: "flex", gap: 6, alignItems: "center", fontSize: 12 }}>
            <input type="checkbox" checked={includeBaselines}
              onChange={(e) => setIncludeBaselines(e.target.checked)} />
            Include naive / seasonal-naive benchmark
          </label>
          <button className="btn btn-primary" onClick={run} disabled={busy || !sessionId || !options}>
            <IconRefresh size={13} /> {busy ? "Evaluating…" : report ? "Re-run evaluation" : "Run evaluation"}
          </button>
        </div>

        <div style={{ marginTop: 12 }} data-testid="evaluation-models">
          <div className="pane-title">Models (leave empty for the Step 11 shortlist + benchmark)</div>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 5 }}>
            {(options?.models ?? []).filter((m) => !m.failed && m.model_id !== "combination").map((m) => (
              <label key={m.model_id} style={{ fontSize: 12, display: "flex", gap: 4, alignItems: "center" }}>
                <input type="checkbox" checked={models.includes(m.model_id)} onChange={() => toggleModel(m.model_id)} />
                {m.display_name}
              </label>
            ))}
          </div>
        </div>

        {longHorizon && (
          <div className="callout callout-warn" style={{ marginTop: 12 }} data-testid="evaluation-horizon-warning">
            <IconAlert size={14} /> H = {horizon} exceeds the advised maximum {options?.recommended_max_h}. The
            rolling origins still score only out-of-sample rows, but a long horizon leaves fewer usable folds.
          </div>
        )}
        {(options?.transform_warnings ?? []).map((w) => (
          <div className="callout callout-warn" key={w} style={{ marginTop: 8, fontSize: 12 }}>{w}</div>
        ))}
      </section>

      {/* --------------------------- metric policy ---------------------------- */}
      {options && (
        <section className="card" style={{ padding: 14 }} data-testid="evaluation-policy">
          <div className="pane-title">Metric-selection policy</div>
          <div style={{ display: "flex", gap: 7, flexWrap: "wrap", margin: "8px 0" }}>
            <span className="pill pill-brand" style={{ fontSize: 10.5 }}>{options.policy.cluster.replace(/_/g, " ")}</span>
            <span className="pill" style={{ fontSize: 10.5 }}>primary: {options.policy.primary}</span>
            <span className="pill" style={{ fontSize: 10.5 }}>data class: {options.data_class}</span>
            <span className="pill" style={{ fontSize: 10.5 }}>target: {options.target_kind}</span>
            <span className="pill" style={{ fontSize: 10.5 }}>benchmark: {options.benchmark_id}</span>
          </div>
          <p style={{ fontSize: 12.5, margin: "0 0 6px" }}>
            <strong>Lead metrics:</strong> {options.policy.metrics.join(", ")}.{" "}
            <strong>Refused:</strong> {options.banned.length ? options.banned.join(", ") : "none by policy"}.
          </p>
          <ul style={{ margin: "4px 0 0", paddingLeft: 18, fontSize: 12, color: "var(--ink-soft)", lineHeight: 1.5 }}>
            {options.policy_reasons.map((r) => <li key={r}>{r}</li>)}
          </ul>
          <p style={{ fontSize: 11.5, color: "var(--ink-faint)", margin: "8px 0 0" }}>
            {options.class_note} Zero share in the training majority: {pct(options.zero_share)}.
            S13 status: {options.s13_status}{options.s13_stale ? " (stale)" : ""}.
          </p>
        </section>
      )}

      {/* ------------------------------ summary ------------------------------- */}
      {report && (
        <section className="card" style={{ padding: 14 }} data-testid="evaluation-summary">
          <div style={{ display: "flex", gap: 9, alignItems: "center", flexWrap: "wrap" }}>
            <span className="pill pill-brand" style={{ fontSize: 10.5 }}>S14 OUT-OF-SAMPLE</span>
            <strong style={{ fontSize: 14 }} data-testid="evaluation-primary-model">{report.accuracy.primary}</strong>
            <span className="pill" style={{ fontSize: 11 }}>
              {report.accuracy.primary_metric} {fmt(primaryValue)}
            </span>
            <span className="pill" style={{ fontSize: 11 }}>{report.models.length} models</span>
            <span className="pill" style={{ fontSize: 11 }}>{text(report.inputs.n)} observations</span>
            <span className="pill" style={{ fontSize: 11 }}>{text(report.inputs.frequency)} frequency</span>
            <span className="pill" style={{ fontSize: 11 }}>{text(report.inputs.folds)} folds · H = {text(report.inputs.horizon)}</span>
            <span className="pill" style={{ fontSize: 11 }}>{text(report.accuracy.n_scored)} scored rows</span>
          </div>
          <p style={{ margin: "10px 0 0", fontSize: 13 }}>
            Evaluation window: <strong>{report.accuracy.evaluation_window}</strong>. Ranking on{" "}
            <strong>{report.accuracy.primary_metric}</strong>: {report.accuracy.ranked.slice(0, 5).join(" → ") || "—"}.{" "}
            Test observations used for fitting: <strong>{text(report.inputs.test_observations_used_for_fitting, "0")}</strong>.
          </p>
          <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginTop: 9 }}>
            <span className="pill" style={{ fontSize: 10.5, borderColor: report.leakage.ok ? "var(--ok)" : "var(--warn)" }}>
              Leakage audit: {report.leakage.ok ? "PASS" : "FAIL"} · {report.leakage.checks.length} checks
            </span>
            <span className="pill" style={{ fontSize: 10.5 }}>
              {text(report.inputs.interval_construction, "coverage + construction stated")}
            </span>
            <span className="pill" style={{ fontSize: 10.5 }}>
              adequacy: {text(report.adequacy.adequacy, "not evaluated")}
            </span>
          </div>
        </section>
      )}

      {/* -------------------------- honesty banner ---------------------------- */}
      {report && (report.warnings.length > 0 || failedLeakage.length > 0 || data?.stale) && (
        <section className="card" style={{ padding: 14, borderLeft: "3px solid var(--warn)" }}
          data-testid="evaluation-warnings">
          <div className="pane-title">What travels with this evaluation</div>
          <ul style={{ margin: "7px 0 0", paddingLeft: 18, fontSize: 12.5, lineHeight: 1.55 }}>
            {data?.stale && <li key="stale">The result is stale: an earlier step changed after this evaluation ran.</li>}
            {failedLeakage.map((c) => <li key={c.name}>Leakage check failed: {c.name} — {c.detail}</li>)}
            {report.warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
        </section>
      )}

      {/* ------------------------- backend result cards ----------------------- */}
      {cards.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }} data-testid="evaluation-results">
          {cards.map((r) => <ResultCard key={r.id} result={r} />)}
        </div>
      ) : loaded ? (
        <div className="empty" data-testid="evaluation-empty">
          <IconChart size={30} />
          <h2 style={{ marginTop: 12 }}>No out-of-sample evaluation yet</h2>
          <p style={{ maxWidth: 680 }}>
            Choose the rolling-origin scheme, then run the evaluation. The policy reason, split chart,
            leakage audit, metric heatmap, MASE-by-horizon, fold boxplots, CRPS/pinball curves,
            PIT/RQR calibration, DM matrix and branch-specific PMF or variance tables will appear here.
            In-sample fit is never reported as forecast accuracy.
          </p>
          <button className="btn" style={{ marginTop: 12 }} onClick={() => setActiveStage("S13")}>
            Go to forecast
          </button>
        </div>
      ) : null}
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <span style={{ fontSize: 11, textTransform: "uppercase", letterSpacing: ".06em", color: "var(--ink-faint)" }}>
        {label}
      </span>
      {children}
      {hint ? <span style={{ fontSize: 11, color: "var(--ink-faint)" }}>{hint}</span> : null}
    </label>
  );
}
