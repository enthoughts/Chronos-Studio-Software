/**
 * panels/ForecastPanel.tsx — S13 (forecast), Part 14 (§5.11).
 *
 * The panel is a *renderer*. Every number, warning, coverage label, construction
 * sentence, adequacy verdict and chart shown here came from the backend's
 * `ForecastPayload` or from a `ResultBase` result stored in the session. The only
 * computation done in this file is parsing the user's own comma-separated driver
 * paths and formatting backend numbers for display.
 *
 * The order of the screen is the argument:
 *   1. the controls (horizon, levels, construction, bias adjustment, combination),
 *   2. the horizon advisor's warning and the explicit "I understand" acknowledgement,
 *   3. the honesty banner (S12 adequacy + warnings),
 *   4. the backend's own result cards — fan chart first, then tables and prose.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconAlert, IconChart, IconRefresh } from "../components/Icons";
import { ResultCard } from "../components/ResultCard";
import { ApiError, api } from "../state/api";
import { useStore } from "../state/store";
import type {
  AnyResult,
  ForecastRequestPayload,
  ForecastResponse,
  HierarchyResponse,
  HorizonAdvicePayload,
} from "../state/types";

const fmt = (v: number | null | undefined, digits = 2): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(digits);

const pct = (v: number): string => `${Math.round(v * 100)}%`;

const METHOD_LABEL: Record<string, string> = {
  naive: "Naive — median, no bias adjustment",
  smearing: "Smearing — empirical mean (Duan)",
  lognormal: "Lognormal — parametric mean",
};

export function ForecastPanel() {
  const { sessionId, pushToast, refresh, results, setActiveStage } = useStore();
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(false);
  const [modelId, setModelId] = useState("");
  const [horizon, setHorizon] = useState(12);
  const [levels, setLevels] = useState<number[]>([0.5, 0.66, 0.8, 0.9, 0.95]);
  const [intervalMethod, setIntervalMethod] = useState<"auto" | "analytic" | "simulation">("auto");
  const [nPaths, setNPaths] = useState(2000);
  const [seedText, setSeedText] = useState("");
  const [ack, setAck] = useState(false);
  const [backTransform, setBackTransform] = useState("");
  const [combination, setCombination] = useState("none");
  const [models, setModels] = useState<string[]>([]);
  const [advice, setAdvice] = useState<HorizonAdvicePayload | null>(null);
  const [driverPaths, setDriverPaths] = useState<Record<string, string>>({});
  const [hierarchyText, setHierarchyText] = useState("");
  const [hierarchyMethod, setHierarchyMethod] = useState("ols");
  const [hierarchyHorizon, setHierarchyHorizon] = useState(6);
  const [hierarchyInfo, setHierarchyInfo] = useState<HierarchyResponse | null>(null);

  const options = data?.options;
  const forecast = data?.forecast;
  const cards: AnyResult[] = useMemo(() => results["S13"] ?? [], [results]);

  const load = useCallback(async () => {
    try {
      const r = await api.forecast();
      setData(r);
      const stored = r.request as Partial<ForecastRequestPayload> | undefined;
      if (stored) {
        setHorizon(stored.horizon ?? r.forecast?.horizon ?? 12);
        setLevels(stored.levels ?? r.options.coverages);
        setIntervalMethod(stored.interval_method ?? "auto");
        setNPaths(stored.n_paths ?? r.options.min_paths);
        setSeedText(stored.seed === null || stored.seed === undefined ? "" : String(stored.seed));
        setAck(Boolean(stored.acknowledge_long_horizon));
        setBackTransform(stored.back_transform ?? "");
        setCombination(stored.combination ?? "none");
        setModels(stored.models ?? []);
        setModelId(stored.model_id ?? r.options.winner ?? "");
      } else {
        setHorizon(r.forecast?.horizon ?? 12);
        setLevels(r.options.coverages);
        setNPaths(r.options.min_paths);
        setModelId((prev) => prev || r.options.winner || "");
      }
      try {
        setHierarchyInfo(await api.hierarchy());
      } catch {
        setHierarchyInfo(null);
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

  useEffect(() => {
    if (!sessionId) return;
    const timer = window.setTimeout(() => {
      void api.horizonAdvice(horizon, ack)
        .then((a) => setAdvice(a))
        .catch(() => setAdvice(null));
    }, 220);
    return () => window.clearTimeout(timer);
  }, [horizon, ack, sessionId]);

  const buildPayload = useCallback((): ForecastRequestPayload => ({
    model_id: modelId || null,
    horizon,
    levels,
    interval_method: intervalMethod,
    n_paths: nPaths,
    seed: seedText.trim() ? Number(seedText.trim()) : null,
    acknowledge_long_horizon: ack,
    back_transform: backTransform
      ? (backTransform as "naive" | "smearing" | "lognormal")
      : null,
    combination: combination as ForecastRequestPayload["combination"],
    models,
  }), [modelId, horizon, levels, intervalMethod, nPaths, seedText, ack,
    backTransform, combination, models]);

  const validate = useCallback((): string | null => {
    if (!levels.length) return "Choose at least one interval level.";
    if (options?.transform_required && !backTransform) {
      return "Choose a bias-adjustment method before back-transforming — nothing is exponentiated silently.";
    }
    if (advice?.requires_acknowledgement && !ack) {
      return "Tick 'I understand' before forecasting beyond one fifth of the history.";
    }
    if (nPaths < (options?.min_paths ?? 2000)) {
      return `Simulation needs at least ${options?.min_paths ?? 2000} seeded paths.`;
    }
    if (seedText.trim() && !Number.isFinite(Number(seedText.trim()))) {
      return "The seed must be a whole number, or blank to use the session seed.";
    }
    return null;
  }, [levels, options, backTransform, advice, ack, nPaths, seedText]);

  const run = async () => {
    const problem = validate();
    if (problem) {
      pushToast({ kind: "error", title: problem, body: "Nothing was computed." });
      return;
    }
    setBusy(true);
    try {
      await api.runForecast(buildPayload());
      await refresh();
      await load();
      pushToast({
        kind: "success",
        title: "Forecast complete.",
        body: "Every band states its coverage and construction; the S12 verdict travels with it.",
      });
    } catch (e) {
      const err = e as ApiError;
      pushToast({ kind: "error", title: err.message, body: err.fix ?? err.code });
    } finally {
      setBusy(false);
    }
  };

  const saveScenario = async () => {
    const paths: Record<string, number[]> = {};
    for (const driver of options?.drivers ?? []) {
      const raw = (driverPaths[driver] ?? "").trim();
      if (!raw) continue;
      const parts = raw.split(/[\s,;]+/).filter(Boolean);
      const vals = parts.map(Number);
      if (vals.some((v) => !Number.isFinite(v))) {
        pushToast({ kind: "error", title: `The path for ${driver} contains a non-number.`, body: "Use comma-separated numbers." });
        return;
      }
      if (vals.length !== horizon) {
        pushToast({
          kind: "error",
          title: `The path for ${driver} has ${vals.length} values.`,
          body: `Supply exactly ${horizon} values — one per forecast step.`,
        });
        return;
      }
      paths[driver] = vals;
    }
    const problem = validate();
    if (problem) {
      pushToast({ kind: "error", title: problem });
      return;
    }
    setBusy(true);
    try {
      await api.setScenario({ horizon, paths });
      await api.runForecast(buildPayload());
      await refresh();
      await load();
      pushToast({
        kind: "success",
        title: "Scenario applied.",
        body: "The forecast was recomputed with your future driver paths; held-flat drivers are listed in the result.",
      });
    } catch (e) {
      const err = e as ApiError;
      pushToast({ kind: "error", title: err.message, body: err.fix ?? err.code });
    } finally {
      setBusy(false);
    }
  };

  const runHierarchy = async () => {
    let bottom: Record<string, number[]>;
    try {
      const parsed: unknown = JSON.parse(hierarchyText);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) throw new Error("shape");
      bottom = parsed as Record<string, number[]>;
    } catch {
      pushToast({
        kind: "error",
        title: "The hierarchy must be JSON.",
        body: 'Use {"North": [ ... ], "South": [ ... ]} with equal-length numeric arrays.',
      });
      return;
    }
    setBusy(true);
    try {
      await api.reconcile({
        bottom,
        // Omitted on purpose: the backend refits the Step 11 winner, which is
        // always a univariate model. Passing an exogenous model here would be
        // refused, and picking a different id in the UI would be a second,
        // undocumented model choice.
        horizon: hierarchyHorizon,
        method: hierarchyMethod as "bottom_up" | "top_down" | "ols",
      });
      await refresh();
      setHierarchyInfo(await api.hierarchy());
      pushToast({
        kind: "success",
        title: "Hierarchy reconciled.",
        body: "The bottom forecasts and the aggregate now add up coherently.",
      });
    } catch (e) {
      const err = e as ApiError;
      pushToast({ kind: "error", title: err.message, body: err.fix ?? err.code });
    } finally {
      setBusy(false);
    }
  };

  const toggleLevel = (c: number) => {
    setLevels((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c].sort((a, b) => a - b));
  };
  const toggleModel = (id: string) => {
    setModels((prev) => prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }} data-testid="forecast-panel">
      {/* -------------------------------- controls -------------------------------- */}
      <section className="card" style={{ padding: 14 }} data-testid="forecast-controls">
        <div style={{ display: "flex", gap: 14, alignItems: "flex-end", flexWrap: "wrap" }}>
          <Field label="Model" hint="Stored fit; defaults to the Step 11 winner">
            <select value={modelId} onChange={(e) => setModelId(e.target.value)} style={{ minWidth: 180 }}>
              <option value="">{options?.winner ? `Winner: ${options.winner}` : "Choose a fit"}</option>
              {(options?.models ?? []).filter((m) => !m.failed).map((m) => (
                <option key={m.model_id} value={m.model_id}>{m.display_name}</option>
              ))}
            </select>
          </Field>
          <Field label="Horizon H" hint={`n = ${options?.n ?? "—"}; recommended max ${options?.recommended_max_h ?? "—"}`}>
            <input type="number" min={1} max={240} value={horizon}
              onChange={(e) => setHorizon(Math.max(1, Number(e.target.value) || 1))} style={{ width: 82 }} />
          </Field>
          <Field label="Intervals" hint="50 / 66 / 80 / 90 / 95 %">
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", paddingTop: 3 }}>
              {(options?.coverages ?? [0.5, 0.66, 0.8, 0.9, 0.95]).map((c) => (
                <label key={c} style={{ fontSize: 12, display: "flex", gap: 3, alignItems: "center" }}>
                  <input type="checkbox" checked={levels.includes(c)} onChange={() => toggleLevel(c)} />
                  {pct(c)}
                </label>
              ))}
            </div>
          </Field>
          <Field label="Construction" hint="analytic + simulation are both computed">
            <select value={intervalMethod} onChange={(e) => setIntervalMethod(e.target.value as typeof intervalMethod)}>
              <option value="auto">auto (simulation for non-Gaussian)</option>
              <option value="analytic">analytic</option>
              <option value="simulation">simulation</option>
            </select>
          </Field>
          <Field label="Paths" hint={`seeded, ≥ ${options?.min_paths ?? 2000}`}>
            <input type="number" min={options?.min_paths ?? 2000} max={20000} step={500} value={nPaths}
              onChange={(e) => setNPaths(Number(e.target.value) || 0)} style={{ width: 90 }} />
          </Field>
          <Field label="Seed" hint="blank = session seed">
            <input type="text" value={seedText} onChange={(e) => setSeedText(e.target.value)}
              placeholder="session" style={{ width: 92 }} />
          </Field>
        </div>

        <div style={{ display: "flex", gap: 14, alignItems: "flex-end", flexWrap: "wrap", marginTop: 12 }}>
          <Field label="Back-transformation" hint={options?.transform_required ? "REQUIRED — no silent exponentiation" : "No transform is active"}>
            {options?.transform_required ? (
              <select value={backTransform} onChange={(e) => setBackTransform(e.target.value)} style={{ minWidth: 250 }}>
                <option value="">Choose a bias adjustment…</option>
                {(options.back_transform_methods).map((m) => (
                  <option key={m} value={m}>{METHOD_LABEL[m] ?? m}</option>
                ))}
              </select>
            ) : (
              <span style={{ fontSize: 12, color: "var(--ink-soft)", padding: "6px 0" }}>
                original units — not needed
              </span>
            )}
          </Field>
          <Field label="Combination" hint="top models from the Step 11 shortlist">
            <select value={combination} onChange={(e) => setCombination(e.target.value)}>
              {(options?.combination_methods ?? ["none", "mean", "median", "ic_weighted"]).map((m) => (
                <option key={m} value={m}>{m.replace(/_/g, " ")}</option>
              ))}
            </select>
          </Field>
          <button className="btn btn-primary" onClick={run} disabled={busy || !sessionId || !options}>
            {busy ? "Forecasting…" : forecast ? "Re-run forecast" : "Run forecast"}
          </button>
        </div>

        {combination !== "none" && (
          <div style={{ marginTop: 10 }}>
            <div className="pane-title">Combination components (leave empty to use the shortlist)</div>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 5 }}>
              {(options?.models ?? []).filter((m) => !m.failed && m.model_id !== "combination").map((m) => (
                <label key={m.model_id} style={{ fontSize: 12, display: "flex", gap: 4, alignItems: "center" }}>
                  <input type="checkbox" checked={models.includes(m.model_id)} onChange={() => toggleModel(m.model_id)} />
                  {m.display_name}
                </label>
              ))}
            </div>
          </div>
        )}

        {options?.transform_required && (
          <div className="callout callout-warn" style={{ marginTop: 12 }} data-testid="forecast-backtransform-warning">
            <IconAlert size={14} />{" "}
            <strong>Transformed series.</strong> The model was fitted to a {String(options.transform_active?.method ?? "transformed")} scale.
            A point forecast must not be exponentiated without an explicit bias-adjustment choice; the formula
            and the affected scale are reported with the result.
          </div>
        )}
        {(options?.transform_warnings ?? []).map((w) => (
          <div className="callout callout-warn" key={w} style={{ marginTop: 8, fontSize: 12 }}>{w}</div>
        ))}
      </section>

      {/* -------------------------- horizon advisor ----------------------------- */}
      {advice && (advice.long_horizon || advice.season_short) && (
        <section className="card" style={{ padding: 14 }} data-testid="forecast-horizon">
          <div className="pane-title">Horizon advisor</div>
          <p style={{ margin: "6px 0 0", fontSize: 13 }}>{advice.advice}</p>
          {advice.warnings.map((w) => (
            <p key={w} style={{ margin: "8px 0 0", fontSize: 12, color: "var(--ink-soft)" }}>{w}</p>
          ))}
          {advice.requires_acknowledgement && (
            <label style={{ display: "flex", gap: 8, marginTop: 10, fontSize: 13, alignItems: "center" }}>
              <input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} />
              <strong>I understand</strong> — proceed with H = {advice.horizon} beyond n/5.
            </label>
          )}
        </section>
      )}

      {/* ----------------------------- scenario -------------------------------- */}
      {(options?.drivers ?? []).length > 0 && (
        <section className="card" style={{ padding: 14 }} data-testid="forecast-scenario">
          <div className="pane-title">Future driver paths (scenario)</div>
          <p style={{ margin: "6px 0 10px", fontSize: 12, color: "var(--ink-soft)", maxWidth: 760 }}>
            Supply {horizon} comma-separated values per driver. Drivers you leave blank are held at their
            last observed value — a scenario, not a forecast of the driver.
          </p>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
            {(options?.drivers ?? []).map((d) => (
              <Field key={d} label={d} hint="one value per step">
                <input type="text" value={driverPaths[d] ?? ""}
                  onChange={(e) => setDriverPaths((p) => ({ ...p, [d]: e.target.value }))}
                  placeholder="held flat" style={{ width: 220 }} />
              </Field>
            ))}
            <button className="btn" onClick={saveScenario} disabled={busy || !options}>
              <IconRefresh size={13} /> Save scenario &amp; forecast
            </button>
          </div>
        </section>
      )}

      {/* ----------------------------- summary --------------------------------- */}
      {forecast && (
        <section className="card" style={{ padding: 14 }} data-testid="forecast-summary">
          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <span className="pill pill-brand" style={{ fontSize: 10.5 }}>S13 FORECAST</span>
            <strong style={{ fontSize: 14 }} data-testid="forecast-model">{forecast.display_name}</strong>
            <span className="pill" style={{ fontSize: 11 }}>H = {forecast.horizon}</span>
            <span className="pill" style={{ fontSize: 11 }}>{forecast.interval_method} intervals</span>
            <span className="pill" style={{ fontSize: 11 }}>{forecast.bands.length} coverage levels</span>
            <span className="pill" style={{ fontSize: 11 }}>
              {forecast.n_paths.toLocaleString()} paths · seed {forecast.seed}
            </span>
            <span className="pill" style={{ fontSize: 11 }}>scale: {forecast.scale}</span>
          </div>
          <p style={{ margin: "10px 0 0", fontSize: 13 }}>
            Point at step 1: <strong>{fmt(forecast.point[0], 3)}</strong>; at step {forecast.horizon}:{" "}
            <strong>{fmt(forecast.point[forecast.horizon - 1], 3)}</strong>.{" "}
            {forecast.adequacy.headline}
          </p>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 9 }}>
            {forecast.bands.map((b) => (
              <span key={b.coverage} className="pill" style={{ fontSize: 10.5 }} title={b.construction}>
                {b.coverage_pct}% · {b.source} · mean width {fmt(b.mean_width, 2)}
              </span>
            ))}
          </div>
          {forecast.combination && (
            <p style={{ margin: "9px 0 0", fontSize: 12, color: "var(--ink-soft)" }}>
              Combination: {forecast.combination.method} of {forecast.combination.component_ids.join(", ")};
              weights {Object.entries(forecast.combination.weights).map(([k, v]) => `${k} ${v.toFixed(3)}`).join(", ")}.
            </p>
          )}
          {forecast.scenario.comparison && (
            <p style={{ margin: "9px 0 0", fontSize: 12, color: "var(--ink-soft)" }}>
              {forecast.scenario.note} Maximum change from the held-flat baseline:{" "}
              <strong>{fmt(forecast.scenario.comparison.max_abs_change, 3)}</strong>.
            </p>
          )}
        </section>
      )}

      {/* -------------------------- honesty banner ------------------------------ */}
      {forecast && forecast.warnings.length > 0 && (
        <section className="card" style={{ padding: 14, borderLeft: "3px solid var(--warn)" }} data-testid="forecast-warnings">
          <div className="pane-title">What travels with this forecast</div>
          <ul style={{ margin: "7px 0 0", paddingLeft: 18, fontSize: 12.5, lineHeight: 1.55 }}>
            {forecast.warnings.map((w) => <li key={w}>{w}</li>)}
          </ul>
        </section>
      )}

      {/* ------------------------- backend result cards ------------------------- */}
      {cards.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }} data-testid="forecast-results">
          {cards.map((r) => <ResultCard key={r.id} result={r} />)}
        </div>
      ) : loaded ? (
        <div className="empty">
          <IconChart size={30} />
          <h2 style={{ marginTop: 12 }}>No forecast yet</h2>
          <p style={{ maxWidth: 640 }}>
            Fit and select a model first, then choose a horizon and an interval construction. The fan
            chart, the point table, the adequacy verdict and every interval's coverage and construction
            will appear here.
          </p>
          <button className="btn" style={{ marginTop: 12 }} onClick={() => setActiveStage("S11")}>
            Go to selection
          </button>
        </div>
      ) : null}

      {/* ------------------------- optional hierarchy --------------------------- */}
      <section className="card" style={{ padding: 14 }} data-testid="forecast-hierarchy">
        <div className="pane-title">Optional: reconcile a hierarchy</div>
        <p style={{ margin: "6px 0 10px", fontSize: 12, color: "var(--ink-soft)", maxWidth: 760 }}>
          Supply bottom-level histories as JSON. The app refits the Step 11 winner to each series and to
          their aggregate, then reconciles them with the summing matrix S = [I; 1′]. Every number in the
          result is computed on the backend.
        </p>
        <textarea
          value={hierarchyText}
          onChange={(e) => setHierarchyText(e.target.value)}
          placeholder={'{"North": [ ... ], "South": [ ... ]}'}
          rows={3}
          style={{ width: "100%", fontFamily: "var(--mono)", fontSize: 12 }}
        />
        <div style={{ display: "flex", gap: 12, alignItems: "flex-end", flexWrap: "wrap", marginTop: 9 }}>
          <Field label="Method">
            <select value={hierarchyMethod} onChange={(e) => setHierarchyMethod(e.target.value)}>
              {(options?.reconciliation_methods ?? ["bottom_up", "top_down", "ols"]).map((m) => (
                <option key={m} value={m}>{m.replace(/_/g, " ")}</option>
              ))}
            </select>
          </Field>
          <Field label="Horizon">
            <input type="number" min={1} max={60} value={hierarchyHorizon}
              onChange={(e) => setHierarchyHorizon(Math.max(1, Number(e.target.value) || 1))}
              style={{ width: 76 }} />
          </Field>
          <button className="btn" onClick={runHierarchy} disabled={busy || !options}>Reconcile</button>
          {hierarchyInfo?.reconciliation && (
            <span style={{ fontSize: 12, color: "var(--ink-soft)" }}>
              {String(hierarchyInfo.reconciliation.method ?? "reconciled")} · coherence error{" "}
              {fmt(hierarchyInfo.reconciliation.coherence_error as number | undefined, 6)}
            </span>
          )}
        </div>
      </section>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
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
