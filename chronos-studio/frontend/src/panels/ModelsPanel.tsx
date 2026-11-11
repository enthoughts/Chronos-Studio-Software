/**
 * panels/ModelsPanel.tsx — S9 (Suggest) and S10 (Fit) in one place (§5.8).
 *
 * The two stages share a screen because they are one decision: the list of candidates *is* the
 * fitting plan, and a user who has to re-select models on a second page will select whatever is
 * already ticked. Layout, in the order the questions get asked:
 *
 *   1. **What can be fitted at all?** — the data class, stated with the number that decided it, plus
 *      a count of the registry. A count series that quietly offers ETS is the bug this prevents.
 *   2. **The candidates** — each with *why it is suggested here* and *what would have to be true for
 *      it to win*. Ticked by default; the ones with assumption warnings are badged, not hidden.
 *   3. **The excluded** — collapsible, each with the measurement that ruled it out. "Why can't I use
 *      a GARCH model?" deserves an answer, not silence.
 *   4. **Fit** — one button, a cost estimate, and a progress note, because a SARIMA grid takes
 *      seconds to minutes and a frozen screen reads as a crash.
 *   5. **The comparison** — ranked by AICc, with baselines always present, failures shown as rows
 *      rather than removed, and a note on every model whose IC is not comparable.
 *
 * Nothing here writes to the series. Fitting is read-only over the prepared data.
 *
 * S9 and S10 render the same component: on S9 the user is choosing, on S10 they are reading. Both
 * need the list, so branching the component would duplicate the candidate cards and let the two
 * stages drift apart.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { IconCheck, IconRefresh } from "../components/Icons";
import { api } from "../state/api";
import { pNumber } from "../state/format";
import { useStore } from "../state/store";
import { errorToastPayload } from "../copy/errors";
import type {
  FitForecast,
  FitRow,
  FitsResponse,
  ModelCard as ModelCardT,
  ModelSuggestion,
  MultivariateResponse,
  SuggestionSet,
  VariancePath,
} from "../state/types";

const FAMILY_TONE: Record<string, { colour: string; wash: string }> = {
  baseline: { colour: "var(--ink-soft)", wash: "var(--stale-wash)" },
  smoothing: { colour: "var(--brand)", wash: "var(--brand-wash)" },
  arma: { colour: "var(--brand)", wash: "var(--brand-wash)" },
  count: { colour: "var(--warn)", wash: "var(--warn-wash)" },
  intermittent: { colour: "var(--warn)", wash: "var(--warn-wash)" },
  structural: { colour: "var(--ink-soft)", wash: "var(--stale-wash)" },
  volatility: { colour: "var(--ink-faint)", wash: "transparent" },
  multivariate: { colour: "var(--brand)", wash: "var(--brand-wash)" },
  ml: { colour: "var(--warn)", wash: "var(--warn-wash)" },
  nonlinear: { colour: "var(--warn)", wash: "var(--warn-wash)" },
};

function report(e: unknown, pushToast: ReturnType<typeof useStore>["pushToast"], fallback: string) {
  pushToast(errorToastPayload(e, fallback));
}

function num(x: number | null | undefined, dp = 2): string {
  return x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(dp);
}

export function ModelsPanel() {
  const { sessionId, session, refresh, pushToast } = useStore();
  const [set, setSet] = useState<SuggestionSet | null>(null);
  const [fits, setFits] = useState<FitsResponse | null>(null);
  const [chosen, setChosen] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  const [showExcluded, setShowExcluded] = useState(false);
  const [openParams, setOpenParams] = useState<string | null>(null);
  // Which fitted model's variance path is on screen. Null until a volatility row is fitted, so a
  // non-volatility session sees no extra card at all rather than an empty one.
  const [varianceFor, setVarianceFor] = useState<string | null>(null);
  const [varianceFc, setVarianceFc] = useState<FitForecast | null>(null);
  const [varianceBusy, setVarianceBusy] = useState(false);
  // Which fitted model's multivariate result is on screen (Granger / IRF / FEVD), and its payload.
  const [multiFor, setMultiFor] = useState<string | null>(null);
  const [multi, setMulti] = useState<MultivariateResponse | null>(null);
  const [multiBusy, setMultiBusy] = useState(false);

  const loadMultivariate = useCallback(
    async (modelId: string) => {
      setMultiBusy(true);
      try {
        setMulti(await api.multivariate(modelId));
      } catch (e) {
        report(e, pushToast, "Could not compute the multivariate result");
        setMulti(null);
      } finally {
        setMultiBusy(false);
      }
    },
    [pushToast],
  );

  const loadVariancePath = useCallback(
    async (modelId: string) => {
      setVarianceBusy(true);
      try {
        const fc = await api.fitForecast(modelId, 60);
        setVarianceFc(fc);
      } catch (e) {
        report(e, pushToast, "Could not compute the variance path");
        setVarianceFc(null);
      } finally {
        setVarianceBusy(false);
      }
    },
    [pushToast],
  );

  const hasData = Boolean(session?.dataset);
  const gateReady = session?.pipeline.stages.S8?.status === "complete";
  const s9done = session?.pipeline.stages.S9?.status === "complete";

  const load = useCallback(async () => {
    if (!sessionId || !hasData) return;
    const [s, f] = await Promise.all([
      api.getSuggestions().catch(() => null),
      api.getFits().catch(() => null),
    ]);
    if (s) {
      setSet(s);
      setChosen((prev) => {
        const next: Record<string, boolean> = {};
        for (const c of s.suggestions) {
          // Default everything compatible to ticked, but remember an explicit user choice.
          next[c.model_id] = prev[c.model_id] ?? true;
        }
        return next;
      });
    }
    if (f) setFits(f);
  }, [sessionId, hasData]);

  useEffect(() => {
    void load().catch((e) => report(e, pushToast, "The model list could not be loaded."));
  }, [load, pushToast]);

  const runSuggest = useCallback(async () => {
    setBusy(true);
    try {
      await api.runSuggest();
      await load();
      await refresh(); // S9 completes server-side; the stepper must agree
      pushToast({ kind: "success", title: "Candidates built.", body: "Tick what to fit, then fit." });
    } catch (e) {
      report(e, pushToast, "The candidate list could not be built.");
    } finally {
      setBusy(false);
    }
  }, [load, refresh, pushToast]);

  const runFit = useCallback(async () => {
    const ids = Object.entries(chosen)
      .filter(([, v]) => v)
      .map(([k]) => k);
    if (ids.length === 0) {
      pushToast({ kind: "error", title: "Nothing selected.", body: "Tick at least one model." });
      return;
    }
    setBusy(true);
    try {
      const res = await api.runFit(ids);
      await load();
      await refresh();
      pushToast({
        kind: "success",
        title: "Fitted.",
        body: res.advisor?.headline ?? "The comparison table is below.",
      });
    } catch (e) {
      report(e, pushToast, "The models could not be fitted.");
    } finally {
      setBusy(false);
    }
  }, [chosen, load, refresh, pushToast]);

  const ticked = useMemo(
    () => Object.values(chosen).filter(Boolean).length,
    [chosen],
  );

  const estimate = useMemo(() => {
    if (!set) return 0;
    return set.suggestions
      .filter((s) => chosen[s.model_id])
      .reduce((sum, s) => sum + (s.expected_seconds ?? 0), 0);
  }, [set, chosen]);

  if (!hasData) {
    return (
      <div className="empty">
        <IconCheck size={30} />
        <h2 style={{ marginTop: 12 }}>No series to model</h2>
        <p>Load data in Step 0, then work through Steps 1–8 before choosing a model.</p>
      </div>
    );
  }

  return (
    <div>
      {/* ---------- 1. what this data is ---------- */}
      {set ? (
        <section className="card" style={{ borderLeft: "3px solid var(--brand)" }}>
          <div className="card-head">
            <span className="card-kind">Data</span>
            <div>
              <div className="card-title" data-testid="model-headline">
                {set.headline}
              </div>
              <div className="card-plain">{set.plain}</div>
            </div>
            <div className="card-actions">
              <span
                className="pill"
                style={{
                  background: FAMILY_TONE[set.data_class]?.wash ?? "var(--stale-wash)",
                  color: FAMILY_TONE[set.data_class]?.colour ?? "var(--ink-soft)",
                }}
              >
                {set.data_class.replace(/_/g, " ")}
              </span>
              {!s9done && (
                <button className="btn btn-sm btn-primary" onClick={() => void runSuggest()} disabled={busy}>
                  Confirm and continue
                </button>
              )}
            </div>
          </div>
        </section>
      ) : (
        <section className="card">
          <div className="card-body" style={{ paddingTop: 12 }}>
            <h2 style={{ margin: 0, fontSize: 14 }}>No candidate list yet</h2>
            <p style={{ margin: "6px 0 12px", fontSize: 12.5, color: "var(--ink-soft)" }}>
              {gateReady
                ? "Build the candidate list: it filters every model in the registry against what the earlier steps measured."
                : "Step 8 (the readiness gate) has not been run. The list of models depends on its verdict, so run that first."}
            </p>
            <button className="btn btn-primary" onClick={() => void runSuggest()} disabled={busy || !gateReady}>
              <IconRefresh size={14} /> {busy ? "Building…" : "Build the candidate list"}
            </button>
          </div>
        </section>
      )}

      {/* ---------- 2. the candidates ---------- */}
      {set && (
        <section className="card" style={{ marginTop: 12 }} aria-labelledby="cand-head">
          <div className="card-head">
            <span className="card-kind">Candidates</span>
            <div>
              <div className="card-title" id="cand-head">
                Each model, the reason it is here, and what would make it win
              </div>
              <div className="card-plain">
                Ticked models are fitted. Baselines are always fitted whether ticked or not — without
                the naive forecast there is no way to tell whether a model earned its complexity.
              </div>
            </div>
          </div>
          <div className="card-body" style={{ paddingTop: 0 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {set.suggestions.map((s) => (
                <CandidateRow
                  key={s.model_id}
                  s={s}
                  ticked={chosen[s.model_id] ?? true}
                  onToggle={() =>
                    setChosen((prev) => ({ ...prev, [s.model_id]: !(prev[s.model_id] ?? true) }))
                  }
                />
              ))}
            </div>
          </div>
        </section>
      )}

      {/* ---------- 3. fit ---------- */}
      {set && (
        <section className="card" style={{ marginTop: 12 }}>
          <div className="card-body" style={{ paddingTop: 12 }}>
            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <button className="btn btn-primary" onClick={() => void runFit()} disabled={busy}>
                <IconCheck size={14} /> {busy ? "Fitting…" : `Fit ${ticked} model${ticked === 1 ? "" : "s"}`}
              </button>
              <span style={{ fontSize: 12, color: "var(--ink-faint)" }}>
                about {estimate.toFixed(1)} s on a series this size · baselines included
              </span>
              <div style={{ flex: 1 }} />
              <button className="btn btn-sm" onClick={() => void runSuggest()} disabled={busy}>
                <IconRefresh size={13} /> Rebuild the list
              </button>
            </div>
            <p style={{ margin: "8px 0 0", fontSize: 11.5, color: "var(--ink-faint)" }}>
              A failed fit is a row in the table with its reason, never a missing row — and never an
              error page.
            </p>
          </div>
        </section>
      )}

      {/* ---------- 4. the excluded ---------- */}
      {set && set.excluded.length > 0 && (
        <section className="card" style={{ marginTop: 12 }}>
          <div className="card-head">
            <span className="card-kind">Excluded</span>
            <div>
              <div className="card-title">
                {set.excluded.length} models in the registry do not apply to this series
              </div>
              <div className="card-plain">
                Each was ruled out by a measurement, quoted below. Nothing here is a judgement call.
              </div>
            </div>
            <div className="card-actions">
              <button className="btn btn-sm" onClick={() => setShowExcluded((v) => !v)}>
                {showExcluded ? "Hide" : "Show"}
              </button>
            </div>
          </div>
          {showExcluded && (
            <div className="card-body" style={{ paddingTop: 0 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
                <tbody>
                  {set.excluded.map((e) => (
                    <tr key={e.model_id} style={{ borderTop: "1px solid var(--rule)" }}>
                      <td style={{ padding: "6px 8px 6px 0", width: 200, verticalAlign: "top" }}>
                        {e.display_name}
                      </td>
                      <td style={{ padding: "6px 0", color: "var(--ink-soft)" }}>
                        {e.incompatibilities.join(" ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      {/* ---------- 5. the comparison ---------- */}
      {fits && fits.status === "complete" && (
        <FitComparison
          fits={fits}
          openParams={openParams}
          setOpenParams={setOpenParams}
          onVariancePath={async (id) => {
            setVarianceFor(id);
            setVarianceFc(null);
            await loadVariancePath(id);
          }}
          varianceFor={varianceFor}
          onMultivariate={async (id) => {
            setMultiFor(id);
            setMulti(null);
            await loadMultivariate(id);
          }}
          multiFor={multiFor}
        />
      )}

      {/* ---------- 6. the second forecast, when one exists ---------- */}
      {varianceFor && varianceBusy && (
        <p style={{ marginTop: 12, fontSize: 12, color: "var(--ink-faint)" }}>
          Computing the variance path for {varianceFor}…
        </p>
      )}
      {varianceFc && varianceFc.variance_path && <VariancePathCard fc={varianceFc} />}

      {/* ---------- 7. the joint result, when the fit was multivariate ---------- */}
      {multiFor && multiBusy && (
        <p style={{ marginTop: 12, fontSize: 12, color: "var(--ink-faint)" }}>
          Computing the impulse responses for {multiFor}…
        </p>
      )}
      {multi && <MultivariateCard m={multi} />}

      {fits && fits.status !== "complete" && (
        <p style={{ marginTop: 14, fontSize: 12, color: "var(--ink-faint)" }}>
          Nothing fitted yet. The comparison table appears here.
        </p>
      )}
    </div>
  );
}

function CandidateRow({
  s,
  ticked,
  onToggle,
}: {
  s: ModelSuggestion;
  ticked: boolean;
  onToggle: () => void;
}) {
  const tone = FAMILY_TONE[s.family] ?? { colour: "var(--ink-soft)", wash: "var(--stale-wash)" };
  return (
    <div
      style={{
        border: "1px solid var(--rule)",
        borderRadius: 8,
        padding: "9px 11px",
        background: s.recommended ? "var(--brand-wash)" : "transparent",
      }}
      data-testid={`candidate-${s.model_id}`}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
        <input
          type="checkbox"
          checked={ticked}
          onChange={onToggle}
          aria-label={`Fit ${s.display_name}`}
        />
        <strong style={{ fontSize: 13 }}>{s.display_name}</strong>
        <span className="pill" style={{ background: tone.wash, color: tone.colour }}>
          {s.family}
        </span>
        {s.is_baseline && <span className="pill" style={{ color: "var(--ink-faint)" }}>benchmark</span>}
        {s.advanced && (
          <span
            className="pill pill-warn"
            title="Listed as advanced in the model registry: it can win, but its assumptions are harder to check."
          >
            advanced
          </span>
        )}
        {s.recommended && (
          <span className="pill" style={{ background: "var(--brand)", color: "#fff" }}>start here</span>
        )}
        <div style={{ flex: 1 }} />
        {s.order_hint && (
          <span style={{ fontSize: 11, color: "var(--ink-faint)", fontFamily: "var(--mono)" }}>
            {s.order_hint}
          </span>
        )}
        <span style={{ fontSize: 11, color: "var(--ink-faint)" }}>
          ~{s.expected_seconds ?? "?"} s
        </span>
      </div>
      <p style={{ margin: "6px 0 0", fontSize: 12.5 }}>
        <strong>Why here:</strong> {s.because}
      </p>
      <p style={{ margin: "3px 0 0", fontSize: 12.5, color: "var(--ink-soft)" }}>
        <strong>Would win if:</strong> {s.would_win_if}
      </p>
      {s.assumption_warnings.length > 0 && (
        <ul style={{ margin: "5px 0 0", paddingLeft: 18, fontSize: 12, color: "var(--warn)" }}>
          {s.assumption_warnings.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function FitComparison({
  fits,
  openParams,
  setOpenParams,
  onVariancePath,
  varianceFor,
  onMultivariate,
  multiFor,
}: {
  fits: FitsResponse;
  openParams: string | null;
  setOpenParams: (v: string | null) => void;
  onVariancePath: (modelId: string) => void | Promise<void>;
  varianceFor: string | null;
  onMultivariate: (modelId: string) => void | Promise<void>;
  multiFor: string | null;
}) {
  return (
    <>
      {fits.stale && (
        <section className="card" style={{ marginTop: 12, borderLeft: "3px solid var(--warn)" }}>
          <div className="card-body" style={{ paddingTop: 10 }}>
            <strong style={{ fontSize: 13 }}>These fits describe an older version of the series.</strong>
            <p style={{ margin: "4px 0 0", fontSize: 12.5, color: "var(--ink-soft)" }}>
              Fitted on version {fits.computed_for_version}; the series is now at version{" "}
              {fits.current_version}. Re-fit before reading the comparison.
            </p>
          </div>
        </section>
      )}

      {fits.failure_summary && (
        <section className="card" style={{ marginTop: 12, borderLeft: "3px solid var(--bad)" }}>
          <div className="card-body" style={{ paddingTop: 10, fontSize: 12.5 }}>
            <strong>Some models did not fit.</strong> {fits.failure_summary}
          </div>
        </section>
      )}

      <section className="card" style={{ marginTop: 12 }} aria-labelledby="cmp-head">
        <div className="card-head">
          <span className="card-kind">Comparison</span>
          <div>
            <div className="card-title" id="cmp-head">
              Every fitted model, ranked by AICc
            </div>
            <div className="card-plain">
              AICc rewards explaining the sample and penalises parameters. It ranks explanation, not
              forecasting — the holdout comparison in Step 11 is what settles the choice.
            </div>
          </div>
          <div className="card-actions">
            <span className="pill" style={{ color: "var(--ink-soft)" }}>
              d = {fits.d} · D = {fits.D} · m = {fits.period ?? "—"}
            </span>
          </div>
        </div>
        <div className="card-body" style={{ paddingTop: 0 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
            <thead>
              <tr style={{ textAlign: "left", color: "var(--ink-faint)", fontSize: 11 }}>
                <th style={{ padding: "4px 6px 4px 0" }}>#</th>
                <th style={{ padding: "4px 6px" }}>Model</th>
                <th style={{ padding: "4px 6px", textAlign: "right" }}>AICc</th>
                <th style={{ padding: "4px 6px", textAlign: "right" }}>BIC</th>
                <th style={{ padding: "4px 6px", textAlign: "right" }}>k</th>
                <th style={{ padding: "4px 6px", textAlign: "right" }}>MASE</th>
                <th style={{ padding: "4px 6px", textAlign: "right" }}>Ljung–Box p</th>
                <th style={{ padding: "4px 0 4px 6px" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {fits.fits.map((f, i) => (
                <>
                  <tr
                    key={f.model_id}
                    style={{ borderTop: "1px solid var(--rule)", opacity: f.failed ? 0.75 : 1 }}
                    data-testid={`fit-row-${f.model_id}`}
                  >
                    <td style={{ padding: "6px 6px 6px 0", color: "var(--ink-faint)" }}>{i + 1}</td>
                    <td style={{ padding: "6px" }}>
                      <button
                        className="btn"
                        style={{
                          background: "none",
                          border: "none",
                          padding: 0,
                          fontWeight: 600,
                          fontSize: 12.5,
                          color: f.failed ? "var(--bad)" : "var(--ink)",
                          cursor: "pointer",
                          textDecoration: "underline dotted",
                        }}
                        onClick={() => setOpenParams(openParams === f.model_id ? null : f.model_id)}
                      >
                        {f.display_name}
                      </button>
                      {f.is_baseline && (
                        <span style={{ fontSize: 10, color: "var(--ink-faint)", marginLeft: 6 }}>
                          benchmark
                        </span>
                      )}
                    </td>
                    <td style={{ padding: "6px", textAlign: "right", fontFamily: "var(--mono)" }}>
                      {num(f.aicc)}
                    </td>
                    <td style={{ padding: "6px", textAlign: "right", fontFamily: "var(--mono)" }}>
                      {num(f.bic)}
                    </td>
                    <td style={{ padding: "6px", textAlign: "right" }}>{f.n_params}</td>
                    <td style={{ padding: "6px", textAlign: "right", fontFamily: "var(--mono)" }}>
                      {num((f.diagnostics?.mase as number | null) ?? null, 3)}
                    </td>
                    <td style={{ padding: "6px", textAlign: "right", fontFamily: "var(--mono)" }}>
                      {pNumber((f.diagnostics?.ljung_box_p as number | null) ?? null, 3)}
                    </td>
                    <td style={{ padding: "6px 0 6px 6px", whiteSpace: "nowrap" }}>
                      {f.family === "volatility" && !f.failed && (
                        <button
                          className="btn"
                          style={{
                            background: "none",
                            border: "1px solid var(--rule)",
                            borderRadius: 4,
                            fontSize: 10.5,
                            padding: "1px 6px",
                            marginRight: 6,
                            color: varianceFor === f.model_id ? "var(--brand)" : "var(--ink-soft)",
                          }}
                          onClick={() => onVariancePath(f.model_id)}
                          title="Show the second forecast: how uncertain each future step is."
                          data-testid={`variance-${f.model_id}`}
                        >
                          variance path
                        </button>
                      )}
                      {f.family === "multivariate" && !f.failed && (
                        <button
                          className="btn"
                          style={{
                            background: "none",
                            border: "1px solid var(--rule)",
                            borderRadius: 4,
                            fontSize: 10.5,
                            padding: "1px 6px",
                            marginRight: 6,
                            color: multiFor === f.model_id ? "var(--brand)" : "var(--ink-soft)",
                          }}
                          onClick={() => onMultivariate(f.model_id)}
                          title="Show what moves what: Granger tests, impulse responses and the variance decomposition."
                          data-testid={`multivariate-${f.model_id}`}
                        >
                          how they move
                        </button>
                      )}
                      {f.failed ? (
                        <span className="pill" style={{ background: "var(--bad-wash)", color: "var(--bad)" }}>
                          failed
                        </span>
                      ) : f.is_baseline ? (
                        <span className="pill" style={{ color: "var(--ink-faint)" }}>benchmark</span>
                      ) : (
                        <span className="pill pill-ok">fitted</span>
                      )}
                    </td>
                  </tr>
                  {(openParams === f.model_id || f.failed) && (
                    <tr key={`${f.model_id}-detail`}>
                      <td colSpan={8} style={{ padding: "0 0 8px 20px" }}>
                        <FitDetail f={f} />
                      </td>
                    </tr>
                  )}
                </>
              ))}
            </tbody>
          </table>
          <p style={{ margin: "8px 0 0", fontSize: 11.5, color: "var(--ink-faint)" }}>
            A dash means the model has no comparable information criterion — Theta, Croston and ARFIMA
            have none, and ranking them against AICc would compare numbers on different scales.
          </p>
        </div>
      </section>
    </>
  );
}

/**
 * The variance path — §8 Part 11's *second* forecast.
 *
 * A GARCH family fit produces two forecasts and only one of them is a value. This card exists because
 * the mean path for a volatility model is usually a flat line at zero: showing that alone would read as
 * "the model has no opinion", when the model's entire opinion is in the spread. The numbers come from
 * the same stored fit, via `/analysis/fit/forecast`, and the card says so.
 */
function VariancePathCard({ fc }: { fc: FitForecast }) {
  const vp: VariancePath | null | undefined = fc.variance_path;
  if (!vp) return null;
  if (vp.error) {
    return (
      <section className="card" style={{ marginTop: 12, borderLeft: "3px solid var(--warn)" }}>
        <div className="card-body" style={{ paddingTop: 10, fontSize: 12.5 }}>
          <strong>The variance path could not be produced.</strong> {vp.error}
        </div>
      </section>
    );
  }
  const horizon = vp.sd.length;
  // Evenly spaced checkpoints rather than every step: 60 rows of six decimals is not a table a reader
  // uses, it is one they scroll past.
  const step = Math.max(1, Math.floor(horizon / 8));
  const rows = vp.sd.map((sd, i) => ({ h: i + 1, sd, variance: vp.variance[i] })).filter(
    (r) => r.h % step === 0 || r.h === 1 || r.h === horizon,
  );
  const first = vp.sd[0];
  const last = vp.sd[horizon - 1];
  const uncond = vp.unconditional_sd;
  const falling = uncond !== null && first > uncond;
  const maxSd = Math.max(...vp.sd);
  return (
    <section className="card" style={{ marginTop: 12 }} aria-labelledby="vpath-head" data-testid="variance-path">
      <div className="card-head">
        <span className="card-kind">Second forecast</span>
        <div>
          <div className="card-title" id="vpath-head">
            How uncertain each step is — the variance path
          </div>
          <div className="card-plain">
            The mean path above answers “what will the next value be”. This answers “how unsure should
            I be about it”, and it is the number a volatility model exists to produce. Every step is the
            fitted variance recursion iterated forward, not a single sample-wide standard deviation
            repeated.
          </div>
        </div>
        <div className="card-actions">
          <span className="pill" style={{ color: "var(--ink-soft)" }}>
            {fc.display_name}
          </span>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0 }}>
        <div style={{ display: "flex", gap: 22, flexWrap: "wrap", marginBottom: 8, fontSize: 12.5 }}>
          <div>
            <div style={{ color: "var(--ink-faint)", fontSize: 11 }}>Volatility now → at step {horizon}</div>
            <div style={{ fontFamily: "var(--mono)" }}>
              {num(first, 3)} → {num(last, 3)}
              {uncond !== null && <span style={{ color: "var(--ink-soft)" }}> · long run {num(uncond, 3)}</span>}
            </div>
          </div>
          <div>
            <div style={{ color: "var(--ink-faint)", fontSize: 11 }}>Persistence</div>
            <div style={{ fontFamily: "var(--mono)" }}>
              {vp.persistence === null ? "—" : num(vp.persistence, 3)}
            </div>
          </div>
          <div>
            <div style={{ color: "var(--ink-faint)", fontSize: 11 }}>Leverage</div>
            <div>{vp.leverage ? (vp.leverage.active ? vp.leverage.reading : "none detected") : "not modelled"}</div>
          </div>
        </div>
        <p style={{ margin: "0 0 6px", fontSize: 12.5, color: "var(--ink-soft)" }}>
          {uncond === null
            ? "There is no long-run level: persistence sits at the integrated-variance boundary, so the path does not converge."
            : falling
              ? `The sample ends more turbulent than its own long run, so the path falls toward ${num(uncond, 3)} as the shock fades. A constant-variance model would report one width for all ${horizon} steps — too wide now, too narrow if the calm returns.`
              : `The sample ends calmer than its own long run, so the path rises toward ${num(uncond, 3)}.`}
        </p>
        <Sparkline values={vp.sd} max={maxSd} />
        <table style={{ borderCollapse: "collapse", fontSize: 12, marginTop: 6 }}>
          <thead>
            <tr style={{ color: "var(--ink-faint)", fontSize: 11 }}>
              <th style={{ textAlign: "left", padding: "2px 12px 2px 0" }}>Step</th>
              <th style={{ textAlign: "right", padding: "2px 12px" }}>Variance σ²</th>
              <th style={{ textAlign: "right", padding: "2px 0 2px 12px" }}>Volatility σ</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.h}>
                <td style={{ padding: "2px 12px 2px 0" }}>{r.h}</td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 12px" }}>
                  {num(r.variance, 4)}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 0 2px 12px" }}>
                  {num(r.sd, 4)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * A plain inline SVG line: the in-app preview has no network access, so no chart library.
 *
 * Width is 100% with `preserveAspectRatio="none"` only in the horizontal direction, which is achieved
 * by drawing in a coordinate system whose height is fixed in CSS pixels (`height={84}`) so the stroke
 * is not stretched. The first version used a viewBox that scaled uniformly inside a wide container and
 * rendered as a short squashed line floating in whitespace.
 */
function Sparkline({ values, max }: { values: number[]; max: number }) {
  const w = 1000;
  const h = 84;
  const pad = 6;
  const n = values.length;
  const y = (v: number) => pad + (1 - v / (max || 1)) * (h - 2 * pad);
  const pts = values.map((v, i) => `${(i / Math.max(n - 1, 1)) * w},${y(v)}`).join(" ");
  return (
    <svg
      width="100%"
      height={h}
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      style={{ display: "block", width: "100%", height: h }}
      role="img"
      aria-label={`Variance path from ${values[0].toFixed(3)} to ${values[n - 1].toFixed(3)} over ${n} steps`}
    >
      <line x1={0} y1={h - 1} x2={w} y2={h - 1} stroke="var(--rule)" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      <polyline points={pts} fill="none" stroke="var(--brand)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

function FitDetail({ f }: { f: FitRow }) {
  if (f.failed) {
    return (
      <p style={{ margin: 0, fontSize: 12, color: "var(--bad)" }} data-testid="fit-failure">
        {f.failure_reason}
      </p>
    );
  }
  return (
    <div style={{ fontSize: 12 }}>
      <p style={{ margin: "0 0 4px" }}>{f.plain_one_liner}</p>
      {f.model_card && <ModelCardView card={f.model_card} />}
      {f.roots_ok === false && (
        <p style={{ margin: "0 0 4px", color: "var(--bad)" }}>{f.roots_note}</p>
      )}
      {f.roots_ok === true && f.roots_note && (
        <p style={{ margin: "0 0 4px", color: "var(--warn)" }}>{f.roots_note}</p>
      )}
      {f.caveats.length > 0 && (
        <ul style={{ margin: "2px 0 4px", paddingLeft: 18, color: "var(--ink-soft)" }}>
          {f.caveats.map((c) => (
            <li key={c}>{c}</li>
          ))}
        </ul>
      )}
      {f.params.length > 0 && (
        <table style={{ borderCollapse: "collapse", fontSize: 11.5, marginTop: 4 }}>
          <thead>
            <tr style={{ color: "var(--ink-faint)" }}>
              <th style={{ textAlign: "left", padding: "2px 10px 2px 0" }}>Parameter</th>
              <th style={{ textAlign: "right", padding: "2px 10px" }}>Estimate</th>
              <th style={{ textAlign: "right", padding: "2px 10px" }}>SE</th>
              <th style={{ textAlign: "right", padding: "2px 0 2px 10px" }}>z</th>
            </tr>
          </thead>
          <tbody>
            {f.params.map((p) => (
              <tr key={p.name}>
                <td style={{ padding: "2px 10px 2px 0" }}>{p.name}</td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 10px" }}>
                  {num(p.value, 3)}
                  {p.at_boundary && (
                    <span style={{ color: "var(--warn)", marginLeft: 4 }} title={p.plain}>
                      ·
                    </span>
                  )}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 10px" }}>
                  {num(p.std_err, 3)}
                </td>
                <td style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 0 2px 10px" }}>
                  {num(p.z, 2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

/**
 * The multivariate result — Granger, impulse responses, and the forecast-error decomposition (§5.8 E).
 *
 * Three questions, three blocks, in the order a reader asks them:
 *
 *   1. **Does x help predict y?** The Granger table, which says "helps predict" and never "causes".
 *      A level rejection that disappears once both series are differenced is labelled as the trend —
 *      that flag is the whole reason the table is trustworthy on trending series.
 *   2. **How far does a shock travel?** The IRF, with its bootstrap band drawn as a shaded envelope.
 *      The band is the point: a bare line cannot distinguish a real response from a coefficient that
 *      happened to come out positive.
 *   3. **How much of the uncertainty is each shock responsible for?** The FEVD, printed *with the
 *      Cholesky ordering that produced it*, because a different ordering is a different answer.
 *
 * The ordering, the lag order and the root are all in the header rather than in a tooltip: they are
 * assumptions the reader has to be able to see while reading the numbers.
 */
function MultivariateCard({ m }: { m: MultivariateResponse }) {
  const horizon = m.irf.horizon;
  const k = m.series.length;
  const [response, setResponse] = useState(m.irf.response_to);
  const paths = m.irf.paths[response] ?? [];
  const band = m.irf.bands_90[response];
  // Flattened abs-max across the path and its band, so the band is never clipped by the line's scale.
  // The band is stepped × series like `paths`, which is why it is flattened rather than spread.
  const peak = Math.max(
    1e-12,
    ...paths.flat().map((v) => Math.abs(v)),
    ...(band ? [...band.lower.flat(), ...band.upper.flat()].map((v) => Math.abs(v)) : [0]),
  );
  const share = m.fevd.shares;
  return (
    <section className="card" style={{ marginTop: 12 }} aria-labelledby="mv-head" data-testid="multivariate-card">
      <div className="card-head">
        <span className="card-kind">Joint result</span>
        <div>
          <div className="card-title" id="mv-head">
            How these series move each other
          </div>
          <div className="card-plain">
            Fitted to {m.fitted_on}, {m.lag_order} lag(s) over {k} series. Ordering{" "}
            <span style={{ fontFamily: "var(--mono)" }}>{m.irf.ordering.join(" → ")}</span>: the first
            series is allowed to affect the others within the same period, the last is not. That is an
            assumption, not a finding, and it changes the decomposition below.
          </div>
        </div>
        <div className="card-actions">
          <span className="pill" style={{ color: "var(--ink-soft)" }}>
            {m.model_id.toUpperCase()}
          </span>
          <span
            className="pill"
            style={{
              color: m.stability.stable ? "var(--ok, var(--ink-soft))" : "var(--bad)",
              background: m.stability.stable ? "transparent" : "var(--bad-wash)",
            }}
          >
            largest root {num(m.stability.max_root, 3)}
          </span>
        </div>
      </div>
      <div className="card-body" style={{ paddingTop: 0 }}>
        {/* ---- 1. Granger ---- */}
        <div style={{ fontSize: 12, color: "var(--ink-faint)", marginBottom: 4 }}>
          Does one help predict the other?
        </div>
        {m.granger.length === 0 ? (
          <p style={{ margin: "0 0 10px", fontSize: 12.5, color: "var(--ink-soft)" }}>
            No directional tests for this row: a VECM reports the long-run relationship instead, and its
            adjustment speeds are in the parameter table.
          </p>
        ) : (
          <table style={{ borderCollapse: "collapse", fontSize: 12, marginBottom: 10 }}>
            <thead>
              <tr style={{ color: "var(--ink-faint)", fontSize: 11 }}>
                <th style={{ textAlign: "left", padding: "2px 12px 2px 0" }}>Cause</th>
                <th style={{ textAlign: "left", padding: "2px 12px" }}>Effect</th>
                <th style={{ textAlign: "right", padding: "2px 12px" }}>F</th>
                <th style={{ textAlign: "right", padding: "2px 12px" }}>p</th>
                <th style={{ textAlign: "right", padding: "2px 0 2px 12px" }}>p on differences</th>
              </tr>
            </thead>
            <tbody>
              {m.granger.map((g) => (
                <tr key={`${g.cause}-${g.effect}`} title={g.sentence}>
                  <td style={{ padding: "2px 12px 2px 0" }}>{g.cause}</td>
                  <td style={{ padding: "2px 12px" }}>{g.effect}</td>
                  <td style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 12px" }}>
                    {num(g.f, 2)}
                  </td>
                  <td
                    style={{
                      textAlign: "right",
                      fontFamily: "var(--mono)",
                      padding: "2px 12px",
                      color: g.significant && !g.spurious_risk ? "var(--ink)" : "var(--ink-soft)",
                      fontWeight: g.significant && !g.spurious_risk ? 600 : 400,
                    }}
                  >
                    {pNumber(g.p, 3)}
                  </td>
                  <td style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 0 2px 12px" }}>
                    {pNumber(g.p_on_differences ?? null, 3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {m.granger.some((g) => g.spurious_risk) && (
          <p style={{ margin: "0 0 10px", fontSize: 12, color: "var(--warn)" }}>
            A rejection above disappears once both series are differenced. On trending series that is
            the signature of a shared trend rather than a relationship, so it is not counted as one.
          </p>
        )}

        {/* ---- 2. the impulse response ---- */}
        <div style={{ display: "flex", gap: 16, alignItems: "baseline", fontSize: 12, marginBottom: 4 }}>
          <span style={{ color: "var(--ink-faint)" }}>Response of one series to a one-SD shock in</span>
          <select
            value={response}
            onChange={(e) => setResponse(e.target.value)}
            style={{ fontSize: 12, padding: "1px 4px" }}
            data-testid="irf-series"
          >
            {m.series.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <span style={{ color: "var(--ink-faint)" }}>
            · {m.irf.n_boot} bootstrap draws, 90% band
          </span>
        </div>
        <IrfChart paths={paths} band={band} peak={peak} series={m.series} />

        {/* ---- 3. the decomposition ---- */}
        <div style={{ fontSize: 12, color: "var(--ink-faint)", margin: "10px 0 4px" }}>
          Where each series&apos; forecast error comes from, at step {horizon}
        </div>
        <table style={{ borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ color: "var(--ink-faint)", fontSize: 11 }}>
              <th style={{ textAlign: "left", padding: "2px 12px 2px 0" }}>Forecast error in</th>
              {m.fevd.ordering.map((s) => (
                <th key={s} style={{ textAlign: "right", padding: "2px 12px" }}>
                  from {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(share).map(([target, parts]) => (
              <tr key={target}>
                <td style={{ padding: "2px 12px 2px 0" }}>{target}</td>
                {m.fevd.ordering.map((s) => (
                  <td
                    key={s}
                    style={{ textAlign: "right", fontFamily: "var(--mono)", padding: "2px 12px" }}
                  >
                    {parts[s] === undefined ? "—" : `${(parts[s] * 100).toFixed(1)}%`}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <p style={{ margin: "6px 0 0", fontSize: 12, color: "var(--ink-soft)" }}>
          Shares sum to 100% down each row. The same shock ordering produced every column — with a
          different ordering the numbers move, which is why the ordering is printed above.
        </p>

        {m.adjustment && m.adjustment.length > 0 && (
          <>
            <div style={{ fontSize: 12, color: "var(--ink-faint)", margin: "10px 0 4px" }}>
              How fast each series is pulled back to the long-run relationship
            </div>
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: "var(--ink-soft)" }}>
              {m.adjustment.map((a) => (
                <li key={a.series}>
                  {a.sentence}
                  {a.t !== null && <span style={{ fontFamily: "var(--mono)" }}> · t = {num(a.t, 2)}</span>}
                </li>
              ))}
            </ul>
          </>
        )}
        {m.caveats.length > 0 && (
          <ul style={{ margin: "8px 0 0", paddingLeft: 18, fontSize: 11.5, color: "var(--ink-soft)" }}>
            {m.caveats.map((c) => (
              <li key={c}>{c}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

/**
 * The impulse response, one panel per series, with the bootstrap band as a shaded envelope.
 *
 * Drawn in plain SVG for the same reason as the sparkline: the in-app preview has no network access.
 * Each panel gets its own fixed height so `preserveAspectRatio="none"` stretches only horizontally and
 * the stroke stays the width it was drawn at.
 */
function IrfChart({
  paths,
  band,
  peak,
  series,
}: {
  paths: number[][];
  band?: { lower: number[][]; upper: number[][] };
  peak: number;
  series: string[];
}) {
  if (!paths.length) return null;
  const w = 1000;
  const h = 62;
  const pad = 5;
  // `paths` is one row per STEP and one column per series, so the horizontal extent is `paths.length`.
  // Taking it from `paths[0].length` (the series count) put every point past the right edge of the
  // viewBox, and the SVG clipped the whole path away — an empty panel that rendered without an error.
  const steps = paths.length;
  const x = (i: number) => (i / Math.max(steps - 1, 1)) * w;
  const y = (v: number) => pad + (1 - (v / peak + 1) / 2) * (h - 2 * pad);
  const zero = y(0);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {series.map((name, j) => {
        const line = paths.map((row) => row[j]);
        const pts = line.map((v, i) => `${x(i)},${y(v)}`).join(" ");
        // Column `j` of each row: the band belongs to *this* response series, not to every series.
        const lower = band?.lower.map((row, i) => `${x(i)},${y(row[j])}`).join(" ");
        const upper = band?.upper
          .map((row, i) => `${x(i)},${y(row[j])}`)
          .slice()
          .reverse()
          .join(" ");
        return (
          <div key={name} style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ width: 96, fontSize: 11.5, color: "var(--ink-soft)", flexShrink: 0 }}>
              {name}
            </span>
            {/* The viewBox is what makes `width="100%"` scale the drawing to the box. Without it an
                SVG with no viewBox clips silently at the edge of its viewport, and the `0..1000`
                coordinate system fell entirely outside — an empty panel with no error anywhere. */}
            <svg
              width="100%"
              height={h}
              viewBox={`0 0 ${w} ${h}`}
              preserveAspectRatio="none"
              style={{ display: "block", flex: 1, minWidth: 0 }}
              aria-label={`impulse response of ${name}`}
            >
              <line x1={0} x2={w} y1={zero} y2={zero} stroke="var(--rule)" strokeWidth={1} />
              {lower && upper && (
                <polygon points={`${lower} ${upper}`} fill="var(--brand-wash)" stroke="none" />
              )}
              <polyline points={pts} fill="none" stroke="var(--brand)" strokeWidth={1.6} />
            </svg>
            <span style={{ width: 56, fontSize: 11, fontFamily: "var(--mono)", color: "var(--ink-faint)", flexShrink: 0, textAlign: "right" }}>
              {num(line[line.length - 1] ?? null, 2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * The tabular-model card (§5.8 F).
 *
 * The spec calls it mandatory for a reason: an ML row without its features, lags and validation scheme
 * is an unreadable number in a comparison table. Everything here is what makes the row checkable —
 * which lags, which drivers, whether the split was time-ordered, and the measured coverage of the
 * interval against the coverage it claims.
 */
function ModelCardView({ card }: { card: ModelCardT }) {
  const [open, setOpen] = useState(false);
  const folds = card.holdout.folds;
  return (
    <div
      style={{ margin: "4px 0 6px", padding: "6px 8px", background: "var(--stale-wash)", borderRadius: 4 }}
      data-testid="model-card"
    >
      <div style={{ fontSize: 11.5 }}>
        <strong>Model card.</strong> {card.n_features} features, {card.lags} lags,{" "}
        {card.scheme} multi-step. Validated by {card.validation}.
      </div>
      <div style={{ fontSize: 11.5, color: "var(--ink-soft)", marginTop: 2 }}>
        Holdout MASE {num(card.holdout.mase, 3)} over {card.holdout.n} rows
        {folds.length > 0 && (
          <>
            {" · "}folds{" "}
            {folds.map((f) => num(f.mase, 3)).join(" / ")}
          </>
        )}
        {card.interval_coverage_90 !== null && (
          <>
            {" · "}90% band covered{" "}
            <strong>{(card.interval_coverage_90 * 100).toFixed(0)}%</strong> out of sample
          </>
        )}
      </div>
      <button
        className="btn btn-sm"
        style={{ background: "none", border: "none", color: "var(--brand)", padding: "2px 0", fontSize: 11 }}
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Hide the feature list" : "Show the feature list"}
      </button>
      {open && (
        <div style={{ fontSize: 11.5, color: "var(--ink-soft)", fontFamily: "var(--mono)" }}>
          {card.features.join(" · ")}
        </div>
      )}
    </div>
  );
}
