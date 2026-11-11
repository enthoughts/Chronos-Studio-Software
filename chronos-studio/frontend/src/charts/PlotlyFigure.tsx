/**
 * charts/PlotlyFigure.tsx — the one place Plotly is mounted.
 *
 * Why a local UMD script: the in-app preview runs in a sandboxed iframe with no
 * network access, so a CDN script would silently fail to load. `plotly.min.js`
 * is served from `public/` and sets `window.Plotly` before this module runs. It
 * also keeps the 4.4 MB minified library out of Rollup's graph — bundling it
 * made `vite build` exhaust the sandbox's memory.
 *
 * Why `Plotly.react`: figures re-render on theme change / resize without leaking
 * DOM nodes, and the config below is the accessibility and export story in one
 * place (screen-reader description, PNG/SVG export, no "edit in Plotly Cloud").
 */

import { useEffect, useId, useRef } from "react";
import { figureValueSummary } from "../state/accessible";

export interface PlotlyFigureProps {
  /** A Plotly figure dict produced by the backend chart factory. */
  figure: Record<string, unknown>;
  /** Accessible description, generated server-side from the numbers. */
  altText?: string;
  height?: number;
  /** Called with the PNG data URL when the user exports. */
  onExport?: (format: "png" | "svg") => void;
}

const CONFIG = {
  displayModeBar: true,
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: [
    "sendDataToCloud",
    "lasso2d",
    "select2d",
    "autoScale2d",
    "toggleSpikelines",
  ] as string[],
  toImageButtonOptions: {
    format: "png" as const,
    filename: "chronos-figure",
    height: 800,
    width: 1200,
    scale: 2,
  },
};

export function PlotlyFigure({ figure, altText, height, onExport }: PlotlyFigureProps) {
  const ref = useRef<HTMLDivElement>(null);
  const summaryId = useId();
  const traces = ((figure.data as unknown[]) ?? []).length;
  const valueSummary = figureValueSummary(altText ?? "", traces);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const Plotly = window.Plotly;
    if (!Plotly) {
      const message = document.createElement("div");
      message.style.cssText =
        "display:grid;place-items:center;height:100%;color:var(--ink-faint);font-size:12.5px";
      message.textContent = altText || "The chart library did not load.";
      el.replaceChildren(message);
      return;
    }
    const layout = {
      ...(figure.layout as Record<string, unknown> | undefined),
      autosize: true,
      ...(height ? { height } : {}),
    };
    const data = (figure.data as unknown[]) ?? [];

    // A figure with no traces is almost always a computation that found nothing;
    // showing an empty axis is friendlier than a blank box.
    if (data.length === 0) {
      el.innerHTML =
        '<div style="display:grid;place-items:center;height:100%;color:var(--ink-faint);font-size:12.5px">Nothing to plot for this result.</div>';
      return;
    }

    Plotly.react(el, data, layout, CONFIG as unknown as Record<string, unknown>);

    if (onExport) {
      (el as unknown as { on: (e: string, cb: () => void) => void }).on?.("plotly_buttonclicked", () => {
        onExport("png");
      });
    }

    return () => {
      try {
        Plotly.purge(el);
      } catch {
        /* already gone */
      }
    };
  }, [figure, height, onExport, altText]);

  return (
    <figure style={{ margin: 0 }}>
      {/* Part 19: the spoken description includes the series count and the
          backend-generated summary of the actual plotted values. */}
      <p id={summaryId} className="sr-only">{valueSummary}</p>
      <div
        ref={ref}
        role="img"
        aria-label={altText || "Chart"}
        aria-describedby={summaryId}
        style={{ width: "100%", minHeight: height ?? 320 }}
      />
      {altText && (
        <figcaption
          style={{
            fontSize: 11,
            color: "var(--ink-faint)",
            marginTop: 6,
            lineHeight: 1.45,
            maxHeight: 48,
            overflow: "hidden",
          }}
        >
          {altText}
        </figcaption>
      )}
    </figure>
  );
}
