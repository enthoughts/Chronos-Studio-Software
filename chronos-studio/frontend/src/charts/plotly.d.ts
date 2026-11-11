/**
 * Minimal typings for Plotly's UMD global.
 *
 * The minified distribution is loaded as a local static script in `index.html`
 * rather than bundled by Rollup (see `vite.config.ts`). We only use
 * react / purge / downloadImage, so we describe exactly those.
 */
interface PlotlyApi {
  react(
    el: HTMLElement,
    data: unknown[],
    layout?: Record<string, unknown>,
    config?: Record<string, unknown>,
  ): Promise<unknown>;
  newPlot(
    el: HTMLElement,
    data: unknown[],
    layout?: Record<string, unknown>,
    config?: Record<string, unknown>,
  ): Promise<unknown>;
  purge(el: HTMLElement): void;
  downloadImage(el: HTMLElement, opts: Record<string, unknown>): Promise<string>;
  Plots: { resize(el: HTMLElement): void };
}

interface Window {
  Plotly?: PlotlyApi;
}
