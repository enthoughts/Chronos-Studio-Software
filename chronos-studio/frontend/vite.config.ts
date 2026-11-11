import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * The UI talks to the API through a relative path (/api) so that the same build
 * works behind the workspace preview proxy, on localhost, and in production.
 * Never hardcode localhost in browser code.
 */
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: true,
    // The preview host is a different origin; allow it explicitly.
    allowedHosts: true,
    proxy: {
      "/api": { target: API_TARGET, changeOrigin: true },
    },
  },
  // Plotly is loaded as a local UMD script (see index.html / scripts/copy-plotly.mjs),
  // so it is not part of the Rollup graph. Source maps for the app code are still
  // cheap; bundling the 4.4 MB minified library with maps exhausted the sandbox.
  build: { outDir: "dist", sourcemap: false },
});
