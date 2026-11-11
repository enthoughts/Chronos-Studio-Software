/**
 * Copy the minified Plotly UMD build into public/ before a production build.
 *
 * Rollup must not see this 4.4 MB file: bundling it (with sourcemaps) exhausted
 * the sandbox's memory. index.html loads it as a local script instead, which also
 * keeps charts working in the network-less in-app preview.
 */
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const source = resolve(root, "node_modules/plotly.js-dist-min/plotly.min.js");
const target = resolve(root, "public/plotly.min.js");

mkdirSync(dirname(target), { recursive: true });
copyFileSync(source, target);
console.log(`Copied ${source} -> ${target}`);
