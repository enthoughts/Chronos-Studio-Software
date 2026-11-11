# Chronos Studio — standalone desktop package

`chronos-studio-standalone_0.1.0_amd64.deb` is an installable, offline-first
desktop package for **Linux Mint 22 / Ubuntu 24.04 on x86_64**. It turns the
two-developer-server workspace into one application:

```text
App-menu icon → private Python 3.12 environment → FastAPI + built React UI
              → one Chromium/Chrome app-mode window on 127.0.0.1:8765
```

There is no Vite dev server, no Node.js at runtime, no Docker, no terminal and
no cloud connection unless you explicitly enable Live AI.

**Package:** approximately 109 MB (about 104 MiB).  
**Integrity:** `chronos-studio-standalone_0.1.0_amd64.deb.sha256` contains the exact SHA-256 for the delivered file.

## Install

```bash
cd /home/en/Downloads/Downloads
sha256sum -c chronos-studio-standalone_0.1.0_amd64.deb.sha256
sudo apt install ./chronos-studio-standalone_0.1.0_amd64.deb
```

Then launch **Chronos Studio** from the application menu, or run:

```bash
chronos-studio
```

Requirements declared by the package: `python3.12`, `python3.12-venv`,
`xdg-utils`, `ca-certificates`, `libgomp1`. A Chromium/Chrome browser is
recommended for the chromeless app window; Firefox and the default browser are
fallbacks.

When Chromium/Chrome is unavailable, Firefox is started with a dedicated
`-no-remote` profile under the data directory. A plain `firefox --new-window`
can merely hand the URL to an already-running Firefox and exit immediately; that
short-lived launcher must not be treated as the application window. If neither
browser is available, the default browser is opened and the server stays alive
until Ctrl+C.

The Assistant's first-use payload preview checks `/api/v1/ai/config` as soon as
the drawer opens, **before** a dataset or analysis session exists. It should
resolve within a second. **Offline only** mode does not show the live-AI
preview. If the status ever remains on “Checking live AI availability…”,
reload the page and confirm the local backend is running.

The first launch creates `~/.local/share/chronos-studio/venv` from the bundled
offline wheel archive. **No pip/npm download is needed at install or first
run.** The launcher stores a stamp of the wheel archive and runtime requirements;
if a reinstalled package changes either one, the private venv is rebuilt
offline on the next launch instead of reusing stale dependencies. Uploads and
the browser profile live under the same per-user data directory; sessions remain
in memory with the existing six-hour TTL. A `.chronos` file is a one-way report
export in this build, not a resumable session: restarting the app requires
re-uploading the data and replaying the analysis.

The residual battery in S12 is compatible with both SciPy 1.16 and 1.17. The
bundled wheelhouse pins SciPy 1.16.3, whose `stats.anderson()` has no
`method="interpolate"` keyword; the code now falls back to the older signature
instead of raising an unclassified `internal.error`. If an installed build still
shows that error on `Run diagnostics`, reinstall the current `.deb` and
hard-refresh the page.

S11 and S12 refresh the pipeline state immediately after a successful run, so
the stepper agrees with the backend without a manual `⋯ More → Refresh`.
Catalogue and report status tolerate Theta’s `battery: None` diagnostic block.
The backend test suite still skips the 300-dpi PNG/PDF publication contracts
that need Kaleido, Playwright and a local Chromium, but the HTML-report and
vector-SVG contracts are exercised through the offline fallback. Direct PDF
and 300-dpi PNG downloads degrade with explicit errors, as documented below.

## What is bundled

- Prebuilt React/Plotly UI (`frontend/dist`) served by FastAPI.
- Full backend application code and example assets.
- Offline wheels for FastAPI, Uvicorn, NumPy, pandas, SciPy, statsmodels,
  Plotly, OpenPyXL, python-docx, scikit-learn, `google-genai>=1.0,<2` and their
  transitive dependencies, pinned to CPython 3.12 Linux x86_64 wheels.
- Five CSV datasets and `chronos_dataset_guides.html`.
- Desktop launcher, application-menu entry and SVG icon.
- `standalone/chronos_studio.py` and `standalone/launch.sh`.

## Live AI, privacy and offline mode

- Offline statistical workflows and the offline Assistant work with **no key
  and no internet**.
- Live Gemini requires your own key in **Assistant → Settings** or
  `GEMINI_API_KEY`, plus an outbound connection. No key is bundled, logged,
  echoed or stored server-side.
- The app binds to `127.0.0.1` by default and sends no telemetry.
- The pasted real Google AI Studio key from earlier in this conversation was
  never used; rotate it if it was real.

## Deliberate size optimizations

To fit the workspace budget, the package omits:

- **Playwright, Kaleido and their Chromium browser** — the one-file HTML report
  and every vector SVG figure now use an offline pure-Python renderer, so the
  report keeps its charts. Direct PDF, 300-dpi PNG and the publication-figure
  ZIP's PNG/PDF members report `export.chromium_missing` or
  `catalog.kaleido_missing` and list the affected files in `errors.txt`. To get
  a PDF without those packages, open the downloaded one-file HTML in your
  browser and use **Print → Save as PDF**. Install Kaleido, Playwright and a
  system Chromium only if direct publication exports are essential; HTML,
  DOCX, XLSX, LaTeX, reproducible scripts, the `.chronos` report archive and
  vector SVG figures all work offline.
- **A bundled Python interpreter** — the package uses the distribution's
  Python 3.12 to create a private venv. This keeps the installer around 100 MiB
  instead of several hundred MiB.
- **Auto-update, code signing and notarization** — this is a private/local
  installer, not a public unattended production release.

The v4 source zip remains the fallback for other Python versions or for a
source install. The standalone package includes `scikit-learn`, so the ML model
families remain available; optional exotic libraries such as Prophet,
`statsforecast`, `sktime`, TBATS, LightGBM, XGBoost, `pmdarima` and `arch` are
still feature-flagged as absent, as in the source build.

## Advanced launcher options

```bash
chronos-studio --no-browser --port 9000       # headless; open the printed URL
CHRONOS_DATA_DIR=/tmp/chronos-demo chronos-studio
CHRONOS_PYTHON_BIN=/usr/bin/python3.12 chronos-studio
```

`--host 0.0.0.0` is available for a remote preview; do not expose it on an
untrusted network. `--app-dir`, `--static-dir` and `--data-dir` override the
installed paths for testing.

## Uninstall

```bash
sudo apt remove chronos-studio-standalone
```

User data under `~/.local/share/chronos-studio` is left in place. Remove it
manually if you also want to delete uploads and the private venv.

## Reproducing the build

`standalone/build-deb.py` assembles the package from the workspace source, the
built frontend, the CSV datasets and the offline wheel archive. It writes the
`.deb` and a basename-based `.sha256` sidecar. No secrets are included.
