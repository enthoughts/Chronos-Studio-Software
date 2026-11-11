# Install Chronos Studio locally on Linux Mint Cinnamon

This guide is for a **private, local install**. It does not need Docker, a public
URL, a database, or a Gemini API key. Internet access is needed only for the
first `pip` / `npm` dependency download.

The clean workspace archive is **`CHRONOS-local-install-v4.zip`**, with
**`CHRONOS-local-install-v4.zip.sha256`**. Its root folder is `CHRONOS/`, so
extracting it into `/home/en/Downloads/Downloads/` creates exactly:

```text
/home/en/Downloads/Downloads/CHRONOS/
```

> **Security note — read this if you downloaded an earlier archive.**  
> The first archive, `CHRONOS-local-install.zip`, accidentally bundled the
> gitignored `backend/.env` containing a Gemini key value. **Delete that archive
> and any folder extracted from it.** If that key was real, revoke or rotate it
> at <https://aistudio.google.com/apikey>. The **v2, v3 and v4** archives contain **no
> `backend/.env` and no API key**; Chronos runs in offline mode without one.
> **Use v4**; v2 and v3 are clean but stale. v4 adds the Live AI capability
> fix and includes every earlier S1, stage-navigation and S7 Apply fix.

---

## 0.1 Alternative: installable desktop package

If you want a double-click application instead of the source/terminal workflow,
use **`chronos-studio-standalone_0.1.0_amd64.deb`** and its `.sha256` sidecar.
It targets Linux Mint 22 / Ubuntu 24.04 x86_64 with Python 3.12, bundles an
offline wheelhouse, serves the built UI and API from one local process, and
opens a Chromium/Chrome app-mode window. No Node.js, Vite dev server or Docker
is needed at runtime.

```bash
cd /home/en/Downloads/Downloads
sha256sum -c chronos-studio-standalone_0.1.0_amd64.deb.sha256
sudo apt install ./chronos-studio-standalone_0.1.0_amd64.deb
```

See [`STANDALONE.md`](STANDALONE.md) for requirements, first-run behavior,
privacy notes and the PDF/300-dpi PNG export caveat. The v4 source archive
below remains the fallback for other Python versions or a source install.

---

## 0. What is in the archive

The archive contains the full project source, docs, tests, example datasets,
`chronos_dataset_guides.html`, the five attached CSVs in `datasets/`,
`frontend/public/plotly.min.js`, deploy files and the audit notes. It **excludes**
generated dependency, secret and cache files such as `frontend/node_modules`,
`frontend/dist`, `frontend/tsconfig.tsbuildinfo`, `backend/.venv`,
`backend/.env`, `.coverage`, `__pycache__`, `.pytest_cache`, `.mypy_cache`,
`.ruff_cache` and `*.pyc`. Those are recreated by the install script.

---

## 1. System prerequisites

Linux Mint 22.x ships with Python 3.12, which is fine. This project declares
`requires-python >= 3.11`; it was validated in the audit sandbox on Python 3.13,
and the Docker images use Python 3.11. If your “latest Python” is 3.14 and a
scientific wheel such as `scipy` or `statsmodels` fails to build, use Python
3.12 or 3.13 instead:

```bash
python3 --version
python3.12 --version 2>/dev/null || true
python3.13 --version 2>/dev/null || true
```

Install the base packages and a current Node.js. Vite 5 / React 18 need Node 18
or newer; **Node 20 LTS is recommended**:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip build-essential curl unzip git ca-certificates
# If Mint's nodejs is older than 18, use NodeSource:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node -v    # should be v20.x
npm -v
```

If you already have Node 20+, skip the NodeSource lines.

---

## 2. Download and extract

In Arena, download **both** `CHRONOS-local-install-v4.zip` and
`CHRONOS-local-install-v4.zip.sha256`. Put both files directly in
`/home/en/Downloads/Downloads/` — your browser saved the earlier archive there,
not in `~/Downloads/`.

```bash
cd /home/en/Downloads/Downloads
ls -l CHRONOS-local-install-v4.zip*

# Verify before extracting:
sha256sum -c CHRONOS-local-install-v4.zip.sha256
# must print: CHRONOS-local-install-v4.zip: OK
# If you did not download the .sha256 file, compare `sha256sum CHRONOS-local-install-v4.zip`
# with the hash printed in the release message.

# Remove any earlier or wrongly nested extraction, then extract HERE:
rm -rf CHRONOS
unzip CHRONOS-local-install-v4.zip
cd CHRONOS
ls
```

`ls` must show `backend/`, `frontend/`, `scripts/`, `README.md`,
`INSTALL_LOCAL.md` and `AUDIT.md` **directly**. The installer must be at:

```text
/home/en/Downloads/Downloads/CHRONOS/scripts/install-local-linux-mint.sh
```

If you instead see
`/home/en/Downloads/Downloads/CHRONOS/CHRONOS/scripts/install-local-linux-mint.sh`,
you ran `unzip` from inside `CHRONOS`; delete the outer `CHRONOS/` and repeat
the commands above from its parent:

```bash
cd /home/en/Downloads/Downloads
rm -rf CHRONOS
unzip CHRONOS-local-install-v4.zip
cd CHRONOS
```

If the browser saved the files one level up, move them first:

```bash
mv ~/Downloads/CHRONOS-local-install-v4.zip* /home/en/Downloads/Downloads/
```

---

## 3. One-command install

The helper script creates `backend/.venv`, installs the deploy requirements,
creates a blank `backend/.env` from the template, generates example datasets if
needed, installs the frontend packages, and checks that the backend imports:

```bash
cd /home/en/Downloads/Downloads/CHRONOS
bash scripts/install-local-linux-mint.sh
```

To use a specific Python interpreter:

```bash
PYTHON_BIN=python3.12 bash scripts/install-local-linux-mint.sh
```

What it installs:

- backend: `backend/requirements-deploy.txt` = core + `python-docx`,
  `playwright`, `scikit-learn`, `google-genai`, plus the test tools;
- frontend: `npm ci` from `frontend/package-lock.json`;
- examples: `backend/scripts/gen_examples.py`;
- `backend/.env` with a **blank** `GEMINI_API_KEY` (offline mode, no secret).

It does **not** install `weasyprint`, `arch`, `pmdarima`, `prophet`,
`statsforecast`, `sktime`, `tbats`, `lightgbm` or `xgboost`. Those optional
families stay visibly disabled in `/api/v1/health` and the UI. PDF and 300-dpi
PNG exports need a Chromium binary; see §6.

---

## 4. Run it

Use the wrapper, which activates the backend virtual environment and then calls
the existing dev script:

```bash
cd /home/en/Downloads/Downloads/CHRONOS
bash scripts/run-local.sh
```

You should see:

```text
→ backend  http://localhost:8000/docs
→ frontend http://localhost:5173
```

Open **<http://localhost:5173>** in your browser. The Vite dev server proxies
relative `/api/` calls to `http://127.0.0.1:8000`, so no hostnames need editing.
Press `Ctrl+C` in the terminal to stop both processes.

If you prefer to run manually:

```bash
cd /home/en/Downloads/Downloads/CHRONOS
source backend/.venv/bin/activate
bash scripts/dev.sh
```

---

## 5. Verify the install

Quick checks:

```bash
curl -fsS http://localhost:8000/api/v1/health
curl -fsS http://localhost:5173/api/v1/health
```

Optional full verification (the backend suite takes several minutes):

```bash
cd /home/en/Downloads/Downloads/CHRONOS/backend
source .venv/bin/activate
PYTHONPATH=. pytest -q
PYTHONPATH=. pytest -q --cov=app/ts --cov-fail-under=80

cd ../frontend
npm run typecheck
npm test
npm run build
```

Historical audit: 1014 backend tests and 91.61% `app/ts` coverage. Current v4
validation: 60 frontend tests, clean typecheck and build; the build still emits
a harmless “chunk larger than 500 kB” advisory.

---

## 6. Optional features

- **Offline assistant:** works with no key. Do nothing.
- **Live Gemini answers:** the installer now installs the compatible
  `google-genai>=1.0,<2` SDK from `backend/requirements-deploy.txt`. Supply
  your own key either in **Assistant → Settings** (kept in this browser tab's
  `sessionStorage`) or through the environment / `backend/.env`:
  ```bash
  cd /home/en/Downloads/Downloads/CHRONOS
  nano backend/.env          # set GEMINI_API_KEY=your-own-key
  # or: export GEMINI_API_KEY=your-own-key
  bash scripts/run-local.sh
  ```
  Never commit, share or screenshot a real key. The archive ships no key. If
  you installed dependencies manually and Live AI still says the SDK is
  missing, run `pip install -r backend/requirements-deploy.txt` inside
  `backend/.venv`, then restart the backend. Assistant Settings now states
  whether the SDK or the key is the missing piece; offline mode always works.
- **PDF / 300-dpi PNG reports:** install Chromium/Chrome via your package
  manager, then set:
  ```bash
  export CHRONOS_CHROME=/usr/bin/chromium
  ```
  or build the Docker image with `INSTALL_BROWSER=1`. Without a browser the
  app writes a clear `errors.txt` entry for those two export kinds instead of
  pretending they succeeded.
- **All optional model families / readers:** from `backend/`, with the venv
  active, install only the packages you actually want, for example:
  ```bash
  pip install -r requirements-deploy.txt
  pip install arch pmdarima statsforecast sktime tbats lightgbm xgboost prophet pyreadstat pyreadr weasyprint kaleido
  ```
  Or add the chosen packages to `requirements-deploy.txt` and rerun the install
  script. `weasyprint` also needs system Pango/Cairo libraries. The
  `pyproject.toml` extras are declarations; the tested local path is the
  requirements files, not an editable `pip install -e .` from `backend/`.
- **No Docker:** local development uses `uvicorn` + `vite` directly. Docker is
  only for deployment.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `unzip: cannot find /home/en/Downloads/CHRONOS-local-install-v4.zip` | The file is in `/home/en/Downloads/Downloads/`; `cd` there first |
| `sha256sum: … No such file or directory` | Download the `.sha256` next to the zip, or use the inline hash from the release message |
| `CHRONOS/CHRONOS/...` nested folder | You unzipped from inside `CHRONOS`; `cd .. && rm -rf CHRONOS && unzip …` |
| `python3 -m venv` fails | `sudo apt install python3-venv` |
| `npm ci` fails or Vite errors | Node is too old; install Node 20 and rerun |
| `pip` cannot build `scipy` / `statsmodels` | Use Python 3.12 or 3.13, not 3.14; `PYTHON_BIN=python3.12 bash scripts/install-local-linux-mint.sh` |
| `uvicorn: command not found` | Activate `backend/.venv` or use `scripts/run-local.sh` |
| Port 5173 or 8000 busy | Stop the other process, or edit `scripts/dev.sh` / `frontend/vite.config.ts` |
| Charts do not render | Confirm `frontend/public/plotly.min.js` exists; run `cd frontend && npm run prebuild` |
| PDF/PNG export missing | Install Chromium and set `CHRONOS_CHROME` |
| `ModuleNotFoundError: app` | Run backend commands from the `backend/` directory with `PYTHONPATH=.` |
| Live AI unavailable | Read the status in Assistant → Settings. Missing SDK: install `backend/requirements-deploy.txt` and restart the backend. Missing key: enter your own key in Settings or set `GEMINI_API_KEY`; offline mode needs neither |

Do not use `sudo pip`. Keep everything inside `backend/.venv`.

---

## 8. Update / uninstall

To update, replace the extracted `CHRONOS/` directory with a newer archive and
rerun `bash scripts/install-local-linux-mint.sh`. To uninstall, stop the dev
server and delete the directory:

```bash
rm -rf /home/en/Downloads/Downloads/CHRONOS
```

There is no database and no system-wide service; uploaded data and sessions
live in memory / a temporary upload directory and disappear on restart.
