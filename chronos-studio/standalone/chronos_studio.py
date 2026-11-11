#!/usr/bin/env python3
"""Chronos Studio standalone desktop launcher.

Serves the prebuilt React UI and the FastAPI API from one private local HTTP
server, then opens a browser window in app mode. No Vite dev server, no Node.js
at runtime, and no cloud connection unless the user enables Live AI.
"""
from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch Chronos Studio as a local desktop app")
    p.add_argument("--host", default=os.getenv("CHRONOS_HOST", "127.0.0.1"),
                   help="Bind address. Use 0.0.0.0 only for a remote preview.")
    p.add_argument("--port", type=int, default=int(os.getenv("CHRONOS_PORT", "0")),
                   help="Port; 0 picks the first free port from 8765, 8766, 8767, then OS-assigned.")
    p.add_argument("--no-browser", action="store_true",
                   help="Do not open a browser window; useful for headless previews.")
    p.add_argument("--app-dir", default=None, help="Override the installed application root.")
    p.add_argument("--static-dir", default=None, help="Override the prebuilt frontend directory.")
    p.add_argument("--data-dir", default=None, help="Override the per-user data directory.")
    return p.parse_args()


def choose_port(host: str, preferred: int) -> int:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    candidates = [preferred] if preferred else [8765, 8766, 8767, 0]
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
                return int(sock.getsockname()[1])
            except OSError:
                continue
    raise RuntimeError("No free local TCP port could be found.")


def find_chromium() -> str | None:
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        found = shutil.which(name)
        if found:
            return found
    return None


def open_app_window(url: str, data_dir: Path, holder: dict) -> None:
    """Open a dedicated browser window and record whether its lifetime is safe to watch.

    A plain `firefox --new-window URL` is only a short-lived launcher when an
    existing Firefox instance is running: it forwards the URL and exits. Watching
    that process made the server shut down immediately. Use a separate profile
    with `-no-remote` for Firefox, and never watch the generic webbrowser fallback.
    """
    chrome = find_chromium()
    if chrome:
        profile = data_dir / "chromium-profile"
        profile.mkdir(parents=True, exist_ok=True)
        cmd = [
            chrome,
            f"--app={url}",
            "--window-size=1440,900",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
        ]
        try:
            holder["proc"] = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            holder["watch"] = True
            return
        except OSError:
            pass
    firefox = shutil.which("firefox")
    if firefox:
        profile = data_dir / "firefox-profile"
        profile.mkdir(parents=True, exist_ok=True)
        cmd = [firefox, "-no-remote", "-profile", str(profile), "-new-window", url]
        try:
            holder["proc"] = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            holder["watch"] = True
            return
        except OSError:
            pass
    # xdg-open / webbrowser launchers often exit after handing off to an
    # existing browser. Keep the server alive and require Ctrl+C instead.
    webbrowser.open_new(url)
    holder["proc"] = None
    holder["watch"] = False


def main() -> int:
    args = parse_args()
    app_dir = Path(args.app_dir or os.getenv("CHRONOS_APP_DIR") or Path(__file__).resolve().parent.parent).resolve()
    static_dir = Path(args.static_dir or os.getenv("CHRONOS_STATIC_DIR") or app_dir / "frontend" / "dist").resolve()
    backend_dir = (app_dir / "backend").resolve()
    default_data = Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))) / "chronos-studio"
    data_dir = Path(args.data_dir or os.getenv("CHRONOS_DATA_DIR") or default_data).resolve()

    if not (static_dir / "index.html").is_file():
        print(f"ERROR: built UI not found at {static_dir / 'index.html'}", file=sys.stderr)
        return 2
    if not (backend_dir / "app" / "main.py").is_file():
        print(f"ERROR: backend not found under {backend_dir}", file=sys.stderr)
        return 2

    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "uploads").mkdir(parents=True, exist_ok=True)

    port = choose_port(args.host, args.port)
    loopback_host = "127.0.0.1" if args.host in {"0.0.0.0", "::", ""} else args.host
    url = f"http://{loopback_host}:{port}/"

    # Environment must be set before app.core.config is imported.
    os.environ["CHRONOS_STANDALONE"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    os.environ["CHRONOS_DEBUG"] = "false"
    os.environ["CHRONOS_ALLOW_ALL_ORIGINS"] = "true" if args.host in {"0.0.0.0", "::"} else "false"
    os.environ["CHRONOS_ALLOWED_ORIGINS"] = f"http://127.0.0.1:{port},http://localhost:{port}"
    os.environ["CHRONOS_UPLOAD_DIR"] = str(data_dir / "uploads")
    os.environ["CHRONOS_SESSION_TTL_HOURS"] = "6"
    os.environ["CHRONOS_APP_DIR"] = str(app_dir)
    os.environ["CHRONOS_STATIC_DIR"] = str(static_dir)
    os.environ["CHRONOS_DATA_DIR"] = str(data_dir)
    chrome = find_chromium()
    if chrome:
        os.environ.setdefault("CHRONOS_CHROME", chrome)

    sys.path.insert(0, str(backend_dir))

    import uvicorn
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from app.main import app as api_app

    # Replace the development JSON root with the built single-page app.
    api_app.router.routes = [
        route for route in api_app.router.routes
        if getattr(route, "path", None) != "/"
    ]

    @api_app.get("/", include_in_schema=False)
    def _serve_index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    api_app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    config = uvicorn.Config(
        api_app,
        host=args.host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    holder: dict = {"proc": None, "watch": False}

    if not args.no_browser:
        def _delayed_open() -> None:
            time.sleep(1.0)
            open_app_window(url, data_dir, holder)

        def _watch_window() -> None:
            # Ignore launchers that exit immediately because they forwarded the
            # URL to an already-running browser. Only a browser that stayed
            # alive past the grace period is treated as the app window.
            grace_until = time.monotonic() + 5.0
            while not server.should_exit:
                proc = holder.get("proc")
                if holder.get("watch") and proc is not None and proc.poll() is not None:
                    if time.monotonic() < grace_until:
                        holder["proc"] = None
                        holder["watch"] = False
                        return
                    server.should_exit = True
                    return
                time.sleep(0.5)

        threading.Thread(target=_delayed_open, daemon=True).start()
        threading.Thread(target=_watch_window, daemon=True).start()

    print(f"Chronos Studio: {url}", flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print("Press Ctrl+C to quit.", flush=True)
    server.run()
    if holder.get("watch"):
        proc = holder.get("proc")
        if proc is not None and proc.poll() is None:
            proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
