#!/usr/bin/env python3
"""Assemble the Chronos Studio standalone .deb and its checksum sidecar."""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", "*.tsbuildinfo", "*.log",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".env",
)


def copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=IGNORE)


def set_tree_perms(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        Path(dirpath).chmod(0o755)
        for name in filenames:
            Path(dirpath, name).chmod(0o644)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/user/chronos-studio")
    ap.add_argument("--wheels", default="/home/user/build/wheels_full.tar.xz")
    ap.add_argument("--datasets", default="/home/user/uploads")
    ap.add_argument("--guide", default="/home/user/chronos_dataset_guides.html")
    ap.add_argument("--stage", default="/home/user/build/deb_stage")
    ap.add_argument("--output", default="/home/user/chronos-studio-standalone_0.1.0_amd64.deb")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    wheels = Path(args.wheels).resolve()
    datasets = Path(args.datasets).resolve()
    guide = Path(args.guide).resolve()
    stage = Path(args.stage).resolve()
    out = Path(args.output).resolve()

    assert wheels.is_file(), f"missing wheel archive: {wheels}"
    assert guide.is_file(), f"missing dataset guide: {guide}"
    csvs = sorted(datasets.glob("*.csv"))
    assert len(csvs) == 5, f"expected 5 CSV datasets, found {len(csvs)}"
    assert (root / "frontend" / "dist" / "index.html").is_file(), "frontend/dist is missing; run npm run build"
    assert (root / "backend" / "app" / "main.py").is_file(), "backend/app/main.py is missing"
    assert (root / "STANDALONE.md").is_file(), "STANDALONE.md is missing"

    if stage.exists():
        shutil.rmtree(stage)
    debian = stage / "DEBIAN"
    debian.mkdir(parents=True)
    opt = stage / "opt" / "chronos-studio"
    (opt / "standalone").mkdir(parents=True)
    (opt / "backend").mkdir(parents=True)
    (opt / "frontend").mkdir(parents=True)
    (opt / "datasets").mkdir(parents=True)
    (stage / "usr" / "bin").mkdir(parents=True)
    (stage / "usr" / "share" / "applications").mkdir(parents=True)
    (stage / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps").mkdir(parents=True)

    copy_tree(root / "backend" / "app", opt / "backend" / "app")
    copy_tree(root / "frontend" / "dist", opt / "frontend" / "dist")
    for name in ("chronos_studio.py", "launch.sh", "requirements-runtime.txt", "README.md"):
        shutil.copy2(root / "standalone" / name, opt / "standalone" / name)
    shutil.copy2(wheels, opt / "standalone" / "wheels.tar.xz")
    shutil.copy2(root / "STANDALONE.md", opt / "README-STANDALONE.md")
    shutil.copy2(guide, opt / "chronos_dataset_guides.html")
    for csv in csvs:
        shutil.copy2(csv, opt / "datasets" / csv.name)

    (stage / "usr" / "bin" / "chronos-studio").write_text(
        "#!/usr/bin/env bash\n"
        "exec /opt/chronos-studio/standalone/launch.sh \"$@\"\n"
    )
    shutil.copy2(root / "standalone" / "chronos-studio.desktop",
                 stage / "usr" / "share" / "applications" / "chronos-studio.desktop")
    shutil.copy2(root / "standalone" / "chronos-studio.svg",
                 stage / "usr" / "share" / "icons" / "hicolor" / "scalable" / "apps" / "chronos-studio.svg")

    set_tree_perms(stage)
    debian.chmod(0o755)
    for executable in (
        stage / "usr" / "bin" / "chronos-studio",
        opt / "standalone" / "launch.sh",
        opt / "standalone" / "chronos_studio.py",
    ):
        executable.chmod(0o755)

    installed_kib = int(subprocess.check_output(["du", "-sk", str(stage)], text=True).split()[0])
    control = f"""Package: chronos-studio-standalone
Version: 0.1.0
Section: science
Priority: optional
Architecture: amd64
Maintainer: Chronos Studio local build <local@example.invalid>
Installed-Size: {installed_kib}
Depends: python3.12, python3.12-venv, xdg-utils, ca-certificates, libgomp1
Recommends: chromium-browser | google-chrome-stable | firefox
Description: Offline-first time-series analysis workbench
 Chronos Studio is a JASP-like application for EDA, stationarity tests,
 preprocessing, model selection, diagnostics, forecasting and reporting.
 .
 This standalone build serves the prebuilt React UI and FastAPI API from one
 private local process, opens an app-mode browser window, and uses bundled
 offline Python wheels on first launch. Live Gemini answers are optional and
 require a user-supplied key; offline analysis and the offline Assistant need
 neither a key nor an internet connection.
"""
    (debian / "control").write_text(control)
    (debian / "control").chmod(0o644)

    if out.exists():
        out.unlink()
    cmd = [
        "dpkg-deb", "--root-owner-group", "-Zxz", "-z6", "--threads-max=1",
        "--build", str(stage), str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode:
        print(result.stderr.strip())
        return result.returncode

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    Path(str(out) + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"DEB {out} {out.stat().st_size} bytes")
    print(f"SHA256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
