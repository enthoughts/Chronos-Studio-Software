"""The .chronos report archive: a reviewer can inspect every stored decision."""

from __future__ import annotations

import io
from zipfile import ZIP_DEFLATED, ZipFile

from app.ts.report.bundle import ReportBundle
from app.ts.report.utils import jsonable, safe_json, settings_payload

README = """CHRONOS STUDIO REPORT ARCHIVE (.chronos)

This ZIP is an explicit user export. It contains:
  manifest.json    schema, session id, seed, data hash, stage status, verdict
  data_hash.txt    sha256 prefix of the observed values, for reproducibility
  settings.json    all replayable settings (secrets and fitted wrappers excluded)
  pipeline.json    the full 16-stage pipeline state and progress
  results.json     every ResultBase card as strict, finite JSON
  catalog.json     the Part 16 plot/table catalogue with data behind
  scorecard.json   the weighted S15 readiness scorecard
  manuscript.json  filled Methods/Results/Selection/Diagnostics/Accuracy prose
  caveats.json     travelling caveats and blockers
  data.csv         the untouched table, when one was stored
  README.txt       this file

Open it with any ZIP reader. This build does not restore a .chronos file into
a live session. To replay the analysis, re-upload your data and download the
separate reproduce_analysis.py export or use the stage_calls in settings.json.
No Gemini API key is stored here.
"""


def render_archive(bundle: ReportBundle) -> bytes:
    settings = settings_payload(bundle.session)
    files: dict[str, bytes] = {
        "manifest.json": safe_json(bundle.manifest(), indent=2).encode("utf-8"),
        "data_hash.txt": (bundle.data_hash + "\n").encode("utf-8"),
        "settings.json": safe_json(settings, indent=2).encode("utf-8"),
        "pipeline.json": safe_json(jsonable(bundle.session_export()), indent=2).encode("utf-8"),
        "results.json": safe_json(jsonable(bundle.results_json()), indent=2).encode("utf-8"),
        "catalog.json": safe_json(jsonable(bundle.catalog), indent=2).encode("utf-8"),
        "scorecard.json": safe_json(bundle.scorecard.model_dump(mode="json"), indent=2).encode("utf-8"),
        "manuscript.json": safe_json(jsonable(bundle.sections), indent=2).encode("utf-8"),
        "caveats.json": safe_json({
            "caveats": list(bundle.session.state.caveats),
            "blockers": list(bundle.session.state.blockers),
        }, indent=2).encode("utf-8"),
        "README.txt": README.encode("utf-8"),
    }
    if bundle.data_csv:
        files["data.csv"] = bundle.data_csv
    out = io.BytesIO()
    with ZipFile(out, "w", compression=ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return out.getvalue()


__all__ = ["README", "render_archive"]
