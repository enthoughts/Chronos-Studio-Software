"""
Regenerate the Part 19 golden-file fixtures.

    cd backend
    PYTHONPATH=. python scripts/generate_golden.py

This is intentionally the ONLY way to change `tests/fixtures/golden_models.json`.
Review the diff before committing: a changed AIC is a statistical regression,
not a fixture refresh. The independent statsmodels/R AirPassengers reference in
`airpassengers_reference.json` is generated separately (see CONTRIBUTING.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from golden_common import GOLDEN_MODELS_PATH, build_fixture  # noqa: E402


def main() -> None:
    fixture = build_fixture()
    GOLDEN_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_MODELS_PATH.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n")
    print(f"WROTE {GOLDEN_MODELS_PATH} ({len(fixture['cases'])} cases)")
    for case in fixture["cases"]:
        print(
            f"  {case['family']:<15} {case['model_id']:<16} "
            f"failed={str(case['failed']):<5} aic={case['aic']}"
        )


if __name__ == "__main__":
    main()
