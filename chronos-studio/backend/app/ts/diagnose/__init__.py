"""
ts/diagnose/__init__.py — the residual battery and the adequacy verdict (§5.9, Part 13).

The spec's battery is long, and the trap in a long battery is that each test gets quoted as if it were
independent evidence. It is not: twelve tests at 5% will reject a good model with probability 1 − 0.95¹²
≈ 46%, and a report that lists twelve p-values and calls the model inadequate because two are low is
doing arithmetic with that fact.

So the battery is organised in two layers:

* `battery.py` — every test, computed once, with its own precondition and its own reading. Nothing is
  aggregated into a score.
* `verdict.py` — the adequacy verdict, which names **the specific failing test** and its remedy. It is
  allowed to say `INADEQUATE` for one reason and to disagree with a low p-value elsewhere, because a
  lone rejection among twelve with no mechanism behind it is what multiple testing looks like.

`calibration.py` holds PIT and randomized quantile residuals — the two diagnostics that test the
*predictive distribution* rather than the mean, and the ones most often implemented in a way that
cannot be interpreted (see its docstring on why PIT must be randomized for discrete models).
"""

from app.ts.diagnose.battery import (
    DIAGNOSTIC_LAG_RULE,
    DiagnosticBattery,
    acf_with_bounds,
    box_pierce,
    cusum_path,
    jarque_bera_test,
    ljung_box_multi,
    runs_test,
    shapiro_test,
    anderson_darling_test,
    sign_test,
    run_battery,
)
from app.ts.diagnose.calibration import (
    PITResult,
    RQRResult,
    coverage_tests,
    pit_from_quantiles,
    rqr_from_draws,
)
from app.ts.diagnose.verdict import (
    Adequacy,
    Verdict,
    diagnose_model,
    judge,
)

__all__ = [
    "DIAGNOSTIC_LAG_RULE",
    "Adequacy",
    "DiagnosticBattery",
    "PITResult",
    "RQRResult",
    "Verdict",
    "acf_with_bounds",
    "anderson_darling_test",
    "box_pierce",
    "coverage_tests",
    "cusum_path",
    "diagnose_model",
    "jarque_bera_test",
    "judge",
    "ljung_box_multi",
    "pit_from_quantiles",
    "rqr_from_draws",
    "run_battery",
    "runs_test",
    "shapiro_test",
    "sign_test",
]
