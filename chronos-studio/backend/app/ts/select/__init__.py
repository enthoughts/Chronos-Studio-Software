"""
ts/select/ — model selection (§5.9, Part 13).

Four modules, in dependency order:

* `ic.py` — AIC / AICc / BIC / HQ, plus the **comparability guard**. Information criteria are only
  comparable across models fitted to the same observations with the same differencing, and the guard
  refuses the comparison rather than printing a number that would mislead.
* `cv.py` — the rolling-origin engine. §5.9 makes this the **primary** selection criterion and the ICs
  secondary, which is the opposite of what most software does by default.
* `tests_compare.py` — Diebold–Mariano, Clark–West and a Mincer–Zarnowitz regression, computed on the
  *same* loss series so the comparison is paired.
* `report.py` — the selection report: one table, a ranked shortlist, and a "why this model won"
  paragraph built from the numbers actually computed (never a template with a blank left in).

The engine never picks a model. It produces the evidence and a rank; the caller decides, and the UI
prints both plus the reason they can disagree.
"""

from app.ts.select.ic import (
    IC,
    ICComparison,
    ICKind,
    FitIdentity,
    comparability,
    ic_from_loglik,
    information_criteria,
    rank_by_ic,
)
from app.ts.select.cv import (
    CVFold,
    CVResult,
    CVScheme,
    FoldScore,
    cross_validate,
    fold_origins,
    rank_by_cv,
)
from app.ts.select.tests_compare import (
    ClarkWestResult,
    DMResult,
    MZResult,
    clark_west,
    diebold_mariano,
    mincer_zarnowitz,
)
from app.ts.select.report import (
    SelectionReport,
    ShortlistEntry,
    build_selection_report,
    why_this_model_won,
)

__all__ = [
    "CVFold",
    "CVResult",
    "CVScheme",
    "ClarkWestResult",
    "DMResult",
    "FitIdentity",
    "FoldScore",
    "IC",
    "ICComparison",
    "ICKind",
    "MZResult",
    "SelectionReport",
    "ShortlistEntry",
    "build_selection_report",
    "clark_west",
    "comparability",
    "cross_validate",
    "diebold_mariano",
    "fold_origins",
    "ic_from_loglik",
    "information_criteria",
    "mincer_zarnowitz",
    "rank_by_cv",
    "rank_by_ic",
    "why_this_model_won",
]
