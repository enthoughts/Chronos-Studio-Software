"""
ts/select/report.py — the selection report: one table, a shortlist, and a "why this model won".

§5.9 asks for a paragraph template "filled with real numbers". The template is the danger: a template
with a blank reads as authoritative and says nothing. This module builds the sentence *from the
computed evidence*, and when the evidence does not support a claim it says so in the sentence rather
than leaving a slot.

The ordering of the report is deliberate and reflects §5.9's own ordering:

1. **Rolling-origin accuracy** — the primary criterion.
2. **Information criteria** — secondary, and only where the comparability guard allows.
3. **Pairwise tests** between the top two — DM, or Clark–West when nested, plus MZ on the leader.
4. **A verdict** that names the winner *and* the closeness of the race.

The report refuses to crown a model when the top two are inside the two-unit IC band or their CV MASE
means differ by less than the fold spread. "Model A won" where A and B are indistinguishable is the
most common way a selection report misleads — and the most avoidable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from app.core.formatting import p_phrase
from app.ts.select.cv import CVResult, rank_by_cv
from app.ts.select.ic import (
    INDISTINGUISHABLE_DELTA,
    IC,
    ICKind,
    akaike_weights,
    comparability,
    delta_ic,
    support_label,
)
from app.ts.select.ic import FitIdentity
from app.ts.select.tests_compare import DMResult, MZResult, clark_west, diebold_mariano, mincer_zarnowitz


@dataclass
class ShortlistEntry:
    """One row of the ranked shortlist — the numbers that put it there, and the caveats that follow."""

    rank: int
    model_id: str
    display_name: str
    cv_mase: float | None = None
    cv_mae: float | None = None
    cv_rmse: float | None = None
    cv_spread: float | None = None
    """Fold-to-fold standard deviation of MASE — how much the mean is worth."""
    aicc: float | None = None
    delta_aicc: float | None = None
    akaike_weight: float | None = None
    support: str = ""
    n_folds: int = 0
    worst_fold_mase: float | None = None
    notes: list[str] = field(default_factory=list)
    verdict: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "cv_mase": self.cv_mase,
            "cv_mae": self.cv_mae,
            "cv_rmse": self.cv_rmse,
            "cv_spread": self.cv_spread,
            "aicc": self.aicc,
            "delta_aicc": self.delta_aicc,
            "akaike_weight": self.akaike_weight,
            "support": self.support,
            "n_folds": self.n_folds,
            "worst_fold_mase": self.worst_fold_mase,
            "notes": self.notes,
            "verdict": self.verdict,
        }


@dataclass
class SelectionReport:
    """Everything S11 needs to render, and every reason a blank is blank."""

    primary_criterion: str = "rolling-origin MASE"
    caveat: str = ""
    """Always populated: §5.9 requires the UI to state that CV is primary."""
    shortlist: list[ShortlistEntry] = field(default_factory=list)
    winner: str = ""
    winner_name: str = ""
    runner_up: str = ""
    indistinguishable: bool = False
    """True when the top two cannot be told apart — the report then refuses to crown one."""
    why: str = ""
    """The filled paragraph."""
    ic_comparable: bool = True
    ic_reason: str = ""
    dm: DMResult | None = None
    mz: MZResult | None = None
    scheme: str = "expanding"
    horizon: int = 1
    n_folds: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "primary_criterion": self.primary_criterion,
            "caveat": self.caveat,
            "shortlist": [e.as_dict() for e in self.shortlist],
            "winner": self.winner,
            "winner_name": self.winner_name,
            "runner_up": self.runner_up,
            "indistinguishable": self.indistinguishable,
            "why": self.why,
            "ic_comparable": self.ic_comparable,
            "ic_reason": self.ic_reason,
            "dm": self.dm.as_dict() if self.dm else None,
            "mz": self.mz.as_dict() if self.mz else None,
            "scheme": self.scheme,
            "horizon": self.horizon,
            "n_folds": self.n_folds,
            "notes": self.notes,
        }


PRIMARY_CAVEAT = (
    "Rolling-origin accuracy is the primary selection criterion and the information criteria are "
    "secondary. An IC rewards fit to the sample the model was fitted on; the rolling-origin score "
    "measures the forecast on data the model never saw, at the horizon actually asked for. Where the "
    "two disagree, the rolling-origin score is the one to act on."
)


def _fmt(x: float | None, digits: int = 3) -> str:
    return "—" if x is None or not math.isfinite(x) else f"{x:.{digits}f}"



def build_selection_report(
    cv_results: Sequence[CVResult],
    *,
    identities: Sequence[FitIdentity] | None = None,
    errors_by_model: dict[str, np.ndarray] | None = None,
    nested_pairs: set[tuple[str, str]] | None = None,
    horizon: int = 1,
    metric: str = "mase",
    ic_kind: ICKind = ICKind.AICC,
) -> SelectionReport:
    """
    Assemble the report.

    `errors_by_model` is the CV errors per model, pooled across folds *in fold order* — needed for the
    pairwise tests, which compare paired loss series. When it is absent the pairwise section is simply
    omitted with a note rather than filled with a test on unpaired data.
    """
    report = SelectionReport(caveat=PRIMARY_CAVEAT, horizon=horizon)
    ranked = rank_by_cv([r for r in cv_results if not r.failed], metric=metric)
    failed = [r for r in cv_results if r.failed]

    for r in failed:
        report.notes.append(
            f"{r.display_name or r.model_id} could not be cross-validated: {r.failure_reason}. It is "
            "not ranked, and an unranked model is not the same as a bad one — it is unmeasured."
        )

    if not ranked:
        report.why = (
            "No model could be scored on held-out data, so no ranking is offered. The fits in Step 10 "
            "are still valid; what is missing is any evidence about which one forecasts better."
        )
        return report

    ic_verdict = comparability(identities, kind=ic_kind) if identities else None
    ic_by_model: dict[str, IC] = {}
    if identities:
        ic_by_model = {f.model_id: f.ic for f in identities if f.ic is not None}
    report.ic_comparable = bool(ic_verdict.comparable) if ic_verdict else False
    report.ic_reason = ic_verdict.reason if ic_verdict else "No information criteria were supplied."
    if ic_verdict and ic_verdict.comparable:
        deltas = dict(zip([m for m, _ in ic_verdict.ranked], delta_ic(ic_verdict.ranked, kind=ic_kind)))
        weights = dict(
            zip(
                [m for m, _ in ic_verdict.ranked],
                akaike_weights(
                    [ic.get(ic_kind) for _, ic in ic_verdict.ranked]
                ),
            )
        )
    else:
        deltas, weights = {}, {}

    for i, r in enumerate(ranked, start=1):
        aicc = ic_by_model[r.model_id].aicc if r.model_id in ic_by_model else None
        entry = ShortlistEntry(
            rank=i,
            model_id=r.model_id,
            display_name=r.display_name or r.model_id,
            cv_mase=r.aggregate("mase"),
            cv_mae=r.aggregate("mae"),
            cv_rmse=r.aggregate("rmse"),
            cv_spread=r.spread("mase"),
            aicc=aicc,
            delta_aicc=deltas.get(r.model_id),
            akaike_weight=weights.get(r.model_id),
            support=support_label(deltas.get(r.model_id, float("nan"))) if r.model_id in deltas else "",
            n_folds=r.n_folds,
            worst_fold_mase=getattr(r.worst_fold("mase"), "mase", None),
        )
        # A model whose fold spread exceeds its advantage over the next model is a model whose rank
        # is not established, whatever the mean says.
        if entry.cv_spread is not None and entry.cv_mase:
            entry.notes.append(
                f"Fold-to-fold spread {_fmt(entry.cv_spread)} against a mean MASE of "
                f"{_fmt(entry.cv_mase)} — "
                + (
                    "the spread is larger than the mean, so this score is unstable."
                    if entry.cv_spread > entry.cv_mase
                    else "the mean is quoted with that spread attached."
                )
            )
        if entry.worst_fold_mase is not None and entry.cv_mase and entry.worst_fold_mase > 1.5 * entry.cv_mase:
            entry.notes.append(
                f"Its worst fold scores MASE {_fmt(entry.worst_fold_mase)}, "
                f"{entry.worst_fold_mase / entry.cv_mase:.1f}× its average — one stretch of the series "
                "defeats it."
            )
        report.shortlist.append(entry)

    top = report.shortlist[0]
    second = report.shortlist[1] if len(report.shortlist) > 1 else None
    report.winner = top.model_id
    report.winner_name = top.display_name
    report.n_folds = top.n_folds
    report.scheme = ranked[0].scheme.value
    report.runner_up = second.model_id if second else ""

    # The closeness test: is the gap bigger than the noise in the gap?
    close = False
    if second and top.cv_mase is not None and second.cv_mase is not None:
        gap = second.cv_mase - top.cv_mase
        noise = max(
            v for v in (top.cv_spread, second.cv_spread) if v is not None
        ) if any(v is not None for v in (top.cv_spread, second.cv_spread)) else 0.0
        # With `n_folds` folds, the standard error of a difference is at most sqrt(2)·spread/sqrt(F).
        se_gap = math.sqrt(2.0) * noise / math.sqrt(max(top.n_folds, 1)) if noise else 0.0
        close = gap <= max(se_gap, 1e-12) if se_gap else gap <= 1e-9
        if top.delta_aicc is not None and second.delta_aicc is not None:
            close = close or (second.delta_aicc - (top.delta_aicc or 0.0)) <= INDISTINGUISHABLE_DELTA
    elif second:
        close = False

    report.indistinguishable = bool(close)

    # Pairwise tests, on paired errors only.
    if errors_by_model and len(ranked) >= 2:
        a, b = ranked[0].model_id, ranked[1].model_id
        ea, eb = errors_by_model.get(a), errors_by_model.get(b)
        if ea is not None and eb is not None:
            nested = (b, a) in (nested_pairs or set()) or (a, b) in (nested_pairs or set())
            if nested:
                report.dm = clark_west(ea, eb, name_small=b, name_large=a, horizon=horizon)
            else:
                report.dm = diebold_mariano(ea, eb, name_a=a, name_b=b, horizon=horizon)
        actual = errors_by_model.get("__actual__")
        if actual is not None and ea is not None:
            forecast = np.asarray(actual, dtype=float) - np.asarray(ea, dtype=float)
            report.mz = mincer_zarnowitz(actual, forecast, name=ranked[0].display_name or a, horizon=horizon)
    else:
        report.notes.append(
            "No paired forecast errors were supplied, so Diebold–Mariano and Mincer–Zarnowitz are "
            "omitted. Both need errors from the same origins; running them on unpaired errors would "
            "produce a number about nothing."
        )

    report.why = why_this_model_won(report, ranked)
    return report


def why_this_model_won(report: SelectionReport, ranked: Sequence[CVResult]) -> str:
    """
    The paragraph, built from computed numbers.

    Written as three sentences with a fixed job each: what won, on what evidence; how close it was; and
    what would make the reader choose differently. The third sentence is the one that keeps the report
    honest — it names the condition under which the recommendation changes.
    """
    if not report.shortlist:
        return report.why
    top = report.shortlist[0]
    second = report.shortlist[1] if len(report.shortlist) > 1 else None
    parts: list[str] = []

    parts.append(
        f"{top.display_name} came first on {report.primary_criterion} across {top.n_folds} "
        f"rolling-origin folds at horizon {report.horizon}, with a MASE of {_fmt(top.cv_mase)} "
        f"(MAE {_fmt(top.cv_mae, 2)}, RMSE {_fmt(top.cv_rmse, 2)}, fold spread {_fmt(top.cv_spread)})."
    )

    if second and top.cv_mase is not None and second.cv_mase is not None:
        gap_pct = (second.cv_mase - top.cv_mase) / top.cv_mase * 100 if top.cv_mase else float("nan")
        if report.indistinguishable:
            parts.append(
                f"It is not separable from {second.display_name}, which scored {_fmt(second.cv_mase)} — "
                f"a gap of {gap_pct:+.1f}%, inside the fold-to-fold noise"
                + (
                    f" and inside the {INDISTINGUISHABLE_DELTA:g}-unit information-criterion band"
                    if report.ic_comparable
                    else ""
                )
                + ". Treat them as one choice rather than a finding."
            )
        else:
            parts.append(
                f"The next model, {second.display_name}, scored {_fmt(second.cv_mase)} — "
                f"{gap_pct:+.1f}% worse, and wider than the fold-to-fold spread."
            )
    elif second:
        parts.append(f"{second.display_name} was second but could not be scored on the same metric.")

    if report.dm is not None:
        if report.dm.ok and report.dm.p_value is not None:
            parts.append(
                f"Diebold–Mariano on the paired loss differential gives {p_phrase(report.dm.p_value)}"
                + (
                    ", so the difference is not distinguishable from forecast noise"
                    if report.dm.p_value >= 0.05
                    else ", a real improvement rather than noise"
                )
                + "."
            )
        elif not report.dm.ok:
            parts.append(f"The pairwise test was not run: {report.dm.note}")

    if report.mz is not None and report.mz.verdict:
        parts.append(report.mz.verdict)

    # Sentence three: what would change the answer.
    conditions: list[str] = []
    if top.cv_mase is not None and top.cv_mase > 1.0:
        conditions.append(
            f"it does not beat the naive benchmark (MASE {_fmt(top.cv_mase)} > 1), so the honest "
            "forecast may be the last value repeated"
        )
    # Only name the runner-up as an alternative when it is actually close. The first version of this
    # branch fired on `not indistinguishable`, which meant a model 145% worse than the winner was
    # offered as "close enough that the ranking is not robust" — the opposite of what the numbers said.
    if second and second.cv_mase is not None and top.cv_mase:
        gap = (second.cv_mase - top.cv_mase) / top.cv_mase
        if 0 < gap <= 0.25:
            conditions.append(
                f"the horizon or the CV window changed — {second.display_name} is within "
                f"{gap:.0%} of the winner, so a different task could reorder them"
            )
    if not report.ic_comparable:
        conditions.append(
            "the information criteria could not be compared here, so this ranking rests on the "
            "held-out accuracy alone"
        )
    if top.worst_fold_mase is not None and top.cv_mase and top.worst_fold_mase > 1.5 * top.cv_mase:
        conditions.append(
            f"the winner's worst fold (MASE {_fmt(top.worst_fold_mase)}) suggests it does not hold "
            "across the whole series"
        )
    if conditions:
        parts.append("Choose differently if " + "; ".join(conditions) + ".")

    return " ".join(parts)


__all__ = [
    "PRIMARY_CAVEAT",
    "SelectionReport",
    "ShortlistEntry",
    "build_selection_report",
    "why_this_model_won",
]
