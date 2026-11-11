"""
ts/select/ic.py — information criteria, and the guard that stops them being compared wrongly.

§5.9 states the rule and then asks for it to be *enforced in code*:

    ICs are only comparable on identical data and identical differencing, and IMA/ARFIMA models' ICs
    are not comparable to ARMA's. Enforce the comparability rule in code and refuse invalid
    comparisons.

This module is that enforcement. Two fits are comparable only when they were scored on the same
number of observations, the same differencing order and the same transform stack. When they are not,
the comparison is **refused with the reason** rather than silently sorted — a sorted column of
incomparable numbers is worse than no column, because it produces a winner.

The formulas (n = observations used in the likelihood, k = free parameters including σ²):

    AIC  = -2ℓ + 2k
    AICc = AIC + 2k(k+1)/(n-k-1)                     (undefined when n ≤ k+1)
    BIC  = -2ℓ + k·ln(n)
    HQ   = -2ℓ + 2k·ln(ln(n))                        (undefined when ln(ln n) ≤ 0, i.e. n ≤ e)

AICc is the default everywhere in this build because forecast comparison almost always has n/k < 40,
where the plain AIC is optimistic enough to change the winner. It is *not* reported when n ≤ k+1:
that is not a large number, it is a division by zero wearing a hat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence


class ICKind(str, Enum):
    """Which criterion. `AICC` is the default for the reasons in the module docstring."""

    AIC = "aic"
    AICC = "aicc"
    BIC = "bic"
    HQ = "hq"


#: The two-unit rule of thumb from Burnham & Anderson: models within 2 IC units of the best are not
#: meaningfully distinguishable, and calling one of them "the winner" is a coin flip reported as a
#: finding. The UI draws this band rather than only ranking.
INDISTINGUISHABLE_DELTA = 2.0

#: Beyond this the model has essentially no support. Used for the wording, not the arithmetic.
NO_SUPPORT_DELTA = 10.0


@dataclass(frozen=True)
class IC:
    """One model's criteria. Any of them may be `None` — absent beats fabricated."""

    aic: float | None = None
    aicc: float | None = None
    bic: float | None = None
    hq: float | None = None
    n: int = 0
    k: int = 0
    loglik: float | None = None
    #: Why a criterion is missing, when it is. Empty when all four are present.
    notes: tuple[str, ...] = ()

    def get(self, kind: ICKind) -> float | None:
        return {
            ICKind.AIC: self.aic,
            ICKind.AICC: self.aicc,
            ICKind.BIC: self.bic,
            ICKind.HQ: self.hq,
        }[kind]

    @property
    def best_available(self) -> ICKind:
        """The strongest criterion this fit can actually offer, in the order the app prefers them."""
        for kind in (ICKind.AICC, ICKind.BIC, ICKind.AIC, ICKind.HQ):
            if self.get(kind) is not None:
                return kind
        return ICKind.AICC

    def as_dict(self) -> dict[str, object]:
        return {
            "aic": self.aic,
            "aicc": self.aicc,
            "bic": self.bic,
            "hq": self.hq,
            "n": self.n,
            "k": self.k,
            "loglik": self.loglik,
            "notes": list(self.notes),
        }


def information_criteria(
    loglik: float | None, *, n: int, k: int, penalty_scale: float = 1.0
) -> IC:
    """
    All four criteria from one log-likelihood.

    `penalty_scale` multiplies k. It exists for the quasi-likelihood rows (the GARCH family's
    Bollerslev–Wooldridge standard errors, the count models' estimated dispersion), where the
    likelihood is not a true likelihood and the honest thing is to scale the penalty up rather than
    pretend the number is exact. It defaults to 1.0 and is recorded in `notes` when it is not.
    """
    notes: list[str] = []
    n = int(n)
    k = int(k)
    if loglik is None or not math.isfinite(loglik):
        return IC(n=n, k=k, loglik=loglik, notes=("No likelihood was computed, so no criterion can be.",))
    if n <= 0 or k < 1:
        return IC(n=n, k=k, loglik=loglik, notes=("Degenerate fit: n or k is not positive.",))

    kk = k * float(penalty_scale)
    if penalty_scale != 1.0:
        notes.append(
            f"The penalty is scaled by {penalty_scale:g} because this row's likelihood is a "
            "quasi-likelihood; its criteria are comparable with each other, not with a true-likelihood row."
        )

    aic = -2.0 * loglik + 2.0 * kk

    aicc: float | None
    if n - kk - 1 > 0:
        aicc = aic + (2.0 * kk * (kk + 1.0)) / (n - kk - 1.0)
    else:
        aicc = None
        notes.append(
            f"AICc is undefined here: {int(kk)} parameters against {n} observations leaves nothing "
            "to divide by. The AIC is reported instead and is optimistic by construction."
        )

    bic = -2.0 * loglik + kk * math.log(n)

    hq: float | None
    if n > math.e:
        hq = -2.0 * loglik + 2.0 * kk * math.log(math.log(n))
    else:
        hq = None
        notes.append("HQ is undefined below n = e; there are too few observations for its penalty.")

    return IC(
        aic=_finite(aic),
        aicc=_finite(aicc),
        bic=_finite(bic),
        hq=_finite(hq),
        n=n,
        k=k,
        loglik=_finite(loglik),
        notes=tuple(notes),
    )


def ic_from_loglik(loglik: float | None, n: int, k: int) -> IC:
    """Alias kept short for the wrappers, which call this on every fit."""
    return information_criteria(loglik, n=n, k=k)


def _finite(x: float | None) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# --------------------------------------------------------------------------- #
# The comparability guard
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FitIdentity:
    """
    What makes two fits comparable: the data they were scored on, and the shape they were fitted in.

    Everything here is a *fact about the fit*, recorded at fit time. The guard compares identities; it
    never re-derives them, because deriving "the same data" from two models that each kept their own
    effective sample is exactly the step that goes wrong.
    """

    model_id: str
    model_class: str = "arma"
    """How the row relates to a plain likelihood: `arma`, `ima` (integrated MA / ARFIMA), `quasi`
    (GARCH / count / intermittent, whose log-likelihood is a quasi-likelihood), `none` (no likelihood
    at all: ML, k-NN, SVR, GP, sets, combination)."""
    n_obs: int = 0
    d: int = 0
    D: int = 0
    transform_stack: tuple[str, ...] = ()
    loglik: float | None = None
    k: int = 0
    ic: IC | None = None

    @property
    def has_ic(self) -> bool:
        return self.ic is not None and self.ic.aicc is not None


@dataclass
class ICComparison:
    """The verdict of the guard, plus the ranked table when it passes."""

    comparable: bool
    reason: str
    """Empty when comparable; otherwise one sentence naming what differs."""
    groups: dict[str, list[str]] = field(default_factory=dict)
    """When not comparable: the identity key → the models that share it. Usually two groups."""
    ranked: list[tuple[str, IC]] = field(default_factory=list)
    """When comparable: (model_id, ic) sorted by the criterion, best first."""

    def as_dict(self) -> dict[str, object]:
        return {
            "comparable": self.comparable,
            "reason": self.reason,
            "groups": {k: list(v) for k, v in self.groups.items()},
            "ranked": [{"model_id": m, "ic": ic.as_dict()} for m, ic in self.ranked],
        }


def _identity_key(f: FitIdentity) -> tuple:
    """Two fits share a key exactly when their criteria may be sorted against each other."""
    return (
        f.n_obs,
        f.d,
        f.D,
        tuple(f.transform_stack),
        f.model_class,
    )


def _group_label(key: tuple, members: Sequence[FitIdentity]) -> str:
    n, d, D, stack, cls = key
    parts = [f"{members[0].model_id}…"]
    parts.append(f"n={n}")
    if d or D:
        parts.append("differencing " + "".join(f"D{d}" if d else "" for _ in (0,)) + f"(d={d}, D={D})")
    if stack:
        parts.append("transform " + "+".join(stack))
    parts.append(f"likelihood class {cls}")
    return " · ".join(parts)


def comparability(
    fits: Sequence[FitIdentity], *, kind: ICKind = ICKind.AICC
) -> ICComparison:
    """
    May these fits' criteria be sorted against each other?

    The rule is applied pairwise-by-group rather than by a global flag: two ARMA fits on an
    untransformed series are comparable even if a GARCH row is also present, because the GARCH row is
    simply not in their comparison. Refusing the *whole table* because one row is incomparable would
    be over-refusal, and silently dropping the row hides that it was fitted.

    The four facts that must match:

    1. **n** — a likelihood computed on 168 observations is not comparable with one computed on 180.
    2. **d/D** — differencing changes the likelihood's scale by the Jacobian, so a model fitted to
       differences and one fitted to levels cannot be sorted by an IC even when both are "ARIMA".
    3. **transform stack** — a log transform is a nonlinear reparameterisation with its own Jacobian.
    4. **likelihood class** — a quasi-likelihood (GARCH, count, Croston) has no true likelihood to
       maximise, and an integrated-MA row's likelihood ignores the initial values that define it.
    """
    usable = [f for f in fits if f.has_ic]
    if not usable:
        return ICComparison(
            comparable=False,
            reason="No fit here has a comparable information criterion, so nothing can be ranked by one.",
        )

    buckets: dict[tuple, list[FitIdentity]] = {}
    for f in usable:
        buckets.setdefault(_identity_key(f), []).append(f)

    groups = {_group_label(k, v): [f.model_id for f in v] for k, v in buckets.items()}

    if len(buckets) == 1:
        key = next(iter(buckets))
        members = buckets[key]
        ranked = sorted(
            ((f.model_id, f.ic) for f in members if f.ic is not None and f.ic.get(kind) is not None),
            key=lambda pair: pair[1].get(kind) or math.inf,  # type: ignore[union-attr]
        )
        return ICComparison(comparable=True, reason="", groups=groups, ranked=ranked)

    # More than one group. Say precisely which fact differs, because "not comparable" alone sends the
    # reader looking for a bug in the IC formula.
    keys = list(buckets)
    first = keys[0]
    differing: list[str] = []
    for other in keys[1:]:
        if other[0] != first[0]:
            differing.append(f"observation counts {first[0]} vs {other[0]}")
        if (other[1], other[2]) != (first[1], first[2]):
            differing.append(
                f"differencing (d={first[1]}, D={first[2]}) vs (d={other[1]}, D={other[2]})"
            )
        if other[3] != first[3]:
            differing.append(
                "transforms " + "+".join(first[3] or ("none",)) + " vs " + "+".join(other[3] or ("none",))
            )
        if other[4] != first[4]:
            differing.append(f"likelihood classes {first[4]} vs {other[4]}")
        break  # one clear example is more useful than an exhaustive list

    join = "; ".join(dict.fromkeys(differing)) or "different fit shapes"
    return ICComparison(
        comparable=False,
        reason=(
            "These fits cannot be ranked by an information criterion because they were not scored on "
            f"the same thing — {join}. An IC differences likelihoods, so the likelihoods have to be "
            "for the same observations. Use the rolling-origin accuracy instead, which is measured on "
            "a common holdout."
        ),
        groups=groups,
    )


def rank_by_ic(fits: Sequence[FitIdentity], *, kind: ICKind = ICKind.AICC) -> list[tuple[str, IC]]:
    """
    Rank by a criterion, or return an empty list when the guard refuses.

    Returning empty rather than "best effort" is the point: a caller that wants a rank must handle
    the refusal, and the only way to handle it is to look at why.
    """
    verdict = comparability(fits, kind=kind)
    return verdict.ranked if verdict.comparable else []


def delta_ic(ranked: Sequence[tuple[str, IC]], *, kind: ICKind = ICKind.AICC) -> list[float]:
    """Δ against the best, the scale the two-unit band is drawn on."""
    if not ranked:
        return []
    values = [ic.get(kind) for _, ic in ranked]
    best = min(v for v in values if v is not None) if any(v is not None for v in values) else None
    if best is None:
        return []
    return [float("nan") if v is None else float(v - best) for v in values]


def support_label(delta: float) -> str:
    """
    Burnham & Anderson's evidence language, as a label rather than a probability.

    The Akaike *weights* are computed in `report.py; this function only names the band, because a
    weight printed to three decimals implies a precision the two-unit rule does not support.
    """
    if not math.isfinite(delta):
        return "not comparable"
    if delta <= INDISTINGUISHABLE_DELTA:
        return "indistinguishable from the best"
    if delta <= 4.0:
        return "considerably less support"
    if delta <= NO_SUPPORT_DELTA:
        return "substantially less support"
    return "essentially no support"


def akaike_weights(values: Iterable[float | None]) -> list[float]:
    """Akaike weights from Δ values. Kept here so the report and the UI use one implementation."""
    vals = [v for v in values]
    deltas = [v if v is not None and math.isfinite(v) else math.inf for v in vals]
    if not deltas or all(not math.isfinite(d) for d in deltas):
        return [0.0 for _ in deltas]
    best = min(deltas)
    exp = [math.exp(-0.5 * (d - best)) if math.isfinite(d) else 0.0 for d in deltas]
    total = sum(exp)
    return [e / total if total > 0 else 0.0 for e in exp]


__all__ = [
    "IC",
    "ICComparison",
    "ICKind",
    "FitIdentity",
    "INDISTINGUISHABLE_DELTA",
    "NO_SUPPORT_DELTA",
    "akaike_weights",
    "comparability",
    "delta_ic",
    "ic_from_loglik",
    "information_criteria",
    "rank_by_ic",
    "support_label",
]
