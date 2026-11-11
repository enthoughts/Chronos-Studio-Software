"""
ts.gate — the model-readiness gate (§5.7).

A pure decision layer: `GateFacts` in, `Gate` out. It computes nothing statistical itself — every
number arrives from a stage that already ran — which is what lets it be tested exhaustively without
a session, a server or a dataset.

    from app.ts.gate import GateFacts, assess

    gate = assess(GateFacts(n=240, n_missing=0, breaks_tested=True, ...))
    gate.verdict          # "ready" | "ready_with_caveats" | "not_ready" | "inconclusive"
    gate.blockers         # what must be fixed before a model is fitted
    gate.caveats_travelling   # what must be carried into every downstream output and the report
"""

from app.ts.gate.readiness import (
    CHECKS,
    HORIZON_RATIO,
    MIN_CYCLES,
    MIN_N,
    REMAINDER_HEAVY,
    SEASONAL_STRONG,
    Criterion,
    CriterionState,
    Gate,
    GateFacts,
    assess,
)

__all__ = [
    "CHECKS",
    "HORIZON_RATIO",
    "MIN_CYCLES",
    "MIN_N",
    "REMAINDER_HEAVY",
    "SEASONAL_STRONG",
    "Criterion",
    "CriterionState",
    "Gate",
    "GateFacts",
    "assess",
]
