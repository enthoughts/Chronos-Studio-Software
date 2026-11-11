"""
core/formatting.py — one convention for p-values in prose.

Two mistakes kept recurring in the first thirteen parts:

* `f"{p:.3f}"` printed **p = 0.000** for a p-value that is merely very small. A p-value is never
  zero, and "0.000" teaches the reader that the test returned an exact impossibility.
* `f"{p:.1e}"` printed **p = 0.0e+00** (or 3.4e-120), which reads like a formatting bug.
* After fixing the second, several modules printed **p = < 0.0001** — an equals sign in front of
  an inequality.

Every sentence that quotes a p-value therefore uses `p_phrase`, which returns a *complete*
comparison: `p < 0.0001`, or `p = 0.0075`. Callers interpolate it directly, without adding "p = ".
A missing value says so in words rather than showing a dash that could be read as a number.
"""

from __future__ import annotations

import math

#: Below this, journals write "p < 0.0001" rather than quoting a number whose accuracy is optical.
SMALL_P = 1e-4


def p_phrase(p: float | None, digits: int = 4) -> str:
    """`p < 0.0001` or `p = 0.0075`, including the comparison operator."""
    if p is None or not math.isfinite(p):
        return "the p-value is not available"
    if p < SMALL_P:
        return f"p < {SMALL_P:.4f}"
    # A fixed-point format with 3 or 4 decimals would round 0.00037 to "0.000";
    # keep three significant digits for the values just above our cutoff.
    if p < 1e-3:
        return f"p = {p:.3g}"
    return f"p = {p:.{digits}f}"


def p_number(p: float | None, digits: int = 4) -> str:
    """Just the number, for a table cell that already has its own `p` column header."""
    if p is None or not math.isfinite(p):
        return "—"
    if p < SMALL_P:
        return f"< {SMALL_P:.4f}"
    return f"{p:.3g}" if p < 1e-3 else f"{p:.{digits}f}"


__all__ = ["SMALL_P", "p_phrase", "p_number"]
