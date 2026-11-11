"""
ts/forecast/horizons.py — the horizon advisor (§5.11, §6.11).

"Advisor warns if H > n/5, explains why, and can still proceed with an explicit
'I understand' flag." Two distinct cautions live here, and they must not be merged:

* **H > n/5** is a *distance* problem: the model is being asked to walk further than
  the history that taught it to walk. It requires an acknowledgement to proceed.
* **n < 2 periods** is a *season* problem: a seasonal model has seen fewer than two
  full cycles, so its seasonal indices are not identified. It is a warning, never a
  blocker, because a non-seasonal model may still forecast honestly.

The acknowledgement is a real boolean in the request, not a checkbox that only
exists in the UI. A client that omits it is refused by the API (§4.4: an advisor
that recommends but cannot act is decoration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

#: The five levels §5.11 asks for. 66% is kept (rather than rounded to 65 or 68)
#: because it is the approximate one-standard-deviation band and readers recognise it.
COVERAGES: tuple[float, ...] = (0.50, 0.66, 0.80, 0.90, 0.95)

#: The spec's "~1/5 of the training length".
LONG_HORIZON_RATIO = 1.0 / 5.0


@dataclass
class HorizonAdvice:
    """Everything the UI needs to explain a horizon before it spends time forecasting."""

    n: int
    horizon: int
    period: int | None
    recommended_max: int
    long_horizon: bool
    season_short: bool
    acknowledged: bool
    requires_acknowledgement: bool
    allowed: bool
    warnings: list[str] = field(default_factory=list)
    advice: str = ""
    method: str = (
        "H > n/5 needs an explicit acknowledgement; n < 2*period warns about seasonal "
        "history. Both are computed from the in-sample series length only."
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "horizon": self.horizon,
            "period": self.period,
            "recommended_max": self.recommended_max,
            "long_horizon": self.long_horizon,
            "season_short": self.season_short,
            "acknowledged": self.acknowledged,
            "requires_acknowledgement": self.requires_acknowledgement,
            "allowed": self.allowed,
            "warnings": list(self.warnings),
            "advice": self.advice,
            "method": self.method,
            "test_observations_used": 0,
        }


def advise_horizon(
    n: int,
    horizon: int,
    *,
    period: int | None = None,
    acknowledged: bool = False,
) -> HorizonAdvice:
    """
    Build the advice. Pure: no session, no fitted model, no test-period data.

    `n` is the number of **training** observations. Passing the whole series when a
    holdout exists would make H look safe; callers must hand it the training length.
    """
    n, horizon = int(n), int(horizon)
    if n < 1:
        raise ValueError("A forecast needs at least one training observation.")
    if horizon < 1:
        raise ValueError("A forecast horizon must be at least one step.")

    recommended_max = max(1, int(math.floor(n * LONG_HORIZON_RATIO)))
    long_horizon = horizon > n * LONG_HORIZON_RATIO
    season_short = bool(period and period >= 2 and n < 2 * period)

    warnings: list[str] = []
    if long_horizon:
        warnings.append(
            f"A {horizon}-step horizon exceeds one fifth of the {n} training observations "
            f"(about {recommended_max} steps). Beyond that distance the forecast is driven by the "
            "model's structure rather than by observed behaviour, and the intervals are only as "
            "good as that structure. Confirm 'I understand' to proceed, or shorten the horizon."
        )
    if season_short:
        warnings.append(
            f"There are {n} observations and a seasonal period of {period}: less than two full "
            f"seasons ({2 * period} observations). Seasonal effects are weakly identified; "
            "seasonal-naive and seasonal-ARIMA intervals are optimistic here."
        )

    allowed = (not long_horizon) or bool(acknowledged)
    if long_horizon and not acknowledged:
        advice = (
            f"H = {horizon} is longer than n/5 = {n * LONG_HORIZON_RATIO:.1f}. "
            f"Set a horizon of at most {recommended_max}, or acknowledge the risk explicitly."
        )
    elif long_horizon:
        advice = (
            f"H = {horizon} exceeds the recommended {recommended_max} steps; the acknowledgement "
            "is recorded and travels with the forecast."
        )
    elif season_short:
        advice = (
            f"H = {horizon} is within the recommended {recommended_max} steps, but seasonal "
            "history is short — read the seasonal intervals with that in mind."
        )
    else:
        advice = (
            f"H = {horizon} is within the recommended {recommended_max} steps for {n} "
            "observations."
        )

    return HorizonAdvice(
        n=n,
        horizon=horizon,
        period=period,
        recommended_max=recommended_max,
        long_horizon=long_horizon,
        season_short=season_short,
        acknowledged=bool(acknowledged),
        requires_acknowledgement=bool(long_horizon),
        allowed=allowed,
        warnings=warnings,
        advice=advice,
    )


__all__ = ["COVERAGES", "HorizonAdvice", "LONG_HORIZON_RATIO", "advise_horizon"]
