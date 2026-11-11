"""
ts/metrics/policy.py — which accuracy metrics are legal here, and why (§5.12).

A metric is not neutral. MAPE divides by the truth, so a single zero makes it
infinite; sMAPE has a 0–200% scale and is not a percentage error; CRPS needs a
predictive distribution, not a point; and a count forecast has a PMF, not a
density. This module is the one place where the app *chooses* a metric set from
measured facts, and attaches the reason to the result. Callers may compute extra
diagnostics, but they may never promote a banned metric into the headline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: A series shorter than this, or shorter than three seasonal cycles, gets the
#: short-series policy: a scaled error and an absolute error, not a percentage.
SHORT_SERIES_N = 40
#: §5.8's intermittent threshold; also used for the "zero-heavy" MAPE ban.
ZERO_HEAVY_SHARE = 0.30

METRIC_LABELS: dict[str, str] = {
    "mae": "MAE", "rmse": "RMSE", "mase": "MASE", "rmsse": "RMSSE",
    "mpe": "MPE", "mape": "MAPE", "smape": "sMAPE", "gmrae": "GMRAE",
    "mdrae": "MdRAE", "rel_mae": "relative MAE", "rel_rmse": "relative RMSE",
    "crps": "CRPS", "pinball": "pinball loss", "log_score": "log score",
    "winkler": "Winkler / interval score", "coverage": "PI coverage",
    "calibration": "calibration curve", "pit": "PIT", "rqr": "RQR",
    "discrete_crps": "discrete CRPS", "randomized_pit": "randomized PIT",
    "qlike": "QLIKE", "mse_variance": "MSE of variance",
    "kupiec": "Kupiec VaR coverage", "christoffersen": "Christoffersen VaR independence",
}


@dataclass(frozen=True)
class MetricPolicy:
    """The chosen metric set, the banned ones, and the reason in plain words."""

    cluster: str
    primary: str
    metrics: tuple[str, ...]
    banned: tuple[str, ...]
    reasons: tuple[str, ...]
    notes: tuple[str, ...] = ()
    probabilistic: bool = False
    discrete: bool = False
    variance: bool = False

    def allows(self, metric: str) -> bool:
        key = metric.lower()
        return key not in self.banned and key in self.metrics

    @property
    def headline(self) -> str:
        return (
            f"{self.cluster.replace('_', ' ').title()} metric set: "
            + ", ".join(METRIC_LABELS.get(m, m) for m in self.metrics)
            + "."
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "cluster": self.cluster,
            "primary": self.primary,
            "primary_label": METRIC_LABELS.get(self.primary, self.primary),
            "metrics": list(self.metrics),
            "metric_labels": {m: METRIC_LABELS.get(m, m) for m in self.metrics},
            "banned": list(self.banned),
            "banned_labels": {m: METRIC_LABELS.get(m, m) for m in self.banned},
            "reasons": list(self.reasons),
            "notes": list(self.notes),
            "headline": self.headline,
            "probabilistic": self.probabilistic,
            "discrete": self.discrete,
            "variance": self.variance,
        }


def choose_policy(
    *,
    data_class: Any,
    n: int,
    period: int | None = None,
    horizon: int = 1,
    zero_share: float = 0.0,
    probabilistic: bool = False,
    discrete: bool = False,
    target_kind: str = "level",
    short_series: bool | None = None,
) -> MetricPolicy:
    """Pick the metric set from facts, never from a UI toggle alone."""
    dc = str(getattr(data_class, "value", data_class) or "continuous").lower()
    zero_heavy = bool(zero_share >= ZERO_HEAVY_SHARE)
    # A declared intermittent series is NOT a PMF/count target. Sending it down the
    # discrete branch would replace the required MASE/RMSSE/CRPS set with density
    # scores it cannot justify. Count/event data wins over a high zero share.
    is_count = dc in {"count", "event_timestamp"} or (discrete and dc != "intermittent")
    is_intermittent = dc == "intermittent" or (zero_heavy and not is_count)
    short = (
        bool(short_series) if short_series is not None else
        int(n) < max(SHORT_SERIES_N, (int(period or 1) * 3) + max(int(horizon), 1))
    )

    if target_kind == "variance":
        return MetricPolicy(
            cluster="volatility_target",
            primary="qlike",
            metrics=("qlike", "mse_variance", "kupiec", "christoffersen"),
            banned=("mape", "smape", "mpe", "crps", "pinball"),
            reasons=(
                "The target is a variance / volatility path, so scoring a mean forecast would "
                "answer a different question.",
                "QLIKE is the proper scoring rule for variance forecasts; VaR backtests judge the "
                "tail separately.",
            ),
            notes=(
                "Realised variance is proxied by squared returns, which is noisy; QLIKE and the "
                "backtests are reported with that limitation.",
                "Kupiec tests unconditional coverage, Christoffersen adds independence of violations; "
                "both have low power in short samples.",
            ),
            variance=True,
        )

    if is_count:
        return MetricPolicy(
            cluster="count_discrete",
            primary="discrete_crps",
            metrics=("discrete_crps", "log_score", "randomized_pit", "mase", "mae", "coverage"),
            banned=("mape", "smape", "mpe", "gmrae", "mdrae"),
            reasons=(
                f"The data class is {dc.replace('_', ' ')} and the model supplies a discrete PMF, "
                "so the score uses the probability of the observed count rather than a density.",
                "MAPE and sMAPE are forbidden on count / intermittent data; MAE is shown only "
                "beside MASE, as §5.12 requires.",
                "Discrete PIT is randomized inside each probability atom; otherwise a correct "
                "Poisson model would look non-uniform on a histogram.",
            ),
            notes=(
                "Log score = -mean(log P(Y = y)); an observed count outside the stored PMF "
                "support has zero probability and refuses the mean instead of using a finite "
                "edge penalty.",
                "MASE still needs the naive / seasonal-naive benchmark fitted on the training "
                "window only (§6.4).",
            ),
            probabilistic=True,
            discrete=True,
        )

    if is_intermittent:
        return MetricPolicy(
            cluster="intermittent_zero_heavy",
            primary="mase",
            metrics=("mase", "rmsse", "crps", "mae", "winkler", "coverage"),
            banned=("mape", "smape", "mpe", "gmrae", "mdrae"),
            reasons=(
                f"Zero-heavy data ({zero_share:.1%} zeros) makes a percentage error undefined or "
                "misleading; MASE and RMSSE scale by the in-sample naive error instead.",
                "A probabilistic forecaster is available, so CRPS and the interval score lead the "
                "distributional comparison.",
            ),
            notes=(
                "Sharpness is read subject to calibration: a narrower band only wins after its "
                "empirical coverage matches the nominal level.",
            ),
            probabilistic=True,
        )

    if short:
        return MetricPolicy(
            cluster="short_series",
            primary="mase",
            metrics=("mase", "mae", "rmse", "coverage"),
            banned=(),
            reasons=(
                f"Only {int(n)} observations are available (seasonal period {period or 'none'}, "
                f"horizon {int(horizon)}), so percentage and probabilistic rankings would be "
                "unstable; MASE + MAE are the robust pair.",
            ),
            notes=("Intervals are approximate at this sample size and are shown with their "
                   "coverage rather than ranked on width alone.",),
        )

    if probabilistic:
        return MetricPolicy(
            cluster="probabilistic",
            primary="crps",
            metrics=("crps", "pinball", "pit", "winkler", "coverage", "mase", "mae", "rmse"),
            banned=(),
            reasons=(
                "At least one fitted model supplies a predictive distribution, so CRPS, pinball "
                "loss and PIT judge the whole distribution rather than just its centre.",
                "Point errors are still shown, but they no longer decide the ranking.",
            ),
            notes=("Sharpness subject to calibration is the stated evaluation philosophy: "
                   "compare widths only among models whose coverage is consistent with nominal.",),
            probabilistic=True,
        )

    return MetricPolicy(
        cluster="continuous",
        primary="mase",
        metrics=("mae", "rmse", "mase", "mape", "smape", "gmrae", "mdrae", "coverage"),
        banned=(),
        reasons=(
            "The target is a continuous level series with no near-zero warning, so absolute, "
            "squared and scaled point errors are reported together.",
        ),
        notes=(
            "MAPE is refused (not silently skipped) if any actual is near zero.",
            "sMAPE is on a 0–200% scale and is not a percentage error; its note travels with the "
            "table.",
            "Intervals are approximate unless the fitted model has a full predictive distribution.",
        ),
    )


__all__ = ["METRIC_LABELS", "MetricPolicy", "SHORT_SERIES_N", "ZERO_HEAVY_SHARE", "choose_policy"]
