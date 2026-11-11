"""S14 must replay the documented S7 seasonal-difference + S3 imputation path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.api.metrics import _needs_original
from app.ts.metrics.protocols import _check_records


def _record(step: str, method: str) -> SimpleNamespace:
    return SimpleNamespace(step=step, method=method)


def test_difference_and_kalman_imputation_are_replayable():
    records = [
        _record("difference", "difference"),
        _record("impute", "kalman_smoother"),
    ]
    assert _check_records(records) == ["difference", "kalman_smoother"]
    assert _needs_original(records, None)[0] is True


def test_difference_alone_still_requires_the_original_levels():
    records = [_record("difference", "difference")]
    assert _check_records(records) == ["difference"]
    assert _needs_original(records, None)[0] is True


@pytest.mark.parametrize(
    "step,method",
    [
        ("trim", "trim"),
        ("aggregate", "aggregate"),
        ("smooth", "moving_average"),
        ("outlier", "delete"),
    ],
)
def test_observation_set_changes_remain_refused(step, method):
    with pytest.raises(ValueError, match="S14 needs the original regular series"):
        _check_records([_record(step, method)])
