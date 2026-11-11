"""
core/seeds.py — determinism helpers (CRAFT R0.7).

Anything stochastic in Chronos Studio (simulation intervals, randomised PIT,
RQR randomisation, bootstrap bands, ML folds) goes through `seed_everything`
or `rng_for`, so that a result can be reproduced from its recorded seed.
"""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed every global RNG this app might touch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def rng_for(seed: int, *tags: str) -> np.random.Generator:
    """
    A deterministic generator derived from a base seed plus string tags.

    Usage:  rng = rng_for(session.seed, "pit", model_id)
    so that adding a new model does not change the draws of an existing one.
    """
    digest = hashlib.sha256((":".join([str(seed), *tags])).encode()).digest()
    entropy = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(entropy)


def derived_seed(seed: int, *tags: str) -> int:
    digest = hashlib.sha256((":".join([str(seed), *tags])).encode()).digest()
    return int.from_bytes(digest[:4], "big", signed=False) % (2**31 - 1)


__all__ = ["seed_everything", "rng_for", "derived_seed"]
