"""predictive/transform.py - optional monotone transform of the regression target.

Selected once in TARGET.target_transform and applied consistently across select / tune /
train / design so every stage optimizes and reports in the same space.

  "none" : identity (any target).
  "lce"  : LCE = -log10(1 - target/100), for a 0-100 % *efficiency* target. It stretches the
           high tail (98-99.9 %) so the model resolves differences that raw % compresses.
"""
from __future__ import annotations

import numpy as np

from molforge.predictive.target import TARGET


def forward(y):
    """Original target -> model space."""
    if TARGET.target_transform == "lce":
        return -np.log10(np.clip(1.0 - np.asarray(y, float) / 100.0, 1e-3, 1.0))
    return np.asarray(y, float)


def inverse(z):
    """Model space -> original target."""
    if TARGET.target_transform == "lce":
        return 100.0 * (1.0 - np.power(10.0, -np.asarray(z, float)))
    return np.asarray(z, float)
