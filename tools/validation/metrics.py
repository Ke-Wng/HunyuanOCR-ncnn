"""Numeric comparison metrics shared by vision replay tests."""

from __future__ import annotations

import numpy as np


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    """Return absolute-error, cosine, and relative-L2 metrics."""

    diff = actual.astype(np.float64) - expected.astype(np.float64)
    actual64 = actual.astype(np.float64).reshape(-1)
    expected64 = expected.astype(np.float64).reshape(-1)
    denom = np.linalg.norm(actual64) * np.linalg.norm(expected64)
    expected_norm = np.linalg.norm(expected64)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "cosine": float(np.dot(actual64, expected64) / denom) if denom else 1.0,
        "relative_l2": float(np.linalg.norm(diff.reshape(-1)) / expected_norm)
        if expected_norm
        else 0.0,
    }
