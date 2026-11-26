"""Paired comparison of two policies over seeds.

Every seed replays the same jobs under both policies, so the per-seed ratio B/A removes most of the
seed-to-seed variation (a busy stretch of the trace is busy for both). The reported effect is the mean of
the per-seed relative changes, with a percentile bootstrap interval over seeds. The absolute difference
B - A is reported the same way, and is the only one given when A is zero for some seed.
"""

from __future__ import annotations

import numpy as np


def bootstrap_mean(values: np.ndarray, resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% percentile-bootstrap interval of the mean."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    low, high = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def paired_change(a: list[float], b: list[float]) -> dict:
    """Change from a to b per seed (negative = b is lower): relative (b/a - 1) and absolute (b - a)."""
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = b_arr - a_arr
    out: dict = {"seeds": len(a_arr), "b_lower_in": int((diff < 0).sum()), "b_higher_in": int((diff > 0).sum())}
    out["abs_mean"], out["abs_low"], out["abs_high"] = bootstrap_mean(diff)
    if np.all(a_arr > 0):
        rel = b_arr / a_arr - 1.0
        out["mean"], out["ci_low"], out["ci_high"] = bootstrap_mean(rel)
        out["per_seed"] = [round(float(x), 5) for x in rel]
    else:
        out["mean"] = out["ci_low"] = out["ci_high"] = None
        out["per_seed"] = None
    return out
