"""Paired statistics for the promotion gate (stdlib only)."""
from __future__ import annotations

import math
import random

# two-sided 97.5% t quantiles by df (df > 30 -> normal approximation)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069,
         24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def paired_ci(diffs: list[float]) -> dict:
    """mean of paired differences with a 95% t-CI. n < 2 -> CI is infinite (never significant)."""
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean": 0.0, "lo": -math.inf, "hi": math.inf}
    m = sum(diffs) / n
    if n < 2:
        return {"n": n, "mean": m, "lo": -math.inf, "hi": math.inf}
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    t = _T975.get(n - 1, 1.96)
    h = t * sd / math.sqrt(n)
    return {"n": n, "mean": m, "lo": m - h, "hi": m + h}


def cluster_bootstrap_ci(groups: list[list[float]], reps: int = 4000, seed: int = 0) -> dict:
    """95% percentile bootstrap CI of the mean paired difference, resampling whole sessions (examples within one session
    are not independent)."""
    flat = [d for g in groups for d in g]
    if not flat or len(groups) < 2:
        return {"n": len(flat), "clusters": len(groups), "mean": (sum(flat) / len(flat)) if flat else 0.0, "lo": -math.inf, "hi": math.inf}
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        s = [groups[rng.randrange(len(groups))] for _ in groups]
        xs = [d for g in s for d in g]
        means.append(sum(xs) / len(xs))
    means.sort()
    return {"n": len(flat), "clusters": len(groups), "mean": sum(flat) / len(flat), "lo": means[int(0.025 * reps)], "hi": means[int(0.975 * reps) - 1]}
