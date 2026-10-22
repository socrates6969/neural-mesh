"""Paired statistics for held-out comparisons (no scipy needed)."""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p. b = base right & pack wrong, c = base wrong & pack right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(base: list[bool], treat: list[bool]) -> dict:
    n = len(base)
    b = sum(1 for x, y in zip(base, treat) if x and not y)
    c = sum(1 for x, y in zip(base, treat) if (not x) and y)
    kb, kt = sum(base), sum(treat)
    # Wilson-style bound on the paired difference via the discordant pairs (conservative)
    diff = (kt - kb) / n if n else 0.0
    lo_c, _ = wilson(c, b + c) if b + c else (0.0, 0.0)
    return {"n": n, "base": kb, "pack": kt, "base_rate": kb / n if n else 0, "pack_rate": kt / n if n else 0,
            "diff": diff, "b_base_only": b, "c_pack_only": c, "p": mcnemar_exact(b, c),
            "discordant_pack_share_lo": lo_c}
