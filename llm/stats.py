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


def _boot_means(groups: list[list[float]], reps: int, seed: int) -> list[float]:
    """sorted means of `reps` cluster-bootstrap resamples (whole groups drawn with replacement; empty groups are skipped by the
    callers). Shared by the CI and the p-values below so they agree with each other."""
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        s = [groups[rng.randrange(len(groups))] for _ in groups]
        xs = [d for g in s for d in g]
        means.append(sum(xs) / len(xs) if xs else 0.0)
    means.sort()
    return means


def cluster_bootstrap_ci(groups: list[list[float]], reps: int = 4000, seed: int = 0) -> dict:
    """95% percentile bootstrap CI of the mean paired difference, resampling whole sessions (examples within one session
    are not independent)."""
    flat = [d for g in groups for d in g]
    if not flat or len(groups) < 2:
        return {"n": len(flat), "clusters": len(groups), "mean": (sum(flat) / len(flat)) if flat else 0.0, "lo": -math.inf, "hi": math.inf}
    means = _boot_means(groups, reps, seed)
    return {"n": len(flat), "clusters": len(groups), "mean": sum(flat) / len(flat), "lo": means[int(0.025 * reps)], "hi": means[int(0.975 * reps) - 1]}


def sign_test_p(wins: int, losses: int) -> float:
    """one-sided exact sign test: P(X >= wins) for X ~ Binomial(wins + losses, 0.5). Ties are dropped by the caller.
    No decisive pairs -> 1.0 (never significant)."""
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone, capped at 1), in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, run = [1.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * pvals[i]))
        adj[i] = run
    return adj


def cluster_bootstrap_p(groups: list[list[float]], alternative: str = "greater", null: float = 0.0, reps: int = 4000, seed: int = 0) -> float:
    """Cluster-bootstrap p-value for the mean of the pooled values vs `null`. alternative "greater" (H1: mean > null), "less"
    (H1: mean < null) or "two-sided". p = (1 + #resample means on the null side) / (reps + 1). Fewer than 2 non-empty clusters -> 1.0."""
    groups = [g for g in groups if g]
    if len(groups) < 2:
        return 1.0
    means = _boot_means(groups, reps, seed)
    le = sum(1 for x in means if x <= null)
    ge = sum(1 for x in means if x >= null)
    p_gt, p_lt = (1 + le) / (reps + 1), (1 + ge) / (reps + 1)
    if alternative == "greater":
        return p_gt
    if alternative == "less":
        return p_lt
    if alternative == "two-sided":
        return min(1.0, 2 * min(p_gt, p_lt))
    raise ValueError(f"alternative must be greater, less or two-sided, not {alternative!r}")


def tost_cluster(groups: list[list[float]], eps: float = 0.01, reps: int = 4000, seed: int = 0) -> dict:
    """Two one-sided tests of equivalence within +-eps on the cluster-bootstrapped mean: p = max(p(mean > -eps), p(mean < +eps)).
    `equivalent` at alpha .05 (callers Holm-adjust `p` with their other tests)."""
    flat = [d for g in groups for d in g]
    p_lo = cluster_bootstrap_p(groups, "greater", -eps, reps, seed)
    p_hi = cluster_bootstrap_p(groups, "less", eps, reps, seed)
    p = max(p_lo, p_hi)
    return {"n": len(flat), "clusters": sum(1 for g in groups if g), "mean": (sum(flat) / len(flat)) if flat else 0.0, "eps": eps,
            "p_lower": p_lo, "p_upper": p_hi, "p": p, "equivalent": p < 0.05}


def wilson_upper(k: int, n: int, z: float = 1.959964) -> float:
    """upper end of the Wilson score interval for k/n (95% two-sided by default); n == 0 -> 1.0 (nothing is known)."""
    if n <= 0:
        return 1.0
    ph = k / n
    den = 1 + z * z / n
    c = ph + z * z / (2 * n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return min(1.0, (c + h) / den)


# ---- exact paired tests, power and minimum detectable effect (E2) ---------------------------------------------------------------------

def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p: the discordant pairs b (A right, B wrong) and c (B right, A wrong) are Binomial(b + c, .5) under H0.
    No discordant pair -> 1.0. Same value as ingest.shadow.mcnemar_exact_p."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _upper_tail(n: int, k: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return min(1.0, sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def critical_wins(m: int, alpha: float = 0.05) -> int | None:
    """smallest k with P(Binomial(m, .5) >= k) <= alpha (one-sided exact sign test); None when even k = m is not significant (m < 5 at .05)."""
    for k in range(m // 2 + 1, m + 1):
        if sum(math.comb(m, i) for i in range(k, m + 1)) / 2 ** m <= alpha:
            return k
    return None


def sign_power(m: int, p: float, alpha: float = 0.05) -> float:
    """exact power of the one-sided sign test with m decisive pairs when each decisive pair is a win with probability p."""
    k = critical_wins(m, alpha)
    return 0.0 if k is None else _upper_tail(m, k, p)


def mde_sign_test(m: int, power: float = 0.8, alpha: float = 0.05) -> float | None:
    """minimum detectable WIN SHARE among m decisive pairs: the smallest p whose exact power is >= `power` (bisection; power is monotone in p).
    None when the test cannot reach significance at all with m pairs, or cannot reach `power` even at p = 1."""
    if critical_wins(m, alpha) is None or sign_power(m, 1.0, alpha) < power:
        return None
    lo, hi = 0.5, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if sign_power(m, mid, alpha) >= power else (mid, hi)
    return hi


def mde_paired(n: int, discordance: float, power: float = 0.8, alpha: float = 0.05) -> dict:
    """minimum detectable paired accuracy difference with n items when a fraction `discordance` of them is discordant (exactly one of the
    two systems right): m = round(n * discordance) decisive pairs, win share p* = mde_sign_test(m), diff = discordance * (2 p* - 1)."""
    m = int(round(n * discordance))
    p = mde_sign_test(m, power, alpha)
    return {"n": n, "discordance": discordance, "decisive": m, "win_share": p, "min_diff": None if p is None else discordance * (2 * p - 1)}


def mde_observed(n: int, b: int, c: int, power: float = 0.8, alpha: float = 0.05) -> dict:
    """MDE for a comparison that produced b + c discordant pairs out of n items (the observed discordance is the planning value)."""
    return mde_paired(n, (b + c) / n if n else 0.0, power, alpha)


def recall_by_class(y_true: list, y_pred: list) -> dict:
    """per-class recall {class: {"n": support, "k": hits, "recall": k/n}} over the classes present in y_true."""
    out: dict = {}
    for t, p in zip(y_true, y_pred):
        d = out.setdefault(t, {"n": 0, "k": 0})
        d["n"] += 1
        d["k"] += int(t == p)
    for d in out.values():
        d["recall"] = d["k"] / d["n"]
    return out


def balanced_accuracy(y_true: list, y_pred: list) -> float | None:
    """macro-average of the per-class recalls over the classes present in y_true (a constant majority predictor scores 1 / n_classes)."""
    r = recall_by_class(y_true, y_pred)
    return sum(d["recall"] for d in r.values()) / len(r) if r else None


def required_decisive(win_share: float, power: float = 0.8, alpha: float = 0.05, max_m: int = 1500) -> int | None:
    """smallest number of decisive pairs m for which the exact one-sided sign test has >= `power` when the true win share is `win_share`
    (scans m upward: exact power is a sawtooth in m, so the first m that reaches it is reported). None if not reached by max_m."""
    for m in range(5, max_m + 1):
        if sign_power(m, win_share, alpha) >= power:
            return m
    return None
