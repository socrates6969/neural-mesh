"""k3_stats: document-level (file-level) bootstrap helpers + Holm. Pure python, deterministic (seeded)."""
import random, math


def f1_from(c):
    tp, fp, fn = c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def boot_ratio(doc_vals, stat, B=1000, seed=7):
    """doc_vals: dict doc -> vector (list of numbers). stat: function(sum_vector)->float. returns (point, lo, hi)."""
    docs = sorted(doc_vals)
    if not docs:
        return (float('nan'),) * 3
    n = len(docs)
    dim = len(doc_vals[docs[0]])
    tot = [sum(doc_vals[d][i] for d in docs) for i in range(dim)]
    pt = stat(tot)
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        s = [0.0] * dim
        for _ in range(n):
            v = doc_vals[docs[rng.randrange(n)]]
            for i in range(dim):
                s[i] += v[i]
        vals.append(stat(s))
    vals.sort()
    return pt, vals[int(0.025 * B)], vals[int(0.975 * B) - 1]


def paired_boot(doc_a, doc_b, B=2000, seed=11):
    """doc_a/doc_b: dict doc -> (sum_metric, n). mean diff a-b with bootstrap over docs; returns (diff, lo, hi, p_two_sided)"""
    docs = sorted(set(doc_a) & set(doc_b))
    n = len(docs)
    if n == 0:
        return (float('nan'),) * 4
    sa = [doc_a[d] for d in docs]
    sb = [doc_b[d] for d in docs]
    tot = lambda idx: (sum(sa[i][0] for i in idx) / max(1, sum(sa[i][1] for i in idx))) - (sum(sb[i][0] for i in idx) / max(1, sum(sb[i][1] for i in idx)))
    pt = tot(range(n))
    rng = random.Random(seed)
    vals = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(tot(idx))
    vals.sort()
    lo, hi = vals[int(0.025 * B)], vals[int(0.975 * B) - 1]
    # two-sided p from the normal approximation to the bootstrap distribution (resolution is not limited by B, so a global Holm is usable)
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    z = abs(pt) / sd if sd > 0 else (float('inf') if pt != 0 else 0.0)
    p = math.erfc(z / math.sqrt(2)) if z != float('inf') else 0.0
    return pt, lo, hi, p


def holm(ps):
    """ps: list of p. returns adjusted list (same order)"""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * ps[i]
        run = max(run, v)
        adj[i] = min(1.0, run)
    return adj
