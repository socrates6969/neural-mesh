"""B3: QUERY-DISJOINT re-evaluation of the real-text retrieval profile (validity fix).
Finding that motivates it: in the h10 real axis, seeds 31-60 (HELD) and 61-90 (CONF) draw from the SAME query pool (|HELD q| ~ |CONF q| ~ 1550, overlap ~1549),
so 'confirm' was a re-sample of the same items, not an independent set. Only TUNE (seeds 1-8) is query-disjoint from them.
Here: all queries of seeds 31-90 are pooled (n~1566), split ONCE into two query-DISJOINT halves (fixed rng seed 20260605) = HALF1 / HALF2, plus the full pool;
per-query reciprocal rank for each config; paired tests are PER QUERY (Wilcoxon signed-rank + exact sign test), Holm over the family. Pre-declared configs:
 default  = hybrid alpha .7, BM25 (1.2,.75)          bm25p    = default + BM25 (.9,.5)
 adaptive = adaptive alpha (.3,1,2), BM25 (1.2,.75)   profile  = adaptive + BM25 (.9,.5)  (R63 recommended)
 dense / alpha.9 = references.
"""
import sys, json
from lib import *
from r02 import Adaptive
import numpy as np
from scipy import stats

W = RealWorld("A")
Q = W.queries


def qset(seeds):
    out = {}
    for s in seeds:
        v = W.inst.get(str(s))
        if v is None: continue
        for r in v["rows"]:
            out.setdefault(r["q"], r["t"][0][0])
    return out


class Fixed:
    def __init__(self, alpha, k1=1.2, b=0.75):
        self.vecs, self.bm, self.a = W.dvec, BM25(W.docs, k1, b), alpha
    def scores(self, qt, qv):
        return self.a * z(self.vecs @ qv) + (1 - self.a) * z(self.bm.score(qt))


class AdaptiveBM(Adaptive):
    def __init__(self, k1, b):
        super().__init__(W.docs, W.dvec, "hybrid", .7, .3, 1.0, 2.0)
        self.bm = BM25(W.docs, k1, b)


def rr_all(ret, qs):
    out = {}
    for q, tid in qs.items():
        rel = set(Q[q]["rel"]); sc = ret.scores(W.texts[tid], W.emb[tid])
        best = min(int((sc >= sc[d]).sum()) - 1 for d in rel)
        out[q] = 1.0 / (1 + best)
    return out


if __name__ == "__main__":
    tune_q = qset(range(1, 9)); pool = qset(range(31, 91))
    rng = np.random.RandomState(20260605)
    ids = sorted(pool); rng.shuffle(ids)
    h1, h2 = set(ids[: len(ids) // 2]), set(ids[len(ids) // 2:])
    print("tune", len(tune_q), "pool", len(pool), "half1", len(h1), "half2", len(h2), "tune&pool", len(set(tune_q) & set(pool)))
    cfgs = {
        "default": Fixed(.7), "bm25p": Fixed(.7, .9, .5), "adaptive": Adaptive(W.docs, W.dvec, "hybrid", .7, .3, 1.0, 2.0),
        "profile": AdaptiveBM(.9, .5), "alpha.9": Fixed(.9), "dense": Fixed(1.0),
    }
    allq = dict(pool); allq.update(tune_q)
    R = {k: rr_all(r, allq) for k, r in cfgs.items()}
    json.dump({k: {str(q): v for q, v in d.items()} for k, d in R.items()}, open(f"{RES}/b3_rr.json", "w"))
    sets = {"TUNE": set(tune_q), "HALF1": h1, "HALF2": h2, "POOL": set(pool)}
    for sn, qs in sets.items():
        qs = sorted(qs)
        print(sn, len(qs), {k: round(float(np.mean([R[k][q] for q in qs])), 4) for k in R})
    fam = []
    for ref, cand in [("default", "bm25p"), ("default", "adaptive"), ("default", "profile"), ("adaptive", "profile"), ("default", "alpha.9"), ("default", "dense")]:
        for sn in ("TUNE", "HALF1", "HALF2", "POOL"):
            qs = sorted(sets[sn]); d = np.array([R[cand][q] - R[ref][q] for q in qs])
            nz = d[d != 0]; w = int((nz > 0).sum()); l = int((nz < 0).sum())
            p_sign = stats.binomtest(w, w + l, .5).pvalue if w + l else 1.0
            p_w = stats.wilcoxon(nz).pvalue if len(nz) > 5 else 1.0
            bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(2000)]
            fam.append((f"{cand}-{ref}", sn, float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), w, l, p_sign, p_w))
    adj = holm([f[8] for f in fam]) if "holm" in globals() else None
    if adj is None:
        from lib import holm_adjust as holm
        adj = holm([f[8] for f in fam])
    for f, a in zip(fam, adj):
        print(f"{f[0]:20} {f[1]:6} d {f[2]:+.4f} [{f[3]:+.4f},{f[4]:+.4f}] W/L {f[5]}/{f[6]} sign p {f[7]:.2e} wilcoxon p {f[8]:.2e} Holm(wilcoxon) {a:.2e}")
