"""vector-lab: does a Poincare-distance ranking (ruvector attention module math) beat cosine on the b1 battery retrieval axes?
Dense arm only. Baseline = cosine. Variant = -d_poincare(s*u, s*v), unit vectors scaled by s (curvature 1). Paired over seeds.
Run from neural-mesh/py/b1:  python ../vec_lab/hyp_battery.py"""
import sys, json, numpy as np
sys.path.insert(0, "../b1")
from lib import *
from battery import run_battery

class HypRetriever:
    def __init__(self, docs, vecs, s):
        self.s = s; self.V = vecs * s; self.n2 = (self.V ** 2).sum(1)
    def scores(self, qtext, qvec):
        q = qvec * self.s; qn = float(q @ q)
        sq = self.n2 + qn - 2 * (self.V @ q)
        arg = 1 + 2 * sq / ((1 - self.n2) * (1 - qn))
        return -np.arccosh(np.maximum(arg, 1.0))
def mk(s):
    return lambda docs, vecs, arm, alpha: HypRetriever(docs, vecs, s)
if __name__ == "__main__":
    D = Data(); AX = ["general_info", "comprehension", "domain_r1", "domain_mrr", "domain_route"]
    def run(make, seeds):
        tab = run_battery(D, seeds, arm="dense", make=make, with_exec=False)
        return {k: [np.mean([tab[(w, s)][k] for w in WORLDS]) for s in seeds] for k in AX}
    out = {}
    for name, seeds in (("TUNE", TUNE), ("HELD", HELD)):
        base = run(None, seeds)
        for s in (0.5, 0.9):
            v = run(mk(s), seeds)
            for k in AX:
                p = paired(v[k], base[k]); out[f"{name}/s={s}/{k}"] = p
                print(name, s, k, f"{np.mean(base[k]):.3f} -> {np.mean(v[k]):.3f} d={p['mean']:+.4f} CI[{p['lo']:+.4f},{p['hi']:+.4f}] n={p['n']} w/l={p['wins']}/{p['losses']}", flush=True)
    json.dump(out, open("../../results/vec_hyp_battery.json", "w"))
