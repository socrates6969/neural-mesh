"""R107: tie re-count of the older quant rounds (R9 int8, R11 int4, R12 binary sign, R13/R14 PCA(+int8)) under TP=1 (exact top ties = miss; rank = #docs >= relevant minus self).
Baseline (f32 hybrid .7) is RE-COMPUTED under the same rule (the cached b1_baseline.json predates it). Usage: TP=1 python r107.py"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
from rounds import *
from eff import *
import eff
def run(name, nbytes, make=None, E=None):
    [lib.TIE.__setitem__(k, 0) for k in lib.TIE]
    rows, loss = eff.eff_round(name, nbytes, make=make, E=E)
    print(f"   TIE COUNT {name}: {lib.TIE['ties']} tied-top decisions of {lib.TIE['n']} argmax/argmin calls; relevant-doc rank ties {lib.TIE['rank_ties']} of {lib.TIE['rank_n']} rank evaluations (TP={int(lib.TP)})", flush=True)
    return rows, loss
if __name__ == "__main__":
    t0 = time.time(); [lib.TIE.__setitem__(k, 0) for k in lib.TIE]
    eff.BASE = base_tab_fresh = run_battery(D, ALLS, with_exec=False, real=True)
    print(f"BASE f32 hybrid recomputed under TP={int(lib.TP)} ({time.time()-t0:.0f}s); ties {lib.TIE}", flush=True)
    only = sys.argv[1:]
    cands = [("R9 int8", 388, mkq(q8, q8), lambda t: q8(D.v(t))),
             ("R10 f16", 768, mkq(f16, f16), lambda t: f16(D.v(t))),
             ("R11 int4", 196, mkq(lambda X: qk(X, 4), lambda X: qk(X, 4)), lambda t: qk(D.v(t), 4)),
             ("R12 binary sign", 48, mkq(sign, sign), lambda t: sign(D.v(t))),
             ("R14a PCA192+int8", 196, pca_q(192, q8), None),
             ("R14b PCA256+int8", 260, pca_q(256, q8), None),
             ("R14c PCA128+int8", 132, pca_q(128, q8), None),
             ("R13 PCA192 f32", 768, pca_tf(192), None)]
    for nm, b, mk, E in cands:
        if only and nm.split()[0] not in only: continue
        run(nm + f" [TP={int(lib.TP)}]", b, make=mk, E=E)
