"""R54: BM25 parameters (k1, b) inside the hybrid on the REAL store (+ domain world A/B via battery axes). TUNE seeds 1-8 pick; verify 31-60 / 61-90."""
from lab import *
from lib import *
import pickle
W = RealWorld("A"); INST = W.inst; Q = W.queries
class R2(Retriever):
    def __init__(self, docs, vecs, arm, alpha, k1, b):
        super().__init__(docs, vecs, arm, alpha); self.bm = BM25(docs, k1, b)
def real(k1, b, seeds, nq=120):
    ret = R2(W.docs, W.dvec, "hybrid", 0.7, k1, b); out = {}
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        rr, h1 = [], []
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; rel = set(Q[row["q"]]["rel"]); sc = ret.scores(W.texts[tid], W.emb[tid]); best = min(int((sc > sc[d]).sum()) for d in rel)
            rr.append(1.0 / (1 + best)); h1.append(float(best == 0))
        out[s] = {"real_mrr": float(np.mean(rr)), "real_h1": float(np.mean(h1))}
    return out
if __name__ == "__main__":
    res = {}
    tune = [s for s in range(1, 9)]
    for k1 in (0.6, 0.9, 1.2, 1.6, 2.0):
        for b in (0.3, 0.5, 0.75, 0.9):
            r = real(k1, b, tune); res[(k1, b)] = r
            print(f"k1={k1} b={b}: TUNE real_mrr {np.mean([r[s]['real_mrr'] for s in tune if s in r]):.4f}")
    pickle.dump(res, open(f"{RES}/r54_tune.pkl", "wb"))
