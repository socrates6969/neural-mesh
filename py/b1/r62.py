"""R62: two-embedder ensemble on the REAL store: z(MiniLM dense) and z(BGE dense) averaged inside the hybrid (BM25 default and k1=.9,b=.5)."""
import sys; sys.path.insert(0, "../")
from r53 import *
class Ens(Retriever):
    def __init__(self, docs, vm, vb, alpha, wb, k1=1.2, b=0.75):
        self.docs, self.vm, self.vb, self.alpha, self.wb = docs, vm, vb, alpha, wb; self.bm = BM25(docs, k1, b)
    def scores(self, qt, qm, qb):
        d = (1 - self.wb) * z(self.vm @ qm) + self.wb * z(self.vb @ qb)
        return self.alpha * d + (1 - self.alpha) * z(self.bm.score(qt))
def runE(wb, alpha=0.7, k1=1.2, b=0.75, nq=120, seeds=range(31, 91)):
    VM = W.dvec; VB = np.stack([cache[d] for d in range(W.ndoc)]); ret = Ens(W.docs, VM, VB, alpha, wb, k1, b); out = {}
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        rr, h1 = [], []
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; rel = set(Q[row["q"]]["rel"]); sc = ret.scores(W.texts[tid], W.emb[tid], cache[tid]); best = min(int((sc > sc[d]).sum()) for d in rel)
            rr.append(1.0 / (1 + best)); h1.append(float(best == 0))
        out[s] = {"real_mrr": float(np.mean(rr)), "real_h1": float(np.mean(h1))}
    return out
if __name__ == "__main__":
    import pickle
    from v2 import gate
    base = runE(0.0)
    res = {}
    tab = lambda r: {("R", s): v for s, v in r.items()}
    for wb in (0.5,):
        for (k1, b) in ((1.2, 0.75), (0.9, 0.5)):
            c = runE(wb, k1=k1, b=b)
            ref = base if (k1, b) == (1.2, 0.75) else runE(0.0, k1=k1, b=b)
            gate(tab(ref), tab(c), ["real_mrr", "real_h1"], name=f"R62 MiniLM+BGE dense ensemble (w_bge={wb}) BM25 k1={k1} b={b} vs MiniLM-only same BM25")
            print("   means HELD/CONF ensemble", round(np.mean([c[s]["real_mrr"] for s in HELD]), 4), round(np.mean([c[s]["real_mrr"] for s in CONF]), 4), "| MiniLM-only", round(np.mean([ref[s]["real_mrr"] for s in HELD]), 4), round(np.mean([ref[s]["real_mrr"] for s in CONF]), 4))
