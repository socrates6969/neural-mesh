from lab import *
from lib import *
import pickle
W = RealWorld("A"); INST = W.inst; Q = W.queries
TOK = np.array([max(1, int(1.33 * len(t.split()))) for t in W.docs])
ret = W.retriever("hybrid", 0.7)
def data(seeds, nq=120):
    out = {}
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        rows = []
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; sc = ret.scores(W.texts[tid], W.emb[tid]); rel = set(Q[row["q"]]["rel"]); top = np.argsort(-sc)[:10]; zs = z(sc)[top]
            rows.append((top, zs, rel))
        out[s] = rows
    return out
def ctx(top, zs, rel, pol):
    n = pol(zs); g = top[:n]; return float(any(d in rel for d in g)), int(TOK[g].sum())
def fixed(k): return lambda zs: k
def gap(g, kmax=5): return lambda zs: max(1, min(kmax, int((zs >= zs[0] - g).sum())))
def ratio(r, kmax=5): return lambda zs: max(1, min(kmax, int((zs >= r * zs[0]).sum())))
if __name__ == "__main__":
    D = data(range(1, 9))
    pols = {f"top{k}": fixed(k) for k in (1, 2, 3, 4, 5, 8)}
    for g in (0.5, 1, 1.5, 2, 3): pols[f"gap{g}"] = gap(g, 8)
    for g in (0.5, 1, 1.5, 2): pols[f"gap{g}k5"] = gap(g, 5)
    for r in (.5, .6, .7, .8): pols[f"ratio{r}"] = ratio(r, 8)
    for n, p in pols.items():
        res = [ctx(*row, p) for s in D for row in D[s]]
        print(f"{n:10s} hit {np.mean([r[0] for r in res]):.3f} tokens {np.mean([r[1] for r in res]):.0f}")
