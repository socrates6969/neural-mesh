"""R47: shared-store corpus GC by near-duplicate removal on the REAL store (6,831 paragraphs). A forgotten doc is mapped to its kept representative (content retained);
a hit on the representative counts for a query whose relevant doc was merged into it. Also reports space saved and exact-duplicate counts."""
from lab import *
from lib import *
W = RealWorld("A"); INST = W.inst; Q = W.queries
Vn = W.dvec / np.linalg.norm(W.dvec, axis=1, keepdims=True)
S = Vn @ Vn.T; np.fill_diagonal(S, -1)
def gc_map(th):
    rep = list(range(W.ndoc)); keep = np.ones(W.ndoc, bool)
    for i in range(W.ndoc):           # keep the lowest index of a duplicate cluster (greedy, deterministic)
        if not keep[i]: continue
        for j in np.where(S[i] > th)[0]:
            if j > i and keep[j]: keep[j] = False; rep[j] = i
    return np.array(rep), keep
def evaluate(rep, keep, seeds, nq=120):
    kept = np.where(keep)[0]; idx = {d: i for i, d in enumerate(kept)}
    docs = [W.docs[d] for d in kept]; ret = Retriever(docs, W.dvec[kept], "hybrid", 0.7); out = {}
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        rr, h1 = [], []
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; sc = ret.scores(W.texts[tid], W.emb[tid]); rel = {idx[rep[d]] for d in Q[row["q"]]["rel"]}
            best = min(int((sc > sc[d]).sum()) for d in rel); rr.append(1.0 / (1 + best)); h1.append(float(best == 0))
        out[s] = {"real_mrr": float(np.mean(rr)), "real_h1": float(np.mean(h1))}
    return out
if __name__ == "__main__":
    import pickle
    seeds = list(range(31, 91)); res = {"full": evaluate(np.arange(W.ndoc), np.ones(W.ndoc, bool), seeds)}
    for th in (0.995, 0.98, 0.95, 0.9):
        rep, keep = gc_map(th); res[th] = evaluate(rep, keep, seeds)
        m = lambda r, sd: float(np.mean([r[s]["real_mrr"] for s in sd if s in r]))
        print(f"th={th}: removed {int((~keep).sum())}/{W.ndoc} ({100*(~keep).mean():.1f}%) | mrr HELD {m(res[th],HELD):.4f} vs full {m(res['full'],HELD):.4f} | CONF {m(res[th],CONF):.4f} vs {m(res['full'],CONF):.4f}")
    pickle.dump(res, open(f"{RES}/r47_res.pkl", "wb"))
