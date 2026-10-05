"""R53: BGE-base vs MiniLM on the REAL doc store (hybrid .7), real_mrr / real_h1 for seeds 31-90 (queries = row texts t[0][0]); also adaptive alpha."""
import sys; sys.path.insert(0, "../")
from lab import *
from lib import *
import pickle, f8_embed
W = RealWorld("A"); INST = W.inst; Q = W.queries
BGE = f8_embed.Embedder(f8_embed.BGE, "cls", 768)
CP = os.path.join(RES, "r53_bge.pkl")
cache = pickle.load(open(CP, "rb")) if os.path.exists(CP) else {}
def ensure(tids):
    todo = [t for t in tids if t not in cache]
    for i in range(0, len(todo), 128):
        vs = BGE.batch([W.texts[t] for t in todo[i:i + 128]])
        for t, v in zip(todo[i:i + 128], vs): cache[t] = v
    if todo: pickle.dump(cache, open(CP, "wb"))
def run(vec_of, dim, alpha=0.7, nq=120, seeds=range(31, 91)):
    V = np.stack([vec_of(d) for d in range(W.ndoc)]); ret = Retriever(W.docs, V, "hybrid", alpha); out = {}
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        rr, h1 = [], []
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; rel = set(Q[row["q"]]["rel"]); sc = ret.scores(W.texts[tid], vec_of(tid)); best = min(int((sc > sc[d]).sum()) for d in rel)
            rr.append(1.0 / (1 + best)); h1.append(float(best == 0))
        out[s] = {"real_mrr": float(np.mean(rr)), "real_h1": float(np.mean(h1))}
    return out
if __name__ == "__main__":
    qt = {row["t"][0][0] for s in range(31, 91) if str(s) in INST for row in INST[str(s)]["rows"][:120]}
    t0 = time.time(); ensure(list(range(W.ndoc)) + sorted(qt)); print("bge embedded", len(cache), round(time.time() - t0), "s")
    res = {"mini_a.7": run(lambda t: W.emb[t], 384), "bge_a.7": run(lambda t: cache[t], 768)}
    for a in (0.3, 0.5, 0.9, 1.0): res[f"bge_a{a}"] = run(lambda t: cache[t], 768, alpha=a); res[f"mini_a{a}"] = run(lambda t: W.emb[t], 384, alpha=a)
    pickle.dump(res, open(f"{RES}/r53_res.pkl", "wb"))
    for k, r in res.items(): print(f"{k:10s} HELD {np.mean([r[s]['real_mrr'] for s in HELD if s in r]):.4f} CONF {np.mean([r[s]['real_mrr'] for s in CONF if s in r]):.4f}")
