"""R43: token-cost accounting for retrieved context on the REAL store. Budget B tokens (tokens ~ 1.33*words). hit@B = a relevant doc is fully inside the packed context.
Packers over the hybrid top-20: plain (rank order, STOP at first doc that does not fit), skip (rank order, skip non-fitting docs), trunc (truncate every doc to T=B/3 tokens, then skip-pack), density (sort by z-score/tokens, skip-pack)."""
from lab import *
from lib import *
W = RealWorld("A"); INST = W.inst; Q = W.queries
TOK = np.array([max(1, int(1.33 * len(t.split()))) for t in W.docs])
ret = W.retriever("hybrid", 0.7)
def pack(order, budget, kind, sc=None):
    used, got = 0, []
    if kind == "density": order = sorted(order, key=lambda d: -(sc[d] / np.sqrt(TOK[d])))
    for d in order:
        c = TOK[d] if kind != "trunc" else min(TOK[d], max(1, budget // 3))
        if used + c <= budget: got.append(d); used += c
        elif kind == "plain": break
    return got, used
def run(seeds, B, kind, nq=120):
    out = {}
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        hit, tok = [], []
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; sc = ret.scores(W.texts[tid], W.emb[tid]); rel = set(Q[row["q"]]["rel"]); top = list(np.argsort(-sc)[:20])
            got, used = pack(top, B, kind, z(sc))
            if kind == "trunc":  # a truncated doc counts only if it was not cut (cut docs may lose the answer)
                got = [d for d in got if TOK[d] <= max(1, B // 3)]
            hit.append(float(any(d in rel for d in got))); tok.append(used)
        out[s] = {"ctx_hit": float(np.mean(hit)), "ctx_tokens": float(np.mean(tok))}
    return out
if __name__ == "__main__":
    import pickle
    print("doc tokens: mean %.0f median %.0f p90 %.0f" % (TOK.mean(), np.median(TOK), np.percentile(TOK, 90)))
    seeds = list(range(31, 91)); res = {}
    for B in (150, 300, 600, 1200):
        for kind in ("plain", "skip", "trunc", "density"):
            r = run(seeds, B, kind); res[(B, kind)] = r
            m = lambda sd, k: float(np.mean([r[s][k] for s in sd if s in r]))
            print(f"B={B:5d} {kind:8s} HELD hit {m(HELD,'ctx_hit'):.3f} CONF {m(CONF,'ctx_hit'):.3f}  tokens used {m(HELD,'ctx_tokens'):.0f}")
    pickle.dump(res, open(f"{RES}/r43_res.pkl", "wb"))
