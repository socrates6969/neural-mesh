from r71_explore import *
LT = np.log(TOK)
def run(lam, k, seeds=range(1, 9), nq=120):
    h, t = [], []
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; sc = z(ret.scores(W.texts[tid], W.emb[tid])); rel = set(Q[row["q"]]["rel"])
            adj = sc - lam * (LT - LT.mean()); top = np.argsort(-adj)[:k]
            h.append(float(any(d in rel for d in top))); t.append(TOK[top].sum())
    return np.mean(h), np.mean(t)
for lam in (0, .1, .2, .3, .5):
    print(lam, " | ".join("k%d hit %.3f tok %.0f" % ((k,) + run(lam, k)) for k in (3, 5)))
