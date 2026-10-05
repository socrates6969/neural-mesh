"""R40: MMR re-ranking for 'bridge' (two-topic) queries. both@5 = both gold docs in top-5; any@5; cov = distinct domains in top-5 / min(5, 2) capped.
Baseline = hybrid(.7) top-5. Candidates: MMR(lambda) over z-fused hybrid scores with dense doc-doc cosine."""
from lab import *
from lib import *
import random
def bridge_axes(world, seed, mmr_lams=(None,), alpha=0.7, nq=24):
    r = random.Random(seed * 9973 + (1 if world == "B" else 0)); dd = D.stores["domain"]; n = len(dd["docs"])
    keep = sorted(r.sample(range(n), int(.8 * n))); idx = {d: i for i, d in enumerate(keep)}
    docs = [dd["docs"][d] for d in keep]; vecs = np.stack([D.v(d) for d in docs]); dom = [dd["dom"][d] for d in keep]
    ret = Retriever(docs, vecs, "hybrid", alpha); qs = {d: (a, b) for (d, _, a, b) in dd["qs"]}
    cand = [d for d in keep if d in qs]
    out = {lam: dict(both=[], anyh=[], cov=[]) for lam in mmr_lams}
    G = vecs @ vecs.T
    for _ in range(nq):
        while True:
            a, b = r.sample(cand, 2)
            if dd["dom"][a] != dd["dom"][b]: break
        qa, qb = qs[a], qs[b]
        q = (qa[0] + " and " + qb[0]) if world == "A" else (typo(r, qa[1], .12) + " and " + typo(r, qb[1], .12))
        qv = D.v(q); sc = ret.scores(q, qv); ia, ib = idx[a], idx[b]
        for lam in mmr_lams:
            if lam is None: top = list(np.argsort(-sc)[:5])
            else:
                zs = z(sc); sel = []; rest = list(np.argsort(-sc)[:40])
                while len(sel) < 5 and rest:
                    best = max(rest, key=lambda i: lam * zs[i] - (1 - lam) * (max(G[i, j] for j in sel) * 4 if sel else 0.0))
                    sel.append(best); rest.remove(best)
                top = sel
            s = set(int(t) for t in top); out[lam]["both"].append(ia in s and ib in s); out[lam]["anyh"].append(ia in s or ib in s)
            out[lam]["cov"].append(len({dom[t] for t in top}) / 5.0)
    return {lam: {"bridge_both5": float(np.mean(v["both"])), "bridge_any5": float(np.mean(v["anyh"])), "bridge_cov5": float(np.mean(v["cov"]))} for lam, v in out.items()}
if __name__ == "__main__":
    lams = (None, 0.9, 0.8, 0.7, 0.5)
    tabs = {l: {} for l in lams}
    for w in WORLDS:
        for s in ALLS:
            o = bridge_axes(w, s, lams)
            for l in lams: tabs[l][(w, s)] = o[l]
    import pickle; pickle.dump(tabs, open(f"{RES}/r40_tabs.pkl", "wb"))
    for l in lams: print(l, {k: round(float(np.mean([tabs[l][(w, s)][k] for w in WORLDS for s in TUNE])), 3) for k in tabs[l][("A", 1)]}, "TUNE")
