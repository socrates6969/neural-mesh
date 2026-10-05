"""R42: forgetting caps on the REAL doc store (6,831 paragraphs): keep fraction f of docs under a policy, evaluate on queries NOT used to build usage stats.
Policies: random | usage (docs relevant to TUNE-seed queries, ranked by count) | usage_ret (docs that TUNE queries retrieve in their top-3, no gold)."""
from lab import *
from lib import *
import json
W = RealWorld("A"); INST = W.inst; Q = W.queries
tune_seeds = [s for s in INST if s.isdigit() and int(s) <= 8]
seen_q = {r["q"] for s in tune_seeds for r in INST[s]["rows"]}
def usage_counts():
    c = np.zeros(W.ndoc)
    for s in tune_seeds:
        for r in INST[s]["rows"]:
            for d in Q[r["q"]]["rel"]: c[d] += 1
    return c
def usage_ret_counts(k=3):
    c = np.zeros(W.ndoc); ret = W.retriever("hybrid", 0.7)
    for s in tune_seeds:
        for r in INST[s]["rows"][:120]:
            tid = r["t"][0][0]; sc = ret.scores(W.texts[tid], W.emb[tid])
            for d in np.argsort(-sc)[:k]: c[d] += 1
    return c
def keep_set(policy, frac, rng):
    n = W.ndoc; m = int(frac * n)
    if policy == "random": return set(rng.sample(range(n), m))
    c = USAGE if policy == "usage" else USAGE_RET
    order = sorted(range(n), key=lambda d: (-c[d], rng.random())); return set(order[:m])
def evaluate(keep, seeds, nq=120):
    keep = sorted(keep); idx = {d: i for i, d in enumerate(keep)}
    docs = [W.docs[d] for d in keep]; vecs = W.dvec[keep]; ret = Retriever(docs, vecs, "hybrid", 0.7); out = {}
    for s in seeds:
        v = INST.get(str(s))
        if v is None: continue
        rr, h1, cov = [], [], []
        for row in v["rows"][:nq * 3]:
            if row["q"] in seen_q: continue
            if len(rr) >= nq: break
            rel = [idx[d] for d in Q[row["q"]]["rel"] if d in idx]
            cov.append(float(bool(rel)))
            if not rel: rr.append(0.0); h1.append(0.0); continue
            tid = row["t"][0][0]; sc = ret.scores(W.texts[tid], W.emb[tid]); best = min(int((sc > sc[d]).sum()) for d in rel)
            rr.append(1.0 / (1 + best)); h1.append(float(best == 0))
        out[s] = {"real_mrr": float(np.mean(rr)), "real_h1": float(np.mean(h1)), "rel_kept": float(np.mean(cov))}
    return out
USAGE = usage_counts(); USAGE_RET = usage_ret_counts()
if __name__ == "__main__":
    import pickle, random
    seeds = list(range(31, 91)); res = {}
    base = evaluate(set(range(W.ndoc)), seeds); res[("full", 1.0)] = base
    for pol in ("random", "usage", "usage_ret"):
        for f in (0.75, 0.5, 0.25):
            res[(pol, f)] = evaluate(keep_set(pol, f, random.Random(7)), seeds)
            m = lambda r, k, sd: float(np.mean([r[s][k] for s in sd if s in r]))
            print(f"{pol:10s} f={f}: HELD mrr {m(res[(pol,f)],'real_mrr',HELD):.3f} (full {m(base,'real_mrr',HELD):.3f}) CONF {m(res[(pol,f)],'real_mrr',CONF):.3f} (full {m(base,'real_mrr',CONF):.3f})  rel_kept {m(res[(pol,f)],'rel_kept',HELD):.3f}")
    pickle.dump(res, open(f"{RES}/r42_res.pkl", "wb"))
