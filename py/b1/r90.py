"""R90: per-store retrieval-budget (k) SELECTION procedure, validated as a procedure (R77 only tabulated hit@k per store). choose_k(ranks, kmax=8, frac=.98) = smallest k whose hit@k >= frac * hit@kmax,
calibrated on n_cal queries of the store and evaluated on DISJOINT test queries of the same store (200 random splits per store and n_cal). Stores: real A (1,939 unique real queries), real B (1,029),
domain A / B (24 queries x 60 seeds each, B typo'd), general A / B (12 x 60). Default hybrid .7. rank = 0-based rank of the first relevant doc (None if not in top-kmax). Tokens = 1.33 x words (proxy)."""
import sys, json; sys.path.insert(0, "../")
from r72 import *
KMAX = 8
def choose_k(ranks, kmax=KMAX, frac=.98, min_q=20):
    n = len(ranks)
    if n < min_q: return None
    hit = [sum(1 for r in ranks if r is not None and r < k) / n for k in range(1, kmax + 1)]
    if hit[-1] == 0: return None
    for k in range(1, kmax + 1):
        if hit[k - 1] >= frac * hit[-1] - 1e-12: return k
def store_items():
    out = {}
    for tag, Wd in (("real A", W), ("real B", RealWorld("B"))):
        ret = Wd.retriever("hybrid", .7); T = np.array([tk(t) for t in Wd.docs]); qm = {}
        for s, v in Wd.inst.items():
            if s == "R": continue
            for row in v["rows"]: qm[row["q"]] = row["t"][0][0]
        it = []
        for q, tid in qm.items():
            rel = set(Wd.queries[q]["rel"]); top = np.argsort(-ret.scores(Wd.texts[tid], Wd.emb[tid]))[:KMAX]
            r = next((i for i, d in enumerate(top) if d in rel), None); it.append((r, np.cumsum(T[top]).tolist()))
        out[tag] = it
    for world in "AB":
        itg, itd = [], []
        for seed in range(1, 61):
            r = random.Random(seed * 7919 + (world == "B")); g = D.stores["general"]; docs = g["docs"]; vecs = np.stack([D.v(d) for d in docs]); ret = Retriever(docs, vecs, "hybrid", .7)
            T = np.array([tk(d) for d in docs]); qs = r.sample(g["qs"], 12)
            for gi, q1, q2, ans in qs:
                q = q1 if world == "A" else typo(r, q2, 0.12); top = np.argsort(-ret.scores(q, D.v(q)))[:KMAX]; itg.append((next((i for i, d in enumerate(top) if d == gi), None), np.cumsum(T[top]).tolist()))
            dd = D.stores["domain"]; n = len(dd["docs"]); qs = r.sample(dd["qs"], 24); keep = sorted(set(r.sample(range(n), int(.7 * n))) | {q[0] for q in qs}); idx = {d: i for i, d in enumerate(keep)}
            ddocs = [dd["docs"][d] for d in keep]; dv = np.stack([D.v(d) for d in ddocs]); rd = Retriever(ddocs, dv, "hybrid", .7); T2 = np.array([tk(d) for d in ddocs])
            for gd, dom, q1, q2 in qs:
                q = q1 if world == "A" else typo(r, q2, 0.12); top = np.argsort(-rd.scores(q, D.v(q)))[:KMAX]; itd.append((next((i for i, d in enumerate(top) if idx[gd] == d), None), np.cumsum(T2[top]).tolist()))
        out[f"general {world}"] = itg; out[f"domain {world}"] = itd
    return out
if __name__ == "__main__":
    items = store_items(); json.dump(items, open(f"{RES}/r90_items.json", "w"))
    rng = np.random.default_rng(90)
    for name, it in items.items():
        n = len(it); ranks = [x[0] for x in it]; cum = np.array([x[1] for x in it])
        hit = lambda idx, k: float(np.mean([ranks[i] is not None and ranks[i] < k for i in idx]))
        print(f"== {name} (n={n}) full-store hit@k 1/2/3/5/8: " + " ".join(f"{hit(range(n), k):.3f}" for k in (1, 2, 3, 5, 8)) + f"  full-store choose_k = {choose_k(ranks)}")
        for ncal in (20, 50, 100, 200):
            if ncal > n // 2: continue
            ks, rat, loss5, tsav, tsav3 = [], [], [], [], []
            for _ in range(200):
                perm = rng.permutation(n); cal, te = perm[:ncal], perm[ncal:]
                k = choose_k([ranks[i] for i in cal]) or 5; ks.append(k)
                h8 = hit(te, 8); rat.append(hit(te, k) / h8 if h8 > 0 else 1.0); loss5.append(hit(te, 5) - hit(te, k))
                tsav.append(1 - cum[te, k - 1].mean() / cum[te, 4].mean())
            kc = np.bincount(ks, minlength=9)[1:]; 
            print(f"   n_cal={ncal:3d}: chosen k mean {np.mean(ks):.2f} (dist k1..8 {kc.tolist()}); test hit@k/hit@8 mean {np.mean(rat):.3f}, share of splits >= .95: {np.mean(np.array(rat) >= .95):.2f}; hit loss vs k=5 {np.mean(loss5):+.3f}; tokens vs k=5 {-100*np.mean(tsav):+.0f}%")
