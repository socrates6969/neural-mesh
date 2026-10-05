"""R78 explore: agreement-based adaptive k. k=k_agree if dense top-1 == BM25 top-1 (and they are the hybrid top-1) else k_dis. TUNE seeds 1-8."""
import sys; sys.path.insert(0, "../")
from r72 import *
def real_rows(seeds, nq=120):
    ret = W.retriever("hybrid", .7)
    for s in seeds:
        v = W.inst.get(str(s))
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; q = W.texts[tid]; qv = W.emb[tid]; dn = ret.vecs @ qv; bm = ret.bm.score(q); sc = .7 * z(dn) + .3 * z(bm)
            yield np.argsort(-sc)[:10], int(np.argmax(dn)), int(np.argmax(bm)), bm.max() > 0, set(Q[row["q"]]["rel"]), RTOK
def dom_rows(world, seeds):
    for s in seeds:
        r = random.Random(s * 7919 + (world == "B")); g = D.stores["general"]; qs = r.sample(g["qs"], 12)
        for gi, q1, q2, ans in qs:
            if world == "B": typo(r, q2, 0.12)
        dd = D.stores["domain"]; n = len(dd["docs"]); qs = r.sample(dd["qs"], 24); keep = sorted(set(r.sample(range(n), int(.7 * n))) | {q[0] for q in qs}); idx = {d: i for i, d in enumerate(keep)}
        docs = [dd["docs"][d] for d in keep]; vecs = np.stack([D.v(d) for d in docs]); ret = Retriever(docs, vecs, "hybrid", .7); T = np.array([tk(d) for d in docs])
        for gd, dom, q1, q2 in qs:
            q = q1 if world == "A" else typo(r, q2, 0.12); dn = vecs @ D.v(q); bm = ret.bm.score(q); sc = .7 * z(dn) + .3 * z(bm)
            yield np.argsort(-sc)[:10], int(np.argmax(dn)), int(np.argmax(bm)), bm.max() > 0, {idx[gd]}, T
def evalp(rows, ka, kd):
    h, t = [], []
    for top, dn, bm, hasbm, rel, T in rows:
        k = ka if (dn == bm and hasbm) else kd; g = top[:k]; h.append(float(any(d in rel for d in g))); t.append(float(T[g].sum()))
    return np.mean(h), np.mean(t)
if __name__ == "__main__":
    S = range(1, 9)
    for name, rows in (("real", list(real_rows(S))), ("domainA", list(dom_rows("A", S))), ("domainB", list(dom_rows("B", S)))):
        print(name, "fixed:", " ".join("k%d %.3f/%.0f" % ((k,) + evalp(rows, k, k)) for k in (1, 2, 3, 5)))
        for ka, kd in ((1, 5), (1, 3), (2, 5), (2, 8), (3, 8)): print(f"   agree->{ka} else {kd}: hit %.3f tok %.0f" % evalp(rows, ka, kd))
    print("---- vs fixed k8/k10 (top lists hold 10)")
    for name, rows in (("real", list(real_rows(S))), ("domainB", list(dom_rows("B", S)))):
        print(name, "fixed k8 %.3f/%.0f k10 %.3f/%.0f" % (evalp(rows, 8, 8) + evalp(rows, 10, 10)))
        for ka, kd in ((3, 10), (4, 10), (5, 10), (4, 8), (5, 8), (6, 10)): print(f"   agree->{ka} else {kd}: hit %.3f tok %.0f" % evalp(rows, ka, kd))
