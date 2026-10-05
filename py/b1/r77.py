"""R77: per-store retrieval-budget table: ctx_hit@k and injected tokens for k=1,2,3,5,8 on every store the battery has (general / domain A,B / real / registry).
Smallest k reaching >= 98% of hit@8 is the recommended cap for that store type."""
import sys; sys.path.insert(0, "../")
from r72 import *
def general(world, seed, ks):
    g = D.stores["general"]; docs = g["docs"]; vecs = np.stack([D.v(d) for d in docs]); ret = Retriever(docs, vecs, "hybrid", .7)
    r = random.Random(seed * 7919 + (world == "B")); qs = r.sample(g["qs"], 12); T = np.array([tk(d) for d in docs]); o = {k: ([], []) for k in ks}
    for gi, q1, q2, ans in qs:
        q = q1 if world == "A" else typo(r, q2, 0.12); sc = ret.scores(q, D.v(q)); top = np.argsort(-sc)
        for k in ks: o[k][0].append(float(gi in top[:k])); o[k][1].append(float(T[top[:k]].sum()))
    return o
def domain(world, seed, ks):
    r = random.Random(seed * 7919 + (world == "B")); g = D.stores["general"]; qs = r.sample(g["qs"], 12)
    for gi, q1, q2, ans in qs:
        if world == "B": typo(r, q2, 0.12)
    dd = D.stores["domain"]; n = len(dd["docs"]); qs = r.sample(dd["qs"], 24); keep = sorted(set(r.sample(range(n), int(.7 * n))) | {q[0] for q in qs}); idx = {d: i for i, d in enumerate(keep)}
    docs = [dd["docs"][d] for d in keep]; vecs = np.stack([D.v(d) for d in docs]); ret = Retriever(docs, vecs, "hybrid", .7); T = np.array([tk(d) for d in docs]); o = {k: ([], []) for k in ks}
    for gd, dom, q1, q2 in qs:
        q = q1 if world == "A" else typo(r, q2, 0.12); top = np.argsort(-ret.scores(q, D.v(q)))
        for k in ks: o[k][0].append(float(idx[gd] in top[:k])); o[k][1].append(float(T[top[:k]].sum()))
    return o
def real(seed, ks, nq=120):
    v = W.inst.get(str(seed)); ret = W.retriever("hybrid", .7); o = {k: ([], []) for k in ks}
    for row in v["rows"][:nq]:
        tid = row["t"][0][0]; rel = set(Q[row["q"]]["rel"]); top = np.argsort(-ret.scores(W.texts[tid], W.emb[tid]))
        for k in ks: o[k][0].append(float(any(d in rel for d in top[:k]))); o[k][1].append(float(RTOK[top[:k]].sum()))
    return o
ks = (1, 2, 3, 5, 8); seeds = HELD + CONF
def agg(fn):
    acc = {k: ([], []) for k in ks}
    for s in seeds:
        o = fn(s)
        for k in ks: acc[k][0].append(np.mean(o[k][0])); acc[k][1].append(np.mean(o[k][1]))
    return {k: (np.mean(acc[k][0]), np.mean(acc[k][1])) for k in ks}
tabs = {"general A": agg(lambda s: general("A", s, ks)), "general B (typos)": agg(lambda s: general("B", s, ks)), "domain A": agg(lambda s: domain("A", s, ks)), "domain B (typos)": agg(lambda s: domain("B", s, ks)), "real docs": agg(lambda s: real(s, ks))}
for n, t in tabs.items():
    print(f"{n:18s} " + " | ".join(f"k{k} hit {t[k][0]:.3f} tok {t[k][1]:5.0f}" for k in ks))
    kk = min(k for k in ks if t[k][0] >= .98 * t[8][0]); print(f"{'':18s} -> smallest k with >=98% of hit@8: k={kk} ({t[kk][1]:.0f} tokens vs {t[5][1]:.0f} at k=5, {100*(1-t[kk][1]/t[5][1]):+.0f}%)")
