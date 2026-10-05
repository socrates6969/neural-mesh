"""R82: does the R63 retrieval profile (adaptive alpha + BM25 .9/.5) buy TOKENS? Compare context policies by (ctx_hit, tok_ctx) per world:
base = default hybrid .7 top-5 (the current context); cand A = profile top-5 (quality-up), cand B = profile top-3 (token-down), cand C = profile top-4."""
import sys; sys.path.insert(0, "../")
from r72 import *
from r60 import mkp
from gate import compare
prof = mkp(.3, 1.0, 2.0)
RPROF = prof(W.docs, W.dvec, "hybrid", .7)
def real_t(seed, make_ret, k, nq=120):
    v = W.inst.get(str(seed))
    if v is None: return None
    ret = make_ret; h, t = [], []
    for row in v["rows"][:nq]:
        tid = row["t"][0][0]; rel = set(Q[row["q"]]["rel"]); top = np.argsort(-ret.scores(W.texts[tid], W.emb[tid]))[:k]
        h.append(float(any(d in rel for d in top))); t.append(float(RTOK[top].sum()))
    return {"ctx_hit": float(np.mean(h)), "tok_ctx": float(np.mean(t))}
def dom_t(world, seed, profile, k):
    r = random.Random(seed * 7919 + (world == "B")); g = D.stores["general"]; qs = r.sample(g["qs"], 12)
    for gi, q1, q2, ans in qs:
        if world == "B": typo(r, q2, 0.12)
    dd = D.stores["domain"]; n = len(dd["docs"]); qs = r.sample(dd["qs"], 24); keep = sorted(set(r.sample(range(n), int(.7 * n))) | {q[0] for q in qs}); idx = {d: i for i, d in enumerate(keep)}
    docs = [dd["docs"][d] for d in keep]; vecs = np.stack([D.v(d) for d in docs]); tags = [dd["dom"][d] for d in keep]
    ret = prof(docs, vecs, "hybrid", .7, tags) if profile else Retriever(docs, vecs, "hybrid", .7); T = np.array([tk(d) for d in docs]); h, t = [], []
    for gd, dom, q1, q2 in qs:
        q = q1 if world == "A" else typo(r, q2, 0.12); top = np.argsort(-ret.scores(q, D.v(q)))[:k]; h.append(float(idx[gd] in top)); t.append(float(T[top].sum()))
    return {"ctx_hit": float(np.mean(h)), "tok_ctx": float(np.mean(t))}
def table(profile, k, seeds):
    tab = {}
    for s in seeds:
        a = real_t(s, RPROF if profile else W.retriever("hybrid", .7), k)
        if a: tab[("R", s)] = a
        for w in WORLDS: tab[(w, s)] = dom_t(w, s, profile, k)
    return tab
if __name__ == "__main__":
    seeds = HELD + CONF; base = table(False, 5, seeds)
    for name, (p, k) in {"profile k5": (True, 5), "profile k4": (True, 4), "profile k3": (True, 3)}.items():
        c = table(p, k, seeds); print(f"== {name} vs default k5")
        m = lambda t, ax, w: np.mean([t[(w, s)][ax] for s in seeds if (w, s) in t])
        print("   " + "  ".join(f"{w}: hit {m(base,'ctx_hit',w):.3f}->{m(c,'ctx_hit',w):.3f} tok {m(base,'tok_ctx',w):.1f}->{m(c,'tok_ctx',w):.1f}" for w in "ABR"))
        compare(base, c, ["tok_ctx", "ctx_hit"], tol=0.002, delta=0.01, quiet=True) and None
        v, rows, gain, loss = compare(base, c, ["tok_ctx", "ctx_hit"], tol=0.002, quiet=True); print("   verdict", v, "| gains", gain, "| losses", loss)
