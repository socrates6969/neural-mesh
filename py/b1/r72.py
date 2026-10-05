"""R72: token-cost as a first-class gate axis (tok_*, lower=better) + length-aware context selection.
Context = top-k docs after re-ranking adj = z(score) - lam*(log len - mean log len). Axes: ctx_hit (a relevant doc inside the context), tok_ctx (injected tokens, 1.33 x words).
domain world A/B: k=3 over the ~70% kept domain docs (24 queries/seed); real world R: k=5 over 6,831 paragraphs (120 queries/seed)."""
import sys; sys.path.insert(0, "../")
from lab import *
from lib import *
import pickle
D = Data(); W = RealWorld("A"); Q = W.queries
tk = lambda t: max(1, int(1.33 * len(t.split())))
RTOK = np.array([tk(t) for t in W.docs]); RLT = np.log(RTOK); RLT = RLT - RLT.mean()
def ctx_axes(world, seed, lam, k_dom=3):
    r = random.Random(seed * 7919 + (1 if world == "B" else 0))
    g = D.stores["general"]; _ = r.sample(g["qs"], 12)
    for it in D.items[f"{world}{seed}"]:
        pass
    # replicate knowledge_axes RNG consumption: typo() draws happen inside general queries (world B) - re-run it exactly
    r = random.Random(seed * 7919 + (1 if world == "B" else 0)); qs = r.sample(g["qs"], 12)
    for gi, q1, q2, ans in qs:
        if world == "B": typo(r, q2, 0.12)
    dd = D.stores["domain"]; n = len(dd["docs"]); qs = r.sample(dd["qs"], 24)
    keep = sorted(set(r.sample(range(n), int(.7 * n))) | {q[0] for q in qs}); idx = {d: i for i, d in enumerate(keep)}
    docs = [dd["docs"][d] for d in keep]; vecs = np.stack([D.v(d) for d in docs]); ret = Retriever(docs, vecs, "hybrid", 0.7)
    T = np.array([tk(d) for d in docs]); L = np.log(T); L = L - L.mean(); h, t = [], []
    for gd, dom, q1, q2 in qs:
        q = q1 if world == "A" else typo(r, q2, 0.12); sc = z(ret.scores(q, D.v(q))); top = np.argsort(-(sc - lam * L))[:k_dom]
        h.append(float(idx[gd] in top)); t.append(float(T[top].sum()))
    return {"ctx_hit": float(np.mean(h)), "tok_ctx": float(np.mean(t))}
def real_axes(seed, lam, k=5, nq=120):
    v = W.inst.get(str(seed))
    if v is None: return None
    ret = W.retriever("hybrid", 0.7); h, t = [], []
    for row in v["rows"][:nq]:
        tid = row["t"][0][0]; rel = set(Q[row["q"]]["rel"]); sc = z(ret.scores(W.texts[tid], W.emb[tid])); top = np.argsort(-(sc - lam * RLT))[:k]
        h.append(float(any(d in rel for d in top))); t.append(float(RTOK[top].sum()))
    return {"ctx_hit": float(np.mean(h)), "tok_ctx": float(np.mean(t))}
def table(lam, seeds):
    tab = {}
    for s in seeds:
        a = real_axes(s, lam)
        if a: tab[("R", s)] = a
        for w in WORLDS: tab[(w, s)] = ctx_axes(w, s, lam)
    return tab
if __name__ == "__main__":
    from gate import compare
    seeds = TUNE[:8] + HELD + CONF
    base = table(0.0, seeds)
    for lam in (0.1, 0.2, 0.3, 0.5):
        c = table(lam, seeds)
        m = lambda t, ax, w: np.mean([t[(w, s)][ax] for s in TUNE[:8] if (w, s) in t])
        print(f"lam={lam} TUNE (seeds1-8): " + "  ".join(f"{w}: hit {m(base,'ctx_hit',w):.3f}->{m(c,'ctx_hit',w):.3f} tok {m(base,'tok_ctx',w):.1f}->{m(c,'tok_ctx',w):.1f}" for w in "ABR"))
        pickle.dump((base, c), open(f"{RES}/r72_lam{lam}.pkl", "wb"))
