"""R83: the R63 profile retrieves longer docs (R82: +8 tokens real at k5). Combine with the R72 length-aware re-rank: z(score) - lam*(log len - mean) on top of the profile; gate vs default hybrid k5."""
import sys; sys.path.insert(0, "../")
from r82 import *
def real_l(seed, lam, k=5, nq=120):
    v = W.inst.get(str(seed))
    if v is None: return None
    h, t = [], []
    for row in v["rows"][:nq]:
        tid = row["t"][0][0]; rel = set(Q[row["q"]]["rel"]); sc = z(RPROF.scores(W.texts[tid], W.emb[tid])); top = np.argsort(-(sc - lam * RLT))[:k]
        h.append(float(any(d in rel for d in top))); t.append(float(RTOK[top].sum()))
    return {"ctx_hit": float(np.mean(h)), "tok_ctx": float(np.mean(t))}
def dom_l(world, seed, lam, k=5):
    r = random.Random(seed * 7919 + (world == "B")); g = D.stores["general"]; qs = r.sample(g["qs"], 12)
    for gi, q1, q2, ans in qs:
        if world == "B": typo(r, q2, 0.12)
    dd = D.stores["domain"]; n = len(dd["docs"]); qs = r.sample(dd["qs"], 24); keep = sorted(set(r.sample(range(n), int(.7 * n))) | {q[0] for q in qs}); idx = {d: i for i, d in enumerate(keep)}
    docs = [dd["docs"][d] for d in keep]; vecs = np.stack([D.v(d) for d in docs]); ret = prof(docs, vecs, "hybrid", .7, [dd["dom"][d] for d in keep]); T = np.array([tk(d) for d in docs]); L = np.log(T); L -= L.mean(); h, t = [], []
    for gd, dom, q1, q2 in qs:
        q = q1 if world == "A" else typo(r, q2, 0.12); top = np.argsort(-(z(ret.scores(q, D.v(q))) - lam * L))[:k]; h.append(float(idx[gd] in top)); t.append(float(T[top].sum()))
    return {"ctx_hit": float(np.mean(h)), "tok_ctx": float(np.mean(t))}
if __name__ == "__main__":
    seeds = HELD + CONF; base = table(False, 5, seeds)
    for lam in (0.1, 0.2, 0.3):
        c = {}
        for s in seeds:
            a = real_l(s, lam)
            if a: c[("R", s)] = a
            for w in WORLDS: c[(w, s)] = dom_l(w, s, lam)
        m = lambda t, ax, w: np.mean([t[(w, s)][ax] for s in seeds if (w, s) in t])
        print(f"== profile + length-aware lam={lam} vs default k5: " + "  ".join(f"{w}: hit {m(base,'ctx_hit',w):.3f}->{m(c,'ctx_hit',w):.3f} tok {m(base,'tok_ctx',w):.1f}->{m(c,'tok_ctx',w):.1f}" for w in "ABR"))
        v, rows, gain, loss = compare(base, c, ["tok_ctx", "ctx_hit"], tol=0.002, quiet=True); print("   verdict", v, "| gains", gain, "| losses", loss)
