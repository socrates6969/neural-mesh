from lab import *
import lab
T = ["domain_r1", "domain_mrr", "domain_route", "real_mrr"]
class AdDP:
    wants_tags = True
    def __init__(s, docs, vecs, arm, alpha, tags=None, g=0.5, lo=.3, hi=1.0, pw=2.0, hard=0):
        s.V, s.bm, s.g, s.lo, s.hi, s.pw, s.tags, s.hard = vecs, BM25(docs), g, lo, hi, pw, tags, hard
        if tags:
            doms = sorted(set(tags)); s.cent = np.stack([vecs[[i for i, t in enumerate(tags) if t == d]].mean(0) for d in doms]); s.cent /= np.linalg.norm(s.cent, axis=1, keepdims=True); s.didx = np.array([doms.index(t) for t in tags])
    def scores(s, qt, qv):
        bm = s.bm.score(qt); seen = {t for t in toks(qt) if t in s.bm.post}; den = sum(s.bm.idf[t] for t in seen) * (s.bm.k1 + 1)
        f = (bm.max() / den) if den > 0 else 0.0; a = s.hi - (s.hi - s.lo) * min(1.0, f) ** s.pw
        sc = a * z(s.V @ qv) + (1 - a) * z(bm)
        if s.tags:
            cs = s.cent @ qv
            if s.hard: 
                top = np.argsort(-cs)[:s.hard]; sc = sc + np.where(np.isin(s.didx, top), 0.0, -100.0)
            else: sc = sc + s.g * z(cs)[s.didx]
        return sc
def mk(**kw):
    f = lambda d, v, ar, al, tags=None: AdDP(d, v, ar, al, tags, **kw); f.wants_tags = True; return f
# new baseline = R2 adaptive (no prior)
lab.BASE = cand(real=True, make=mk(g=0.0))
run("R7a prior g=.5 on top of adaptive alpha (vs adaptive base)", T, make=mk(g=0.5))
run("R7b prior g=1.0 on top of adaptive alpha (vs adaptive base)", T, make=mk(g=1.0))
run("R8a hard route top-1 domain on top of adaptive", T, make=mk(hard=1))
run("R8b hard route top-2 domains on top of adaptive", T, make=mk(hard=2))
