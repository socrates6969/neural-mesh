from lab import *
T = ["domain_r1", "domain_mrr", "domain_route", "real_mrr"]
# R3: RRF fusion
class RRF:
    def __init__(s, docs, vecs, arm, alpha, k=60): s.v, s.bm, s.k = vecs, BM25(docs), k
    def scores(s, qt, qv):
        d = s.v @ qv; b = s.bm.score(qt); rd = np.empty(len(d)); rd[np.argsort(-d)] = np.arange(len(d)); rb = np.empty(len(d)); rb[np.argsort(-b)] = np.arange(len(b))
        return 1 / (s.k + rd) + (1 / (s.k + rb)) * (b > 0)
#run("R3 RRF(k=60) fusion", T, make=lambda d, v, a, al: RRF(d, v, a, al))
# R4: corpus-mean centering of the dense channel (+ hybrid)
class Cen:
    def __init__(s, docs, vecs, arm, alpha):
        s.mu = vecs.mean(0); V = vecs - s.mu; s.V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9); s.bm = BM25(docs); s.a = alpha
    def scores(s, qt, qv):
        q = qv - s.mu; q = q / (np.linalg.norm(q) + 1e-9); return s.a * z(s.V @ q) + (1 - s.a) * z(s.bm.score(qt))
#run("R4 centered dense (corpus mean) + hybrid .7", T, make=lambda d, v, a, al: Cen(d, v, a, al))
# R5: dense PRF (Rocchio) on top of hybrid
class PRF:
    def __init__(s, docs, vecs, arm, alpha, beta=0.5, k=3): s.V, s.bm, s.a, s.beta, s.k = vecs, BM25(docs), alpha, beta, k
    def scores(s, qt, qv):
        b = z(s.bm.score(qt)); sc = s.a * z(s.V @ qv) + (1 - s.a) * b
        top = np.argsort(-sc)[:s.k]; q2 = qv + s.beta * s.V[top].mean(0); q2 /= np.linalg.norm(q2)
        return s.a * z(s.V @ q2) + (1 - s.a) * b
#run("R5 Rocchio PRF beta .5 k3 on hybrid .7", T, make=lambda d, v, a, al: PRF(d, v, a, al))
# R6: domain-centroid soft prior (needs doc tags; domain store only; others unchanged)
class DomPrior:
    wants_tags = True
    def __init__(s, docs, vecs, arm, alpha, tags=None, g=0.5):
        s.V, s.bm, s.a, s.g = vecs, BM25(docs), alpha, g; s.tags = tags
        if tags:
            doms = sorted(set(tags)); s.cent = np.stack([vecs[[i for i, t in enumerate(tags) if t == d]].mean(0) for d in doms]); s.cent /= np.linalg.norm(s.cent, axis=1, keepdims=True)
            s.didx = np.array([doms.index(t) for t in tags])
    def scores(s, qt, qv):
        sc = s.a * z(s.V @ qv) + (1 - s.a) * z(s.bm.score(qt))
        if s.tags: sc = sc + s.g * z((s.cent @ qv))[s.didx]
        return sc
def mkdp(g, a=None):
    f = lambda d, v, ar, al, tags=None: DomPrior(d, v, ar, al if a is None else a, tags, g); f.wants_tags = True; return f
#run("R6a domain-centroid soft prior g=.5 (hybrid .7)", T, make=mkdp(0.5))
#run("R6b domain-centroid soft prior g=1.0 (hybrid .7)", T, make=mkdp(1.0))
