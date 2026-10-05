"""fixture for the Rust test `bm25_params_match_python_reference`: adaptive hybrid scores with BM25 k1=.9, b=.5 on the same docs/queries as b1_adaptive_fixture."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lex import *
import numpy as np
class LexP(Lex):
    def _init_bm(s, docs):
        super()._init_bm(docs); b = s.bm; b.k1, b.b = .9, .5
        b.post = {}; b.dl = np.array([len(s.tok(d)) for d in docs], np.float32)
        for i, d in enumerate(docs):
            c = {}
            for t in s.tok(d): c[t] = c.get(t, 0) + 1
            for t, f in c.items(): b.post.setdefault(t, []).append((i, f))
        b.avg = float(b.dl.mean()) if b.dl.mean() > 0 else 1.0
        b.idf = {t: math.log((b.n - len(p) + .5) / (len(p) + .5) + 1) for t, p in b.post.items()}
fx = json.load(open(f"{RES}/b1_adaptive_fixture.json"))
docs = fx["docs"]; vecs = np.array(fx["dvec"], np.float32).reshape(len(docs), 384)
o = LexP.__new__(LexP); Lex.__init__(o, docs, vecs, "hybrid", 0.7, None)
out = dict(k1=.9, b=.5, lo=.3, hi=1.0, pow=2.0, queries=[])
for q in fx["queries"]:
    qv = np.array(q["vec"], np.float32); bm = o.bm.score(q["text"]); seen = {t for t in o.tok(q["text"]) if t in o.bm.post}
    den = sum(o.bm.idf[t] for t in seen) * (o.bm.k1 + 1); f = (bm.max() / den) if den > 0 else 0.0; a = 1.0 - 0.7 * min(1.0, f) ** 2.0
    sc = o.scores(q["text"], qv); out["queries"].append(dict(text=q["text"], alpha=float(a), top=int(np.argmax(sc)), scores=[float(x) for x in sc]))
json.dump(out, open(f"{RES}/b1_bm25p_fixture.json", "w")); print(len(out["queries"]))
