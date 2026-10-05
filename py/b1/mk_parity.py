import json
from rounds import *
from r02 import Adaptive
dd = D.stores["domain"]; docs = dd["docs"]; vecs = np.stack([D.v(d) for d in docs])
ad = Adaptive(docs, vecs, "hybrid", 0.7, 0.3, 1.0, 2.0)
import random
r = random.Random(5); qs = []
for gd, dom, q1, q2 in r.sample(dd["qs"], 20): qs += [q1, typo(r, q2, .15)]
out = dict(docs=docs, dvec=[float(x) for x in vecs.ravel()], queries=[], lo=0.3, hi=1.0, pow=2.0)
for q in qs:
    qv = D.v(q); bm = ad.bm.score(q); seen = {t for t in toks(q) if t in ad.bm.post}; den = sum(ad.bm.idf[t] for t in seen) * (ad.bm.k1 + 1)
    f = (bm.max() / den) if den > 0 else 0.0; a = 1.0 - 0.7 * min(1.0, f) ** 2.0
    sc = ad.scores(q, qv)
    out["queries"].append(dict(text=q, vec=[float(x) for x in qv], alpha=float(a), top=int(np.argmax(sc)), scores=[float(x) for x in sc]))
json.dump(out, open(f"{RES}/b1_adaptive_fixture.json", "w")); print(len(qs))
