"""R65: kNN router learning from feedback: add confirmed (request, capability) pairs from a stream of NOVEL requests (ROUTE2 half A) and measure accuracy on the other half B (and vice versa),
vs adding the same number of AUTHORED exemplars (ROUTE3). 30 random splits."""
from lab import *
from lib import *
from data_route2 import ROUTE2
from data_route3 import ROUTE3
import random, pickle
X0, y0 = [], []
for w in WORLDS:
    for s in TUNE:
        for t, lab in routing_items(D, w, s): X0.append(D.v(t)); y0.append(lab)
X0, y0 = np.stack(X0), np.array(y0)
def acc(X, y, items, k=5):
    ok = 0
    for t, lab in items:
        s = X @ D.v(t); top = np.argsort(-s)[:k]; v = {}
        for i in top: v[y[i]] = v.get(y[i], 0) + float(s[i])
        ok += max(v, key=v.get) == lab
    return ok / len(items)
res = {}
for rep in range(30):
    r = random.Random(rep); A, B = [], []
    for lab, qs in ROUTE2.items():
        qq = qs[:]; r.shuffle(qq); h = len(qq) // 2; A += [(q, lab) for q in qq[:h]]; B += [(q, lab) for q in qq[h:]]
    r.shuffle(A)
    for n in (0, 6, 12, 24, 48):
        add = A[:n]
        X = np.vstack([X0] + [D.v(t)[None] for t, _ in add]) if add else X0; y = np.concatenate([y0, [l for _, l in add]]) if add else y0
        res.setdefault(("feedback", n), []).append(acc(X, y, B))
        # authored exemplars of the same count (class balanced)
        per = max(0, n // 6); ex = [(q, lab) for lab, qs in ROUTE3.items() for q in r.sample(qs, per)] if per else []
        Xa = np.vstack([X0] + [D.v(t)[None] for t, _ in ex]) if ex else X0; ya = np.concatenate([y0, [l for _, l in ex]]) if ex else y0
        res.setdefault(("authored", n), []).append(acc(Xa, ya, B))
for n in (0, 6, 12, 24, 48):
    f, a = res[("feedback", n)], res[("authored", n)]
    print(f"n={n:2d}: feedback-from-stream {np.mean(f):.3f}  authored(ROUTE3, {n//6}/class) {np.mean(a):.3f}")
pickle.dump(res, open(f"{RES}/r65_res.pkl", "wb"))
