"""fixture for the Rust test `knn_router_matches_python_reference`: a small exemplar set + queries with python (lib.KNNRouter-style) vote shares."""
import json, sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lab import *
from lib import *
from data_route2 import ROUTE2
from data_route3 import ROUTE3
r = random.Random(3); ex = []
for w in WORLDS:
    for s in range(1, 4): ex += routing_items(D, w, s)
ex = r.sample(ex, 120)
for lab, qs in ROUTE3.items(): ex += [(q, lab) for q in r.sample(qs, 5)]
qs = [(q, lab) for lab, L in ROUTE2.items() for q in r.sample(L, 6)]
X = np.stack([D.v(t) for t, _ in ex]); y = [l for _, l in ex]
def votes(t, k=5):
    s = X @ D.v(t); top = np.argsort(-s, kind="stable")[:k]; v = {}
    for i in top: v[y[i]] = v.get(y[i], 0) + float(s[i])
    tot = sum(v.values()); return {a: b / tot for a, b in v.items()}
out = dict(dim=384, k=5, exemplars=[dict(label=l, vec=[round(float(x), 6) for x in D.v(t)]) for t, l in ex],
           queries=[dict(text=t, vec=[round(float(x), 6) for x in D.v(t)], votes=votes(t)) for t, _ in qs])
json.dump(out, open(f"{RES}/b1_knn_fixture.json", "w")); print(len(ex), len(qs))
