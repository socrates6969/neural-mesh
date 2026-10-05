"""R79: token / quality view of routing on NOVEL phrasing (ROUTE2 dev, ROUTE4 blind): tokens forwarded to the LLM, tool_ok, tool_catch, accuracy for kw / kNN / kNN+72 ROUTE3 exemplars."""
from lab import *
from lib import *
from data_route2 import ROUTE2
from data_route3 import ROUTE3
from data_route4 import ROUTE4
TOOLS = {"calc", "units", "csp", "sandbox"}
D = Data()
X0 = np.stack([D.v(t) for w in WORLDS for s in TUNE for t, _ in routing_items(D, w, s)]); y0 = np.array([l for w in WORLDS for s in TUNE for _, l in routing_items(D, w, s)])
ex = [(q, l) for l, qs in ROUTE3.items() for q in qs]; X1 = np.vstack([X0] + [D.v(t)[None] for t, _ in ex]); y1 = np.concatenate([y0, [l for _, l in ex]])
def knn(X, y, k=5):
    def f(t):
        s = X @ D.v(t); top = np.argsort(-s)[:k]; v = {}
        for i in top: v[y[i]] = v.get(y[i], 0) + float(s[i])
        return max(v, key=v.get)
    return f
def evaluate(items, f):
    pred = [f(t) if f else None for t, _ in items]; tk = [1.33 * len(t.split()) for t, _ in items]
    if f is None: return 1.0, 0.0, 0.0, float(np.mean(tk))
    nt = [l not in TOOLS for _, l in items]; ok = np.mean([p not in TOOLS for p, n in zip(pred, nt) if not n and False] or [0])
    tw = np.mean([p in TOOLS for p, n in zip(pred, nt) if n]); tm = np.mean([p not in TOOLS for p, n in zip(pred, nt) if not n])
    return 1 - tw, 1 - tm, float(np.mean([p == l for p, (_, l) in zip(pred, items)])), float(np.mean([t if p not in TOOLS else 0 for t, p in zip(tk, pred)]))
for name, S in (("ROUTE2 (dev)", ROUTE2), ("ROUTE4 (blind)", ROUTE4)):
    items = [(q, l) for l, qs in S.items() for q in qs]
    print(f"{name} n={len(items)}")
    for an, f in (("all-LLM", None), ("kw", route_kw), ("kNN", knn(X0, y0)), ("kNN+ROUTE3", knn(X1, y1))):
        o, c, a, t = evaluate(items, f); print(f"   {an:11s} tool_ok {o:.3f} tool_catch {c:.3f} acc {a:.3f} tok_llm/item {t:.1f}")
