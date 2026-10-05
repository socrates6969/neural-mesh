"""R80: abstain option for the kNN router (forward to the LLM when the top vote share < tau or the best similarity < smin). Tune on ROUTE2 (dev); test on ROUTE4 (blind) + battery HELD/CONF (in-distribution guard)."""
from r79 import *
def knn_conf(X, y, k=5):
    def f(t):
        s = X @ D.v(t); top = np.argsort(-s)[:k]; v = {}
        for i in top: v[y[i]] = v.get(y[i], 0) + float(s[i])
        lab = max(v, key=v.get); return lab, v[lab] / sum(v.values()), float(s[top[0]])
    return f
F = knn_conf(X1, y1)
def metrics(items, tau, smin):
    pred = []
    for t, l in items:
        lab, sh, sm = F(t); pred.append("llm" if (sh < tau or sm < smin) else lab)
    nt = [l not in TOOLS for _, l in items]; tk = [1.33 * len(t.split()) for t, _ in items]
    tw = np.mean([p in TOOLS for p, n in zip(pred, nt) if n]); tm = np.mean([p not in TOOLS for p, n in zip(pred, nt) if not n])
    return 1 - tw, 1 - tm, float(np.mean([p == l for p, (_, l) in zip(pred, items)])), float(np.mean([t if p not in TOOLS else 0 for t, p in zip(tk, pred)])), float(np.mean([p in TOOLS and l in TOOLS and p != l for p, (_, l) in zip(pred, items)]))
R2 = [(q, l) for l, qs in ROUTE2.items() for q in qs]; R4 = [(q, l) for l, qs in ROUTE4.items() for q in qs]
B = [(t, l) for w in WORLDS for s in HELD + CONF[:10] for t, l in routing_items(D, w, s)]
for tau in (0, .5, .6, .7, .8):
    for smin in (0, .5, .6):
        r = [metrics(S, tau, smin) for S in (R2, R4, B)]
        print(f"tau={tau} smin={smin}: " + " | ".join(f"{n}: ok {x[0]:.3f} catch {x[1]:.3f} acc {x[2]:.3f} tok {x[3]:.1f} wrongtool {x[4]:.3f}" for n, x in zip(("R2", "R4", "bat"), r)))
