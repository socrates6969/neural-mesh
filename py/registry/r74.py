"""R74: which injected-candidate policy reaches the highest hit@k for a fixed token budget on the capability registry?
Arms: hybrid .7 (default), dedup-by-name (same kind+name only once), kind-MMR, adaptive alpha. Sets: Q (dev), Q2 (clean), Q3 (blind, q3.py). Paired exact sign tests, Holm."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_eval import *
from q3 import Q3
ents, texts, E = load(); bm = BM25(texts)
TK = lambda e: 1.33 * len(f"{e['kind']} {e['name']}: {e['desc']}".split())
def order_default(sc): return list(np.argsort(-sc))
def order_dedup(sc):
    seen, out = set(), []
    for j in np.argsort(-sc)[:200]:
        key = ents[j]["name"].lower()
        if key in seen: continue
        seen.add(key); out.append(j)
    return out
def hitk(order, acc, k): return float(any(ents[j]["name"].lower() in {a.lower() for a in acc} for j in order[:k]))
def evalset(qs, name):
    QE = embed([q for q, _ in qs]); out = {}
    for arm, (ordf, scf) in {"hybrid": (order_default, lambda q, d: .7 * z(d) + .3 * z(bm.score(q))), "dedup": (order_dedup, lambda q, d: .7 * z(d) + .3 * z(bm.score(q))),
                             "adaptive+dedup": (order_dedup, lambda q, d: adaptive(bm, q, d)), "dense+dedup": (order_dedup, lambda q, d: d)}.items():
        H = {k: np.array([hitk(ordf(scf(q, E @ QE[i])), acc, k) for i, (q, acc) in enumerate(qs)]) for k in (1, 3, 5, 10)}
        T = {k: np.mean([sum(TK(ents[j]) for j in ordf(scf(q, E @ QE[i]))[:k]) for i, (q, acc) in enumerate(qs)]) for k in (3, 5)}
        out[arm] = (H, T)
    print(f"-- {name} (n={len(qs)})")
    for arm, (H, T) in out.items(): print(f"  {arm:15s} " + " ".join(f"hit@{k} {H[k].mean():.3f}" for k in H) + f" | tokens k3 {T[3]:.0f} k5 {T[5]:.0f}")
    return out
if __name__ == "__main__":
    res = {n: evalset(qs, n) for n, qs in (("Q dev", Q), ("Q2 clean", Q2), ("Q3 blind", Q3))}
    allq = Q + Q2 + Q3; R = evalset(allq, "ALL")
    for k in (3, 5):
        for arm in ("dedup", "adaptive+dedup", "dense+dedup"):
            a, b = R[arm][0][k], R["hybrid"][0][k]; w, l = int((a > b).sum()), int((a < b).sum())
            print(f"k={k} {arm} vs hybrid: d={a.mean()-b.mean():+.3f} W/L {w}/{l} p={stats.binomtest(w, w+l, .5).pvalue if w+l else 1:.3g}")
