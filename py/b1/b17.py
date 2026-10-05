"""B17: selective routing for the kNN capability router: abstain (hand back to the caller / LLM) when the top-class vote share (or top1-top2 margin) is below tau.
Pre-declared: tau chosen on ROUTE2 (n=84) as the SMALLEST threshold whose accepted-accuracy >= .95 (and also the one giving >= 80% coverage reported); CONFIRMED on the blind ROUTE4 (n=60): accuracy on accepted items, coverage, exact Clopper-Pearson CI,
and the same rule applied to a RANDOM abstention of equal coverage as the null (accepted-accuracy of a random subset)."""
from lab import *
from lib import *
from data_route2 import ROUTE2
from data_route4 import ROUTE4
from scipy import stats
X0, y0 = [], []
for w in WORLDS:
    for s in TUNE:
        for t, lab in routing_items(D, w, s): X0.append(D.v(t)); y0.append(lab)
X0, y0 = np.stack(X0), np.array(y0); labs = sorted(set(y0))
def items(R): return [(q, lab) for lab, qs in R.items() for q in qs]
I2, I4 = items(ROUTE2), items(ROUTE4)
def vote(q, k=5):
    s = X0 @ q; top = np.argsort(-s)[:k]; v = {l: 0.0 for l in labs}
    for i in top: v[y0[i]] += float(s[i])
    t = sum(v.values()) or 1.0; sh = sorted(((x / t, l) for l, x in v.items()), reverse=True)
    return sh[0][1], sh[0][0], sh[0][0] - sh[1][0]
def eval_set(I):
    out = []
    for t, lab in I:
        pred, share, margin = vote(D.v(t)); out.append((pred == lab, share, margin))
    return out
E2, E4 = eval_set(I2), eval_set(I4)
print("overall accuracy ROUTE2 %.3f (n=%d) ROUTE4 %.3f (n=%d)" % (np.mean([e[0] for e in E2]), len(E2), np.mean([e[0] for e in E4]), len(E4)))
def cp(k, n):
    lo = stats.beta.ppf(.025, k, n - k + 1) if k else 0.0; hi = stats.beta.ppf(.975, k + 1, n - k) if k < n else 1.0; return lo, hi
for name, ix in (("vote share", 1), ("margin", 2)):
    taus = sorted(set(e[ix] for e in E2))
    rows = []
    for tau in taus:
        acc = [e for e in E2 if e[ix] >= tau]; rows.append((tau, len(acc) / len(E2), np.mean([e[0] for e in acc])))
    cand = [r for r in rows if r[2] >= .95]
    tau95 = min(cand, key=lambda r: r[0])[0] if cand else None
    cand80 = [r for r in rows if r[1] >= .80]; tau80 = max(cand80, key=lambda r: r[0])[0] if cand80 else None
    print(f"--- confidence = {name}; ROUTE2 curve (tau, coverage, accepted-acc), sample:", [(round(r[0], 2), round(r[1], 2), round(r[2], 3)) for r in rows[::max(1, len(rows) // 8)]])
    for lab, tau in (("tau@acc>=.95 (ROUTE2)", tau95), ("tau@coverage>=.80 (ROUTE2)", tau80)):
        if tau is None: print("  ", lab, "no threshold"); continue
        a2 = [e for e in E2 if e[ix] >= tau]; a4 = [e for e in E4 if e[ix] >= tau]
        k4 = sum(e[0] for e in a4); lo, hi = cp(k4, len(a4)) if a4 else (0, 0)
        rng = np.random.RandomState(0); null = np.mean([np.mean([E4[i][0] for i in rng.choice(len(E4), len(a4), replace=False)]) for _ in range(2000)]) if a4 else float('nan')
        # p-value that accepted-acc exceeds overall-acc by chance (exact: hypergeometric upper tail)
        tot = sum(e[0] for e in E4); p = stats.hypergeom.sf(k4 - 1, len(E4), tot, len(a4)) if a4 else 1.0
        print(f"   {lab}: tau {tau:.3f} | ROUTE2 coverage {len(a2)/len(E2):.2f} acc {np.mean([e[0] for e in a2]):.3f} | ROUTE4 coverage {len(a4)/len(E4):.2f} ({len(a4)}/{len(E4)}) accepted-acc {k4/max(1,len(a4)):.3f} CP95 [{lo:.3f},{hi:.3f}] vs overall {tot/len(E4):.3f}; random-subset null {null:.3f}; hypergeometric p {p:.3f}")
