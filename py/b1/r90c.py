"""R90c: cross-fitted gate for the marginal-return k rule (s = 0.5 hit per 1,000 tokens, pre-declared from the r90b sweep) vs fixed k=5, on 7 stores incl. a FRESH one (capability registry,
124 queries of R31/R32/R74, never used for this procedure). Each store: queries split in 2 item-disjoint folds by a fixed permutation; calibrate on one fold, test on the other (F1->F2 = held, F2->F1 = confirm).
Per-query paired tests (exact McNemar for hit, paired t for tokens), Holm over every (store x fold x axis) test. Tokens = 1.33 x words (proxy)."""
import sys, os, json, math; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registry")); sys.path.insert(0, "../")
import numpy as np
from scipy import stats
from r90b import pick_marg, pick_frac, curve, items as ITEMS, KMAX, RES
RULE = sys.argv[1] if len(sys.argv) > 1 else "frac98"
PICK = {"frac98": lambda h, t: pick_frac(h, t, .98), "marg05": lambda h, t: pick_marg(h, t, .5), "frac98_cap5": lambda h, t: pick_frac(h[:5], t[:5], .98) if False else next((k for k in range(1, 6) if h[k - 1] >= .98 * h[4] - 1e-12), 5)}[RULE]
import lookup_eval as LE
from q3 import Q3
from lookup_eval import *
# --- registry store items (default hybrid .7, kind+name+desc text)
ents, texts, E = LE.load(); bm = LE.BM25(texts); TK = lambda e: 1.33 * len(f"{e['kind']} {e['name']}: {e['desc']}".split())
qs = list(LE.Q) + list(getattr(LE, "Q2", [])) + list(Q3); QE = LE.embed([q for q, _ in qs]); reg = []
for i, (q, acc) in enumerate(qs):
    sc = .7 * LE.z(E @ QE[i]) + .3 * LE.z(bm.score(q)); top = list(np.argsort(-sc)[:KMAX]); accs = {a.lower() for a in acc}
    r = next((j for j, d in enumerate(top) if ents[d]["name"].lower() in accs), None); reg.append((r, np.cumsum([TK(ents[d]) for d in top]).tolist()))
ITEMS = dict(ITEMS); ITEMS["registry (fresh)"] = reg
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); r_ = 0
    for r, i in enumerate(o): r_ = max(r_, min(1, ps[i] * (m - r))); out[i] = r_
    return out
rows = []
for name, it in ITEMS.items():
    n = len(it); perm = np.random.default_rng(90).permutation(n); folds = {"F1": perm[:n // 2], "F2": perm[n // 2:]}
    for tn, (cal_f, te_f) in {"held(F1->F2)": ("F1", "F2"), "conf(F2->F1)": ("F2", "F1")}.items():
        cal = [it[i] for i in folds[cal_f]]; te = [it[i] for i in folds[te_f]]; h, t = curve(cal); k = PICK(h, t)
        a = [float(r is not None and r < k) for r, _ in te]; b = [float(r is not None and r < 5) for r, _ in te]
        w = sum(x > y for x, y in zip(a, b)); l = sum(x < y for x, y in zip(a, b)); p_hit = 1.0 if w + l == 0 else float(stats.binomtest(w, w + l, .5).pvalue)
        ta = np.array([c[k - 1] for _, c in te]); tb = np.array([c[4] for _, c in te]); d = ta - tb; p_tok = 1.0 if d.std() == 0 else float(stats.ttest_1samp(d, 0).pvalue)
        rows.append(dict(store=name, split=tn, k=k, dh=float(np.mean(a) - np.mean(b)), W=w, L=l, p_hit=p_hit, tok_b=float(tb.mean()), tok_c=float(ta.mean()), dt=float(d.mean()), p_tok=p_tok, n=len(te)))
adj = holm([x for r in rows for x in (r["p_hit"], r["p_tok"])])
for i, r in enumerate(rows): r["ph_hit"], r["ph_tok"] = adj[2 * i], adj[2 * i + 1]
print("RULE", RULE)
for r in rows: print(f"{r['store']:18s} {r['split']:13s} k={r['k']}  hit d={r['dh']:+.3f} (W/L {r['W']}/{r['L']}, Holm p={r['ph_hit']:.3g})  tokens {r['tok_b']:.0f}->{r['tok_c']:.0f} ({100*(r['tok_c']/r['tok_b']-1):+.0f}%, Holm p={r['ph_tok']:.3g})  n={r['n']}")
json.dump(rows, open(f"{RES}/r90c_{RULE}.json", "w"), default=float)
