"""R91b gate: int8 dense shadow (Rust `HybridIndex::with_int8_dense`) vs f32 FAST dense scan, default hybrid .7, real corpora A (1,939 unique queries) and B (1,029); rank of first relevant doc comes from the Rust bin
`n11_hybrid_q8` (results/r91_ranks.json, ranks searched to 1000). Item-disjoint halves (fixed permutation), per-query paired t, Holm over 16 tests, TOST +-.01 on MRR."""
import json, math, os, random, numpy as np
from scipy import stats
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"); TP = os.environ.get("TP") == "1"; R = json.load(open(f"{RES}/r91_ranks{'_tp' if TP else ''}.json"))
def rr(x): return 0.0 if x < 0 else 1.0 / (1 + x)
def h5(x): return float(0 <= x < 5)
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); r = 0
    for k, i in enumerate(o): r = max(r, min(1, ps[i] * (m - k))); out[i] = r
    return out
rows = []
for tag in "AB":
    a = np.array([rr(x) for x in R[tag]["rank_tp_f32" if TP else "rank_f32"]]); b = np.array([rr(x) for x in R[tag]["rank_tp_i8" if TP else "rank_i8"]]); ha = np.array([h5(x) for x in R[tag]["rank_tp_f32" if TP else "rank_f32"]]); hb = np.array([h5(x) for x in R[tag]["rank_tp_i8" if TP else "rank_i8"]])
    n = len(a); ids = list(range(n)); random.Random(91).shuffle(ids); halves = {"H1": ids[:n // 2], "H2": ids[n // 2:]}
    print(f"{tag}: MRR f32 {a.mean():.4f} int8 {b.mean():.4f}; hit@5 {ha.mean():.4f} {hb.mean():.4f}")
    for hn, hid in halves.items():
        for ax, x, y in (("mrr", a, b), ("hit5", ha, hb)):
            d = (y - x)[hid]; p = 1.0 if d.std() == 0 else float(stats.ttest_1samp(d, 0).pvalue)
            se = d.std(ddof=1) / math.sqrt(len(d)) + 1e-12; m = d.mean(); tost = float(max(1 - stats.t.cdf((m + .01) / se, len(d) - 1), stats.t.cdf((m - .01) / se, len(d) - 1)))
            rows.append(dict(corpus=tag, half=hn, ax=ax, d=float(m), p=p, tost=tost, W=int((d > 0).sum()), L=int((d < 0).sum())))
adj = holm([r["p"] for r in rows])
for r, a_ in zip(rows, adj): r["ph"] = a_; print(f"  {r['corpus']}{r['half']} {r['ax']:5s} d={r['d']:+.4f} W/L {r['W']}/{r['L']} Holm p={r['ph']:.3g}  TOST(+-.01) p={r['tost']:.2g}")
json.dump(rows, open(f"{RES}/r91{'_tp' if TP else ''}.json", "w"), default=float)
