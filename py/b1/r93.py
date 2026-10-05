"""R93: the two speed levers COMBINED: int8 query ENCODER (R85, docs encoded once with fp32) + int8 dense shadow scan (R91). Emulated exactly in python: docs and query vectors are per-vector symmetric int8
quantised then de-quantised, so the hybrid dense scores equal the integer-dot scores. Arms vs the fp32 / f32 default: (1) int8 encoder only, (2) int8 dense only, (3) both. Real corpora A, B; item-disjoint halves; per-query paired t; Holm over 24 tests; TOST +-.01 on MRR."""
import sys, os, json, math, random; sys.path.insert(0, "../")
from lib import *
from scipy import stats
SP = os.environ["R85_SP"]; Q = json.load(open(f"{SP}/qtexts.json")); EQ8 = np.load(f"{SP}/E_q8.npy"); EQ32 = np.load(f"{SP}/E_fp32.npy")
def q8(M):
    out = np.zeros_like(M)
    for i, v in enumerate(M):
        m = np.abs(v).max(); s = m / 127.0 if m > 0 else 1.0; out[i] = np.round(v / s).clip(-127, 127) * s
    return out
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); r = 0
    for k, i in enumerate(o): r = max(r, min(1, ps[i] * (m - k))); out[i] = r
    return out
rows = []; off = 0
for tag in "AB":
    W = RealWorld(tag); n = len(Q[tag]["q"]); eq32 = W.emb[Q[tag]["tid"]]  # fp32 query vectors from the store (equal to EQ32 up to ORT noise)
    eq8 = EQ8[off:off + n]; off += n; D32 = W.dvec; D8 = q8(D32)
    arms = {"base": (D32, eq32), "enc_i8": (D32, eq8), "dense_i8": (D8, q8(eq32)), "both": (D8, q8(eq8))}
    res = {}
    for name, (dv, qv) in arms.items():
        r = Retriever(W.docs, dv, "hybrid", .7); rr, h5 = [], []
        for i, (qid, tid) in enumerate(zip(Q[tag]["q"], Q[tag]["tid"])):
            rel = set(W.queries[qid]["rel"]); sc = r.scores(Q[tag]["text"][i], qv[i]); best = min(int((sc > sc[d]).sum()) for d in rel); rr.append(1.0 / (1 + best)); h5.append(float(best < 5))
        res[name] = (np.array(rr), np.array(h5))
    ids = list(range(n)); random.Random(93).shuffle(ids); halves = {"H1": ids[:n // 2], "H2": ids[n // 2:]}
    print(tag, "MRR " + " ".join(f"{k} {v[0].mean():.4f}" for k, v in res.items()), "| hit@5 " + " ".join(f"{k} {v[1].mean():.4f}" for k, v in res.items()))
    for arm in ("enc_i8", "dense_i8", "both"):
        for hn, hid in halves.items():
            for ax, k in (("mrr", 0), ("hit5", 1)):
                d = (res[arm][k] - res["base"][k])[hid]; p = 1.0 if d.std() == 0 else float(stats.ttest_1samp(d, 0).pvalue); se = d.std(ddof=1) / math.sqrt(len(d)) + 1e-12; m = d.mean()
                tost = float(max(1 - stats.t.cdf((m + .01) / se, len(d) - 1), stats.t.cdf((m - .01) / se, len(d) - 1))) if d.std() > 0 else 0.0
                rows.append(dict(corpus=tag, arm=arm, half=hn, ax=ax, d=float(m), p=p, tost=tost))
adj = holm([r["p"] for r in rows])
for r, a in zip(rows, adj): r["ph"] = a
for arm in ("enc_i8", "dense_i8", "both"):
    print(f"{arm:9s} " + "  ".join(f"{r['corpus']}{r['half']}/{r['ax']} {r['d']:+.4f}{'*' if r['ph']<.05 else ''}{'' if r['ax']!='mrr' else f'(T{r['tost']:.0e})'}" for r in rows if r["arm"] == arm))
json.dump(rows, open(f"{RES}/r93.json", "w"), default=float)
