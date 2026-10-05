"""R85: query-encoder latency levers for memory lookup: int8 dynamic-quantised MiniLM, layer truncation (5/4/3 of 6 layers), and combined.
DOCS stay fp32 full-model vectors (existing store); only the QUERY is encoded with the variant (asymmetric). Retrieval = default hybrid .7 and dense-only,
real corpora A (crate docs, 1,939 unique queries) and B (project prose, 1,029 unique queries); per-query paired tests vs the fp32 encoder, item-disjoint halves (fixed permutation).
Variant embeddings/latencies were produced by scratchpad scripts in an isolated venv (onnx 1.23.1, onnxruntime) - see RD-LOG R85. Latency = per-query p50, python ORT, short real queries."""
import sys, os, json, random, math; sys.path.insert(0, "../")
from lib import *
from scipy import stats
SP = os.environ["R85_SP"]
Q = json.load(open(f"{SP}/qtexts.json")); VARS = ["q8", "t5", "t4", "t3", "t4q8"]
E = {v: np.load(f"{SP}/E_{v}.npy") for v in ["fp32"] + VARS}
lat1 = json.load(open(f"{SP}/lat_t1.json")); lat4 = json.load(open(f"{SP}/lat_t4.json"))
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); r_ = 0
    for r, i in enumerate(o): r_ = max(r_, min(1, ps[i] * (m - r))); out[i] = r_
    return out
def tostp(d, delta):
    n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n) + 1e-12
    return float(max(1 - stats.t.cdf((m + delta) / se, n - 1), stats.t.cdf((m - delta) / se, n - 1)))
off = 0; recs = []
for tag in "AB":
    W = RealWorld(tag); n = len(Q[tag]["q"]); Eq = {v: E[v][off:off + n] for v in E}; off += n
    ret = {"hyb": W.retriever("hybrid", .7), "dense": W.retriever("dense", 1.0)}
    res = {}
    for mode, r in ret.items():
        for v in ["fp32"] + VARS:
            rr, h5 = [], []
            for i, (qid, tid) in enumerate(zip(Q[tag]["q"], Q[tag]["tid"])):
                rel = set(W.queries[qid]["rel"]); sc = r.scores(Q[tag]["text"][i], Eq[v][i]); best = min(int((sc >= sc[d]).sum()) - 1 for d in rel)
                rr.append(1.0 / (1 + best)); h5.append(float(best < 5))
            res[(mode, v)] = (np.array(rr), np.array(h5))
    ids = list(range(n)); random.Random(85).shuffle(ids); halves = {"H1": ids[:n // 2], "H2": ids[n // 2:]}
    cos = {v: float(np.mean(np.sum(Eq[v] * Eq["fp32"], 1))) for v in VARS}
    for mode in ret:
        for v in VARS:
            for hn, hid in halves.items():
                for ax, k in (("mrr", 0), ("hit5", 1)):
                    a = res[(mode, v)][k][hid]; b = res[(mode, "fp32")][k][hid]; d = a - b
                    p = 1.0 if d.std() == 0 else float(stats.ttest_1samp(d, 0).pvalue)
                    recs.append(dict(corpus=tag, mode=mode, var=v, half=hn, ax=ax, base=float(b.mean()), cand=float(a.mean()), d=float(d.mean()), p=p, tost=tostp(d, .01) if d.std() > 0 else 0.0))
    print(tag, "mean cosine(variant, fp32) =", {k: round(x, 4) for k, x in cos.items()})
    for mode in ret: print(f"  {tag} {mode} fp32: MRR {res[(mode,'fp32')][0].mean():.4f} hit@5 {res[(mode,'fp32')][1].mean():.4f}")
adj = holm([r["p"] for r in recs])
for r, a in zip(recs, adj): r["ph"] = a
print("\nlatency p50 ms (1 thr | 4 thr):", {v: (round(lat1[v]["p50_ms"], 2), round(lat4[v]["p50_ms"], 2)) for v in lat1})
for v in VARS:
    for mode in ("hyb", "dense"):
        sub = [r for r in recs if r["var"] == v and r["mode"] == mode]
        print(f"{v:5s} {mode:5s} " + "  ".join(f"{r['corpus']}{r['half']}/{r['ax']} {r['d']:+.4f}{'*' if r['ph'] < .05 else ''}" for r in sub))
json.dump(recs, open(f"{RES}/r85.json", "w"), default=float)
