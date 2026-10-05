"""R96: confidence-gated ENCODER SKIP. If the lexical (BM25) side is already confident, answer from BM25 alone and never encode the query (saves the 2-4 ms encoder, which dominates lookup latency, R75/R85).
Gate signal = R2 lexical coverage f = max BM25 / ((k1+1) * sum idf of in-vocab query tokens) (+ optional top-1 margin). Rule family: skip iff f >= tau. tau chosen on corpus A TUNE queries (seeds 1-8, 365 q) with the
pre-declared criterion: smallest tau whose SKIPPED set loses <= .005 mean RR vs hybrid .7 on TUNE; evaluated on (a) corpus A queries NOT in TUNE (item-disjoint, halves H1/H2) and (b) corpus B (never tuned; halves).
Baseline = default hybrid .7 (fp32 dense). Cascade = BM25 ranking if f >= tau else hybrid. Per-query paired t on RR / hit@5, Holm, TOST +-.01 on MRR."""
import sys, os, json, math, random; sys.path.insert(0, "../")
from lib import *
from scipy import stats
def qstats(Wd, qmap):
    bm = BM25(Wd.docs); hyb = Wd.retriever("hybrid", .7); out = {}
    for q, tid in qmap.items():
        txt = Wd.texts[tid]; b = bm.score(txt); seen = {t for t in toks(txt) if t in bm.post}; den = sum(bm.idf[t] for t in seen) * (bm.k1 + 1); f = float(b.max() / den) if den > 0 else 0.0
        s2 = np.sort(b)[-2] if len(b) > 1 else 0.0; marg = float((b.max() - s2) / b.max()) if b.max() > 0 else 0.0
        rel = set(Wd.queries[q]["rel"]); sh = hyb.scores(txt, Wd.emb[tid]); rb = min(int((b > b[d]).sum()) for d in rel) if b.max() > 0 else 10**6; rh = min(int((sh > sh[d]).sum()) for d in rel)
        out[q] = dict(f=f, marg=marg, rr_b=1.0 / (1 + rb), rr_h=1.0 / (1 + rh), h5_b=float(rb < 5), h5_h=float(rh < 5))
    return out
def uq(Wd, seeds=None):
    m = {}
    for s, v in Wd.inst.items():
        if s == "R" or (seeds is not None and int(s) not in seeds): continue
        for row in v["rows"]: m[row["q"]] = row["t"][0][0]
    return m
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); r = 0
    for k, i in enumerate(o): r = max(r, min(1, ps[i] * (m - k))); out[i] = r
    return out
if __name__ == "__main__":
    WA, WB = RealWorld("A"), RealWorld("B"); qa_all = uq(WA); qa_tune = uq(WA, set(range(1, 9))); qa_rest = {q: t for q, t in qa_all.items() if q not in qa_tune}
    SA, SB = qstats(WA, qa_all), qstats(WB, uq(WB)); print("queries: A tune", len(qa_tune), "A rest", len(qa_rest), "B", len(SB))
    tune = [SA[q] for q in qa_tune]; fs = np.array([x["f"] for x in tune]); print("coverage f quantiles on TUNE (10/25/50/75/90):", np.round(np.percentile(fs, [10, 25, 50, 75, 90]), 3))
    print("TUNE: hybrid MRR %.4f, BM25-only MRR %.4f" % (np.mean([x["rr_h"] for x in tune]), np.mean([x["rr_b"] for x in tune])))
    cands = []
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]:
        sk = [x for x in tune if x["f"] >= tau]
        if len(sk) < 10: print(f"  tau {tau}: skipped {len(sk)} (too few)"); continue
        d = np.mean([x["rr_b"] - x["rr_h"] for x in sk]); print(f"  tau {tau}: skipped {len(sk)/len(tune):.3f} of queries; skipped-set dRR (BM25 - hybrid) {d:+.4f}")
        if d >= -0.005: cands.append((tau, len(sk) / len(tune)))
    tau = min(c[0] for c in cands) if cands else None; print("tau* (pre-declared rule) =", tau)
    rows = []
    if tau is not None:
        for name, S in (("A-rest", {q: SA[q] for q in qa_rest}), ("B", SB)):
            ids = sorted(S); random.Random(96).shuffle(ids); halves = {"H1": ids[:len(ids) // 2], "H2": ids[len(ids) // 2:]}
            sk = [q for q in ids if S[q]["f"] >= tau]
            print(f"{name}: skipped {len(sk)}/{len(ids)} = {len(sk)/len(ids):.3f}; skipped-set RR BM25 {np.mean([S[q]['rr_b'] for q in sk]):.4f} vs hybrid {np.mean([S[q]['rr_h'] for q in sk]):.4f}")
            for hn, hid in halves.items():
                for ax, kb, kh in (("mrr", "rr_b", "rr_h"), ("hit5", "h5_b", "h5_h")):
                    d = np.array([(S[q][kb] if S[q]["f"] >= tau else S[q][kh]) - S[q][kh] for q in hid]); p = 1.0 if d.std() == 0 else float(stats.ttest_1samp(d, 0).pvalue)
                    se = d.std(ddof=1) / math.sqrt(len(d)) + 1e-12; m = d.mean(); tost = float(max(1 - stats.t.cdf((m + .01) / se, len(d) - 1), stats.t.cdf((m - .01) / se, len(d) - 1))) if d.std() > 0 else 0.0
                    rows.append(dict(set=name, half=hn, ax=ax, d=float(m), p=p, tost=tost, skipped=len([q for q in hid if S[q]["f"] >= tau]) / len(hid)))
        adj = holm([r["p"] for r in rows])
        for r, a in zip(rows, adj): r["ph"] = a; print(f"  {r['set']}{r['half']} {r['ax']:5s} cascade-hybrid d={r['d']:+.4f}  Holm p={a:.3g}  TOST p={r['tost']:.2g}  skipped {r['skipped']:.3f}")
    json.dump(dict(tau=tau, rows=rows), open(f"{RES}/r96.json", "w"), default=float)
