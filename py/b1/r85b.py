"""R85b: SYMMETRIC variants (docs AND queries encoded with the same variant encoder): q8 (int8 dynamic), t4/t3 (4/3 of 6 layers), t4q8. Reference = fp32 full MiniLM docs+queries.
Same corpora/halves/tests as r85.py (hybrid .7 default and dense-only)."""
import sys, os, json, random, math; sys.path.insert(0, "../")
from lib import *
from r85 import holm, tostp, Q, E, lat1, lat4
from scipy import stats
SP = os.environ["R85_SP"]; VARS = ["q8", "t4", "t3", "t4q8"]
recs = []; off = 0
for tag in "AB":
    W = RealWorld(tag); n = len(Q[tag]["q"]); ids = list(range(n)); random.Random(85).shuffle(ids); halves = {"H1": ids[:n // 2], "H2": ids[n // 2:]}
    res = {}
    for v in ["fp32"] + VARS:
        dv = W.dvec if v == "fp32" else np.load(f"{SP}/D_{v}_{tag}.npy"); qv = E[v][off:off + n]
        for mode, (arm, al) in {"hyb": ("hybrid", .7), "dense": ("dense", 1.0)}.items():
            r = Retriever(W.docs, dv, arm, al); rr, h5 = [], []
            for i, (qid, tid) in enumerate(zip(Q[tag]["q"], Q[tag]["tid"])):
                rel = set(W.queries[qid]["rel"]); sc = r.scores(Q[tag]["text"][i], qv[i]); best = min(int((sc >= sc[d]).sum()) - 1 for d in rel); rr.append(1.0 / (1 + best)); h5.append(float(best < 5))
            res[(mode, v)] = (np.array(rr), np.array(h5))
    off += n
    for mode in ("hyb", "dense"):
        print(f"{tag} {mode} fp32 MRR {res[(mode,'fp32')][0].mean():.4f}  " + "  ".join(f"{v} {res[(mode,v)][0].mean():.4f}" for v in VARS))
        for v in VARS:
            for hn, hid in halves.items():
                for ax, k in (("mrr", 0), ("hit5", 1)):
                    a = res[(mode, v)][k][hid]; b = res[(mode, "fp32")][k][hid]; d = a - b; p = 1.0 if d.std() == 0 else float(stats.ttest_1samp(d, 0).pvalue)
                    recs.append(dict(corpus=tag, mode=mode, var=v, half=hn, ax=ax, base=float(b.mean()), cand=float(a.mean()), d=float(d.mean()), p=p, tost=tostp(d, .01) if d.std() > 0 else 0.0))
adj = holm([r["p"] for r in recs])
for r, a in zip(recs, adj): r["ph"] = a
for v in VARS:
    for mode in ("hyb", "dense"):
        sub = [r for r in recs if r["var"] == v and r["mode"] == mode]
        print(f"SYM {v:5s} {mode:5s} " + "  ".join(f"{r['corpus']}{r['half']}/{r['ax']} {r['d']:+.4f}{'*' if r['ph'] < .05 else ''}" for r in sub))
json.dump(recs, open(f"{RES}/r85b.json", "w"), default=float)
