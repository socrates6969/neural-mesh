"""k4-1 report: Pareto frontier (MRR / hit@8 vs scan cost) of the two-stage configs on A held (first 10 seeds) + a single-thread latency micro-benchmark
of flat vs two-stage (cluster-contiguous arrays).  Reads results/k4_taxo.json.  Writes results/k4_taxo_pareto.txt"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
from k4_common import *
from k4_taxo import Taxo, K_REL

R = json.load(open(os.path.join(RES, "k4_taxo.json")))
L = []
for wn in ("A", "pydoc9", "crates8"):
    pa = R["phases"]["held|%s" % wn]["pareto10"]
    flat = [p for p in pa if p[0] == "flat"][0]
    L.append("== held|%s (first 10 seeds): flat mrr %.4f hit8 %.4f" % (wn, flat[2], flat[3]))
    for kind in ("ts|", "tree|"):
        for tau_zero in (True, False):
            sub = [p for p in pa if p[0].startswith(kind) and (p[0].endswith("t0.00") == tau_zero)]
            for cap in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9):
                c = [p for p in sub if p[1] <= cap]
                if c:
                    b = max(c, key=lambda p: p[2])
                    L.append("  %-5s %-12s cost<=%.2f best: %-44s cost %.3f mrr %.4f (%+.4f) hit8 %.4f (%+.4f)" % (kind, "no-fallback" if tau_zero else "fallback-gated", cap, b[0], b[1], b[2], b[2] - flat[2], b[3], b[3] - flat[3]))
# latency micro-benchmark on A: flat hybrid vs two-stage (k1.0 cent m=2, no fallback) with cluster-contiguous arrays, single thread
w = H10World("A"); st = w.store
tx = Taxo(st, int(round(np.sqrt(st.n))))
q = w.qset(31)
mem = tx.members
Dc = [st.D[m] for m in mem]
Wc = [st.W[m] for m in mem]


def flat(i):
    e = q.E[i:i + 1]; b = q.B[i:i + 1]
    d = e @ st.D.T; bm = (b @ st.W.T).toarray()
    return top_idx(ALPHA * zrows(d) + (1 - ALPHA) * zrows(bm), 10)


def two(i, m=2):
    e = q.E[i:i + 1]; b = q.B[i:i + 1]
    s = (e @ tx.cent.T)[0]
    sel = np.argsort(-s)[:m]
    ids = np.concatenate([mem[c] for c in sel])
    d = np.concatenate([e @ Dc[c].T for c in sel], 1)
    bm = np.concatenate([(b @ Wc[c].T).toarray() for c in sel], 1)
    f = ALPHA * zrows(d) + (1 - ALPHA) * zrows(bm)
    return ids[top_idx(f, 10)[0]]


for fn in (flat, two):
    for i in range(20):
        fn(i)
    ts = []
    for rep in range(5):
        t0 = time.perf_counter()
        for i in range(q.n):
            fn(i)
        ts.append((time.perf_counter() - t0) / q.n * 1e6)
    L.append("latency %-5s : %.0f us/query (min over 5 reps; single thread; box noisy +-50%%), mean docs scanned flat %d, two-stage m=2 k=%d: %.0f" % (fn.__name__, min(ts), st.n, tx.k, np.mean([sum(len(mem[c]) for c in np.argsort(-(q.E[i] @ tx.cent.T))[:2]) for i in range(q.n)])))
open(os.path.join(RES, "k4_taxo_pareto.txt"), "w").write("\n".join(L))
print("\n".join(L))
