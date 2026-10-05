"""R91: export corpus A doc vectors + 400 real query vectors as raw f32 for `n10_kernels real` (results/n10_docs_A.f32, n10_queries_A.f32; not committed: 10 MB)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
W = RealWorld("A"); qm = {}
for s, v in W.inst.items():
    if s == "R": continue
    for row in v["rows"]: qm[row["q"]] = row["t"][0][0]
tids = [qm[q] for q in sorted(qm)][:400]
W.dvec.astype(np.float32).tofile(os.path.join(RES, "n10_docs_A.f32")); W.emb[tids].astype(np.float32).tofile(os.path.join(RES, "n10_queries_A.f32")); print(W.dvec.shape, len(tids))
