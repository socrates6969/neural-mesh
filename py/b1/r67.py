"""R67: PRE-REGISTERED blind retest of the verbal recipes on a freshly authored pool (data_verbal3.py, authored before any scorer was run on it).
Hypotheses (declared in RD-LOG before running): H1 vocab BGE>MiniLM; H2 analogies BGE>MiniLM; H3 cloze BGE+mix > MiniLM plain; H4 cloze MiniLM+mix > MiniLM plain; H5 cloze BGE plain vs MiniLM (expect loss).
One-sided-direction tests via exact McNemar (two-sided p reported), Holm over 5."""
import sys; sys.path.insert(0, "../")
import random, itertools, numpy as np
from scipy import stats
import data_verbal3 as V
import v2, r49
from r66 import ARMS
from lib import holm
r = random.Random(67)
vocab = []
for wd, syn, ant in V.VOCAB3:
    others = [x[1] for x in V.VOCAB3 if x[0] != wd]; opts = [syn, ant] + r.sample(others, 2); r.shuffle(opts); vocab.append(dict(t=wd, opts=opts, ans=opts.index(syn)))
cloze = []
for st, an, ds in V.CLOZE3:
    opts = [an] + ds; r.shuffle(opts); cloze.append(dict(st=st, opts=opts, ans=opts.index(an)))
ana = []
for rn, pairs in V.REL3.items():
    for _ in range(30):
        P = r.sample(pairs, 4); (a, b), (c, d) = P[0], P[1]; others = [P[2][1], P[3][1]]
        rest = [p[1] for p in pairs if p[1] not in (b, d) and p[1] not in others]; opts = [d] + others + [r.choice(rest)]; r.shuffle(opts)
        ana.append(dict(abc=[a, b, c], opts=opts, ans=opts.index(d)))
SK = {"vocab": vocab, "analogies": ana, "cloze": cloze}
ts = [t for sk, L in SK.items() for x in L for t in v2.base_texts(sk, x)]
v2.embed_many(ts); r49.bembed_many(ts)
H = [("H1", "vocab", "bge", "mini"), ("H2", "analogies", "bge", "mini"), ("H3", "cloze", "bge+mix", "mini"), ("H4", "cloze", "mini+mix", "mini"), ("H5", "cloze", "bge", "mini")]
res = []
for h, sk, cand, base in H:
    X = SK[sk]; Cc = np.array([ARMS[cand](sk, x) == x["ans"] for x in X]); Cb = np.array([ARMS[base](sk, x) == x["ans"] for x in X])
    b = int((Cc & ~Cb).sum()); c = int((~Cc & Cb).sum()); p = stats.binomtest(b, b + c, .5).pvalue if b + c else 1.0
    res.append((h, sk, cand, base, len(X), Cc.mean(), Cb.mean(), b, c, p))
adj = holm([x[-1] for x in res])
for x, a in zip(res, adj): print(f"{x[0]} {x[1]:9s} n={x[4]:3d} {x[2]:9s} {x[5]:.3f} vs {x[3]:5s} {x[6]:.3f}  d={x[5]-x[6]:+.3f} W/L {x[7]}/{x[8]} p={x[9]:.3g} holm={a:.3g}")
for sk in SK: print(sk, {a: round(float(np.mean([ARMS[a](sk, x) == x["ans"] for x in SK[sk]])), 3) for a in ARMS})
