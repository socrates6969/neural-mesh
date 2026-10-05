"""B4f: R1 (fixed .7 + BM25 (.9,.5)) vs R0 (library default) and shipped adaptive profile (R63) vs R0, per family, on the three item-disjoint folds."""
import numpy as np
from scipy import stats
from lib import holm
from b4c import TB, ai
from b4d import groups, vec
R0=groups(vec(TB,(1.2,.75),lambda f:.7))
for name,(p,rule) in {"R1 bm25p (alpha .7, BM25 .9/.5)":((.9,.5),lambda f:.7),"R63 profile (adaptive .3/1/2 + BM25 .9/.5)":((.9,.5),lambda f:1-(1-.3)*min(1,f["cov"])**2)}.items():
    C=groups(vec(TB,p,rule)); rows=[]
    for key in sorted(C):
        ids=sorted(C[key]); d=np.array([C[key][i]-R0[key][i] for i in ids]); nz=d[np.abs(d)>1e-12]
        pw=stats.wilcoxon(nz).pvalue if len(nz)>5 else 1.0
        rng=np.random.RandomState(1); bs=[d[rng.randint(0,len(d),len(d))].mean() for _ in range(2000)]
        rows.append((key,len(d),d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),int((nz>0).sum()),int((nz<0).sum()),pw))
    adj=holm([r[7] for r in rows]); print("==",name,"- R0 default (12 tests, Holm)")
    for r,h in zip(rows,adj): print(f"  {r[0][0]:6} {r[0][1]:6} n={r[1]:4} d {r[2]:+.4f} [{r[3]:+.4f},{r[4]:+.4f}] W/L {r[5]}/{r[6]} Holm {h:.2e}")
