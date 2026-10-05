"""B4 verification (pre-declared): candidate = adaptive alpha (lo .3, hi .9, pw 2) with BM25 (.9,.5). Chosen on TUNE only (b4c.py).
References: R0 = lib default (fixed .7, BM25 1.2/.75); R1 = fixed .7 + BM25 (.9,.5) ('bm25p', R54); R2 = shipped adaptive (.3,1,2)+BM25(.9,.5) ('profile', R63).
Tests per item (domain B: mean over its 3 typo draws), on the two query/item-DISJOINT halves; Holm over all tests of the candidate; loss counts if Holm p<.05 and |d|>.002."""
import pickle, numpy as np
from scipy import stats
from lib import RES, holm
from b4c import TB, ai, AL
def cand_alpha(f,hi=.9,lo=.3,pw=2): return hi-(hi-lo)*min(1,f["cov"])**pw
def vec(T,p,rule):
    out={}
    for k,(f,v) in T[p].items():
        out[k]=v[ai(rule(f))]
    return out
def groups(vv):
    g={}
    for k,x in vv.items():
        grp={"H":"realH","S":"realS"}.get(k[2]) if k[0]=="real" else k[0]
        key=(grp,k[1]); 
        if k[0]=="domB":
            item=int(''.join(c for c in k[3].split('B')[0] if c.isdigit())); g.setdefault(key,{}).setdefault(item,[]).append(x)
        else: g.setdefault(key,{})[k[3]]=[x]
    return {k:{i:float(np.mean(v)) for i,v in d.items()} for k,d in g.items()}
refs={"R0 default":("(1.2, 0.75)",lambda f:.7),"R1 bm25p":("(0.9, 0.5)",lambda f:.7),"R2 profile":("(0.9, 0.5)",lambda f:1-(1-.3)*min(1,f["cov"])**2)}
refs={"R0 default":((1.2,.75),lambda f:.7),"R1 bm25p":((.9,.5),lambda f:.7),"R2 profile":((.9,.5),lambda f:1-(1-.3)*min(1,f["cov"])**2)}
C=groups(vec(TB,(.9,.5),cand_alpha))
for rn,(p,rule) in refs.items():
    Rg=groups(vec(TB,p,rule)); rows=[]
    for (g,part) in sorted(C):
        if part=="TUNE": continue
        ids=sorted(C[(g,part)]); a=np.array([C[(g,part)][i] for i in ids]); b=np.array([Rg[(g,part)][i] for i in ids]); d=a-b
        nz=d[np.abs(d)>1e-12]; w=int((nz>0).sum()); l=int((nz<0).sum())
        pw=stats.wilcoxon(nz).pvalue if len(nz)>5 else 1.0
        rng=np.random.RandomState(1); bs=[d[rng.randint(0,len(d),len(d))].mean() for _ in range(2000)]
        rows.append((g,part,len(d),d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),w,l,pw,a.mean(),b.mean()))
    adj=holm([r[8] for r in rows])
    print("== candidate adapt(.3,.9,2)+BM25(.9,.5) vs",rn)
    for r,h in zip(rows,adj):
        print(f"  {r[0]:6} {r[1]:6} n={r[2]:4} cand {r[9]:.4f} ref {r[10]:.4f} d {r[3]:+.4f} [{r[4]:+.4f},{r[5]:+.4f}] W/L {r[6]}/{r[7]} Holm {h:.2e}")
