"""B7: fusion-normalisation variants for the hybrid on the REAL store (item-disjoint folds TUNE / HALF1 / HALF2, families H,S) + domain thirds.
Baseline = R1: .7*z(dense)+.3*z(bm25) with BM25 (.9,.5), z over ALL docs. Candidates (pre-declared): 
 localz(K): z computed over each channel's top-K candidates only (K=100,300), docs outside both tops get the min;
 maxnorm(l): cos + l*bm25/max(bm25)  (l in .1,.2,.3,.5);  
 sigm: z-scores squashed through tanh(z/3) (limits single-channel outliers).
Selection on TUNE (mean of group gains with realH/realS gain >= -.001), verify on HALF1/HALF2 (Holm)."""
import numpy as np, pickle, itertools
from lib import *
from b4lib import RealComp, DomainComp, rr_of
from scipy import stats
p=(.9,.5)
RC=RealComp(bmparams=(p,)); W=RC.W
DC=DomainComp(bmparams=(p,))
def raw_real(q):
    d=W.dvec@W.emb[RC.qid[q]]; b=RC.bms[p].score(RC.qt[q]); return d,b
def raw_dom(row):
    rid,w,t,v,j,part=row; d=DC.vecs@v; b=DC.bms[p].score(t); return d,b
def zloc(x,K):
    top=np.sort(x)[-K:]; m,s=top.mean(),top.std()+1e-9
    return np.minimum((x-m)/s, 3.0)
cands={"base":lambda d,b:.7*z(d)+.3*z(b)}
for K in (100,300): 
    cands[f"localz{K}"]=(lambda K:lambda d,b:.7*zloc(d,K)+.3*zloc(b,K))(K)
for l in (.1,.2,.3,.5,1.0): cands[f"maxnorm{l}"]=(lambda l:lambda d,b: d+l*b/(b.max()+1e-9))(l)
cands["tanh"]=lambda d,b:.7*np.tanh(z(d)/3)+.3*np.tanh(z(b)/3)
for a in (.6,.8): cands[f"base{a}"]=(lambda a:lambda d,b:a*z(d)+(1-a)*z(b))(a)
rows=[]  # (group,part,item,{cand:rr})
for q in RC.q:
    d,b=raw_real(q); g="realH" if RC.fam[q]=="H" else "realS"
    rows.append((g,RC.split[q],q,{n:rr_of(f(d,b),RC.rel[q]) for n,f in cands.items()}))
dom={}
for row in DC.rows:
    d,b=raw_dom(row); item=row[4]; g="dom"+row[1]
    dom.setdefault((g,row[5],item),[]).append({n:rr_of(f(d,b),[row[4]]) for n,f in cands.items()})
for (g,part,item),lst in dom.items(): rows.append((g,part,("d",item),{n:float(np.mean([x[n] for x in lst])) for n in cands}))
pickle.dump(rows,open(f"{RES}/b7_rows.pkl","wb"))
GR=["realH","realS","domA","domB"]
def gm(part,n): return {g:float(np.mean([r[3][n] for r in rows if r[0]==g and r[1]==part])) for g in GR}
for part in ("TUNE","HALF1","HALF2"):
    b=gm(part,"base"); print(part,"base",{g:round(v,3) for g,v in b.items()})
    for n in cands:
        if n=="base": continue
        t=gm(part,n); print(f"   {n:10} gains "+" ".join(f"{g}{t[g]-b[g]:+.3f}" for g in GR))
