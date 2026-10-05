"""B8: top-K local z-normalisation inside the hybrid (candidate from B7: K=300, cap 3 gave real +.02..+.035 on all three folds, domain B -.00..-.03).
Pre-declared grid, selection on TUNE only: K in {150,300,600,1000,2000}, cap in {none,3,5}, alpha in {.6,.7,.8}; criterion = mean(realH,realS gain vs R1) s.t. domain A/B gains >= -.003 on TUNE.
Verification on the item-disjoint HALF1/HALF2 vs R1 (and vs R0 default): per-item paired tests, Holm over the family. Local z uses mean/std of the channel's top-K scores."""
import numpy as np, pickle, itertools
from lib import *
from b4lib import RealComp, DomainComp, rr_of
from scipy import stats
p=(.9,.5)
RC=RealComp(bmparams=(p,(1.2,.75))); W=RC.W
DC=DomainComp(bmparams=(p,(1.2,.75)))
def zk(x,K,cap):
    if K>=len(x): m,s=x.mean(),x.std()+1e-9
    else:
        top=np.partition(x,-K)[-K:]; m,s=top.mean(),top.std()+1e-9
    r=(x-m)/s
    return r if cap is None else np.minimum(r,cap)
grid=[(K,cap,a) for K in (150,300,600,1000,2000) for cap in (None,3,5) for a in (.6,.7,.8)]
names=["R0","R1"]+[f"K{K}/cap{cap}/a{a}" for K,cap,a in grid]
def rrs(d,b,b0,rel):
    out=[rr_of(.7*z(d)+.3*z(b0),rel), rr_of(.7*z(d)+.3*z(b),rel)]
    for K,cap,a in grid: out.append(rr_of(a*zk(d,K,cap)+(1-a)*zk(b,K,cap),rel))
    return out
rows=[]
for q in RC.q:
    d=W.dvec@W.emb[RC.qid[q]]; b=RC.bms[p].score(RC.qt[q]); b0=RC.bms[(1.2,.75)].score(RC.qt[q])
    rows.append(("realH" if RC.fam[q]=="H" else "realS",RC.split[q],("r",q),rrs(d,b,b0,RC.rel[q])))
dom={}
for (rid,w,t,v,j,part) in DC.rows:
    d=DC.vecs@v; b=DC.bms[p].score(t); b0=DC.bms[(1.2,.75)].score(t)
    dom.setdefault(("dom"+w,part,j),[]).append(rrs(d,b,b0,[j]))
for (g,part,j),l in dom.items(): rows.append((g,part,("d",j),list(np.mean(l,axis=0))))
pickle.dump((names,rows),open(f"{RES}/b8_rows.pkl","wb"))
GR=["realH","realS","domA","domB"]
def gm(part,i): return {g:float(np.mean([r[3][i] for r in rows if r[0]==g and r[1]==part])) for g in GR}
b=gm("TUNE",1); res=[]
for i,n in enumerate(names[2:],2):
    t=gm("TUNE",i); g={k:t[k]-b[k] for k in GR}
    ok=min(g["domA"],g["domB"])>=-.003
    res.append((np.mean([g["realH"],g["realS"]]),n,g,ok))
res.sort(key=lambda x:-x[0])
print("TUNE base R1",{k:round(v,3) for k,v in b.items()}); 
for s,n,g,ok in res[:14]: print(f"  {n:18} {'OK ' if ok else 'dom-loss'} gains "+" ".join(f"{k}{v:+.3f}" for k,v in g.items()))
sel=[r for r in res if r[3]][:1]
print("SELECTED (TUNE):",sel[0][1] if sel else None)
if sel:
    i=names.index(sel[0][1])
    fam=[]
    for ref,ri in (("R1",1),("R0",0)):
        for g in GR:
            for part in ("HALF1","HALF2"):
                rr=[r for r in rows if r[0]==g and r[1]==part]
                d=np.array([r[3][i]-r[3][ri] for r in rr]); nz=d[np.abs(d)>1e-12]
                pw=stats.wilcoxon(nz).pvalue if len(nz)>5 else 1.0
                rng=np.random.RandomState(1); bs=[d[rng.randint(0,len(d),len(d))].mean() for _ in range(2000)]
                fam.append((ref,g,part,len(d),d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),int((nz>0).sum()),int((nz<0).sum()),pw))
    adj=holm([f[9] for f in fam])
    for f,h in zip(fam,adj): print(f"  vs {f[0]} {f[1]:6} {f[2]:6} n={f[3]:4} d {f[4]:+.4f} [{f[5]:+.4f},{f[6]:+.4f}] W/L {f[7]}/{f[8]} Holm {h:.2e}")
