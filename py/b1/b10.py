"""B10: one-parameter DOCUMENT-LENGTH PRIOR on the R1 fusion: score = .7 z(dense) + .3 z(bm25) + beta * (ln(1+doclen_tokens) - mean_ln).
Motivation: B9c ablation - a learned model with only (fused score, log doc length) captures most of the re-ranker's gain on S queries. Standard IR practice (length prior).
Protocol: beta grid {0,.05,.1,.15,.2,.3,.5,.75,1.0} pre-declared; CRATE-DISJOINT 3-fold CV (beta picked by mean(H,S) MRR on the two training folds, evaluated on the third) + the fixed beta=.2 reported; domain thirds as a no-regress check (beta applied unchanged; z over 144 docs).
Ties counted against the relevant doc. Per-query Wilcoxon, Holm over (fold x family)."""
import numpy as np, json
from lib import *
from b4lib import RealComp, DomainComp
from scipy import stats
p=(.9,.5); RC=RealComp(bmparams=(p,)); W=RC.W; bm=RC.bms[p]
DC=DomainComp(bmparams=(p,))
BETAS=[0,.05,.1,.15,.2,.3,.5,.75,1.0]
ll=np.log1p(bm.dl); llc=ll-ll.mean()
def rrt(sc,rel): return 1.0/(1+min(int((sc>=sc[d]).sum())-1 for d in rel))
CQ=json.load(open(f"{RES}/h10_corpus_A.json"))["queries"]; grp={q["id"]:q["grp"] for q in CQ}
crates=sorted(set(grp[q] for q in RC.q)); rg=np.random.RandomState(11); rg.shuffle(crates); cf={c:["C1","C2","C3"][i%3] for i,c in enumerate(crates)}
R={}  # q -> rr per beta
for q in RC.q:
    f=.7*RC.zD[q]+.3*RC.zB[p][q]
    R[q]=np.array([rrt(f+b*llc,RC.rel[q]) for b in BETAS])
folds={fn:[q for q in RC.q if cf[grp[q]]==fn] for fn in ("C1","C2","C3")}
print("pooled mean RR by beta:",{b:round(float(np.mean([R[q][i] for q in RC.q])),4) for i,b in enumerate(BETAS)})
for g in "HS": print(" fam",g,{b:round(float(np.mean([R[q][i] for q in RC.q if RC.fam[q]==g])),4) for i,b in enumerate(BETAS)})
fam=[]
for test,qs in folds.items():
    tr=[q for fn,l in folds.items() if fn!=test for q in l]
    sc=[np.mean([R[q][i] for q in tr]) for i in range(len(BETAS))]; bi=int(np.argmax(sc))
    for g in "HS":
        qq=[q for q in qs if RC.fam[q]==g]
        for label,i in (("cv-beta",bi),("beta.2",BETAS.index(.2))):
            d=np.array([R[q][i]-R[q][0] for q in qq]); nz=d[np.abs(d)>1e-12]
            pw=stats.wilcoxon(nz).pvalue if len(nz)>5 else 1.0
            rng=np.random.RandomState(1); bs=[d[rng.randint(0,len(d),len(d))].mean() for _ in range(2000)]
            fam.append((label,test,BETAS[i],g,len(d),float(np.mean([R[q][0] for q in qq])),d.mean(),np.percentile(bs,2.5),np.percentile(bs,97.5),int((nz>0).sum()),int((nz<0).sum()),pw))
adj=holm([f[11] for f in fam])
for f,h in zip(fam,adj): print(f"  {f[0]:8} test {f[1]} beta {f[2]:<4} fam {f[3]} n={f[4]:3} R1 {f[5]:.4f} d {f[6]:+.4f} [{f[7]:+.4f},{f[8]:+.4f}] W/L {f[9]}/{f[10]} Holm {h:.2e}")
# domain no-regress
dl=np.log1p(np.array([len(toks(d)) for d in DC.docs],float)); dlc=dl-dl.mean()
for w in "AB":
    for part in ("TUNE","HALF1","HALF2"):
        rows=[r for r in DC.rows if r[1]==w and r[5]==part]
        out=[]
        for b in (0,.1,.2,.3,.5):
            out.append(np.mean([rrt(.7*DC.zD[r[0]]+.3*DC.zB[p][r[0]]+b*dlc,[r[4]]) for r in rows]))
        print("domain",w,part,"beta 0/.1/.2/.3/.5:"," ".join(f"{x:.4f}" for x in out))
