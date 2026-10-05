"""B38: SHARD-AFFINITY CALIBRATION of the fused score in a union of specialist shards (B26/B27 follow-up; NOT a local-shard gate: it uses the query's affinity to EACH contributing shard).
Rule PRE-DECLARED before any run:
  aff_s(q) = mean of the top-3 raw dense cosines of q within shard s (store-independent). bonus(doc in shard s) = lam * (aff_s - max_s' aff_s').  (<=0; the best-matching shard is untouched)
  fused' = fused_union + bonus ; fused_union = .7 z(cos) + .3 z(BM25) on the union (R1, as B26).
  lam in {0,2,5,10,20,40} chosen on TUNE queries per (partition kind, M) (max mean MRR(all)), averaged over 8 salts; FIXED, then evaluated on HALF1 (held-out) and HALF2 (confirm).
  Comparators: ungated union (lam=0). Primary metric: MRR(all) paired per-query (mean over salts), Wilcoxon, Holm over 2 kinds x 2 M(2,4) x 2 splits = 8 tests.
  Secondary: the distractor cost on queries answerable in the own (machine 0) store. Tie-pessimistic ranks (>= minus self)."""
import json, hashlib, numpy as np
from scipy.stats import wilcoxon
from lib import *
from b4lib import RealComp
RC=RealComp(bmparams=((.9,.5),)); p=(.9,.5); W=RC.W
C=json.load(open(f"{RES}/h10_corpus_A.json"))["docs"]; src=[d["src"] for d in C]; N=len(W.docs)
rng=np.random.RandomState(11); S={}
for sp in ("TUNE","HALF1","HALF2"):
    L=[q for q in RC.q if RC.split[q]==sp]; S[sp]=list(rng.choice(L,min(400,len(L)),replace=False))
LAMS=[0,2,5,10,20,40]
def assign(kind,salt):
    if kind=="random": return np.random.RandomState(salt).randint(0,4,N)
    return np.array([int(hashlib.md5(f"{salt}:{s}".encode()).hexdigest(),16)%4 for s in src])
def run(m,M,qlist):
    idx=np.where(m<M)[0]; pos={d:i for i,d in enumerate(idx)}; vecs=W.dvec[idx]; bm=BM25([W.docs[i] for i in idx],*p); shard=m[idx]
    own=(shard==0); out=np.zeros((len(LAMS),len(qlist))); ans=np.zeros(len(qlist),bool)
    for j,q in enumerate(qlist):
        rel=[pos[d] for d in RC.rel[q] if d in pos]
        if not rel: continue
        cs=vecs@W.emb[RC.qid[q]]; base=.7*z(cs)+.3*z(bm.score(RC.qt[q]))
        aff=np.array([np.sort(cs[shard==s])[-3:].mean() if (shard==s).any() else -9 for s in range(M)]); b=(aff-aff.max())[shard]
        for k,lam in enumerate(LAMS):
            sc=base+lam*b; out[k,j]=1.0/(1+min(int((sc>=sc[d]).sum())-1 for d in rel))
        ans[j]=any(own[r] for r in rel)
    return out,ans
res={}
for kind in ("random","clustered"):
  for M in (2,4):
    for sp in S:
        R=[];A=[]
        for salt in range(8):
            o,a=run(assign(kind,salt),M,S[sp]); R.append(o); A.append(a)
        res[(kind,M,sp)]=(np.array(R),np.array(A))  # [salt,lam,q]
    print("done",kind,M,flush=True)
rows=[];ps=[]
for kind in ("random","clustered"):
  for M in (2,4):
    R,_=res[(kind,M,"TUNE")]; tm=R.mean((0,2)); k=int(np.argmax(tm)); lam=LAMS[k]
    print(f"{kind} M={M}: TUNE curve {[round(x,4) for x in tm]} -> lam={lam}",flush=True)
    for sp in ("HALF1","HALF2"):
        R,A=res[(kind,M,sp)]; d=(R[:,k]-R[:,0]).mean(0); nz=d[d!=0]; pv=wilcoxon(nz).pvalue if len(nz)>5 else 1.0
        ci=(d.mean()-1.96*d.std(ddof=1)/np.sqrt(len(d)),d.mean()+1.96*d.std(ddof=1)/np.sqrt(len(d)))
        # own-answerable cost: per salt mean over answerable queries
        ca=[(R[s,k][A[s]]-R[s,0][A[s]]).mean() for s in range(8) if A[s].any()]
        print(f"  {sp}: union {R[:,0].mean():.4f} cal {R[:,k].mean():.4f} d {d.mean():+.4f} CI[{ci[0]:+.4f},{ci[1]:+.4f}] W/L {(d>0).sum()}/{(d<0).sum()} p={pv:.3g} | on own-answerable cal-union {np.mean(ca):+.4f}",flush=True)
        rows.append((kind,M,sp,d.mean(),pv)); ps.append(pv)
order=np.argsort(ps); m_=len(ps); run_=0; adj={}
for r,i in enumerate(order): run_=max(run_,(m_-r)*ps[i]); adj[i]=min(1,run_)
print("Holm:")
for i,(k_,M,sp,dm,pv) in enumerate(rows): print(f"  {k_} M={M} {sp}: d {dm:+.4f} raw p {pv:.3g} Holm {adj[i]:.3g}")
