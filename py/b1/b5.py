"""B5: retrieval value of SHARING a doc corpus across machines (real h10 store, R1 retrieval: fixed alpha .7, BM25 .9/.5; query-disjoint HALF1/HALF2 from b4lib).
Machine m holds a random fraction f of the 6,831 docs (independent draws, so union coverage = 1-(1-f)^M). We evaluate on queries (H+S) from the disjoint halves:
 (a) MRR over ALL queries (a query whose relevant docs are all absent scores 0)  = what a user of that store gets;
 (b) MRR CONDITIONAL on the query having >=1 relevant doc present in the evaluated store = the 'distractor' effect of adding foreign docs.
Arms: own (machine 0 only) vs union of M machines. Honest expectation: (a) rises with coverage with diminishing returns; (b) is the check that foreign docs do not hurt."""
import pickle, numpy as np
from lib import *
from b4lib import RealComp
RC = RealComp(bmparams=((.9,.5),))
p=(.9,.5); W=RC.W
qs=[q for q in RC.q if RC.split[q]!="TUNE"]
print("queries",len(qs))
# precompute per-query raw fused score vector over all docs is NOT subset-invariant (z-scoring uses the whole corpus): recompute z on the subset
def eval_store(mask, qlist):
    idx=np.where(mask)[0]; pos={d:i for i,d in enumerate(idx)}
    vecs=W.dvec[idx]; docs=[W.docs[i] for i in idx]; bm=BM25(docs,*p)
    out=[]
    for q in qlist:
        rel=[pos[d] for d in RC.rel[q] if d in pos]
        tid=None
        sc=.7*z(vecs@W.emb[RC.qid[q]])+.3*z(bm.score(RC.qt[q]))
        if not rel: out.append(None); continue
        out.append(1.0/(1+min(int((sc>=sc[d]).sum())-1 for d in rel)))
    return out
rng=np.random.RandomState(7)
qs=list(rng.choice(qs,500,replace=False))
N=len(W.docs)
for f in (.25,.5):
    for M in (1,2,3,4,8):
        A=[];U=[];S=[];C=[]
        for draw in range(8):
            r=np.random.RandomState(1000*draw+int(f*100))
            masks=[r.rand(N)<f for _ in range(8)]
            own=masks[0]; uni=np.any(masks[:M],axis=0)
            x=eval_store(own,qs); y=eval_store(uni,qs)
            A.append(np.mean([0 if v is None else v for v in x])); U.append(np.mean([0 if v is None else v for v in y]))
            both=[(a,b) for a,b in zip(x,y) if a is not None]   # queries answerable in OWN store: pure distractor effect
            S.append(np.mean([b-a for a,b in both])); C.append(np.mean([a for a,b in both]))
        d=np.array(U)-np.array(A)
        print(f"f={f} M={M}: MRR(all) own {np.mean(A):.4f} union {np.mean(U):.4f} d {d.mean():+.4f} (sd {d.std(ddof=1):.4f}) | on queries answerable in OWN store: own MRR {np.mean(C):.4f}, union-own {np.mean(S):+.4f} (sd {np.std(S,ddof=1):.4f}, {int(np.sum(np.array(S)<0))}/8 draws negative)",flush=True)
