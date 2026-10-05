"""B9b (CRATE-DISJOINT folds C1-C3: all queries whose source crate is in the fold; stricter than query-disjoint): learned re-ranker ('feedback-trained retrieval') on the REAL store. Candidates = top-100 of R1 fusion (.7/.3, BM25 .9/.5). Pointwise model (HistGradientBoosting and logistic regression)
over per-(query,doc) features: fused z, dense z, bm25 z, raw cosine, bm25/max, fused/dense/bm25 ranks, log doc length, query length, query-token coverage by the doc, doc-token precision.
3-fold CROSS-VALIDATION over the query-disjoint folds (TUNE 365 q, HALF1 ~787, HALF2 ~787; train on two, test on the third), so every query is scored by a model that never saw it
(or any query of its fold). Metric = reciprocal rank (ties against), per-query paired Wilcoxon vs R1, Holm over (fold x family) tests. Relevant docs outside the top-100 keep their R1 rank (no gain)."""
import numpy as np, pickle
from lib import *
from b4lib import RealComp
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
p=(.9,.5); RC=RealComp(bmparams=(p,)); W=RC.W; bm=RC.bms[p]
K=100
docs_tok=[set(toks(d)) for d in W.docs]; dlen=np.log1p(bm.dl)
def feats(q):
    d=W.dvec@W.emb[RC.qid[q]]; b=bm.score(RC.qt[q]); zd,zb=z(d),z(b); f=.7*zd+.3*zb
    top=np.argsort(-f)[:K]
    rd=np.empty(len(f),int); rd[np.argsort(-d)]=np.arange(len(f)); rb=np.empty(len(f),int); rb[np.argsort(-b)]=np.arange(len(f))
    qt=set(toks(RC.qt[q])); nq=max(1,len(qt)); bmax=b.max()+1e-9
    X=[]
    for r,i in enumerate(top):
        inter=len(qt&docs_tok[i])
        X.append([f[i],zd[i],zb[i],d[i],b[i]/bmax,r,rd[i],rb[i],dlen[i],nq,inter/nq,inter/max(1,len(docs_tok[i]))])
    full_rank={int(i):r for r,i in enumerate(np.argsort(-f))}
    return top,np.array(X,np.float32),f
data={}
for q in RC.q:
    top,X,f=feats(q); rel=set(RC.rel[q]); y=np.array([int(i in rel) for i in top])
    rrbase=1.0/(1+min(int((f>=f[d]).sum())-1 for d in rel))
    data[q]=(top,X,y,rrbase,rel)
print("queries",len(data), "frac with rel in top100", np.mean([d[2].sum()>0 for d in data.values()]))
import zlib, json
CQ=json.load(open(f"{RES}/h10_corpus_A.json"))["queries"]
grp={q["id"]:q["grp"] for q in CQ}
crates=sorted(set(grp[q] for q in RC.q)); rg=np.random.RandomState(11); rg.shuffle(crates)
cf={c:["C1","C2","C3"][i%3] for i,c in enumerate(crates)}
folds={fn:[q for q in RC.q if cf[grp[q]]==fn] for fn in ("C1","C2","C3")}
print({k:len(v) for k,v in folds.items()}, "crates",len(crates))
def fit(model,train):
    X=np.vstack([data[q][1] for q in train]); y=np.concatenate([data[q][2] for q in train])
    if model=="hgb": m=HistGradientBoostingClassifier(max_depth=3,learning_rate=.05,max_iter=200,random_state=0)
    else:
        mu,sd=X.mean(0),X.std(0)+1e-9; m=LogisticRegression(C=1.0,max_iter=3000); m.fit((X-mu)/sd,y); m.mu,m.sd=mu,sd; return m
    return m.fit(X,y)
def score(m,model,q):
    X=data[q][1]; return m.predict_proba(X if model=="hgb" else (X-m.mu)/m.sd)[:,1]
def rr_new(m,model,q):
    top,X,y,rrb,rel=data[q]
    if y.sum()==0: return rrb
    s=score(m,model,q); order=np.argsort(-s,kind="stable"); 
    # ties against: rank = count(s>=s_rel)-1 among candidates
    best=min(int((s>=s[j]).sum())-1 for j in range(len(top)) if y[j])
    return 1.0/(1+best)

names=["fused","zd","zb","cos","bm/max","rank_f","rank_d","rank_b","loglen","nq","q_cov","doc_prec"]
subsets={"fused only":[0],"fused+loglen":[0,8],"fused+ranks":[0,5,6,7],"fused+loglen+ranks":[0,5,6,7,8],"scores(zd,zb,cos,bm/max)":[1,2,3,4],"scores+loglen":[1,2,3,4,8],"all":list(range(12))}
def run(cols):
    out={}
    for test in folds:
        train=[q for fn,qs in folds.items() if fn!=test for q in qs]
        X=np.vstack([data[q][1][:,cols] for q in train]); y=np.concatenate([data[q][2] for q in train]); mu,sd=X.mean(0),X.std(0)+1e-9
        m=LogisticRegression(C=1.0,max_iter=3000).fit((X-mu)/sd,y)
        for q in folds[test]:
            top,Xq,yq,rrb,rel=data[q]
            if yq.sum()==0: out[q]=rrb; continue
            s_=m.predict_proba((Xq[:,cols]-mu)/sd)[:,1]
            out[q]=1.0/(1+min(int((s_>=s_[j]).sum())-1 for j in range(len(top)) if yq[j]))
    return out
for nm,cols in subsets.items():
    r=run(cols); row=[]
    for g in "HS":
        qq=[q for q in RC.q if RC.fam[q]==g]; a=np.array([r[q] for q in qq]); b=np.array([data[q][3] for q in qq]); d=a-b
        nz=d[np.abs(d)>1e-12]; row.append(f"{g}: d {d.mean():+.4f} W/L {int((nz>0).sum())}/{int((nz<0).sum())} p {stats.wilcoxon(nz).pvalue:.1e}")
    print(f"{nm:28}"," | ".join(row))
