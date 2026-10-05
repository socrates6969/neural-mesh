"""B4e: 3-fold cross-validated selection of an alpha rule (folds = TUNE, HALF1, HALF2; items disjoint). Tune on two folds, test on the third.
Selection objective: mean group gain vs R1 (fixed .7 + BM25(.9,.5)) over groups {realH, realS, domA, domB}, constraint min(realH, realS) gain >= -.001 on the tuning folds."""
import pickle, numpy as np
from b4c import *
p=(.9,.5); T=TB[p]
_,_,_=None,None,None
import b4c
rules,base,out=b4c.table(p)
PARTS=["TUNE","HALF1","HALF2"]
def ev(rule,parts):
    res={g:[] for g in GR}
    for k,(f,v) in T.items():
        if k[1] in parts: res[grp(k)].append(v[ai(rule(f))])
    return {g:float(np.mean(x)) for g,x in res.items()}
fix=rules["fix0.7"]
for test in PARTS:
    tr=[x for x in PARTS if x!=test]
    b_tr=ev(fix,tr); b_te=ev(fix,[test])
    best=None
    for name,r in rules.items():
        t=ev(r,tr); g={k:t[k]-b_tr[k] for k in GR}
        if min(g["realH"],g["realS"])<-.001: continue
        sc=np.mean(list(g.values()))
        if best is None or sc>best[0]: best=(sc,name,r,g)
    sc,name,r,g=best
    t=ev(r,[test]); gt={k:t[k]-b_te[k] for k in GR}
    print(f"test={test}: chosen on {tr}: {name} (tune gains "+" ".join(f"{k}{v:+.3f}" for k,v in g.items())+") -> TEST gains "+" ".join(f"{k}{v:+.3f}" for k,v in gt.items()))
    ex=rules["adapt(0.3,1.0,2)"] if "adapt(0.3,1.0,2)" in rules else None
    if ex: 
        t=ev(ex,[test]); print("    shipped adaptive on this fold: "+" ".join(f"{k}{t[k]-b_te[k]:+.3f}" for k in GR))
