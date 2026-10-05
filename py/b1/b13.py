"""B13: soft SOURCE prior on the real store (docs carry their source crate/file as a tag; no query-side tag): score = R1 + g * z(cos(q, centroid(source(d)))), g pre-declared {0,.25,.5,1.0}; also a FILE-level tag.
Selection on TUNE queries only; verification on HALF1/HALF2 (item-disjoint) with Holm; ties against; per-query Wilcoxon."""
import numpy as np, json
from lib import *
from b4lib import RealComp, rr_of
from scipy import stats
p=(.9,.5); RC=RealComp(bmparams=(p,)); W=RC.W
C=json.load(open(f"{RES}/h10_corpus_A.json"))["docs"]
def centroids(key):
    tags=[d[key] if key!="file" else d["src"]+"/"+d["file"] for d in C]
    ids={t:i for i,t in enumerate(sorted(set(tags)))}; lab=np.array([ids[t] for t in tags])
    cen=np.zeros((len(ids),W.dvec.shape[1]),np.float32)
    np.add.at(cen,lab,W.dvec); cen/=np.linalg.norm(cen,axis=1,keepdims=True)+1e-9
    return lab,cen
GS=[0,.25,.5,1.0]; KEYS=["src","file"]
cen={k:centroids(k) for k in KEYS}
print({k:len(v[1]) for k,v in cen.items()})
R={}
for q in RC.q:
    f=.7*RC.zD[q]+.3*RC.zB[p][q]; qv=W.emb[RC.qid[q]]; row=[]
    for k in KEYS:
        lab,ce=cen[k]; pri=z((ce@qv)[lab])
        for g in GS: row.append(rr_of(f+g*pri,RC.rel[q]))
    R[q]=np.array(row)
cols=[(k,g) for k in KEYS for g in GS]
def mean(part,fam,c): return float(np.mean([R[q][c] for q in RC.q if RC.split[q]==part and RC.fam[q]==fam]))
for part in ("TUNE","HALF1","HALF2"):
    for fam in "HS":
        print(part,fam," ".join(f"{k}{g}:{mean(part,fam,i):.4f}" for i,(k,g) in enumerate(cols)))
fam=[]
for i,(k,g) in enumerate(cols):
    if g==0: continue
    for part in ("HALF1","HALF2"):
        for fm in "HS":
            qq=[q for q in RC.q if RC.split[q]==part and RC.fam[q]==fm]; b0=KEYS.index(k)*len(GS)
            d=np.array([R[q][i]-R[q][b0] for q in qq]); nz=d[np.abs(d)>1e-12]
            pw=stats.wilcoxon(nz).pvalue if len(nz)>5 else 1.0
            fam.append((k,g,part,fm,len(d),d.mean(),int((nz>0).sum()),int((nz<0).sum()),pw))
adj=holm([f[8] for f in fam])
for f,h in zip(fam,adj): print(f"  {f[0]:4} g={f[1]:<4} {f[2]} fam {f[3]} n={f[4]} d {f[5]:+.4f} W/L {f[6]}/{f[7]} Holm {h:.1e}")
