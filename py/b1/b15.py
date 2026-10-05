"""B15: end-to-end check of the Rust shared-folder CORPUS sync (MeshStore::sync_docs_folder, int8 vectors) on the REAL 6,831-paragraph store.
4 machines each hold an independent random 40% of the docs (overlapping); `meshstore sync-docs` x2 rounds; machine 0 afterwards must hold EXACTLY the union of ids; its (dequantised int8) vectors
are compared to the exact f32 union: (a) id-set equality, (b) max / mean |cos error|, (c) retrieval: R1 hybrid (.7 / BM25 .9,.5) MRR on 500 queries from the query-disjoint halves, synced vs exact union,
paired per query, TOST +-.01 (non-inferiority) and top-10 id overlap. Folder bytes vs the f32 shards it replaces."""
import os, subprocess, json, shutil, numpy as np, tempfile
from lib import *
from b4lib import RealComp, rr_of
from scipy import stats
EXE=os.path.join(ROOT,"target","release","meshstore.exe")
RC=RealComp(bmparams=((.9,.5),)); W=RC.W; p=(.9,.5)
N=len(W.docs); dim=W.dvec.shape[1]
base=tempfile.mkdtemp(prefix="b15_"); folder=os.path.join(base,"folder"); NS="kb"
rng=np.random.RandomState(5); M=4; f=.4
masks=[rng.rand(N)<f for _ in range(M)]
def run(*a): 
    r=subprocess.run([EXE,*a],capture_output=True,text=True); 
    if r.returncode: raise RuntimeError(r.stderr)
    return r.stdout
roots=[]
for k in range(M):
    root=os.path.join(base,f"m{k}"); run("stats",root); d=os.path.join(root,"knowledge",NS); os.makedirs(d)
    idx=np.where(masks[k])[0]
    with open(os.path.join(d,"docs.jsonl"),"w",encoding="utf-8") as fh:
        for i in idx: fh.write(json.dumps({"id":f"d{i}","text":W.docs[i],"tag":"x"})+"\n")
    W.dvec[idx].astype("<f4").tofile(os.path.join(d,"vecs.f32")); json.dump({"dim":dim},open(os.path.join(d,"meta.json"),"w"))
    roots.append(root)
rep=[]
for rnd in range(2):
    for k in range(M): rep.append(json.loads(run("sync-docs",roots[k],folder,f"m{k}",NS)))
print("sync reports (last round):",rep[-M:])
uni=np.any(masks,axis=0); uidx=np.where(uni)[0]
docs=[json.loads(l) for l in open(os.path.join(roots[0],"knowledge",NS,"docs.jsonl"),encoding="utf-8")]
vec=np.fromfile(os.path.join(roots[0],"knowledge",NS,"vecs.f32"),dtype="<f4").reshape(-1,dim)[:len(docs)]
ids=[int(d["id"][1:]) for d in docs]
print("machine0 docs",len(docs),"union",len(uidx),"id-set equal:",set(ids)==set(uidx.tolist()), "dups:",len(ids)-len(set(ids)))
text_ok=all(W.docs[i]==d["text"] for i,d in zip(ids,docs)); print("texts intact:",text_ok)
ex=W.dvec[ids]; cos=(vec*ex).sum(1)/(np.linalg.norm(vec,axis=1)*np.linalg.norm(ex,axis=1)+1e-12)
own=np.where(masks[0])[0]; own_pos={i:j for j,i in enumerate(ids)}
exact_own=np.mean([cos[own_pos[i]] for i in own]) 
print("cos(dequantised, exact): min %.6f mean %.6f; own-origin docs (never re-quantised on machine 0) mean %.6f"%(cos.min(),cos.mean(),exact_own))
fb=sum(os.path.getsize(os.path.join(folder,x)) for x in os.listdir(folder)); f32=int(sum(m.sum() for m in masks)*dim*4)
print("folder bytes",fb,"vs f32 vectors alone for the same shards",f32,"(shards also carry text)")
# retrieval
rel_ok=lambda q:[d for d in RC.rel[q] if uni[d]]
qs=[q for q in RC.q if RC.split[q]!="TUNE" and rel_ok(q)]
rs=np.random.RandomState(3); qs=list(rs.choice(qs,500,replace=False))
def store_eval(docidx,vecs):
    pos={d:j for j,d in enumerate(docidx)}; bm=BM25([W.docs[i] for i in docidx],*p); out=[];tops=[]
    for q in qs:
        sc=.7*z(vecs@W.emb[RC.qid[q]])+.3*z(bm.score(RC.qt[q])); rel=[pos[d] for d in RC.rel[q] if d in pos]
        out.append(rr_of(sc,rel)); tops.append(set(np.array(docidx)[np.argsort(-sc)[:10]].tolist()))
    return np.array(out),tops
rx,tx=store_eval(uidx.tolist(),W.dvec[uidx]); rs_,ts=store_eval(ids,vec)
d=rs_-rx; nz=d[np.abs(d)>1e-12]
se=d.std(ddof=1)/np.sqrt(len(d)); lo=d.mean()-1.645*se; hi=d.mean()+1.645*se
print(f"MRR exact-union {rx.mean():.4f} synced(int8) {rs_.mean():.4f} d {d.mean():+.5f} 90%CI [{lo:+.5f},{hi:+.5f}] TOST(+-.01) {'PASS' if lo>-.01 and hi<.01 else 'FAIL'} W/L {int((nz>0).sum())}/{int((nz<0).sum())}")
print("mean top-10 overlap exact vs synced: %.4f ; identical top-10 sets: %d/%d"%(np.mean([len(a&b)/10 for a,b in zip(tx,ts)]),sum(a==b for a,b in zip(tx,ts)),len(qs)))
shutil.rmtree(base,ignore_errors=True)
