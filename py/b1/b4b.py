import pickle, itertools, numpy as np, sys
from lib import RES
zD,zB,cov,qt,fam,rel,split=pickle.load(open(f"{RES}/b4_real.pkl","rb"))
dzD,dzB,dcov,drows=pickle.load(open(f"{RES}/b4_dom.pkl","rb"))
AL=np.round(np.arange(0,1.0001,.05),2)
def rr(sc,rl): 
    b=min(int((sc>=sc[d]).sum())-1 for d in rl); return 1.0/(1+b)
def build(p):
    T={}  # key -> list of (feat dict, rr vector)
    for q in zD:
        v=np.array([rr(a*zD[q]+(1-a)*zB[p][q],rel[q]) for a in AL])
        f=dict(mD=float(zD[q].max()), mB=float(zB[p][q].max()), cov=cov[p][q], ntok=len(qt[q].split()),
               sD=float(np.sort(zD[q])[-1]-np.sort(zD[q])[-2]), sB=float(np.sort(zB[p][q])[-1]-np.sort(zB[p][q])[-2]),
               agree=float(zD[q].argmax()==zB[p][q].argmax()))
        T[("real",split[q],fam[q],q)]=(f,v)
    for (rid,w,t,vv,j,part) in drows:
        v=np.array([rr(a*dzD[rid]+(1-a)*dzB[p][rid],[j]) for a in AL])
        f=dict(mD=float(dzD[rid].max()), mB=float(dzB[p][rid].max()), cov=dcov[p][rid], ntok=len(t.split()),
               sD=float(np.sort(dzD[rid])[-1]-np.sort(dzD[rid])[-2]), sB=float(np.sort(dzB[p][rid])[-1]-np.sort(dzB[p][rid])[-2]),
               agree=float(dzD[rid].argmax()==dzB[p][rid].argmax()))
        T[("dom"+w,part,"",rid)]=(f,v)
    return T
if __name__=="__main__":
    out={}
    for p in [(1.2,.75),(.9,.5)]:
        out[p]=build(p); print(p,len(out[p]))
    pickle.dump(out,open(f"{RES}/b4_table.pkl","wb"))
