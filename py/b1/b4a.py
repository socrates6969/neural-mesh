import pickle, time
from b4lib import *
t=time.time()
RC=RealComp(); DC=DomainComp()
print("built", time.time()-t)
pickle.dump((RC.zD,RC.zB,RC.cov,RC.qt,RC.fam,RC.rel,RC.split),open(f"{RES}/b4_real.pkl","wb"))
pickle.dump((DC.zD,DC.zB,DC.cov,DC.rows),open(f"{RES}/b4_dom.pkl","wb"))
alphas=[0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]
P=(.9,.5)
def groups_real():
    g={}
    for q in RC.q:
        g.setdefault((RC.split[q],RC.fam[q]),[]).append(q)
    return g
G=groups_real()
for p in [(1.2,.75),(.9,.5)]:
    print("BM25",p)
    for key,qs in sorted(G.items()):
        row=[]
        for a in alphas:
            row.append(np.mean([RC.rr(q,a*RC.zD[q]+(1-a)*RC.zB[p][q]) for q in qs]))
        orc=np.mean([max(RC.rr(q,a*RC.zD[q]+(1-a)*RC.zB[p][q]) for a in alphas) for q in qs])
        ad=np.mean([RC.rr(q,(1-(1-.3)*min(1,RC.cov[p][q])**2)*RC.zD[q]+(1-(1-(1-.3)*min(1,RC.cov[p][q])**2))*RC.zB[p][q]) for q in qs])
        print(key,len(qs)," ".join(f"{x:.3f}" for x in row),"| oracle %.3f adaptive %.3f"%(orc,ad))
    for w in "AB":
        for part in ["TUNE","HALF1","HALF2"]:
            rows=[r for r in DC.rows if r[1]==w and r[5]==part]
            row=[np.mean([rr_of(a*DC.zD[r[0]]+(1-a)*DC.zB[p][r[0]],[r[4]]) for r in rows]) for a in alphas]
            orc=np.mean([max(rr_of(a*DC.zD[r[0]]+(1-a)*DC.zB[p][r[0]],[r[4]]) for a in alphas) for r in rows])
            ad=np.mean([rr_of((lambda al: al*DC.zD[r[0]]+(1-al)*DC.zB[p][r[0]])(1-(1-.3)*min(1,DC.cov[p][r[0]])**2),[r[4]]) for r in rows])
            print("dom",w,part,len(rows)," ".join(f"{x:.3f}" for x in row),"| oracle %.3f adaptive %.3f"%(orc,ad))
