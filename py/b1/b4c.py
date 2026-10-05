import pickle, itertools, numpy as np
from lib import RES
TB=pickle.load(open(f"{RES}/b4_table.pkl","rb"))
AL=np.round(np.arange(0,1.0001,.05),2)
def ai(a): return np.clip(np.round(np.asarray(a)/.05).astype(int),0,20)
GR=["realH","realS","domA","domB"]
def grp(k): return {"H":"realH","S":"realS"}.get(k[2]) if k[0]=="real" else k[0]
def evalrule(T,rule,part):
    res={g:[] for g in GR}
    for k,(f,v) in T.items():
        if k[1]!=part: continue
        res[grp(k)].append(v[ai(rule(f))])
    return {g:float(np.mean(x)) for g,x in res.items()}
def table(p):
    T=TB[p]; out=[]
    rules={}
    for a in (.6,.7,.8,.9): rules[f"fix{a}"]=(lambda a: lambda f:a)(a)
    for lo,hi,pw in itertools.product((.3,.5,.6),(.8,.9,1.0),(1,2,4)):
        def r(f,lo=lo,hi=hi,pw=pw): return hi-(hi-lo)*min(1,f["cov"])**pw
        rules[f"adapt({lo},{hi},{pw})"]=r
    for a0,k,s in itertools.product((.6,.7,.8),(.1,.2,.3),(1,2,4)):
        rules[f"conf({a0},{k},{s})"]=(lambda a0,k,s: lambda f:a0+k*np.tanh((f["mD"]-f["mB"])/s))(a0,k,s)
        rules[f"gap({a0},{k},{s})"]=(lambda a0,k,s: lambda f:a0+k*np.tanh((f["sD"]-f["sB"])/s))(a0,k,s)
    for a1,a2 in itertools.product((.6,.7,.8),(.8,.9,1.0)):
        rules[f"agree({a1},{a2})"]=(lambda a1,a2: lambda f:a1 if f["agree"]>.5 else a2)(a1,a2)
    for n0,a1,a2 in itertools.product((6,8,12),(.6,.7),(.8,.9,1.0)):
        rules[f"ntok<{n0}:({a2},{a1})"]=(lambda n0,a1,a2: lambda f:a2 if f["ntok"]<n0 else a1)(n0,a1,a2)
    base=evalrule(T,rules["fix0.7"],"TUNE")
    for name,r in rules.items():
        t=evalrule(T,r,"TUNE"); gains={g:t[g]-base[g] for g in GR}
        out.append((name,t,gains))
    return rules,base,out
if __name__=="__main__":
    for p in [(1.2,.75),(.9,.5)]:
        rules,base,out=table(p)
        print("BM25",p,"TUNE base fix.7:",{g:round(v,3) for g,v in base.items()})
        # score = mean group gain, require no real group below -.002
        ok=[o for o in out if min(o[2]["realH"],o[2]["realS"])>=-.002]
        ok.sort(key=lambda o:-np.mean(list(o[2].values())))
        for name,t,g in ok[:12]:
            print(f"  {name:22} gains " + " ".join(f"{k}{v:+.3f}" for k,v in g.items()))
        print("  existing adaptive(.3,1.0,2):",[ (o[0],{k:round(v,3) for k,v in o[2].items()}) for o in out if o[0]=="adapt(0.3,1.0,2)"])
