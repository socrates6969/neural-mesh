import sys, time
sys.path.insert(0,'py')
from k3_eval import *
import k3_extract as X
tables,_=load_all()
U=[T for T in tables if usable(T)]
split=sys.argv[1]; kn={'v1':X.KNOW_V1,'v2':X.KNOW_V2,'v3':X.KNOW_V3,'ind':X.KNOW_IND}[sys.argv[2]]
lays=sys.argv[3].split(',') if len(sys.argv)>3 else LAYOUTS
sel=[T for T in U if T['split']==split]
t0=time.time()
rs=run_grid(sel, lays, ['v0','mix'], kn)
print('time',round(time.time()-t0,1),'renders',len(rs))
a=agg(rs,key=lambda r:(r['layout'],r['vfmt']))
for lay in lays:
    row=[]
    for vf in ['v0','mix']:
        v=a.get((lay,vf))
        row.append('%.3f/%.3f/%.3f(n=%d)'%(v[0],v[1],v[2],v[3]+v[5]) if v else '-')
    print('%-7s'%lay,'  '.join(row))
