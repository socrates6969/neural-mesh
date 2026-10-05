import sys; sys.path.insert(0, "../")
from lab import *
import lab, pickle
from gate import compare
from lex import mk
import f8_embed
BGE = f8_embed.Embedder(f8_embed.BGE, "cls", 768)
class DataBGE(Data):
    def __init__(s, base):
        s.__dict__.update(base.__dict__); s.cache = {}
        pp = os.path.join(RES, "b1_bge_cache.pkl")
        if os.path.exists(pp): s.cache = pickle.load(open(pp, "rb"))
    def v(s, t):
        if t not in s.cache:
            s.cache[t] = BGE.batch([t])[0]
        return s.cache[t]
    def save(s): pickle.dump(s.cache, open(os.path.join(RES, "b1_bge_cache.pkl"), "wb"))
D2 = DataBGE(D)
lab.BASE = {}
tab = json.load(open(f"{RES}/b1_adaptive_table.json")); lab.BASE = {(k[0], int(k[1:])): v for k, v in tab.items()}
c = {}
t0 = time.time()
for w in WORLDS:
    for s in ALLS: c[(w, s)] = knowledge_axes(D2, w, s, make=mk())
print("bge embed time", time.time() - t0); D2.save()
v, rows, gain, loss = compare(lab.BASE, c, ["domain_r1", "domain_mrr", "domain_route", "general_info"], tol=0.002, quiet=False, axes=["domain_r1","domain_mrr","domain_route","general_info","comprehension"])
