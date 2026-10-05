"""R60: re-tune the adaptive-alpha (lo, hi, pow) jointly with BM25 (k1=.9, b=.5) on TUNE (domain_mrr + real_mrr), verify on HELD/CONF vs adaptive(.3,1,2)+BM25(.9,.5) AND vs the original adaptive default."""
from lex import *
import lex, json, itertools
from v2 import gate
class LexP(Lex):
    def _init_bm(s, docs):
        super()._init_bm(docs); b = s.bm; b.k1, b.b = .9, .5
        b.post = {}; b.dl = np.array([len(s.tok(d)) for d in docs], np.float32)
        for i, d in enumerate(docs):
            c = {}
            for t in s.tok(d): c[t] = c.get(t, 0) + 1
            for t, f in c.items(): b.post.setdefault(t, []).append((i, f))
        b.avg = float(b.dl.mean()) if b.dl.mean() > 0 else 1.0
        b.idf = {t: math.log((b.n - len(p) + .5) / (len(p) + .5) + 1) for t, p in b.post.items()}
def mkp(lo, hi, pw):
    def f(d, v, ar, al, tags=None):
        o = LexP.__new__(LexP); Lex.__init__(o, d, v, ar, al, tags, lo=lo, hi=hi, pw=pw); return o
    f.wants_tags = True; return f
if __name__ == "__main__":
    tab = json.load(open(f"{RES}/b1_adaptive_table.json")); ADAPT = {(k[0], int(k[1:])): v for k, v in tab.items()}
    rows = []
    for lo, hi, pw in itertools.product((0.2, 0.3, 0.5), (0.9, 1.0), (1.0, 2.0, 3.0)):
        c = cand(seeds=TUNE, real=False, make=mkp(lo, hi, pw)); cr = cand(seeds=[s for s in range(1, 9)], real=True, make=mkp(lo, hi, pw))
        dm = np.mean([c[(w, s)]["domain_mrr"] for w in WORLDS for s in TUNE]); rm = np.mean([cr[("R", s)]["real_mrr"] for s in range(1, 9) if ("R", s) in cr])
        rows.append(((lo, hi, pw), dm, rm)); print((lo, hi, pw), round(dm, 4), round(rm, 4), "obj", round(dm + rm, 4))
    best = max(rows, key=lambda x: x[1] + x[2]); print("BEST on TUNE:", best)
    json.dump([(list(a), b, c) for a, b, c in rows], open(f"{RES}/r60_tune.json", "w"))
