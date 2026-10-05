from rounds import *
import itertools
class Adaptive:
    def __init__(self, docs, vecs, arm, alpha, lo=0.5, hi=0.95, pw=1.0):
        self.vecs, self.bm, self.lo, self.hi, self.pw = vecs, BM25(docs), lo, hi, pw
    def scores(self, qtext, qvec):
        bm = self.bm.score(qtext); seen = {t for t in toks(qtext) if t in self.bm.post}
        den = sum(self.bm.idf[t] for t in seen) * (self.bm.k1 + 1)
        f = (bm.max() / den) if den > 0 else 0.0
        a = self.hi - (self.hi - self.lo) * min(1.0, f) ** self.pw
        return a * z(self.vecs @ qvec) + (1 - a) * z(bm)
def mk(lo, hi, pw=1.0): return lambda d, v, arm, al: Adaptive(d, v, arm, al, lo, hi, pw)
if __name__ == "__main__":
    print("R2 adaptive alpha: tune on TUNE (domain_mrr + real_mrr)/2")
    best = None
    for lo, hi, pw in itertools.product((0.3, 0.5, 0.7), (0.9, 1.0), (1.0, 2.0)):
        t = cand(TUNE, make=mk(lo, hi, pw), real=True)
        dm = tune_mean(t, "domain_mrr"); rm = float(np.mean([t[("R", s)]["real_mrr"] for s in TUNE if ("R", s) in t]))
        print(lo, hi, pw, round(dm, 4), round(rm, 4)); sc = dm + rm
        if best is None or sc > best[0]: best = (sc, lo, hi, pw)
    print("best", best)
    c = cand(make=mk(*best[1:]), real=True); b = json.load(open(f"{RES}/b1_baseline.json"))["hybrid"]; b = {(k[0], int(k[1:])): v for k, v in b.items()}
    compare(b, c, ["domain_r1", "domain_mrr", "domain_route", "real_mrr", "real_h1"])
