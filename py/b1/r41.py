"""R41: ensemble keyword + kNN router on a NOVEL-phrasing set. Per seed: 48 items sampled from ROUTE2 (6 classes); world A plain, world B chatty prefix + 12% transposition typos."""
from lab import *
from lib import *
from data_route2 import ROUTE2
import random, re
PRE = ["hey, quick question: ", "pls help - ", "could you tell me, ", "ok so ", "sorry to bother, ", "one thing - "]
def seed_items(world, seed, n=8):
    r = random.Random(seed * 131 + (1 if world == "B" else 0)); out = []
    for lab, qs in ROUTE2.items():
        for q in r.sample(qs, n):
            if world == "B": q = r.choice(PRE) + typo(r, q, 0.12)
            out.append((q, lab))
    return out
KNN = KNNRouter(D, TUNE)
def kw_conf(text):
    t = text.lower()
    for lab, pat in RULES:
        if re.search(pat, t): return lab, True
    return "calc", False
def knn_votes(t, k=5):
    s = KNN.X @ D.v(t); top = np.argsort(-s)[:k]; v = {}
    for i in top: v[KNN.y[i]] = v.get(KNN.y[i], 0) + float(s[i])
    tot = sum(v.values()); return {a: b / tot for a, b in v.items()}
def r_kw(t): return kw_conf(t)[0]
def r_knn(t): v = knn_votes(t); return max(v, key=v.get)
def r_e1(t):  # kw when a rule fired, else knn
    l, m = kw_conf(t); return l if m else r_knn(t)
def mk_e2(tau):  # knn when confident else kw
    def f(t):
        v = knn_votes(t); l = max(v, key=v.get); return l if v[l] >= tau else r_kw(t)
    return f
def mk_e3(w):  # vote sum
    def f(t):
        v = knn_votes(t); l, m = kw_conf(t)
        if m: v[l] = v.get(l, 0) + w
        return max(v, key=v.get)
    return f
ARMS = {"kw": r_kw, "knn": r_knn, "e1_kw_then_knn": r_e1, "e2_knn_tau.6": mk_e2(.6), "e2_knn_tau.8": mk_e2(.8), "e3_vote_w.3": mk_e3(.3), "e3_vote_w.6": mk_e3(.6), "e3_vote_w1": mk_e3(1.0)}
def table(seeds=ALLS):
    tabs = {a: {} for a in ARMS}
    for w in WORLDS:
        for s in seeds:
            its = seed_items(w, s)
            for a, f in ARMS.items(): tabs[a][(w, s)] = {"route2_acc": float(np.mean([f(t) == l for t, l in its]))}
    return tabs
if __name__ == "__main__":
    import pickle; T = table(); pickle.dump(T, open(f"{RES}/r41_tabs.pkl", "wb"))
    for a in ARMS: print(f"{a:16s}", {sp: round(float(np.mean([T[a][(w, s)]["route2_acc"] for w in WORLDS for s in sd])), 3) for sp, sd in (("TUNE", TUNE), ("HELD", HELD), ("CONF", CONF))})
