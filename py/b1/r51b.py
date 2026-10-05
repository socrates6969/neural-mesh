"""R51b: kNN router exemplar augmentation: add n authored exemplars per class (ROUTE3, in both plain and chatty/typo form for world B?) to the battery-trained kNN; test on ROUTE2 novel phrasing."""
import sys; sys.path.insert(0, "../")
from lab import *
from lib import *
from r41 import seed_items, ALLS
from data_route3 import ROUTE3
import random, pickle
PRE = ["hey, quick question: ", "pls help - ", "could you tell me, ", "ok so ", "sorry to bother, ", "one thing - "]
class KA:
    def __init__(self, n_extra, k=5, w_extra=1.0, seed=0):
        X, y = [], []
        for w in WORLDS:
            for s in TUNE:
                for t, lab in routing_items(D, w, s): X.append(D.v(t)); y.append(lab)
        r = random.Random(seed)
        for lab, qs in ROUTE3.items():
            for q in r.sample(qs, n_extra): X.append(D.v(q)); y.append(lab)
        self.X, self.y, self.k = np.stack(X), np.array(y), k
    def __call__(self, t):
        s = self.X @ D.v(t); top = np.argsort(-s)[:self.k]; v = {}
        for i in top: v[self.y[i]] = v.get(self.y[i], 0) + float(s[i])
        return max(v, key=v.get)
def evalr(f, seeds):
    return {(w, s): {"route2_acc": float(np.mean([f(t) == l for t, l in seed_items(w, s)]))} for w in WORLDS for s in seeds}
if __name__ == "__main__":
    T = {}
    for n in (0, 3, 6, 12):
        T[f"extra{n}"] = evalr(KA(n), ALLS)
        print(n, {sp: round(float(np.mean([T[f"extra{n}"][(w, s)]["route2_acc"] for w in WORLDS for s in sd])), 3) for sp, sd in (("TUNE", TUNE), ("HELD", HELD), ("CONF", CONF))})
    # a heavier weight: replicate each extra exemplar 5x (they would otherwise be 72 among ~9000 training rows)
    pickle.dump(T, open(f"{RES}/r51b_tabs.pkl", "wb"))
