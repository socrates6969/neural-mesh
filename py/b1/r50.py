"""R50: improve the kNN router on NOVEL phrasing: embedding model (MiniLM vs BGE), k, training-set size (seeds 1-30 vs 1-10)."""
import sys; sys.path.insert(0, "../")
from lab import *
from lib import *
from r41 import seed_items, ALLS
from data_route2 import ROUTE2
import pickle, f8_embed
from r22 import DataBGE, D2
class K:
    def __init__(self, Dm, seeds, k=5):
        X, y = [], []
        for w in WORLDS:
            for s in seeds:
                for t, lab in routing_items(D, w, s): X.append(Dm.v(t)); y.append(lab)
        self.X, self.y, self.k, self.D = np.stack(X), np.array(y), k, Dm
    def __call__(self, t):
        s = self.X @ self.D.v(t); top = np.argsort(-s)[:self.k]; v = {}
        for i in top: v[self.y[i]] = v.get(self.y[i], 0) + float(s[i])
        return max(v, key=v.get)
def evalr(f, seeds):
    return {(w, s): {"route2_acc": float(np.mean([f(t) == l for t, l in seed_items(w, s)]))} for w in WORLDS for s in seeds}
if __name__ == "__main__":
    # make sure every text is embedded (BGE batch)
    texts = set()
    for w in WORLDS:
        for s in TUNE: texts.update(t for t, _ in routing_items(D, w, s))
        for s in ALLS: texts.update(t for t, _ in seed_items(w, s))
    todo = [t for t in texts if t not in D2.cache]; print("bge to embed", len(todo))
    for i in range(0, len(todo), 128):
        for t, v in zip(todo[i:i+128], f8_embed.Embedder.batch(f8_embed.Embedder(f8_embed.BGE, "cls", 768), todo[i:i+128])): D2.cache[t] = v
    D2.save()
    T = {}
    T["mini_k5"] = evalr(K(D, TUNE, 5), ALLS)
    for k in (1, 3, 9): T[f"mini_k{k}"] = evalr(K(D, TUNE, k), ALLS)
    T["bge_k5"] = evalr(K(D2, TUNE, 5), ALLS)
    T["bge_k3"] = evalr(K(D2, TUNE, 3), ALLS)
    T["mini_k5_train10"] = evalr(K(D, TUNE[:10], 5), ALLS)
    pickle.dump(T, open(f"{RES}/r50_tabs.pkl", "wb"))
    for a, t in T.items(): print(f"{a:16s}", {sp: round(float(np.mean([t[(w, s)]["route2_acc"] for w in WORLDS for s in sd])), 3) for sp, sd in (("TUNE", TUNE), ("HELD", HELD), ("CONF", CONF))})
