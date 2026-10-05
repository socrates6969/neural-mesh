"""B6: linear-probe / centroid / ensemble capability routers vs the shipped kNN (k=5) on NOVEL phrasing. Train: battery routing items (seeds 1-30, both worlds).
Pre-declared protocol: pick the candidate hyper-parameters on ROUTE2 (n=96 hand-written, authored after the battery data was frozen); CONFIRM on ROUTE4 (n=60, blind set from R65), exact McNemar vs kNN.
Candidates: logreg(C in .1,1,10,100) on L2-normalised MiniLM; nearest-centroid; kNN+logreg prob average. Both worlds A (plain) and B (chatty+typos) of ROUTE2 as in R41 use data_route2 helpers if present."""
from lab import *
from lib import *
from data_route2 import ROUTE2
from data_route4 import ROUTE4
from sklearn.linear_model import LogisticRegression
from scipy import stats
X0, y0 = [], []
for w in WORLDS:
    for s in TUNE:
        for t, lab in routing_items(D, w, s): X0.append(D.v(t)); y0.append(lab)
X0, y0 = np.stack(X0), np.array(y0)
labs = sorted(set(y0))
def items(R): return [(q, lab) for lab, qs in R.items() for q in qs]
I2, I4 = items(ROUTE2), items(ROUTE4)
print("train", X0.shape, "route2", len(I2), "route4", len(I4), labs)
def knn_scores(X, y, q, k=5):
    s = X @ q; top = np.argsort(-s)[:k]; v = {l: 0.0 for l in labs}
    for i in top: v[y[i]] += float(s[i])
    t = sum(v.values()) or 1.0
    return np.array([v[l] / t for l in labs])
def cen_scores(C, q):
    s = C @ q; e = np.exp((s - s.max()) * 20); return e / e.sum()
cent = np.stack([X0[y0 == l].mean(0) for l in labs]); cent /= np.linalg.norm(cent, axis=1, keepdims=True)
models = {}
for C in (.1, 1, 10, 100):
    m = LogisticRegression(C=C, max_iter=2000).fit(X0, y0); models[f"lr{C}"] = m
def predict(name, t):
    q = D.v(t)
    if name == "knn5": return labs[int(np.argmax(knn_scores(X0, y0, q)))]
    if name == "centroid": return labs[int(np.argmax(cent @ q))]
    if name.startswith("lr"): return models[name].predict(q[None])[0]
    if name.startswith("knn+lr"):
        m = models["lr" + name[6:]]; p = m.predict_proba(q[None])[0]; return labs[int(np.argmax(knn_scores(X0, y0, q) + p))]
names = ["knn5", "centroid", "lr0.1", "lr1", "lr10", "lr100", "knn+lr1", "knn+lr10", "knn+lr100"]
def accs(I): return {n: np.array([predict(n, t) == l for t, l in I]) for n in names}
A2, A4 = accs(I2), accs(I4)
for n in names: print(f"{n:10} ROUTE2 {A2[n].mean():.3f}  ROUTE4 {A4[n].mean():.3f}")
def mcn(a, b):
    w = int((a & ~b).sum()); l = int((~a & b).sum()); return w, l, (stats.binomtest(w, w + l, .5).pvalue if w + l else 1.0)
print("vs knn5 (McNemar exact):")
for n in names[1:]: print(f"  {n:10} ROUTE2 W/L/p {mcn(A2[n],A2['knn5'])}   ROUTE4 W/L/p {mcn(A4[n],A4['knn5'])}")
