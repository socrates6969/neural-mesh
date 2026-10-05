"""B1 battery evaluation library. Axes are per-(world, seed) scalars in [0,1] unless noted (latency us etc.).
Splits: TUNE 1-30, HELD 31-60, CONF 61-90 (iron rule). Worlds A,B.
All metrics are SYNTHETIC PROXIES (see RD-LOG.md scope statement)."""
import json, math, os, random, re, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
RES = os.path.join(ROOT, "results")
TUNE, HELD, CONF = list(range(1, 31)), list(range(31, 61)), list(range(61, 91))
WORLDS = "AB"
STOP = set("the a an of to and in is it for on that this with as are be by or at from was can if you we not will has have its".split())

def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]

class Data:
    def __init__(self):
        j = json.load(open(os.path.join(RES, "b1_items.json")))
        self.items, self.stores = j["items"], j["stores"]
        self.texts = json.load(open(os.path.join(RES, "b1_texts.json")))
        self.t2i = {t: i for i, t in enumerate(self.texts)}
        self.E = np.fromfile(os.path.join(RES, "b1_emb.f32"), dtype=np.float32).reshape(len(self.texts), 384)
        import pickle
        pp = os.path.join(RES, "b1_extra.pkl"); self.extra = pickle.load(open(pp, "rb")) if os.path.exists(pp) else {}
        self.exec = json.load(open(os.path.join(RES, "b1_exec.json"))) if os.path.exists(os.path.join(RES, "b1_exec.json")) else {}
    _sess = None
    def v(self, t):
        i = self.t2i.get(t)
        if i is not None: return self.E[i]
        x = self.extra.get(t)
        if x is None:
            from embed import embed
            x = embed([t])[0]; self.extra[t] = x; self.dirty = True
        return x
    def save_extra(self):
        if getattr(self, "dirty", False):
            import pickle; pickle.dump(self.extra, open(os.path.join(RES, "b1_extra.pkl"), "wb")); self.dirty = False

class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.post = {}
        self.dl = np.array([len(toks(d)) for d in docs], np.float32)
        for i, d in enumerate(docs):
            c = {}
            for t in toks(d): c[t] = c.get(t, 0) + 1
            for t, f in c.items(): self.post.setdefault(t, []).append((i, f))
        n = len(docs); self.n = n
        self.avg = float(self.dl.mean()) if n and self.dl.mean() > 0 else 1.0
        self.idf = {t: math.log((n - len(p) + .5) / (len(p) + .5) + 1) for t, p in self.post.items()}
    def score(self, q):
        out = np.zeros(self.n, np.float32); seen = set()
        for t in toks(q):
            if t in seen or t not in self.post: continue
            seen.add(t)
            for d, tf in self.post[t]:
                out[d] += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[d] / self.avg))
        return out

def z(x):
    s = x.std()
    return np.zeros_like(x) if s < 1e-9 else (x - x.mean()) / s

class Retriever:
    """arm in dense|bm25|hybrid. docs: list[str]; vecs: (n,384)."""
    def __init__(self, docs, vecs, arm="hybrid", alpha=0.7):
        self.docs, self.vecs, self.arm, self.alpha = docs, vecs, arm, alpha
        self.bm = BM25(docs) if arm != "dense" else None
    def scores(self, qtext, qvec):
        if self.arm == "dense": return self.vecs @ qvec
        if self.arm == "bm25": return self.bm.score(qtext)
        return self.alpha * z(self.vecs @ qvec) + (1 - self.alpha) * z(self.bm.score(qtext))

def typo(r, s, rate):
    w = s.split(" ")
    for i, x in enumerate(w):
        if len(x) >= 5 and r.random() < rate:
            k = r.randrange(1, len(x) - 1); w[i] = x[:k] + x[k + 1] + x[k] + x[k + 2:]
    return " ".join(w)

# ---------------------------------------------------------------- semantic skills (embedding-only arm)
def cosm(a, B):
    return (B @ a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1) + 1e-9)

def semantic_acc(D, items, E=None):
    """E: function text->vec (default MiniLM table). Returns {skill: accuracy}"""
    Eall = E or D.v
    res = {}
    for it in items:
        sk = it["skill"]
        if sk not in ("vocab", "similarities", "classification", "verbal_analogies", "sentence_completion"): continue
        E = Eall.get(sk, D.v) if isinstance(Eall, dict) else Eall
        opts = it["opts"]; O = np.stack([E(o) for o in opts])
        if sk == "vocab": s = cosm(E(it["target"]), O); p = int(np.argmax(s))
        elif sk == "similarities": s = cosm(E(it["pair"][0]) + E(it["pair"][1]), O); p = int(np.argmax(s))
        elif sk == "classification":
            S = O @ O.T / (np.linalg.norm(O, axis=1)[:, None] * np.linalg.norm(O, axis=1)[None] + 1e-9)
            p = int(np.argmin((S.sum(1) - 1) / (len(opts) - 1)))
        elif sk == "verbal_analogies":
            a, b, c = [E(x) for x in it["abc"]]; s = cosm(b - a + c, O); p = int(np.argmax(s))
        else:
            st = it["stem"]; F = np.stack([E(st.replace("___", o)) for o in opts]); s = cosm(E(st), F); p = int(np.argmax(s))
        res.setdefault(sk, []).append(p == it["ans"])
    return {k: float(np.mean(v)) for k, v in res.items()}

# ---------------------------------------------------------------- knowledge axes
def knowledge_axes(D, world, seed, arm="hybrid", alpha=0.7, make=None):
    """make(docs, vecs, arm, alpha)->retriever-like with .scores(qtext,qvec). Returns dict of axes."""
    make = make or (lambda docs, vecs, arm, alpha: Retriever(docs, vecs, arm, alpha))
    r = random.Random(seed * 7919 + (1 if world == "B" else 0))
    out = {}
    # general_info: all docs indexed; 12 queries
    g = D.stores["general"]; docs = g["docs"]; vecs = np.stack([D.v(d) for d in docs]); ret = make(docs, vecs, arm, alpha)
    qs = r.sample(g["qs"], 12); hit = []
    for gi, q1, q2, ans in qs:
        q = q1 if world == "A" else typo(r, q2, 0.12)
        qv = D.v(q)  # the typo'd text is embedded too (fixed R1 artifact: earlier typos hit only BM25)
        hit.append(int(np.argmax(ret.scores(q, qv))) == gi)
    out["general_info"] = float(np.mean(hit))
    # comprehension
    ch = []
    for it in D.items[f"{world}{seed}"]:
        if it["skill"] != "comprehension": continue
        sv = np.stack([D.v(s) for s in it["sents"]]); rr = make(it["sents"], sv, arm, alpha)
        ch.append(int(np.argmax(rr.scores(it["text"], D.v(it["text"])))) == it["gold"])
    out["comprehension"] = float(np.mean(ch))
    # domain
    dd = D.stores["domain"]; n = len(dd["docs"]); qs = r.sample(dd["qs"], 24)
    keep = set(r.sample(range(n), int(.7 * n))) | {q[0] for q in qs}; keep = sorted(keep)
    idx = {d: i for i, d in enumerate(keep)}
    docs = [dd["docs"][d] for d in keep]; vecs = np.stack([D.v(d) for d in docs])
    ret = make(docs, vecs, arm, alpha, tags=[dd["dom"][d] for d in keep]) if getattr(make, "wants_tags", False) else make(docs, vecs, arm, alpha)
    h1 = []; mrr = []; dh = []
    for gd, dom, q1, q2 in qs:
        q = q1 if world == "A" else typo(r, q2, 0.12); qv = D.v(q)
        sc = ret.scores(q, qv); top = int(np.argmax(sc)); gi = idx[gd]
        h1.append(top == gi); mrr.append(1.0 / (1 + int((sc > sc[gi]).sum())))
        dh.append(dd["dom"][keep[top]] == dom)
    out["domain_r1"], out["domain_mrr"], out["domain_route"] = float(np.mean(h1)), float(np.mean(mrr)), float(np.mean(dh))
    return out

# ---------------------------------------------------------------- L2 exec axes (from Rust b1_exec)
def exec_axes(D, world, seed):
    e = D.exec.get(f"{world}{seed}", {}); o = {}
    for sk, (n, ex, cl, us, cost) in e.items():
        o[f"exec_{sk}"] = ex / n; o[f"claim_{sk}"] = cl / (2 * n); o[f"us_{sk}"] = us
    return o

# ---------------------------------------------------------------- L1 routing
RULES = [("units", r"kilomet|metres|convert|miles|\bkg\b"), ("csp", r"clues|each finish|distinct place"),
         ("semantic", r"closest in meaning|alike|does not belong|\bis to\b|fills the blank|best fills"),
         ("knowledge", r"capital|chemical symbol|wrote|author|credited|where did|what did|which day|passage|to which place|which object|located|is .*thing|\bwhat is (the )?[a-z]+ of\b|^(what|how|why|which) (is|are|does|do|can)\b"),
         ("sandbox", r"digits|reorder|backwards|array|symbol|count how many|rotated|folded|repeating|series|sequence|figure|block"),
         ("calc", r"\d")]
def route_kw(text):
    t = text.lower()
    for lab, pat in RULES:
        if re.search(pat, t): return lab
    return "calc"

def routing_items(D, world, seed):
    r = random.Random(seed * 31 + (1 if world == "B" else 0)); out = [(it["text"], it["route"]) for it in D.items[f"{world}{seed}"] if it["skill"] != "general_info"]
    g = D.stores["general"]["qs"]; d = D.stores["domain"]["qs"]
    for gi, q1, q2, a in r.sample(g, 6): out.append((q1 if world == "A" else q2, "knowledge"))
    for di, dom, q1, q2 in r.sample(d, 6): out.append((q1 if world == "A" else q2, "knowledge"))
    return out

class KNNRouter:
    def __init__(self, D, seeds, k=5):
        X, y = [], []
        for w in WORLDS:
            for s in seeds:
                for t, lab in routing_items(D, w, s): X.append(D.v(t)); y.append(lab)
        self.X, self.y, self.k, self.D = np.stack(X), np.array(y), k, D
    def __call__(self, t):
        s = self.X @ self.D.v(t); top = np.argsort(-s)[:self.k]
        votes = {}
        for i in top: votes[self.y[i]] = votes.get(self.y[i], 0) + float(s[i])
        return max(votes, key=votes.get)

def routing_axes(D, world, seed, routers):
    items = routing_items(D, world, seed); o = {}
    for name, f in routers.items():
        o[f"route_{name}"] = float(np.mean([f(t) == lab for t, lab in items]))
    return o

# ---------------------------------------------------------------- stats
def tcrit(df):
    from scipy import stats
    return float(stats.t.ppf(0.975, df))

def paired(a, b):
    d = np.array(a) - np.array(b); n = len(d)
    m, sd = d.mean(), d.std(ddof=1) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    from scipy import stats
    t = m / se if se > 0 else (0.0 if m == 0 else float("inf") * np.sign(m))
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1)) if se > 0 else (1.0 if m == 0 else 0.0)
    c = tcrit(n - 1) * se
    return dict(mean=float(m), lo=float(m - c), hi=float(m + c), p=float(p), n=n, wins=int((d > 0).sum()), losses=int((d < 0).sum()))

def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.zeros(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * ps[i]); adj[i] = min(1.0, run)
    return adj.tolist()

def tost(a, b, delta):
    """equivalence within +-delta: both one-sided p; returns max p (small => equivalent)"""
    from scipy import stats
    d = np.array(a) - np.array(b); n = len(d); m = d.mean(); se = d.std(ddof=1) / math.sqrt(n) + 1e-12
    p1 = 1 - stats.t.cdf((m + delta) / se, n - 1); p2 = stats.t.cdf((m - delta) / se, n - 1)
    return float(max(p1, p2))

# ---------------------------------------------------------------- REAL-text regression axis (h10 fixtures, corpus A: 6831 real doc paragraphs)
class RealWorld:
    def __init__(self, tag="A"):
        self.texts = json.load(open(os.path.join(RES, f"h10_texts_{tag}.json")))
        self.emb = np.fromfile(os.path.join(RES, f"h10_emb_{tag}.f32"), dtype=np.float32).reshape(len(self.texts), 384)
        j = json.load(open(os.path.join(RES, f"h10_inst_{tag}.json")))
        self.ndoc, self.queries, self.inst = j["ndoc"], j["queries"], j["inst"]
        self.docs, self.dvec = self.texts[:self.ndoc], self.emb[:self.ndoc]
        self._ret = {}
    def retriever(self, arm, alpha, make=None):
        key = (arm, alpha, id(make))
        if key not in self._ret:
            mk = make or (lambda d, v, a, al: Retriever(d, v, a, al)); self._ret[key] = (make, mk(self.docs, self.dvec, arm, alpha))  # hold ref to make so id() is never recycled
        return self._ret[key][1]
    def axes(self, seed, arm="hybrid", alpha=0.7, make=None, nq=120):
        v = self.inst.get(str(seed))
        if v is None: return None
        ret = self.retriever(arm, alpha, make); rr, h1 = [], []
        for row in v["rows"][:nq]:
            tid = row["t"][0][0]; rel = set(self.queries[row["q"]]["rel"])
            sc = ret.scores(self.texts[tid], self.emb[tid]); best = min(int((sc > sc[d]).sum()) for d in rel)
            rr.append(1.0 / (1 + best)); h1.append(best == 0)
        return {"real_mrr": float(np.mean(rr)), "real_h1": float(np.mean(h1))}
