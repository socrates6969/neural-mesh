"""k4 common: world loaders (h10 A/B real-text fixtures, h20 pydoc9/crates8), hybrid z+BM25 retrieval (src/hybrid.rs semantics: z(.7)+BM25, spell
already applied in the saved query variants), metrics and paired statistics (seed t-CI, query-cluster bootstrap, Holm, TOST).
Everything REAL text / REAL structure; typos and any outcome model are SYNTHETIC and labelled where used.  Python 3 + numpy + scipy (+sklearn)."""
import collections, hashlib, importlib.util, json, math, os, re, sys
import numpy as np
import scipy.sparse as sp
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
STOP = set("the a an of to and in is it for on that this with as are be by or at from was can if you we not will has have its".split())
TOK = re.compile(r"[a-z0-9]+")
ALPHA = 0.7
K1, B = 1.2, 0.75


def tokens(s):
    return [t for t in TOK.findall(s.lower()) if t not in STOP]


def zrows(M):
    M = np.asarray(M, dtype=np.float32)
    mu = M.mean(1, keepdims=True)
    sd = M.std(1, keepdims=True)
    out = (M - mu) / np.maximum(sd, 1e-9)
    out[(M.max(1) - M.min(1)) <= 0] = 0.0
    return out


def top_idx(S, k=10):
    k = min(k, S.shape[1] - 1)
    p = np.argpartition(-S, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, p, 1), axis=1)
    return np.take_along_axis(p, o, 1)


class Store:
    """a document store: texts, dense vectors D (unit norm), BM25 matrix, per-doc metadata."""

    def __init__(self, name, texts, D, meta=None):
        self.name = name
        self.texts = texts
        self.n = len(texts)
        self.D = np.asarray(D, dtype=np.float32)
        self.meta = meta or [{} for _ in texts]
        toks = [tokens(t) for t in texts]
        self.toks = toks
        vocab = {}
        for t in toks:
            for w in t:
                if w not in vocab:
                    vocab[w] = len(vocab)
        self.vocab = vocab
        rows, cols, vals = [], [], []
        for i, t in enumerate(toks):
            for w, f in collections.Counter(t).items():
                rows.append(i); cols.append(vocab[w]); vals.append(float(f))
        self.C = sp.csr_matrix((vals, (rows, cols)), shape=(self.n, len(vocab)), dtype=np.float32)
        self.dl = np.array([len(t) for t in toks], dtype=np.float32)
        self.W = self.bm25(np.arange(self.n))
        self.df = np.asarray((self.C > 0).sum(0)).ravel()

    def bm25(self, ids):
        C = self.C[ids].tocsr()
        N = len(ids)
        df = np.asarray((C > 0).sum(0)).ravel().astype(np.float32)
        idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        avg = max(float(self.dl[ids].mean()), 1.0)
        W = C.copy()
        rows = np.repeat(np.arange(N), np.diff(W.indptr))
        tf = W.data
        W.data = (idf[W.indices] * tf * (K1 + 1) / (tf + K1 * (1 - B + B * self.dl[ids][rows] / avg))).astype(np.float32)
        return W

    def qbow(self, texts):
        rows, cols = [], []
        for i, t in enumerate(texts):
            for w in set(tokens(t)):
                j = self.vocab.get(w)
                if j is not None:
                    rows.append(i); cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(texts), len(self.vocab)))

    def parts(self, qE, qB):
        return qE @ self.D.T, (qB @ self.W.T).toarray()

    def fused(self, qE, qB, ids=None):
        """z(.7)+BM25 hybrid over the index `ids` (None = all docs); z-scored over that index"""
        dense, bm = self.parts(qE, qB)
        if ids is not None:
            dense, bm = dense[:, ids], bm[:, ids]
        return ALPHA * zrows(dense) + (1 - ALPHA) * zrows(bm)


class QSet:
    """a seed's query set: spelled-variant embeddings E, BoW B (over store vocab), rel (list of index arrays), fam, qid"""

    def __init__(self, E, Bw, rel, fam, qid, typo=None):
        self.E, self.B, self.rel, self.fam, self.qid = E, Bw, rel, np.array(fam), np.array(qid)
        self.typo = typo if typo is not None else np.zeros(len(qid), bool)
        self.n = len(qid)


def hnum(*a):
    return int(hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()[:12], 16)


class H10World:
    """h10 real-text world A (6831 crate-README paragraphs) or B (4181 repo/ruvnet paragraphs); queries: H = section title, S = held-out sentence"""

    def __init__(self, tag):
        self.tag = tag
        c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
        self.docs = c["docs"]
        self.queries = c["queries"]
        texts = json.load(open(os.path.join(RES, "h10_texts_%s.json" % tag), encoding="utf8"))
        self.alltexts = texts
        inst = json.load(open(os.path.join(RES, "h10_inst_%s.json" % tag), encoding="utf8"))
        self.inst = inst["inst"]
        nd = inst["ndoc"]
        emb = np.fromfile(os.path.join(RES, "h10_emb_%s.f32" % tag), dtype=np.float32).reshape(len(texts), 384)
        self.emb = emb
        meta = [dict(src=d["src"], file=d["file"], id=d["id"]) for d in self.docs]
        self.store = Store("h10" + tag, [d["text"] for d in self.docs], emb[:nd], meta)
        self.qtext = {q["id"]: q["text"] for q in self.queries}

    def seeds(self, phase):
        if self.tag == "A":
            return {"tune": range(1, 9), "held": range(31, 61), "confirm": range(61, 91)}[phase]
        return {"confirm": range(61, 91)}[phase]

    def qset(self, seed, typo_frac=0.25):
        rows = self.inst[str(seed)]["rows"]
        E, texts, rel, fam, qid, ty = [], [], [], [], [], []
        for j, r in enumerate(rows):
            u = (hnum("typo", self.tag, seed, r["q"], j) % 1000) / 1000.0
            v = 0 if u >= typo_frac else 1 + (hnum("v", seed, r["q"], j) % 2)
            ti = r["t"][v][1]  # S1-spelled variant of instance v
            E.append(self.emb[ti]); texts.append(self.alltexts[ti])
            rel.append(np.array(self.queries[r["q"]]["rel"], dtype=int)); fam.append(r["fam"]); qid.append(r["q"]); ty.append(v > 0)
        return QSet(np.stack(E), self.store.qbow(texts), rel, fam, qid, np.array(ty))


class H20World:
    """h20 real-text world (pydoc9 / crates8): one query per doc (held-out sentence), domain label = provenance"""

    def __init__(self, tag):
        self.tag = tag
        w = json.load(open(os.path.join(RES, "h20_world_%s.json" % tag), encoding="utf8"))
        docs = w["docs"]
        self.docs = docs
        n = len(docs)
        E = np.fromfile(os.path.join(RES, "h20_emb_%s.f32" % tag), dtype=np.float32).reshape(n, 5, 384)
        self.Q = E[:, 1:, :]
        names = sorted(set(d["domain"] for d in docs))
        di = {x: i for i, x in enumerate(names)}
        self.names = names
        self.dom = np.array([di[d["domain"]] for d in docs])
        meta = [dict(src=d.get("src", ""), domain=d["domain"], id=i) for i, d in enumerate(docs)]
        self.store = Store("h20" + tag, [d["text"] for d in docs], E[:, 0, :], meta)
        h = np.array([int(hashlib.sha1(("%s|%d" % (tag, i)).encode()).hexdigest()[:8], 16) % 4 for i in range(n)])
        self.bucket = {"tune": np.where(h == 0)[0], "held": np.where(h == 1)[0], "confirm": np.where(h == 2)[0]}

    def seeds(self, phase):
        return {"tune": range(1, 9), "held": range(31, 61), "confirm": range(61, 91)}[phase]

    def qset(self, seed, phase, nq_cap=150, typo_frac=0.5):
        rng = np.random.RandomState(seed * 1000 + 17)
        pool = self.bucket[phase]
        nq = min(nq_cap, int(0.75 * len(pool)))
        gold = rng.choice(pool, nq, replace=False)
        typo = rng.rand(nq) < typo_frac
        var_sp = np.where(typo, 3, 2)
        E = self.Q[gold, var_sp]
        keys = ["qs_typo" if t else "qs_clean" for t in typo]
        texts = [self.docs[g][k] for g, k in zip(gold, keys)]
        return QSet(E, self.store.qbow(texts), [np.array([g]) for g in gold], ["S"] * nq, gold, typo)


# ----------------------------------------------------------------------------------------------- metrics
def rank_metrics(top, rel, kk=(1, 3, 8)):
    """top: (nq, >=10) doc ids (-1 = none); rel: list of arrays.  returns dict of per-query arrays: hit@k (any relevant in top-k), mrr@10, f1@3 (as h13)"""
    nq = len(rel)
    out = {"mrr": np.zeros(nq)}
    for k in kk:
        out["hit%d" % k] = np.zeros(nq)
    out["f1@3"] = np.zeros(nq)
    for i in range(nq):
        rs = set(rel[i].tolist())
        row = top[i].tolist()
        pos = next((j for j, d in enumerate(row[:10]) if d in rs), None)
        if pos is not None:
            out["mrr"][i] = 1.0 / (pos + 1)
        for k in kk:
            out["hit%d" % k][i] = 1.0 if any(d in rs for d in row[:k]) else 0.0
        h3 = sum(1 for d in row[:3] if d in rs)
        p = h3 / 3.0; r = h3 / max(1, len(rs))
        out["f1@3"][i] = 0.0 if h3 == 0 else 2 * p * r / (p + r)
    return out


# ----------------------------------------------------------------------------------------------- statistics
def t_ci(d, conf=0.95):
    d = np.asarray(d, float)
    n = len(d)
    m = d.mean()
    if n < 2:
        return m, m, m, 1.0
    se = d.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return m, m, m, 1.0 if m == 0 else 0.0
    tc = stats.t.ppf(0.5 + conf / 2, n - 1)
    p = 2 * (1 - stats.t.cdf(abs(m / se), n - 1))
    return m, m - tc * se, m + tc * se, p


def tost(d, margin, conf=0.95):
    """TOST equivalence: True if the 90% CI of the mean paired diff lies inside (-margin, +margin)"""
    m, lo, hi, _ = t_ci(d, 0.90)
    return (lo > -margin) and (hi < margin)


def holm(ps):
    ps = np.asarray(ps, float)
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def cluster_boot(per_q, qids, reps=2000, seed=1):
    """per-query paired diffs with a query-cluster bootstrap over DISTINCT query ids -> (mean, lo, hi, p two-sided)"""
    per_q = np.asarray(per_q, float)
    qids = np.asarray(qids)
    u, inv = np.unique(qids, return_inverse=True)
    s = np.bincount(inv, weights=per_q)
    c = np.bincount(inv).astype(float)
    rng = np.random.RandomState(seed)
    m = s.sum() / c.sum()
    bs = np.empty(reps)
    for r in range(reps):
        ix = rng.randint(0, len(u), len(u))
        bs[r] = s[ix].sum() / c[ix].sum()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()) + 1.0 / reps)
    return m, lo, hi, p


def cmp_arms(runs_a, runs_b, key):
    """seeds-paired t-CI of mean(key) a - b over per-seed means; runs_x: list over seeds of dict key -> per-query array"""
    d = np.array([r_a[key].mean() - r_b[key].mean() for r_a, r_b in zip(runs_a, runs_b)])
    return t_ci(d)


def jdump(obj, name):
    with open(os.path.join(RES, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
