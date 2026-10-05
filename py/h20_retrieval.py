"""h20-B: DOMAIN-CORPUS retrieval.  Domain router -> domain index (or domain prior on the pooled index) vs pooled retrieval vs no retrieval, on
held-out real/synthetic text, with end-to-end research-task quality from a DECLARED quality model.   (SYNTHETIC outcome model; real text for pydoc9/crates8.)

Worlds (built by py/h20_corpus.py, py/h20_syn.py, py/h20_embed.py):  pydoc9 (real python docstrings, 7 domains), crates8 (real crate READMEs/lib docs, 7 domains),
syn10 (SYNTHETIC role-domain corpora, 10 domains).  Controls: <world>+shuf = domain labels randomly permuted (NO domain structure).
Protocol: tune seeds 1-8, held-out 31-60, confirm 61-90 (config frozen from tune).  Query pools are disjoint doc buckets: tune bucket0, held bucket1, confirm bucket2, check bucket3.

Quality model (declared assumption; a gain exists by construction when the gold paragraph is in context): per task
   q = clip(base + G*hit5 - H*offdom5 + noise, 0, 1),  base ~ N(.45,.10), noise ~ N(0,.05) (common random numbers across arms),
   hit5 = gold paragraph in the k=5 retrieved contexts, offdom5 = fraction of the 5 contexts from a domain other than the task's.  Default G=.30, H=.10, k=5.
Pooled baselines: dense raw / dense+spell / hybrid z(alpha .7)+BM25 + spell (src/hybrid.rs semantics; z over ALL docs of the searched index) / hybrid raw.
Domain arms: routers cent | knn | fuse (probabilities), arms  H = hard route to top-1 domain index (z over that index), P = pooled hybrid + lam*P(domain),
M = pooled hybrid masked to top-m domains, each optionally confidence-gated (margin >= tau); ALL domain arms are disabled when the label-structure
self-check fails (kNN label purity excess < .15; pre-declared, not tuned).  Oracle = true domain.
Usage: python py/h20_retrieval.py --tune | --phase held|confirm | --sens held   (python 3.14 + numpy + scipy)
"""
import argparse, collections, hashlib, itertools, json, math, os, re, sys, time
import numpy as np
import scipy.sparse as sp
from scipy import stats

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
STOP = set("the a an of to and in is it for on that this with as are be by or at from was can if you we not will has have its".split())
TOK = re.compile(r"[a-z0-9]+")
ALPHA = 0.7
K1, B = 1.2, 0.75
WORLDS = ["pydoc9", "crates8", "syn10"]
SELF_EXCESS = 0.15


def tokens(s):
    return [t for t in TOK.findall(s.lower()) if t not in STOP]


def zrows(M):
    M = np.asarray(M, dtype=np.float32)
    mu = M.mean(1, keepdims=True)
    sd = M.std(1, keepdims=True)
    out = (M - mu) / np.maximum(sd, 1e-9)
    out[(M.max(1) - M.min(1)) <= 0] = 0.0
    return out


class World:
    def __init__(self, tag, shuffle=False):
        w = json.load(open(os.path.join(RES, "h20_world_%s.json" % tag), encoding="utf8"))
        self.tag = tag + ("+shuf" if shuffle else "")
        self.docs = w["docs"]
        n = len(self.docs)
        self.n = n
        E = np.fromfile(os.path.join(RES, "h20_emb_%s.f32" % tag), dtype=np.float32).reshape(n, 5, 384)
        self.D, self.Q = E[:, 0, :], E[:, 1:, :]  # Q[:,0]=clean raw, 1=typo raw, 2=clean spelled, 3=typo spelled
        names = sorted(set(d["domain"] for d in self.docs))
        di = {x: i for i, x in enumerate(names)}
        self.names = names
        self.dom = np.array([di[d["domain"]] for d in self.docs])
        self.dom_true = self.dom.copy()
        if shuffle:
            rng = np.random.RandomState(777)
            self.dom = rng.permutation(self.dom)  # index labels are noise; dom_true keeps the real domain for the harm model
        self.nd = len(names)
        # tokens / vocab
        toks = [tokens(d["text"]) for d in self.docs]
        vocab = {}
        for t in toks:
            for w_ in t:
                if w_ not in vocab:
                    vocab[w_] = len(vocab)
        self.vocab = vocab
        rows, cols, vals = [], [], []
        for i, t in enumerate(toks):
            c = collections.Counter(t)
            for w_, f in c.items():
                rows.append(i)
                cols.append(vocab[w_])
                vals.append(float(f))
        self.C = sp.csr_matrix((vals, (rows, cols)), shape=(n, len(vocab)), dtype=np.float32)
        self.dl = np.array([len(t) for t in toks], dtype=np.float32)
        allids = np.arange(n)
        self.pool = self.bm25(allids)
        self.dom_ids = [np.where(self.dom == d)[0] for d in range(self.nd)]
        self.dom_bm = [self.bm25(ix) for ix in self.dom_ids]
        self.lr = None
        cen = np.stack([self.D[ix].mean(0) for ix in self.dom_ids])
        self.cent = cen / np.linalg.norm(cen, axis=1, keepdims=True)
        # buckets
        h = np.array([int(hashlib.sha1(("%s|%d" % (tag, i)).encode()).hexdigest()[:8], 16) % 4 for i in range(n)])
        self.bucket = {"tune": np.where(h == 0)[0], "held": np.where(h == 1)[0], "confirm": np.where(h == 2)[0], "check": np.where(h == 3)[0]}
        # domain specificity of each doc's clean spelled query (for the confusable tertile)
        df_d = np.zeros((self.nd, len(vocab)), dtype=np.float32)
        for d in range(self.nd):
            df_d[d] = np.asarray((self.C[self.dom_ids[d]] > 0).sum(0)).ravel()
        dft = df_d.sum(0)
        self.spec_tok = np.where(dft >= 3, df_d.max(0) / np.maximum(dft, 1), np.nan)
        spec = []
        for d in self.docs:
            v = [self.spec_tok[vocab[t]] for t in set(tokens(d["qs_clean"])) if t in vocab]
            v = [x for x in v if not np.isnan(x)]
            spec.append(np.mean(v) if v else 1.0)
        self.spec = np.array(spec)
        self.conf_thr = np.quantile(self.spec[self.bucket["held"]], 1 / 3)  # lowest tertile of the HELD pool, declared before runs
        # label-structure self-check: kNN label purity excess on the check bucket (dense neighbours, excluding self)
        ci = self.bucket["check"][:400]
        S = self.D[ci] @ self.D.T
        S[np.arange(len(ci)), ci] = -9
        top = np.argpartition(-S, 5, axis=1)[:, :5]
        pur = np.mean(self.dom[top] == self.dom[ci][:, None])
        base = float(np.sum((np.bincount(self.dom) / n) ** 2))
        self.purity, self.purity_base = float(pur), base
        self.excess = float(pur - base)
        self.domain_ok = self.excess >= SELF_EXCESS

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
            for w_ in set(tokens(t)):
                j = self.vocab.get(w_)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(texts), len(self.vocab)))

    def fused(self, qE, qB, ids, W):
        """fused z-score matrix (nq x len(ids)) over index `ids` with BM25 matrix W (hybrid.rs semantics)"""
        dense = qE @ self.D[ids].T
        bm = (qB @ W.T).toarray()
        return ALPHA * zrows(dense) + (1 - ALPHA) * zrows(bm)


def softmax(x, T):
    z = x / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def lr_probs(w, qE_s):
    if w.lr is None:
        from sklearn.linear_model import LogisticRegression
        w.lr = LogisticRegression(C=20.0, max_iter=300).fit(w.D, w.dom)
    P = np.full((len(qE_s), w.nd), 1e-6)
    P[:, w.lr.classes_] = w.lr.predict_proba(qE_s)
    return P / P.sum(1, keepdims=True)


def route_probs(w, qE_s, Fp, kind):
    if kind in ("lr", "lrk"):
        pl = lr_probs(w, qE_s)
        if kind == "lr":
            return pl
    pc = softmax(qE_s @ w.cent.T, 0.05)
    top = np.argsort(-Fp, 1)[:, :10]
    pk = np.zeros((len(Fp), w.nd))
    for r in range(10):
        np.add.at(pk, (np.arange(len(Fp)), w.dom[top[:, r]]), 1.0 / (r + 1))
    pk = (pk + 0.02) / (pk + 0.02).sum(1, keepdims=True)
    out = {"cent": pc, "knn": pk, "fuse": 0.5 * pc + 0.5 * pk}
    if kind == "lrk":
        return 0.5 * pl + 0.5 * pk
    return out[kind]


def top_idx(S, k=10):
    p = np.argpartition(-S, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(S, p, 1), axis=1)
    return np.take_along_axis(p, o, 1)


def search_domain(w, qE, qB, pred, gate):
    """hard route: per-query in-domain fused search; where gate is False use pooled search. returns top-10 ids"""
    out = np.zeros((len(qE), 10), dtype=int)
    Fp = None
    for d in range(w.nd):
        rows = np.where((pred == d) & gate)[0]
        if len(rows) == 0:
            continue
        ids = w.dom_ids[d]
        F = w.fused(qE[rows], qB[rows], ids, w.dom_bm[d])
        kk = min(10, len(ids))
        t = top_idx(F, kk)
        res = ids[t]
        if kk < 10:
            res = np.concatenate([res, np.full((len(rows), 10 - kk), -1)], 1)
        out[rows] = res
    return out


def metrics(top, gold, dom_task, w, k=5):
    nq = len(gold)
    pos = np.full(nq, 99)
    for r in range(10):
        m = (top[:, r] == gold) & (pos == 99)
        pos[m] = r
    hit5 = (pos < k).astype(float)
    mrr = np.where(pos < 10, 1.0 / (pos + 1.0), 0.0)
    r1 = (pos == 0).astype(float)
    dd = np.where(top[:, :k] >= 0, w.dom_true[np.maximum(top[:, :k], 0)], -1)
    off = np.mean(dd != dom_task[:, None], axis=1)
    return {"hit5": hit5, "mrr": mrr, "r1": r1, "off": off}


def quality(m, base, noise, G, H):
    return np.clip(base + G * m["hit5"] - H * m["off"] + noise, 0, 1)


def run_seed(w, seed, phase, cfgs, G=0.30, H=0.10, k=5, nq_cap=150):
    rng = np.random.RandomState(seed * 1000 + 17)
    pool = w.bucket[phase]
    nq = min(nq_cap, int(0.75 * len(pool)))
    gold = rng.choice(pool, nq, replace=False)
    typo = rng.rand(nq) < 0.5
    var_raw, var_sp = np.where(typo, 1, 0), np.where(typo, 3, 2)
    qraw = w.Q[gold, var_raw]
    qsp = w.Q[gold, var_sp]
    kraw = ["query_typo" if t else "query" for t in typo]
    ksp = ["qs_typo" if t else "qs_clean" for t in typo]
    Bs = w.qbow([w.docs[g][kk] for g, kk in zip(gold, ksp)])
    Br = w.qbow([w.docs[g][kk] for g, kk in zip(gold, kraw)])
    dom_task = w.dom[gold]  # index label (oracle / router accuracy)
    dom_true = w.dom_true[gold]  # real domain (harm model)
    base = rng.normal(0.45, 0.10, nq)
    noise = rng.normal(0, 0.05, nq)
    allids = np.arange(w.n)
    res = {}
    # pooled arms
    Fp = w.fused(qsp, Bs, allids, w.pool)
    Fr = w.fused(qraw, Br, allids, w.pool)
    arms = {"none": None, "dense_raw": top_idx(qraw @ w.D.T), "dense_spell": top_idx(qsp @ w.D.T), "hyb_raw": top_idx(Fr), "hyb_spell": top_idx(Fp)}
    # oracle
    arms["oracle_H"] = search_domain(w, qsp, Bs, dom_task, np.ones(nq, bool))
    mask = np.where(w.dom[None, :] == dom_task[:, None], 0.0, -1e3)
    arms["oracle_M"] = top_idx(Fp + mask)
    # domain arms from cfgs: (name, kind, arm, param, tau)
    probs = {}
    for name, kind, arm, par, tau in cfgs:
        if kind not in probs:
            probs[kind] = route_probs(w, qsp, Fp, kind)
        P = probs[kind]
        order = np.argsort(-P, 1)
        pred = order[:, 0]
        margin = P[np.arange(nq), pred] - P[np.arange(nq), order[:, 1]]
        gate = (margin >= tau) & w.domain_ok
        if arm == "H":
            top = search_domain(w, qsp, Bs, pred, gate)
            fb = arms["hyb_spell"]
            top = np.where(gate[:, None], top, fb)
        elif arm == "P":
            F2 = Fp + par * P[:, w.dom]
            top = top_idx(F2)
            if not w.domain_ok:
                top = arms["hyb_spell"]
        else:  # M: mask to top-m domains
            m_ = int(par)
            allowed = np.zeros((nq, w.nd), bool)
            for r in range(m_):
                allowed[np.arange(nq), order[:, r]] = True
            msk = np.where(allowed[:, w.dom], 0.0, -1e3)
            top = top_idx(Fp + msk)
            top = np.where(gate[:, None], top, arms["hyb_spell"])
        arms[name] = top
        res.setdefault("_router_acc_" + kind, float(np.mean(pred == dom_task)))
    spec = w.spec[gold]
    conf = spec <= w.conf_thr
    out = {}
    for a, top in arms.items():
        if a == "none":
            q = np.clip(base + noise, 0, 1)
            out[a] = {"q": q, "hit5": np.zeros(nq), "mrr": np.zeros(nq), "r1": np.zeros(nq), "off": np.zeros(nq)}
            continue
        m = metrics(top, gold, dom_true, w, k)
        m["q"] = quality(m, base, noise, G, H)
        out[a] = m
    summ = {}
    for a, m in out.items():
        s = {}
        for key in ("q", "hit5", "mrr", "r1", "off"):
            s[key] = float(m[key].mean())
            s[key + "_typo"] = float(m[key][typo].mean()) if typo.any() else float("nan")
            s[key + "_clean"] = float(m[key][~typo].mean()) if (~typo).any() else float("nan")
            s[key + "_conf"] = float(m[key][conf].mean()) if conf.any() else float("nan")
        summ[a] = s
    summ["_meta"] = {"nq": nq, "typo_frac": float(typo.mean()), "conf_frac": float(conf.mean()),
                     **{k_: v for k_, v in res.items()}}
    return summ


def seeds_of(phase):
    return {"tune": range(1, 9), "held": range(31, 61), "confirm": range(61, 91)}[phase]


def default_cfgs():
    cf = []
    for kind in ("cent", "knn", "fuse", "lr", "lrk"):
        for tau in (0.0, 0.2, 0.4):
            cf.append(("H|%s|t%.1f" % (kind, tau), kind, "H", 0, tau))
        for lam in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            cf.append(("P|%s|l%.1f" % (kind, lam), kind, "P", lam, 0.0))
        for m in (1, 2, 3):
            for tau in (0.0, 0.2):
                cf.append(("M|%s|m%d|t%.1f" % (kind, m, tau), kind, "M", m, tau))
    return cf


def collect(worlds, phase, cfgs, force_ok=False, **kw):
    R = {}
    for w in worlds:
        saved = w.domain_ok
        if force_ok:
            w.domain_ok = True
        R[w.tag] = [run_seed(w, s, phase, cfgs, **kw) for s in seeds_of(phase)]
        w.domain_ok = saved
    return R


def mean_metric(runs, arm, key="q"):
    return np.array([r[arm][key] for r in runs])


def paired(a, b, tost=None):
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, se = d.mean(), d.std(ddof=1) / math.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(m / max(se, 1e-12)), n - 1)
    out = {"diff": float(m), "lo": float(m - t * se), "hi": float(m + t * se), "p": float(p), "n": n}
    if tost is not None:
        pl = stats.t.sf((m + tost) / max(se, 1e-12), n - 1)
        pu = stats.t.cdf((m - tost) / max(se, 1e-12), n - 1)
        out["tost_p"] = float(max(pl, pu))
        out["equiv"] = bool(out["tost_p"] < 0.05)
    return out


def holm(ps, alpha=0.05):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    ok = [False] * len(ps)
    for rank, i in enumerate(idx):
        if ps[i] <= alpha / (len(ps) - rank):
            ok[i] = True
        else:
            break
    return ok


def tune(worlds):
    cfgs = default_cfgs()
    R = collect(worlds, "tune", cfgs)
    rows = []
    for c in cfgs:
        name = c[0]
        gains = []
        for w in worlds:
            if w.tag.endswith("+shuf"):
                continue
            runs = R[w.tag]
            gains.append(mean_metric(runs, name).mean() - mean_metric(runs, "hyb_spell").mean())
        rows.append((float(np.mean(gains)), name, gains))
    rows.sort(reverse=True)
    best_by_arm = {}
    for g, name, gains in rows:
        best_by_arm.setdefault(name.split("|")[0], (name, g, gains))
    pooled = {a: float(np.mean([mean_metric(R[w.tag], a).mean() for w in worlds if not w.tag.endswith("+shuf")])) for a in ("dense_raw", "dense_spell", "hyb_raw", "hyb_spell")}
    best_pooled = max(("dense_spell", "hyb_spell"), key=lambda a: pooled[a])
    out = {"grid": [(n, g, gs) for g, n, gs in rows], "best_by_family": best_by_arm, "pooled_tune_q": pooled, "best_pooled": best_pooled,
           "frozen_domain_arm": rows[0][1], "objective": "mean over non-control worlds of e2e quality(q) minus hyb_spell, seeds 1-8"}
    json.dump(out, open(os.path.join(RES, "h20_retrieval_tune.json"), "w"), indent=1)
    print("TUNE top 8:")
    for g, n, gs in rows[:8]:
        print("  %-22s %+.4f  per-world %s" % (n, g, " ".join("%+.4f" % x for x in gs)))
    print("best by family:", {k: (v[0], round(v[1], 4)) for k, v in best_by_arm.items()})
    print("pooled tune q:", pooled)
    return out


def evaluate(worlds, phase, tuned, **kw):
    fam = tuned["best_by_family"]
    names = [v[0] for v in fam.values()]
    cfgs = [c for c in default_cfgs() if c[0] in names]
    R = collect(worlds, phase, cfgs, **kw)
    fz = tuned["frozen_domain_arm"]
    R2 = collect(worlds, phase, [c for c in default_cfgs() if c[0] == fz], force_ok=True, **kw)
    for tag in R:
        for r, r2 in zip(R[tag], R2[tag]):
            r[fz + "#nocheck"] = r2[fz]
    names = names + [fz + "#nocheck"]
    res = {"phase": phase, "tuned": {"frozen": tuned["frozen_domain_arm"], "families": {k: v[0] for k, v in fam.items()}, "best_pooled": tuned["best_pooled"]}, "worlds": {}}
    bp = tuned["best_pooled"]
    arms = ["none", "dense_raw", "dense_spell", "hyb_raw", "hyb_spell", "oracle_H", "oracle_M"] + names
    allp = []
    keys = []
    for w in worlds:
        runs = R[w.tag]
        W = {"n_docs": w.n, "domains": w.names, "purity": w.purity, "purity_base": w.purity_base, "excess": w.excess, "domain_mode_enabled": bool(w.domain_ok), "arms": {}, "contrasts": {}}
        for a in arms:
            W["arms"][a] = {k: float(np.mean([r[a][k] for r in runs])) for k in runs[0][a].keys()}
        W["router_acc"] = {k: float(np.mean([r["_meta"][k] for r in runs])) for k in runs[0]["_meta"] if k.startswith("_router")}
        for a in names + ["oracle_H", "oracle_M", "hyb_spell", "dense_spell"]:
            for ref in ("none", bp):
                if a == ref:
                    continue
                for key in ("q", "hit5", "mrr"):
                    c = paired(mean_metric(runs, a, key), mean_metric(runs, ref, key), tost=0.01)
                    W["contrasts"]["%s_vs_%s_%s" % (a, ref, key)] = c
                    if key == "q":
                        allp.append(c["p"])
                        keys.append((w.tag, a, ref))
        for sub in ("typo", "clean", "conf"):
            for a in names + [bp]:
                W["arms"][a]["_sub_" + sub] = {k: float(np.nanmean([r[a][k + "_" + sub] for r in runs])) for k in ("q", "hit5", "mrr")}
        res["worlds"][w.tag] = W
    ok = holm(allp)
    res["holm_q_contrasts"] = [{"world": k[0], "arm": k[1], "ref": k[2], "p": p, "holm_sig": o} for k, p, o in zip(keys, allp, ok)]
    return res


def print_eval(res):
    fam = res["tuned"]["families"]
    print("== phase", res["phase"], "frozen:", res["tuned"]["frozen"], "families:", fam, "best pooled:", res["tuned"]["best_pooled"])
    for wt, W in res["worlds"].items():
        print("-- %s  n=%d  structure purity %.3f (base %.3f, excess %+.3f) domain_mode=%s router_acc=%s" % (wt, W["n_docs"], W["purity"], W["purity_base"], W["excess"], W["domain_mode_enabled"], {k[-4:]: round(v, 3) for k, v in W["router_acc"].items()}))
        for a, m in W["arms"].items():
            print("   %-22s q %.4f  hit5 %.3f  mrr %.3f  r1 %.3f  off %.3f" % (a, m["q"], m["hit5"], m["mrr"], m["r1"], m["off"]))
        bp = res["tuned"]["best_pooled"]
        for a in list(fam.values()) + [res["tuned"]["frozen"], "oracle_H", "oracle_M"]:
            for key in ("q", "hit5", "mrr"):
                c = W["contrasts"].get("%s_vs_%s_%s" % (a, bp, key))
                if c:
                    print("   %s - %s  [%s] %+.4f [%+.4f,%+.4f] p=%.3g%s" % (a, bp, key, c["diff"], c["lo"], c["hi"], c["p"], ("  TOST+-.01 %s" % c["equiv"]) if key == "q" else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--phase", default=None)
    ap.add_argument("--sens", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    worlds = []
    for t in WORLDS:
        worlds.append(World(t))
    for t in WORLDS:
        worlds.append(World(t, shuffle=True))
    for w in worlds:
        print("[world] %-12s n=%d dom=%d purity %.3f base %.3f excess %+.3f domain_ok=%s conf_thr(spec)=%.3f" % (w.tag, w.n, w.nd, w.purity, w.purity_base, w.excess, w.domain_ok, w.conf_thr), flush=True)
    if a.tune:
        tune(worlds)
        return
    tuned = json.load(open(os.path.join(RES, "h20_retrieval_tune.json")))
    tuned = {"best_by_family": {k: tuple(v) for k, v in tuned["best_by_family"].items()}, "frozen_domain_arm": tuned["frozen_domain_arm"], "best_pooled": tuned["best_pooled"]}
    if a.phase:
        res = evaluate(worlds, a.phase, tuned)
        print_eval(res)
        json.dump(res, open(a.out or os.path.join(RES, "h20_retrieval_%s.json" % a.phase), "w"), indent=1)
    if a.sens:
        sens = {}
        for label, kw in [("G.15", dict(G=0.15)), ("G.45", dict(G=0.45)), ("H0", dict(H=0.0)), ("H.20", dict(H=0.20)), ("k3", dict(k=3)), ("k8", dict(k=8))]:
            r = evaluate(worlds, a.sens, tuned, **kw)
            fz = tuned["frozen_domain_arm"]
            bp = tuned["best_pooled"]
            sens[label] = {wt: {"frozen_minus_pooled_q": W["contrasts"].get("%s_vs_%s_q" % (fz, bp), {}).get("diff"), "none_q": W["arms"]["none"]["q"], "pooled_q": W["arms"][bp]["q"], "frozen_q": W["arms"][fz]["q"]} for wt, W in r["worlds"].items()}
            print(label, {wt: round(v["frozen_minus_pooled_q"], 4) for wt, v in sens[label].items()}, flush=True)
        json.dump(sens, open(os.path.join(RES, "h20_retrieval_sens_%s.json" % a.sens), "w"), indent=1)


if __name__ == "__main__":
    main()
