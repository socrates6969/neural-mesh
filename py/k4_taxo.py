"""k4-1: AUTOMATIC TAXONOMY of the knowledge store (k-means on MiniLM paragraph vectors + c-TF-IDF labels; optional ward tree over the centroids) and
TWO-STAGE retrieval (coarse cluster routing -> fine hybrid z+BM25 search inside the selected clusters) vs the existing best FLAT hybrid z(.7)+BM25 + spell,
with a pre-declared cluster-confidence FALLBACK to flat search.
Worlds: h10 A (tune 1-8 / held 31-60 / confirm 61-90, same pool fresh draws), h10 B (independent confirm 61-90), h20 pydoc9 / crates8 (held 31-60, confirm 61-90).
Selection rule (pre-declared, tune = world A seeds 1-8 only): the config with the smallest scan cost whose tune MRR@10 >= flat MRR@10 - 0.005.
Outcome model (SYNTHETIC, only for pydoc9/crates8 where provenance domains exist): the h20 end-to-end quality  q = clip(base + .30*hit@5 - .10*offdomain@5 + noise).
Usage: python py/k4_taxo.py            (writes results/k4_taxo.json, results/k4_taxo.txt)"""
import itertools, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4_common import *
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster

K_REL = [0.25, 0.5, 1.0, 2.0]
MS = [1, 2, 3, 4]
TAUS = [0.0, 0.02, 0.04, 0.08]
TOL = 0.005
G, H_, KK = 0.30, 0.10, 5


class Taxo:
    """k-means taxonomy of a store; c-TF-IDF labels; centroid + label routers"""

    def __init__(self, store, k, seed=0):
        self.store = store
        n = store.n
        km = KMeans(n_clusters=k, n_init=3, random_state=seed).fit(store.D)
        self.k = k
        self.cl = km.labels_
        cen = np.stack([store.D[self.cl == c].mean(0) if (self.cl == c).any() else km.cluster_centers_[c] for c in range(k)])
        self.cent = cen / np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-9)
        # c-TF-IDF (BERTopic): class tf of concatenated docs, weight log(1 + A / f_t)
        V = len(store.vocab)
        R = sp.csr_matrix((np.ones(n), (self.cl, np.arange(n))), shape=(k, n))
        TF = (R @ store.C).tocsr().astype(np.float32)
        ft = np.asarray(TF.sum(0)).ravel() + 1e-9
        A = float(TF.sum() / k)
        idf = np.log(1 + A / ft).astype(np.float32)
        rowsum = np.asarray(TF.sum(1)).ravel() + 1e-9
        self.ctf = sp.csr_matrix(TF.multiply(1.0 / rowsum[:, None]).multiply(idf[None, :]))
        inv = {j: w for w, j in store.vocab.items()}
        self.labels = []
        for c in range(k):
            r = self.ctf.getrow(c)
            if r.nnz == 0:
                self.labels.append([]); continue
            o = np.argsort(-r.data)[:6]
            self.labels.append([inv[r.indices[i]] for i in o])
        self.members = [np.where(self.cl == c)[0] for c in range(k)]
        self.sizes = np.array([len(m) for m in self.members])

    def tree(self, g):
        Z = linkage(self.cent, method="ward")
        grp = fcluster(Z, t=g, criterion="maxclust") - 1
        return grp

    def route_scores(self, qE, qB, router):
        s = qE @ self.cent.T
        if router == "cent":
            return s
        lab = (qB @ self.ctf.T).toarray()
        return 0.5 * zrows(s) + 0.5 * zrows(lab) if router == "cent+label" else s


def masked_fused(dense, bm, M):
    """hybrid z-fusion where z-scores are computed over the masked index only (M: bool nq x n); outside the mask -> -inf"""
    Mf = M.astype(np.float32)
    cnt = np.maximum(Mf.sum(1, keepdims=True), 1)

    def z(X):
        mu = (X * Mf).sum(1, keepdims=True) / cnt
        var = (((X - mu) ** 2) * Mf).sum(1, keepdims=True) / cnt
        return (X - mu) / np.maximum(np.sqrt(var), 1e-9)
    F = ALPHA * z(dense) + (1 - ALPHA) * z(bm)
    F = np.where(M, F, -1e9)
    return F


def search(store, tx, q, cfg, dense, bm, Ffull, topflat, grp=None):
    """cfg = dict(kind 'flat'|'ts'|'tree', router, m, arm 'zsub'|'mask', tau, g, mg).  returns (top (nq,10), cost fraction per query, gate bool per query)"""
    nq = q.n
    n = store.n
    if cfg["kind"] == "flat":
        return topflat, np.ones(nq), np.zeros(nq, bool)
    S = tx.route_scores(q.E, q.B, cfg["router"])
    if cfg["kind"] == "ts":
        order = np.argsort(-S, 1)
        m = cfg["m"]
        sel = order[:, :m]
        gap = np.take_along_axis(S, order[:, m - 1:m], 1)[:, 0] - np.take_along_axis(S, order[:, m:m + 1], 1)[:, 0]
        allowed = np.zeros((nq, tx.k), bool)
        np.put_along_axis(allowed, sel, True, 1)
        M = allowed[:, tx.cl]
        over = tx.k
    else:  # tree: route to coarse groups of the ward tree over centroids, search every doc of the selected groups
        grp_of_cl = grp
        g = grp.max() + 1
        gs = np.stack([S[:, grp_of_cl == j].max(1) for j in range(g)], 1)
        order = np.argsort(-gs, 1)
        mg = cfg["mg"]
        gap = np.take_along_axis(gs, order[:, mg - 1:mg], 1)[:, 0] - np.take_along_axis(gs, order[:, min(mg, g - 1):min(mg, g - 1) + 1], 1)[:, 0]
        allowed = np.zeros((nq, g), bool)
        np.put_along_axis(allowed, order[:, :mg], True, 1)
        M = allowed[:, grp_of_cl[tx.cl]]
        over = tx.k
    gate = gap >= cfg["tau"]  # confident -> two-stage; else FALLBACK to flat
    if cfg["tau"] == 0.0:
        gate = np.ones(nq, bool)
    if cfg["arm"] == "mask":
        F = np.where(M, Ffull, -1e9)
    else:
        F = masked_fused(dense, bm, M)
    top = top_idx(F, 10)
    top = np.where(gate[:, None], top, topflat)
    cost = np.where(gate, (M.sum(1) + over) / n, 1.0 + over / n)  # fallback pays routing + flat
    return top, cost, gate


def eval_cfgs(store, tx, qsets, cfgs, extra=None, grps=None):
    """-> {cfg name: list over seeds of dict metric -> per-query array (+ cost, gate)}; flat included"""
    out = {c["name"]: [] for c in cfgs}
    for q in qsets:
        dense, bm = store.parts(q.E, q.B)
        Ffull = ALPHA * zrows(dense) + (1 - ALPHA) * zrows(bm)
        topflat = top_idx(Ffull, 10)
        for c in cfgs:
            grp = grps.get((c.get("k_rel"), c.get("g"))) if grps else None
            top, cost, gate = search(store, tx[c["k_rel"]] if c["kind"] != "flat" else None, q, c, dense, bm, Ffull, topflat, grp)
            m = rank_metrics(top, q.rel)
            m["cost"] = cost; m["gate"] = gate.astype(float); m["top"] = top; m["qid"] = q.qid
            out[c["name"]].append(m)
    return out


def mk_cfgs():
    cfgs = [dict(name="flat", kind="flat", k_rel=None)]
    for kr in K_REL:
        for router in ("cent", "cent+label"):
            for m in MS:
                for arm in ("zsub", "mask"):
                    for tau in TAUS:
                        cfgs.append(dict(name="ts|k%.2f|%s|m%d|%s|t%.2f" % (kr, router, m, arm, tau), kind="ts", k_rel=kr, router=router, m=m, arm=arm, tau=tau))
    for kr in (0.5, 1.0):
        for g in (4, 8, 16):
            for mg in (1, 2):
                for tau in TAUS:
                    cfgs.append(dict(name="tree|k%.2f|g%d|mg%d|t%.2f" % (kr, g, mg, tau), kind="tree", k_rel=kr, router="cent", arm="zsub", g=g, mg=mg, tau=tau))
    return cfgs


def summarize(res, name, keys=("mrr", "hit1", "hit3", "hit8", "f1@3", "cost", "gate")):
    return {k: float(np.mean([r[k].mean() for r in res[name]])) for k in keys}


def e2e(world, res, name, seeds, qsets):
    """h20 outcome model (SYNTHETIC): returns per-seed mean quality for arm `name`; common random numbers across arms (rng by seed)"""
    dom = world.dom
    qs = []
    for s, q, r in zip(seeds, qsets, res[name]):
        rng = np.random.RandomState(s * 1000 + 99)
        base = rng.normal(0.45, 0.10, q.n); noise = rng.normal(0, 0.05, q.n)
        top = r["top"][:, :KK]
        hit = np.array([1.0 if any(d in set(rel.tolist()) for d in row) else 0.0 for row, rel in zip(top, q.rel)])
        gold_dom = np.array([dom[rel[0]] for rel in q.rel])
        off = np.mean(dom[top] != gold_dom[:, None], axis=1)
        qs.append(float(np.clip(base + G * hit - H_ * off + noise, 0, 1).mean()))
    return np.array(qs)


def main():
    t0 = time.time()
    cfgs = mk_cfgs()
    worlds = {"A": H10World("A"), "B": H10World("B"), "pydoc9": H20World("pydoc9"), "crates8": H20World("crates8")}
    tx = {}
    info = {}
    for wn, w in worlds.items():
        st = w.store
        tx[wn] = {}
        for kr in K_REL:
            k = max(4, int(round(kr * math.sqrt(st.n))))
            tx[wn][kr] = Taxo(st, k)
        base = tx[wn][1.0]
        inf = dict(n=st.n, k_by_krel={str(kr): tx[wn][kr].k for kr in K_REL})
        if hasattr(w, "dom"):
            inf["nmi_domain_by_krel"] = {str(kr): float(normalized_mutual_info_score(w.dom, tx[wn][kr].cl)) for kr in K_REL}
        inf["label_examples_k1.0"] = [dict(c=c, size=int(base.sizes[c]), label=base.labels[c]) for c in np.argsort(-base.sizes)[:6]]
        srcs = np.array([m.get("src", "") for m in st.meta])
        inf["nmi_src_by_krel"] = {str(kr): float(normalized_mutual_info_score(srcs, tx[wn][kr].cl)) for kr in K_REL}
        info[wn] = inf
        print(wn, "taxonomy built", inf["k_by_krel"], flush=True)
    grps = {}
    for wn in worlds:
        grps[wn] = {(kr, g): tx[wn][kr].tree(g) for kr in (0.5, 1.0) for g in (4, 8, 16)}

    def qsets(wn, phase):
        w = worlds[wn]
        if isinstance(w, H10World):
            return list(w.seeds(phase)), [w.qset(s) for s in w.seeds(phase)]
        return list(w.seeds(phase)), [w.qset(s, phase) for s in w.seeds(phase)]

    # ------------------------------------------------ tune on world A seeds 1-8
    sd, qs = qsets("A", "tune")
    tune = eval_cfgs(worlds["A"].store, tx["A"], qs, cfgs, grps=grps["A"])
    flat_t = summarize(tune, "flat")
    rows = []
    for c in cfgs[1:]:
        s = summarize(tune, c["name"])
        rows.append((c["name"], s))
    ok = [(n, s) for n, s in rows if s["mrr"] >= flat_t["mrr"] - TOL]
    pick = min(ok, key=lambda x: (x[1]["cost"], -x[1]["mrr"]))[0] if ok else None
    # also record the best config per kind for reporting (cheapest within tolerance)
    pick_by = {}
    for kind in ("ts", "tree"):
        okk = [(n, s) for n, s in ok if n.startswith(kind + "|")]
        pick_by[kind] = min(okk, key=lambda x: (x[1]["cost"], -x[1]["mrr"]))[0] if okk else None
    # unconstrained best-MRR ts config w/o fallback loss: also record the best fallback-gated (tau>0) config
    gated = [(n, s) for n, s in ok if not n.endswith("t0.00")]
    pick_gated = min(gated, key=lambda x: (x[1]["cost"], -x[1]["mrr"]))[0] if gated else None
    print("tune flat", flat_t, "\n pick", pick, pick_by, "gated pick", pick_gated, flush=True)
    tune_tab = {n: s for n, s in rows}
    cfg_by = {c["name"]: c for c in cfgs}
    frozen = [n for n in {pick, pick_by["ts"], pick_by["tree"], pick_gated} if n]
    # ------------------------------------------------ held / confirm with frozen configs (+ the full grid on held for the Pareto)
    result = dict(info=info, tune_flat=flat_t, pick=pick, pick_by=pick_by, pick_gated=pick_gated, tune_table=tune_tab, phases={})
    lines = []
    for phase, wns in (("held", ["A", "pydoc9", "crates8"]), ("confirm", ["A", "B", "pydoc9", "crates8"])):
        for wn in wns:
            sd, qs = qsets(wn, phase)
            use = [cfg_by["flat"]] + [cfg_by[n] for n in frozen]
            # the cfg's own taxonomy resolution (k_rel) applies to every world (scale-free via sqrt(n))
            res = eval_cfgs(worlds[wn].store, tx[wn], qs, use, grps=grps[wn])
            full = None
            if phase == "held":
                full = eval_cfgs(worlds[wn].store, tx[wn], qs[:10], cfgs, grps=grps[wn])  # Pareto on the first 10 held seeds
            blk = dict(summ={n: summarize(res, n) for n in ["flat"] + frozen}, contrasts={})
            for n in frozen:
                cc = {}
                for key in ("mrr", "hit1", "hit3", "hit8", "f1@3"):
                    d = np.array([a[key].mean() - b[key].mean() for a, b in zip(res[n], res["flat"])])
                    m, lo, hi, p = t_ci(d)
                    cc[key] = dict(d=m, lo=lo, hi=hi, p=p, tost_pm01=bool(tost(d, 0.01)), tost_pm02=bool(tost(d, 0.02)))
                # query-cluster bootstrap on MRR / hit8 (distinct queries across seeds)
                pq = np.concatenate([a["mrr"] - b["mrr"] for a, b in zip(res[n], res["flat"])])
                qid = np.concatenate([a["qid"] for a in res[n]])
                cc["mrr_boot"] = dict(zip(("d", "lo", "hi", "p"), cluster_boot(pq, qid, 1000)))
                pq = np.concatenate([a["hit8"] - b["hit8"] for a, b in zip(res[n], res["flat"])])
                cc["hit8_boot"] = dict(zip(("d", "lo", "hi", "p"), cluster_boot(pq, qid, 1000)))
                cc["cost"] = blk["summ"][n]["cost"]; cc["gate_on"] = blk["summ"][n]["gate"]
                blk["contrasts"][n] = cc
            if hasattr(worlds[wn], "dom"):
                allq = {n: e2e(worlds[wn], res, n, sd, qs) for n in ["flat"] + frozen}
                blk["e2e_q_SYNTHETIC"] = {n: float(v.mean()) for n, v in allq.items()}
                blk["e2e_dq_vs_flat_SYNTHETIC"] = {n: dict(zip(("d", "lo", "hi", "p"), t_ci(allq[n] - allq["flat"]))) for n in frozen}
                # domain-purity of the context under the arms
                blk["offdomain5"] = {n: float(np.mean([np.mean(worlds[wn].dom[r["top"][:, :KK]] != np.array([worlds[wn].dom[rel[0]] for rel in q.rel])[:, None]) for r, q in zip(res[n], qs)])) for n in ["flat"] + frozen}
            if full is not None:
                par = []
                for c in cfgs:
                    s = summarize(full, c["name"])
                    par.append((c["name"], s["cost"], s["mrr"], s["hit8"]))
                blk["pareto10"] = par
            result["phases"]["%s|%s" % (phase, wn)] = blk
            print(phase, wn, "done %.0fs" % (time.time() - t0), flush=True)
    jdump(result, "k4_taxo.json")
    # text report
    L = ["k4-1 TAXONOMY + TWO-STAGE RETRIEVAL  (real text; typos synthetic)  tune pick=%s ; by kind=%s ; cheapest gated=%s" % (pick, pick_by, pick_gated), "tune flat: %s" % flat_t, ""]
    for key, blk in result["phases"].items():
        L.append("== %s" % key)
        for n, s in blk["summ"].items():
            L.append("  %-44s mrr %.4f hit1 %.4f hit8 %.4f f1@3 %.4f cost %.3f gate_on %.2f" % (n, s["mrr"], s["hit1"], s["hit8"], s["f1@3"], s["cost"], s["gate"]))
        for n, cc in blk["contrasts"].items():
            L.append("  d vs flat  %-40s mrr %+.4f [%+.4f,%+.4f] boot[%+.4f,%+.4f] TOST.01=%s | hit8 %+.4f [%+.4f,%+.4f] boot[%+.4f,%+.4f] TOST.02=%s | hit1 %+.4f" % (n, cc["mrr"]["d"], cc["mrr"]["lo"], cc["mrr"]["hi"], cc["mrr_boot"]["lo"], cc["mrr_boot"]["hi"], cc["mrr"]["tost_pm01"], cc["hit8"]["d"], cc["hit8"]["lo"], cc["hit8"]["hi"], cc["hit8_boot"]["lo"], cc["hit8_boot"]["hi"], cc["hit8"]["tost_pm02"], cc["hit1"]["d"]))
        if "e2e_dq_vs_flat_SYNTHETIC" in blk:
            for n, v in blk["e2e_dq_vs_flat_SYNTHETIC"].items():
                L.append("  e2e dq (SYNTHETIC) %-40s %+.4f [%+.4f,%+.4f]  offdomain5 flat %.3f arm %.3f" % (n, v["d"], v["lo"], v["hi"], blk["offdomain5"]["flat"], blk["offdomain5"][n]))
    open(os.path.join(RES, "k4_taxo.txt"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
