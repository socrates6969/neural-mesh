"""k4-4: KNOWLEDGE ACQUISITION POLICY.  Gap log (queries the store cannot support: low top-1 cosine) -> acquisition-priority ranking of the NOT-yet-ingested
documents ("units" = (source, file) real documents) -> EXTEND simulation: ingest the best-ranked next units under a document budget and measure coverage / recall growth
on fresh future queries vs random / largest-first / most-recent (reverse corpus order) / topic-gap / oracle ingestion order.
REAL: text, units, relevance (section / source paragraph membership -> unit).  SYNTHETIC (declared): the DEMAND model -- queries come from a latent topic mix (units clustered into
12 topics by their 'abstract' = mean of the first two paragraph vectors; per seed 3 random hot topics get 80% of the demand) -- and the initial store = 25% random units.
Controls: UNIFORM demand (no hot topics; gap-priority must tie random) and DRIFT (evaluation hot topics differ from the history's hot topics; harm must be bounded).
Protocol: tune = world A seeds 1-8 on the tune query bank (gap threshold theta, softmax temperature T, size exponent gamma by coverage-AUC gain over random);
held = 31-60 (A, pydoc9, crates8) ; confirm = 61-90 (A, B, pydoc9, crates8) frozen.  History queries and evaluation queries come from disjoint halves of the phase bank.
Usage: python py/k4_extend.py -> results/k4_extend.json, results/k4_extend.txt"""
import collections, hashlib, itertools, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4_common import *
from k4_taxo import masked_fused
from sklearn.cluster import KMeans

BUDGETS = [0.05, 0.10, 0.20, 0.30, 0.50]
NH, NE = 300, 300
THETAS = [0.35, 0.45, 0.55]
TEMPS = [0.03, 0.06, 0.12, 0.25]
GAMMAS = [0.0, 1.0]
NTOP = 12
HOT = 3
PHOT = 0.8


class Bank:
    """query bank of a world: E (spelled clean variant), BoW, rel docs, unit of the relevant docs, phase"""

    def __init__(self, w):
        st = w.store
        self.st = st
        if isinstance(w, H10World):
            first = {}
            for seed, blk in w.inst.items():
                if seed == "R":
                    continue
                for r in blk["rows"]:
                    if r["q"] not in first:
                        first[r["q"]] = r["t"][0][1]
            qids = sorted(first)
            self.E = np.stack([w.emb[first[q]] for q in qids])
            self.texts = [w.alltexts[first[q]] for q in qids]
            self.rel = [np.array(w.queries[q]["rel"], dtype=int) for q in qids]
            self.qid = np.array(qids)
            self.unit_of_doc = {}
            ukeys = {}
            self.udoc = np.zeros(st.n, dtype=int)
            for i, d in enumerate(w.docs):
                k = (d["src"], d["file"])
                self.udoc[i] = ukeys.setdefault(k, len(ukeys))
        else:
            n = st.n
            typo = np.array([hnum("ty", w.tag, i) % 2 for i in range(n)]) == 1
            self.E = np.where(typo[:, None], w.Q[:, 3], w.Q[:, 2])
            self.texts = [w.docs[i]["qs_typo" if typo[i] else "qs_clean"] for i in range(n)]
            self.rel = [np.array([i]) for i in range(n)]
            self.qid = np.arange(n)
            ukeys = {}
            self.udoc = np.zeros(n, dtype=int)
            for i, d in enumerate(w.docs):
                self.udoc[i] = ukeys.setdefault(d["src"], len(ukeys))
        self.nu = int(self.udoc.max()) + 1
        self.B = st.qbow(self.texts)
        self.qunit = np.array([self.udoc[r[0]] for r in self.rel])
        ph = np.array([hnum("bank", w.tag, int(q)) % 10 for q in self.qid])
        self.phase = np.where(ph < 2, 0, np.where(ph < 6, 1, 2))  # tune 20% / held 40% / confirm 40%
        self.half = np.array([hnum("half", w.tag, int(q)) % 2 for q in self.qid])
        # unit abstracts: mean of the first two paragraph vectors
        self.members = [[] for _ in range(self.nu)]
        for i in range(st.n):
            self.members[self.udoc[i]].append(i)
        A = np.stack([st.D[m[:2]].mean(0) for m in self.members])
        self.abstract = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-9)
        self.usize = np.array([len(m) for m in self.members])
        km = KMeans(NTOP, n_init=3, random_state=0).fit(self.abstract)
        self.utopic = km.labels_
        self.qtopic = self.utopic[self.qunit]
        self.w = w


def sample_queries(bank, phase, half, n, hot, p_hot, rng):
    ix = np.where((bank.phase == phase) & (bank.half == half))[0]
    by = collections.defaultdict(list)
    for i in ix:
        by[bank.qtopic[i]].append(i)
    topics = [t for t in range(NTOP) if len(by.get(t, [])) > 0]
    hots = [t for t in hot if len(by.get(t, [])) > 0]
    out = []
    while len(out) < n:
        if hots and rng.rand() < p_hot:
            t = hots[rng.randint(len(hots))]
        else:
            t = topics[rng.randint(len(topics))]
        out.append(by[t][rng.randint(len(by[t]))])
    return np.array(out)


def store_mask(bank, units):
    m = np.zeros(bank.nu, bool)
    m[list(units)] = True
    return m[bank.udoc]


def retrieve(bank, qi, dmask, dense, bm):
    """top-8 over the docs in dmask for the queries qi (rows of dense/bm already computed for qi); z-scores over the store"""
    ids = np.where(dmask)[0]
    F = ALPHA * zrows(dense[:, ids]) + (1 - ALPHA) * zrows(bm[:, ids])
    return ids[top_idx(F, 8)]


def run_seed(bank, seed, phase, params, hot_mode="hot"):
    st = bank.st
    rng = np.random.RandomState(seed * 977 + 13)
    nu = bank.nu
    s0 = rng.rand(nu) < 0.25
    s0_units = set(np.where(s0)[0].tolist())
    hot_h = list(rng.choice(NTOP, HOT, replace=False))
    if hot_mode == "uniform":
        hot_e, p = hot_h, 0.0
    elif hot_mode == "drift":
        rest = [t for t in range(NTOP) if t not in hot_h]
        hot_e, p = list(rng.choice(rest, HOT, replace=False)), PHOT
    else:
        hot_e, p = hot_h, PHOT
    hq = sample_queries(bank, phase, 0, NH, hot_h, 0.0 if hot_mode == "uniform" else PHOT, rng)
    eq = sample_queries(bank, phase, 1, NE, hot_e, p, rng)
    # --- gap log on the initial store
    m0 = store_mask(bank, s0_units)
    dh = bank.E[hq] @ st.D.T
    bh = (bank.B[hq] @ st.W.T).toarray()
    smax = np.where(m0[None, :], dh, -1).max(1)
    unans = ~s0[bank.qunit[hq]]
    from scipy.stats import rankdata
    auc = None
    if unans.any() and (~unans).any():
        r = rankdata(-smax)
        auc = float((r[unans].sum() - unans.sum() * (unans.sum() + 1) / 2) / (unans.sum() * (~unans).sum()))
    theta, T, gamma = params["theta"], params["T"], params["gamma"]
    gaps = hq[smax < theta]
    cand = np.array([u for u in range(nu) if not s0[u]])
    Ac = bank.abstract[cand]
    # gap-priority score: expected number of gap queries a unit would satisfy (softmax over candidates of cosine / T)
    if len(gaps):
        S = bank.E[gaps] @ Ac.T
        P = np.exp((S - S.max(1, keepdims=True)) / T)
        P = P / P.sum(1, keepdims=True)
        w_gap = P.sum(0)
    else:
        w_gap = np.zeros(len(cand))
    # topic-gap prior: units of the topics where most gaps fall (ties random)
    tg = np.bincount(bank.qtopic[gaps], minlength=NTOP).astype(float) if len(gaps) else np.zeros(NTOP)  # LEAKS the true topic of failed queries -> labelled topic_oracle
    cen_t = np.stack([bank.abstract[bank.utopic == t].mean(0) if (bank.utopic == t).any() else np.zeros(bank.abstract.shape[1]) for t in range(NTOP)])
    cen_t = cen_t / np.maximum(np.linalg.norm(cen_t, axis=1, keepdims=True), 1e-9)
    if len(gaps):
        St = bank.E[gaps] @ cen_t.T
        Pt = np.exp((St - St.max(1, keepdims=True)) / T)
        tp = (Pt / Pt.sum(1, keepdims=True)).sum(0)  # soft predicted-topic gap mass (no labels)
    else:
        tp = np.zeros(NTOP)
    # oracle: number of eval queries whose relevant unit is u
    orc = np.bincount(bank.qunit[eq], minlength=nu)[cand].astype(float)
    rnd_key = rng.rand(len(cand))
    sizes = bank.usize[cand].astype(float)
    orders = {
        "random": np.argsort(rnd_key),
        "largest": np.argsort(-sizes + 1e-6 * rnd_key),
        "recent": np.argsort(-cand),
        "topic_oracle": np.lexsort((rnd_key, -tg[bank.utopic[cand]])),
        "topic_pred": np.lexsort((rnd_key, -tp[bank.utopic[cand]])),
        "gap": np.argsort(-(w_gap / sizes ** gamma) + 1e-9 * rnd_key),
        "oracle": np.argsort(-(orc / sizes ** gamma) + 1e-9 * rnd_key),
    }
    # --- evaluation of the growth curves
    de = bank.E[eq] @ st.D.T
    be = (bank.B[eq] @ st.W.T).toarray()
    tot = bank.usize[cand].sum()
    out = {}
    for name, order in orders.items():
        cov, rec = [], []
        for bf in BUDGETS:
            budget = bf * tot
            units = set(s0_units); used = 0
            for k in order:
                u = cand[k]
                if used + bank.usize[u] > budget:
                    continue  # skip units that do not fit (greedy knapsack by priority)
                units.add(int(u)); used += bank.usize[u]
            mk = store_mask(bank, units)
            top = retrieve(bank, eq, mk, de, be)
            hit = np.array([float(any(d in set(bank.rel[q].tolist()) for d in row)) for row, q in zip(top, eq)])
            ans = np.array([float(any(mk[d] for d in bank.rel[q])) for q in eq])
            cov.append(ans.mean()); rec.append(hit.mean())
        out[name] = dict(cov=cov, rec=rec, cov_auc=float(np.mean(cov)), rec_auc=float(np.mean(rec)))
    out["_meta"] = dict(n_gap=int(len(gaps)), n_hist=NH, detect_auc=auc, cov0=float(s0[bank.qunit[eq]].mean()), n_cand_docs=int(tot))
    return out


def main():
    t0 = time.time()
    worlds = {"A": H10World("A"), "B": H10World("B"), "pydoc9": H20World("pydoc9"), "crates8": H20World("crates8")}
    banks = {k: Bank(w) for k, w in worlds.items()}
    for k, b in banks.items():
        print(k, "units", b.nu, "bank", len(b.qid), "phase counts", np.bincount(b.phase), flush=True)
    # ------------------------------------------------ tune on A seeds 1-8 (tune bank)
    tab = {}
    for th, T, g in itertools.product(THETAS, TEMPS, GAMMAS):
        rs = [run_seed(banks["A"], s, 0, dict(theta=th, T=T, gamma=g)) for s in range(1, 9)]
        d = np.array([r["gap"]["cov_auc"] - r["random"]["cov_auc"] for r in rs])
        d2 = np.array([r["gap"]["rec_auc"] - r["random"]["rec_auc"] for r in rs])
        tab[(th, T, g)] = (float(d.mean()), float(d2.mean()))
        print("tune", th, T, g, tab[(th, T, g)], flush=True)
    best = max(tab, key=lambda k: tab[k][0])
    params = dict(theta=best[0], T=best[1], gamma=best[2])
    print("frozen", params, flush=True)
    L = ["k4-4 ACQUISITION POLICY (real text/relevance; demand model SYNTHETIC)", "tune table (theta,T,gamma) -> (cov_auc gain, rec_auc gain vs random): %s" % {str(k): v for k, v in tab.items()}, "frozen %s" % params, ""]
    res = {}
    plan = (("held", 1, ["A", "pydoc9", "crates8"], range(31, 61)), ("confirm", 2, ["A", "B", "pydoc9", "crates8"], range(61, 91)))
    for label, ph, wns, seeds in plan:
        for wn in wns:
            for mode in ("hot", "uniform", "drift"):
                rs = [run_seed(banks[wn], s, ph, params, mode) for s in seeds]
                key = "%s|%s|%s" % (label, wn, mode)
                blk = dict(mean={a: dict(cov_auc=float(np.mean([r[a]["cov_auc"] for r in rs])), rec_auc=float(np.mean([r[a]["rec_auc"] for r in rs])),
                                       cov=[float(np.mean([r[a]["cov"][i] for r in rs])) for i in range(len(BUDGETS))], rec=[float(np.mean([r[a]["rec"][i] for r in rs])) for i in range(len(BUDGETS))]) for a in ("random", "largest", "recent", "topic_oracle", "topic_pred", "gap", "oracle")},
                           meta=dict(n_gap=float(np.mean([r["_meta"]["n_gap"] for r in rs])), detect_auc=float(np.nanmean([r["_meta"]["detect_auc"] if r["_meta"]["detect_auc"] is not None else np.nan for r in rs])), cov0=float(np.mean([r["_meta"]["cov0"] for r in rs]))), contrast={})
                for a in ("largest", "recent", "topic_oracle", "topic_pred", "gap", "oracle"):
                    for met in ("cov_auc", "rec_auc"):
                        d = np.array([r[a][met] - r["random"][met] for r in rs])
                        m, lo, hi, p = t_ci(d)
                        blk["contrast"]["%s|%s" % (a, met)] = dict(d=m, lo=lo, hi=hi, p=p, tost01=bool(tost(d, 0.01)), tost02=bool(tost(d, 0.02)))
                d = np.array([r["gap"]["cov_auc"] - r["topic_pred"]["cov_auc"] for r in rs])
                m, lo, hi, p = t_ci(d)
                blk["contrast"]["gap-topic_pred|cov_auc"] = dict(d=m, lo=lo, hi=hi, p=p)
                res[key] = blk
                L.append("== %s  (cov0 %.3f, mean gaps %.0f of %d hist queries, gap-detector AUC %.3f)" % (key, blk["meta"]["cov0"], blk["meta"]["n_gap"], NH, blk["meta"]["detect_auc"]))
                for a, v in blk["mean"].items():
                    L.append("   %-10s cov_auc %.4f rec_auc %.4f  cov@budgets %s" % (a, v["cov_auc"], v["rec_auc"], [round(x, 3) for x in v["cov"]]))
                for k2, c in blk["contrast"].items():
                    if k2.startswith("gap|") or k2.startswith("recent") or k2.startswith("largest") or k2.startswith("gap-") or k2.startswith("topic"):
                        L.append("   d %-24s %+.4f [%+.4f,%+.4f] p=%.3g%s" % (k2, c["d"], c["lo"], c["hi"], c["p"], (" TOST.01=%s TOST.02=%s" % (c["tost01"], c["tost02"])) if "tost01" in c else ""))
                print(key, "done %.0fs" % (time.time() - t0), flush=True)
    # Holm across the main gap-vs-random contrasts (cov_auc, hot mode) per phase
    for label in ("held", "confirm"):
        ks = [k for k in res if k.startswith(label) and k.endswith("|hot")]
        ps = [res[k]["contrast"]["gap|cov_auc"]["p"] for k in ks]
        for k, pa in zip(ks, holm(ps)):
            res[k]["contrast"]["gap|cov_auc"]["p_holm"] = float(pa)
            L.append("Holm %s gap-vs-random cov_auc (hot demand): p_holm=%.3g" % (k, pa))
    jdump(dict(params=params, tune={str(k): v for k, v in tab.items()}, results=res), "k4_extend.json")
    open(os.path.join(RES, "k4_extend.txt"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
