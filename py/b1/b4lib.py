"""B4 shared library: per-query component scores (dense z, BM25 z) on (a) the real h10 store (family H = heading queries, S = sentence->neighbour-paragraph queries;
TUNE queries = seeds 1-8, POOL queries = seeds 31-90, query-disjoint from TUNE; POOL split once into two disjoint halves) and (b) the hand-written domain store
(144 docs; world A keyword question, world B paraphrase with deterministic typos; items split into 3 DISJOINT thirds by doc index mod 3 -> TUNE/HALF1/HALF2).
Per query we keep zD (dense z-scored scores over all docs) and zB per BM25 parameterisation so any alpha rule can be evaluated by arithmetic only."""
import json, os, random
import numpy as np
from lib import *
import data_domain

_cache = {}


def real_queries():
    W = RealWorld("A")
    out = {}
    for s in list(range(1, 9)) + list(range(31, 91)):
        v = W.inst.get(str(s))
        for r in v["rows"]:
            out.setdefault(r["q"], r["t"][0][0])
    seeds_tune = set()
    for s in range(1, 9):
        for r in W.inst[str(s)]["rows"]: seeds_tune.add(r["q"])
    pool = sorted(q for q in out if q not in seeds_tune)
    rng = np.random.RandomState(20260605)
    ids = list(pool); rng.shuffle(ids)
    h = len(ids) // 2
    split = {q: "TUNE" for q in seeds_tune}
    for q in ids[:h]: split[q] = "HALF1"
    for q in ids[h:]: split[q] = "HALF2"
    return W, out, split


class RealComp:
    """component scores for all real queries"""
    def __init__(self, bmparams=((1.2, .75), (.9, .5))):
        W, qs, split = real_queries()
        self.W, self.split = W, split
        self.q = sorted(qs)
        self.fam = {q: W.queries[q]["fam"] for q in self.q}
        self.rel = {q: list(W.queries[q]["rel"]) for q in self.q}
        self.bms = {p: BM25(W.docs, *p) for p in bmparams}
        self.zD, self.zB, self.cov, self.qt, self.qid = {}, {p: {} for p in bmparams}, {p: {} for p in bmparams}, {}, {}
        for q in self.q:
            tid = qs[q]; t = W.texts[tid]; self.qt[q] = t; self.qid[q] = tid
            self.zD[q] = z(W.dvec @ W.emb[tid]).astype(np.float32)
            for p, bm in self.bms.items():
                raw = bm.score(t); self.zB[p][q] = z(raw).astype(np.float32)
                seen = {x for x in toks(t) if x in bm.post}
                den = sum(bm.idf[x] for x in seen) * (bm.k1 + 1)
                self.cov[p][q] = float(raw.max() / den) if den > 0 else 0.0
    def rr(self, q, sc):
        best = min(int((sc > sc[d]).sum()) for d in self.rel[q])
        return 1.0 / (1 + best)


class DomainComp:
    def __init__(self, bmparams=((1.2, .75), (.9, .5)), typo_draws=3):
        D = Data()
        items = []
        for dom, lst in data_domain.DOMAINS.items():
            for i, (doc, qk, qp) in enumerate(lst): items.append((dom, i, doc, qk, qp))
        self.docs = [x[2] for x in items]
        self.vecs = np.stack([D.v(t) for t in self.docs])
        self.bms = {p: BM25(self.docs, *p) for p in bmparams}
        self.rows = []  # (id, world, text, vec, relidx, part)
        for j, (dom, i, doc, qk, qp) in enumerate(items):
            part = ["TUNE", "HALF1", "HALF2"][j % 3]
            self.rows.append((f"{j}A", "A", qk, D.v(qk), j, part))
            for d in range(typo_draws):
                r = random.Random(1000 * (d + 1) + j)
                tq = typo(r, qp, 0.12)
                self.rows.append((f"{j}B{d}", "B", tq, D.v(tq), j, part))
        D.save_extra()
        self.zD, self.zB, self.cov = {}, {p: {} for p in bmparams}, {p: {} for p in bmparams}
        for (rid, w, t, v, j, part) in self.rows:
            self.zD[rid] = z(self.vecs @ v).astype(np.float32)
            for p, bm in self.bms.items():
                raw = bm.score(t); self.zB[p][rid] = z(raw).astype(np.float32)
                seen = {x for x in toks(t) if x in bm.post}
                den = sum(bm.idf[x] for x in seen) * (bm.k1 + 1)
                self.cov[p][rid] = float(raw.max() / den) if den > 0 else 0.0


def rr_of(sc, rel):
    """reciprocal rank with TIES COUNTED AGAINST the relevant doc (pessimistic): a rule that creates ties (e.g. a score cap) cannot gain from them"""
    best = min(int((sc >= sc[d]).sum()) - 1 for d in rel)
    return 1.0 / (1 + best)
