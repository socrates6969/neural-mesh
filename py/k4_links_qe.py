"""k4-2b: graph-based QUERY EXPANSION (pseudo-relevance feedback through links): q' = normalize(q + beta * mean(vectors of the linked neighbours of the top-2 hits)); the BM25 leg keeps the
original query; k=8 total slots, same pairs/protocol as py/k4_links.py (beta tuned on A tune pool seeds 1-8; held 31-60; confirm 61-90 + world B).  Graph = union_nc (never the type that
defined the pair) and the degree-matched random control.  Usage: python py/k4_links_qe.py -> results/k4_links_qe.json/.txt"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4_common import *
import k4_links as KL

BETAS = [0.5, 1.0, 2.0]


def run_qe(st, emb, cfgs, phase_pairs, seed):
    rng = np.random.RandomState(seed * 7 + 3)
    by = {}
    for k, p in enumerate(phase_pairs):
        by.setdefault(p["type"], []).append(k)
    sel = []
    for t, ks in by.items():
        ks = np.array(ks)
        sel.extend(ks if len(ks) <= KL.NPAIR else rng.choice(ks, KL.NPAIR, replace=False))
    P = [phase_pairs[k] for k in sorted(sel)]
    qE = np.stack([emb[p["rq"]] for p in P]); qB = st.qbow([p["q"] for p in P])
    F = st.fused(qE, qB); top = top_idx(F, 24)
    ptype = np.array([p["type"] for p in P])
    pid = np.array([p["p1"] * 10 + ["ADJ", "SECNEXT", "DEP", "XREF", "XLINK"].index(p["type"]) for p in P])

    def score(lists):
        a = np.array([KL.hop_metrics(l, p["p1"], set(p["t2"])) for l, p in zip(lists, P)])
        return {"hop1": a[:, 0], "hop2": a[:, 1], "both": a[:, 2], "ptype": ptype, "pid": pid}
    res = {"flat": score([list(t[:KL.KB]) for t in top])}
    for gname, gfun in cfgs.items():
        for beta in BETAS:
            lists = []
            for k, p in enumerate(P):
                g = gfun(p["type"])
                nb = []
                for s in top[k][:2]:
                    nb.extend(g[s].keys())
                nb = [j for j in nb if j not in set(top[k][:2])][:6]
                if not nb:
                    lists.append(list(top[k][:KL.KB])); continue
                e = qE[k] + beta * st.D[nb].mean(0)
                e = e / np.linalg.norm(e)
                f = st.fused(e[None], qB[k:k + 1])
                lists.append(list(top_idx(f, 8)[0]))
            res["QE|%s|b%.1f" % (gname, beta)] = score(lists)
    return res


def main():
    P = json.load(open(os.path.join(RES, "k4_mh_pairs.json")))
    emb = np.fromfile(os.path.join(RES, "k4_embed.f32"), dtype=np.float32).reshape(-1, 384)
    for p in P["A"]:
        if p["type"] in ("XREF", "XLINK"):
            p["pool"] = "held"
    wA, stA = KL.build_store("A", P["A"], emb)
    wB, stB = KL.build_store("B", P["B"], emb)
    GA = KL.graphs_for("A", wA, stA); GB = KL.graphs_for("B", wB, stB)
    kinds = ["adj", "sec", "rare", "dep", "xref", "xlink"]
    gtkind = {"ADJ": ["adj", "sec"], "SECNEXT": ["adj", "sec"], "DEP": ["dep"], "XREF": ["xref"], "XLINK": ["xlink"]}
    UA = KL.union_graph(GA, kinds, stA.n)
    UA_nc = {t: KL.union_graph(GA, [k for k in kinds if k not in gtkind[t]], stA.n) for t in gtkind}
    SH = KL.shuffled_like(UA, stA.n)
    cfgA = {"union_nc": lambda t: UA_nc[t], "shuf": lambda t: SH}
    UB_nc = KL.union_graph(GB, ["rare"], stB.n)
    cfgB = {"union_nc": lambda t: UB_nc, "shuf": lambda t: KL.shuffled_like(KL.union_graph(GB, ["adj", "rare"], stB.n), stB.n)}
    pa = {ph: [p for p in P["A"] if p["pool"] == ph] for ph in ("tune", "held", "confirm")}
    tr = [run_qe(stA, emb, cfgA, pa["tune"], s) for s in range(1, 9)]
    tab = {b: float(np.mean([KL.contrast(tr, "QE|union_nc|b%.1f" % b, "flat", "both", t)["d"] for t in ("ADJ", "SECNEXT", "DEP")])) for b in BETAS}
    bb = max(tab, key=lambda b: tab[b])
    L = ["k4-2b QUERY EXPANSION through links (real structure): tune beta table %s -> beta %.1f" % (tab, bb)]
    out = dict(tune=tab, beta=bb, results={})
    for phase, seeds in (("held", range(31, 61)), ("confirm", range(61, 91))):
        runs = [run_qe(stA, emb, cfgA, pa[phase], s) for s in seeds]
        for t in KL.TYPES_A:
            if not any((r["flat"]["ptype"] == t).any() for r in runs):
                continue
            for g in ("union_nc", "shuf"):
                c = KL.contrast(runs, "QE|%s|b%.1f" % (g, bb), "flat", "both", t)
                out["results"]["A|%s|%s|%s" % (phase, t, g)] = c
                L.append("A %-7s %-8s QE|%-8s both@8 flat %.3f  d %+.3f [%+.3f,%+.3f] boot[%+.3f,%+.3f] p=%.3g TOST.01=%s" % (phase, t, g, KL.summarize_runs(runs, "flat", t)["both"], c["d"], c["lo"], c["hi"], c["boot"]["lo"], c["boot"]["hi"], c["p"], c["tost01"]))
    runs = [run_qe(stB, emb, cfgB, P["B"], s) for s in range(61, 91)]
    for g in ("union_nc", "shuf"):
        c = KL.contrast(runs, "QE|%s|b%.1f" % (g, bb), "flat", "both", "ADJ")
        out["results"]["B|confirm|ADJ|%s" % g] = c
        L.append("B confirm ADJ QE|%-8s both@8 flat %.3f  d %+.3f [%+.3f,%+.3f] boot[%+.3f,%+.3f] p=%.3g TOST.01=%s" % (g, KL.summarize_runs(runs, "flat", "ADJ")["both"], c["d"], c["lo"], c["hi"], c["boot"]["lo"], c["boot"]["hi"], c["p"], c["tost01"]))
    jdump(out, "k4_links_qe.json")
    open(os.path.join(RES, "k4_links_qe.txt"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
