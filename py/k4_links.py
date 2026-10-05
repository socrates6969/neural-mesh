"""k4-2: KNOWLEDGE LINKS.  Link graph from REAL structure (same-file neighbours / same section / shared rare terms / explicit links / Cargo.toml deps)
and LINK-EXPANSION retrieval (retrieve hits, spend r of the k=8 context slots on their linked neighbours) vs the flat hybrid z+BM25 at the SAME budget k=8,
on multi-hop questions whose ground-truth pair (P1, T2) comes from the same document structure (py/k4_mh_prep.py).
Protocol: tune = world A tune pool seeds 1-8 (choose r only); held = A held pool seeds 31-60; confirm = A confirm pool seeds 61-90 and world B (independent text) 61-90.
CIRCULARITY (declared): graph type == GT type encodes the label (ceiling, reported separately); the claim is made with the NON-circular graph `union_nc`
(union of every graph type EXCEPT the one that defined the pair type) and with controls (degree-matched random links, single-hop questions).
Usage: python py/k4_links.py   -> results/k4_links.json, results/k4_links.txt"""
import collections, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4_common import *

KB = 8
RS = [0, 1, 2, 3, 4]
TYPES_A = ["ADJ", "SECNEXT", "DEP", "XREF"]
NPAIR = 100


def cname(g):
    import re
    return re.sub(r"-\d+\.\d+.*$", "", g)


def build_store(tag, pairs, emb):
    w = H10World(tag)
    texts = [d["text"] for d in w.docs]
    D = w.store.D.copy()
    for p in pairs:
        if p["rdoc"] >= 0:
            texts[p["p1"]] = p["newtxt"]
            D[p["p1"]] = emb[p["rdoc"]]
    meta = [dict(src=d["src"], file=d["file"], id=d["id"]) for d in w.docs]
    st = Store("k4" + tag, texts, D, meta)
    return w, st


def graphs_for(tag, w, st):
    """-> dict type -> list (len n) of dict nbr->weight"""
    n = st.n
    docs = w.docs
    G = {t: [dict() for _ in range(n)] for t in ("adj", "sec", "rare", "dep", "xref", "xlink")}
    for i in range(n - 1):
        if docs[i]["src"] == docs[i + 1]["src"] and docs[i]["file"] == docs[i + 1]["file"] and docs[i]["file"] != "(doc comment)":
            G["adj"][i][i + 1] = 1.0; G["adj"][i + 1][i] = 1.0
    # rare terms (df<=3, len>=5)
    inv = collections.defaultdict(list)
    for t, j in st.vocab.items():
        if st.df[j] <= 3 and len(t) >= 5:
            pass
    toks = st.toks
    term_docs = collections.defaultdict(list)
    for i, tk in enumerate(toks):
        for t in set(tk):
            j = st.vocab[t]
            if 2 <= st.df[j] <= 3 and len(t) >= 5:
                term_docs[t].append(i)
    for t, ds in term_docs.items():
        wgt = math.log(n / len(ds))
        for a in ds:
            for b in ds:
                if a != b and abs(a - b) > 3:
                    G["rare"][a][b] = G["rare"][a].get(b, 0) + wgt
    for i in range(n):  # keep the 5 strongest, normalise to <=1
        if len(G["rare"][i]) > 5:
            G["rare"][i] = dict(sorted(G["rare"][i].items(), key=lambda x: -x[1])[:5])
        if G["rare"][i]:
            mx = max(G["rare"][i].values())
            G["rare"][i] = {k: v / mx for k, v in G["rare"][i].items()}
    if tag == "A":
        S = json.load(open(os.path.join(RES, "k4_struct_A.json")))
        secs, sd, deps = S["secs"], S["docs"], S["deps"]
        for s in secs:
            ds = s["docs"]
            for a_i, a in enumerate(ds):
                for b_i in range(max(0, a_i - 3), min(len(ds), a_i + 4)):
                    if b_i != a_i and abs(b_i - a_i) > 1:
                        G["sec"][a][ds[b_i]] = 0.7
        byname = collections.defaultdict(list)
        for i, d in enumerate(docs):
            byname[cname(d["src"])].append(i)
        for i, d in enumerate(docs):
            me = cname(d["src"])
            for y in deps.get(d["src"], []):
                if y != me and y in byname:
                    for j in byname[y][:3]:
                        G["dep"][i][j] = 1.0
        import re
        names = set(byname)
        for i, d in enumerate(sd):
            me = cname(docs[i]["src"])
            for l in d["raw"]:
                for seg in re.split(r"[/#?]", l):
                    y = seg.lower().replace("_", "-")
                    if y in names and y != me and not me.startswith(y) and not y.startswith(me):
                        for j in byname[y][:3]:
                            G["xref"][i][j] = 1.0
            for kind, t in d["tgt"]:
                if kind == "sec":
                    for j in secs[t]["docs"][:3]:
                        G["xlink"][i][j] = 1.0
    return G


def union_graph(G, kinds, n):
    U = [dict() for _ in range(n)]
    for k in kinds:
        for i in range(n):
            for j, wv in G[k][i].items():
                if wv > U[i].get(j, 0):
                    U[i][j] = wv
    return U


def shuffled_like(U, n, seed=3):
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        m = len(U[i])
        out.append({int(j): 1.0 for j in rng.randint(0, n, m)} if m else {})
    return out


def le_list(top_flat_row, graph, r, kb=KB):
    """flat top-(kb-r) seeds + the r best linked neighbours (score = edge weight / (1 + seed rank)), filled from flat if neighbours run out"""
    seeds = list(top_flat_row[:kb - r])
    sset = set(seeds)
    cand = {}
    for rk, s in enumerate(seeds):
        for j, wv in graph[s].items():
            if j in sset:
                continue
            sc = wv / (1.0 + rk)
            if sc > cand.get(j, 0):
                cand[j] = sc
    add = [j for j, _ in sorted(cand.items(), key=lambda x: -x[1])[:r]]
    out = seeds + add
    k = kb - r
    while len(out) < kb:
        j = top_flat_row[k]
        k += 1
        if j not in out:
            out.append(j)
    return out


def hop_metrics(lst, p1, t2):
    h1 = float(p1 in lst)
    h2 = float(any(d in t2 for d in lst))
    return h1, h2, h1 * h2


class Arm:
    pass


def run_pairs(st, emb, cfgs, phase_pairs, seed, rng_n=NPAIR):
    """one seed: sample pairs per type, evaluate arms -> dict arm -> dict metric -> per-pair arrays (+ pair idx, type)"""
    rng = np.random.RandomState(seed * 7 + 3)
    sel = []
    by = collections.defaultdict(list)
    for k, p in enumerate(phase_pairs):
        by[p["type"]].append(k)
    for t, ks in by.items():
        ks = np.array(ks)
        sel.extend(ks if len(ks) <= rng_n else rng.choice(ks, rng_n, replace=False))
    sel = np.array(sorted(sel))
    P = [phase_pairs[k] for k in sel]
    qE = np.stack([emb[p["rq"]] for p in P])
    qB = st.qbow([p["q"] for p in P])
    F = st.fused(qE, qB)
    top = top_idx(F, 24)
    res = {}
    ptype = np.array([p["type"] for p in P])
    pid = np.array([p["p1"] * 10 + ["ADJ", "SECNEXT", "DEP", "XREF", "XLINK"].index(p["type"]) for p in P])

    def score(lists):
        a = np.array([hop_metrics(l, p["p1"], set(p["t2"])) for l, p in zip(lists, P)])
        return {"hop1": a[:, 0], "hop2": a[:, 1], "both": a[:, 2], "ptype": ptype, "pid": pid}

    res["flat"] = score([list(t[:KB]) for t in top])
    # decomposition baseline (extra retrieval per clause; 2-clause types only)
    lists = []
    for k, p in enumerate(P):
        if p["rc2"] < 0:
            lists.append(list(top[k][:KB])); continue
        e1 = emb[p["rs1"]][None]; e2 = emb[p["rc2"]][None]
        f1 = st.fused(e1, st.qbow([p["s1"]]))[0]; f2 = st.fused(e2, st.qbow([p["c2"]]))[0]
        a = list(top_idx(f1[None], 8)[0][:4]); b = [d for d in top_idx(f2[None], 8)[0] if d not in a][:4]
        out = a + b
        for d in top[k]:
            if len(out) >= KB: break
            if d not in out: out.append(d)
        lists.append(out[:KB])
    res["split"] = score(lists)
    for gname, gfun in cfgs.items():
        for r in RS[1:]:
            lists = []
            for k, p in enumerate(P):
                g = gfun(p["type"])
                lists.append(le_list(top[k], g, r))
            res["LE|%s|r%d" % (gname, r)] = score(lists)
    return res


def summarize_runs(runs, arm, mask_type=None):
    out = {}
    for key in ("hop1", "hop2", "both"):
        v = []
        for r in runs:
            a = r[arm][key]
            if mask_type is not None:
                a = a[r[arm]["ptype"] == mask_type]
            v.append(a.mean() if len(a) else np.nan)
        out[key] = float(np.nanmean(v))
    return out


def contrast(runs, arm, base, key, typ):
    d = []; pq = []; ids = []
    for r in runs:
        m = r[arm]["ptype"] == typ
        if m.sum() == 0:
            continue
        diff = r[arm][key][m] - r[base][key][m]
        d.append(diff.mean()); pq.append(diff); ids.append(r[arm]["pid"][m])
    if not d:
        return None
    d = np.array(d)
    m_, lo, hi, p = t_ci(d)
    bm = cluster_boot(np.concatenate(pq), np.concatenate(ids), 1000)
    return dict(d=m_, lo=lo, hi=hi, p=p, boot=dict(zip(("d", "lo", "hi", "p"), bm)), tost01=bool(tost(d, 0.01)), n_seeds=len(d))


def main():
    t0 = time.time()
    P = json.load(open(os.path.join(RES, "k4_mh_pairs.json")))
    emb = np.fromfile(os.path.join(RES, "k4_embed.f32"), dtype=np.float32).reshape(-1, 384)
    out = dict(note="real text + real structure; no outcome model; labels from document structure")
    for p in P["A"]:
        if p["type"] in ("XREF", "XLINK"):
            p["pool"] = "held"  # too few pairs to split: reported once (held), r was tuned on other types
    lines = []
    wA, stA = build_store("A", P["A"], emb)
    wB, stB = build_store("B", P["B"], emb)
    GA = graphs_for("A", wA, stA)
    GB = graphs_for("B", wB, stB)
    nA, nB = stA.n, stB.n
    degs = {k: float(np.mean([len(x) for x in GA[k]])) for k in GA}
    degsB = {k: float(np.mean([len(x) for x in GB[k]])) for k in ("adj", "rare")}
    print("mean degree A", degs, "B", degsB, flush=True)
    out["degree_A"] = degs; out["degree_B"] = degsB
    kinds = ["adj", "sec", "rare", "dep", "xref", "xlink"]
    gtkind = {"ADJ": ["adj", "sec"], "SECNEXT": ["adj", "sec"], "DEP": ["dep"], "XREF": ["xref"], "XLINK": ["xlink"]}
    UA = union_graph(GA, kinds, nA)
    UA_nc = {t: union_graph(GA, [k for k in kinds if k not in gtkind[t]], nA) for t in gtkind}
    SH = shuffled_like(UA, nA)
    # oracle edges: P1 -> first 3 docs of T2 (circular ceiling)
    OR = [dict() for _ in range(nA)]
    for p in P["A"]:
        for j in p["t2"][:3]:
            OR[p["p1"]][j] = 1.0
    UB = union_graph(GB, ["adj", "rare"], nB)
    UB_nc = {"ADJ": union_graph(GB, ["rare"], nB)}
    cfgsA = {
        "adj": lambda t: GA["adj"], "sec": lambda t: GA["sec"], "rare": lambda t: GA["rare"], "dep": lambda t: GA["dep"], "xref": lambda t: GA["xref"],
        "union": lambda t: UA, "union_nc": lambda t: UA_nc[t], "shuf": lambda t: SH, "oracle": lambda t: OR,
    }
    cfgsB = {"adj": lambda t: GB["adj"], "rare": lambda t: GB["rare"], "union": lambda t: UB, "union_nc": lambda t: UB_nc[t], "shuf": lambda t: shuffled_like(UB, nB)}
    # ------------------------------------------------ tune r on A tune pool seeds 1-8 (objective: mean both@8 gain of union_nc over flat on ADJ, SECNEXT, DEP)
    ptune = [p for p in P["A"] if p["pool"] == "tune"]
    ptune = [p for p in P["A"] if p["pool"] == "tune"]
    pa = {"tune": ptune, "held": [p for p in P["A"] if p["pool"] == "held"], "confirm": [p for p in P["A"] if p["pool"] == "confirm"]}
    # run_pairs needs index into emb by p fields only; phase_pairs are the pair dicts
    tune_runs = [run_pairs(stA, emb, cfgsA, pa["tune"], s) for s in range(1, 9)]
    tab = {}
    for r in RS[1:]:
        gains = [np.mean([contrast(tune_runs, "LE|union_nc|r%d" % r, "flat", "both", t)["d"] for t in ("ADJ", "SECNEXT", "DEP")])]
        tab[r] = float(gains[0])
    rbest = max(tab, key=lambda r: tab[r])
    print("tune: mean both@8 gain by r", tab, "-> r =", rbest, flush=True)
    out["tune_r_gain"] = tab; out["r_frozen"] = rbest
    lines.append("TUNE (A tune pool seeds 1-8, objective mean both@8 gain of LE|union_nc over flat8 on ADJ/SECNEXT/DEP): %s -> r=%d" % (tab, rbest))
    # ------------------------------------------------ held / confirm
    def block(label, runs, typs, arms, ref="flat"):
        lines.append("== %s  (k=%d total slots; r=%d expansion slots)" % (label, KB, rbest))
        blk = {}
        for t in typs:
            if not any((r[arms[0]]["ptype"] == t).any() for r in runs):
                continue
            lines.append("  type %s" % t)
            for arm in arms:
                s = summarize_runs(runs, arm, t)
                c = contrast(runs, arm, ref, "both", t) if arm != ref else None
                c1 = contrast(runs, arm, ref, "hop1", t) if arm != ref else None
                c2 = contrast(runs, arm, ref, "hop2", t) if arm != ref else None
                blk[(t, arm)] = dict(summ=s, both=c, hop1=c1, hop2=c2)
                if c:
                    lines.append("    %-22s hop1 %.3f hop2 %.3f both %.3f | d both %+.3f [%+.3f,%+.3f] boot[%+.3f,%+.3f] p=%.3g | d hop1 %+.3f d hop2 %+.3f TOST.01(both)=%s" % (arm, s["hop1"], s["hop2"], s["both"], c["d"], c["lo"], c["hi"], c["boot"]["lo"], c["boot"]["hi"], c["p"], c1["d"], c2["d"], c["tost01"]))
                else:
                    lines.append("    %-22s hop1 %.3f hop2 %.3f both %.3f" % (arm, s["hop1"], s["hop2"], s["both"]))
        return blk
    armsA = ["flat", "split"] + ["LE|%s|r%d" % (g, rbest) for g in ("adj", "sec", "rare", "dep", "xref", "union", "union_nc", "shuf", "oracle")]
    resd = {}
    for phase, seeds in (("held", range(31, 61)), ("confirm", range(61, 91))):
        runs = [run_pairs(stA, emb, cfgsA, pa[phase], s) for s in seeds]
        blk = block("A %s seeds %d-%d" % (phase, seeds[0], seeds[-1]), runs, TYPES_A, armsA)
        resd["A|" + phase] = {"%s|%s" % k: v for k, v in blk.items()}
        # r sweep (descriptive)
        sweep = {}
        for r in RS[1:]:
            sweep[r] = {t: (lambda c: c["d"] if c else None)(contrast(runs, "LE|union_nc|r%d" % r, "flat", "both", t)) for t in TYPES_A}
        resd["A|%s|r_sweep_union_nc" % phase] = sweep
        lines.append("  r sweep (union_nc, both@8 d vs flat) %s" % sweep)
    pbB = [p for p in P["B"]]
    runsB = [run_pairs(stB, emb, cfgsB, pbB, s) for s in range(61, 91)]
    armsB = ["flat"] + ["LE|%s|r%d" % (g, rbest) for g in ("adj", "rare", "union", "union_nc", "shuf")]
    blk = block("B confirm seeds 61-90 (independent text world)", runsB, ["ADJ"], armsB)
    resd["B|confirm"] = {"%s|%s" % k: v for k, v in blk.items()}
    out["results"] = resd
    # ------------------------------------------------ control: single-hop S questions (A held 31-60): does LE hurt plain retrieval of the single target?
    ctl = []
    for s in range(31, 61):
        q = wA.qset(s)
        F = stA.fused(q.E, q.B)
        top = top_idx(F, 24)
        rel = q.rel
        isS = q.fam == "S"
        row = {"flat": np.array([float(any(d in set(rl.tolist()) for d in t[:KB])) for t, rl in zip(top, rel)])[isS]}
        for g, gr in (("union", UA), ("shuf", SH)):
            row["LE|%s" % g] = np.array([float(any(d in set(rl.tolist()) for d in le_list(t, gr, rbest))) for t, rl in zip(top, rel)])[isS]
        ctl.append(row)
    lines.append("== CONTROL single-hop S questions (A held 31-60, hit@8 of the one relevant paragraph), r=%d" % rbest)
    cres = {}
    for a in ("LE|union", "LE|shuf"):
        d = np.array([r[a].mean() - r["flat"].mean() for r in ctl])
        m, lo, hi, p = t_ci(d)
        cres[a] = dict(d=m, lo=lo, hi=hi, p=p, tost02=bool(tost(d, 0.02)), tost01=bool(tost(d, 0.01)))
        lines.append("  %s - flat: %+.4f [%+.4f,%+.4f] p=%.3g TOST+-.01=%s TOST+-.02=%s  (flat hit@8 %.3f)" % (a, m, lo, hi, p, cres[a]["tost01"], cres[a]["tost02"], np.mean([r["flat"].mean() for r in ctl])))
    out["control_singlehop"] = cres
    jdump(out, "k4_links.json")
    open(os.path.join(RES, "k4_links.txt"), "w").write("\n".join(lines))
    print("\n".join(lines))
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
