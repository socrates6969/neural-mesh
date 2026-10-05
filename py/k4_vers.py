"""k4-3: STALENESS / CONTRADICTION signal between versions of the same source (real README paragraphs of on-disk crate versions).
Arms (top-1 answer after retrieval over ALL ingested versions): flat hybrid z+BM25 | key: structural key (crate|heading|ordinal) newest-wins | keylist: same but only
siblings already inside the retrieved top-10 | sem: unsupervised semantic sibling (same crate, cos >= tau, newer version, different text; tau tuned on the tune crates) | oracle.
Primary metric: answer-correct@1 on 'changed fact' queries (the masked changed number/entity must be in the returned paragraph).
Controls: STABLE (identical across versions; resolver must not matter) and ONLYOLD (paragraph exists only in an older version; a resolver must not hijack it).
Statistics: queries are a small finite set -> crate-cluster bootstrap (4000 reps) instead of seeds; Holm over the resolver contrasts per phase.
Usage: python py/k4_vers.py -> results/k4_vers.json, k4_vers.txt"""
import collections, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from k4_common import *

TAUS = [0.55, 0.65, 0.75, 0.85, 0.92]


def main():
    Dd = json.load(open(os.path.join(RES, "k4_vers_q.json"), encoding="utf8"))
    rows, Q = Dd["rows"], Dd["queries"]
    emb = np.fromfile(os.path.join(RES, "k4_vers_emb.f32"), dtype=np.float32).reshape(-1, 384)
    n = len(rows)
    st = Store("vers", [r["text"] for r in rows], emb[:n], [dict(src=r["crate"], key=r["key"], vord=r["vord"]) for r in rows])
    key_of = [r["key"] for r in rows]
    vord = np.array([r["vord"] for r in rows])
    crate = [r["crate"] for r in rows]
    by_key = collections.defaultdict(list)
    by_crate = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_key[r["key"]].append(i); by_crate[r["crate"]].append(i)
    qE = np.stack([emb[q["rq"]] for q in Q])
    qB = st.qbow([q["text"] for q in Q])
    F = st.fused(qE, qB)
    top = top_idx(F, 10)
    kinds = np.array([q["kind"] for q in Q]); pools = np.array([q["pool"] for q in Q]); crates = np.array([q["crate"] for q in Q])

    def latest_sibling(d):
        """newest version of the same structural key whose text differs from d (None if d is already newest/identical)"""
        cand = [j for j in by_key[key_of[d]] if vord[j] > vord[d] and rows[j]["text"] != rows[d]["text"]]
        return max(cand, key=lambda j: vord[j]) if cand else None

    def sem_sibling(d, tau):
        best = None; bc = -1
        for j in by_crate[crate[d]]:
            if vord[j] > vord[d] and rows[j]["text"] != rows[d]["text"]:
                c = float(st.D[d] @ st.D[j])
                if c >= tau and (best is None or (vord[j], c) > (vord[best], bc)):
                    best, bc = j, c
        return best

    def arm_top1(i, arm, tau=None):
        row = list(top[i])
        d = row[0]
        if arm == "flat":
            return d
        if arm == "key":
            s = latest_sibling(d)
            return s if s is not None else d
        if arm == "keylist":
            s = latest_sibling(d)
            return s if (s is not None and s in row) else d
        if arm == "sem":
            s = sem_sibling(d, tau)
            return s if s is not None else d
        if arm == "oracle":
            return Q[i]["gold"]

    def correct(i, d):
        q = Q[i]
        if q["kind"] in ("NUM", "ENT"):
            return float(all(t in rows[d]["text"] for t in q["answer"]))
        if q["kind"] == "STABLE":
            return float(key_of[d] == q["key"])
        return float(rows[d]["key"] == q["key"] and rows[d]["text"] == rows[q["gold"]]["text"])  # ONLYOLD: the (only) paragraph itself

    def run(arm, idx, tau=None):
        return np.array([correct(i, arm_top1(i, arm, tau)) for i in idx])

    # ------------------------------------------------ tune tau on tune crates: objective = mean(changed correct) + mean(ONLYOLD correct) + mean(STABLE correct) (all three kinds, equal weight)
    ti = np.where(pools == "tune")[0]
    obj = {}
    for tau in TAUS:
        v = []
        for kset in (("NUM", "ENT"), ("STABLE",), ("ONLYOLD",)):
            ix = ti[np.isin(kinds[ti], kset)]
            v.append(run("sem", ix, tau).mean())
        obj[tau] = float(np.mean(v))
    tau_star = max(obj, key=lambda t: (obj[t], t))
    L = ["k4-3 STALENESS (real README versions; masked-changed-fact queries)", "tune tau objective %s -> tau=%.2f" % (obj, tau_star)]
    out = dict(tau_obj=obj, tau=tau_star, n_docs=n, n_queries={k: int((kinds == k).sum()) for k in ("NUM", "ENT", "STABLE", "ONLYOLD")})
    # ------------------------------------------------ evaluation
    arms = ["flat", "key", "keylist", "sem", "oracle"]
    res = {}
    for phase in ("tune", "held", "confirm"):
        for grp, kset in (("changed", ("NUM", "ENT")), ("NUM", ("NUM",)), ("ENT", ("ENT",)), ("STABLE", ("STABLE",)), ("ONLYOLD", ("ONLYOLD",))):
            ix = np.where((pools == phase) & np.isin(kinds, kset))[0]
            if len(ix) == 0:
                continue
            vals = {a: run(a, ix, tau_star) for a in arms}
            blk = dict(n=len(ix), mean={a: float(v.mean()) for a, v in vals.items()}, contrast={})
            for a in ("key", "keylist", "sem"):
                d = vals[a] - vals["flat"]
                m, lo, hi, p = cluster_boot(d, crates[ix], 4000)
                blk["contrast"][a] = dict(d=m, lo=lo, hi=hi, p=p, helped=int((d > 0).sum()), hurt=int((d < 0).sum()))
            res["%s|%s" % (phase, grp)] = blk
    # holm over the 3 resolver contrasts on held and confirm changed
    for phase in ("held", "confirm"):
        b = res.get("%s|changed" % phase)
        if b:
            ps = [b["contrast"][a]["p"] for a in ("key", "keylist", "sem")]
            for a, pa in zip(("key", "keylist", "sem"), holm(ps)):
                b["contrast"][a]["p_holm"] = float(pa)
    out["results"] = res
    for k, b in res.items():
        L.append("%s n=%d  correct@1: %s" % (k, b["n"], {a: round(v, 3) for a, v in b["mean"].items()}))
        for a, c in b["contrast"].items():
            L.append("    %-8s - flat: %+.3f [%+.3f,%+.3f] boot p=%.3g%s  helped %d hurt %d" % (a, c["d"], c["lo"], c["hi"], c["p"], (" holm=%.3g" % c["p_holm"]) if "p_holm" in c else "", c["helped"], c["hurt"]))
    # ------------------------------------------------ staleness signal quality (no answer labels used): conflict flag in the retrieved top-10
    def flag_struct(i):
        ks = collections.defaultdict(set)
        for d in top[i]:
            ks[key_of[d]].add(rows[d]["text"])
        return any(len(v) > 1 for v in ks.values())

    def flag_top1_has_newer(i):
        return latest_sibling(top[i][0]) is not None
    fl = {}
    for kd in ("NUM", "ENT", "STABLE", "ONLYOLD"):
        ix = np.where(kinds == kd)[0]
        f1 = np.array([flag_struct(i) for i in ix]); f2 = np.array([flag_top1_has_newer(i) for i in ix])
        s1 = np.array([correct(i, top[i][0]) == 0 for i in ix])
        fl[kd] = dict(n=len(ix), flag_conflict_in_top10=float(f1.mean()), flag_top1_has_newer_version=float(f2.mean()), flat_top1_wrong=float(s1.mean()), precision_flag2_wrong=float(s1[f2].mean()) if f2.any() else None, recall_wrong_caught=float(f2[s1].mean()) if s1.any() else None)
    out["signal"] = fl
    L.append("SIGNAL (all pools): " + json.dumps(fl))
    # ------------------------------------------------ structural key reliability: how often is the same paragraph under the same key across versions? (key drift)
    pairs = tot = 0
    cs = []
    for crate_, ids in by_crate.items():
        lv = max(vord[j] for j in ids)
        newer = [j for j in ids if vord[j] == lv]
        older = [j for j in ids if vord[j] < lv]
        if not older:
            continue
        Do = st.D[older]
        for j in newer:
            s = Do @ st.D[j]
            b = int(np.argmax(s))
            if s[b] >= 0.90:
                tot += 1
                cs.append(float(s[b]))
                if key_of[older[b]] == key_of[j]:
                    pairs += 1
    out["key_reliability"] = dict(newest_paragraphs_with_near_copy_in_older=tot, same_key_share=(pairs / tot if tot else None))
    L.append("KEY RELIABILITY: newest-version paragraphs with a semantically near-identical (cos>=.90) older paragraph: %d, of which under the SAME structural key: %.3f" % (tot, pairs / max(tot, 1)))
    jdump(out, "k4_vers.json")
    open(os.path.join(RES, "k4_vers.txt"), "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
