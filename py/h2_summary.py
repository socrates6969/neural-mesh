"""h2: print tables from results/h2_route*.json and apply the declared KEEP rule across corpora.
Usage: python py/h2_summary.py            (reads results/h2_route.json (A held-out 31-60) + h2_route_confirm_{B,C2,D2}.json (frozen, 61-90))
KEEP rule (declared before running): vs the SAME front without retrieval (guarded(hebr)), H2000, Holm over the family:
  >= 2 treatment worlds {sib16, rand16, sib64, sib16-drift, sib16-weakid50} are Holm WINS  AND  no control world {ctl-irrel, ctl-oneoff, ctl-noinfo} is a Holm LOSS
  on corpus A held-out AND on >= 2 of the 3 fresh frozen confirms (B, C2, D2).  mixed50 is reported as a strict flag.
"""
import json, os, sys

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
TREAT = ["sib16", "rand16", "sib64", "sib16-drift", "sib16-weakid50"]
CTRL = ["ctl-irrel", "ctl-oneoff", "ctl-noinfo"]
WORLDS = ["sib16", "rand16", "sib64", "sib16-drift", "sib16-weakid50", "sib16-mixed50", "ctl-irrel", "ctl-oneoff", "ctl-noinfo"]
SHORT = {"sib16": "sib16", "rand16": "rand16", "sib64": "sib64", "sib16-drift": "drift", "sib16-weakid50": "weakid", "sib16-mixed50": "mixed50", "ctl-irrel": "c-irrel", "ctl-oneoff": "c-1off", "ctl-noinfo": "c-noinf"}


def load(p):
    p = os.path.join(RES, p)
    return json.load(open(p)) if os.path.exists(p) else None


def comp(d, fam, arm, world, metric="H2000", vs=None):
    for c in d["comparisons"]:
        if c["fam"].startswith(fam) and c["arm"] == arm and c["world"] == world and c["metric"] == metric and (vs is None or c["vs"] == vs):
            return c
    return None


def mk(c):
    return "*" if c["verdict"] == "WIN" else ("!" if c["verdict"] == "LOSS" else " ")


def table(d, name, fam, title, arms=None):
    print("\n##", name, title)
    print("%-20s" % "arm" + "".join("%9s" % SHORT[w] for w in WORLDS))
    for a in d["arms"]:
        an = a["name"]
        if arms and an not in arms:
            continue
        cs = [comp(d, fam, an, w) for w in WORLDS]
        if any(c is None for c in cs):
            continue
        print("%-20s" % an[:20] + "".join("%+8.3f%s" % (c["diff"], mk(c)) for c in cs))


def keep(d, an):
    tw = [w for w in TREAT if (comp(d, "F1", an, w) or {}).get("verdict") == "WIN"]
    cl = [w for w in CTRL if (comp(d, "F1", an, w) or {}).get("verdict") == "LOSS"]
    mx = (comp(d, "F1", an, "sib16-mixed50") or {}).get("verdict")
    return len(tw) >= 2 and not cl, tw, cl, mx


def main():
    runs = [("A(held 31-60)", load("h2_route.json"))] + [("%s(frozen 61-90)" % t, load("h2_route_confirm_%s.json" % t)) for t in ("B", "C2", "D2")]
    runs = [(n, d) for n, d in runs if d]
    for n, d in runs:
        print("\n" + "=" * 100 + "\n" + n)
        print("picks:", [(a["name"], a["label"], "k%d thr%.2f thr2%.2f mn%d" % (a["k"], a["thr"], a["thr2"], a["min_n"])) for a in d["arms"] if not a["name"].endswith("oracle")][:0])
        table(d, n, "F1", "H2000 diff vs SAME front guarded(hebr)  (* Holm win, ! Holm loss)")
        table(d, n, "F3", "H2000 diff vs spell-only (dense+spell, same gate)")
        table(d, n, "F4", "H2000 diff vs g16 reference hybrid(.6) thr .75 (same gate)")
        table(d, n, "F5", "H2000 diff vs raw-dense (NO spell correction, same gate): the spell-correct lever")
        print("\nkeep rule per arm:")
        for a in d["arms"]:
            k, tw, cl, mx = keep(d, a["name"])
            print("  %-18s %-16s %s tw=%s cl=%s mixed50=%s" % (a["name"], a["label"], "KEEP" if k else "----", [SHORT[x] for x in tw], [SHORT[x] for x in cl], mx))
    # cross-corpus verdict
    if runs:
        print("\n" + "=" * 100 + "\nCROSS-CORPUS KEEP VERDICT (A held-out AND >=2 of 3 fresh confirms)")
        names = [a["name"] for a in runs[0][1]["arms"]]
        for an in names:
            ks = [(n, keep(d, an)[0]) for n, d in runs if any(a["name"] == an for a in d["arms"])]
            ka = ks[0][1]
            kc = sum(1 for n, k in ks[1:] if k)
            fin = ka and kc >= 2
            print("  %-18s A:%s confirms keep %d/%d -> %s" % (an, "Y" if ka else "n", kc, len(ks) - 1, "KEEP" if fin else "reject"))
    # costs
    d = runs[0][1]
    print("\ncost (A, single thread sib16): ")
    for c in d["cost"]:
        print("  %-20s %8.1f us/task calls/task %6.2f charged %8.1f us/task" % (c["arm"], c["us_per_task"], c["calls_per_task"], c["charged_us_per_task"]))


if __name__ == "__main__":
    main()


def extra():
    """second (stricter) rule + non-inferiority CI containment + retrieval-vs-end-to-end juxtaposition"""
    runs = [("A", load("h2_route.json"))] + [(t, load("h2_route_confirm_%s.json" % t)) for t in ("B", "C2", "D2")]
    runs = [(n, d) for n, d in runs if d]
    print("\n" + "=" * 100 + "\nSTRICT RULE: does the variant ADD over the incumbent (spell-only dense, same gate)?  needs Holm WIN vs spell-only in >=2 treatment worlds AND no Holm LOSS in any of the 9 worlds, on A and >=2 of 3 confirms")
    names = [a["name"] for a in runs[0][1]["arms"] if not a["name"].endswith("oracle") and not a["name"].endswith("|dense") and "raw-dense" not in a["name"]]
    for an in names:
        flags = []
        for n, d in runs:
            ws = [(w, comp(d, "F3", an, w)) for w in WORLDS]
            wins = [w for w, c in ws if c and c["verdict"] == "WIN"]
            loss = [w for w, c in ws if c and c["verdict"] == "LOSS"]
            tw = [w for w in wins if w in TREAT]
            flags.append((n, len(tw) >= 2 and not loss, [SHORT[w] for w in wins], [SHORT[w] for w in loss]))
        ok = flags[0][1] and sum(1 for f in flags[1:] if f[1]) >= 2
        print("  %-18s %s | %s" % (an, "ADDS" if ok else "no-add", "; ".join("%s: wins%s loss%s" % (f[0], f[2], f[3]) for f in flags)))
    print("\nNON-INFERIORITY (post-hoc, margin +-0.01 as in g16): worlds where the 95% CI of (variant - spell-only, H2000) lies inside [-0.01,+0.01] / 9, per corpus")
    for an in names:
        row = []
        for n, d in runs:
            cs = [comp(d, "F3", an, w) for w in WORLDS]
            row.append("%s:%d" % (n, sum(1 for c in cs if c and c["lo"] > -0.01 and c["hi"] < 0.01)))
        print("  %-18s %s" % (an, " ".join(row)))
    r = load("h2_retrieval.json")
    if r:
        print("\nRETRIEVAL-LEVEL vs END-TO-END (A held-out): doc-level delta vs dense+spell (F1@3, MRR, hit1) | end-to-end H2000 delta vs spell-only, sib16 / rand16 (g1 gate | safe gate)")
        d = runs[0][1]
        m = {"RRF(k=60)": "rrf", "RSF": "rsf", "hybrid-z": "hybrid", "rerank20-overlap": "rerank-jac", "rerank20-bm25": "rerank-bm25", "rerank20-logit": "rerank-logit"}
        base = [s for s in r["systems"] if s["system"].startswith("dense")][0]
        for s in r["systems"]:
            for k, v in m.items():
                if s["system"].startswith(k):
                    e = []
                    for reg in ("g1", "safe"):
                        cs = [comp(d, "F3", "%s|%s" % (reg, v), w) for w in ("sib16", "rand16")]
                        e.append("/".join("%+.3f%s" % (c["diff"], mk(c)) for c in cs))
                    print("  %-34s doc dF1@3 %+.3f dMRR %+.3f dhit1 %+.3f | e2e g1 %s | safe %s" % (s["system"], s["f13"] - base["f13"], s["mrr"] - base["mrr"], s["hit1"] - base["hit1"], e[0], e[1]))


if __name__ == "__main__":
    extra()
