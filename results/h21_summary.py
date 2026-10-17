#!/usr/bin/env python
"""H21 summary: held (31-60) | confirm (61-90) group means, paired deltas, Holm counts, ablations, controls.
usage: python results/h21_summary.py [suffix]   suffix '' = main run, '_mu0.1' = strict exchange rate, '--extra' = post-hoc extra controls"""
import json, sys

suf = sys.argv[1] if len(sys.argv) > 1 else ""
if suf == "--extra":
    H, C = "results/h21_extra.json", "results/h21_extra_confirm61-90.json"
else:
    H, C = f"results/h21_research{suf}.json", f"results/h21_research{suf}_confirm61-90.json"
h, c = json.load(open(H)), json.load(open(C))


def grp(w):
    return "genA" if w.startswith("res-") else "genB" if w.startswith("ana-") else "controls"


G = ["genA", "genB", "controls"]
if suf == "--extra":
    G = ["controls"]


def gm(d, arm, metric):
    out = {}
    for g in G:
        v = [d["worlds"][w]["arms"][arm][metric] for w in d["worlds"] if grp(w) == g]
        out[g] = sum(v) / len(v) if v else float("nan")
    return out


def pair(d, fam, a, b, metric):
    out = {}
    for g in G:
        v = [t["mean"] for t in d["tests"] if t["family"] == fam and t["treatment"] == a and t["metric"] == metric and (b is None or t["vs"] == b) and grp(t["world"]) == g]
        out[g] = sum(v) / len(v) if v else float("nan")
    return out


print(f"== {H} | {C}")
print("-- ABSOLUTE group means (held | confirm): q_h1000 | ok_h1000 | cost(ref calls) | util_c")
arms = ["ss-keyword", "ss-react", "ss-bok3", "oracle-stack", "L1", "L2", "L2t", "L3x", "L4", "L4w", "L5", "BoK", "Stack", "Stack-w"]
first = next(iter(h["worlds"]))
for a in arms:
    if a not in h["worlds"][first]["arms"]:
        continue
    row = []
    for g in G:
        s = [f"{gm(h, a, m)[g]:.3f}|{gm(c, a, m)[g]:.3f}" for m in ["q_h1000", "ok_h1000", "cost", "util_c"]]
        row.append(f"{g}: " + " ".join(s))
    print(f"  {a:<12} " + " || ".join(row))
print("-- PAIRED deltas (mean of per-world means; held | confirm): q | ok | cost | okpc")
cmp = [("F1-vs-bestfair", "L1", None), ("F1-vs-bestfair", "L2t", None), ("F1-vs-bestfair", "L5", None), ("F1-vs-bestfair", "Stack", None),
       ("F1s-vs-beststrong", "L2t", None), ("F1s-vs-beststrong", "L5", None), ("F1s-vs-beststrong", "Stack", None),
       ("F2-vs-prev", "L2", "L1"), ("F2-vs-prev", "L2t", "L2"), ("F2-vs-prev", "L3x", "L2t"), ("F2-vs-prev", "L4", "L3x"), ("F2-vs-prev", "L5", "L4"), ("F2-vs-prev", "Stack", "L5"),
       ("F2-vs-prev", "L4w", "L3x"), ("F2-vs-prev", "L4", "L4o"), ("F2-vs-prev", "BoK", "L2t"), ("F2-vs-prev", "Stack-w", "Stack"),
       ("F4-vs-react", "L2t", None), ("F4-vs-react", "L3x", None), ("F4-vs-react", "L5", None), ("F4-vs-react", "Stack", None)]
for fam, a, b in cmp:
    row = []
    for g in G:
        s = []
        for m in ["q_h1000", "ok_h1000", "cost", "okpc"]:
            if m == "cost" and fam in ("F1-vs-bestfair", "F1s-vs-beststrong"):
                s.append("  n/a")
                continue
            s.append(f"{pair(h, fam, a, b, m)[g]:+.3f}|{pair(c, fam, a, b, m)[g]:+.3f}")
        row.append(f"{g}: " + " ".join(s))
    print(f"  {a:<8}-{(b or fam.split('-')[-1]):<8} " + " || ".join(row))
print("-- ABLATION (lever removed minus Stack; held | confirm): ok | q | cost | okpc")
for a in ["S-noretr", "S-notools", "S-noverify", "S-nogate", "S-noreplan", "S-norouting", "S-nocross", "S-novote"]:
    row = []
    for g in G:
        s = [f"{pair(h, 'F5-ablation', a, 'Stack', m)[g]:+.3f}|{pair(c, 'F5-ablation', a, 'Stack', m)[g]:+.3f}" for m in ["ok_h1000", "q_h1000", "cost", "okpc"]]
        row.append(f"{g}: " + " ".join(s))
    print(f"  {a:<12} " + " || ".join(row))
if suf != "--extra":
    print("-- HOLM counts (W/L/ns held | confirm)")
    cn_h, cn_c = h["counts"], c["counts"]

    def cnt(k):
        return " ; ".join(f"{g} {'/'.join(map(str, cn_h.get(k + g, ['?'] * 3)))} | {'/'.join(map(str, cn_c.get(k + g, ['?'] * 3)))}" for g in G)

    for k in ["q_h1000|L2t|", "ok_h1000|L5|", "okpc|Stack|", "util_c|Stack|"]:
        print("  vs-best-plain", k, cnt(k))
    for k in ["strong|q_h1000|L2t|", "strong|ok_h1000|L5|", "strong|ok_h1000|Stack|", "strong|okpc|L5|", "strong|util_c|Stack|"]:
        print("  vs-strong", k, cnt(k))
    for k in ["prev|ok_h1000|L4-L3x|", "prev|okpc|L4-L3x|", "prev|ok_h1000|L5-L4|", "prev|ok_h1000|Stack-L5|", "prev|okpc|Stack-L5|", "prev|ok_h1000|L3x-L2t|", "prev|ok_h1000|L2t-L2|", "prev|ok_h1000|L2-L1|", "prev|util_a|L3x-L2t|", "prev|util_a|L2t-L2|"]:
        print("  vs-prev", k, cnt(k))
    for k in ["react|ok_h1000|L2t|", "react|ok_h1000|L5|", "react|ok_h1000|Stack|", "react|okpc|L5|"]:
        print("  vs-react", k, cnt(k))
print("-- SAMPLE EFFICIENCY: tasks-to-95%-of-own-asymptote vs best fair plain, Holm W/L/ns across worlds (held | confirm)")
for lvl in ["L1", "L2", "L2t", "L3x", "L4", "L5", "Stack"]:
    def f(d):
        w = l = n = 0
        for t in d["tests"]:
            if t["family"] == "F3-tte-own" and t["treatment"] == lvl:
                if t["verdict"] == "WIN":
                    w += 1
                elif t["verdict"] == "LOSS":
                    l += 1
                else:
                    n += 1
        return f"{w}/{l}/{n}"

    def mo(d):
        v = [t["mean"] for t in d["tests"] if t["family"] == "F3-tte-own" and t["treatment"] == lvl]
        return sum(v) / max(1, len(v))

    print(f"  {lvl:<6} W/L/ns {f(h)} | {f(c)}   mean delta tasks {mo(h):+.0f} | {mo(c):+.0f}")
print("-- EARLY (q H250) vs late (q H1000), mean over treatment worlds")
ws = [w for w in h["worlds"] if grp(w) != "controls"] or list(h["worlds"])
for a in ["ss-react", "L2t", "L3x", "L4", "L5", "Stack"]:
    q2 = sum(h["worlds"][w]["arms"][a]["q_h250"] for w in ws) / len(ws)
    q1 = sum(h["worlds"][w]["arms"][a]["q_h1000"] for w in ws) / len(ws)
    print(f"  {a:<8} {q2:.3f} | {q1:.3f}")
print("-- CONTROLS pre-declared equivalence checks (held | confirm)")
for wh, wc in zip(h["controls"], c["controls"]):
    for ch, cc in zip(wh["checks"], wc["checks"]):
        print(f"  {wh['world']:<18} {ch['a']}-{ch['b']} {ch['metric']:<8} held {ch['mean']:+.4f} [{ch['lo']:+.4f},{ch['hi']:+.4f}] {'EQ' if ch['equivalent'] else '--'} | conf {cc['mean']:+.4f} [{cc['lo']:+.4f},{cc['hi']:+.4f}] {'EQ' if cc['equivalent'] else '--'}")
print("-- PER-WORLD all-correct (held | confirm): best-plain, ss-react, L2t, L3x, L5, Stack ; cost L5")
for w in h["worlds"]:
    bf = h["worlds"][w]["best_fair"]
    bfc = c["worlds"][w]["best_fair"]

    def g(d, a, m="ok_h1000"):
        return d["worlds"][w]["arms"][a][m]

    print(f"  {w:<18} bf {g(h, bf):.3f}|{g(c, bfc):.3f}  react {g(h, 'ss-react'):.3f}|{g(c, 'ss-react'):.3f}  L2t {g(h, 'L2t'):.3f}|{g(c, 'L2t'):.3f}  L3x {g(h, 'L3x'):.3f}|{g(c, 'L3x'):.3f}  L5 {g(h, 'L5'):.3f}|{g(c, 'L5'):.3f}  Stack {g(h, 'Stack'):.3f}|{g(c, 'Stack'):.3f}  cost L5 {g(h, 'L5', 'cost'):.2f}|{g(c, 'L5', 'cost'):.2f} react {g(h, 'ss-react', 'cost'):.2f}")
