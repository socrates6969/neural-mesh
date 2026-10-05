"""K5 summary: frontier table (gain in dependent worlds vs loss in controls) from results/k5_{held,confA,confB}.json.
usage: python py/k5_summary.py > results/k5_summary.txt"""
import json, statistics as st
SPL = [("held", "A held 31-60 (not clean: viewed during design)"), ("confA", "A confirm 61-90"), ("confB", "B confirm 61-90")]
DEP = ["T12", "T12-drift", "T48"]
CTL = ["ctl-irrel", "ctl-oneoff", "ctl-noinfo"]
BF7 = ["keyword", "eps-greedy", "pooled-ts", "hebbian-robust", "warm(hebr)", "guarded", "global-ts"]
J = {k: json.load(open(f"results/k5_{k}.json")) for k, _ in SPL}
arms = list(J["held"]["levels"].keys())
def lv(sp, a, w, m="H2000"):
    return J[sp]["levels"][a][w][m]
def bf(sp, w, names=BF7, m="H2000"):
    return max(lv(sp, n, w, m) for n in names)
print("FRONTIER: mean H2000 difference vs the BEST FAIR baseline of each world (BF7 incl. global-ts | BF6 without), averaged over the 3 dependent worlds (T12,T12-drift,T48) and the 3 controls (ctl-irrel, ctl-oneoff, ctl-noinfo)")
print("%-18s %s" % ("arm", "  ".join("%-34s" % s for s, _ in SPL)))
print("%-18s %s" % ("", "  ".join("dep(BF7)  ctl(BF7)  worstctl  ctl(BF6)" for _ in SPL)))
for a in arms:
    row = []
    for sp, _ in SPL:
        d = st.mean(lv(sp, a, w) - bf(sp, w) for w in DEP)
        c = [lv(sp, a, w) - bf(sp, w) for w in CTL]
        c6 = st.mean(lv(sp, a, w) - bf(sp, w, BF7[:6]) for w in CTL)
        row.append("%+.4f   %+.4f   %+.4f   %+.4f" % (d, st.mean(c), min(c), c6))
    print("%-18s %s" % (a, "  ".join(row)))
print()
print("COLD START H250 / H500 / H2000 (A confirm and B confirm), selected arms, worlds T12 | ctl-irrel | ctl-noinfo")
for a in ["guarded", "global-ts", "meta3-front", "g1|guard", "g1|warm", "g1|meta3", "P2w", "P2w+", "P3", "P2g", "local|meta3", "ab(g1)|meta3"]:
    if a not in arms: continue
    out = []
    for sp in ("confA", "confB"):
        for w in ("T12", "ctl-irrel", "ctl-noinfo"):
            out.append("%.3f/%.3f/%.3f" % (lv(sp, a, w, "H250"), lv(sp, a, w, "H500"), lv(sp, a, w, "H2000")))
    print("%-14s %s" % (a, "  ".join(out)))
print()
print("LAYER DIAGNOSTICS (confA | confB): vote share, front share (guard/warm/gts), A/B holdout share, A/B on-share, injection precision")
for a in arms:
    if a in BF7 or a.endswith("-front"): continue
    out = []
    for sp in ("confA", "confB"):
        for w in ("T12", "ctl-irrel", "ctl-oneoff"):
            l = J[sp]["levels"][a][w]
            out.append("%s v%.2f g%.2f/w%.2f/t%.2f h%.2f on%.2f p%.2f" % (w[:6], l["vote_frac"], l["fshare_guard"], l["fshare_warm"], l["fshare_gts"], l["hold_frac"], (l["on_frac"] if l["on_frac"] is not None else -1), (l["inj_prec"] if l["inj_prec"] is not None else -1)))
    print("%-18s %s" % (a, " | ".join(out)))

print()
print("PRICE OF ONLINE FRONT SELECTION = arm minus ORACLE-FRONT-PER-WORLD (g1|warm in dependent worlds + ctl-oneoff, global-ts in ctl-irrel/ctl-noinfo), H2000")
W6 = ["T12", "T12-drift", "T48", "ctl-irrel", "ctl-oneoff", "ctl-noinfo"]
for sp, _ in SPL:
    L = J[sp]["levels"]
    for a in ["P2w+", "P2w", "g1|meta3"]:
        out = []
        for w in W6:
            orc = L["g1|warm"][w]["H2000"] if w in ("T12", "T12-drift", "T48", "ctl-oneoff") else L["global-ts"][w]["H2000"]
            out.append("%s %+.4f" % (w, L[a][w]["H2000"] - orc))
        print("%-6s %-9s %s" % (sp, a, " | ".join(out)))
print()
print("RETRIEVAL VALUE ON THE SAME FRONT (H2000): arm minus its front-only reference")
for sp, _ in SPL:
    L = J[sp]["levels"]
    for a, b in [("g1|meta3", "meta3-front"), ("feat|meta3", "meta3-front"), ("g1|warm", "warm(hebr)"), ("feat|warm", "warm(hebr)"), ("g1|guard", "guarded"), ("safe|guard", "guarded")]:
        print("%-6s %-11s - %-11s %s" % (sp, a, b, " ".join("%s %+.3f" % (w[:9], L[a][w]["H2000"] - L[b][w]["H2000"]) for w in W6)))
