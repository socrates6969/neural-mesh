"""Combine h4 held-out + confirm JSON into the claim table (analysis only; reads results/h4_held.json, results/h4_confirm.json).
Rerun: python results/h4_combine.py > results/h4_combined.md
A cell is CLAIMED (EQ) only if Holm-adjusted TOST p<0.05 at margin 0.01 in BOTH stages (strict: margin 0.005); NI likewise."""
import json, sys

def load(p):
    return {(r["world"], r["cell"]): r for r in json.load(open(p))}

H = load("results/h4_held.json")
C = load("results/h4_confirm.json")
MAIN = ["A-zipf", "B-burst"]
CTRL = ["A-allvalued", "A-noreuse"]
FAM = {"fifo", "lru", "value", "merge", "tiered-int8", "tiered-f16", "tiered-int8+merge"}

def ok(r, k):
    v = r.get(k)
    return v is not None and v == v and v < 0.05

print("# h4 combined claim table (held-out seeds 31-60 AND confirm seeds 61-90; synthetic-until-validated)\n")
print("EQ01/EQ005 = Holm-adjusted TOST (margin 0.01 / 0.005) significant in BOTH stages; NI = non-inferiority (delta > -0.01) in BOTH stages. Controls use raw p (not in Holm family).\n")
for w in MAIN + CTRL:
    print(f"## {w}\n")
    print("| cell | dSucc held | dSucc confirm | bytes frac (D=64) | proj D=384 | items | lat | dvs full-rd (held/conf) | EQ.01 both | EQ.005 both | NI both |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (ww, cell), rh in H.items():
        if ww != w or rh["frac"] is None or rh["policy"] not in FAM:
            continue
        rc = C[(ww, cell)]
        fam = w in MAIN
        k01, k005, kni = ("adj_01", "adj_005", "adj_ni_01") if fam else ("p_tost_01", "p_tost_005", "p_ni_01")
        eq01 = ok(rh, k01) and ok(rc, k01)
        eq005 = ok(rh, k005) and ok(rc, k005)
        ni = ok(rh, kni) and ok(rc, kni)
        print(f"| {cell} | {rh['d']:+.4f} [{rh['lo']:+.4f},{rh['hi']:+.4f}] | {rc['d']:+.4f} [{rc['lo']:+.4f},{rc['hi']:+.4f}] | "
              f"{(rh['bytes_frac']+rc['bytes_frac'])/2:.3f} | {(rh['proj384_frac']+rc['proj384_frac'])/2:.3f} | "
              f"{(rh['items_frac']+rc['items_frac'])/2:.3f} | {(rh['lat_ratio']+rc['lat_ratio'])/2:.2f} | "
              f"{rh.get('d_vs_fullrd', float('nan')):+.4f}/{rc.get('d_vs_fullrd', float('nan')):+.4f} | "
              f"{'EQ' if eq01 else '-'} | {'EQ' if eq005 else '-'} | {'NI' if ni else '-'} |")
    print()

print("## Knee per (world, policy): smallest byte fraction with EQ(0.01) in both stages\n")
print("| world | policy | knee bytes frac | knee cell | bytes saved | proj D=384 saved |")
print("|---|---|---|---|---|---|")
for w in MAIN:
    for pol in sorted(FAM):
        best = None
        for (ww, cell), rh in H.items():
            if ww != w or rh["policy"] != pol or rh["frac"] is None:
                continue
            rc = C[(ww, cell)]
            if ok(rh, "adj_01") and ok(rc, "adj_01"):
                b = (rh["bytes_frac"] + rc["bytes_frac"]) / 2
                pj = (rh["proj384_frac"] + rc["proj384_frac"]) / 2
                if best is None or b < best[0]:
                    best = (b, cell, pj)
        if best:
            print(f"| {w} | {pol} | {best[0]:.3f} | {best[1]} | {1-best[0]:.1%} | {1-best[2]:.1%} |")
        else:
            print(f"| {w} | {pol} | none | - | 0% | 0% |")

# ---- paired contrasts between forgetting policies at EQUAL budget (pooled held+confirm, n=60 seeds, 95% t-CI) ----
import math
def pooled(w, cell):
    return H[(w, cell)]["d_per_seed"] + C[(w, cell)]["d_per_seed"]
def ci(x):
    n = len(x); m = sum(x) / n
    s = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1))
    t = 2.001 if n >= 60 else 2.045
    hw = t * s / math.sqrt(n)
    return m, m - hw, m + hw
print("\n## Paired contrasts at equal byte budget, pooled held+confirm (60 seeds): policy minus FIFO\n")
print("| world | budget | value-fifo | merge-fifo | lru-fifo | tiered-int8-fifo | tiered-int8+merge-fifo | merge-value |")
print("|---|---|---|---|---|---|---|---|")
for w in MAIN + ["A-allvalued"]:
    for b in ["50%", "25%", "10%"]:
        def d(a, c):
            x = [p - q for p, q in zip(pooled(w, f"{a}@{b}"), pooled(w, f"{c}@{b}"))]
            m, lo, hi = ci(x)
            return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]"
        print(f"| {w} | {b} | {d('value','fifo')} | {d('merge','fifo')} | {d('lru','fifo')} | {d('tiered-int8','fifo')} | {d('tiered-int8+merge','fifo')} | {d('merge','value')} |")
