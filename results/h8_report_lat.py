#!/usr/bin/env python
"""H8 latency tables (stdlib only). usage: python results/h8_report_lat.py lat2|lat > results/h8_lat2.md"""
import json, os, sys, statistics as st

R = os.path.dirname(os.path.abspath(__file__))
DS = ["synth384-n30000", "real-mini", "real-bge"]


def avg(arr, k):
    return sum(a[k] for a in arr) / len(arr)


def lat2():
    for ds in DS:
        p = os.path.join(R, f"h8_lat2_{ds}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        print(f"\n#### {ds} (n={d['n']}, dim={d['dim']}) - 3 builds per (M,efc); latency = p50/p99 of per-query MINIMUM over 7 interleaved passes (load-robust); mean [min-max] over builds")
        print("| M | efc | ef | recall@10 | p50 us (lib) | p99 us (lib) | p50 us (avx2) | p99 us (avx2) | avx2/lib p50 (paired) | heap B/vec | build s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        rows = []
        for r in d["results"]:
            L = r["lib"]
            X = r["avx2"]
            def f(arr, k):
                v = [a[k] for a in arr]
                return sum(v) / len(v), min(v), max(v)
            rec, p50, p99 = f(L, "recall"), f(L, "p50_min_us"), f(L, "p99_min_us")
            xs, ratio = "- | -", "-"
            if X:
                x50, x99 = f(X, "p50_min_us"), f(X, "p99_min_us")
                xs = f"{x50[0]:.0f} | {x99[0]:.0f}"
                ratio = f"{st.mean(x['p50_min_us'] / l['p50_min_us'] for x, l in zip(X, L)):.2f}"
            hp, bs = avg(L, "heap_bytes_per_vec"), avg(L, "build_s")
            print(f"| {r['m']} | {r['efc']} | {r['ef']} | {rec[0]:.3f} [{rec[1]:.3f}-{rec[2]:.3f}] | {p50[0]:.0f} [{p50[1]:.0f}-{p50[2]:.0f}] | {p99[0]:.0f} [{p99[1]:.0f}-{p99[2]:.0f}] | {xs} | {ratio} | {hp:.0f} | {bs:.1f} |")
            rows.append(dict(m=r["m"], efc=r["efc"], ef=r["ef"], rec=rec[0], recmin=rec[1], p50=p50[0], p99=p99[0], hp=hp))
        print("\nMinimum p99 (lib, per-query-min stat) at recall target (mean over builds; and requiring all 3 builds >= target):")
        print("| recall >= | min p99 us | cfg M/efc/ef | p50 | recall | heap B/vec | min p99 us (all builds >= target) | cfg |")
        print("|---|---|---|---|---|---|---|---|")
        for t in [0.95, 0.98, 0.99, 0.995]:
            c = [r for r in rows if r["rec"] >= t]
            c2 = [r for r in rows if r["recmin"] >= t]
            a = min(c, key=lambda r: r["p99"]) if c else None
            b = min(c2, key=lambda r: r["p99"]) if c2 else None
            fa = f"{a['p99']:.0f} | {a['m']}/{a['efc']}/{a['ef']} | {a['p50']:.0f} | {a['rec']:.3f} | {a['hp']:.0f}" if a else "not reached | | | |"
            fb = f"{b['p99']:.0f} | {b['m']}/{b['efc']}/{b['ef']}" if b else "- | -"
            print(f"| {t} | {fa} | {fb} |")
        print("\nBest recall at fixed p99 (per-query-min) budget:")
        print("| p99 budget us | best mean recall | cfg | p99 | p50 |")
        print("|---|---|---|---|---|")
        for bud in [100, 150, 200, 300, 500]:
            c = [r for r in rows if r["p99"] <= bud]
            if c:
                b = max(c, key=lambda r: (r["rec"], -r["p99"]))
                print(f"| {bud} | {b['rec']:.3f} | {b['m']}/{b['efc']}/{b['ef']} | {b['p99']:.0f} | {b['p50']:.0f} |")
            else:
                print(f"| {bud} | none | | | |")


def lat():
    for ds in DS:
        p = os.path.join(R, f"h8_lat_{ds}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        print(f"\n#### {ds}: sweep-picked configs, 3 builds, interleaved lib/avx2, per-query-min stat")
        print("| cfg | recall lib / avx2 | p50 lib | p99 lib | p50 avx2 | p99 avx2 |")
        print("|---|---|---|---|---|---|")
        for r in d["results"]:
            print(f"| M{r['m']} efc{r['efc']} ef{r['ef']} | {avg(r['lib'],'recall'):.3f} / {avg(r['avx2'],'recall'):.3f} | {avg(r['lib'],'p50_min_us'):.0f} | {avg(r['lib'],'p99_min_us'):.0f} | {avg(r['avx2'],'p50_min_us'):.0f} | {avg(r['avx2'],'p99_min_us'):.0f} |")


if __name__ == "__main__":
    {"lat2": lat2, "lat": lat}[sys.argv[1]]()
