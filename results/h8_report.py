#!/usr/bin/env python
"""H8 report generator (stdlib only): turns results/h8_*.json into the markdown tables used in h8_board_entry.md.
usage: python results/h8_report.py sweep|confirm|cross|memory|scale|conc|selfcheck > results/h8_<x>.md
"""
import json, sys, glob, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)))
def J(n):
    return json.load(open(os.path.join(R, n)))

def sweep():
    for ds in ["synth384-n30000", "real-mini", "real-bge"]:
        p = os.path.join(R, f"h8_hnsw_sweep_{ds}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rows = []
        for c in d["configs"]:
            for r in c["rows"]:
                b = r["bench"]
                rows.append(dict(rec=b["recall_at_10"], p50=b["p50_us"]["mean"], p99=b["p99_us"]["mean"], m=c["m"], efc=c["efc"], ef=r["ef"],
                                 heap=c["heap_bytes_per_vec"], build=c["build_s"], sr=c["self_recall"]))
        print(f"\n#### {ds} (n={d['n']}, dim={d['dim']}, nq={d['nq']}) - fixed crate, ONE build per cell, p50/p99 = mean of 3 passes")
        print("Pareto frontier (recall up, p99 down):\n| recall@10 | p50 us | p99 us | M | efc | ef | heap B/vec | build s |\n|---|---|---|---|---|---|---|---|")
        for f in d["pareto"]:
            m = [r for r in rows if r["m"] == f["cfg"]["m"] and r["efc"] == f["cfg"]["efc"] and r["ef"] == f["cfg"]["ef"]][0]
            print(f"| {m['rec']:.3f} | {m['p50']:.0f} | {m['p99']:.0f} | {m['m']} | {m['efc']} | {m['ef']} | {m['heap']:.0f} | {m['build']:.1f} |")
        print("\nBest recall at fixed p99 budget:\n| p99 budget us | best recall | config (M/efc/ef) | p50 | p99 |\n|---|---|---|---|---|")
        for bud in [150, 250, 400, 800, 1500]:
            c = [r for r in rows if r["p99"] <= bud]
            if c:
                b = max(c, key=lambda r: (r["rec"], -r["p99"]))
                print(f"| {bud} | {b['rec']:.3f} | {b['m']}/{b['efc']}/{b['ef']} | {b['p50']:.0f} | {b['p99']:.0f} |")
        print("\nMinimum p99 at recall target:\n| recall >= | min p99 us | config | p50 | recall |\n|---|---|---|---|---|")
        for t in [0.90, 0.95, 0.98, 0.99, 0.995]:
            c = [r for r in rows if r["rec"] >= t]
            if c:
                b = min(c, key=lambda r: r["p99"])
                print(f"| {t} | {b['p99']:.0f} | {b['m']}/{b['efc']}/{b['ef']} | {b['p50']:.0f} | {b['rec']:.3f} |")
            else:
                print(f"| {t} | not reached | | | |")
        tot = [(c["m"], c["efc"], c["self_recall"]) for c in d["configs"]]
        print("\nSelf-recall (own vector as query, ef=64, sample): " + "; ".join(f"M{m}/efc{e}: {s['not_top1']}/{s['queried']} not top-1, {s['absent_top10']} absent from top-10" for m, e, s in tot))
        by_m = {}
        for r in rows:
            if r["ef"] == 64 and r["efc"] == 200:
                by_m[r["m"]] = r["rec"]
        print("\nrecall at ef=64, efc=200 by M: " + ", ".join(f"M{k}: {v:.3f}" for k, v in sorted(by_m.items())))

def confirm():
    for ds in ["synth384-n30000", "real-mini", "real-bge"]:
        p = os.path.join(R, f"h8_hnsw_confirm_{ds}.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        print(f"\n#### {ds} (n={d['n']}, dim={d['dim']}) - 3 independent builds per row; mean [min-max] over builds of the per-build 3-pass mean")
        print("| target recall | cfg | kernel | recall@10 | p50 us | p99 us | build s | heap B/vec | self not-top1 (ef64, per build) |\n|---|---|---|---|---|---|---|---|---|")
        for r in d["results"]:
            def s(k, f="{:.0f}"):
                x = r[k]
                return f"{f.format(x['mean'])} [{f.format(x['min'])}-{f.format(x['max'])}]"
            k = "lib" if "lib" in r["kernel"] else "avx2"
            print(f"| >={r['target_recall']} | M{r['m']} efc{r['efc']} ef{r['ef']} | {k} | {s('recall_per_build','{:.3f}')} | {s('p50_us_per_build')} | {s('p99_us_per_build')} | {r['build_s']['mean']:.1f} | {r['heap_bytes_per_vec']['mean']:.0f} | {r['self_not_top1_per_build']} |")

def selfcheck(tag=""):
    p = os.path.join(R, f"h8_selfcheck{tag}.json")
    d = json.load(open(p))
    print("| dataset | M/efc | builds | ef64 not-top1 per build | ef512 not-top1 per build | ef512 absent-from-top10 per build |\n|---|---|---|---|---|---|")
    keys = []
    for r in d["rows"]:
        k = (r["dataset"], r["m"], r["efc"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        rr = [r for r in d["rows"] if (r["dataset"], r["m"], r["efc"]) == k]
        print(f"| {k[0]} (n={rr[0]['n']}) | M{k[1]}/efc{k[2]} | {len(rr)} | {[r['ef64']['not_top1'] for r in rr]} | {[r['ef512']['not_top1'] for r in rr]} | {[r['ef512']['absent_top10'] for r in rr]} |")

def cands(fn):
    d = J(fn)
    sizes = d["sizes"] if "sizes" in d else [d]
    for s in sizes:
        if "rows" not in s:
            print(f"N={s.get('n')}: {s.get('skipped')}")
            continue
        print(f"\n#### {fn} n={s['n']} dim={s.get('dim')}")
        print("| config | index heap B/vec | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us | build s |\n|---|---|---|---|---|---|---|---|---|")
        for r in s["rows"]:
            if r.get("extra") and isinstance(r["extra"], dict) and r["extra"].get("skipped"):
                print(f"| {r['name']} | - | - | - | - | - | - | - | - |")
                continue
            b = r["bench"]
            print(f"| {r['name']} | {r['index_heap_bytes_per_vec']:.1f} | {r['ram_bytes_per_vec_total']:.1f} | {r['disk_f32_rerank_bytes_per_vec']:.0f} | {r['m_vectors_per_decimal_GB_ram']:.2f} | {b['recall_at_10']:.3f} | {b['p50_us']['mean']:.0f} | {b['p99_us']['mean']:.0f} | {r['build_s']:.1f} |")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "sweep": sweep()
    elif m == "confirm": confirm()
    elif m == "lat2": lat2()
    elif m == "lat": lat()
    elif m == "selfcheck": selfcheck(sys.argv[2] if len(sys.argv) > 2 else "")
    elif m == "memory":
        for f in sorted(glob.glob(os.path.join(R, "h8_memory_*.json"))):
            cands(os.path.basename(f))
    elif m == "scale": cands("h8_scale.json")
