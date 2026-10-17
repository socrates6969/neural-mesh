#!/usr/bin/env python
"""H8 max-vectors-per-GB 3-D Pareto (stdlib only): python results/h8_report_mem.py > results/h8_pareto_mem.md
Non-dominated set over (RAM bytes/vec measured with the counting allocator, recall@10, p99 latency), recall >= 0.90 only.
Two views: (A) RAM-only (the f32 rerank copy, if used, is counted in RAM; no SSD); (B) f32 rerank rows allowed on SSD (direct I/O, cold), RAM = index only.
'SSD buffered' rows are excluded from (B) because the warm OS page cache is RAM."""
import json, os, glob

R = os.path.dirname(os.path.abspath(__file__))


def sets():
    for f in sorted(glob.glob(os.path.join(R, "h8_memory_*.json"))):
        d = json.load(open(f))
        yield d["dataset"] + " [memory mode]", d["n"], d["dim"], d["rows"]
    d = json.load(open(os.path.join(R, "h8_scale.json")))
    for s in d["sizes"]:
        if "rows" in s:
            yield f"synth384-n{s['n']} [scale mode, streamed]", s["n"], s["dim"], s["rows"]


def front(rr):
    out = []
    for x in rr:
        dom = any(y is not x and y["ram"] <= x["ram"] and y["rec"] >= x["rec"] and y["p99"] <= x["p99"] and (y["ram"] < x["ram"] or y["rec"] > x["rec"] or y["p99"] < x["p99"]) for y in rr)
        if not dom:
            out.append(x)
    return sorted(out, key=lambda x: x["ram"])


def show(title, rr):
    print(f"\n{title}\n| config | RAM B/vec | SSD B/vec | M vec/GB RAM | recall@10 | p50 us | p99 us |\n|---|---|---|---|---|---|---|")
    for x in front(rr):
        print(f"| {x['name']} | {x['ram']:.0f} | {x['ssd']:.0f} | {x['mpg']:.2f} | {x['rec']:.3f} | {x['p50']:.0f} | {x['p99']:.0f} |")


def main():
    for name, n, dim, rows in sets():
        rr = []
        for r in rows:
            if isinstance(r.get("extra"), dict) and r["extra"].get("skipped"):
                continue
            b = r["bench"]
            x = dict(name=r["name"], ram=r["ram_bytes_per_vec_total"], ssd=r["disk_f32_rerank_bytes_per_vec"], rec=b["recall_at_10"],
                     p50=b["p50_us"]["mean"], p99=b["p99_us"]["mean"], mpg=r["m_vectors_per_decimal_GB_ram"], buf="SSD buffered" in r["name"])
            if x["rec"] == x["rec"] and x["rec"] >= 0.90:
                rr.append(x)
        print(f"\n### {name} (n={n}, dim={dim})")
        show("(A) RAM-only 3-D Pareto (RAM B/vec, recall, p99):", [x for x in rr if x["ssd"] == 0 and not x["buf"]])
        show("(B) with f32 rerank rows on SSD (cold direct I/O) 3-D Pareto:", [x for x in rr if not x["buf"] and x["rec"] >= 0.90])


main()
