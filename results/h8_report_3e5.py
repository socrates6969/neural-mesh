#!/usr/bin/env python
"""N=3e5 HNSW with 3 independent builds per config: python results/h8_report_3e5.py [tag]"""
import json, os, sys, re, collections
R = os.path.dirname(os.path.abspath(__file__))
tag = sys.argv[1] if len(sys.argv) > 1 else "hnsw3b"
d = json.load(open(os.path.join(R, f"h8_scale_{tag}.json")))
rows = d["sizes"][0]["rows"]
g = collections.OrderedDict()
for r in rows:
    m = re.match(r"HNSW M(\d+) efc(\d+) ef(\d+) \[(\w+) stored\]", r["name"])
    if not m:
        continue
    k = (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))
    g.setdefault(k, []).append(r)
print(f"| M | efc | ef | recall@10 mean [min-max] (3 builds) | p50 us | p99 us | heap B/vec | build s |\n|---|---|---|---|---|---|---|---|")
for (m, efc, ef, st), rs in g.items():
    rec = [r["bench"]["recall_at_10"] for r in rs]
    p50 = sum(r["bench"]["p50_us"]["mean"] for r in rs) / len(rs)
    p99 = sum(r["bench"]["p99_us"]["mean"] for r in rs) / len(rs)
    print(f"| {m} | {efc} | {ef} | {sum(rec)/len(rec):.3f} [{min(rec):.3f}-{max(rec):.3f}] | {p50:.0f} | {p99:.0f} | {rs[0]['index_heap_bytes_per_vec']:.0f} | {sum(r['build_s'] for r in rs)/len(rs):.0f} |")
