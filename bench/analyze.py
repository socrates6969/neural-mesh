"""bench/analyze.py - paired statistics over bench/results/<tag>/raw.jsonl.
For each category and overall: accuracy per condition, completion rate, input completion ("parsed"),
mean total tokens per item (API usage fields), mean wall-clock seconds and API-reported seconds per item.
Paired 95% CIs: item-level bootstrap (10,000 resamples, seed 1) of mean(cond_x - cond_A) over items present in both.
Usage: python bench/analyze.py [tag] [--json out.json]"""
import json, random, sys, pathlib, collections
HERE = pathlib.Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "main"
rows = [json.loads(l) for l in open(HERE / "results" / tag / "raw.jsonl", encoding="utf8") if l.strip()]
by = collections.defaultdict(dict)       # (cond) -> id -> row
for r in rows: by[r["cond"]][r["id"]] = r
CATS = ["math", "physics", "code", "research", "research:public", "research:private", "ALL", "ALL_excl_private"]
def sel(r, cat):
    if cat == "ALL": return True
    if cat == "ALL_excl_private": return not (r["cat"] == "research" and r["sub"] == "private")
    if ":" in cat: c, s = cat.split(":"); return r["cat"] == c and r["sub"] == s
    return r["cat"] == cat
def val(r, m):
    if m == "acc": return float(r["correct"])
    if m == "complete": return float(r["completed"])
    if m == "parsed": return float(r.get("parsed", False))
    if m == "tokens": return float(r["tokens"]["total"]) if r.get("tokens") else None
    if m == "wall": return r["wall_s"]
    if m == "api_s": return (r["api_ms"] or 0) / 1000 if r.get("api_ms") is not None else None
def boot(d, n=10000, seed=1):
    if not d: return (float("nan"),) * 3
    rng = random.Random(seed); m = sum(d) / len(d)
    ms = sorted(sum(d[rng.randrange(len(d))] for _ in range(len(d))) / len(d) for _ in range(n))
    return m, ms[int(.025 * n)], ms[int(.975 * n)]
out = {}
for cat in CATS:
    out[cat] = {}
    for cond in sorted(by):
        rs = [r for r in by[cond].values() if sel(r, cat)]
        if not rs: continue
        o = {"n": len(rs)}
        for m in ("acc", "complete", "parsed", "tokens", "wall", "api_s"):
            v = [x for x in (val(r, m) for r in rs) if x is not None]
            o[m] = sum(v) / len(v) if v else None
        out[cat][cond] = o
    for cond in sorted(by):
        if cond == "A" or "A" not in by: continue
        ids = [i for i in by[cond] if i in by["A"] and sel(by[cond][i], cat)]
        if not ids: continue
        paired = {}
        for m in ("acc", "tokens", "wall", "api_s"):
            d = []
            for i in ids:
                a, b = val(by["A"][i], m), val(by[cond][i], m)
                if a is not None and b is not None: d.append(b - a)
            paired[m] = boot(d)
        out[cat].setdefault("paired_vs_A", {})[cond] = {"n": len(ids), **{m: [round(x, 4) for x in v] for m, v in paired.items()}}
json.dump(out, open(HERE / "results" / tag / "summary.json", "w"), indent=1)
f = lambda x, p=3: "-" if x is None else f"{x:.{p}f}"
for cat in CATS:
    if not out[cat]: continue
    print(f"== {cat}")
    for cond in sorted(by):
        o = out[cat].get(cond)
        if o: print(f"  {cond}: n={o['n']:3d} acc={f(o['acc'])} complete={f(o['complete'])} parsed={f(o['parsed'])} tok={f(o['tokens'],0)} wall_s={f(o['wall'],1)} api_s={f(o['api_s'],1)}")
    for cond, p in out[cat].get("paired_vs_A", {}).items():
        print(f"  {cond}-A (n={p['n']}): acc {p['acc'][0]:+.3f} [{p['acc'][1]:+.3f},{p['acc'][2]:+.3f}]  tokens {p['tokens'][0]:+.0f} [{p['tokens'][1]:+.0f},{p['tokens'][2]:+.0f}]  wall {p['wall'][0]:+.1f}s [{p['wall'][1]:+.1f},{p['wall'][2]:+.1f}]  api {p['api_s'][0]:+.1f}s [{p['api_s'][1]:+.1f},{p['api_s'][2]:+.1f}]")
