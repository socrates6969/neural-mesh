"""R98 scorer: model-in-the-loop selection accuracy among the top-5 registry candidates, by line format (full / <=15 words / <=8 words / name only).
Metrics: sel_acc = the chosen candidate is acceptable (queries with >= 1 acceptable candidate in the top-5, n=116 of 124); abstain_ok = answered 0 on the queries with none (n=8).
Baselines: retrieval top-1 acceptable. Paired exact McNemar of each arm vs `full`, Holm over 3. Tokens = cl100k PROXY of the listed lines."""
import sys, os, json, re
from scipy import stats
d = sys.argv[1]; key = json.load(open(os.path.join(d, "sel_key.json"))); ARMS = ["full", "w15", "w8", "name"]; TOK = {"full": 23948, "w15": 18842, "w8": 14852, "name": 6962}
ans = {}
for a in ARMS:
    p = os.path.join(d, f"ans_{a}.txt")
    if not os.path.exists(p): print("missing", p); continue
    m = {}
    for line in open(p, encoding="utf-8", errors="ignore"):
        g = re.match(r"\s*Q(\d+)\s*:\s*(\d)", line)
        if g: m[int(g.group(1))] = int(g.group(2))
    ans[a] = m; print(a, "answers parsed:", len(m), "of", len(key))
has = [i for i, k in enumerate(key) if any(k["ok"])]; none = [i for i, k in enumerate(key) if not any(k["ok"])]
def correct(a, i):
    c = ans[a].get(i)
    return c is not None and c > 0 and key[i]["ok"][c - 1]
print(f"retrieval top-1 acceptable among the {len(has)} answerable queries: {sum(key[i]['ok'][0] for i in has)}/{len(has)} = {sum(key[i]['ok'][0] for i in has)/len(has):.3f}")
res = {}
for a in ans:
    sa = [correct(a, i) for i in has]; ab = [ans[a].get(i) == 0 for i in none]; res[a] = sa
    print(f"{a:5s} sel_acc {sum(sa)}/{len(has)} = {sum(sa)/len(has):.3f}  abstain_ok {sum(ab)}/{len(none)}  tokens(proxy) {TOK[a]} ({100*(TOK[a]/TOK['full']-1):+.0f}% vs full)  chose-0 on answerable: {sum(1 for i in has if ans[a].get(i)==0)}")
ps = []; names = []
for a in ans:
    if a == "full" or "full" not in ans: continue
    w = sum(1 for x, y in zip(res[a], res["full"]) if x and not y); l = sum(1 for x, y in zip(res[a], res["full"]) if y and not x); p = 1.0 if w + l == 0 else stats.binomtest(w, w + l, .5).pvalue; ps.append(p); names.append((a, w, l))
adj = [min(1, p * (len(ps) - r)) for r, p in enumerate(sorted(ps))] if ps else []
for (a, w, l), p in zip(names, ps): print(f"  {a} vs full: W/L {w}/{l} exact p={p:.3g} (x3 Holm-bound {min(1,p*3):.3g})")
