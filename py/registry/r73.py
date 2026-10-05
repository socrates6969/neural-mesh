"""R73: token cost of injecting the capability catalog (per entry / per agent) vs retrieving top-k and trimming descriptions.
Injected line = '<kind> <name>: <desc>' (tokens = 1.33 x words, as R43). Retrieval = hybrid .7 on the FULL text (unchanged); trimming only affects what is injected."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lookup_eval import *
ents, texts, E = load()
TK = lambda s: 1.33 * len(s.split())
def line(e, nw=None):
    d = e["desc"] if nw is None else " ".join(e["desc"].split()[:nw])
    return f"{e['kind']} {e['name']}: {d}"
full = np.array([TK(line(e)) for e in ents]); kinds = np.array([e["kind"] for e in ents])
print("entries", len(ents), "catalog tokens (all)", int(full.sum()))
for k in ("agent", "skill", "command", "plugin", "cli", "pip", "mcp"):
    m = kinds == k; print(f"  {k:8s} n={m.sum():4d} mean tok/entry {full[m].mean():5.1f} p90 {np.percentile(full[m],90):5.1f} total {int(full[m].sum())}")
bm = BM25(texts); allq = Q + Q2; QE = embed([q for q, _ in allq])
R = np.array([ranks(ents, .7 * z(E @ QE[i]) + .3 * z(bm.score(q)), acc) for i, (q, acc) in enumerate(allq)])
tops = [np.argsort(-(.7 * z(E @ QE[i]) + .3 * z(bm.score(q))))[:10] for i, (q, acc) in enumerate(allq)]
print("n queries", len(allq), "hybrid hit@1 %.3f @3 %.3f @5 %.3f @10 %.3f" % tuple(np.mean(R <= k) for k in (1, 3, 5, 10)))
print("inject-everything baseline: %d tokens/query, hit 1.000 by construction" % int(full.sum()))
for nw in (None, 40, 25, 15, 8):
    for k in (3, 5, 10):
        t = [sum(TK(line(ents[j], nw)) for j in tp[:k]) for tp in tops]
        print(f"desc<={str(nw):4s} words top-{k:2d}: hit {np.mean(R<=k):.3f}  tokens/query {np.mean(t):6.0f}  ({100*np.mean(t)/full.sum():.1f}% of catalog)")
print("misses at k=5:")
for i, (q, acc) in enumerate(allq):
    if R[i] > 5: print("  rank", R[i], q, sorted(acc), "| top3:", [ents[j]["name"] for j in tops[i][:3]])
