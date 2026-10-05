"""R66: ITEM-DISJOINT verbal splits. The V2 pools (50 vocab, 39/40 cloze, 48 analogy pairs per world) were shared by all seed splits, so
TUNE/HELD/CONF drew the same words (validity hole). Here every pool is partitioned into 3 DISJOINT parts (TUNE/HELD/CONF); each item appears once;
per-item paired exact McNemar tests, Holm over the family. Scorers are the R38/R49 ones (MiniLM default, BGE, cloze mix lam .5)."""
import sys; sys.path.insert(0, "../")
import random, itertools, numpy as np
from scipy import stats
import data_verbal2 as V
import v2, r49
from v2 import emb, cosm
from lib import holm
from r49 import bemb

def part(lst, seed):
    l = list(lst); random.Random(seed).shuffle(l); n = len(l); k = [0, round(n / 3), round(2 * n / 3), n]
    return [l[k[i]:k[i + 1]] for i in range(3)]

def items(w, seed=66):
    vocab = V.VOCAB2_A if w == "A" else V.VOCAB2_B; rel = V.REL2_A if w == "A" else V.REL2_B; cloze = V.CLOZE2_A if w == "A" else V.CLOZE2_B
    out = [dict(vocab=[], analogies=[], cloze=[]) for _ in range(3)]
    r = random.Random(seed * 13 + (w == "B"))
    for pi, P in enumerate(part(vocab, seed)):
        for wd, syn, ant in P:
            others = [x[1] for x in vocab if x[0] != wd]; opts = [syn, ant] + r.sample(others, 2); r.shuffle(opts)
            out[pi]["vocab"].append(dict(t=wd, opts=opts, ans=opts.index(syn)))
    for pi, P in enumerate(part(cloze, seed)):
        for st, an, ds in P:
            opts = [an] + ds; r.shuffle(opts); out[pi]["cloze"].append(dict(st=st, opts=opts, ans=opts.index(an)))
    for rn, pairs in rel.items():
        for pi, G in enumerate(part(pairs, seed)):
            for i, j in itertools.permutations(range(len(G)), 2):
                (a, b), (c, d) = G[i], G[j]; others = [G[k][1] for k in range(len(G)) if k not in (i, j)]
                rest = [p[1] for p in pairs if p[1] not in (b, d) and p[1] not in others]
                opts = [d] + others + [r.choice(rest)]; r.shuffle(opts)
                out[pi]["analogies"].append(dict(abc=[a, b, c], opts=opts, ans=opts.index(d)))
    return out

def mix(E, x, lam=.5):
    F = np.stack([E(x["st"].replace("___", o)) for o in x["opts"]]); O = np.stack([E(o) for o in x["opts"]]); s = E(x["st"])
    return int(np.argmax(cosm(s, F) + lam * cosm(s, O)))
ARMS = {"mini": lambda sk, x: v2.base_pred(sk, x, emb), "bge": lambda sk, x: v2.base_pred(sk, x, bemb),
        "mini+mix": lambda sk, x: mix(emb, x) if sk == "cloze" else v2.base_pred(sk, x, emb),
        "bge+mix": lambda sk, x: mix(bemb, x) if sk == "cloze" else v2.base_pred(sk, x, bemb)}
if __name__ == "__main__":
    IT = {w: items(w) for w in "AB"}
    # embed anything missing
    tsm, tsb = [], []
    for w in "AB":
        for pi in range(3):
            for sk, L in IT[w][pi].items():
                for x in L: tsm += v2.base_texts(sk, x)
    v2.embed_many(tsm); r49.bembed_many(tsm)
    names = ["TUNE", "HELD", "CONF"]; rows = []
    for sk in ("vocab", "analogies", "cloze"):
        for pi in range(3):
            X = [x for w in "AB" for x in IT[w][pi][sk]]
            C = {a: np.array([ARMS[a](sk, x) == x["ans"] for x in X]) for a in ARMS}
            print(f"{sk:9s} {names[pi]:4s} n={len(X):3d} " + " ".join(f"{a} {C[a].mean():.3f}" for a in ARMS))
            for cand, base in (("bge", "mini"), ("mini+mix", "mini"), ("bge+mix", "mini"), ("bge+mix", "bge")):
                if sk != "cloze" and "mix" in cand: continue
                b = int(((C[cand]) & ~C[base]).sum()); c = int((~C[cand] & C[base]).sum())
                p = stats.binomtest(b, b + c, .5).pvalue if b + c else 1.0
                rows.append(dict(sk=sk, sp=names[pi], cand=cand, base=base, d=float(C[cand].mean() - C[base].mean()), w=b, l=c, p=p, n=len(X)))
    ev = [r for r in rows if r["sp"] != "TUNE"]; adj = holm([r["p"] for r in ev])
    for r, a in zip(ev, adj): r["ph"] = a
    print("\nHELD/CONF paired exact McNemar, Holm over", len(ev), "tests")
    for r in ev: print(f"  {r['sk']:9s} {r['sp']} {r['cand']:9s} vs {r['base']:5s} d={r['d']:+.3f} W/L {r['w']}/{r['l']} p={r['p']:.3g} holm={r['ph']:.3g}")
    print("\nPOOLED HELD+CONF (disjoint from TUNE), exact McNemar")
    for sk in ("vocab", "analogies", "cloze"):
        X = [x for w in "AB" for pi in (1, 2) for x in IT[w][pi][sk]]
        C = {a: np.array([ARMS[a](sk, x) == x["ans"] for x in X]) for a in ARMS}
        for cand, base in (("bge", "mini"), ("bge+mix", "mini"), ("mini+mix", "mini"), ("bge+mix", "bge")):
            if sk != "cloze" and "mix" in cand: continue
            b = int((C[cand] & ~C[base]).sum()); c = int((~C[cand] & C[base]).sum()); p = stats.binomtest(b, b + c, .5).pvalue if b + c else 1
            print(f"  {sk:9s} n={len(X)} {cand:9s} vs {base:5s} {C[cand].mean():.3f} vs {C[base].mean():.3f} W/L {b}/{c} p={p:.3g}")
    for w in "AB":
        for sk in ("vocab","cloze"):
            X = [x for pi in (1, 2) for x in IT[w][pi][sk]]
            C = {a: np.array([ARMS[a](sk, x) == x["ans"] for x in X]) for a in ("mini","bge","bge+mix")}
            print(f"  world {w} {sk} n={len(X)} " + " ".join(f"{a} {C[a].mean():.3f}" for a in C))
