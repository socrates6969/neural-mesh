"""R84: resolve R83 (profile + length-aware re-rank, lam chosen by looking at held/conf) with a FRESH real-text corpus.
TUNE: corpus A (crate docs) seeds 1-8 ONLY (pre-declared rule below).  CONFIRM: corpus B (this project's prose + ruvnet docs; 4,181
paragraphs, 1,034 real queries, never used for any tuning) with ITEM-DISJOINT halves of its unique queries (fixed permutation): H1 = held, H2 = confirm.
Rule (declared before B was touched): lam* = the lam in {0,.05,.1,.15,.2,.25,.3} with the fewest context tokens subject to
ctx_hit(A-TUNE, profile+lam) >= ctx_hit(A-TUNE, default hybrid .7 top-5) - 0.004.   Candidates on B: profile+lam* and the R83 lams .1/.2/.3.
Tokens = 1.33 x words (a PROXY).  Per-query paired tests: exact McNemar (hit), paired t (tokens); Holm over family."""
import sys, random, math; sys.path.insert(0, "../")
from lab import *
from lib import *
from r60 import mkp
from scipy import stats
tk = lambda t: max(1, int(1.33 * len(t.split())))
prof = mkp(.3, 1.0, 2.0)
def uq(Wd, seeds=None):
    out = {}
    for s, v in Wd.inst.items():
        if s == "R" or (seeds is not None and int(s) not in seeds): continue
        for row in v["rows"]: out[row["q"]] = row["t"][0][0]
    return out
def run(Wd, ret_fn, qmap, k=5, lam=0.0):
    T = np.array([tk(t) for t in Wd.docs]); L = np.log(T); L = L - L.mean(); res = {}
    for q, tid in qmap.items():
        rel = set(Wd.queries[q]["rel"]); sc = z(ret_fn.scores(Wd.texts[tid], Wd.emb[tid])); top = np.argsort(-(sc - lam * L))[:k]
        res[q] = (float(any(d in rel for d in top)), float(T[top].sum()))
    return res
def mcnemar(a, b):
    w = sum(1 for x, y in zip(a, b) if x > y); l = sum(1 for x, y in zip(a, b) if x < y)
    return w, l, (1.0 if w + l == 0 else float(stats.binomtest(w, w + l, .5).pvalue))
def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); out = [0] * len(ps); m = len(ps); run_ = 0
    for r, i in enumerate(o): run_ = max(run_, min(1, ps[i] * (m - r))); out[i] = run_
    return out
if __name__ == "__main__":
    WA, WB = RealWorld("A"), RealWorld("B")
    LAMS = [0, .05, .1, .15, .2, .25, .3]
    qa = uq(WA, set(range(1, 9))); print("A tune unique queries", len(qa))
    base = run(WA, WA.retriever("hybrid", .7), qa); RPA = prof(WA.docs, WA.dvec, "hybrid", .7)
    bh = np.mean([v[0] for v in base.values()]); bt = np.mean([v[1] for v in base.values()]); print(f"A-TUNE default k5 hit {bh:.4f} tok {bt:.1f}")
    ok = []
    for lam in LAMS:
        r = run(WA, RPA, qa, lam=lam); h = np.mean([v[0] for v in r.values()]); t = np.mean([v[1] for v in r.values()]); print(f"  profile lam={lam}: hit {h:.4f} tok {t:.1f}")
        if h >= bh - 0.004: ok.append((t, lam))
    lam_star = min(ok)[1] if ok else None; print("lam* (pre-declared rule) =", lam_star, "(None = no lam satisfies the rule on A-TUNE)")
    qb = uq(WB); ids = sorted(qb); rr = random.Random(84); rr.shuffle(ids); half = {"H1": ids[:len(ids) // 2], "H2": ids[len(ids) // 2:]}
    print("B unique queries", len(ids), {k: len(v) for k, v in half.items()})
    bb = run(WB, WB.retriever("hybrid", .7), qb); RPB = prof(WB.docs, WB.dvec, "hybrid", .7)
    cands = sorted({x for x in (lam_star, 0.0, .1, .2, .3) if x is not None}); rows = []
    for lam in cands:
        r = run(WB, RPB, qb, lam=lam)
        for hn, hid in half.items():
            a = [r[q][0] for q in hid]; b = [bb[q][0] for q in hid]; w, l, p = mcnemar(a, b)
            ta = np.array([r[q][1] for q in hid]); tb = np.array([bb[q][1] for q in hid]); d = ta - tb; tp = float(stats.ttest_1samp(d, 0).pvalue)
            rows.append(dict(lam=lam, half=hn, hit_b=np.mean(b), hit_c=np.mean(a), dh=np.mean(a) - np.mean(b), W=w, L=l, p_hit=p, tok_b=tb.mean(), tok_c=ta.mean(), dt=d.mean(), p_tok=tp))
    ps = [x for r in rows for x in (r["p_hit"], r["p_tok"])]; adj = holm(ps)
    for i, r in enumerate(rows): r["ph_hit"], r["ph_tok"] = adj[2 * i], adj[2 * i + 1]
    print("B: default hybrid .7 top-5 vs profile + length term (all vs default):")
    for r in rows: print(f"  lam={r['lam']:<5} {r['half']}: hit {r['hit_b']:.4f}->{r['hit_c']:.4f} (d={r['dh']:+.4f} W/L {r['W']}/{r['L']} Holm p={r['ph_hit']:.3g})  tok {r['tok_b']:.1f}->{r['tok_c']:.1f} (d={r['dt']:+.2f} Holm p={r['ph_tok']:.3g})")
    # profile without length term vs default on B (is the R82 token penalty reproduced on fresh text?)
    r0 = run(WB, RPB, qb, lam=0.0)
    for hn, hid in half.items():
        print(f"  [R82 check] profile lam=0 {hn}: hit {np.mean([bb[q][0] for q in hid]):.4f}->{np.mean([r0[q][0] for q in hid]):.4f}  tok {np.mean([bb[q][1] for q in hid]):.1f}->{np.mean([r0[q][1] for q in hid]):.1f}")
    json.dump(dict(lam_star=lam_star, rows=rows), open(f"{RES}/r84.json", "w"), default=float)
