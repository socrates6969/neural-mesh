"""R36+ enlarged verbal battery (V2): 30 items/skill/seed (was 6). Axes: v2_vocab, v2_analogies, v2_cloze per world A,B.
Seeds 1-90 (TUNE/HELD/CONF as iron rule). Candidates are scoring functions; all texts embedded by the local MiniLM (embed.py)."""
import os, sys, random, pickle, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_verbal2 as V
from lib import *
N2 = 30
CACHE = os.path.join(RES, "v2_emb.pkl")
_E = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
def emb(t):
    if t not in _E: embed_many([t])
    return _E[t]
def embed_many(ts):
    ts = [t for t in dict.fromkeys(ts) if t not in _E]
    if not ts: return
    from embed import embed
    vs = embed(ts)
    for t, v in zip(ts, vs): _E[t] = v
    pickle.dump(_E, open(CACHE, "wb"))

def gen_items(w, seed):
    vocab = V.VOCAB2_A if w == "A" else V.VOCAB2_B; rel = V.REL2_A if w == "A" else V.REL2_B; cloze = V.CLOZE2_A if w == "A" else V.CLOZE2_B
    r = random.Random(seed * 7001 + (1 if w == "B" else 0)); out = {"vocab": [], "analogies": [], "cloze": []}
    for wd, syn, ant in r.sample(vocab, N2):
        others = [x[1] for x in vocab if x[0] != wd]; opts = [syn, ant] + r.sample(others, 2); r.shuffle(opts)
        out["vocab"].append(dict(t=wd, opts=opts, ans=opts.index(syn)))
    for _ in range(N2):
        rn = r.choice(list(rel)); pairs = r.sample(rel[rn], 4); (a, b), (c, d) = pairs[0], pairs[1]
        rest = [p[1] for p in rel[rn] if p[1] not in (b, d) and p[1] not in [q[1] for q in pairs[2:]]]
        opts = [d] + [p[1] for p in pairs[2:]] + [r.choice(rest)]; r.shuffle(opts)
        out["analogies"].append(dict(abc=[a, b, c], opts=opts, ans=opts.index(d)))
    for st, an, ds in r.sample(cloze, min(N2, len(cloze))):
        opts = [an] + ds; r.shuffle(opts); out["cloze"].append(dict(st=st, opts=opts, ans=opts.index(an)))
    return out

_ITEMS = {}
def items(w, s):
    if (w, s) not in _ITEMS: _ITEMS[(w, s)] = gen_items(w, s)
    return _ITEMS[(w, s)]

def all_texts(fns_texts, seeds):
    ts = []
    for w in WORLDS:
        for s in seeds:
            it = items(w, s)
            for sk in it:
                for x in it[sk]: ts += fns_texts(sk, x)
    return ts

def cosm(a, B): return (B @ a) / (np.linalg.norm(a) * np.linalg.norm(B, axis=1) + 1e-9)

# ---- scorers: (sk, item, E) -> predicted index
def base_pred(sk, x, E=emb):
    O = np.stack([E(o) for o in x["opts"]])
    if sk == "vocab": return int(np.argmax(cosm(E(x["t"]), O)))
    if sk == "analogies":
        a, b, c = [E(q) for q in x["abc"]]; return int(np.argmax(cosm(b - a + c, O)))
    st = x["st"]; F = np.stack([E(st.replace("___", o)) for o in x["opts"]]); return int(np.argmax(cosm(E(st), F)))
def base_texts(sk, x):
    if sk == "vocab": return [x["t"]] + x["opts"]
    if sk == "analogies": return x["abc"] + x["opts"]
    return [x["st"]] + x["opts"] + [x["st"].replace("___", o) for o in x["opts"]]

def run_v2(pred=base_pred, textsf=base_texts, seeds=None, skills=("vocab", "analogies", "cloze")):
    seeds = seeds or (TUNE + HELD + CONF)
    embed_many(all_texts(textsf, seeds))
    tab = {}
    for w in WORLDS:
        for s in seeds:
            it = items(w, s); tab[(w, s)] = {f"v2_{sk}": float(np.mean([pred(sk, x) == x["ans"] for x in it[sk]])) for sk in skills}
    return tab

def tab_mean(tab, ax, seeds):
    return float(np.mean([tab[(w, s)][ax] for w in WORLDS for s in seeds]))

def gate(base, cand, target, tol=0.002, name="", log=True):
    from gate import compare
    # compare() also loops world 'R' which is absent -> skipped automatically
    v, rows, gain, loss = compare(base, cand, target, tol=tol, quiet=True)
    print(f"=== {name}: {v} | target gains {[g for g in gain if g[0] in target] or 'none'} | losses {loss or 'none'}")
    for ax in sorted({r['ax'] for r in rows}):
        print("   ", f"{ax:14s}", " ".join(f"{r['w']}{r['sp'][0]} {r['mean']:+.3f}(ph{r['ph']:.2g},W/L {r['wins']}/{r['losses']})" for r in rows if r["ax"] == ax))
    if log:
        with open(f"{RES}/b1_rounds.jsonl", "a") as f: f.write(json.dumps(dict(name=name, verdict=v, gains=gain, losses=loss, target=target)) + "\n")
    return v, rows

if __name__ == "__main__":
    t0 = time.time(); b = run_v2(); print("embedded+scored", round(time.time() - t0))
    for sp, sd in (("TUNE", TUNE), ("HELD", HELD), ("CONF", CONF)):
        print(sp, {ax: round(tab_mean(b, ax, sd), 3) for ax in ("v2_vocab", "v2_analogies", "v2_cloze")})
    pickle.dump(b, open(os.path.join(RES, "v2_base.pkl"), "wb"))
