from lab import *
import lab, re, math
import lib
T = ["domain_r1", "domain_mrr", "domain_route", "real_mrr"]
QSTOP = set("what how why which does do can i me my you your should would could there their they he she them who whom whose when where".split())
def stem(t):
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(t) > len(suf) + 3 and t.endswith(suf): return t[:-len(suf)]
    return t
class Lex:
    wants_tags = True
    def __init__(s, docs, vecs, arm, alpha, tags=None, tok=None, expand=0, spell=False, lo=.3, hi=1.0, pw=2.0):
        s.V, s.lo, s.hi, s.pw = vecs, lo, hi, pw
        s.tok = tok or toks
        dd = list(docs)
        if expand and tags:
            by = {}
            for d, t in zip(docs, tags):
                for w in s.tok(d): by.setdefault(t, {}); by[t][w] = by[t].get(w, 0) + 1
            df = {}
            for t, c in by.items():
                for w in c: df[w] = df.get(w, 0) + 1
            top = {t: [w for w, _ in sorted(c.items(), key=lambda x: -x[1] * math.log(1 + len(by) / df[x[0]]))[:expand]] for t, c in by.items()}
            dd = [d + " " + " ".join(top[t]) for d, t in zip(docs, tags)]
        s.bm = lib.BM25.__new__(lib.BM25); s._init_bm(dd)
        s.spell = None
        if spell:
            freq = {}
            for d in docs:
                for w in re.findall(r"[a-z]+", d.lower()): freq[w] = freq.get(w, 0) + 1
            s.freq = freq; s.known = set(freq)
    def _init_bm(s, docs):
        b = s.bm; b.k1, b.b = 1.2, .75; b.post = {}; b.dl = np.array([len(s.tok(d)) for d in docs], np.float32)
        for i, d in enumerate(docs):
            c = {}
            for t in s.tok(d): c[t] = c.get(t, 0) + 1
            for t, f in c.items(): b.post.setdefault(t, []).append((i, f))
        b.n = len(docs); b.avg = float(b.dl.mean()) if b.dl.mean() > 0 else 1.0
        b.idf = {t: math.log((b.n - len(p) + .5) / (len(p) + .5) + 1) for t, p in b.post.items()}
    def fix(s, q):
        out = []
        for w in re.findall(r"[A-Za-z]+|[^A-Za-z]+", q):
            lw = w.lower()
            if w.isalpha() and len(w) >= 4 and lw not in s.known:
                best = None
                for cand in s._edits(lw):
                    if cand in s.freq and s.freq[cand] >= 2 and len(cand) >= 3 and (best is None or s.freq[cand] > s.freq[best]): best = cand
                out.append(best or w)
            else: out.append(w)
        return "".join(out)
    def _edits(s, w):
        L = "abcdefghijklmnopqrstuvwxyz"; sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        r = {a + b[1:] for a, b in sp if b} | {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1} | {a + c + b[1:] for a, b in sp if b for c in L} | {a + c + b for a, b in sp for c in L}
        return r
    def scores(s, qt, qv):
        q = s.fix(qt) if s.spell is not None or hasattr(s, "freq") else qt
        bm = s.bm.score(q); seen = {t for t in s.tok(q) if t in s.bm.post}; den = sum(s.bm.idf[t] for t in seen) * (s.bm.k1 + 1)
        f = (bm.max() / den) if den > 0 else 0.0; a = s.hi - (s.hi - s.lo) * min(1.0, f) ** s.pw
        return a * z(s.V @ qv) + (1 - a) * z(bm)
def mk(**kw):
    f = lambda d, v, ar, al, tags=None: Lex(d, v, ar, al, tags, **kw); f.wants_tags = True; return f
if __name__ == "__main__":
    lab.BASE = cand(real=True, make=mk())
    json.dump({f"{k[0]}{k[1]}": v for k, v in lab.BASE.items()}, open(f"{RES}/b1_adaptive_table.json", "w"))
    run("R17 adaptive + spell-correction (doc vocab, OSA-1)", T, make=mk(spell=True))
    run("R18 adaptive + light stemmer in BM25", T, make=mk(tok=lambda s: [stem(t) for t in toks(s)]))
    run("R19 adaptive + query-stopwords (what/how/why...) removed", T, make=mk(tok=lambda s: [t for t in toks(s) if t not in QSTOP]))
    run("R20 adaptive + domain keyword doc-expansion (tags, top 6 terms)", T, make=mk(expand=6))
    run("R21 adaptive + domain keyword doc-expansion (tags, top 12 terms)", T, make=mk(expand=12))
