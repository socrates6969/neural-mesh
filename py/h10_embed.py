"""h10: build the query INSTANCES (synthetic keyboard-adjacency typos on REAL query text, + two spell-correctors) and embed
everything offline with the on-disk MiniLM-L6 (reuses py/f8_embed.py Embedder; no network, no downloads).

Typo model (SYNTHETIC; "published-style"): per eligible word (alphabetic, >= 4 letters) one single edit drawn with
  substitution by a QWERTY-adjacent key .35, deletion .25, insertion (adjacent key or doubled letter) .20, adjacent transposition .20
  (single-edit dominance, Damerau 1964 / Pollock & Zamora 1984; adjacent-key confusions, Kernighan-Church-Gale 1990); the first letter is
  edited with prob .07 (first-letter errors are rare).  Real-word errors are NOT filtered out (about a third of real typos are real words).
Variants per query instance: clean | typo1 (1 content word, 1 edit) | typo2 (2 different words, 1 edit each).
Spell correctors (computed from the DOCUMENT vocabulary only, ed<=1 incl. transposition, word must be OOV, >= 4 letters, alphabetic):
  S1 strict    : known = words of the doc store; candidates need doc freq >= 2; pick max freq.
  S2 g1-style  : S1 + known also includes every word used by >= 3 distinct pool queries (the g1 'query-log' rule; mildly transductive).
Outputs per world W: results/h10_texts_W.json (docs first), h10_emb_W.f32, h10_inst_W.json, h10_embmeta_W.json.
Usage: python py/h10_embed.py A|B
"""
import collections, hashlib, importlib.util, json, os, random, re, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("f8e", os.path.join(ROOT, "py", "f8_embed.py"))
f8e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f8e)

TOK = re.compile(r"[A-Za-z]+|[^A-Za-z]+")
WORD = re.compile(r"[a-z]+")
KB = {}
for row in ("qwertyuiop", "asdfghjkl", "zxcvbnm"):
    for i, c in enumerate(row):
        n = []
        if i > 0:
            n.append(row[i - 1])
        if i < len(row) - 1:
            n.append(row[i + 1])
        KB[c] = n
_rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
for r, row in enumerate(_rows):
    for i, c in enumerate(row):
        for rr in (r - 1, r + 1):
            if 0 <= rr < 3:
                for j in (i - 1, i, i + 1) if rr == r - 1 else (i - 1, i):
                    if 0 <= j < len(_rows[rr]):
                        KB[c].append(_rows[rr][j])


def srng(*a):
    h = hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def typo_word(w, r):
    n = len(w)
    for _ in range(10):
        op = r.random()
        pos = 0 if r.random() < 0.07 else r.randrange(1, n)
        c = w[pos].lower()
        if op < 0.35:  # substitution (adjacent key)
            nb = KB.get(c)
            if not nb:
                continue
            x = r.choice(nb)
            out = w[:pos] + (x.upper() if w[pos].isupper() else x) + w[pos + 1:]
        elif op < 0.60:  # deletion
            out = w[:pos] + w[pos + 1:]
        elif op < 0.80:  # insertion (doubled letter or adjacent key)
            x = c if r.random() < 0.5 else r.choice(KB.get(c, [c]))
            out = w[:pos] + x + w[pos:]
        else:  # adjacent transposition
            p = min(pos, n - 2)
            out = w[:p] + w[p + 1] + w[p] + w[p + 2:]
        if out != w and out.lower() != w.lower():
            return out
    return w[:-1] if n > 4 else w


def inject(text, k, r):
    toks = TOK.findall(text)
    el = [i for i, t in enumerate(toks) if t.isalpha() and len(t) >= 4]
    r.shuffle(el)
    for i in el[:k]:
        toks[i] = typo_word(toks[i], r)
    return "".join(toks)


def del1(w):
    return {w[:i] + w[i + 1:] for i in range(len(w))}


def osa1(a, b):
    """True iff optimal-string-alignment distance(a,b) <= 1 (a != b)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1 or a == b:
        return False
    if la == lb:
        d = [i for i in range(la) if a[i] != b[i]]
        if len(d) == 1:
            return True
        return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]
    s, l = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(s) and s[i] == l[i]:
        i += 1
    return s[i:] == l[i + 1:]


class Speller:
    def __init__(self, freq, known):
        self.freq, self.known = freq, known
        self.idx = collections.defaultdict(list)
        for w, f in freq.items():
            if f >= 2 and len(w) >= 3:
                self.idx[w].append(w)
                for d in del1(w):
                    self.idx[d].append(w)
        self.cache = {}

    def fix(self, t):
        if t in self.known or len(t) < 4:
            return t
        if t in self.cache:
            return self.cache[t]
        cand = set(self.idx.get(t, ()))
        for d in del1(t):
            cand |= set(self.idx.get(d, ()))
        c1 = [x for x in cand if osa1(t, x)]
        out = max(c1, key=lambda x: (self.freq[x], x)) if c1 else t
        self.cache[t] = out
        return out

    def text(self, s):
        return "".join(self.fix(t.lower()) if t.isalpha() else t for t in TOK.findall(s))


def main():
    tag = sys.argv[1]
    c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
    docs, queries = c["docs"], c["queries"]
    freq = collections.Counter(w for d in docs for w in WORD.findall(d["text"].lower()))
    qdf = collections.Counter(w for q in queries for w in set(WORD.findall(q["text"].lower())))
    known1 = set(freq)
    known2 = known1 | {w for w, n in qdf.items() if n >= 3}
    sp1, sp2 = Speller(freq, known1), Speller(freq, known2)

    texts, tid = [], {}

    def T(s):
        if s not in tid:
            tid[s] = len(texts)
            texts.append(s)
        return tid[s]

    for d in docs:  # docs are appended WITHOUT de-duplication so that text index == doc id
        texts.append(d["text"])
    ndoc = len(texts)
    assert ndoc == len(docs)
    byfam = collections.defaultdict(list)
    for q in queries:
        byfam[q["fam"]].append(q["id"])

    def split_of(q):
        key = (q["sec"] if q["fam"] == "H" else q["rel"][0])
        return int(hashlib.sha1(("split%d" % key).encode()).hexdigest()[:6], 16) % 5

    plan = []  # (seed, pool name, n_H, n_S)
    if tag == "A":
        plan = [(s, "tune", 120, 120) for s in range(1, 9)] + [(s, "held", 120, 120) for s in range(31, 91)]
    else:
        plan = [(s, "all", 100, 100) for s in range(61, 91)]
    inst = {}
    for seed, pool, nh, ns in plan:
        rows = []
        for fam, n in (("H", nh), ("S", ns)):
            ids = byfam[fam]
            if tag == "A":
                ids = [i for i in ids if (split_of(queries[i]) == 0) == (pool == "tune")]
            r = random.Random(seed * 7919 + (1 if fam == "H" else 2))
            pick = r.sample(ids, min(n, len(ids)))
            for qi in pick:
                q = queries[qi]
                vs = [q["text"]]
                for k in (1, 2):
                    vs.append(inject(q["text"], k, srng(tag, seed, qi, k)))
                rows.append(dict(q=qi, fam=fam, t=[[T(v), T(sp1.text(v)), T(sp2.text(v))] for v in vs]))
        inst["%d" % seed] = dict(pool=pool, rows=rows)
    # real-typo family R: fixed set (no resampling): text = as written by the human, clean = typo fixed
    real = []
    for q in queries:
        if q["fam"] == "R":
            real.append(dict(q=q["id"], fam="R", t=[[T(q["text"]), T(sp1.text(q["text"])), T(sp2.text(q["text"]))], [T(q["clean"]), T(q["clean"]), T(q["clean"])]]))
    inst["R"] = dict(pool="real", rows=real)
    print("world", tag, "docs", ndoc, "unique texts", len(texts), "seeds", len(plan), "real-typo queries", len(real), flush=True)
    t0 = time.time()
    m = f8e.Embedder(f8e.MINI, "mean", 384)
    emb = m.all(texts)
    t_all = time.time() - t0
    emb.tofile(os.path.join(RES, "h10_emb_%s.f32" % tag))
    sample = random.Random(7).sample(texts[ndoc:], min(150, len(texts) - ndoc))
    ts = []
    for t in sample[:5]:
        m.batch([t])
    for t in sample:
        t1 = time.perf_counter()
        m.batch([t])
        ts.append((time.perf_counter() - t1) * 1000)
    json.dump(texts, open(os.path.join(RES, "h10_texts_%s.json" % tag), "w", encoding="utf8"))
    json.dump(dict(ndoc=ndoc, queries=[{k: q[k] for k in ("id", "fam", "rel")} for q in queries], inst=inst),
              open(os.path.join(RES, "h10_inst_%s.json" % tag), "w"))
    # spell-correction cost (python, per query text)
    t1 = time.perf_counter()
    for s in sample:
        sp1.cache.clear()
        sp1.text(s)
    spell_ms = (time.perf_counter() - t1) * 1000 / max(1, len(sample))
    json.dump(dict(tag=tag, n_texts=len(texts), n_docs=ndoc, embed_all_s=round(t_all, 1), single_ms_mean=round(float(np.mean(ts)), 2),
                   single_ms_p50=round(float(np.median(ts)), 2), single_ms_p95=round(float(np.percentile(ts, 95)), 2),
                   spell_ms_per_query_py=round(spell_ms, 3), vocab=len(freq), seeds=[p[0] for p in plan]),
              open(os.path.join(RES, "h10_embmeta_%s.json" % tag), "w"), indent=1)
    print("embedded in %.1fs; single-query MiniLM %.2f ms; spell %.3f ms/query" % (t_all, float(np.mean(ts)), spell_ms))


if __name__ == "__main__":
    main()
