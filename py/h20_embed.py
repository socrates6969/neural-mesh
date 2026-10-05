"""h20: build query variants + embed offline with the on-disk MiniLM-L6 (py/f8_embed.py Embedder; no network, no downloads).
For every world W in {pydoc9, crates8, syn10} (corpora from py/h20_corpus.py / py/h20_syn.py):
  keep domains with >= MINDOM docs; query = held-out sentence (real worlds: random sentence of the paragraph, seeded; the doc text = paragraph minus
  that sentence; syn10: the query sentence defined by the generator);
  variants: clean | typo1 (one content word, one keyboard-adjacency edit, SYNTHETIC typo model of py/h10_embed.py), each also SPELL-CORRECTED with the
  document-vocabulary corrector (same algorithm as src/hybrid.rs SpellCorrector: OOV, >=4 letters, osa-1, max freq, freq>=2 targets) built from the world's docs.
Outputs: results/h20_world_W.json (docs with texts + the 4 query strings), results/h20_emb_W.f32 shape (n,5,384): doc, q_clean, q_typo, qs_clean, qs_typo.
Usage: C:/Users/mariu/.cognee-plugin/venv/Scripts/python.exe py/h20_embed.py [pydoc9 crates8 syn10]
"""
import collections, importlib.util, json, os, random, re, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
MINDOM = 60


def load(name, f):
    s = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "py", f))
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


h10e = load("h10e", "h10_embed.py")
h10c = load("h10c", "h10_corpus.py")
f8e = h10e.f8e

WORD = re.compile(r"[a-z]+")


class Speller:
    def __init__(self, texts):
        self.freq = collections.Counter()
        for t in texts:
            self.freq.update(WORD.findall(t.lower()))
        self.known = set(self.freq)
        self.idx = collections.defaultdict(list)
        for w, f in self.freq.items():
            if f >= 2 and len(w) >= 3:
                self.idx[w].append(w)
                for d in self.del1(w):
                    self.idx[d].append(w)

    @staticmethod
    def del1(w):
        return {w[:i] + w[i + 1:] for i in range(len(w))}

    @staticmethod
    def osa1(a, b):
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

    def fix(self, t):
        if t in self.known or len(t) < 4:
            return t
        cand = set(self.idx.get(t, []))
        for d in self.del1(t):
            cand.update(self.idx.get(d, []))
        cand = [x for x in cand if self.osa1(t, x)]
        if not cand:
            return t
        return max(cand, key=lambda x: (self.freq[x], x))

    def correct(self, s):
        return re.sub(r"[A-Za-z]+", lambda m: self.fix(m.group(0).lower()), s)


def build_world(tag):
    c = json.load(open(os.path.join(RES, "h20_corpus_%s.json" % tag), encoding="utf8"))
    cnt = collections.Counter(d["domain"] for d in c["docs"])
    keep = {d for d, n in cnt.items() if n >= MINDOM}
    docs = []
    for d in c["docs"]:
        if d["domain"] not in keep:
            continue
        if tag == "syn10":
            text, q = d["text"], d["query"]
        else:
            ss = h10c.sentences(d["text"])
            r = random.Random(h10e.srng(tag, d["id"], "cut").random())
            k = r.randrange(len(ss))
            q = ss[k]
            text = " ".join(s for i, s in enumerate(ss) if i != k)
            if len(text) < 40 or len(q) < 25:
                continue
        docs.append({"id": len(docs), "domain": d["domain"], "src": d["src"], "text": text, "query": q})
    sp = Speller([d["text"] for d in docs])
    for d in docs:
        qt = h10e.inject(d["query"], 1, h10e.srng(tag, d["id"], "typo"))
        d["query_typo"] = qt
        d["qs_clean"] = sp.correct(d["query"])
        d["qs_typo"] = sp.correct(qt)
    return docs, sorted(keep)


def main():
    tags = sys.argv[1:] or ["pydoc9", "crates8", "syn10"]
    emb = f8e.Embedder(f8e.MINI, "mean", 384, threads=4)
    for tag in tags:
        t0 = time.time()
        docs, keep = build_world(tag)
        texts = []
        for d in docs:
            texts += [d["text"], d["query"], d["query_typo"], d["qs_clean"], d["qs_typo"]]
        E = emb.all(texts).astype(np.float32)
        E.tofile(os.path.join(RES, "h20_emb_%s.f32" % tag))
        json.dump({"docs": docs, "domains": keep, "tag": tag, "dim": 384, "per_doc": 5, "typo_model": "py/h10_embed.py inject(k=1), SYNTHETIC"},
                  open(os.path.join(RES, "h20_world_%s.json" % tag), "w", encoding="utf8"))
        print(tag, len(docs), "docs", len(texts), "texts", "%.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
