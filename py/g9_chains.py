"""g9 (W10): extend a g1 corpus with FOLLOW-UP queries for the multi-step session world (src/bin/g9_multistep.rs).

A follow-up query is what an agent writes at step k+1 of a session after reading the evidence of step k: it asks for another procedure
(topic T') of the SAME detail pool, filling the detail slot with a surface form copied from the retrieved evidence document
(the detail of that evidence doc, d).  So the text space is: topic T' (30) x act paraphrase a (4) x pattern p (2) x detail d (8) x surface form f (3)
= 5760 clean machine-written queries.  They are appended to the corpus' query list (cls = "chain", with a "chain" field = [T', a, p, d, f]) and the
result is written as results/g1_corpus_<tag>c.json, which py/g1_embed.py <tag>c can embed with the on-disk MiniLM / BGE models (no download).
Usage: python py/g9_chains.py C      -> results/g1_corpus_Cc.json
Imports json/re/sys/collections/importlib only (deterministic, no network).
"""
import collections, importlib.util, json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("g1c", os.path.join(ROOT, "py", "g1_corpus.py"))
g1c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g1c)

PATS = [g1c.QUERY_PAT[0], g1c.QUERY_PAT[2]]  # "How do I {a}?" / "Need help: I have to {a}."


def main():
    tag = sys.argv[1]
    c = json.load(open(os.path.join(RES, "g1_corpus_%s.json" % tag), encoding="utf8"))
    docs, queries, concepts = c["docs"], c["queries"], c["concepts"]
    # same spell-corrector vocabulary as g1_corpus.build(): doc vocab + words recurring >= 3x in the (original) query log
    freq = collections.Counter(w for d in docs for w in g1c.words(d["text"]))
    qcnt = collections.Counter(w for q in queries for w in set(g1c.words(q["text"])))
    known = set(freq) | {w for w, n in qcnt.items() if n >= 3}

    def correct(w):
        if w in known or not w.isalpha() or len(w) < 4:
            return w
        c1 = [x for x in g1c.edits1(w) if x in freq]
        return max(c1, key=lambda x: freq[x]) if c1 else w

    def spell(text):
        toks = re.findall(r"\w+|\W+", text)
        return "".join(correct(t.lower()) if t.isalpha() else t for t in toks)

    cid = {(k["topic"], k["detail"]): k["id"] for k in concepts}
    forms = {(k["topic"], k["detail"]): k["forms"] for k in concepts}
    new = []
    for ti, (pool, acts) in enumerate(g1c.TOPICS):
        for a, act in enumerate(acts):
            for p, pat in enumerate(PATS):
                for d in range(8):
                    for f in range(3):
                        text = pat.format(a=act.format(D=forms[(ti, d)][f]))
                        new.append(dict(id=len(queries) + len(new), concept=cid[(ti, d)], topic=ti, cls="chain", text=text, spell=spell(text), kw=spell(text), chain=[ti, a, p, d, f]))
    c["queries"] = queries + new
    json.dump(c, open(os.path.join(RES, "g1_corpus_%sc.json" % tag), "w", encoding="utf8"), indent=0)
    print("tag", tag, "orig queries", len(queries), "chain queries", len(new), "total", len(c["queries"]))
    print(" sample:", new[0]["text"], "|", new[777]["text"])


if __name__ == "__main__":
    main()
