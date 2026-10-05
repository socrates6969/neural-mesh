"""k1 prep (KNOWLEDGE-STORE growth/compaction/freshness on REAL-text worlds A/B = saved h10 fixtures; world B is NOT rebuilt from live files).

REAL: paragraphs (h10 docs), their src/file structure, section structure (H-query rel lists), the queries, MiniLM embeddings of the docs.
SYNTHETIC (labelled): the edited versions we generate here:
  * OLD versions ("superseded revisions") of a fixed candidate set of units (a unit = an H-section or a single paragraph): v-1 (and for 1/3 of
    them also v-2) = the real current paragraph with a synthetic edit (word substitution 15-30% / sentence splice from another paragraph of the
    same crate / trim / extend). The edit strength is drawn per paragraph so cosine(old, new) covers a wide range.
  * NEAR-DUPLICATE variants (cos ~ .93-.995): case/punctuation/prefix/whitespace/sentence-order edits of the same paragraph.
  * NOISE records: real paragraphs of the OTHER world (never relevant), with a cluster = cluster of the nearest local doc.
All new texts are embedded OFFLINE with the on-disk MiniLM-L6 (same Embedder as py/h10_embed.py); no network, no download.
Outputs per world W: results/k1_meta_W.json (keys, units, queries, variants), results/k1_var_emb_W.f32 (variant+noise embeddings, 384 f32 each).
Usage: python py/k1_prep.py A|B
"""
import collections, hashlib, importlib.util, json, os, random, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
DIM = 384
STOP = set("the and for that this with from have been were their which will would could should there where when what into your more than then them they these those such also only other some most many much very just like over about after before while because between through under again further being does done doing each both same than too can may might must not are was has had its our you your use used using uses via per any all one two new".split())
WORD = re.compile(r"[A-Za-z]{4,}")
SENT = re.compile(r"(?<=[.!?])\s+")


def srng(*a):
    h = hashlib.sha1("|".join(map(str, a)).encode()).hexdigest()
    return random.Random(int(h[:12], 16))


def sentences(t):
    return [s for s in SENT.split(t.strip()) if s]


def edit_old(text, doc_i, depth, srcvocab, src_sents, strength):
    """synthetic OLD version of `text` (real paragraph). returns a text != text."""
    r = srng("old", doc_i, depth)
    ss = sentences(text)
    op = r.random()
    out = text
    if op < 0.25 and len(ss) >= 2 and src_sents:
        k = r.randrange(len(ss))
        ss[k] = r.choice(src_sents)
        out = " ".join(ss)
    elif op < 0.45 and len(ss) >= 2:
        out = " ".join(ss[:-1]) if r.random() < 0.5 else " ".join(ss + [r.choice(src_sents)] if src_sents else ss[:-1])
    if out == text:
        frac = [0.15, 0.3][strength]
        words = list(WORD.finditer(text))
        cand = [m for m in words if m.group(0).lower() not in STOP]
        n = max(1, int(round(frac * len(cand))))
        pick = set(r.sample(range(len(cand)), min(n, len(cand)))) if cand else set()
        pieces, last = [], 0
        for i, m in enumerate(cand):
            if i in pick and srcvocab:
                pieces.append(text[last:m.start()])
                pieces.append(r.choice(srcvocab))
                last = m.end()
        pieces.append(text[last:])
        out = "".join(pieces)
        out = re.sub(r"\d+", lambda m: str((int(m.group(0)) + 1 + r.randrange(8)) % 1000) if len(m.group(0)) < 4 else m.group(0), out)
    if out == text:
        out = text + " (legacy behaviour, superseded)"
    return out


def edit_near(text, doc_i, j):
    r = srng("near", doc_i, j)
    ss = sentences(text)
    for _ in range(8):
        t = text
        ops = r.sample(["period", "lower", "prefix", "ws", "order", "dropstop", "bullet"], 1 + (r.random() < 0.4))
        for op in ops:
            if op == "period":
                t = t.rstrip(".") if t.endswith(".") else t + "."
            elif op == "lower":
                t = t[0].lower() + t[1:]
            elif op == "prefix":
                t = r.choice(["Note: ", "Summary: ", "See also: ", "Docs: "]) + t
            elif op == "ws":
                t = t.replace(", ", ",  ").replace(". ", " . ")
            elif op == "order" and len(ss) >= 2:
                s2 = ss[:]
                s2[0], s2[1] = s2[1], s2[0]
                t = " ".join(s2)
            elif op == "dropstop":
                t = re.sub(r"\b(the|a|an)\s+", "", t, count=1, flags=re.I)
            elif op == "bullet":
                t = "- " + t
        if t != text:
            return t
    return text + " ."


def main():
    tag = sys.argv[1]
    other = "B" if tag == "A" else "A"
    spec = importlib.util.spec_from_file_location("f8e", os.path.join(ROOT, "py", "f8_embed.py"))
    f8e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f8e)
    c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
    docs = c["docs"]
    n = len(docs)
    E = np.fromfile(os.path.join(RES, "h10_emb_%s.f32" % tag), dtype=np.float32).reshape(-1, DIM)
    # structural keys: (src, file, ordinal of the paragraph inside its file)
    ordn = collections.Counter()
    key = []
    for d in docs:
        g = (d["src"], d["file"])
        key.append("%s/%s#%d" % (d["src"], d["file"], ordn[g]))
        ordn[g] += 1
    assert len(set(key)) == n
    # queries: clean text id from the saved instances (same text ids as the h10 retrieval used)
    inst = json.load(open(os.path.join(RES, "h10_inst_%s.json" % tag)))
    tid = {}
    for k, v in inst["inst"].items():
        for r in v["rows"]:
            if r["fam"] in ("H", "S") and r["q"] not in tid:
                tid[r["q"]] = r["t"][0][0]
    qs = []
    for q in c["queries"]:
        if q["fam"] in ("H", "S") and q["id"] in tid:
            qs.append(dict(id=q["id"], fam=q["fam"], rel=q["rel"], tid=tid[q["id"]], grp=q["grp"]))
    print("queries", len(qs), "of", len(c["queries"]), flush=True)
    # units: H-sections (disjoint rel lists) else singleton paragraphs
    unit_of = [-1] * n
    units = []
    for q in qs:
        if q["fam"] == "H":
            if all(unit_of[d] < 0 for d in q["rel"]):
                for d in q["rel"]:
                    unit_of[d] = len(units)
                units.append(list(q["rel"]))
    for d in range(n):
        if unit_of[d] < 0:
            unit_of[d] = len(units)
            units.append([d])
    # per-src vocabulary / sentences for the synthetic edits
    srcv = collections.defaultdict(collections.Counter)
    srcs = collections.defaultdict(list)
    for i, d in enumerate(docs):
        for m in WORD.finditer(d["text"]):
            w = m.group(0)
            if w.lower() not in STOP:
                srcv[d["src"]][w] += 1
        for s in sentences(d["text"]):
            if 30 <= len(s) <= 260:
                srcs[d["src"]].append((i, s))
    variants = []
    texts = []
    # OLD candidates: fixed 33% of units; 1/3 of them also get v-2
    for u, mem in enumerate(units):
        r = srng("cand", tag, u)
        if r.random() >= 0.33:
            continue
        deep = r.random() < 1 / 3
        for d in mem:
            sv = [w for w, _ in srcv[docs[d]["src"]].most_common(60)] or ["system"]
            ssl = [s for (j, s) in srcs[docs[d]["src"]] if j != d]
            strength = srng("str", d).randrange(2)
            t1 = edit_old(docs[d]["text"], d, 1, sv, ssl, strength)
            variants.append(dict(kind="old", base=d, depth=1, unit=u, strong=strength))
            texts.append(t1)
            if deep:
                t2 = edit_old(t1, d, 2, sv, ssl, srng("str2", d).randrange(2))
                variants.append(dict(kind="old", base=d, depth=2, unit=u, strong=0))
                texts.append(t2)
    # NEAR duplicates: fixed 20% of paragraphs; 1/3 get a second variant
    for d in range(n):
        r = srng("nd", tag, d)
        if r.random() >= 0.20:
            continue
        for j in range(1 + (r.random() < 1 / 3)):
            variants.append(dict(kind="near", base=d, depth=j))
            texts.append(edit_near(docs[d]["text"], d, j))
    nold = sum(1 for v in variants if v["kind"] == "old")
    nnear = sum(1 for v in variants if v["kind"] == "near")
    # NOISE: real paragraphs of the other world (fixed sample, 30% of n docs); embeddings copied (already computed)
    oc = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % other), encoding="utf8"))
    OE = np.fromfile(os.path.join(RES, "h10_emb_%s.f32" % other), dtype=np.float32).reshape(-1, DIM)
    nn = int(0.30 * n)
    pick = random.Random(77).sample(range(len(oc["docs"])), min(nn, len(oc["docs"])))
    noise_texts = [oc["docs"][i]["text"] for i in pick]
    noise_emb = OE[pick]
    cl12 = json.load(open(os.path.join(RES, "h13_world_%s.json" % tag)))["cl"]["12"]
    nn_doc = np.argmax(noise_emb @ E[:n].T, axis=1)
    noise_cl = [int(cl12[j]) for j in nn_doc]
    print("variants old", nold, "near", nnear, "noise", len(noise_texts), "units", len(units), flush=True)
    m = f8e.Embedder(f8e.MINI, "mean", DIM, threads=2)
    VE = m.all(texts)
    allemb = np.concatenate([VE, noise_emb.astype(np.float32)], axis=0)
    allemb.tofile(os.path.join(RES, "k1_var_emb_%s.f32" % tag))
    # cos(old|near, base) distributions
    for kind in ("old", "near"):
        cs = [float(VE[i] @ E[v["base"]]) for i, v in enumerate(variants) if v["kind"] == kind and v["depth"] == 1]
        print(kind, "cos to final: mean %.3f p5 %.3f p50 %.3f p95 %.3f min %.3f" % (np.mean(cs), np.percentile(cs, 5), np.median(cs), np.percentile(cs, 95), min(cs)))
    meta = dict(tag=tag, ndoc=n, key=key, src=[d["src"] for d in docs], units=units, unit_of=unit_of, queries=qs, variants=variants,
                var_texts=texts, noise_texts=noise_texts, noise_cl=noise_cl, cl12=cl12)
    json.dump(meta, open(os.path.join(RES, "k1_meta_%s.json" % tag), "w", encoding="utf8"))
    print("done", tag, "variant rows", len(texts), "+ noise", len(noise_texts))


if __name__ == "__main__":
    main()
