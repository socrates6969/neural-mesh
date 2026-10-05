"""h13: corpus statistics that could explain the g1-vs-h10 discrepancy (templated synthetic vs real paragraphs).
Reads only saved fixtures: results/h10_{corpus,texts,emb,inst}_{A,B}.*, results/g1_{corpus,texts}_{A,B}.json + g1_emb_{A,B}_mini.f32.
Per corpus: doc length, vocabulary, share of hapax terms; NEAR-DUPLICATE density (max cosine of each doc to any other doc, share >= .90/.95;
token-Jaccard >= .6 partners); SIBLING crowding of the relevant docs (number of non-relevant docs closer than .85 cosine to a relevant doc);
query-relevant-doc dense cosine and its gap to the best non-relevant doc.  (Lexical query-doc overlap is computed in Rust: results/h13_retrieval.json.)
Output: results/h13_corpus_stats.json   Usage: python py/h13_corpus_stats.py
"""
import collections, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
DIM = 384
STOP = set("the a an of to and in is it for on that this with as are be by or at from was can if you we not will has have its".split())


def toks(s):
    return [t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in STOP]


def doc_stats(docs, E, rels, qE):
    """docs: list[str]; E: n x DIM unit vectors; rels: list of relevant doc-index lists per query; qE: q x DIM clean-query vectors"""
    n = len(docs)
    T = [toks(d) for d in docs]
    lens = np.array([len(t) for t in T])
    vocab = collections.Counter(w for t in T for w in set(t))
    allw = collections.Counter(w for t in T for w in t)
    out = {"n_docs": n, "mean_tokens": float(lens.mean()), "median_tokens": float(np.median(lens)), "vocab": len(vocab),
           "type_token_ratio": len(allw) / sum(allw.values()), "hapax_share_of_vocab": sum(1 for w, c in vocab.items() if c == 1) / len(vocab)}
    # near duplicates by embedding
    mx = np.empty(n, dtype=np.float32)
    for i in range(0, n, 1024):
        S = E[i:i + 1024] @ E.T
        for j in range(S.shape[0]):
            S[j, i + j] = -1
        mx[i:i + 1024] = S.max(1)
    out["nn_cos"] = {"mean": float(mx.mean()), "p50": float(np.median(mx)), "share_ge_.90": float((mx >= 0.90).mean()), "share_ge_.95": float((mx >= 0.95).mean()), "share_ge_.98": float((mx >= 0.98).mean())}
    # near duplicates by token Jaccard (inverted index over terms with df <= 150)
    sets = [set(t) for t in T]
    post = collections.defaultdict(list)
    for i, s in enumerate(sets):
        for w in s:
            if vocab[w] <= 150:
                post[w].append(i)
    has_partner = 0
    for i, s in enumerate(sets):
        cnt = collections.Counter()
        for w in s:
            if vocab[w] <= 150:
                for j in post[w]:
                    if j != i:
                        cnt[j] += 1
        ok = False
        for j, c in cnt.items():
            if c / (len(s) + len(sets[j]) - c) >= 0.6:
                ok = True
                break
        has_partner += ok
    out["share_docs_with_jaccard_ge_.6_partner"] = has_partner / n
    # sibling crowding around relevant docs, and the query-doc cosine gap
    crowd, qcos, gap = [], [], []
    for qi, rel in enumerate(rels):
        if not rel:
            continue
        r0 = rel[0]
        s = E @ E[r0]
        s[rel] = -1
        crowd.append(int((s >= 0.85).sum()))
        sq = E @ qE[qi]
        best_rel = max(sq[r] for r in rel)
        mask = np.ones(n, bool)
        mask[rel] = False
        qcos.append(float(best_rel))
        gap.append(float(best_rel - sq[mask].max()))
    crowd = np.array(crowd)
    out["siblings_ge_.85_of_relevant_doc"] = {"mean": float(crowd.mean()), "share_with_any": float((crowd > 0).mean()), "p90": float(np.percentile(crowd, 90))}
    out["query_to_relevant_cos"] = {"mean": float(np.mean(qcos)), "median": float(np.median(qcos))}
    out["best_relevant_minus_best_nonrelevant_cos"] = {"mean": float(np.mean(gap)), "share_negative": float((np.array(gap) < 0).mean())}
    return out


def h10(tag):
    c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
    docs = [d["text"] for d in c["docs"]]
    n = len(docs)
    E = np.fromfile(os.path.join(RES, "h10_emb_%s.f32" % tag), dtype=np.float32).reshape(-1, DIM)
    texts = json.load(open(os.path.join(RES, "h10_texts_%s.json" % tag), encoding="utf8"))
    inst = json.load(open(os.path.join(RES, "h10_inst_%s.json" % tag)))
    qs = inst["queries"]
    seen, rels, qv = set(), [], []
    for s in ("31", "32", "33", "61", "62", "63") if tag == "A" else ("61", "62", "63"):
        for r in inst["inst"][s]["rows"]:
            if r["q"] in seen:
                continue
            seen.add(r["q"])
            rels.append(qs[r["q"]]["rel"])
            qv.append(E[r["t"][0][0]])
    return doc_stats(docs, E[:n], rels, np.array(qv)), len(rels)


def g1(tag):
    c = json.load(open(os.path.join(RES, "g1_corpus_%s.json" % tag), encoding="utf8"))
    texts = json.load(open(os.path.join(RES, "g1_texts_%s.json" % tag), encoding="utf8"))
    tix = {t: i for i, t in enumerate(texts)}
    Eall = np.fromfile(os.path.join(RES, "g1_emb_%s_mini.f32" % tag), dtype=np.float32).reshape(-1, DIM)
    docs = [d["text"] for d in c["docs"]]
    E = np.array([Eall[tix[d]] for d in docs])
    dconc = np.array([d["concept"] for d in c["docs"]])
    rels, qv = [], []
    for q in c["queries"]:
        if q["concept"] < 0:
            continue
        rels.append([int(i) for i in np.where(dconc == q["concept"])[0]])
        qv.append(Eall[tix[q["text"]]])
    return doc_stats(docs, E, rels, np.array(qv)), len(rels)


def main():
    out = {}
    for tag in "AB":
        out["h10_" + tag], nq = h10(tag)
        out["h10_" + tag]["n_queries_used"] = nq
        print("h10", tag, json.dumps(out["h10_" + tag]), flush=True)
    for tag in "AB":
        out["g1_" + tag], nq = g1(tag)
        out["g1_" + tag]["n_queries_used"] = nq
        print("g1", tag, json.dumps(out["g1_" + tag]), flush=True)
    json.dump(out, open(os.path.join(RES, "h13_corpus_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
