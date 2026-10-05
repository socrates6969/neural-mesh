"""h13 prep: per-DOC topic clusters (same spherical k-means + seeds as py/h10_replay.py, K=12/48), keyword->modal-cluster tokens, and the
keyword FEATURES of every query text used by the routing replay (<=6 rarest recurring content words, df>=6), for the saved h10 fixtures.
No new data: reads results/h10_corpus_{A,B}.json, h10_emb_{A,B}.f32, h10_texts_{A,B}.json, h10_inst_{A,B}.json (md5 in results/h10_corpus.md5).
Output: results/h13_world_{A,B}.json   Usage: python py/h13_prep.py
"""
import collections, json, os, re
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
DIM = 384
STOP = set("""the and for that this with from have been were their which will would could should there where when what into your more than then them they these those such also only other some most many much very just like over about after before while because between through under again further being does done doing each both same than too can may might must not are was has had its our you your use used using uses via per any all one two new""".split())
WORD = re.compile(r"[A-Za-z]+")


def skmeans(X, k, seed, iters=30):  # verbatim from py/h10_replay.py
    rng = np.random.RandomState(seed)
    n = len(X)
    cent = [X[rng.randint(n)]]
    for _ in range(1, k):
        d = 1 - np.max(X @ np.array(cent).T, axis=1)
        d = np.clip(d, 0, None) ** 2
        cent.append(X[rng.choice(n, p=d / d.sum())])
    C = np.array(cent)
    for _ in range(iters):
        a = np.argmax(X @ C.T, axis=1)
        for j in range(k):
            m = X[a == j]
            if len(m):
                v = m.mean(0)
                C[j] = v / np.linalg.norm(v)
    return np.argmax(X @ C.T, axis=1)


def main():
    for tag in "AB":
        c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
        docs = c["docs"]
        n = len(docs)
        E = np.fromfile(os.path.join(RES, "h10_emb_%s.f32" % tag), dtype=np.float32).reshape(-1, DIM)[:n]
        toks = [sorted({w.lower() for w in WORD.findall(d["text"]) if len(w) >= 4 and w.lower() not in STOP}) for d in docs]
        df = collections.Counter(w for t in toks for w in t)
        ok = {w for w, f in df.items() if 6 <= f <= max(10, int(0.05 * n))}
        out = {"n_docs": n, "cl": {}, "kwtok": {}, "feat": {}}
        for K in (12, 48):
            lab = skmeans(E, K, seed=7 + K)
            out["cl"][str(K)] = [int(x) for x in lab]
            dist = collections.defaultdict(collections.Counter)
            for di, t in enumerate(toks):
                for w in t:
                    if w in ok:
                        dist[w][int(lab[di])] += 1
            kw = {}
            for w, cc in dist.items():
                tot = sum(cc.values())
                j, m = cc.most_common(1)[0]
                if tot >= 3 and m / tot >= 0.4:
                    kw[w] = j
            out["kwtok"][str(K)] = kw
        texts = json.load(open(os.path.join(RES, "h10_texts_%s.json" % tag), encoding="utf8"))
        inst = json.load(open(os.path.join(RES, "h10_inst_%s.json" % tag)))
        ids = set()
        for k, v in inst["inst"].items():
            for r in v["rows"]:
                for tri in r["t"]:
                    ids.add(tri[0])
        for tid in sorted(ids):
            ws = sorted({w.lower() for w in WORD.findall(texts[tid]) if len(w) >= 4 and w.lower() in ok}, key=lambda w: (df[w], w))
            out["feat"][str(tid)] = ws[:6]
        nf = sum(1 for v in out["feat"].values() if len(v) > 0)
        print(tag, "docs", n, "ok-vocab", len(ok), "raw text ids", len(ids), "with >=1 feature", nf, "kw tokens K12/K48", len(out["kwtok"]["12"]), len(out["kwtok"]["48"]))
        json.dump(out, open(os.path.join(RES, "h13_world_%s.json" % tag), "w"))


if __name__ == "__main__":
    main()
