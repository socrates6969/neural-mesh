"""h22 prep: REAL sentence embeddings for the h10 replay tasks (MiniLM-L6, offline, on-disk model; no download).

Re-derives the exact task list of py/h10_replay.py (min_df=6 worlds A and B), ASSERTS it equals results/h10_replay_{A,B}.json tasks,
embeds every task sentence (REAL 384-d MiniLM vectors, L2-normalised), and writes
  results/h22_emb_W.f32   (n x 384 float32)         REAL sentence embeddings
  results/h22_pca_W.f32   (n x 64 float32)          PCA-whitened coordinates (unsupervised, pool statistics only)
  results/h22_pool_W.json {tasks, sents, para, clusters{K}, kwtok{K}, anchors{M}}
clusters: spherical k-means of the SENTENCE embeddings (K=12, 48; seeds fixed here); kwtok = token -> modal cluster (df>=3, share>=.4)
anchors:  nearest-centroid ids of an independent k-means (M=32, 128) = the idealised offline version of the repo's knowledge-anchor feature kn:<id>.
Usage: python py/h22_prep.py
"""
import collections, importlib.util, json, os, re, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("f8e", os.path.join(ROOT, "py", "f8_embed.py"))
f8e = importlib.util.module_from_spec(spec); spec.loader.exec_module(f8e)
spec2 = importlib.util.spec_from_file_location("h10r", os.path.join(ROOT, "py", "h10_replay.py"))
h10r = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(h10r)

STOP, WORD, sents, skmeans = h10r.STOP, h10r.WORD, h10r.sents, h10r.skmeans
DIM = 384


def build(tag, min_df=6):
    c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
    docs = c["docs"]; n = len(docs)
    toks = [sorted({w.lower() for w in WORD.findall(d["text"]) if len(w) >= 4 and w.lower() not in STOP}) for d in docs]
    df = collections.Counter(w for t in toks for w in t)
    ok = {w for w, f in df.items() if min_df <= f <= max(10, int(0.05 * n))}
    tasks, para, texts = [], [], []
    for di, d in enumerate(docs):
        for s in sents(d["text"]):
            if not 6 <= len(s.split()) <= 40:
                continue
            ws = sorted({w.lower() for w in WORD.findall(s) if len(w) >= 4 and w.lower() in ok}, key=lambda w: (df[w], w))
            if len(ws) < 2:
                continue
            tasks.append(ws[:6]); para.append(di); texts.append(s)
    fx = json.load(open(os.path.join(RES, "h10_replay_%s.json" % tag)))
    assert fx["tasks"] == tasks, "task list differs from h10 fixture"
    assert fx["para"] == para
    return tasks, para, texts


def main():
    m = f8e.Embedder(f8e.MINI, "mean", DIM)
    for tag in "AB":
        t0 = time.time()
        tasks, para, texts = build(tag)
        E = m.all(texts)
        n = len(texts)
        print(tag, "tasks", n, "embedded in %.1fs" % (time.time() - t0), "norm mean", float(np.linalg.norm(E, axis=1).mean()), flush=True)
        E.astype(np.float32).tofile(os.path.join(RES, "h22_emb_%s.f32" % tag))
        X = E - E.mean(0)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        Z = (X @ Vt[:64].T) / (S[:64] / np.sqrt(n))
        Z.astype(np.float32).tofile(os.path.join(RES, "h22_pca_%s.f32" % tag))
        ev = (S ** 2) / (S ** 2).sum()
        print(tag, "PCA var share top8/16/64: %.3f %.3f %.3f" % (ev[:8].sum(), ev[:16].sum(), ev[:64].sum()), flush=True)
        out = dict(n=n, tasks=tasks, sents=texts, para=para, clusters={}, kwtok={}, anchors={}, pca_var=[float(ev[:8].sum()), float(ev[:16].sum()), float(ev[:64].sum())])
        for K in (12, 48):
            lab = skmeans(E, K, seed=1000 + K)
            out["clusters"][str(K)] = [int(x) for x in lab]
            dist = collections.defaultdict(collections.Counter)
            for i, t in enumerate(tasks):
                for w in t:
                    dist[w][int(lab[i])] += 1
            kw = {}
            for w, cc in dist.items():
                tot = sum(cc.values()); j, mm = cc.most_common(1)[0]
                if tot >= 3 and mm / tot >= 0.4:
                    kw[w] = j
            out["kwtok"][str(K)] = kw
            print(tag, "K", K, "sizes top5", sorted(collections.Counter(int(x) for x in lab).values(), reverse=True)[:5], "kw tokens", len(kw), flush=True)
        for M in (32, 128):
            lab = skmeans(E, M, seed=2000 + M)
            out["anchors"][str(M)] = [int(x) for x in lab]
        json.dump(out, open(os.path.join(RES, "h22_pool_%s.json" % tag), "w"))


if __name__ == "__main__":
    main()
