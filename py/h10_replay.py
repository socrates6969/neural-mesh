"""h10 task 2: REAL-TEXT-SHAPED routing replay fixture.

What is REAL: the task texts (sentences of real documentation paragraphs), their keyword features (a long-tailed real vocabulary), and the
topic structure (spherical k-means over the REAL MiniLM paragraph embeddings -> cluster per paragraph; a sentence inherits its paragraph's
cluster).  What is SIMULATED (Rust side, py/h10_replay -> src/bin/h10_replay.rs): which agent is competent on which topic cluster, the
keyword->agent prior and every Bernoulli outcome.  So: real text and real topic structure, SIMULATED agents and outcomes.

Outputs results/h10_replay_{A,B}.json: {k: {K: {"cluster": [per task], ...}}, tasks:[feature lists], kwtok:{K:{token: modal_cluster}}}
Usage: python py/h10_replay.py   (after py/h10_embed.py A and B)
"""
import collections, json, os, re, sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
DIM = 384
STOP = set("""the and for that this with from have been were their which will would could should there where when what into your more than then them they these those such also only other some most many much very just like over about after before while because between through under again further being does done doing each both same than too can may might must not are was has had its our you your use used using uses via per any all one two new""".split())
WORD = re.compile(r"[A-Za-z]+")


def sents(p):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", p) if s.strip()]


def skmeans(X, k, seed, iters=30):
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
    a = np.argmax(X @ C.T, axis=1)
    return a


def main():
    stats = {}
    for tag, suffix, min_df in [(t, sf, md) for t in "AB" for sf, md in (("", 6), ("_tail", 2))]:
        c = json.load(open(os.path.join(RES, "h10_corpus_%s.json" % tag), encoding="utf8"))
        docs = c["docs"]
        n = len(docs)
        E = np.fromfile(os.path.join(RES, "h10_emb_%s.f32" % tag), dtype=np.float32).reshape(-1, DIM)[:n]
        # token df over paragraphs
        toks = [sorted({w.lower() for w in WORD.findall(d["text"]) if len(w) >= 4 and w.lower() not in STOP}) for d in docs]
        df = collections.Counter(w for t in toks for w in t)
        ok = {w for w, f in df.items() if min_df <= f <= max(10, int(0.05 * n))}
        tasks, para = [], []
        for di, d in enumerate(docs):
            for s in sents(d["text"]):
                if not 6 <= len(s.split()) <= 40:
                    continue
                ws = sorted({w.lower() for w in WORD.findall(s) if len(w) >= 4 and w.lower() in ok}, key=lambda w: (df[w], w))
                if len(ws) < 2:
                    continue
                tasks.append(ws[:6])
                para.append(di)
        out = dict(n_docs=n, n_tasks=len(tasks), tasks=tasks, para=para, K={})
        for K in (8, 12, 48):
            lab = skmeans(E, K, seed=7 + K)
            cl = [int(lab[p]) for p in para]
            # modal cluster of each token (over paragraphs), keep tokens with df>=3 and modal share >= .4
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
            sizes = collections.Counter(int(x) for x in lab)
            out["K"][str(K)] = dict(cluster=cl, kwtok=kw, sizes=sorted(sizes.values(), reverse=True))
            print(tag, "K", K, "cluster sizes (paragraphs) top5", sorted(sizes.values(), reverse=True)[:5], "kw tokens", len(kw), flush=True)
        feat = collections.Counter(w for t in tasks for w in t)
        stats[tag + suffix] = dict(min_df=min_df, n_tasks=len(tasks), distinct_features=len(feat), mean_feats=float(np.mean([len(t) for t in tasks])),
                          top_feature_share=feat.most_common(1)[0][1] / sum(feat.values()), singletons_share=sum(1 for v in feat.values() if v == 1) / len(feat))
        print(tag + suffix, stats[tag + suffix])
        json.dump(out, open(os.path.join(RES, "h10_replay_%s%s.json" % (tag, suffix)), "w"))
    json.dump(stats, open(os.path.join(RES, "h10_replay_stats.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
