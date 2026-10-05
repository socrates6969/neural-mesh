"""h2: embed the FRESH h2 corpora (C2 = g1_corpus.py seed 5, D2 = seed 6) with the already-on-disk MiniLM-L6 ONNX only (no BGE needed, no download, no network).
Same text ordering / output conventions as py/g1_embed.py but with the h2_ file prefix (g1_* files are not touched).
Usage: python py/h2_embed.py <C2|D2>
Outputs: results/h2_texts_<tag>.json, h2_qtexts_<tag>.json, h2_emb_<tag>_mini.f32, h2_embmeta_<tag>.json
"""
import importlib.util, json, os, random, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("f8e", os.path.join(ROOT, "py", "f8_embed.py"))
f8e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f8e)


def main():
    tag = sys.argv[1]
    c = json.load(open(os.path.join(RES, "h2_corpus_%s.json" % tag), encoding="utf8"))
    texts, seen = [], {}

    def add(t):
        if t not in seen:
            seen[t] = len(texts)
            texts.append(t)

    for d in c["docs"]:
        add(d["text"])
        for s in d["sents"]:
            add(s)
    q_texts = []
    for q in c["queries"]:
        for k in ("text", "spell", "kw"):
            add(q[k])
            q_texts.append(q[k])
    q_uniq = sorted(set(q_texts))
    json.dump(texts, open(os.path.join(RES, "h2_texts_%s.json" % tag), "w", encoding="utf8"))
    json.dump(q_uniq, open(os.path.join(RES, "h2_qtexts_%s.json" % tag), "w", encoding="utf8"))
    print("unique texts", len(texts), "unique query texts", len(q_uniq))
    meta = dict(tag=tag, n_texts=len(texts), n_qtexts=len(q_uniq), models={})
    t0 = time.time()
    m = f8e.Embedder(f8e.MINI, "mean", 384)
    load = time.time() - t0
    t0 = time.time()
    emb = m.all(texts)
    t_all = time.time() - t0
    emb.tofile(os.path.join(RES, "h2_emb_%s_mini.f32" % tag))
    sample = random.Random(7).sample(q_uniq, 150)
    for t in sample[:5]:
        m.batch([t])
    ts = []
    for t in sample:
        t1 = time.perf_counter()
        m.batch([t])
        ts.append((time.perf_counter() - t1) * 1000)
    meta["models"]["mini"] = dict(dim=384, load_s=round(load, 2), embed_all_s=round(t_all, 1), single_ms_mean=round(float(np.mean(ts)), 2),
                                  model_bytes=os.path.getsize(os.path.join(f8e.MINI, "model.onnx")))
    print(meta["models"]["mini"], flush=True)
    json.dump(meta, open(os.path.join(RES, "h2_embmeta_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main()
