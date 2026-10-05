"""g1: embed the g1 corpus (docs, sentence chunks, queries + spell/keyword rewrites) offline with MiniLM-L6 and BGE-base (ONNX CPU).

Reuses ONLY already-on-disk, already-audited models (py/models/bge-base from f8, ~/.ruvector/models/all-MiniLM-L6-v2 from f5).
No network, no downloads, no new packages.  A bigger model was searched for on disk (HF cache, ~/.cache, ~/.ruvector, cognee venv): none exists.
Models are run SEQUENTIALLY (one ONNX session alive at a time) to keep RAM modest (BGE python working set ~0.5 GB).
Outputs (A/B = corpus tag):
  results/g1_texts_<A>.json            ordered unique texts (docs, sentences, query raw/spell/kw)
  results/g1_emb_<A>_<model>.f32       (n_texts x dim) f32 LE, L2-normalised     model in {mini, bge}
  results/g1_emb_<A>_bgei.f32          BGE with the query instruction prefix on the QUERY texts only (raw/spell/kw), same order as q_texts
  results/g1_embmeta_<A>.json          shapes, latencies, sizes, ram
Usage: python py/g1_embed.py <A|B>
"""
import importlib.util, json, os, random, sys, time
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("f8e", os.path.join(ROOT, "py", "f8_embed.py"))
f8e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f8e)

INSTR = "Represent this sentence for searching relevant passages: "


def main():
    tag = sys.argv[1]
    c = json.load(open(os.path.join(RES, "g1_corpus_%s.json" % tag), encoding="utf8"))
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
    json.dump(texts, open(os.path.join(RES, "g1_texts_%s.json" % tag), "w", encoding="utf8"))
    json.dump(q_uniq, open(os.path.join(RES, "g1_qtexts_%s.json" % tag), "w", encoding="utf8"))
    print("unique texts", len(texts), "unique query texts", len(q_uniq))
    meta = dict(tag=tag, n_texts=len(texts), n_qtexts=len(q_uniq), models={})
    rss0 = f8e.rss_mb()
    for name, d, pool, dim, path in (("mini", f8e.MINI, "mean", 384, f8e.MINI), ("bge", f8e.BGE, "cls", 768, f8e.BGE)):
        t0 = time.time()
        m = f8e.Embedder(d, pool, dim)
        load = time.time() - t0
        t0 = time.time()
        emb = m.all(texts)
        t_all = time.time() - t0
        emb.tofile(os.path.join(RES, "g1_emb_%s_%s.f32" % (tag, name)))
        # single-text latency on 150 random QUERY texts (batch 1, intra_op=4) -- what a deployed system pays per incoming task
        sample = random.Random(7).sample(q_uniq, 150)
        for t in sample[:5]:
            m.batch([t])
        ts = []
        for t in sample:
            t1 = time.perf_counter()
            m.batch([t])
            ts.append((time.perf_counter() - t1) * 1000)
        info = dict(dim=dim, load_s=round(load, 2), embed_all_s=round(t_all, 1), single_ms_mean=round(float(np.mean(ts)), 2),
                    single_ms_p50=round(float(np.median(ts)), 2), single_ms_p95=round(float(np.percentile(ts, 95)), 2),
                    model_bytes=os.path.getsize(os.path.join(path, "model.onnx")), ws_mb=round(f8e.rss_mb()[0]), peak_mb=round(f8e.rss_mb()[1]))
        if name == "bge":
            embi = m.all([INSTR + t for t in q_uniq])
            embi.tofile(os.path.join(RES, "g1_emb_%s_bgei.f32" % tag))
            info["instr_prefix"] = INSTR
        meta["models"][name] = info
        print(name, info, flush=True)
        del m
    meta["ws_baseline_mb"] = round(rss0[0])
    json.dump(meta, open(os.path.join(RES, "g1_embmeta_%s.json" % tag), "w"), indent=1)


if __name__ == "__main__":
    main()
