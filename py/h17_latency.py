"""h17: cost of the REAL embedding path (offline python onnxruntime, on-disk all-MiniLM-L6-v2): file sizes, load time, RSS, per-text latency at batch 1 and 32,
on REAL h10 texts (doc paragraphs and query texts).  Hermetic: json/os/time/random/numpy/onnxruntime/tokenizers via py/f8_embed.py (no network, no download).
Output: results/h17_latency.json     Usage: C:/Users/mariu/.cognee-plugin/venv/Scripts/python.exe py/h17_latency.py [threads]
"""
import importlib.util, json, os, random, sys, time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("f8e", os.path.join(ROOT, "py", "f8_embed.py"))
f8e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f8e)

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MINI = f8e.MINI
res = {"threads": threads}
res["minilm_model_onnx_bytes"] = os.path.getsize(os.path.join(MINI, "model.onnx"))
res["minilm_tokenizer_json_bytes"] = os.path.getsize(os.path.join(MINI, "tokenizer.json"))
res["bge_base_dir_bytes_for_reference"] = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(f8e.BGE) for f in fs)
rss0 = f8e.rss_mb()
t0 = time.time()
emb = f8e.Embedder(MINI, "mean", 384, threads=threads)
res["load_s"] = time.time() - t0
rss1 = f8e.rss_mb()
res["rss_mb_before_after_load"] = [rss0[0], rss1[0]]
texts = json.load(open(os.path.join(RES, "h10_texts_A.json"), encoding="utf8"))
inst = json.load(open(os.path.join(RES, "h10_inst_A.json")))
ndoc = inst["ndoc"]
docs = texts[:ndoc]
qids = sorted({r["t"][0][0] for k, v in inst["inst"].items() if isinstance(v, dict) and "rows" in v for r in v["rows"]})
queries = [texts[i] for i in qids]
r = random.Random(17)
sd, sq = r.sample(docs, 192), r.sample(queries, 192)
res["mean_chars_doc"] = float(np.mean([len(t) for t in sd]))
res["mean_chars_query"] = float(np.mean([len(t) for t in sq]))
# sanity: harness reproduces the saved fixture rows (same texts -> same vectors)
ref = np.fromfile(os.path.join(RES, "h10_emb_A.f32"), dtype=np.float32).reshape(-1, 384)
chk = emb.batch(docs[:16])
res["max_abs_diff_vs_saved_h10_fixture_docs0_15"] = float(np.abs(chk - ref[:16]).max())
emb.batch(sq[:8])  # warm up


def lat_single(ts):
    out = []
    for t in ts:
        a = time.perf_counter()
        emb.batch([t])
        out.append((time.perf_counter() - a) * 1000)
    return out


def lat_batch(ts, B=32):
    a = time.perf_counter()
    for s in range(0, len(ts), B):
        emb.batch(ts[s:s + B])
    return (time.perf_counter() - a) * 1000 / len(ts)


def q(x):
    x = sorted(x)
    return {"mean": float(np.mean(x)), "p50": x[len(x) // 2], "p95": x[int(len(x) * 0.95)]}


for name, ts in (("query", sq), ("doc", sd)):
    res["batch1_ms_per_text_" + name] = q(lat_single(ts))
    res["batch32_ms_per_text_" + name] = lat_batch(sorted(ts, key=len))  # length-sorted like emb.all
res["rss_mb_after_run_peak"] = f8e.rss_mb()[1]
res["vector_bytes_f32_384"] = 384 * 4
res["note"] = "python onnxruntime CPU, incl. tokenisation and mean-pool, box shared with other workers (+-50-100%); compare with results/f6_py_latency.json"
json.dump(res, open(os.path.join(RES, "h17_latency.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
