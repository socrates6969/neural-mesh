"""f8: embed the f5 SYNTHETIC corpus with BGE-base-en-v1.5 (ONNX, CPU) and time BGE vs MiniLM on the same harness.

Hermetic: imports only json/os/time/hashlib/ctypes/random/numpy/onnxruntime/tokenizers; reads only
py/models/bge-base/{model.onnx,tokenizer.json} (audited, see results/f8_audit.json + board), the local MiniLM files and
results/f5_corpus.json; opens no sockets; writes only results/f8_*.  Pooling = CLS + L2 normalise (BGE 1_Pooling config),
NO query instruction (symmetric text-to-text similarity).  Threads: intra_op=4.
Run: C:/Users/mariu/.cognee-plugin/venv/Scripts/python.exe py/f8_embed.py
"""
import ctypes, hashlib, json, os, random, time
from ctypes import wintypes
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = os.path.join(ROOT, "results")
BGE = os.path.join(ROOT, "py", "models", "bge-base")
MINI = os.path.expanduser("~/.ruvector/models/all-MiniLM-L6-v2")


def rss_mb():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    h = ctypes.windll.kernel32.GetCurrentProcess()
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / 2**20, pmc.PeakWorkingSetSize / 2**20


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class Embedder:
    def __init__(self, d, pool, dim, threads=4):
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=128)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(os.path.join(d, "model.onnx"), so, providers=["CPUExecutionProvider"])
        self.names = [i.name for i in self.sess.get_inputs()]
        self.pool, self.dim = pool, dim

    def batch(self, texts):
        enc = self.tok.encode_batch(texts)
        L = max(len(e.ids) for e in enc)
        ids = np.zeros((len(texts), L), dtype=np.int64)
        am = np.zeros_like(ids)
        for r, e in enumerate(enc):
            n = len(e.ids)
            ids[r, :n] = e.ids
            am[r, :n] = 1
        feed = {"input_ids": ids, "attention_mask": am}
        if "token_type_ids" in self.names:
            feed["token_type_ids"] = np.zeros_like(ids)
        hs = self.sess.run(None, feed)[0]
        if self.pool == "cls":
            p = hs[:, 0, :]
        else:
            m = am[:, :, None].astype(np.float32)
            p = (hs * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)

    def all(self, texts, B=32):
        enc = self.tok.encode_batch(texts)
        order = sorted(range(len(texts)), key=lambda i: len(enc[i].ids))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for s in range(0, len(order), B):
            idx = order[s:s + B]
            out[idx] = self.batch([texts[i] for i in idx])
        return out


def main():
    f5 = json.load(open(os.path.join(RES, "f5_corpus.json"), encoding="utf8"))
    items = f5["items"]
    texts = [it["text"] for it in items]
    rss0 = rss_mb()
    t0 = time.time()
    bge = Embedder(BGE, "cls", 768)
    load_bge = time.time() - t0
    rss_bge = rss_mb()
    t0 = time.time()
    emb = bge.all(texts)
    t_bge_all = time.time() - t0
    emb.tofile(os.path.join(RES, "f8_bge_embeddings.f32"))
    q = [i for i, it in enumerate(items) if it["kind"] == "q"]
    rng = random.Random(1)
    same, sib, diff, dist = [], [], [], []
    for _ in range(4000):
        i, j = rng.sample(q, 2)
        a, b = items[i], items[j]
        (same if a["concept"] == b["concept"] else sib if a["topic"] == b["topic"] else diff).append(float(emb[i] @ emb[j]))
    d = [i for i, it in enumerate(items) if it["kind"] == "dist"]
    for _ in range(2000):
        i, j = rng.sample(d, 2)
        dist.append(float(emb[i] @ emb[j]))
    mean = lambda x: round(float(np.mean(x)), 3)
    sanity = dict(same_concept=mean(same), sibling=mean(sib), diff_topic=mean(diff), distractor_distractor=mean(dist))
    print("BGE mean cos", sanity, "norm err", float(np.abs(np.linalg.norm(emb, axis=1) - 1).max()))
    mini = Embedder(MINI, "mean", 384)
    rss_both = rss_mb()
    ref = np.fromfile(os.path.join(RES, "f5_embeddings.f32"), dtype=np.float32).reshape(len(texts), 384)
    chk = mini.all(texts[:64])
    mini_maxdiff = float(np.abs(chk - ref[:64]).max())
    print("MiniLM harness reproduces f5 fixture, max abs diff", mini_maxdiff)
    sample = random.Random(7).sample(range(len(texts)), 150)
    lat = {}
    for name, m in (("bge-base", bge), ("minilm", mini)):
        for i in sample[:5]:
            m.batch([texts[i]])
        ts = []
        for i in sample:
            t0 = time.perf_counter()
            m.batch([texts[i]])
            ts.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        m.all([texts[i] for i in sample], B=32)
        tb = (time.perf_counter() - t0) * 1000 / len(sample)
        lat[name] = dict(single_ms_mean=round(float(np.mean(ts)), 2), single_ms_p50=round(float(np.median(ts)), 2),
                         single_ms_p95=round(float(np.percentile(ts, 95)), 2), batched32_ms_per_text=round(tb, 2))
    cache = {texts[i]: emb[i] for i in range(len(texts))}
    t0 = time.perf_counter()
    for _ in range(20):
        for i in sample:
            cache[texts[i]]
    lat["python_dict_cached_ms"] = round((time.perf_counter() - t0) * 1000 / (20 * len(sample)), 6)
    print(json.dumps(lat, indent=1))
    meta = dict(
        note="BGE-base-en-v1.5 vectors for the SAME 2424 SYNTHETIC f5 texts (results/f5_corpus.json order). CLS pooling, L2-normalised, 768-d, row-major f32 LE, no query instruction.",
        dim=768, n=len(texts), model="BAAI/bge-base-en-v1.5@a5beb1e3e68b9ab74eb54cfd186867f64f240e1a (onnx/model.onnx)",
        model_sha256=sha(os.path.join(BGE, "model.onnx")), tokenizer_sha256=sha(os.path.join(BGE, "tokenizer.json")),
        model_bytes=os.path.getsize(os.path.join(BGE, "model.onnx")), fixture_bytes=int(emb.nbytes),
        onnxruntime=ort.__version__, sanity_mean_cos=sanity, minilm_harness_maxdiff_vs_f5=mini_maxdiff,
        latency_ms=lat, load_seconds_bge=round(load_bge, 2), embed_all_seconds_bge=round(t_bge_all, 1),
        ram_mb=dict(baseline_ws=round(rss0[0]), after_bge_load_ws=round(rss_bge[0]), after_bge_load_peak=round(rss_bge[1]),
                    final_ws_both_models=round(rss_both[0]), final_peak=round(rss_both[1])),
    )
    json.dump(meta, open(os.path.join(RES, "f8_bge_meta.json"), "w"), indent=1)
    vec = {}
    for i, it in enumerate(items):
        if it["kind"] != "dist":
            vec[it["text"]] = [round(float(x), 6) for x in emb[i]]
    json.dump(dict(name="bge-base-en-v1.5-f5", dim=768, vectors=vec, meta=dict(n=len(vec), source="f8_embed.py")), open(os.path.join(RES, "f8_bge_table.json"), "w"))
    print("wrote", len(texts), "vectors; table json", len(vec))


if __name__ == "__main__":
    main()
