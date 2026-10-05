"""k4: embed extra texts with the on-disk MiniLM-L6 (py/f8_embed.py Embedder; no network, no downloads, deterministic).
Inputs  results/k4_embed_req.json (list of texts: multi-hop questions, clauses, re-embedded P1 docs), results/k4_vers_req.json (version corpus docs + queries).
Outputs results/k4_embed.f32 (n,384), results/k4_vers_emb.f32 (m,384).
Usage: C:/Users/mariu/.cognee-plugin/venv/Scripts/python.exe py/k4_embed.py [mh|vers|ext]"""
import importlib.util, json, os, sys, time
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RES = os.path.join(ROOT, "results")
spec = importlib.util.spec_from_file_location("f8e", os.path.join(ROOT, "py", "f8_embed.py"))
f8e = importlib.util.module_from_spec(spec); spec.loader.exec_module(f8e)


def run(req, out):
    texts = json.load(open(os.path.join(RES, req), encoding="utf8"))
    t0 = time.time()
    m = f8e.Embedder(f8e.MINI, "mean", 384)
    emb = m.all(texts)
    emb.astype(np.float32).tofile(os.path.join(RES, out))
    print(req, len(texts), "texts embedded in %.1fs" % (time.time() - t0), emb.shape)


if __name__ == "__main__":
    which = sys.argv[1:] or ["mh"]
    if "mh" in which:
        run("k4_embed_req.json", "k4_embed.f32")
    if "vers" in which:
        run("k4_vers_req.json", "k4_vers_emb.f32")
    if "ext" in which:
        run("k4_ext_req.json", "k4_ext_emb.f32")
