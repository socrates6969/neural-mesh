"""vector-lab: neural-mesh store arms vs ruvector 0.3.3 (VectorDb HNSW, RVF) on real MiniLM vectors (results/b1_emb.f32, n=25,866).
Per seed: random corpus of N, disjoint NQ queries; truth = exact f32 cosine top-10. Output JSON per arm per seed.
Usage: python run_store_cmp.py <scratch_dir> <vec2_dir> <out.json> [N] [NQ] [seeds]"""
import json, os, subprocess, sys, numpy as np
scr, vec2, outp = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
NQ = int(sys.argv[5]) if len(sys.argv) > 5 else 200
SEEDS = int(sys.argv[6]) if len(sys.argv) > 6 else 30
E = np.fromfile(r"C:/Users/mariu/neural-mesh/results/b1_emb.f32", dtype=np.float32).reshape(-1, 384)
exe = r"C:/Users/mariu/neural-mesh/target/release/v1_vec_cmp.exe"
os.makedirs(scr, exist_ok=True); res = {}
for seed in range(1, SEEDS + 1):
    p = np.random.default_rng(seed).permutation(len(E))
    C, Q = E[p[:N]], E[p[N:N + NQ]]
    C.tofile(f"{scr}/c.f32"); Q.tofile(f"{scr}/q.f32")
    truth = np.argsort(-(Q @ C.T), axis=1)[:, :10]
    subprocess.run([exe, f"{scr}/c.f32", f"{scr}/q.f32", f"{scr}/nm.json"], check=True)
    subprocess.run(["node", "ruv_arm.js", f"{scr}/c.f32", f"{scr}/q.f32", f"{scr}/ruv.json", f"{scr}/w"], check=True, cwd=vec2)
    arms = {**json.load(open(f"{scr}/nm.json")), **json.load(open(f"{scr}/ruv.json"))}
    res[seed] = {}
    for k, v in arms.items():
        rec = np.mean([len(set(t) & set(truth[i])) / 10 for i, t in enumerate(v["top"])])
        res[seed][k] = dict(recall10=float(rec), p50_us=v["p50_us"], p95_us=v["p95_us"], bytes_per_vec=v["bytes"] / N, build_s=v["build_s"])
    print(seed, {k: round(x["recall10"], 4) for k, x in res[seed].items()}, flush=True)
    json.dump(res, open(outp, "w"))
