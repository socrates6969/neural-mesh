"""Efficiency rounds: vector compression gated by the battery. quality axes must show no Holm loss and TOST-equivalence within +-.01;
the gain is bytes/vector (deterministic)."""
from lab import *
from gate import compare
ALLQ = ["domain_r1", "domain_mrr", "domain_route", "general_info", "comprehension", "real_mrr", "real_h1"] + [f"sem_{k}" for k in ("vocab","similarities","classification","verbal_analogies","sentence_completion")]
def q8(X):
    s = np.abs(X).max(axis=-1, keepdims=True) / 127.0 + 1e-12; return np.round(X / s).clip(-127, 127) * s
def f16(X): return X.astype(np.float16).astype(np.float32)
def sign(X): return np.sign(X) / np.sqrt(X.shape[-1])
def qk(X, bits):
    lv = 2 ** (bits - 1) - 1; s = np.abs(X).max(axis=-1, keepdims=True) / lv + 1e-12; return np.round(X / s).clip(-lv, lv) * s
class QR:
    def __init__(s, docs, vecs, arm, alpha, tf, qtf=None):
        s.V = tf(vecs); s.tf, s.qtf, s.bm, s.a = tf, qtf or (lambda x: x), BM25(docs), alpha
    def scores(s, qt, qv): return s.a * z(s.V @ s.qtf(qv)) + (1 - s.a) * z(s.bm.score(qt))
def mkq(tf, qtf=None): return lambda d, v, a, al: QR(d, v, a, al, tf, qtf)
def pca_tf(k):
    def make(d, v, a, al):
        mu = v.mean(0); U, S, Vt = np.linalg.svd(v - mu, full_matrices=False); P = Vt[:k].T
        nrm = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
        class R:
            def __init__(s): s.V = nrm((v - mu) @ P); s.bm = BM25(d)
            def scores(s, qt, qv): return al * z(s.V @ nrm((qv - mu) @ P)) + (1 - al) * z(s.bm.score(qt))
        return R()
    return make
def pca_q(k, tfq):
    base = pca_tf(k)
    def make(d, v, a, al):
        mu = v.mean(0); U, S, Vt = np.linalg.svd(v - mu, full_matrices=False); P = Vt[:k].T
        nrm = lambda X: X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)
        class R:
            def __init__(s): s.V = tfq(nrm((v - mu) @ P)); s.bm = BM25(d)
            def scores(s, qt, qv): return al * z(s.V @ tfq(nrm((qv - mu) @ P))) + (1 - al) * z(s.bm.score(qt))
        return R()
    return make
def eff_round(name, bytes_per_vec, make=None, E=None):
    t0 = time.time(); c = run_battery(D, ALLS, make=make, E=E, with_exec=False, real=True)
    v, rows, gain, loss = compare(BASE, c, ["__none__"], tol=0.002, quiet=True, axes=ALLQ)
    worst = {}
    for r in rows:
        k = r["ax"]; worst[k] = min(worst.get(k, 9), r["mean"])
    tostmax = max(r["tost"] for r in rows if r["w"] in "ABR")
    print(f"=== {name} ({time.time()-t0:.0f}s) bytes/vec={bytes_per_vec} (f32=1536)  Holm losses: {loss or 'none'}")
    print("   worst per-axis diff:", {k: round(x, 4) for k, x in worst.items() if x < -0.003} or "none worse than -.003",
          "| max TOST p (+-.01) over all tests: %.3g" % tostmax)
    with open(f"{RES}/b1_rounds.jsonl", "a") as f: f.write(json.dumps(dict(name=name, bytes=bytes_per_vec, losses=loss, tostmax=tostmax, worst=worst)) + "\n")
    return rows, loss
if __name__ == "__main__":
    eff_round("R9 (tol .002) int8 per-vector", 388, make=mkq(q8, q8), E=lambda t: q8(D.v(t)))
    eff_round("R14a PCA192 + int8", 196, make=pca_q(192, q8))
    eff_round("R14b PCA256 + int8", 260, make=pca_q(256, q8))
    eff_round("R14c PCA128 + int8", 132, make=pca_q(128, q8))
