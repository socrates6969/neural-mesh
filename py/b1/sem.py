from lab import *
import lab
SEMT = [f"sem_{k}" for k in ("vocab","similarities","classification","verbal_analogies","sentence_completion")]
# fit unsupervised whitening on ALL battery texts (no labels used)
mu = D.E.mean(0); C = np.cov((D.E - mu).T); w, U = np.linalg.eigh(C)
def whiten(shrink, k=384):
    idx = np.argsort(-w)[:k]; W = U[:, idx] / np.sqrt(w[idx] + shrink)
    def tf(X):
        Y = (X - mu) @ W; return Y / (np.linalg.norm(Y, axis=-1, keepdims=True) + 1e-9)
    return tf
def semrun(name, E=None, make=None, target=SEMT):
    c = run_battery(D, ALLS, E=E, make=make, with_exec=False, real=True)
    from gate import compare
    v, rows, gain, loss = compare(BASE, c, target, tol=0.002, quiet=True)
    print(f"=== {name}: {v} | gains {[g for g in gain if g[0] in target] or 'none'} | losses {loss or 'none'}")
    for ax in target:
        print("   ", ax, " ".join(f"{r['w']}{r['sp'][0]} {r['mean']:+.3f}(p{r['ph']:.2g})" for r in rows if r["ax"] == ax))
    with open(f"{RES}/b1_rounds.jsonl", "a") as f: f.write(json.dumps(dict(name=name, verdict=v, gains=gain, losses=loss)) + "\n")
    return v
if __name__ == "__main__":
    for sh in (1e-4, 1e-3, 1e-2):
        tf = whiten(sh)
        semrun(f"R15 ZCA/PCA whitening shrink={sh} (semantic E only)", E=lambda t, tf=tf: tf(D.v(t)))
