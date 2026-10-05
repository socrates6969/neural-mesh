"""R90b: policy knobs for the per-store k procedure, from the cached per-query items of r90 (results/r90_items.json): frac sweep (.90/.95/.98) and a marginal-return rule
(grow k while the next doc adds >= s hit per 1,000 injected tokens; stop at the first step below s). n_cal=100 calibration queries, 200 random splits, disjoint test queries; test numbers vs fixed k=5."""
import json, os, numpy as np
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results"); items = json.load(open(f"{RES}/r90_items.json")); KMAX = 8
def curve(sub):
    n = len(sub); hit = np.array([np.mean([r is not None and r < k for r, _ in sub]) for k in range(1, KMAX + 1)]); tok = np.mean([c for _, c in sub], 0); return hit, tok
def pick_frac(hit, tok, frac):
    if hit[-1] == 0: return 5
    return next(k for k in range(1, KMAX + 1) if hit[k - 1] >= frac * hit[-1] - 1e-12)
def pick_marg(hit, tok, s):
    k = 1
    while k < KMAX and (hit[k] - hit[k - 1]) / max(tok[k] - tok[k - 1], 1e-9) * 1000 >= s: k += 1
    return k
RULES = {"frac .90": lambda h, t: pick_frac(h, t, .90), "frac .95": lambda h, t: pick_frac(h, t, .95), "frac .98": lambda h, t: pick_frac(h, t, .98), "marg 0.5/1k": lambda h, t: pick_marg(h, t, .5), "marg 1.0/1k": lambda h, t: pick_marg(h, t, 1.0), "marg 2.0/1k": lambda h, t: pick_marg(h, t, 2.0), "fixed k=3": lambda h, t: 3}
rng = np.random.default_rng(91); print(f"{'store':10s} {'rule':12s} chosen k | test hit (k=5: ...) | tokens vs k=5")
for name, it in items.items():
    n = len(it); ncal = 100
    for rn, rule in RULES.items():
        ks, hs, ts, h5s, t5s = [], [], [], [], []
        rng = np.random.default_rng(91)
        for _ in range(200):
            perm = rng.permutation(n); cal = [it[i] for i in perm[:ncal]]; te = [it[i] for i in perm[ncal:]]
            h, t = curve(cal); k = rule(h, t); ht, tt = curve(te); ks.append(k); hs.append(ht[k - 1]); ts.append(tt[k - 1]); h5s.append(ht[4]); t5s.append(tt[4])
        print(f"{name:10s} {rn:12s} k={np.mean(ks):.2f} | hit {np.mean(hs):.3f} (k5 {np.mean(h5s):.3f}, d {np.mean(hs)-np.mean(h5s):+.3f}) | tok {np.mean(ts):6.1f} vs {np.mean(t5s):6.1f} ({100*(np.mean(ts)/np.mean(t5s)-1):+.0f}%)")
