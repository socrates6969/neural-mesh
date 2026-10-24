"""Linear NOTEARS structure learning (Zheng, Aragam, Ravikumar, Xing 2018, "DAGs with NO TEARS", NeurIPS).

Written from scratch on numpy/scipy (no extra dependency). Solves
    min_W  1/(2n) ||X - X W||_F^2 + lambda1 ||W||_1   subject to  h(W) = tr(exp(W o W)) - d = 0
with the augmented Lagrangian (rho, alpha) and L-BFGS-B on W = W+ - W- (W+, W- >= 0, which makes the l1 term
smooth), then prunes |w| < w_threshold. Runs inside the sandbox; also importable on a host with scipy.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt


def h_acyclic(W: np.ndarray) -> tuple[float, np.ndarray]:
    """h(W) = tr(e^{W o W}) - d (zero iff W is a DAG) and its gradient."""
    E = sla.expm(W * W)
    return float(np.trace(E) - W.shape[0]), E.T * W * 2


def notears_linear(X, lambda1=0.1, w_threshold=0.3, max_iter=100, h_tol=1e-8, rho_max=1e16):
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)

    def unpack(w):
        return (w[: d * d] - w[d * d:]).reshape(d, d)

    def objective(w, rho, alpha):
        W = unpack(w)
        R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum()
        g_loss = -1.0 / n * X.T @ R
        h, g_h = h_acyclic(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        g = g_loss + (rho * h + alpha) * g_h
        return obj, np.concatenate([g + lambda1, -g + lambda1], axis=None)

    # no self-loops: bound the diagonal of both halves to 0
    bounds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
    w, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf
    for _ in range(max_iter):
        while rho < rho_max:
            sol = sopt.minimize(objective, w, args=(rho, alpha), method="L-BFGS-B", jac=True, bounds=bounds)
            h_new, _ = h_acyclic(unpack(sol.x))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w, h = sol.x, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W = unpack(w)
    W[np.abs(W) < w_threshold] = 0
    return W, h_acyclic(W)[0], h


def discover(path, lambda1=0.1, w_threshold=0.3, max_cols=20, max_rows=5000, standardize=False):
    import pandas as pd
    df = pd.read_csv(path).select_dtypes(include=[np.number]).dropna()
    cols = list(df.columns)[:max_cols]
    if len(cols) < 2 or len(df) < 20:
        raise ValueError("need >= 2 numeric columns and >= 20 complete rows")
    df = df[cols]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)
    x = df.to_numpy(dtype=float)
    keep = x.std(axis=0) > 0
    cols, x = [c for c, k in zip(cols, keep) if k], x[:, keep]
    if standardize:
        x = (x - x.mean(0)) / x.std(0)
    W, h_final, h_opt = notears_linear(x, lambda1=float(lambda1), w_threshold=float(w_threshold))
    if not h_final < 1e-8:
        raise ValueError("NOTEARS result is not acyclic after thresholding (h=%.3g); raise w_threshold" % h_final)
    edges = [{"from": cols[i], "to": cols[j], "weight": float(W[i, j])}
             for i in range(len(cols)) for j in range(len(cols)) if W[i, j] != 0]
    edges.sort(key=lambda e: -abs(e["weight"]))
    return {"algorithm": "NOTEARS linear (Zheng et al. 2018), l2 loss, lambda1=%g, w_threshold=%g%s"
                         % (lambda1, w_threshold, ", standardized columns" if standardize else ""),
            "n_rows": int(len(df)), "columns": cols, "edges": edges, "h": h_final, "h_before_threshold": h_opt,
            "label": "EXPLORATORY: assumes a linear SEM with no hidden confounders; NOTEARS is not scale-invariant "
                     "(it uses the raw columns by default; rescaling a column can flip edges) "
                     "and the threshold is arbitrary; edges are hypotheses to check with domain knowledge, not "
                     "findings."}
