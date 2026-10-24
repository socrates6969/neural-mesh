"""Invariant Causal Prediction (Peters, Buehlmann, Meinshausen 2016, JRSS-B), linear-Gaussian version.

For every subset S of the predictors: regress the target on S with all environments pooled, then test whether the
residuals have the same distribution in every environment: equal means (one-way ANOVA F-test) and equal variances
(Levene, median-centred). The subset p-value is Bonferroni over those two tests. S is accepted when p_S > alpha;
the estimated causal parents are the intersection of all accepted sets, which contains only true parents with
probability >= 1 - alpha when the model holds. correction="subsets" additionally Bonferroni-corrects over the
number of subsets (more conservative, not needed for the guarantee). Runs inside the sandbox.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats


def _subset_pvalue(y, X, env, envs):
    A = np.column_stack([np.ones(len(y))] + ([X] if X.shape[1] else []))
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    groups = [r[env == e] for e in envs]
    p_mean = stats.f_oneway(*groups).pvalue
    p_var = stats.levene(*groups, center="median").pvalue
    return min(1.0, 2 * min(p_mean, p_var)), p_mean, p_var


def icp(y, X, env, names, alpha=0.05, correction="tests", max_predictors=12):
    y, X, env = np.asarray(y, float), np.asarray(X, float), np.asarray(env)
    d = X.shape[1]
    if d > max_predictors:
        raise ValueError("ICP checks all 2^d subsets; use at most %d predictors (got %d)" % (max_predictors, d))
    envs = sorted(set(env.tolist()))
    if len(envs) < 2:
        raise ValueError("need at least 2 environments in env_col")
    if min(int((env == e).sum()) for e in envs) < 5:
        raise ValueError("every environment needs at least 5 rows")
    subsets = [s for k in range(d + 1) for s in combinations(range(d), k)]
    level = alpha / len(subsets) if correction == "subsets" else alpha
    rows = []
    for s in subsets:
        p, pm, pv = _subset_pvalue(y, X[:, list(s)], env, envs)
        rows.append({"set": [names[i] for i in s], "p": float(p), "p_mean": float(pm), "p_var": float(pv),
                     "accepted": bool(p > level)})
    accepted = [set(r["set"]) for r in rows if r["accepted"]]
    if not accepted:
        parents, status = [], ("no subset is invariant: the linear-Gaussian model with these predictors is rejected "
                               "(hidden confounding, nonlinearity, or the target itself is intervened on)")
    else:
        parents = sorted(set.intersection(*accepted), key=names.index)
        status = "ok" if parents else ("accepted sets do not overlap: no predictor is identified as a parent at this "
                                       "level (too little environment variation, or weak effects)")
    # per-variable evidence: the largest p among sets that EXCLUDE the variable (small = it is needed)
    evidence = {n: max([r["p"] for r in rows if n not in r["set"]] or [1.0]) for n in names}
    top = sorted(rows, key=lambda r: -r["p"])[:8]
    return {"parents": parents, "status": status, "alpha": alpha, "level_per_subset": level,
            "correction": correction, "n_subsets": len(subsets), "n_accepted": len(accepted),
            "environments": [str(e) for e in envs], "variable_evidence": evidence, "top_sets": top}


def run(path, target, env_col, alpha=0.05, predictors=None, correction="tests"):
    import pandas as pd
    df = pd.read_csv(path)
    for c in [target, env_col] + list(predictors or []):
        if c not in df.columns:
            raise ValueError("column %r not in CSV (have: %s)" % (c, list(df.columns)[:40]))
    if predictors:
        preds = [c for c in predictors if c not in (target, env_col)]
    else:
        preds = [c for c in df.select_dtypes(include=[np.number]).columns if c not in (target, env_col)]
    df = df[[target, env_col] + preds].dropna()
    out = icp(df[target].to_numpy(), df[preds].to_numpy(), df[env_col].to_numpy(), preds, alpha, correction)
    out.update({"target": target, "env_col": env_col, "predictors": preds, "n_rows": int(len(df))})
    return out
