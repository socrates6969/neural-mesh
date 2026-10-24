"""Statistical tests with effect sizes (SciPy + statsmodels). Runs inside the sandbox; also importable on a host with SciPy.

compute(kind, data) -> dict. Every result reports an effect size and, where defined, a confidence interval.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

KINDS = ("ttest", "paired_t", "welch", "mannwhitney", "chi2", "mcnemar", "prop_ci", "cohens_d", "multitest")


def _arr(data, key):
    if key not in data:
        raise ValueError("data needs '%s'" % key)
    a = np.asarray(data[key], dtype=float)
    a = a[~np.isnan(a)]
    if a.size < 2:
        raise ValueError("'%s' needs at least 2 numeric values" % key)
    return a


def _d_pooled(a, b):
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else float("nan")
    g = d * (1 - 3 / (4 * (na + nb) - 9))  # Hedges small-sample correction
    se = math.sqrt((na + nb) / (na * nb) + d * d / (2 * (na + nb)))
    return d, g, (d - 1.96 * se, d + 1.96 * se)


def _magnitude_d(d):
    d = abs(d)
    return "negligible" if d < 0.2 else "small" if d < 0.5 else "medium" if d < 0.8 else "large"


def _ttest(data, alpha):
    a = _arr(data, "a")
    if "b" not in data:  # one-sample
        mu = float(data.get("mu", 0.0))
        r = stats.ttest_1samp(a, mu)
        ci = r.confidence_interval(1 - alpha)
        d = (a.mean() - mu) / a.std(ddof=1)
        return {"test": "one-sample t-test", "n": len(a), "mean": a.mean(), "mu": mu, "t": r.statistic,
                "df": r.df, "p": r.pvalue, "mean_ci": [ci.low, ci.high],
                "effect": {"cohens_d": d, "magnitude": _magnitude_d(d)}}
    b = _arr(data, "b")
    r = stats.ttest_ind(a, b, equal_var=True)
    ci = r.confidence_interval(1 - alpha)
    d, g, dci = _d_pooled(a, b)
    return {"test": "Student two-sample t-test (equal variances)", "n_a": len(a), "n_b": len(b),
            "mean_a": a.mean(), "mean_b": b.mean(), "t": r.statistic, "df": r.df, "p": r.pvalue,
            "mean_diff": a.mean() - b.mean(), "mean_diff_ci": [ci.low, ci.high],
            "effect": {"cohens_d": d, "hedges_g": g, "cohens_d_ci95": list(dci), "magnitude": _magnitude_d(d)}}


def _welch(data, alpha):
    a, b = _arr(data, "a"), _arr(data, "b")
    r = stats.ttest_ind(a, b, equal_var=False)
    ci = r.confidence_interval(1 - alpha)
    d, g, dci = _d_pooled(a, b)
    return {"test": "Welch two-sample t-test (unequal variances)", "n_a": len(a), "n_b": len(b),
            "mean_a": a.mean(), "mean_b": b.mean(), "t": r.statistic, "df": r.df, "p": r.pvalue,
            "mean_diff": a.mean() - b.mean(), "mean_diff_ci": [ci.low, ci.high],
            "effect": {"cohens_d": d, "hedges_g": g, "cohens_d_ci95": list(dci), "magnitude": _magnitude_d(d)}}


def _paired(data, alpha):
    a, b = _arr(data, "a"), _arr(data, "b")
    if len(a) != len(b):
        raise ValueError("paired_t needs a and b of equal length")
    r = stats.ttest_rel(a, b)
    ci = r.confidence_interval(1 - alpha)
    diff = a - b
    dz = diff.mean() / diff.std(ddof=1)
    return {"test": "paired t-test", "n": len(a), "mean_diff": diff.mean(), "t": r.statistic, "df": r.df,
            "p": r.pvalue, "mean_diff_ci": [ci.low, ci.high],
            "effect": {"cohens_dz": dz, "magnitude": _magnitude_d(dz)}}


def _mannwhitney(data, alpha):
    a, b = _arr(data, "a"), _arr(data, "b")
    r = stats.mannwhitneyu(a, b, alternative=data.get("alternative", "two-sided"))
    cl = r.statistic / (len(a) * len(b))  # P(A > B) + 0.5 P(A = B)
    return {"test": "Mann-Whitney U", "n_a": len(a), "n_b": len(b), "U": r.statistic, "p": r.pvalue,
            "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            "effect": {"rank_biserial_r": 2 * cl - 1, "common_language_P(a>b)": cl}}


def _chi2(data, alpha):
    t = np.asarray(data.get("table"), dtype=float)
    if t.ndim != 2 or min(t.shape) < 2:
        raise ValueError("chi2 needs 'table' as a 2-D contingency table (at least 2x2)")
    chi, p, dof, exp = stats.chi2_contingency(t, correction=bool(data.get("correction", True)))
    n = t.sum()
    v = math.sqrt(chi / (n * (min(t.shape) - 1)))
    out = {"test": "chi-square test of independence", "n": n, "chi2": chi, "df": dof, "p": p,
           "min_expected": float(exp.min()), "effect": {"cramers_v": v}}
    if exp.min() < 5:
        out["warning"] = "expected count < 5 in some cell; consider Fisher exact test"
    return out


def _mcnemar(data, alpha):
    from statsmodels.stats.contingency_tables import mcnemar
    t = np.asarray(data.get("table"), dtype=float)
    if t.shape != (2, 2):
        raise ValueError("mcnemar needs 'table' as [[both_pos, a_only],[b_only, both_neg]] (2x2 paired counts)")
    b, c = t[0, 1], t[1, 0]
    exact = bool(data.get("exact", (b + c) < 25))
    r = mcnemar(t, exact=exact, correction=True)
    n = t.sum()
    diff = (b - c) / n  # P(A+) - P(B+)
    se = math.sqrt(max((b + c) / n - diff * diff, 0) / n)
    z = stats.norm.ppf(1 - alpha / 2)
    return {"test": "McNemar (%s)" % ("exact binomial" if exact else "chi-square, continuity-corrected"),
            "n_pairs": n, "discordant": [b, c], "statistic": r.statistic, "p": r.pvalue,
            "effect": {"odds_ratio_b_over_c": (b / c) if c else float("inf"), "prop_diff": diff,
                       "prop_diff_ci": [diff - z * se, diff + z * se]}}


def _prop_ci(data, alpha):
    from statsmodels.stats.proportion import proportion_confint
    k, n = int(data.get("successes", -1)), int(data.get("n", 0))
    if n <= 0 or not 0 <= k <= n:
        raise ValueError("prop_ci needs integer 'successes' and 'n' with 0 <= successes <= n")
    method = data.get("method", "wilson")
    lo, hi = proportion_confint(k, n, alpha=alpha, method=method)
    return {"test": "proportion confidence interval (%s)" % method, "successes": k, "n": n,
            "p_hat": k / n, "ci": [lo, hi], "confidence": 1 - alpha}


def _cohens_d(data, alpha):
    a, b = _arr(data, "a"), _arr(data, "b")
    d, g, dci = _d_pooled(a, b)
    return {"test": "Cohen d (pooled SD)", "n_a": len(a), "n_b": len(b), "mean_diff": a.mean() - b.mean(),
            "effect": {"cohens_d": d, "hedges_g": g, "cohens_d_ci95": list(dci), "magnitude": _magnitude_d(d)}}


def _multitest(data, alpha):
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(data.get("pvalues"), dtype=float)
    if p.ndim != 1 or p.size == 0 or ((p < 0) | (p > 1)).any():
        raise ValueError("multitest needs 'pvalues' as a list of numbers in [0, 1]")
    method = {"holm": "holm", "bh": "fdr_bh", "fdr_bh": "fdr_bh", "bonferroni": "bonferroni"}.get(
        str(data.get("method", "holm")).lower())
    if method is None:
        raise ValueError("method must be holm, bh (fdr_bh) or bonferroni")
    rej, adj, _, _ = multipletests(p, alpha=alpha, method=method)
    return {"test": "multiple-comparison correction (%s)" % method, "m": int(p.size), "alpha": alpha,
            "p_raw": p.tolist(), "p_adjusted": adj.tolist(), "reject": [bool(x) for x in rej],
            "n_rejected": int(rej.sum())}


_FN = {"ttest": _ttest, "paired_t": _paired, "welch": _welch, "mannwhitney": _mannwhitney, "chi2": _chi2,
       "mcnemar": _mcnemar, "prop_ci": _prop_ci, "cohens_d": _cohens_d, "multitest": _multitest}


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def compute(kind: str, data: dict) -> dict:
    kind = str(kind).lower()
    if kind not in _FN:
        raise ValueError("unknown kind %r; use one of %s" % (kind, ", ".join(KINDS)))
    if not isinstance(data, dict):
        raise ValueError("data_json must be a JSON object")
    alpha = float(data.get("alpha", 0.05))
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return _plain(_FN[kind](data, alpha))
