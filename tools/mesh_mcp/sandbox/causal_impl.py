"""Causal effect estimation (DoWhy) and exploratory graph discovery (causal-learn PC). Runs inside the sandbox only.

The CSV is mounted read-only at /data/input.csv. Results are plain dicts; the server renders them.
"""
from __future__ import annotations

import logging
import math
import re
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

METHODS = {
    "backdoor": "backdoor.linear_regression",
    "linear_regression": "backdoor.linear_regression",
    "backdoor.linear_regression": "backdoor.linear_regression",
    "propensity_score_weighting": "backdoor.propensity_score_weighting",
    "backdoor.propensity_score_weighting": "backdoor.propensity_score_weighting",
    "propensity_score_matching": "backdoor.propensity_score_matching",
    "backdoor.propensity_score_matching": "backdoor.propensity_score_matching",
    "propensity_score_stratification": "backdoor.propensity_score_stratification",
    "backdoor.propensity_score_stratification": "backdoor.propensity_score_stratification",
}
MAX_ROWS = 200000


def _load(path, cols):
    df = pd.read_csv(path)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError("columns not in CSV: %s (have: %s)" % (missing, list(df.columns)[:40]))
    df = df[cols].dropna()
    if len(df) < 20:
        raise ValueError("need at least 20 complete rows, have %d" % len(df))
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=0)
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            raise ValueError("column %r is not numeric; encode it first" % c)
    return df


def _f(x):
    try:
        x = float(np.asarray(x).ravel()[0]) if np.ndim(x) else float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError, IndexError):
        return None


def _backdoor_factory(treatment, outcome, common_causes):
    from dowhy import CausalModel
    cols = [treatment, outcome] + list(common_causes)

    def make(frame):
        # DoWhy writes propensity_score/ips_weight columns into the frame it is given and reuses them later,
        # so every model gets its own copy of only the analysis columns.
        return CausalModel(data=frame[cols].copy(), treatment=treatment, outcome=outcome,
                           common_causes=common_causes or None)
    return make


def _fit(make, frame, method):
    m = make(frame)
    return _f(m.estimate_effect(m.identify_effect(proceed_when_unidentifiable=True), method_name=method,
                                target_units="ate").value)


def _bootstrap_ci(df, make, method, n_boot, seed):
    rng = np.random.default_rng(seed)
    vals = [v for v in (_fit(make, df.iloc[rng.integers(0, len(df), len(df))].reset_index(drop=True), method)
                        for _ in range(n_boot)) if v is not None]
    if len(vals) < max(10, n_boot // 2):
        return None
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _placebo(df, treatment, make, method, value, n_sims, seed):
    """Permutation placebo: shuffle the treatment column, re-run the same estimator, repeat.

    Done by hand: DoWhy's placebo refuter reuses the propensity_score column fitted on the real treatment.
    p_value = share of placebo estimates at least as large in magnitude as the real estimate.
    """
    rng = np.random.default_rng(seed + 1)
    vals = []
    try:
        for _ in range(n_sims):
            s = df.copy()
            s[treatment] = rng.permutation(s[treatment].to_numpy())
            v = _fit(make, s, method)
            if v is not None:
                vals.append(v)
    except Exception as e:  # noqa: BLE001
        return {"refuter": "placebo_permutation", "error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    if not vals:
        return {"refuter": "placebo_permutation", "error": "no placebo estimates"}
    mean = float(np.mean(vals))
    p = (1 + sum(abs(v) >= abs(value or 0) for v in vals)) / (1 + len(vals))
    return {"refuter": "placebo_permutation", "new_effect": mean, "p_value": p,
            "passed": abs(mean) < 0.2 * max(abs(value or 0), 1e-9) and p <= 0.1,
            "expectation": "mean effect of a permuted (placebo) treatment should be near 0 (%d permutations)"
                           % len(vals)}


def _random_common_cause(model, ident, est, value, n_sims, seed):
    try:
        r = model.refute_estimate(ident, est, method_name="random_common_cause", num_simulations=int(n_sims),
                                  random_seed=seed)
        new = _f(r.new_effect)
        res = getattr(r, "refutation_result", None) or {}
        return {"refuter": "random_common_cause", "new_effect": new,
                "p_value": _f(res.get("p_value")) if res else None,
                "passed": new is not None and abs(new - (value or 0)) < 0.1 * max(abs(value or 0), 1e-9),
                "expectation": "adding a random common cause should leave the estimate ~unchanged"}
    except Exception as e:  # noqa: BLE001
        return {"refuter": "random_common_cause", "error": "%s: %s" % (type(e).__name__, str(e)[:200])}


_NODE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def frontdoor_graph(treatment, outcome, mediator, common_causes):
    """Default frontdoor DAG: hidden U confounds treatment and outcome; treatment -> mediator -> outcome is the only
    causal path; observed common causes point into treatment, mediator and outcome."""
    edges = ["U_hidden->%s" % treatment, "U_hidden->%s" % outcome, "%s->%s" % (treatment, mediator),
             "%s->%s" % (mediator, outcome)]
    for w in common_causes:
        edges += ["%s->%s" % (w, treatment), "%s->%s" % (w, mediator), "%s->%s" % (w, outcome)]
    return 'digraph {U_hidden[label="Unobserved"]; %s;}' % "; ".join(edges)


def estimate_frontdoor(path, treatment, outcome, mediator, common_causes, graph=None, n_sims=20, seed=0):
    """Front-door ATE with frontdoor.two_stage_regression. Returns {"identifiable": False, "reason": ...} instead of
    falling back to another estimand when the front-door criterion does not hold for the stated graph."""
    from dowhy import CausalModel
    mname = "frontdoor.two_stage_regression"

    def no(reason):
        return {"method": mname, "identifiable": False, "reason": reason, "treatment": treatment,
                "outcome": outcome, "mediator": mediator}
    if not mediator:
        return no("method=frontdoor needs mediator=<column on the causal path treatment -> mediator -> outcome>")
    if mediator in (treatment, outcome):
        return no("the mediator must differ from treatment and outcome")
    common_causes = [c for c in common_causes if c not in (mediator, treatment, outcome)]
    cols = [treatment, outcome, mediator] + common_causes
    for c in cols:
        if not _NODE.match(c):
            return no("column %r is not a plain identifier; rename it to use it in a causal graph" % c)
    df = _load(path, cols)
    graph = graph or frontdoor_graph(treatment, outcome, mediator, common_causes)

    def make(frame):
        return CausalModel(data=frame[cols].copy(), treatment=treatment, outcome=outcome, graph=graph)
    np.random.seed(seed)
    model = make(df)
    ident = model.identify_effect(proceed_when_unidentifiable=True)
    fd = list(ident.get_frontdoor_variables() or [])
    if not fd:
        return no("the front-door criterion does not hold for this graph: no set of observed variables intercepts "
                  "every directed path %s -> %s with no unblocked back-door path into it (a direct %s -> %s edge, "
                  "or a confounder of %s and the mediator, breaks it). Graph: %s"
                  % (treatment, outcome, treatment, outcome, treatment, graph))
    if mediator not in fd:
        return no("DoWhy found the front-door set %s, which does not contain mediator %r; pass that set's "
                  "variable as mediator" % (fd, mediator))
    est = model.estimate_effect(ident, method_name=mname, target_units="ate")
    value = _f(est.value)
    ci = _bootstrap_ci(df, make, mname, 40, seed)
    refutations = [_placebo(df, treatment, make, mname, value, int(n_sims), seed),
                   _random_common_cause(model, ident, est, value, n_sims, seed)]
    assumptions = [
        "%s intercepts every directed path from %s to %s (no direct effect of %s on %s)."
        % (mediator, treatment, outcome, treatment, outcome),
        "No unblocked back-door path from %s to %s (nothing confounds treatment and mediator)." % (treatment, mediator),
        "Every back-door path from %s to %s is blocked by %s (confounding of treatment and outcome may be "
        "unmeasured)." % (mediator, outcome, treatment),
        "Both stages are linear and additive (two-stage regression: %s on %s, %s on %s and %s)."
        % (mediator, treatment, outcome, mediator, treatment),
        "SUTVA; rows are i.i.d. samples (rows with missing values were dropped).",
    ]
    return {"method": mname, "identifiable": True, "estimand": "ATE (front-door)", "n_rows": int(len(df)),
            "treatment": treatment, "outcome": outcome, "mediator": mediator, "frontdoor_set": fd,
            "common_causes": common_causes, "graph": graph,
            "binary_treatment": bool(set(pd.unique(df[treatment])) <= {0, 1, True, False}),
            "estimate": value, "ci95": ci,
            "ci_kind": "bootstrap percentile 95% (40 resamples)" if ci else None,
            "refutations": refutations, "assumptions": assumptions}

def estimate(path, treatment, outcome, common_causes, method="backdoor", n_sims=20, seed=0):
    key = str(method).strip().lower()
    if key not in METHODS:
        raise ValueError("method must be one of %s" % sorted(set(METHODS)))
    mname = METHODS[key]
    common_causes = [c for c in (common_causes or []) if c not in (treatment, outcome)]
    df = _load(path, [treatment, outcome] + common_causes)
    binary = set(pd.unique(df[treatment])) <= {0, 1, True, False}
    if "propensity" in mname:
        if not binary:
            raise ValueError("propensity-score methods need a binary 0/1 treatment column")
        df[treatment] = df[treatment].astype(bool)
    np.random.seed(seed)
    make = _backdoor_factory(treatment, outcome, common_causes)
    model = make(df)
    ident = model.identify_effect(proceed_when_unidentifiable=True)
    est = model.estimate_effect(ident, method_name=mname, target_units="ate",
                                confidence_intervals=(mname == "backdoor.linear_regression"))
    value = _f(est.value)
    ci, ci_kind = None, None
    if mname == "backdoor.linear_regression":
        try:
            lo, hi = np.asarray(est.get_confidence_intervals()).ravel()[:2]
            ci, ci_kind = [float(lo), float(hi)], "OLS 95% interval"
        except Exception:  # noqa: BLE001 - fall back to bootstrap below
            ci = None
    if ci is None:
        ci = _bootstrap_ci(df, make, mname, 40, seed)
        ci_kind = "bootstrap percentile 95% (40 resamples)" if ci else None
    refutations = [_placebo(df, treatment, make, mname, value, int(n_sims), seed),
                   _random_common_cause(model, ident, est, value, n_sims, seed)]
    assumptions = [
        "No unmeasured confounding: %s blocks every back-door path from %s to %s."
        % (common_causes or "the empty set", treatment, outcome),
        "Common causes are pre-treatment (not caused by %s) and are not colliders or mediators." % treatment,
        "Positivity/overlap: every covariate stratum has a non-zero chance of each treatment level.",
        "SUTVA: one unit's treatment does not affect another unit's outcome; one version of treatment.",
        "Rows are i.i.d. samples from the target population (rows with missing values were dropped).",
    ]
    if mname == "backdoor.linear_regression":
        assumptions.append("Outcome is linear and additive in treatment and common causes (no interactions).")
    else:
        assumptions.append("Propensity model (logistic regression on common causes) is correctly specified.")
    return {"method": mname, "estimand": "ATE", "n_rows": int(len(df)), "treatment": treatment,
            "outcome": outcome, "common_causes": common_causes, "binary_treatment": bool(binary),
            "estimate": value, "ci95": ci, "ci_kind": ci_kind, "refutations": refutations,
            "assumptions": assumptions}


def discover(path, alpha=0.05, max_cols=20, max_rows=5000):
    from causallearn.search.ConstraintBased.PC import pc
    df = pd.read_csv(path).select_dtypes(include=[np.number]).dropna()
    cols = list(df.columns)[:max_cols]
    if len(cols) < 2 or len(df) < 20:
        raise ValueError("need >= 2 numeric columns and >= 20 complete rows")
    df = df[cols]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=0)
    x = df.to_numpy(dtype=float)
    keep = x.std(axis=0) > 0
    cols = [c for c, k in zip(cols, keep) if k]
    x = x[:, keep]
    cg = pc(x, alpha=alpha, indep_test="fisherz", show_progress=False, node_names=cols)
    g = cg.G.graph  # g[j,i]==1 and g[i,j]==-1 means i --> j
    edges = []
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = g[i, j], g[j, i]
            if a == 0 and b == 0:
                continue
            if a == -1 and b == 1:
                edges.append("%s --> %s" % (cols[i], cols[j]))
            elif a == 1 and b == -1:
                edges.append("%s --> %s" % (cols[j], cols[i]))
            elif a == -1 and b == -1:
                edges.append("%s --- %s (direction undetermined)" % (cols[i], cols[j]))
            else:
                edges.append("%s <-> %s" % (cols[i], cols[j]))
    return {"algorithm": "PC (causal-learn), Fisher-z test, alpha=%g" % alpha, "n_rows": int(len(df)),
            "columns": cols, "edges": edges,
            "label": "EXPLORATORY: assumes causal sufficiency, faithfulness, linear-Gaussian relations; "
                     "edges are hypotheses to check with domain knowledge, not findings."}
