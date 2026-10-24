"""Sandboxed tools: math_solve, stats_test, causal_analysis, discover_graph. All model-supplied code and data
processing runs in the network-less grader container (image MESH_TOOLS_IMAGE, default meshtrain-grader:2)."""
from __future__ import annotations

import json
from pathlib import Path

from tools.mesh_mcp import phack
from tools.mesh_mcp.common import as_json, cap, run_sandbox

MAX_CSV_BYTES = 50 * 1024 * 1024

_RUNNER = """import json, sys, traceback
a = json.load(open("args.json"))
try:
%s
    print("@@RESULT@@" + json.dumps(out, default=str))
except Exception as e:
    print("@@ERROR@@" + json.dumps({"error": "%%s: %%s" %% (type(e).__name__, e)}))
"""


def _call(body: str, args: dict, mounts=None, timeout_s=180, memory="1g") -> dict:
    out, err, code = run_sandbox(_RUNNER % body, args, mounts=mounts, timeout_s=timeout_s, memory=memory)
    for line in out.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):])
        if line.startswith("@@ERROR@@"):
            raise ValueError(json.loads(line[len("@@ERROR@@"):])["error"])
    raise RuntimeError("sandbox failed (exit %d): %s" % (code, (err or out)[-1500:]))


def _fmt(x):
    if isinstance(x, float):
        return "%.6g" % x
    if isinstance(x, list):
        return "[" + ", ".join(_fmt(v) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join("%s=%s" % (k, _fmt(v)) for k, v in x.items()) + "}"
    return str(x)


def math_solve(code: str, timeout_s: int = 120) -> str:
    if not str(code or "").strip():
        raise ValueError("code is required")
    out, err, rc = run_sandbox(str(code), {}, timeout_s=timeout_s)
    text = "exit code %d\n--- stdout ---\n%s" % (rc, out[-4500:])
    if err.strip():
        text += "\n--- stderr ---\n" + err[-1200:]
    return cap(text)


def stats_test(kind: str, data_json, label: str | None = None, look: int | None = None) -> str:
    data = as_json(data_json, "data_json")
    r = _call("    import stats_impl\n    out = stats_impl.compute(a['kind'], a['data'])", {"kind": kind, "data": data},
              timeout_s=60)
    phack.record_test(str(kind).lower(), data, r, label, look)  # session log read by phack_guard
    lines = [r.pop("test")]
    eff = r.pop("effect", None)
    for k, v in r.items():
        lines.append("%s: %s" % (k, _fmt(v)))
    if eff:
        lines.append("EFFECT SIZE: " + _fmt(eff))
    lines.append("Report the effect size and interval with the p-value; p alone does not show magnitude.")
    lines.append("Logged for phack_guard%s." % (" as outcome %r" % label if label else " (pass label=<outcome> to "
                                                                       "track repeated looks)"))
    return cap("\n".join(lines))


def _csv(path: str) -> str:
    p = Path(str(path or "")).expanduser()
    if not p.is_file():
        raise ValueError("csv_path not found: %s" % path)
    if p.suffix.lower() not in (".csv", ".txt"):
        raise ValueError("csv_path must be a .csv file")
    if p.stat().st_size > MAX_CSV_BYTES:
        raise ValueError("CSV larger than %d MB" % (MAX_CSV_BYTES // 2**20))
    return str(p.resolve())


def _cols(value):
    if not value:
        return []
    if isinstance(value, str):
        value = json.loads(value) if value.strip().startswith("[") else value.split(",")
    return [str(c).strip() for c in value if str(c).strip()]


def causal_analysis(csv_path: str, treatment: str, outcome: str, common_causes=None, method: str = "backdoor",
                    num_simulations: int = 20, mediator: str | None = None, graph: str | None = None) -> str:
    if not treatment or not outcome:
        raise ValueError("treatment and outcome are required")
    args = {"t": treatment, "y": outcome, "cc": _cols(common_causes), "m": method, "med": mediator,
            "g": graph, "n": max(5, min(int(num_simulations or 20), 100))}
    frontdoor = str(method or "").strip().lower() in ("frontdoor", "frontdoor.two_stage_regression")
    body = ("    import causal_impl\n    out = causal_impl.estimate_frontdoor('/data/input.csv', a['t'], a['y'], "
            "a['med'], a['cc'], a['g'], a['n'])" if frontdoor else
            "    import causal_impl\n    out = causal_impl.estimate('/data/input.csv', a['t'], a['y'], a['cc'], "
            "a['m'], a['n'])")
    r = _call(body, args, mounts=[(_csv(csv_path), "/data/input.csv")], timeout_s=300, memory="2g")
    if r.get("identifiable") is False:
        return cap("NOT IDENTIFIABLE (%s): %s\nNo estimate was computed; no other estimand was substituted."
                   % (r["method"], r["reason"]))
    head = "Causal effect of %s on %s (%s, %s, n=%d rows)" % (r["treatment"], r["outcome"], r["estimand"],
                                                              r["method"], r["n_rows"])
    lines = [head]
    if r.get("mediator"):
        lines.append("mediator (front-door set): %s; assumed graph: %s" % (r["frontdoor_set"], r["graph"]))
    lines += ["common causes adjusted: %s" % (r["common_causes"] or "none"),
              "ESTIMATE: %s" % _fmt(r["estimate"]),
              "95%% CI: %s (%s)" % (_fmt(r["ci95"]) if r["ci95"] else "unavailable", r["ci_kind"] or "-"),
              "Refutations:"]
    for x in r["refutations"]:
        if "error" in x:
            lines.append("  - %s: ERROR %s" % (x["refuter"], x["error"]))
        else:
            lines.append("  - %s: new_effect=%s p=%s -> %s (%s)" % (
                x["refuter"], _fmt(x["new_effect"]), _fmt(x["p_value"]), "PASS" if x["passed"] else "FAIL",
                x["expectation"]))
    lines.append("Assumptions (the estimate is only causal if these hold):")
    lines += ["  - " + s for s in r["assumptions"]]
    return cap("\n".join(lines))


def discover_dag_notears(csv_path: str, lambda1: float = 0.1, w_threshold: float = 0.3,
                         standardize: bool = False) -> str:
    r = _call("    import notears_impl\n    out = notears_impl.discover('/data/input.csv', float(a['l1']), "
              "float(a['wt']), standardize=bool(a['std']))",
              {"l1": 0.1 if lambda1 is None else lambda1, "wt": 0.3 if w_threshold is None else w_threshold,
               "std": bool(standardize)},
              mounts=[(_csv(csv_path), "/data/input.csv")], timeout_s=300, memory="2g")
    lines = [r["label"], "%s, n=%d rows, columns: %s" % (r["algorithm"], r["n_rows"], ", ".join(r["columns"])),
             "acyclicity h(W)=%.3g after thresholding (checked < 1e-8)" % r["h"],
             "Edges (%d), strongest first:" % len(r["edges"])]
    lines += ["  %s --> %s  (w=%.3f)" % (e["from"], e["to"], e["weight"]) for e in r["edges"]] or ["  (none)"]
    return cap("\n".join(lines))

def discover_graph(csv_path: str, alpha: float = 0.05) -> str:
    r = _call("    import causal_impl\n    out = causal_impl.discover('/data/input.csv', float(a['alpha']))",
              {"alpha": float(alpha or 0.05)}, mounts=[(_csv(csv_path), "/data/input.csv")], timeout_s=300,
              memory="2g")
    lines = [r["label"], "%s, n=%d rows, columns: %s" % (r["algorithm"], r["n_rows"], ", ".join(r["columns"])),
             "Edges (%d):" % len(r["edges"])] + ["  " + e for e in r["edges"] or ["(none)"]]
    return cap("\n".join(lines))

def invariant_causal_prediction(csv_path: str, target: str, env_col: str, alpha: float = 0.05,
                                predictors=None, correction: str = "tests") -> str:
    if not target or not env_col:
        raise ValueError("target and env_col are required")
    if correction not in ("tests", "subsets"):
        raise ValueError("correction must be 'tests' (Peters et al.) or 'subsets' (also Bonferroni over subsets)")
    r = _call("    import icp_impl\n    out = icp_impl.run('/data/input.csv', a['y'], a['e'], float(a['alpha']), "
              "a['x'] or None, a['c'])",
              {"y": target, "e": env_col, "alpha": float(alpha or 0.05), "x": _cols(predictors), "c": correction},
              mounts=[(_csv(csv_path), "/data/input.csv")], timeout_s=300, memory="2g")
    lines = ["Invariant Causal Prediction (Peters et al. 2016, linear-Gaussian) for target %s, environments from %s "
             "(%s), n=%d rows" % (r["target"], r["env_col"], ", ".join(r["environments"]), r["n_rows"]),
             "predictors: %s; %d subsets tested, %d accepted at %.3g per subset (alpha=%g, correction=%s)"
             % (", ".join(r["predictors"]), r["n_subsets"], r["n_accepted"], r["level_per_subset"], r["alpha"],
                r["correction"]),
             "ESTIMATED CAUSAL PARENTS: %s" % (", ".join(r["parents"]) or "none identified"),
             "status: %s" % r["status"],
             "per-variable evidence (largest p among sets WITHOUT the variable; small = needed):"]
    lines += ["  %s: %.3g" % (k, v) for k, v in sorted(r["variable_evidence"].items(), key=lambda kv: kv[1])]
    lines.append("best subsets (p = Bonferroni of F-test on residual means and Levene on residual variances):")
    lines += ["  {%s}: p=%.3g (means %.3g, variances %.3g)%s" % (", ".join(t["set"]) or "empty", t["p"], t["p_mean"],
                                                                 t["p_var"], " accepted" if t["accepted"] else "")
              for t in r["top_sets"][:6]]
    lines.append("Guarantee: the parents set contains only true causes with prob >= 1-alpha IF the target's "
                 "mechanism is linear, unchanged across environments and not confounded; it may miss parents.")
    return cap("\n".join(lines))
