"""Sandboxed tools: math_solve, stats_test, causal_analysis, discover_graph. All model-supplied code and data
processing runs in the network-less grader container (image MESH_TOOLS_IMAGE, default meshtrain-grader:2)."""
from __future__ import annotations

import json
from pathlib import Path

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


def stats_test(kind: str, data_json) -> str:
    data = as_json(data_json, "data_json")
    r = _call("    import stats_impl\n    out = stats_impl.compute(a['kind'], a['data'])", {"kind": kind, "data": data},
              timeout_s=60)
    lines = [r.pop("test")]
    eff = r.pop("effect", None)
    for k, v in r.items():
        lines.append("%s: %s" % (k, _fmt(v)))
    if eff:
        lines.append("EFFECT SIZE: " + _fmt(eff))
    lines.append("Report the effect size and interval with the p-value; p alone does not show magnitude.")
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


def causal_analysis(csv_path: str, treatment: str, outcome: str, common_causes=None, method: str = "backdoor",
                    num_simulations: int = 20) -> str:
    if not treatment or not outcome:
        raise ValueError("treatment and outcome are required")
    cc = common_causes or []
    if isinstance(cc, str):
        cc = [c.strip() for c in (json.loads(cc) if cc.strip().startswith("[") else cc.split(",")) if c.strip()]
    args = {"t": treatment, "y": outcome, "cc": list(cc), "m": method,
            "n": max(5, min(int(num_simulations or 20), 100))}
    r = _call("    import causal_impl\n    out = causal_impl.estimate('/data/input.csv', a['t'], a['y'], a['cc'], "
              "a['m'], a['n'])", args, mounts=[(_csv(csv_path), "/data/input.csv")], timeout_s=300, memory="2g")
    lines = ["Causal effect of %s on %s (%s, %s, n=%d rows)" % (r["treatment"], r["outcome"], r["estimand"],
                                                                 r["method"], r["n_rows"]),
             "common causes adjusted: %s" % (r["common_causes"] or "none"),
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


def discover_graph(csv_path: str, alpha: float = 0.05) -> str:
    r = _call("    import causal_impl\n    out = causal_impl.discover('/data/input.csv', float(a['alpha']))",
              {"alpha": float(alpha or 0.05)}, mounts=[(_csv(csv_path), "/data/input.csv")], timeout_s=300,
              memory="2g")
    lines = [r["label"], "%s, n=%d rows, columns: %s" % (r["algorithm"], r["n_rows"], ", ".join(r["columns"])),
             "Edges (%d):" % len(r["edges"])] + ["  " + e for e in r["edges"] or ["(none)"]]
    return cap("\n".join(lines))

