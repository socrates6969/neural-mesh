"""Router SHADOW MODE: the fixed phase rule, the shadow log and `python -m ingest shadow-report`.

hooks/route_shadow.py (PreToolUse on Task|Agent) writes one row per dispatch to ``data/lifelong/route_shadow.jsonl``:
    {ts, session, tool_use_id, actual, mesh_pick, rule_pick, n_cand, mesh_ms}
It prints nothing and never blocks. This module reads those rows and says whether the mesh router has earned a (printed) suggestion.

The fixed rule is the pipeline-phase rule of scratchpad/router-eval/common.mjs (freight-recovery's fr-* roster): the FIRST token, left to
right, that is a phase keyword decides; default fr-builder. It only knows the fr-* agents, so it is "in domain" only for dispatches whose
ACTUAL agent is one of them. The promotion statistics therefore use the in-domain dispatches (a fair fight: the rule can be right there);
the all-dispatch line is printed for information but is biased towards the mesh (the rule can never pick general-purpose).

PROMOTION RULE (explicit; nothing is turned on by this module):
    ELIGIBLE  iff  n_in_domain >= 100  and  Wilson-95% LOWER bound of the mesh accuracy  >  the rule's accuracy (point estimate)
                   on the in-domain shadow dispatches (rows without an actual agent, and duplicate tool_use_ids, are dropped;
                   a missing mesh pick counts as a miss).
Only when the report says ELIGIBLE may the queen let the hook print a one-line suggestion.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .util import iter_jsonl

RULE = [
    ("fr-planner", ["plan", "planning", "spec", "requirements", "answers", "amend"]),
    ("fr-builder", ["build", "fix", "implement", "add", "apply", "create", "wire"]),
    ("fr-tester", ["test", "tests", "testing", "acceptance", "qa", "ci"]),
    ("fr-reviewer", ["review", "audit", "verify"]),
    ("fr-scriber", ["doc", "docs", "documentation", "readme", "architecture"]),
    ("fr-shipper", ["ship", "push", "pr", "merge", "release", "submission", "submit"]),
]
RULE_DEFAULT = "fr-builder"
FR_ROSTER = ("fr-planner", "fr-builder", "fr-tester", "fr-reviewer", "fr-scriber", "fr-shipper", "fr-ci-diagnoser", "fr-security-reviewer")
MIN_N = 100


def rule_pick(task_text: str) -> str:
    """Phase rule: first phase keyword (left to right) decides, else fr-builder."""
    table = {kw: agent for agent, kws in RULE for kw in kws}
    for tok in re.split(r"[^a-z0-9]+", str(task_text).lower()):
        if tok in table:
            return table[tok]
    return RULE_DEFAULT


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial p for the discordant pairs (b = mesh right/rule wrong, c = mesh wrong/rule right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_rows(path: Path) -> list[dict]:
    seen, out = set(), []
    for r in iter_jsonl(path):
        if not r.get("actual"):
            continue
        tid = r.get("tool_use_id")
        if tid:
            if tid in seen:
                continue
            seen.add(tid)
        out.append(r)
    out.sort(key=lambda r: (r.get("ts") or ""))
    return out


def _acc(rows: list[dict], who: str) -> dict:
    k = sum(1 for r in rows if r.get(who) == r["actual"])
    lo, hi = wilson(k, len(rows))
    return {"correct": k, "n": len(rows), "acc": round(k / len(rows), 4) if rows else None,
            "ci95": [None if math.isnan(lo) else round(lo, 4), None if math.isnan(hi) else round(hi, 4)]}


def summarize(rows: list[dict], window: int = 50) -> dict:
    dom = [r for r in rows if r["actual"] in FR_ROSTER]
    b = sum(1 for r in dom if r.get("mesh_pick") == r["actual"] and r.get("rule_pick") != r["actual"])
    c = sum(1 for r in dom if r.get("mesh_pick") != r["actual"] and r.get("rule_pick") == r["actual"])
    mesh, rule = _acc(dom, "mesh_pick"), _acc(dom, "rule_pick")
    eligible = len(dom) >= MIN_N and mesh["ci95"][0] is not None and rule["acc"] is not None and mesh["ci95"][0] > rule["acc"]
    why = []
    if len(dom) < MIN_N:
        why.append(f"only {len(dom)} in-domain shadow dispatches (< {MIN_N})")
    elif not eligible:
        why.append(f"mesh Wilson lower bound {mesh['ci95'][0]} is not above the rule accuracy {rule['acc']}")
    last = dom[-window:]
    return {
        "dispatches_logged": len(rows), "in_domain": len(dom), "mesh_pick_missing": sum(1 for r in rows if not r.get("mesh_pick")),
        "in_domain_mesh": mesh, "in_domain_rule": rule,
        "paired_in_domain": {"mesh_only_right": b, "rule_only_right": c, "mcnemar_exact_p": round(mcnemar_exact_p(b, c), 4)},
        "rolling": {"window": window, "mesh": _acc(last, "mesh_pick"), "rule": _acc(last, "rule_pick")},
        "all_dispatches_info_only": {"mesh": _acc(rows, "mesh_pick"), "rule": _acc(rows, "rule_pick")},
        "verdict": "ELIGIBLE" if eligible else "NOT ELIGIBLE", "reasons": why,
        "promotion_rule": f"ELIGIBLE iff n_in_domain >= {MIN_N} and mesh Wilson-95% lower bound > rule accuracy (in-domain)",
    }


def report(cfg, window: int = 50) -> dict:
    path = cfg.data / "route_shadow.jsonl"
    rows = load_rows(path)
    out = summarize(rows, window)
    out["log"] = str(path)
    return out


def format_report(s: dict) -> str:
    m, r = s["in_domain_mesh"], s["in_domain_rule"]
    ci = lambda x: "-" if x["acc"] is None else f"{x['acc']:.3f} [{x['ci95'][0]:.3f},{x['ci95'][1]:.3f}] ({x['correct']}/{x['n']})"
    lines = [f"shadow log: {s['log']}  dispatches={s['dispatches_logged']}  in-domain(fr-*)={s['in_domain']}  mesh pick missing={s['mesh_pick_missing']}",
             f"in-domain  mesh {ci(m)}", f"in-domain  rule {ci(r)}",
             f"paired: mesh-only right {s['paired_in_domain']['mesh_only_right']}, rule-only right {s['paired_in_domain']['rule_only_right']}, "
             f"McNemar exact p={s['paired_in_domain']['mcnemar_exact_p']}",
             f"last {s['rolling']['window']} in-domain  mesh {ci(s['rolling']['mesh'])}   rule {ci(s['rolling']['rule'])}",
             f"all dispatches (info only, biased to the mesh)  mesh {ci(s['all_dispatches_info_only']['mesh'])}   rule {ci(s['all_dispatches_info_only']['rule'])}",
             f"rule: {s['promotion_rule']}", f"VERDICT: {s['verdict']}" + (f" ({'; '.join(s['reasons'])})" if s["reasons"] else "")]
    return "\n".join(lines)
