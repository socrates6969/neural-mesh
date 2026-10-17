"""Prequential router evaluation with variants (R122): replay the recorded dispatches in time order into a FRESH scratch router state; before each
update, ask `route` which agent it would pick and compare with the agent the leader actually used.

Variants (each runs in its own scratch state, in parallel threads):
  feed   : "label"   = `--label <actual agent>` (env failures: `--agent X --reward 0 --failure-class env`)           [the production policy]
           "success" = like label, but a defect / fix-round / blocked outcome is fed as `--agent X --reward <r>` (outcome-aware: only work that
                       succeeded is imitated, failed work is penalised instead of imitated)
           "reward"  = legacy `--agent X --reward r` for everything
  scope  : "perproject" = like "project" but ALSO a separate router state per project (no cross-project wording noise)
           "global"  = candidates are all agents seen so far (what hooks/route_shadow.py does today)
           "project" = candidates are the agents seen so far IN THE SAME PROJECT (a roster-aware hook; the project is known from the cwd)
  prior  : None, or the phase rule's pick passed as `--prior` with prior_k = k (the warm prior is followed with probability k/(k+n); n = mean
           per-feature evidence), i.e. "phase rule first, learned correction as evidence accumulates"
  units  : "dispatch" = only kind=dispatch rows are predicted/fed;  "all" = continuations too (they are not routing decisions: a SendMessage goes to an
           agent already chosen, so scoring them mostly rewards remembering the previous message)

Reported per variant: accuracy over dispatches, over the in-domain (fr-*) dispatches next to the fixed phase rule on the same rows, and the
prequential MAJORITY-SO-FAR baseline (the cheapest thing to beat). Wilson 95% intervals. Chronological and leak-free by construction (predict, then feed).

CLI: python -m ingest.router_eval [--outcomes PATH] [--variants default|full] [--workers N] [--json OUT]
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .meshio import clean_agent, clean_features, feedback_args, is_env_failure
from .shadow import FR_ROSTER, mcnemar_exact_p, rule_pick, wilson
from .util import iter_jsonl

TIMEOUT = 60


def _run(cmd: list[str]) -> dict | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
        return json.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _feed_args(row: dict, feed: str) -> list[str]:
    agent = clean_agent(row)
    if is_env_failure(row):
        return ["--agent", agent, "--reward", "0", "--failure-class", "env"]
    if feed == "label":
        return ["--label", agent]
    if feed == "success":
        if row.get("outcome") == "success":
            return ["--label", agent]
        return ["--agent", agent, "--reward", f"{max(0.0, min(1.0, float(row.get('reward', 0.0)))):.4f}"]
    return feedback_args(row, "reward")


def run_variant(exe: Path, rows: list[dict], feed: str = "label", scope: str = "global", prior_k: float | None = None, units: str = "dispatch",
                seed0: int = 7, roster: bool = False, late_agent_init: float | None = None) -> dict:
    work = Path(tempfile.mkdtemp(prefix="router-eval-"))
    state_g = work / "state.json"
    states: dict[str, Path] = {}
    seen_all: list[str] = []
    seen_proj: dict[str, list[str]] = {}
    hist: Counter = Counter()
    hist_proj: dict[str, Counter] = {}
    n = ok = maj = 0
    dn = dok = dmaj = drule = 0
    per_proj: dict[str, list[int]] = {}
    trace: list[dict] = []
    inited: set[Path] = set()

    def ensure(state: Path) -> None:
        """late_agent_init=None keeps the binary default (0.1); a number is applied to each fresh scratch state (live state: 0.0)."""
        if late_agent_init is not None and state not in inited:
            inited.add(state)
            _run([str(exe), "config", "--state", str(state), "--late-agent-init", str(late_agent_init)])
    try:
        for i, r in enumerate(rows):
            if units == "dispatch" and r.get("kind") != "dispatch":
                continue
            agent = clean_agent(r)
            feats = clean_features(r)
            proj = r.get("project") or "default"
            cands = seen_all if scope == "global" else list(seen_proj.get(proj, []))
            if roster and scope != "global" and proj == "freight-recovery":     # the project's declared agents (.claude/agents/fr-*.md) are known up front
                cands += [a for a in FR_ROSTER if a not in cands]
            counter = hist if scope == "global" else hist_proj.setdefault(proj, Counter())
            state = state_g
            if scope == "perproject":        # a separate router state per project (the hook knows the cwd): no cross-project wording noise
                state = states.setdefault(proj, work / f"state-{len(states)}.json")
            ensure(state)
            if len(cands) >= 2 and not is_env_failure(r):
                cmd = [str(exe), "route", "--state", str(state), "--features", feats, "--candidates", ",".join(cands), "--seed", str(seed0 + i)]
                if prior_k is not None:
                    cmd += ["--prior", rule_pick(r.get("task_text") or ""), "--prior-k", str(prior_k)]
                pick = (_run(cmd) or {}).get("agent")
                top = counter.most_common(1)[0][0] if counter else None
                n += 1
                ok += pick == agent
                maj += top == agent
                rp = rule_pick(r.get("task_text") or "")
                trace.append({"i": i, "proj": proj, "actual": agent, "mesh": pick == agent, "maj": top == agent,
                              "rule": rp == agent, "fr": agent in FR_ROSTER, "pick": pick, "top": top, "rule_pick": rp})
                s = per_proj.setdefault(proj, [0, 0])
                s[0] += 1
                s[1] += pick == agent
                if agent in FR_ROSTER:
                    dn += 1
                    dok += pick == agent
                    dmaj += top == agent
                    drule += rule_pick(r.get("task_text") or "") == agent
            # feed
            _run([str(exe), "feedback", "--state", str(state), "--features", feats, *_feed_args(r, feed)])
            if agent not in seen_all:
                seen_all.append(agent)
            sp = seen_proj.setdefault(proj, [])
            if agent not in sp:
                sp.append(agent)
            hist[agent] += 1
            hist_proj.setdefault(proj, Counter())[agent] += 1
    finally:
        shutil.rmtree(work, ignore_errors=True)

    def pack(k, m):
        lo, hi = wilson(k, m)
        return {"k": k, "n": m, "acc": round(k / m, 4) if m else None, "ci95": [round(lo, 4), round(hi, 4)] if m else None}
    def paired(a_key, b_key, only_fr=False):
        t = [x for x in trace if x["fr"] or not only_fr]
        b = sum(1 for x in t if x[a_key] and not x[b_key])
        c = sum(1 for x in t if x[b_key] and not x[a_key])
        return {"a_only_right": b, "b_only_right": c, "mcnemar_exact_p": round(mcnemar_exact_p(b, c), 4), "n": len(t)}
    res = {"feed": feed, "scope": scope, "prior_k": prior_k, "units": units, "roster": roster, "late_agent_init": late_agent_init,
           "paired_mesh_vs_majority_all": paired("mesh", "maj"), "paired_mesh_vs_rule_in_domain": paired("mesh", "rule", only_fr=True),
           "all": pack(ok, n), "all_majority_so_far": pack(maj, n),
           "in_domain_fr": pack(dok, dn), "in_domain_fr_majority_so_far": pack(dmaj, dn), "in_domain_fr_phase_rule": pack(drule, dn)}
    res.update(baseline_report(trace))
    return res


# ---- E2: balanced accuracy, per-class recall, McNemar vs each baseline, per-project no-regression, MDE --------------------------------
def baseline_report(trace: list[dict]) -> dict:
    """Side-by-side mesh / majority-so-far / phase rule over the predicted dispatches (`trace`), all and in-domain (fr-*).
    Majority-so-far predicts the most frequent agent seen so far (None = wrong when nothing was seen yet); the phase rule only ever names fr-* agents.
    Balanced accuracy = macro mean of per-class recall over the actual agents present. no_regression follows rGi evaluateCandidate: the
    candidate must not have FEWER successes than the baseline in any project."""
    from llm import stats as S

    def section(rows: list[dict]) -> dict:
        y = [x["actual"] for x in rows]
        preds = {"mesh": [x["pick"] for x in rows], "majority_so_far": [x["top"] for x in rows], "phase_rule": [x["rule_pick"] for x in rows]}
        top_class = Counter(y).most_common(1)[0][0] if y else None
        out = {"n": len(rows), "majority_class": top_class, "systems": {}}
        for name, p in preds.items():
            rec = S.recall_by_class(y, p)
            minority = [(a, b) for a, b in zip(y, p) if a != top_class]
            out["systems"][name] = {"acc": round(sum(a == b for a, b in zip(y, p)) / len(y), 4) if y else None,
                                    "balanced_acc": None if not y else round(S.balanced_accuracy(y, p), 4),
                                    "minority_recall": None if not minority else round(sum(a == b for a, b in minority) / len(minority), 4),
                                    "minority_n": len(minority),
                                    "recall": {a: [d["k"], d["n"]] for a, d in sorted(rec.items(), key=lambda kv: -kv[1]["n"])}}
        for base, key in (("majority_so_far", "maj"), ("phase_rule", "rule")):
            b = sum(1 for x in rows if x["mesh"] and not x[key])
            c = sum(1 for x in rows if x[key] and not x["mesh"])
            projs: dict[str, list[int]] = {}
            for x in rows:
                s = projs.setdefault(x["proj"], [0, 0, 0])
                s[0] += 1
                s[1] += x["mesh"]
                s[2] += x[key]
            worse = sorted(p for p, s in projs.items() if s[1] < s[2])
            out[f"mesh_vs_{base}"] = {"mesh_only_right": b, "base_only_right": c, "mcnemar_exact_p": round(S.mcnemar_exact_p(b, c), 4),
                                      "mde_at_observed": S.mde_observed(len(rows), b, c) if rows else None,
                                      "mde_planning_d30": S.mde_paired(len(rows), 0.3) if rows else None,
                                      "per_project": {p: {"n": s[0], "mesh": s[1], "base": s[2]} for p, s in sorted(projs.items(), key=lambda kv: -kv[1][0])},
                                      "no_regression": not worse, "regressed_projects": worse}
        return out
    return {"e2_all": section(trace), "e2_in_domain_fr": section([x for x in trace if x["fr"]])}


def e2_line(r: dict) -> str:
    """E2 summary, two lines (all dispatches / fr-*): balanced accuracy mesh|majority|rule, minority recall, McNemar p, MDE, no-regression verdicts."""
    def part(tag, s):
        sy = s["systems"]
        mm, mr = s["mesh_vs_majority_so_far"], s["mesh_vs_phase_rule"]
        mde = (mm["mde_at_observed"] or {}).get("min_diff")
        return (f"  {tag} n={s['n']} balanced mesh/maj/rule={sy['mesh']['balanced_acc']}/{sy['majority_so_far']['balanced_acc']}/{sy['phase_rule']['balanced_acc']} "
                f"minority-recall mesh/maj={sy['mesh']['minority_recall']}/{sy['majority_so_far']['minority_recall']} | McNemar p vs maj={mm['mcnemar_exact_p']} "
                f"vs rule={mr['mcnemar_exact_p']} MDE={None if mde is None else round(mde, 3)} | no-regression vs maj={mm['no_regression']} vs rule={mr['no_regression']}")
    return part("ALL", r["e2_all"]) + "\n" + part("FR ", r["e2_in_domain_fr"])

def default_variants() -> list[dict]:
    return [
        dict(feed="label", scope="global", prior_k=None, units="all"),         # = today's `rebuild-router --eval`
        dict(feed="label", scope="global", prior_k=None, units="dispatch"),
        dict(feed="label", scope="project", prior_k=None, units="dispatch"),
        dict(feed="success", scope="project", prior_k=None, units="dispatch"),
        dict(feed="reward", scope="project", prior_k=None, units="dispatch"),
        dict(feed="label", scope="project", prior_k=5.0, units="dispatch"),
        dict(feed="label", scope="project", prior_k=50.0, units="dispatch"),
        dict(feed="label", scope="project", prior_k=500.0, units="dispatch"),
        dict(feed="success", scope="project", prior_k=50.0, units="dispatch"),
    ]


def grid2_variants() -> list[dict]:
    """Per-project state: label feed, with and without the phase-rule prior at increasing prior_k (k/(k+n) follow probability)."""
    v = [dict(feed="label", scope="perproject", prior_k=None, units="dispatch"), dict(feed="success", scope="perproject", prior_k=None, units="dispatch"),
         dict(feed="label", scope="perproject", prior_k=None, units="dispatch", roster=True)]
    v += [dict(feed="label", scope="perproject", prior_k=k, units="dispatch", roster=True) for k in (1.0, 5.0, 20.0, 100.0, 500.0, 5000.0)]
    v += [dict(feed="label", scope="perproject", prior_k=k, units="dispatch") for k in (500.0,)]
    return v


def evaluate(cfg, outcomes: Path | None = None, variants: list[dict] | None = None, workers: int = 8) -> list[dict]:
    rows = sorted(iter_jsonl(outcomes or cfg.outcomes), key=lambda r: (r.get("ts") or "", r.get("id") or ""))
    variants = variants or default_variants()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_variant, cfg.mesh_exe, rows, **v) for v in variants]
        return [f.result() for f in futs]


def main(argv=None) -> int:
    import argparse
    from .config import Config
    p = argparse.ArgumentParser(prog="python -m ingest.router_eval")
    p.add_argument("--outcomes", type=Path)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--variants", choices=("default", "grid2"), default="default")
    p.add_argument("--late-agent-init", type=float, default=None, help="apply this late_agent_init to every scratch state (live config: 0.0; default: binary default 0.1)")
    p.add_argument("--json", type=Path, help="write the result list here")
    a = p.parse_args(argv)
    vs = grid2_variants() if a.variants == "grid2" else default_variants()
    if a.late_agent_init is not None:
        vs = [dict(v, late_agent_init=a.late_agent_init) for v in vs]
    res = evaluate(Config(), a.outcomes, variants=vs, workers=a.workers)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in res:
        f = lambda x: "-" if not x["n"] else f"{x['acc']:.3f} ({x['k']}/{x['n']})"
        print(f"{r['feed']:7s} {r['scope']:10s} roster={int(r['roster'])} lai={r.get('late_agent_init')} prior_k={str(r['prior_k']):8s} units={r['units']:8s} all={f(r['all'])} maj={f(r['all_majority_so_far'])} | "
              f"fr={f(r['in_domain_fr'])} rule={f(r['in_domain_fr_phase_rule'])} maj={f(r['in_domain_fr_majority_so_far'])}")
        print(e2_line(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
