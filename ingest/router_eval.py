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
                seed0: int = 7, roster: bool = False) -> dict:
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
            if len(cands) >= 2 and not is_env_failure(r):
                cmd = [str(exe), "route", "--state", str(state), "--features", feats, "--candidates", ",".join(cands), "--seed", str(seed0 + i)]
                if prior_k is not None:
                    cmd += ["--prior", rule_pick(r.get("task_text") or ""), "--prior-k", str(prior_k)]
                pick = (_run(cmd) or {}).get("agent")
                top = counter.most_common(1)[0][0] if counter else None
                n += 1
                ok += pick == agent
                maj += top == agent
                trace.append({"i": i, "proj": proj, "actual": agent, "mesh": pick == agent, "maj": top == agent,
                              "rule": rule_pick(r.get("task_text") or "") == agent, "fr": agent in FR_ROSTER})
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
    return {"feed": feed, "scope": scope, "prior_k": prior_k, "units": units, "roster": roster,
            "paired_mesh_vs_majority_all": paired("mesh", "maj"), "paired_mesh_vs_rule_in_domain": paired("mesh", "rule", only_fr=True),
            "all": pack(ok, n), "all_majority_so_far": pack(maj, n),
            "in_domain_fr": pack(dok, dn), "in_domain_fr_majority_so_far": pack(dmaj, dn), "in_domain_fr_phase_rule": pack(drule, dn)}


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
    p.add_argument("--json", type=Path, help="write the result list here")
    a = p.parse_args(argv)
    res = evaluate(Config(), a.outcomes, variants=grid2_variants() if a.variants == "grid2" else None, workers=a.workers)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for r in res:
        f = lambda x: "-" if not x["n"] else f"{x['acc']:.3f} ({x['k']}/{x['n']})"
        print(f"{r['feed']:7s} {r['scope']:10s} roster={int(r['roster'])} prior_k={str(r['prior_k']):8s} units={r['units']:8s} all={f(r['all'])} maj={f(r['all_majority_so_far'])} | "
              f"fr={f(r['in_domain_fr'])} rule={f(r['in_domain_fr_phase_rule'])} maj={f(r['in_domain_fr_majority_so_far'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
