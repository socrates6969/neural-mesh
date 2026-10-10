"""Feed outcomes into neural-mesh's `feedback` CLI state (plus its MeshStore journal, so a new router state can warm-start).

Feed policy (mesh-coder's R117-R121, router-eval: label-fed 67% vs reward-only 33% prequential on 69 real dispatches):
* an ENVIRONMENT failure (turn limit, rate limit / 429 / overloaded, worktree block, API error: signals ``turn_limit`` / ``redispatch_cue`` /
  ``api_error`` on a re-dispatch outcome) is sent as ``--agent X --reward 0 --failure-class env``: it only moves the agent's dispatch-risk score,
  never the skill weights, health or benching;
* every other outcome is sent as ``--label <agent that actually did the work>`` (supervised: the leader's actual choice is the label, which is what
  shadow mode measures); ``INGEST_MESH_FEED=reward`` restores the old reward-only feed (``--agent X --reward r``, env failures included).
* if the binary rejects the new flags (built before R117) the event is re-sent as the legacy reward feed, so nothing is lost (``stats.fallback``).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .util import iter_jsonl, sha16

ENV_SIGNALS = {"turn_limit", "redispatch_cue", "api_error"}


def is_env_failure(outcome: dict) -> bool:
    return outcome.get("outcome") == "re-dispatch" and bool(ENV_SIGNALS & set(outcome.get("signals") or []))


def clean_agent(outcome: dict) -> str:
    return (outcome.get("agent") or "general-purpose").replace(",", "_")


def clean_features(outcome: dict) -> str:
    return ",".join([f for f in outcome.get("features") or [] if f and "," not in f][:24] or ["_none"])


def feedback_args(outcome: dict, mode: str = "label") -> list[str]:
    """The policy-specific tail of the `feedback` command line (features/state/store are added by the caller)."""
    agent = clean_agent(outcome)
    if mode == "reward":      # the legacy feed (INGEST_MESH_FEED=reward, and the fallback for a binary that predates --label): reward only, no classes
        return ["--agent", agent, "--reward", f"{max(0.0, min(1.0, float(outcome['reward']))):.4f}"]
    if is_env_failure(outcome):
        return ["--agent", agent, "--reward", "0", "--failure-class", "env"]
    return ["--label", agent]


def world_of(outcome: dict) -> str:
    return "".join(c if c.isalnum() or c in "-_:" else "_" for c in (outcome.get("project") or "default"))[:64] or "default"


def project_state(cfg, outcome: dict) -> Path:
    """R122: one router state per project next to the global one (data/lifelong/mesh/projects/<world>.json). Routing inside a project with its own
    state and its own candidate roster beat the global state on prequential replay (ingest/router_eval.py); the global state is unchanged."""
    return cfg.router_state.parent / "projects" / (world_of(outcome).replace(":", "_") + ".json")


class MeshFeedback:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fed_path: Path = cfg.state / "mesh_fed.txt"
        self.fed: set[str] = set()
        if self.fed_path.exists():
            self.fed = {x.strip() for x in self.fed_path.read_text(encoding="utf-8").splitlines() if x.strip()}
        self.stats = {"fed": 0, "failed": 0, "skipped": 0, "label": 0, "env": 0, "reward": 0, "fallback": 0, "project_fed": 0}

    def available(self) -> bool:
        return self.cfg.mesh_exe.exists()

    def command(self, outcome: dict, state: Path | None = None, with_store: bool = True, mode: str | None = None) -> list[str]:
        cmd = [str(self.cfg.mesh_exe), "feedback", "--state", str(state or self.cfg.router_state), "--features", clean_features(outcome),
               *feedback_args(outcome, mode or self.cfg.mesh_feed_mode)]
        if with_store:
            cmd += ["--store", str(self.cfg.meshstore_root), "--world", world_of(outcome)]
        return cmd

    def feed(self, outcome: dict) -> None:
        """At-most-once per outcome id: the id is recorded BEFORE the call, so a crash can lose one event but never double-count it."""
        oid = outcome["id"]
        if oid in self.fed:
            self.stats["skipped"] += 1
            return
        self.fed_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.fed_path, "a", encoding="utf-8") as f:
            f.write(oid + "\n")
        self.fed.add(oid)
        self.cfg.router_state.parent.mkdir(parents=True, exist_ok=True)
        kind = "env" if is_env_failure(outcome) else ("label" if self.cfg.mesh_feed_mode == "label" else "reward")
        ok = self._run(self.command(outcome))
        if not ok and kind != "reward":
            # A neural-mesh.exe built before --label / --failure-class existed rejects them. Never lose the event: fall back to the legacy feed.
            ok = self._run(self.command(outcome, mode="reward"))
            if ok:
                kind = "reward"
                self.stats["fallback"] += 1
        self.stats["fed" if ok else "failed"] += 1
        if ok:
            self.stats[kind] += 1
        if ok and self.cfg.mesh_project_states:       # same event, same policy, into the project's own state (no journal: --store omitted)
            pst = project_state(self.cfg, outcome)
            pst.parent.mkdir(parents=True, exist_ok=True)
            pmode = "reward" if kind == "reward" else None
            if self._run(self.command(outcome, state=pst, with_store=False, mode=pmode)):
                self.stats["project_fed"] += 1

    @staticmethod
    def _run(cmd: list[str]) -> bool:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return r.returncode == 0 and '"ok":true' in r.stdout.replace(" ", "")
        except (OSError, subprocess.SubprocessError):
            return False


def feature_key(features: list[str]) -> str:
    return sha16(*features)


def rebuild_router(cfg, dry_run: bool = False, evaluate: bool = False) -> dict:
    """Replay every recorded outcome (time order) into a FRESH router state with the current feed policy.
    The live state is backed up first (router_state.json.pre-rebuild) and replaced only when not dry-run. The MeshStore journal is NOT written
    (--store omitted) so the journal is not duplicated; mesh_fed.txt already holds these ids, so the next nightly run does not re-feed them.
    ``evaluate``: prequential top-1 accuracy of `route` (candidates = agents seen so far) against the actual agent, before each update,
    reported next to the fixed phase rule on the in-domain (fr-*) dispatches."""
    import shutil
    from .shadow import FR_ROSTER, rule_pick
    mf = MeshFeedback(cfg)
    if not mf.available():
        return {"ok": False, "error": "neural-mesh.exe missing"}
    rows = sorted(iter_jsonl(cfg.outcomes), key=lambda r: (r.get("ts") or "", r.get("id") or ""))
    live = cfg.router_state
    scratch = live.with_name("router_state.rebuild.tmp")
    scratch.unlink(missing_ok=True)
    pscratch: dict[Path, Path] = {}      # live project state -> scratch state (R122)
    done_p: set[Path] = set()
    seen: list[str] = []
    n = ok = 0
    ev = {"n": 0, "mesh": 0, "dom_n": 0, "dom_mesh": 0, "dom_rule": 0}
    for r in rows:
        agent = clean_agent(r)
        if evaluate and len(seen) >= 2 and not is_env_failure(r):
            pick = _route(cfg, scratch, clean_features(r), seen)
            ev["n"] += 1
            ev["mesh"] += pick == agent
            if agent in FR_ROSTER:
                ev["dom_n"] += 1
                ev["dom_mesh"] += pick == agent
                ev["dom_rule"] += rule_pick(r.get("task_text") or "") == agent
        n += 1
        ok += MeshFeedback._run(mf.command(r, state=scratch, with_store=False))
        if cfg.mesh_project_states:
            lp = project_state(cfg, r)
            sp = pscratch.setdefault(lp, lp.with_name(lp.name + ".rebuild.tmp"))
            if lp not in done_p:
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.unlink(missing_ok=True)
                done_p.add(lp)
            MeshFeedback._run(mf.command(r, state=sp, with_store=False))
        if agent not in seen:
            seen.append(agent)
    out = {"ok": True, "outcomes": n, "fed_ok": ok, "mode": cfg.mesh_feed_mode, "agents": len(seen), "dry_run": dry_run}
    if evaluate and ev["n"]:
        out["prequential"] = {"all_dispatches_mesh": [ev["mesh"], ev["n"], round(ev["mesh"] / ev["n"], 4)]}
        if ev["dom_n"]:
            out["prequential"]["in_domain_fr_mesh"] = [ev["dom_mesh"], ev["dom_n"], round(ev["dom_mesh"] / ev["dom_n"], 4)]
            out["prequential"]["in_domain_fr_rule"] = [ev["dom_rule"], ev["dom_n"], round(ev["dom_rule"] / ev["dom_n"], 4)]
    out["project_states"] = len(pscratch)
    if dry_run:
        scratch.unlink(missing_ok=True)
        for sp in pscratch.values():
            sp.unlink(missing_ok=True)
        return out
    if live.exists():
        shutil.copy2(live, live.with_name("router_state.json.pre-rebuild"))
    if scratch.exists():
        scratch.replace(live)
    for lp, sp in pscratch.items():
        if lp.exists():
            shutil.copy2(lp, lp.with_name(lp.name + ".pre-rebuild"))
        if sp.exists():
            sp.replace(lp)
    return out


def _route(cfg, state: Path, feats: str, cands: list[str]) -> str | None:
    try:
        r = subprocess.run([str(cfg.mesh_exe), "route", "--state", str(state), "--features", feats, "--candidates", ",".join(cands)],
                           capture_output=True, text=True, timeout=30)
        return json.loads(r.stdout).get("agent") if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
