#!/usr/bin/env python
"""Claude Code PreToolUse hook (matcher "Task|Agent"): SHADOW-log what the mesh router and the fixed phase rule would have picked.

It PRINTS NOTHING, NEVER BLOCKS (always exit 0, no JSON decision), and writes one row per dispatch to data/lifelong/route_shadow.jsonl:
    {ts, session, tool_use_id, actual, mesh_pick, rule_pick, hybrid_pick, n_cand, mesh_ms, hybrid_ms}
`python -m ingest shadow-report` turns the log into accuracy-vs-actual with Wilson CIs and the explicit promotion verdict (ingest/shadow.py).
Disable with NEURAL_MESH_SHADOW=off (checked before any import).  Budget NEURAL_MESH_SHADOW_BUDGET_S (default 2.5 s) via a watchdog + os._exit.

mesh_pick = `neural-mesh route --state <router_state.json> --features <same features the nightly ingest trains on> --candidates <agents in the state>`.
The state file is never modified (the route call runs on a temporary copy; checked by hash in hooks/tests). No prompt text is logged: only the chosen agent names and the
timestamp/session/tool_use_id, so the file carries nothing to scrub.

MESH_ROUTER=v2 (hive router-v2; default v1): mesh_pick comes from router v2 instead (ingest/router_v2.py: MiniLM embedding of the same
"description :: prompt" text the ingest stores, contextual bandit, read-only on router_state.v2.json, candidates = agents in that state),
and the row gets "router": "v2". The hybrid (v1 project state + phase-rule prior) column is unchanged.
"""
import os
import sys

if os.environ.get("NEURAL_MESH_SHADOW", "").strip().lower() == "off":
    sys.exit(0)

import json
import shutil
import subprocess
import tempfile
import threading
import time

T0 = time.monotonic()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET_S = 2.5
ROUTE_TIMEOUT_S = 2.0


def mesh_pick(cfg, feats, cands):
    """(agent or None, ms). Never raises. Routes against a throw-away COPY of the state: newer builds of `route` bump telemetry counters
    (calls, routes, ...) in the state file they are given, and a read-modify-write on the live file could race the nightly feedback."""
    t = time.monotonic()
    tmp = None
    try:
        if not (cfg.mesh_exe.exists() and cfg.router_state.exists() and cands):
            return None, 0
        fd, tmp = tempfile.mkstemp(prefix="route-shadow-", suffix=".json")
        os.close(fd)
        shutil.copyfile(cfg.router_state, tmp)
        cmd = [str(cfg.mesh_exe), "route", "--state", tmp, "--features", ",".join(feats) or "_none", "--candidates", ",".join(cands)]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=ROUTE_TIMEOUT_S)
        agent = json.loads(r.stdout).get("agent") if r.returncode == 0 else None
        return (agent if isinstance(agent, str) and agent else None), round((time.monotonic() - t) * 1000)
    except Exception:
        return None, round((time.monotonic() - t) * 1000)
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def v2_pick(cfg, text, feats, project):
    """(agent or None, ms, n candidates) from router v2. Never raises. The hook runs `python -S`: site-packages (numpy, onnxruntime) are enabled here only."""
    t = time.monotonic()
    try:
        import site
        site.main()
        from ingest import router_v2
        from ingest.meshio import v2_state
        st = v2_state(cfg)
        if not st.exists():
            return None, 0, 0
        out = router_v2.route_cmd(st, feats, None, text=text, project=project)
        agent = out.get("agent") if out.get("ok") else None
        return (agent if isinstance(agent, str) and agent else None), round((time.monotonic() - t) * 1000), int(out.get("n_cand") or 0)
    except Exception:
        return None, round((time.monotonic() - t) * 1000), 0


HYBRID_PRIOR_K = 500.0   # R122: with the declared roster as candidates, k >= 500 follows the phase rule almost always (router_eval: ties the rule in-domain)


def roster_of(cwd):
    """agent names declared by the project (<cwd>/.claude/agents/*.md stems), [] when there are none"""
    try:
        d = os.path.join(cwd, ".claude", "agents")
        return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".md") and "," not in f)[:64]
    except Exception:
        return []


def hybrid_pick(cfg, cwd, feats, rule, seed_key):
    """R122 shadow-only: the PROJECT's own router state + its declared roster as candidates + the phase rule as warm prior (route --prior). None if the project has no state."""
    try:
        from ingest.extract import project_of
        from ingest.meshio import project_state
        import zlib
        st = project_state(cfg, {"project": project_of(cwd, "")})
        if not (cfg.mesh_exe.exists() and st.exists()):
            return None
        with open(st, encoding="utf-8") as f:
            agents = [a for a in json.load(f).get("agents", []) if isinstance(a, str) and a and "," not in a]
        cands = agents + [a for a in roster_of(cwd) if a not in agents]
        if len(cands) < 2:
            return None
        fd, tmp = tempfile.mkstemp(prefix="route-shadow-h-", suffix=".json")
        os.close(fd)
        try:
            shutil.copyfile(st, tmp)
            cmd = [str(cfg.mesh_exe), "route", "--state", tmp, "--features", ",".join(feats) or "_none", "--candidates", ",".join(cands),
                   "--prior", rule, "--prior-k", str(HYBRID_PRIOR_K), "--seed", str(zlib.crc32(str(seed_key).encode()))]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=ROUTE_TIMEOUT_S)
            agent = json.loads(r.stdout).get("agent") if r.returncode == 0 else None
            return agent if isinstance(agent, str) and agent else None
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
    except Exception:
        return None


def state_agents(cfg):
    try:
        with open(cfg.router_state, encoding="utf-8") as f:
            return [a for a in json.load(f).get("agents", []) if isinstance(a, str) and a and "," not in a]
    except Exception:
        return []


def build_row(data, cfg):
    """The shadow row for one PreToolUse payload, or None when it is not an agent dispatch."""
    from ingest.extract import features
    from ingest.shadow import rule_pick
    ti = data.get("tool_input") or {}
    if not isinstance(ti, dict):
        return None
    desc, prompt = str(ti.get("description") or ""), str(ti.get("prompt") or "")
    actual = str(ti.get("subagent_type") or "general-purpose")
    feats = [f for f in features(desc + " " + prompt[:400]) if "," not in f]
    v2 = os.environ.get("MESH_ROUTER", "v1").strip().lower() == "v2"
    if v2:
        from ingest.extract import project_of
        text = desc + (" :: " + prompt[:400] if prompt else "")          # = the task_text the ingest stores (ingest/pipeline.py)
        pick, ms, n_cand = v2_pick(cfg, text, feats, project_of(str(data.get("cwd") or ""), "") or "default")
    else:
        cands = state_agents(cfg)
        pick, ms = mesh_pick(cfg, feats, cands)
        n_cand = len(cands)
    rule = rule_pick(desc + " :: " + prompt[:400])
    t_h = time.monotonic()
    hyb = hybrid_pick(cfg, str(data.get("cwd") or ""), feats, rule, data.get("tool_use_id") or data.get("session_id") or "")
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z", "session": data.get("session_id"), "tool_use_id": data.get("tool_use_id"),
           "actual": actual, "mesh_pick": pick, "rule_pick": rule, "hybrid_pick": hyb, "n_cand": n_cand, "mesh_ms": ms,
           "hybrid_ms": round((time.monotonic() - t_h) * 1000)}
    if v2:
        row["router"] = "v2"
    return row


def log_shadow(data, cfg=None):
    from ingest.config import Config
    from ingest.util import append_jsonl
    cfg = cfg or Config()
    if data.get("tool_name") not in ("Task", "Agent"):
        return False
    row = build_row(data, cfg)
    if row is None:
        return False
    append_jsonl(cfg.data / "route_shadow.jsonl", [row])
    return True


def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        if isinstance(data, dict):
            if ROOT not in sys.path:
                sys.path.insert(0, ROOT)
            box = {}

            def work():
                try:
                    box["ok"] = log_shadow(data)
                except BaseException:
                    box["ok"] = False
            th = threading.Thread(target=work, daemon=True)
            th.start()
            th.join(max(0.0, float(os.environ.get("NEURAL_MESH_SHADOW_BUDGET_S", BUDGET_S)) - (time.monotonic() - T0)))
    except BaseException:
        pass
    os._exit(0)


if __name__ == "__main__":
    main()
