"""Tests for hooks/route_shadow.py: logs the right fields, prints nothing, never blocks, never touches the router state, silent on any error.
Run from the repo root:  python -m unittest hooks.tests.test_route_shadow -v"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ingest.config import Config  # noqa: E402
from ingest.meshio import MeshFeedback  # noqa: E402

HOOK = ROOT / "hooks" / "route_shadow.py"
EXE = Config().mesh_exe


def trained_data():
    d = Path(tempfile.mkdtemp(prefix="shadow-test-"))
    mf = MeshFeedback(Config(data=d))
    for i in range(10):
        for ag, feats in (("fr-tester", ["acceptance", "tests", "black", "box"]), ("fr-reviewer", ["review", "audit", "diff", "security"]),
                          ("general-purpose", ["research", "summarize", "papers"])):
            mf.feed({"id": f"o:{ag}{i}", "agent": ag, "outcome": "success", "signals": [], "reward": 1.0, "features": feats, "project": "p"})
    return d


def payload(subagent="fr-tester", desc="Run acceptance tests", prompt="black box tests for the API", tool="Agent", tid="toolu_1"):
    return {"session_id": "sess-9", "cwd": "C:/x", "hook_event_name": "PreToolUse", "tool_name": tool, "tool_use_id": tid,
            "tool_input": {"subagent_type": subagent, "description": desc, "prompt": prompt}}


def run(data_dir, pl=None, raw=None, env_extra=None):
    # pin the router: the user env may set MESH_ROUTER=v2 live; these tests exercise the default (v1) path
    env = dict(os.environ, INGEST_DATA=str(data_dir), MESH_ROUTER="v1")
    env.update(env_extra or {})
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-S", str(HOOK)], input=raw if raw is not None else json.dumps(pl), capture_output=True, text=True,
                       encoding="utf-8", env=env, timeout=30)
    return r, time.perf_counter() - t


def log_rows(d):
    p = Path(d) / "route_shadow.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


@unittest.skipUnless(EXE.exists(), "neural-mesh.exe not built")
class ShadowHook(unittest.TestCase):
    def test_logs_actual_mesh_and_rule_prints_nothing_and_leaves_state_untouched(self):
        d = trained_data()
        state = d / "mesh" / "router_state.json"
        before = hashlib.sha256(state.read_bytes()).hexdigest()
        r, _ = run(d, payload())
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        rows = log_rows(d)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual((row["actual"], row["session"], row["tool_use_id"]), ("fr-tester", "sess-9", "toolu_1"))
        self.assertEqual(row["mesh_pick"], "fr-tester")           # the mesh learned acceptance/tests -> fr-tester
        self.assertEqual(row["rule_pick"], "fr-tester")           # first phase keyword: "acceptance"
        self.assertEqual(row["n_cand"], 3)
        self.assertEqual(hashlib.sha256(state.read_bytes()).hexdigest(), before)
        self.assertNotIn("prompt", json.dumps(row))                # no prompt text in the log

    def test_hybrid_pick_uses_the_project_state_and_is_null_for_an_unknown_project(self):
        d = trained_data()
        pl = payload()
        pl["cwd"] = "C:/p"                                           # project "p" has its own state (fed by trained_data)
        r, _ = run(d, pl)
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        self.assertEqual(log_rows(d)[-1]["hybrid_pick"], "fr-tester")
        self.assertIn("hybrid_ms", log_rows(d)[-1])
        r, _ = run(d, payload(tid="toolu_2"))                       # cwd C:/x: no state for project "x"
        self.assertIsNone(log_rows(d)[-1]["hybrid_pick"])

    def test_candidates_are_the_agents_seen_in_the_state(self):
        d = trained_data()
        run(d, payload(subagent="fr-reviewer", desc="Review the security diff", prompt="audit the diff"))
        row = log_rows(d)[0]
        self.assertEqual((row["mesh_pick"], row["rule_pick"], row["actual"]), ("fr-reviewer", "fr-reviewer", "fr-reviewer"))

    def test_task_tool_name_is_accepted_and_other_tools_are_ignored(self):
        d = trained_data()
        run(d, payload(tool="Task", tid="a"))
        run(d, payload(tool="Bash", tid="b"))
        run(d, payload(tool="Read", tid="c"))
        self.assertEqual([r["tool_use_id"] for r in log_rows(d)], ["a"])

    def test_missing_subagent_type_defaults_to_general_purpose(self):
        d = trained_data()
        pl = payload()
        del pl["tool_input"]["subagent_type"]
        run(d, pl)
        self.assertEqual(log_rows(d)[0]["actual"], "general-purpose")

    def test_disabled_by_env(self):
        d = trained_data()
        r, dt = run(d, payload(), env_extra={"NEURAL_MESH_SHADOW": "off"})
        self.assertEqual((r.returncode, r.stdout, log_rows(d)), (0, "", []))
        self.assertLess(dt, 0.5)

    def test_garbage_stdin_is_silent(self):
        d = trained_data()
        for raw in ["not json", "", "[]", '{"tool_name": "Agent"}', '{"tool_name": "Agent", "tool_input": "x"}']:
            r, _ = run(d, raw=raw)
            self.assertEqual((r.returncode, r.stdout), (0, ""), raw)

    def test_missing_binary_still_logs_with_null_mesh_pick(self):
        d = trained_data()
        r, _ = run(d, payload(), env_extra={"INGEST_MESH_EXE": str(d / "nope.exe")})
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        row = log_rows(d)[0]
        self.assertIsNone(row["mesh_pick"])
        self.assertEqual(row["rule_pick"], "fr-tester")

    def test_missing_state_logs_null_mesh_pick(self):
        d = Path(tempfile.mkdtemp())
        r, _ = run(d, payload())
        self.assertEqual(r.returncode, 0)
        self.assertEqual(log_rows(d)[0]["n_cand"], 0)

    def test_zero_budget_never_blocks_or_prints(self):
        d = trained_data()
        r, dt = run(d, payload(), env_extra={"NEURAL_MESH_SHADOW_BUDGET_S": "0"})
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        self.assertLess(dt, 3.0)

    def test_unwritable_log_is_silent(self):
        d = trained_data()
        (d / "route_shadow.jsonl").mkdir()                    # a directory where the file should be -> append fails
        r, _ = run(d, payload())
        self.assertEqual((r.returncode, r.stdout), (0, ""))


if __name__ == "__main__":
    unittest.main()
