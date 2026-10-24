"""Tests for ingest/router_v2.py (MiniLM + contextual bandit router) and its MESH_ROUTER=v2 switches in ingest/meshio.py and
hooks/route_shadow.py. Embeddings are the deterministic hashed stand-in (ROUTER_V2_FAKE_EMBED=1), so no model is loaded.
Run from the repo root:  python -m unittest ingest.tests.test_router_v2 -v"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

os.environ["ROUTER_V2_FAKE_EMBED"] = "1"
ROOT = Path(__file__).resolve().parents[2]
from ingest import router_v2 as V  # noqa: E402
from ingest.config import Config  # noqa: E402
from ingest.meshio import MeshFeedback, v2_state  # noqa: E402

TASKS = {"fr-tester": "run the acceptance tests black box api", "fr-reviewer": "review the diff security audit",
         "general-purpose": "research and summarize papers"}


def emb(t):
    return V.embed([t])[0]


def trained(cfg=None, n=6):
    b = V.Bandit(cfg)
    for _ in range(n):
        for a, t in TASKS.items():
            b.feedback_label(a, emb(t), "p", "")
    return b


def cli(*args, env=None):
    r = subprocess.run([sys.executable, "-m", "ingest.router_v2", *map(str, args)], capture_output=True, text=True, cwd=ROOT,
                       env=dict(os.environ, ROUTER_V2_FAKE_EMBED="1", **(env or {})), timeout=60)
    return r.returncode, json.loads(r.stdout)


class Bandit(unittest.TestCase):
    def test_learns_each_agent_from_label_feed(self):
        b = trained()
        for a, t in TASKS.items():
            self.assertEqual(b.route(emb(t), "p", "", list(TASKS)), a)

    def test_pick_is_always_one_of_the_callers_candidates(self):
        b = trained()
        self.assertEqual(b.route(emb(TASKS["fr-tester"]), "p", "", ["fr-reviewer", "general-purpose"]) in ("fr-reviewer", "general-purpose"), True)
        self.assertIsNone(b.route(emb("x"), "p", "", []))

    def test_untried_agent_never_beats_a_well_evidenced_agent_on_uncertainty_alone(self):
        for beta in (0.05, 0.2, 0.5):
            b = trained(replace(V.Config(), beta=beta))
            e = emb(TASKS["fr-tester"])
            self.assertEqual(b.route(e, "p", "", ["brand-new-agent", "fr-tester"]), "fr-tester", beta)
            sc = b.scores(e, "p", "", ["brand-new-agent", "fr-tester"])
            # the untried agent's optimism is capped at kappa x the evidenced agent's sd: its edge is at most beta*(kappa-1)*sd_min
            edge = sc["brand-new-agent"]["ucb"] - sc["brand-new-agent"]["mean"] - (sc["fr-tester"]["ucb"] - sc["fr-tester"]["mean"])
            self.assertLessEqual(edge, beta * (b.cfg.kappa - 1) * sc["fr-tester"]["sd"] + 1e-12)

    def test_the_cap_is_what_stops_uncertainty_from_winning(self):
        b = trained(replace(V.Config(), beta=5.0, kappa=1e9))      # uncapped LinUCB: the untried agent's prior sd wins
        self.assertEqual(b.route(emb(TASKS["fr-tester"]), "p", "", ["brand-new-agent", "fr-tester"]), "brand-new-agent")

    def test_untried_agent_gets_prior_mean_and_sd(self):
        sc = V.Bandit().scores(emb("anything"), "p", "physics", ["x"])["x"]
        self.assertEqual(sc["mean"], 0.0)
        self.assertAlmostEqual(sc["sd"], (1 + 0.25 + 0.5 + 0.5) ** 0.5)

    def test_label_feed_penalises_the_wrong_greedy_pick(self):
        b = trained()
        t = emb(TASKS["fr-tester"])
        n_rev = len(b.arms["fr-reviewer"].y)
        routed = b.feedback_label("fr-reviewer", t, "p", "")
        self.assertEqual(routed, "fr-tester")
        self.assertEqual(b.arms["fr-tester"].y[-1], 0.0)
        self.assertEqual((len(b.arms["fr-reviewer"].y), b.arms["fr-reviewer"].y[-1]), (n_rev + 1, 1.0))

    def test_env_failure_updates_no_skill_estimate(self):
        b = trained()
        before = {a: len(r.y) for a, r in b.arms.items()}
        self.assertEqual(V.apply_outcome(b, {"agent": "fr-tester", "outcome": "re-dispatch", "signals": ["turn_limit"], "kind": "dispatch"}), "env")
        self.assertEqual({a: len(r.y) for a, r in b.arms.items()}, before)
        self.assertEqual(b.env_failures, {"fr-tester": 1})

    def test_failed_outcome_is_a_reward_not_a_label(self):
        b = V.Bandit()
        k = V.apply_outcome(b, {"agent": "fr-builder", "outcome": "defect", "reward": 0.2, "kind": "dispatch", "task_text": "build it"})
        self.assertEqual((k, b.arms["fr-builder"].y), ("reward", [0.2]))

    def test_incremental_inverse_matches_a_full_refit(self):
        b = trained(n=10)
        arm = b.arms["general-purpose"]
        e = emb("research and summarize new papers")
        mu, sd = arm.predict(e, "p", "", b.cfg)
        arm._inv = None
        mu2, sd2 = arm.predict(e, "p", "", b.cfg)
        self.assertAlmostEqual(mu, mu2, places=8)
        self.assertAlmostEqual(sd, sd2, places=8)

    def test_json_roundtrip_keeps_predictions(self):
        b = trained()
        b2 = V.Bandit.from_json(json.loads(json.dumps(b.to_json())))
        e = emb(TASKS["fr-reviewer"])
        for a in TASKS:
            self.assertAlmostEqual(b.arms[a].predict(e, "p", "", b.cfg)[0], b2.arms[a].predict(e, "p", "", b2.cfg)[0], places=2)

    def test_field_and_project_features(self):
        self.assertEqual(V.field_of(["field_physics", "gas"]), "physics")
        self.assertEqual(V.field_of(["gas"], "math"), "math")
        self.assertEqual(V.task_text(None, ["a", "_none", "b"]), "a b")
        self.assertEqual(V.task_text("real text", ["a"]), "real text")

    def test_observation_cap_drops_the_oldest(self):
        b = V.Bandit(replace(V.Config(), cap=3))
        for i in range(5):
            b.update("a", emb(f"t{i}"), "p", "", i / 10)
        self.assertEqual(b.arms["a"].y, [0.2, 0.3, 0.4])
        self.assertEqual(b.arms["a"].E.shape, (3, V.DIM))


class Cli(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="v2-cli-"))
        self.st = self.d / "router_state.v2.json"

    def test_feedback_then_route_and_route_is_read_only(self):
        for _ in range(4):
            for a, t in TASKS.items():
                rc, out = cli("feedback", "--state", self.st, "--text", t, "--features", "x", "--label", a, "--store", "S", "--world", "w")
                self.assertEqual((rc, out["ok"]), (0, True))
        self.assertIn("--store", out["ignored"])
        h = hashlib.sha256(self.st.read_bytes()).hexdigest()
        rc, out = cli("route", "--state", self.st, "--features", "x", "--text", TASKS["fr-reviewer"], "--candidates", "fr-tester,fr-reviewer",
                      "--seed", "3", "--prior", "fr-tester", "--prior-k", "500")
        self.assertEqual((rc, out["agent"], out["router"], out["n_cand"]), (0, "fr-reviewer", "v2", 2))
        self.assertEqual(hashlib.sha256(self.st.read_bytes()).hexdigest(), h)

    def test_route_without_text_embeds_the_keyword_bag(self):
        for _ in range(3):
            cli("feedback", "--state", self.st, "--features", "acceptance,tests", "--label", "fr-tester")
            cli("feedback", "--state", self.st, "--features", "review,audit", "--label", "fr-reviewer")
        self.assertEqual(cli("route", "--state", self.st, "--features", "review,audit")[1]["agent"], "fr-reviewer")

    def test_reward_and_env_feedback(self):
        rc, out = cli("feedback", "--state", self.st, "--features", "x", "--agent", "a", "--reward", "0.3")
        self.assertEqual((rc, out["kind"]), (0, "reward"))
        rc, out = cli("feedback", "--state", self.st, "--features", "x", "--agent", "a", "--reward", "0", "--failure-class", "env")
        self.assertEqual((rc, out["kind"]), (0, "env"))
        self.assertEqual(cli("stats", "--state", self.st)[1]["agents"]["a"]["n"], 1)

    def test_refuses_to_write_the_v1_state(self):
        v1 = self.d / "router_state.json"
        rc, out = cli("feedback", "--state", v1, "--features", "x", "--label", "a")
        self.assertEqual((rc, out["ok"]), (1, False))
        self.assertFalse(v1.exists())
        v1.write_text('{"version": 3, "agents": []}', encoding="utf-8")
        other = self.d / "other.json"
        other.write_text('{"version": 3}', encoding="utf-8")
        self.assertFalse(cli("feedback", "--state", other, "--features", "x", "--label", "a")[1]["ok"])

    def test_rebuild_writes_a_new_file_only(self):
        rows = [{"id": f"o{i}", "ts": f"2026-10-0{1 + i % 9}T00:00:{i:02d}Z", "kind": "dispatch", "agent": a, "outcome": "success", "reward": 1.0,
                 "signals": [], "features": [], "task_text": t, "project": "p"} for i, (a, t) in enumerate(list(TASKS.items()) * 3)]
        rows.append({"id": "c1", "ts": "2026-10-09T00:00:00Z", "kind": "continuation", "agent": "x", "outcome": "success", "task_text": "t"})
        src = self.d / "outcomes.jsonl"
        src.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        rc, out = cli("rebuild", "--outcomes", src, "--out", self.st, "--beta", "0.2")
        self.assertEqual((rc, out["rows"], out["kinds"], out["config"]["beta"]), (0, 9, {"label": 9}, 0.2))
        self.assertNotIn("x", V.load(self.st).arms)
        rc, out = cli("rebuild", "--outcomes", src, "--out", self.d / "router_state.json")
        self.assertEqual(rc, 1)


class MeshioSwitch(unittest.TestCase):
    def outcome(self, i, kind="dispatch"):
        return {"id": f"out:{i}", "kind": kind, "agent": "fr-tester", "outcome": "success", "signals": [], "reward": 1.0,
                "features": ["acceptance"], "task_text": "run acceptance tests", "project": "p"}

    def test_default_is_v1_only(self):
        cfg = Config(data=Path(tempfile.mkdtemp()))
        with mock.patch.dict(os.environ, {"MESH_ROUTER": ""}):
            mf = MeshFeedback(cfg)
            mf.feed(self.outcome(1))
        self.assertFalse(v2_state(cfg).exists())
        self.assertEqual(mf.stats["v2"], 0)

    def test_v2_switch_also_feeds_the_v2_state_and_skips_continuations(self):
        cfg = Config(data=Path(tempfile.mkdtemp()))
        cfg.mesh_exe = cfg.data / "missing.exe"          # the v1 feed fails here; the v2 feed must still happen
        with mock.patch.dict(os.environ, {"MESH_ROUTER": "v2"}):
            mf = MeshFeedback(cfg)
            mf.feed(self.outcome(1))
            mf.feed(self.outcome(2, kind="continuation"))
        self.assertEqual(v2_state(cfg).name, "router_state.v2.json")
        self.assertEqual((mf.stats["v2"], mf.stats["v2_failed"]), (1, 0))
        self.assertEqual(len(V.load(v2_state(cfg)).arms["fr-tester"].y), 1)
        self.assertFalse(cfg.router_state.exists())


class RouteShadowSwitch(unittest.TestCase):
    def test_v2_row_uses_the_v2_state(self):
        d = Path(tempfile.mkdtemp(prefix="v2-shadow-"))
        st = d / "mesh" / "router_state.v2.json"
        b = trained()
        V.save(b, st)
        h = hashlib.sha256(st.read_bytes()).hexdigest()
        pl = {"session_id": "s", "cwd": "C:/x/p", "tool_name": "Agent", "tool_use_id": "t1",
              "tool_input": {"subagent_type": "fr-reviewer", "description": "review the diff", "prompt": "security audit"}}
        env = dict(os.environ, INGEST_DATA=str(d), MESH_ROUTER="v2", ROUTER_V2_FAKE_EMBED="1", NEURAL_MESH_SHADOW_BUDGET_S="20")
        r = subprocess.run([sys.executable, "-S", str(ROOT / "hooks" / "route_shadow.py")], input=json.dumps(pl), capture_output=True,
                           text=True, env=env, timeout=60)
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        row = json.loads((d / "route_shadow.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual((row["router"], row["mesh_pick"], row["n_cand"]), ("v2", "fr-reviewer", 3))
        self.assertEqual(hashlib.sha256(st.read_bytes()).hexdigest(), h)


if __name__ == "__main__":
    unittest.main()