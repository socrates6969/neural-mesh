"""Tests for the router shadow-mode statistics (ingest/shadow.py) and the nightly feed policy (ingest/meshio.py). No network."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ingest.config import Config
from ingest.meshio import MeshFeedback, feedback_args, is_env_failure, rebuild_router
from ingest.shadow import MIN_N, mcnemar_exact_p, report, rule_pick, summarize, wilson

EXE = Config().mesh_exe


def rows(n, mesh_right, rule_right, actual="fr-builder"):
    """n in-domain rows; the first `mesh_right` have a right mesh pick, the LAST `rule_right` a right rule pick (overlap is whatever falls out)."""
    out = []
    for i in range(n):
        out.append({"ts": f"2026-10-10T00:{i // 60:02d}:{i % 60:02d}Z", "actual": actual, "tool_use_id": f"t{i}",
                    "mesh_pick": actual if i < mesh_right else "fr-tester", "rule_pick": actual if i >= n - rule_right else "fr-tester"})
    return out


class RuleAndStats(unittest.TestCase):
    def test_rule_first_keyword_wins_and_defaults_to_builder(self):
        self.assertEqual(rule_pick("Review the diff :: then fix it"), "fr-reviewer")
        self.assertEqual(rule_pick("Write the docs for the API"), "fr-scriber")
        self.assertEqual(rule_pick("fix the build"), "fr-builder")
        self.assertEqual(rule_pick("Run the acceptance tests"), "fr-tester")
        self.assertEqual(rule_pick("something with no phase word at all"), "fr-builder")
        self.assertEqual(rule_pick(""), "fr-builder")

    def test_wilson_known_values(self):
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 0.4038, places=3)
        self.assertAlmostEqual(hi, 0.5962, places=3)
        self.assertEqual(wilson(0, 0)[0] != wilson(0, 0)[0], True)      # NaN for n=0
        lo, hi = wilson(100, 100)
        self.assertAlmostEqual(hi, 1.0, places=6)
        self.assertGreater(lo, 0.96)

    def test_mcnemar_exact(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact_p(10, 0), 2 / 1024, places=6)
        self.assertEqual(mcnemar_exact_p(5, 5), 1.0)


class PromotionRule(unittest.TestCase):
    def test_eligible_when_mesh_lower_bound_beats_rule_accuracy_with_enough_n(self):
        s = summarize(rows(120, 108, 72))                    # mesh .90 (lower ~.83) vs rule .60
        self.assertEqual(s["verdict"], "ELIGIBLE")
        self.assertEqual(s["in_domain"], 120)

    def test_not_eligible_below_min_n_even_if_mesh_is_perfect(self):
        s = summarize(rows(MIN_N - 1, MIN_N - 1, 10))
        self.assertEqual(s["verdict"], "NOT ELIGIBLE")
        self.assertIn("< 100", s["reasons"][0])

    def test_not_eligible_when_lower_bound_does_not_clear_the_rule(self):
        s = summarize(rows(120, 84, 78))                      # mesh .70 (lower ~.61) vs rule .65
        self.assertEqual(s["verdict"], "NOT ELIGIBLE")
        self.assertIn("not above", s["reasons"][0])

    def test_out_of_domain_dispatches_do_not_count_towards_n(self):
        r = rows(60, 60, 0) + rows(80, 80, 0, actual="general-purpose")
        s = summarize(r)
        self.assertEqual((s["dispatches_logged"], s["in_domain"], s["verdict"]), (140, 60, "NOT ELIGIBLE"))

    def test_missing_mesh_pick_counts_as_a_miss(self):
        r = rows(100, 100, 0)
        for x in r[:30]:
            x["mesh_pick"] = None
        s = summarize(r)
        self.assertEqual(s["in_domain_mesh"]["correct"], 70)
        self.assertEqual(s["mesh_pick_missing"], 30)

    def test_report_dedupes_tool_use_ids_and_drops_rows_without_actual(self):
        d = Path(tempfile.mkdtemp())
        base = rows(5, 5, 0)
        lines = base + [dict(base[0])] + [{"ts": "x", "mesh_pick": "a"}]
        (d / "route_shadow.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines) + "{torn", encoding="utf-8")
        s = report(Config(data=d))
        self.assertEqual(s["dispatches_logged"], 5)


class FeedPolicy(unittest.TestCase):
    def out(self, outcome, signals=(), reward=1.0, agent="fr-builder"):
        return {"id": "out:1", "agent": agent, "outcome": outcome, "signals": list(signals), "reward": reward, "features": ["fix", "bug"], "project": "p"}

    def test_env_failures_use_failure_class_env_with_reward_zero(self):
        for sig in ("turn_limit", "redispatch_cue", "api_error"):
            o = self.out("re-dispatch", [sig], 0.1)
            self.assertTrue(is_env_failure(o))
            self.assertEqual(feedback_args(o), ["--agent", "fr-builder", "--reward", "0", "--failure-class", "env"])

    def test_non_env_redispatch_and_other_outcomes_are_labels(self):
        for o in (self.out("re-dispatch", ["redispatched_later"], 0.1), self.out("success"), self.out("defect", ["defect_cue"], 0.4),
                  self.out("fix-round", ["fix_task"], 0.6)):
            self.assertFalse(is_env_failure(o))
            self.assertEqual(feedback_args(o), ["--label", "fr-builder"])

    def test_env_signal_on_a_success_is_not_an_env_failure(self):
        self.assertFalse(is_env_failure(self.out("success", ["api_error"])))

    def test_reward_mode_restores_legacy_feed(self):
        self.assertEqual(feedback_args(self.out("defect", reward=0.4), "reward"), ["--agent", "fr-builder", "--reward", "0.4000"])

    def test_agent_names_with_commas_are_sanitised(self):
        self.assertEqual(feedback_args(self.out("success", agent="a,b"))[-1], "a_b")

    def test_command_line_has_state_features_and_store(self):
        cfg = Config(data=Path(tempfile.mkdtemp()))
        cmd = MeshFeedback(cfg).command(self.out("success"))
        self.assertEqual(cmd[1:3], ["feedback", "--state"])
        self.assertIn("--label", cmd)
        self.assertIn("--store", cmd)
        self.assertNotIn("--reward", cmd)


FAKE_MESH = """import sys
bad = '--label' in sys.argv or '--failure-class' in sys.argv
print('{"ok":false,"error":"unknown option"}' if bad else '{"ok":true}')
sys.exit(1 if bad else 0)
"""


class StaleBinaryFallback(unittest.TestCase):
    def test_binary_without_label_support_falls_back_to_reward_feed_and_loses_nothing(self):
        d = Path(tempfile.mkdtemp())
        fake = d / "fake_mesh.py"
        fake.write_text(FAKE_MESH, encoding="utf-8")
        bat = d / "fake_mesh.bat"
        bat.write_text('@"' + sys.executable + '" "' + str(fake) + '" %*' + chr(13) + chr(10), encoding="utf-8")
        cfg = Config(data=d)
        cfg.mesh_exe = bat
        mf = MeshFeedback(cfg)
        mf.feed({"id": "out:a", "agent": "x", "outcome": "success", "signals": [], "reward": 1.0, "features": ["f"], "project": "p"})
        mf.feed({"id": "out:b", "agent": "x", "outcome": "re-dispatch", "signals": ["turn_limit"], "reward": 0.1, "features": ["f"], "project": "p"})
        self.assertEqual((mf.stats["fed"], mf.stats["failed"], mf.stats["fallback"], mf.stats["reward"]), (2, 0, 2, 2))


@unittest.skipUnless(EXE.exists(), "neural-mesh.exe not built")
class FeedAgainstRealBinary(unittest.TestCase):
    def test_label_feed_teaches_route_and_env_feed_is_accepted(self):
        d = Path(tempfile.mkdtemp())
        cfg = Config(data=d)
        mf = MeshFeedback(cfg)
        for i in range(12):
            for ag, feats in (("fr-tester", ["acceptance", "tests"]), ("fr-reviewer", ["review", "audit"])):
                o = {"id": f"out:{ag}{i}", "agent": ag, "outcome": "success", "signals": [], "reward": 1.0, "features": feats, "project": "p"}
                mf.feed(o)
        env = {"id": "out:env1", "agent": "fr-tester", "outcome": "re-dispatch", "signals": ["turn_limit"], "reward": 0.1, "features": ["acceptance"], "project": "p"}
        mf.feed(env)
        self.assertEqual(mf.stats["failed"], 0)
        self.assertEqual((mf.stats["label"], mf.stats["env"]), (24, 1))
        for feats, want in (("acceptance,tests", "fr-tester"), ("review,audit", "fr-reviewer")):
            r = subprocess.run([str(EXE), "route", "--state", str(cfg.router_state), "--features", feats, "--candidates", "fr-tester,fr-reviewer"],
                               capture_output=True, text=True)
            self.assertEqual(json.loads(r.stdout)["agent"], want)
        mf2 = MeshFeedback(cfg)
        mf2.feed(env)                                       # at-most-once
        self.assertEqual(mf2.stats["skipped"], 1)

    def test_rebuild_dry_run_leaves_live_state_alone(self):
        d = Path(tempfile.mkdtemp())
        cfg = Config(data=d)
        (d / "outcomes.jsonl").write_text("".join(json.dumps({"id": f"out:{i}", "ts": f"2026-10-0{1 + i % 9}T00:00:{i:02d}Z", "agent": "fr-builder" if i % 2 else "fr-tester",
                                                              "outcome": "success", "signals": [], "reward": 1.0, "features": ["x%d" % (i % 2)], "task_text": "build x"}) + "\n"
                                                  for i in range(8)), encoding="utf-8")
        cfg.router_state.parent.mkdir(parents=True)
        cfg.router_state.write_text("LIVE", encoding="utf-8")
        res = rebuild_router(cfg, dry_run=True, evaluate=True)
        self.assertTrue(res["ok"])
        self.assertEqual(cfg.router_state.read_text(encoding="utf-8"), "LIVE")
        res = rebuild_router(cfg)
        self.assertEqual(res["fed_ok"], 8)
        self.assertEqual(cfg.router_state.with_name("router_state.json.pre-rebuild").read_text(encoding="utf-8"), "LIVE")
        self.assertIn('"version"', cfg.router_state.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
