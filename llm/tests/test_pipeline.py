"""Unit tests that need no GPU or Ollama: split, selection, gate, stats, judge parsing, lock, Modelfile, front-end hook.
Run: llm/.venv/Scripts/python -m unittest discover -s llm/tests -t ."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from llm import config, data, evaluate, judge, stats


class Split(unittest.TestCase):
    def test_heldout_is_latest_sessions_and_disjoint(self):
        rows = data.synthetic_rows(n_sessions=10, per_session=3)
        tr, te = data.split_by_session_time(rows, 0.2)
        self.assertEqual({r["session"] for r in te}, {"syn-session-08", "syn-session-09"})
        self.assertFalse({r["session"] for r in tr} & {r["session"] for r in te})
        self.assertGreater(min(r["ts"] for r in te), max(r["ts"] for r in tr))
        self.assertEqual(data.split_by_session_time(rows[:3])[1], [])  # one session -> nothing held out

    def test_select_new_and_replay(self):
        rows = data.synthetic_rows(n_sessions=4, per_session=5)
        seen = {r["id"] for r in rows[:15]}
        new, rep = data.select_for_round(rows, seen, replay_ratio=1.0)
        self.assertEqual(len(new), 5)
        self.assertEqual(len(rep), 5)
        self.assertTrue(all(r["id"] in seen for r in rep) and not any(r["id"] in seen for r in new))
        self.assertEqual(data.select_for_round(rows, {r["id"] for r in rows}), ([], []))

    def test_meta_heldout_flag_wins_and_weighted_replay_prefers_singletons(self):
        rows = data.synthetic_rows(n_sessions=4, per_session=5)
        meta = {r["id"]: {"heldout": r["session"] == "syn-session-00", "replay_weight": 1.0} for r in rows}
        tr, te = data.split(rows, meta)
        self.assertEqual({r["session"] for r in te}, {"syn-session-00"})       # the flag, not time order
        self.assertEqual(data.split(rows, {})[1], data.split_by_session_time(rows)[1])
        old = {r["id"] for r in rows[:10]}
        w = {r["id"]: (100.0 if i == 3 else 0.001) for i, r in enumerate(rows)}
        hits = sum(data.select_for_round(rows, old, replay_ratio=0.1, seed=s, weights=w)[1][0]["id"] == rows[3]["id"] for s in range(20))
        self.assertGreaterEqual(hits, 18)

    def test_read_jsonl_skips_torn_lines_and_dups(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.jsonl"
            r = {"id": "a", "instruction": "x", "response": "y"}
            p.write_text(json.dumps(r) + "\n" + json.dumps(r) + "\n{\"id\": \"b\", \"instr", encoding="utf-8")
            self.assertEqual([x["id"] for x in data.read_jsonl(p)], ["a"])

    def test_synthetic_is_labelled(self):
        self.assertTrue(all(r["source"] == "synthetic" and r["project"] == "SYNTHETIC" for r in data.synthetic_rows()))


class Gate(unittest.TestCase):
    ci = staticmethod(lambda n, m, lo, hi: {"n": n, "mean": m, "lo": lo, "hi": hi})
    pair = staticmethod(lambda n, lo: {"n_judged": n, "winrate_ci": {"n": n, "mean": lo + .1, "lo": lo, "hi": lo + .2}})
    det_ok = {"regressed": []}

    def test_promotes_only_on_significant_pairwise_win_without_regression(self):
        g, ci, pair = evaluate.gate, self.ci, self.pair
        self.assertTrue(g(ci(50, .1, .02, .2), pair(40, .55), self.det_ok, 5)[0])
        self.assertFalse(g(ci(50, .1, .02, .2), pair(40, .50), self.det_ok, 5)[0])                 # lower bound must be > 0.5
        self.assertFalse(g(ci(50, .1, .02, .2), pair(39, .60), self.det_ok, 5)[0])                 # < 40 judged items
        self.assertFalse(g(ci(50, .1, .02, .2), pair(60, .60), {"regressed": ["code"]}, 5)[0])     # deterministic regression
        self.assertFalse(g(ci(50, -.2, -.3, -.1), pair(60, .60), self.det_ok, 5)[0])               # loss significantly worse
        self.assertFalse(g(ci(50, .1, .02, .2), pair(60, .60), self.det_ok, 2)[0])                 # too few clusters
        self.assertFalse(g(ci(50, .1, .02, .2), pair(60, .60), self.det_ok, 5, (False, "x"))[0])   # judge not calibrated: closed

    def test_deterministic_regression_rule(self):
        items = [{"cat": "math"}] * 20 + [{"cat": "code"}] * 20
        same = [True] * 40
        self.assertEqual(evaluate.det_regressions(items, same, same)["regressed"], [])
        worse = [True] * 15 + [False] * 5 + [True] * 20          # math: 5 net losses of 20 = -0.25
        self.assertEqual(evaluate.det_regressions(items, worse, same)["regressed"], ["math", "pooled"])
        better = same[:]; cand = [False] + [True] * 39             # one lost item of 20 = -0.05, sign p .5: tolerated
        self.assertEqual(evaluate.det_regressions(items, cand, better)["regressed"], [])

    def test_stats(self):
        c = stats.paired_ci([1.0, 1.1, 0.9, 1.0])
        self.assertTrue(c["lo"] > 0 and abs(c["mean"] - 1.0) < 1e-9)
        self.assertEqual(stats.paired_ci([1.0])["lo"], float("-inf"))
        b = stats.cluster_bootstrap_ci([[1.0, 1.2], [0.8], [1.1, 0.9, 1.0]], reps=500)
        self.assertTrue(b["lo"] > 0 and b["clusters"] == 3)
        self.assertEqual(stats.cluster_bootstrap_ci([[1.0, 2.0]])["lo"], float("-inf"))  # one session: never significant

    def test_sign_test(self):
        self.assertAlmostEqual(stats.sign_test_p(10, 0), 1 / 1024)
        self.assertAlmostEqual(stats.sign_test_p(5, 5), 0.623046875)
        self.assertEqual(stats.sign_test_p(0, 0), 1.0)


class Judge(unittest.TestCase):
    def test_parse_fails_closed(self):
        self.assertEqual(judge.parse_score('{"score": 1, "why": "ok"}'), 1.0)
        self.assertEqual(judge.parse_score('{"score": 0.5}'), 0.5)
        self.assertEqual(judge.parse_score('blah "score": 7 blah'), 1.0)
        self.assertEqual(judge.parse_score("no json"), 0.0)
        self.assertEqual(judge.parse_score('{"why": "x"}'), 0.0)


class Lock(unittest.TestCase):
    def test_second_holder_is_refused_and_stale_lock_is_taken_over(self):
        from llm.__main__ import lock
        with tempfile.TemporaryDirectory() as d:
            old = config.LOCK, config.MODELS
            config.MODELS = Path(d); config.LOCK = Path(d) / "x.lock"
            try:
                with lock():
                    self.assertTrue(config.LOCK.exists())
                    with self.assertRaises(SystemExit) as e:
                        with lock():
                            pass
                    self.assertEqual(e.exception.code, 75)
                self.assertFalse(config.LOCK.exists())
                config.LOCK.write_text("1 0")
                os.utime(config.LOCK, (0, 0))
                with lock():
                    pass
            finally:
                config.LOCK, config.MODELS = old


class Promote(unittest.TestCase):
    def test_modelfile_points_at_safetensors_dir(self):
        from llm.promote import modelfile
        m = modelfile(Path("C:/x/merged/v1"))
        self.assertTrue(m.startswith("FROM C:/x/merged/v1\n") and "<|im_end|>" in m and "TEMPLATE" in m)


class FrontendHook(unittest.TestCase):
    @unittest.skipUnless(__import__("llm.frontend", fromlist=["BIN"]).BIN.exists(), "neural-mesh binary not built")
    def test_rust_policy_default_off_then_local_then_ood_forward(self):
        from llm.frontend import rust_decide
        sig = {"local-conf": -0.05, "local-tokens": 30, "max-sim": 0.8}
        self.assertEqual(rust_decide({}, sig, [])["decision"], "forward")
        self.assertEqual(rust_decide({"enabled": 1, "min-conf": -0.1, "min-sim": 0.5}, sig, [])["decision"], "local")
        d = rust_decide({"enabled": 1, "min-conf": -0.1, "min-sim": 0.9}, sig, [{"id": "m", "score": 0.5, "tokens": 10}])
        self.assertEqual((d["decision"], d["context"]), ("forward", ["m"]))


if __name__ == "__main__":
    unittest.main()
