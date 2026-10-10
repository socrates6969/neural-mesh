"""SFT admission scheduling + decoy contrast (ingest/schedule.py) and the pipeline retry of budget-starved dispatches."""
import unittest

import numpy as np

from ingest import schedule as sc


class FakeJudge:
    """scores by a table keyed on (target); records the order of paid calls; budget limits paid calls (cached = free)."""
    def __init__(self, scores, budget=100):
        self.scores, self.budget, self.calls, self.cache = scores, budget, [], {}

    def grade(self, target, task, output):
        if target in self.cache:
            return self.cache[target]
        if self.budget <= 0:
            return None
        self.budget -= 1
        self.calls.append(target)
        g = {"score": self.scores.get(target, 1.0), "model": "fake", "rubric_version": "t"}
        self.cache[target] = g
        return g


class FakeVS:
    """instruction / output vectors from a dict text -> vector"""
    def __init__(self, table):
        self.table = table

    def matrix(self, texts):
        return np.stack([np.array(self.table[t], dtype="f4") for t in texts])


def unit(*v):
    a = np.array(v, dtype="f4")
    return (a / np.linalg.norm(a)).tolist()


def cand(t, s, ts, src="agent_report", instr=None, out=None):
    return sc.Cand(t, f"task {t}", instr or f"instr {t}", out or f"out {t}", s, ts, src)


class ScheduleTests(unittest.TestCase):
    def test_order_round_robin_sessions_and_sources(self):
        cs = [cand("a1", "A", "2026-10-04T01"), cand("a2", "A", "2026-10-04T02"), cand("a3", "A", "2026-10-04T03"),
              cand("t1", "A", "2026-10-04T04", "user_turn"),
              cand("b1", "B", "2026-10-09T01"), cand("c1", "C", "2026-10-10T01", "user_turn")]
        o = [c.target for c in sc.order(cs)]
        self.assertEqual(o[:3], ["c1", "b1", "t1"])         # most recent session first; one per session per round; A leads with its turn
        self.assertEqual(set(o), {"a1", "a2", "a3", "t1", "b1", "c1"})
        self.assertEqual(o.index("a3") > o.index("a2"), False)   # newest first inside a session

    def test_small_budget_spreads_over_sessions(self):
        cs = [cand(f"a{i}", "A", f"2026-10-04T0{i}") for i in range(6)] + [cand("b0", "B", "2026-10-09T00"), cand("c0", "C", "2026-10-10T00", "user_turn")]
        j = FakeJudge({}, budget=3)
        adm = sc.grade_all(j, cs, None, 0.7)
        self.assertEqual({c for c in j.calls}, {"c0", "b0", "a5"})
        self.assertTrue(adm["c0"].accepted and adm["b0"].accepted)       # no vectors -> admitted on the real grade
        self.assertIsNone(adm["a4"].accepted)                            # pending, not rejected

    def test_decoy_contrast_admits_only_when_real_beats_decoy(self):
        # q1/q2 are near requests with different answers (a decoy for each other); q3 is alone far away
        cs = [cand("q1", "A", "2026-10-04T01", instr="q1", out="o1"), cand("q2", "A", "2026-10-04T02", instr="q2", out="o2"),
              cand("q3", "A", "2026-10-04T03", instr="q3", out="o3")]
        vs = FakeVS({"q1": unit(1, 0.1, 0), "q2": unit(1, 0.2, 0), "q3": unit(0, 0, 1), "o1": unit(1, 0, 0), "o2": unit(0, 1, 0), "o3": unit(0, 0, 1)})
        # q1's decoy answer (o2) is graded 0.9 -> not rejected; q2's decoy answer (o1) graded 0.1 -> rejected
        j = FakeJudge({"q1": 1.0, "q2": 1.0, "q3": 1.0, "decoy:q1": 0.9, "decoy:q2": 0.1, "decoy:q3": 0.0})
        adm = sc.grade_all(j, cs, vs, 0.7)
        self.assertFalse(adm["q1"].accepted)
        self.assertTrue(adm["q2"].accepted)
        self.assertEqual(adm["q2"].decoy_score, 0.1)
        self.assertEqual(adm["q1"].decoy_target, "q2")

    def test_decoy_budget_shortage_leaves_row_pending_with_cached_real_grade(self):
        cs = [cand("q1", "A", "2026-10-04T01", instr="q1", out="o1"), cand("q2", "A", "2026-10-04T02", instr="q2", out="o2")]
        vs = FakeVS({"q1": unit(1, 0.1), "q2": unit(1, 0.2), "o1": unit(1, 0), "o2": unit(0, 1)})
        j = FakeJudge({}, budget=2)                                       # both real grades, no decoy call left
        adm = sc.grade_all(j, cs, vs, 0.7)
        self.assertIsNone(adm["q1"].accepted)
        self.assertIsNone(adm["q2"].accepted)
        j.budget = 2                                                      # next night: only the two decoy calls are needed (real grades cached)
        adm = sc.grade_all(j, cs, vs, 0.7)
        self.assertEqual(j.calls[2:], ["decoy:q2", "decoy:q1"])
        self.assertTrue(adm["q1"].accepted is not None and adm["q2"].accepted is not None)

    def test_paraphrase_answers_are_not_decoys(self):
        cs = [cand("q1", "A", "2026-10-04T01", instr="q1", out="o1"), cand("q2", "A", "2026-10-04T02", instr="q2", out="o2")]
        vs = FakeVS({"q1": unit(1, 0.1), "q2": unit(1, 0.2), "o1": unit(1, 0.01), "o2": unit(1, 0.02)})   # answers cos ~ 1 -> same answer, no decoy
        self.assertEqual(sc.pick_decoys(cs, vs), {})

    def test_low_real_grade_rejected_without_decoy_call(self):
        cs = [cand("q1", "A", "2026-10-04T01", instr="q1", out="o1"), cand("q2", "A", "2026-10-04T02", instr="q2", out="o2")]
        vs = FakeVS({"q1": unit(1, 0.1), "q2": unit(1, 0.2), "o1": unit(1, 0), "o2": unit(0, 1)})
        j = FakeJudge({"q1": 0.4, "q2": 0.4})
        adm = sc.grade_all(j, cs, vs, 0.7)
        self.assertFalse(adm["q1"].accepted)
        self.assertFalse(any(c.startswith("decoy:") for c in j.calls))


if __name__ == "__main__":
    unittest.main()


class PipelineRetryTests(unittest.TestCase):
    """An outcome row written while the judge had no budget must still get its SFT chance on a later run."""
    def setUp(self):
        import os
        os.environ["INGEST_EMBED_EXE"] = "Z:/missing/embed_texts.exe"
        from ingest.tests.test_pipeline import Env, build_tree
        self.env = Env()
        build_tree(self.env)

    def tearDown(self):
        import os
        os.environ.pop("INGEST_EMBED_EXE", None)
        self.env.tmp.cleanup()

    def test_sft_row_appears_on_a_later_run_with_budget(self):
        from unittest import mock
        from ingest import pipeline
        from ingest.util import iter_jsonl
        r1 = pipeline.run(self.env.cfg)                                   # judge_budget = 0 in Env: outcomes only
        self.assertEqual(len(list(iter_jsonl(self.env.cfg.sft))), 0)
        self.assertEqual(len(list(iter_jsonl(self.env.cfg.outcomes))), 2)

        class J:
            def __init__(self, *a, **k):
                self.stats = {"called": 0, "cached": 0, "failed": 0, "skipped_budget": 0, "unavailable": 0, "prefiltered": 0}

            def grade(self, target, task, output):
                return {"id": "j:" + target, "target": target, "score": 0.95, "reason": "", "model": "fake", "rubric_version": "t"}
        with mock.patch.object(pipeline, "Judge", J):
            pipeline.run(self.env.cfg)
        rows = list(iter_jsonl(self.env.cfg.sft))
        self.assertTrue(rows)                                              # the success dispatch (toolu_A) got its row although its outcome already existed
        self.assertTrue(all(r["session"] == "sess-1" for r in rows))
        self.assertEqual(len(list(iter_jsonl(self.env.cfg.outcomes))), 2)  # no duplicate outcome rows
        self.assertTrue(r1["ok"] if "ok" in r1 else True)
