"""Pairwise judge tests (no network, no GPU): order randomisation, both-order agreement, content-hash cache, hard budget stop,
usage accounting, the $0 structural pre-check, calibration verdict. Run: llm/.venv/Scripts/python -m unittest discover -s llm/tests -t ."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ingest import structural
from llm import calibrate, pairjudge as PJ

USAGE = {"input": 100, "cache_creation": 10, "cache_read": 5, "output": 20}


class Fake:
    """stands in for `claude -p`: verdict_fn(prompt) -> 'A' | 'B' | 'tie'. Records every prompt it is called with."""

    def __init__(self, verdict_fn):
        self.fn, self.prompts = verdict_fn, []

    def __call__(self, prompt, **_):
        self.prompts.append(prompt)
        return {"text": json.dumps({"verdict": self.fn(prompt), "reason": "r"}), "ok": True, "usage": dict(USAGE), "cost_usd": 0.001,
                "resolved_model": "claude-haiku-4-5-test", "err": None}


def prefers(marker):
    """a perfect judge: picks whichever answer position contains `marker`."""
    def fn(prompt):
        a = prompt.split("ANSWER A:\n")[1].split("\n\nANSWER B:")[0]
        b = prompt.split("ANSWER B:\n")[1].split("\n\nJSON verdict:")[0]
        return "A" if marker in a and marker not in b else ("B" if marker in b and marker not in a else "tie")
    return fn


class Env:
    def __init__(self, cap=100, fn=None, precheck=False, seed=0):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.fake = Fake(fn or prefers("GOOD"))
        self.ledger = PJ.Ledger(path=self.dir / "ledger.json", kind="t", cap=cap, today="2026-01-01")
        self.judge = PJ.PairwiseJudge(self.ledger, cache_path=self.dir / "cache.jsonl", runner=self.fake, model="haiku", seed=seed, precheck=precheck)

    def new_judge(self, **kw):
        return PJ.PairwiseJudge(self.ledger, cache_path=self.dir / "cache.jsonl", runner=self.fake, model="haiku", precheck=False, **kw)


class Order(unittest.TestCase):
    def test_both_orders_are_always_shown_and_the_first_one_is_randomised(self):
        e = Env(cap=1000)
        firsts = []
        for i in range(60):
            e.fake.prompts.clear()
            r = e.judge.compare(f"q{i}", "ref", f"GOOD answer {i}", f"bad answer {i}")
            self.assertEqual(r["winner"], "x")
            self.assertEqual(r["how"], "both-orders")
            self.assertEqual(len(e.fake.prompts), 2)
            pos = ["A" if "GOOD" in p.split("ANSWER A:\n")[1].split("\n\nANSWER B:")[0] else "B" for p in e.fake.prompts]
            self.assertEqual(sorted(pos), ["A", "B"])                      # swapped between the two calls
            firsts.append(pos[0])
        self.assertTrue(20 <= firsts.count("A") <= 40, firsts.count("A"))   # a fair coin over 60 pairs, not a fixed order

    def test_coin_is_deterministic_per_content_and_seed(self):
        k = PJ.pair_key("q", "r", "x", "y")
        self.assertEqual(PJ._coin(0, k), PJ._coin(0, k))
        self.assertNotEqual({PJ._coin(s, k) for s in range(40)}, {0})       # the seed can flip it

    def test_position_biased_judge_cannot_produce_a_win(self):
        e = Env(fn=lambda p: "A")                                          # always says "A better" whatever the content
        res = [e.judge.compare(f"q{i}", "r", f"GOOD {i}", f"bad {i}") for i in range(20)]
        self.assertTrue(all(r["winner"] == "tie" and r["how"] == "order-disagree" for r in res))
        s = PJ.summarize(res)
        self.assertEqual((s["x_wins"], s["y_wins"], s["ties"]), (0, 0, 20))
        self.assertEqual(s["position_bias"], 1.0)

    def test_win_needs_both_orders_and_tie_vs_win_is_a_tie(self):
        d = PJ.PairwiseJudge.decide
        self.assertEqual(d([{"who": "x"}, {"who": "x"}])["winner"], "x")
        self.assertEqual(d([{"who": "y"}, {"who": "y"}])["winner"], "y")
        self.assertEqual(d([{"who": "x"}, {"who": "tie"}])["winner"], "tie")
        self.assertEqual(d([{"who": "x"}, {"who": "y"}])["how"], "order-disagree")
        self.assertEqual(d([{"who": "x"}, {"who": None}])["how"], "error")


class Cache(unittest.TestCase):
    def test_second_identical_pair_costs_nothing_even_in_a_new_process(self):
        e = Env()
        e.judge.compare("q", "r", "GOOD", "bad")
        n = len(e.fake.prompts)
        self.assertEqual(n, 2)
        e.judge.compare("q", "r", "GOOD", "bad")
        self.assertEqual(len(e.fake.prompts), n)
        j2 = e.new_judge()                                                  # fresh object, same cache file
        r = j2.compare("q", "r", "GOOD", "bad")
        self.assertEqual((len(e.fake.prompts), r["fresh"], r["winner"]), (n, 0, "x"))
        self.assertEqual(j2.stats["cache_hits"], 2)
        self.assertEqual(e.ledger.used(), 2)                                # cached pairs never touch the budget

    def test_key_changes_with_any_content_model_or_rubric(self):
        e = Env()
        k = e.judge._key(PJ.build_prompt("q", "r", "a", "b"))
        self.assertNotEqual(k, e.judge._key(PJ.build_prompt("q", "r", "a", "c")))
        self.assertNotEqual(k, e.judge._key(PJ.build_prompt("q2", "r", "a", "b")))
        other = PJ.PairwiseJudge(e.ledger, cache_path=e.dir / "c2.jsonl", runner=e.fake, model="opus")
        self.assertNotEqual(k, other._key(PJ.build_prompt("q", "r", "a", "b")))

    def test_failed_calls_are_not_cached(self):
        e = Env()
        e.judge.runner = lambda prompt, **_: {"text": "not json", "ok": True, "usage": dict(USAGE), "cost_usd": 0, "resolved_model": None, "err": None}
        r = e.judge.compare("q", "r", "GOOD", "bad")
        self.assertEqual((r["winner"], r["how"]), ("unjudged", "error"))
        self.assertEqual(e.judge.cache, {})
        self.assertEqual(e.judge.stats["failed"], 2)


class Budget(unittest.TestCase):
    def test_hard_stop_never_exceeds_cap_and_never_half_spends_a_pair(self):
        e = Env(cap=5)
        res = [e.judge.compare(f"q{i}", "r", f"GOOD {i}", f"bad {i}") for i in range(6)]
        self.assertEqual([r["winner"] for r in res], ["x", "x", "unjudged", "unjudged", "unjudged", "unjudged"])
        self.assertEqual(len(e.fake.prompts), 4)                            # 5th call would have been half a pair: not made
        self.assertEqual(e.ledger.used(), 4)
        self.assertEqual(e.judge.stats["unjudged_budget"], 4)
        self.assertEqual(PJ.summarize(res)["unjudged"], 4)
        self.assertIsNone(PJ.score_of(res[-1]))

    def test_budget_persists_across_processes_and_resets_per_day(self):
        e = Env(cap=4)
        e.judge.compare("q0", "r", "GOOD", "bad")
        e.judge.compare("q1", "r", "GOOD", "bad")
        again = PJ.Ledger(path=e.ledger.path, kind="t", cap=4, today="2026-01-01")
        self.assertEqual((again.used(), again.remaining(), again.reserve(1)), (4, 0, False))
        self.assertEqual(PJ.Ledger(path=e.ledger.path, kind="t", cap=4, today="2026-01-02").remaining(), 4)
        self.assertEqual(PJ.Ledger(path=e.ledger.path, kind="other", cap=4, today="2026-01-01").remaining(), 4)

    def test_parallel_workers_respect_the_cap(self):
        e = Env(cap=20)
        res = e.judge.compare_many([(f"q{i}", "r", f"GOOD {i}", f"bad {i}") for i in range(30)], workers=6)
        self.assertEqual(len(e.fake.prompts), 20)
        self.assertEqual(sum(r["winner"] == "unjudged" for r in res), 20)

    def test_token_usage_is_recorded_from_usage_fields(self):
        e = Env()
        e.judge.compare("q", "r", "GOOD", "bad")
        self.assertEqual(e.judge.usage["input"], 200)
        self.assertEqual(e.judge.usage["output"], 40)
        self.assertEqual(e.judge.resolved_model, "claude-haiku-4-5-test")
        snap = e.ledger.snapshot()
        self.assertEqual((snap["calls"], snap["input"], snap["cache_creation"], snap["cache_read"], snap["output"]), (2, 200, 20, 10, 40))
        self.assertAlmostEqual(snap["cost_usd"], 0.002)


class Structural(unittest.TestCase):
    def test_check_codes(self):
        c = structural.check
        self.assertEqual(c(""), "empty")
        self.assertEqual(c("  \n"), "empty")
        self.assertEqual(c("I'm sorry, but I can't help with that."), "refusal")
        self.assertEqual(c("As an AI language model I do not have opinions."), "refusal")
        self.assertEqual(c("Here is the code:\n```python\ndef f(x):\n    return x"), "truncation")
        self.assertEqual(c("the same line repeated here\n" * 8), "repetition")
        self.assertEqual(c("one two three four five six " * 12), "repetition")
        self.assertEqual(c("not json", expect="json"), "parse")
        self.assertEqual(c("```python\ndef f(:\n  pass\n```", expect="python"), "parse")
        self.assertIsNone(c('{"a": 1}', expect="json"))
        self.assertIsNone(c("```python\ndef f(x):\n    return x\n```", expect="python"))
        self.assertIsNone(c("Fixed the bug in api/auth.ts:42; 31 tests pass. I can't verify the deploy, which needs prod access."))
        self.assertIsNone(c("Sorry about the wait. " + "The migration is applied and verified with 12 passing tests. " * 3))   # not a short refusal
        self.assertEqual(structural.expect_of("see ```python\nx=1\n```"), "python")

    def test_junk_never_costs_a_call(self):
        e = Env(precheck=True)
        r = e.judge.compare("q", "ref", "", "A real, reasoned answer with detail.")
        self.assertEqual((r["winner"], r["how"], r["fresh"]), ("y", "structural", 0))
        r2 = e.judge.compare("q", "ref", "good answer here", "I'm sorry, I cannot help.")
        self.assertEqual(r2["winner"], "x")
        r3 = e.judge.compare("q", "ref", "", "  ")
        self.assertEqual(r3["winner"], "tie")
        self.assertEqual(e.fake.prompts, [])
        self.assertEqual((e.ledger.used(), e.judge.stats["structural"]), (0, 3))

    def test_precheck_off_sends_everything_to_the_judge(self):
        e = Env(precheck=False)
        e.judge.compare("q", "ref", "", "GOOD answer")
        self.assertEqual(len(e.fake.prompts), 2)

    def test_ingest_local_judge_prefilters_before_the_model(self):
        from ingest.judge import Judge
        with tempfile.TemporaryDirectory() as d:
            j = Judge(Path(d) / "j.jsonl", "http://127.0.0.1:9", "m", budget=5)
            g = j.grade("t1", "do the thing", "")
            self.assertEqual((g["score"], g["model"], j.stats["prefiltered"], j.stats["unavailable"]), (0.0, "structural", 1, 0))
            self.assertEqual(j.budget, 5)
            self.assertFalse((Path(d) / "j.jsonl").exists())


class Parse(unittest.TestCase):
    def test_verdicts(self):
        p = PJ.parse_verdict
        self.assertEqual(p('{"verdict": "A", "reason": "x"}')[0], "A")
        self.assertEqual(p('```json\n{"verdict": "b", "reason": "x"}\n```')[0], "B")
        self.assertEqual(p('{"verdict": "TIE"}')[0], "tie")
        self.assertEqual(p('sure! {"verdict":"B","reason":"cites file"} done')[0], "B")
        self.assertIsNone(p("A is better"))
        self.assertIsNone(p('{"verdict": "C"}'))
        self.assertIsNone(p(""))

    def test_call_claude_uses_the_bench_flags_and_reads_usage(self):
        out = {"result": '{"verdict":"A"}', "stop_reason": "end_turn", "total_cost_usd": 0.0007,
               "usage": {"input_tokens": 440, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 4, "output_tokens": 55},
               "modelUsage": {"claude-haiku-4-5-20251001": {}}}
        with mock.patch("subprocess.run") as run:
            run.return_value = mock.Mock(stdout=json.dumps(out), stderr="")
            r = PJ.call_claude("p", model="haiku", think="1024")
        cmd, kw = run.call_args.args[0], run.call_args.kwargs
        self.assertEqual(cmd[cmd.index("--model") + 1], "haiku")
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
        self.assertEqual(cmd[cmd.index("--tools") + 1], "")
        self.assertEqual(kw["env"]["MAX_THINKING_TOKENS"], "1024")
        self.assertIn("--strict-mcp-config", cmd)
        self.assertEqual((r["ok"], r["usage"], r["resolved_model"]), (True, {"input": 440, "cache_creation": 3, "cache_read": 4, "output": 55}, "claude-haiku-4-5-20251001"))


class Calibration(unittest.TestCase):
    ok = {"n_judged": 44, "rate": 0.95}

    def test_verdict_thresholds(self):
        d, t = self.ok, {"n_judged": 30, "rate": 0.9}
        self.assertTrue(calibrate.verdict(d, t)[0])
        self.assertFalse(calibrate.verdict({"n_judged": 44, "rate": 0.89}, t)[0])
        self.assertFalse(calibrate.verdict({"n_judged": 39, "rate": 1.0}, t)[0])
        self.assertFalse(calibrate.verdict(d, {"n_judged": 30, "rate": 0.84})[0])
        self.assertFalse(calibrate.verdict(d, {"n_judged": 10, "rate": 1.0})[0])

    def test_gate_stays_closed_without_a_passing_calibration_for_this_rubric(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "c.json"
            self.assertFalse(calibrate.trusted("haiku", p)[0])
            p.write_text(json.dumps({"passed": True, "rubric": PJ.RUBRIC_VERSION, "model": "haiku", "reason": "x"}))
            self.assertTrue(calibrate.trusted("haiku", p)[0])
            self.assertFalse(calibrate.trusted("sonnet", p)[0])
            p.write_text(json.dumps({"passed": True, "rubric": "old", "model": "haiku"}))
            self.assertFalse(calibrate.trusted("haiku", p)[0])
            p.write_text(json.dumps({"passed": False, "rubric": PJ.RUBRIC_VERSION, "model": "haiku", "reason": "decoy 0.5"}))
            self.assertFalse(calibrate.trusted("haiku", p)[0])

    def test_run_pairs_counts_a_hit_only_when_both_orders_pick_the_real_answer(self):
        e = Env()
        pairs = [{"id": str(i), "question": f"q{i}", "reference": "r", "real": f"GOOD {i}", "decoy": f"bad {i}"} for i in range(10)]
        r = calibrate.run_pairs(e.judge, pairs)
        self.assertEqual((r["n_judged"], r["hits"], r["rate"]), (10, 10, 1.0))
        bad = Env(fn=lambda p: "A")
        r2 = calibrate.run_pairs(bad.judge, pairs)
        self.assertEqual((r2["hits"], r2["position_bias"]), (0, 1.0))


if __name__ == "__main__":
    unittest.main()
