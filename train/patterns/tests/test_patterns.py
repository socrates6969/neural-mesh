import tempfile
import unittest
from pathlib import Path

from train.patterns import Candidate, answer_key, best_of_n, evolve, token_novelty
from train.patterns.evolve import cosine_novelty

TMP = Path(tempfile.mkdtemp())


def stub_runner(replies):
    """Stub agent: attempt i answers replies[i] (a string, or an Exception-like dict {"error": ...})."""
    calls = []

    def run(agent, task, pack, workdir, **kw):
        i = int(Path(workdir).name[3:])
        calls.append((agent["name"], i))
        r = replies[i]
        if isinstance(r, dict):
            return {"agent": agent["name"], "text": r.get("text", ""), "error": r["error"], "cost": 0.0, "secs": 0}
        return {"agent": agent["name"], "text": r, "error": None, "cost": 0.01, "secs": 1}
    run.calls = calls
    return run


INT = {"id": "t1", "field": "math", "kind": "int", "prompt": "?", "answer": 42}
A = {"name": "a"}


class BestOfNTest(unittest.TestCase):
    def test_majority_and_agreement(self):
        run = stub_runner(["ANSWER: 42", "ANSWER: 41", "work... ANSWER: 42"])
        r = best_of_n(A, INT, 3, TMP, run_fn=run, parallel=False)
        self.assertEqual((r["answer"], r["ok"], r["agreement"], r["method"]), (42, True, 0.667, "majority"))
        self.assertTrue(r["pass_at_n"])
        self.assertEqual(r["chosen"], 0)

    def test_errors_left_out_of_vote(self):
        run = stub_runner([{"error": "quota: limit"}, "ANSWER: 7", {"error": "timeout"}])
        r = best_of_n(A, INT, 3, TMP, run_fn=run)
        self.assertEqual((r["valid"], r["answer"], r["agreement"], r["ok"]), (1, 7, 1.0, False))

    def test_all_errors(self):
        run = stub_runner([{"error": "timeout"}] * 3)
        r = best_of_n(A, INT, 3, TMP, run_fn=run)
        self.assertIsNone(r["chosen"])
        self.assertFalse(r["ok"])

    def test_numeric_tolerance_groups(self):
        task = {"id": "n", "field": "physics", "kind": "numeric", "prompt": "?", "answer": "9.81"}
        run = stub_runner(["ANSWER: 9.81", "ANSWER: 9.8", "ANSWER: 3.2"])
        r = best_of_n(A, task, 3, TMP, run_fn=run)
        self.assertEqual((r["agreement"], r["ok"]), (0.667, True))

    def test_verified_minority_beats_unverified_majority(self):
        run = stub_runner(["ANSWER: 41", "ANSWER: 41", "checked ANSWER: 42"])
        r = best_of_n(A, INT, 3, TMP, run_fn=run, verify=lambda t, text: "checked" in text)
        self.assertEqual((r["answer"], r["method"], r["verified"], r["ok"]), (42, "verified-majority", 1, True))

    def test_code_task_uses_verified_first_and_cycles_agents(self):
        task = {"id": "c", "field": "coding", "kind": "code", "prompt": "?"}
        run = stub_runner(["```python\nx=1\n```", "```python\nx=2\n```", "```python\nx=3\n```"])
        r = best_of_n([{"name": "a"}, {"name": "b"}], task, 3, TMP, run_fn=run, parallel=False,
                      verify=lambda t, text: "x=2" in text, grade_fn=lambda t, text: "x=2" in text)
        self.assertEqual((r["chosen"], r["method"], r["ok"]), (1, "verified-first", True))
        self.assertEqual([a for a, _ in run.calls], ["a", "b", "a"])

    def test_answer_key(self):
        self.assertEqual(answer_key({"kind": "mcq"}, "ANSWER: (c) and A"), "AC")
        self.assertIsNone(answer_key(INT, "no final line"))
        self.assertIsNone(answer_key({"kind": "code"}, "ANSWER: 1"))


def count_a(s):
    return s.count("a") - 0.1 * max(0, len(s) - 10)


def mutate_add_a(p, ctx):
    rng = ctx["rng"]
    s = p.content
    i = rng.randrange(len(s) + 1)
    ch = "a" if ctx["mode"] == "normal" else rng.choice("xyz")
    return s[:i] + ch + s[i:]


class EvolveTest(unittest.TestCase):
    def test_improves_toy_objective_and_is_deterministic(self):
        r1 = evolve(["bbb", "ccc"], count_a, mutate_add_a, generations=6, pop_size=6, paradigm_every=0, seed=1)
        r2 = evolve(["bbb", "ccc"], count_a, mutate_add_a, generations=6, pop_size=6, paradigm_every=0, seed=1)
        self.assertGreater(r1["best"].score, count_a("bbb"))
        self.assertEqual(r1["best"].content, r2["best"].content)
        self.assertEqual([h["best"] for h in r1["history"]], sorted(h["best"] for h in r1["history"]))

    def test_negative_memory_rejects_before_scoring(self):
        scored = []

        def scorer(s):
            scored.append(s)
            return 1.0

        def mutate(p, ctx):
            return "retry the failed cache idea now" if ctx["gen"] == 1 else p.content + " v%d" % ctx["gen"]
        r = evolve(["seed idea"], scorer, mutate, negative=["retry the failed cache idea now"], generations=1,
                   pop_size=3, elite=1, paradigm_every=0)
        self.assertGreater(r["rejected_negative"] + r["rejected_duplicate"], 0)
        self.assertNotIn("retry the failed cache idea now", scored)

    def test_fail_below_feeds_negative_memory(self):
        r = evolve(["good"], lambda s: 1.0 if s == "good" else -1.0, lambda p, ctx: "bad %d" % ctx["rng"].random(),
                   generations=1, pop_size=3, elite=1, fail_below=0.0, paradigm_every=0)
        self.assertTrue(any(n.startswith("bad") for n in r["negative"]))

    def test_paradigm_shift_bans_best_key_idea(self):
        seen = []

        def mutate(p, ctx):
            seen.append((ctx["mode"], tuple(ctx["banned"])))
            if ctx["mode"] == "paradigm_shift":
                n = len(seen)
                return ("use caching\nvariant %d" % n) if n % 2 else ("use batching\nvariant %d" % n)
            return p.content + "!"
        r = evolve(["use caching\nbase"], lambda s: len(s), mutate, generations=2, pop_size=4, elite=1,
                   paradigm_every=2)
        shifts = [b for m, b in seen if m == "paradigm_shift"]
        self.assertTrue(shifts and all(b == ("use caching",) for b in shifts))
        self.assertGreater(r["rejected_banned"], 0)
        self.assertTrue(r["history"][2]["paradigm_shift"])

    def test_novelty_bonus_prefers_novel_at_equal_score(self):
        r = evolve(["alpha beta gamma", "alpha beta gamma delta", "zeta eta theta"], lambda s: 1.0,
                   lambda p, ctx: "", generations=1, pop_size=2, elite=0, novelty_weight=1.0,
                   archive=["alpha beta gamma"], paradigm_every=0)
        self.assertIn("zeta eta theta", [c.content for c in r["population"]])

    def test_max_evals_and_target(self):
        r = evolve(["b"], count_a, mutate_add_a, generations=50, pop_size=4, max_evals=6, paradigm_every=0)
        self.assertLessEqual(r["evals"], 6)
        r = evolve(["aaa"], count_a, mutate_add_a, generations=50, pop_size=4, target=2.0, paradigm_every=0)
        self.assertEqual(len(r["history"]), 1)

    def test_novelty_functions(self):
        self.assertEqual(token_novelty("a b", []), 1.0)
        self.assertEqual(token_novelty("a b", ["b a"]), 0.0)
        nov = cosine_novelty(lambda s: [1.0, 0.0] if "x" in s else [0.0, 1.0])
        self.assertAlmostEqual(nov("x", ["x1"]), 0.0)
        self.assertAlmostEqual(nov("x", ["y"]), 1.0)
        self.assertIsInstance(Candidate("c").meta, dict)


if __name__ == "__main__":
    unittest.main()
