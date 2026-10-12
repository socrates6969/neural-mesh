"""Self-distillation anchors: rejection sampling, battery-leak guard, deterministic sampling (no GPU, no network)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm import anchors as A


class Parsing(unittest.TestCase):
    def test_final_answer_takes_the_last_answer_line(self):
        self.assertEqual(A.final_answer("ANSWER: 3\nwait\nANSWER: **42**"), "42")
        self.assertIsNone(A.final_answer("the answer is 42"))

    def test_correct_numeric_and_letter(self):
        self.assertTrue(A.correct("gsm8k", "1,200", "1200"))
        self.assertFalse(A.correct("gsm8k", "1199", "1200"))
        self.assertTrue(A.correct("mmlu", "B. 4 m/s", "B"))
        self.assertFalse(A.correct("mmlu", "C", "B"))
        self.assertFalse(A.correct("mmlu", None, "B"))


class LeakGuard(unittest.TestCase):
    def test_a_battery_prompt_leaks_and_a_new_one_does_not(self):
        bank = A.battery_shingles()
        battery = json.loads((A.config.REPO / "bench" / "battery.json").read_text(encoding="utf-8"))
        long_prompt = next(i["prompt"] for i in battery if len(i["prompt"].split()) > 20)
        self.assertTrue(A.leaks(long_prompt, bank))
        self.assertFalse(A.leaks("Natalia sold clips to 48 of her friends in April and half as many in May; how many in total?", bank))


class Build(unittest.TestCase):
    def test_keeps_only_correct_non_leaking_answers_and_is_idempotent(self):
        cands = [
            {"source": "gsm8k", "q": "Tom has 3 apples and buys 4 more. How many?", "gold": "7"},
            {"source": "gsm8k", "q": "Sara has 10 pens and gives away 4. How many left?", "gold": "6"},
            {"source": "mmlu", "q": "Which is a vector?\n\nA. mass\nB. velocity", "gold": "B"},
        ]
        answers = {"Tom": "3+4=7\nANSWER: 7", "Sara": "10-4=5\nANSWER: 5", "Which": "Velocity has direction.\nANSWER: B"}
        gen = lambda p: next(v for k, v in answers.items() if p.startswith(k))  # noqa: E731
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "path", lambda: Path(d) / "anchors.jsonl"), \
                mock.patch.object(A, "battery_shingles", lambda: set()):
            st = A.build(n=4, gen=gen, cands=cands, log=lambda *_: None)
            self.assertEqual((st["kept_new"], st["wrong"]), (2, 1))   # Sara's wrong answer is rejected
            rows = A.load()
            self.assertEqual({r["gold"] for r in rows}, {"7", "B"})
            self.assertTrue(all(r["instruction"].endswith(A.FORMAT) and r["license"].startswith("MIT") for r in rows))
            calls = []
            A.build(n=4, gen=lambda p: calls.append(p) or "ANSWER: 0", cands=cands[:1], log=lambda *_: None)
            self.assertEqual(calls, [])                                # already kept: not regenerated

    def test_leaking_candidate_is_never_generated(self):
        cands = [{"source": "gsm8k", "q": "one two three four five six seven eight nine", "gold": "1"}]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(A, "path", lambda: Path(d) / "anchors.jsonl"), \
                mock.patch.object(A, "battery_shingles", lambda: A._shingles("one two three four five six seven eight nine")):
            st = A.build(n=2, gen=lambda p: self.fail("generated a leaking prompt"), cands=cands, log=lambda *_: None)
            self.assertEqual(st["leak"], 1)

    def test_sample_is_deterministic_and_bounded(self):
        rows = [{"id": f"anc-{i}", "instruction": "q", "response": "a", "source": "gsm8k"} for i in range(10)]
        with mock.patch.object(A, "load", lambda: rows):
            self.assertEqual(A.sample(4, seed=1), A.sample(4, seed=1))
            self.assertEqual(len(A.sample(50, seed=1)), 10)
            self.assertEqual(A.sample(0, seed=1), [])


if __name__ == "__main__":
    unittest.main()
