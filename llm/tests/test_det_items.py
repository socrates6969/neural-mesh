"""det_items must give every code item the precomputed `expected` outputs that host-side sandbox scoring needs
(the 2026-10-12 nightly eval crashed with KeyError: 'expected')."""
import json
import unittest

from llm import config, evaluate


class DetItemsExpected(unittest.TestCase):
    def test_every_code_item_has_expected_matching_the_tune_view(self):
        code = [i for i in evaluate.det_items(1000) if i["kind"] == "code"]
        self.assertGreater(len(code), 0)
        tune = json.loads((config.REPO / "bench" / "battery_tune.json").read_text(encoding="utf-8"))["items"]
        tune_exp = {i["id"]: i["expected"] for i in tune if i.get("kind") == "code"}
        for it in code:
            self.assertIn(it["id"], tune_exp, it["id"])
            self.assertEqual(it["expected"], tune_exp[it["id"]], it["id"])

    def test_non_code_items_unchanged(self):
        for it in evaluate.det_items(1000):
            if it["kind"] != "code":
                self.assertNotIn("expected", it)


class DetCorrect(unittest.TestCase):
    """score() returns a dict; the battery must use its `correct` flag, not the truthiness of the dict."""
    class Bench:
        def __init__(self, result): self.result = result
        def score(self, item, text): return self.result

    def test_wrong_answer_counts_as_wrong(self):
        self.assertFalse(evaluate.det_correct(self.Bench({"correct": False, "infra": False}), {"id": "m1"}, "42"))

    def test_right_answer_counts_as_right(self):
        self.assertTrue(evaluate.det_correct(self.Bench({"correct": True, "infra": False}), {"id": "m1"}, "42"))

    def test_empty_answer_is_wrong_without_scoring(self):
        self.assertFalse(evaluate.det_correct(self.Bench(None), {"id": "m1"}, "  "))

    def test_infra_failure_fails_closed(self):
        with self.assertRaises(RuntimeError):
            evaluate.det_correct(self.Bench({"correct": False, "infra": True}), {"id": "c1"}, "def f(): pass")

    def test_real_scorer_rejects_a_wrong_math_answer(self):
        bench = evaluate._bench_module()
        item = next(i for i in evaluate.det_items(1000) if i["kind"] != "code")
        self.assertFalse(evaluate.det_correct(bench, item, "The answer is: definitely-not-the-answer-zzz"))


if __name__ == "__main__":
    unittest.main()
