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


if __name__ == "__main__":
    unittest.main()
