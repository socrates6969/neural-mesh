"""Known-value tests for the E2 statistics (llm/stats.py) and the baseline-aware quality report (llm/evaluate.py). No GPU, no network."""
import math
import random
import unittest

from llm import stats
from llm.evaluate import quality_report


class ExactTests(unittest.TestCase):
    def test_mcnemar_known(self):
        self.assertAlmostEqual(stats.mcnemar_exact_p(7, 1), 0.0703125)       # 2 * (1 + 8) / 256
        self.assertEqual(stats.mcnemar_exact_p(5, 5), 1.0)
        self.assertEqual(stats.mcnemar_exact_p(0, 0), 1.0)
        self.assertAlmostEqual(stats.mcnemar_exact_p(10, 0), 2 / 1024)
        self.assertEqual(stats.mcnemar_exact_p(3, 9), stats.mcnemar_exact_p(9, 3))

    def test_agrees_with_shadow(self):
        from ingest.shadow import mcnemar_exact_p
        rng = random.Random(1)
        for _ in range(200):
            b, c = rng.randrange(0, 40), rng.randrange(0, 40)
            self.assertAlmostEqual(stats.mcnemar_exact_p(b, c), mcnemar_exact_p(b, c))

    def test_critical_wins(self):
        self.assertIsNone(stats.critical_wins(4))        # 2^-4 = .0625 > .05: no outcome is significant
        self.assertEqual(stats.critical_wins(5), 5)      # 1/32
        self.assertEqual(stats.critical_wins(10), 9)     # P(X>=8) = 56/1024 = .0547 > .05, P(X>=9) = 11/1024
        self.assertEqual(stats.critical_wins(20), 15)    # P(X>=15) = .0207, P(X>=14) = .0577

    def test_sign_power_known(self):
        self.assertAlmostEqual(stats.sign_power(10, 0.9), 10 * 0.9 ** 9 * 0.1 + 0.9 ** 10)   # P(X >= 9 | p = .9)
        self.assertAlmostEqual(stats.sign_power(10, 0.5), 11 / 1024)                          # the size of the test at the null
        self.assertEqual(stats.sign_power(4, 1.0), 0.0)

    def test_mde_matches_brute_force(self):
        for m in (5, 6, 10, 20, 21, 40, 60, 100):
            p = stats.mde_sign_test(m)
            if p is None:
                self.assertLess(stats.sign_power(m, 1.0), 0.8)
                continue
            self.assertGreaterEqual(stats.sign_power(m, p), 0.8 - 1e-9)
            self.assertLess(stats.sign_power(m, p - 2e-3), 0.8)                 # smallest, up to the bisection tolerance
        self.assertAlmostEqual(stats.mde_sign_test(6), 0.8 ** (1 / 6))          # m = 6 needs 6/6 wins: p ** 6 >= .8
        self.assertIsNone(stats.mde_sign_test(4))

    def test_mde_shrinks_with_n(self):
        vals = [stats.mde_sign_test(m) for m in (25, 50, 100, 200, 400)]
        self.assertTrue(all(a > b for a, b in zip(vals, vals[1:])))

    def test_mde_paired(self):
        r = stats.mde_paired(20, 0.3)
        self.assertEqual(r["decisive"], 6)
        self.assertAlmostEqual(r["win_share"], 0.8 ** (1 / 6))
        self.assertAlmostEqual(r["min_diff"], 0.3 * (2 * 0.8 ** (1 / 6) - 1))
        self.assertIsNone(stats.mde_paired(10, 0.2)["min_diff"])                 # 2 decisive pairs: can never be significant
        self.assertIsNone(stats.mde_observed(20, 0, 0)["min_diff"])

    def test_required_decisive(self):
        m = stats.required_decisive(0.6)
        self.assertGreaterEqual(stats.sign_power(m, 0.6), 0.8)
        self.assertTrue(all(stats.sign_power(k, 0.6) < 0.8 for k in range(5, m)))
        self.assertTrue(140 <= m <= 170)                                         # the frontier-research estimate was ~150
        self.assertLess(stats.required_decisive(0.75), 40)


class ClassMetrics(unittest.TestCase):
    def test_majority_predictor_is_chance(self):
        y = ["a"] * 90 + ["b"] * 10
        self.assertAlmostEqual(stats.balanced_accuracy(y, ["a"] * 100), 0.5)      # 90% accuracy, 50% balanced
        self.assertAlmostEqual(stats.balanced_accuracy(y, y), 1.0)
        self.assertIsNone(stats.balanced_accuracy([], []))

    def test_recall_by_class(self):
        r = stats.recall_by_class(["a", "a", "b", "c", "c", "c"], ["a", "b", "b", "a", "c", "c"])
        self.assertEqual((r["a"]["k"], r["a"]["n"]), (1, 2))
        self.assertAlmostEqual(r["c"]["recall"], 2 / 3)
        self.assertEqual(r["b"]["recall"], 1.0)


class QualityReport(unittest.TestCase):
    """The stored 20261012-160854 verdict: research 4/20 vs 9/20 (0 cand-only, 5 prod-only), pairwise W13 / L8 / T6, 33 unjudged."""
    DET = {"categories": {
        "math": {"n": 20, "cand": 10, "prod": 8, "cand_only": 3, "prod_only": 1},
        "physics": {"n": 20, "cand": 11, "prod": 8, "cand_only": 4, "prod_only": 1},
        "code": {"n": 20, "cand": 20, "prod": 18, "cand_only": 2, "prod_only": 0},
        "research": {"n": 20, "cand": 4, "prod": 9, "cand_only": 0, "prod_only": 5}}}
    PAIR = {"n_items": 60, "n_judged": 27, "wins": 13, "losses": 8, "ties": 6}

    def test_numbers(self):
        q = quality_report(self.DET, self.PAIR)
        self.assertAlmostEqual(q["categories"]["research"]["mcnemar_p_two_sided"], 2 / 32)
        self.assertAlmostEqual(q["categories"]["research"]["sign_p_cand_better"], 1.0)
        self.assertAlmostEqual(q["pooled"]["acc_cand"], 45 / 80)
        self.assertAlmostEqual(q["pooled"]["balanced_acc_cand"], (0.5 + 0.55 + 1.0 + 0.2) / 4)
        self.assertAlmostEqual(q["pooled"]["balanced_acc_prod"], (0.4 + 0.4 + 0.9 + 0.45) / 4)
        self.assertAlmostEqual(q["pooled"]["mcnemar_p_two_sided"], stats.mcnemar_exact_p(9, 7))
        self.assertEqual(q["pairwise"]["decisive"], 21)
        self.assertAlmostEqual(q["pairwise"]["mcnemar_p_two_sided"], stats.mcnemar_exact_p(13, 8))
        self.assertEqual(q["pairwise"]["noop_win_rate"], 0.5)
        self.assertTrue(q["underpowered"]["battery_for_10pt"])
        self.assertTrue(q["underpowered"]["pairwise_for_target"])

    def test_large_n_is_powered(self):
        det = {"categories": {"x": {"n": 400, "cand": 240, "prod": 200, "cand_only": 60, "prod_only": 20}}}
        pair = {"n_items": 300, "n_judged": 300, "wins": 150, "losses": 100, "ties": 50}
        q = quality_report(det, pair)
        self.assertFalse(q["underpowered"]["battery_for_10pt"])
        self.assertTrue(q["underpowered"]["pairwise_for_target"] is False)


if __name__ == "__main__":
    unittest.main()
