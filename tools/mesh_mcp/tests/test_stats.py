"""stats_impl against known SciPy/statsmodels values (host, needs scipy+statsmodels) and through the sandbox."""
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sandbox"))
try:
    import numpy as np
    from scipy import stats
    import stats_impl as S
    HAVE = True
except ImportError:  # host without scipy: only the sandbox test can run
    HAVE = False

from tools.mesh_mcp.common import docker_available

A = [5.1, 4.9, 6.2, 5.8, 6.0, 5.5, 5.3, 6.1, 5.7, 5.9]
B = [4.1, 4.5, 5.0, 4.8, 4.2, 4.9, 4.4, 5.1, 4.6, 4.3, 4.0, 4.7]


@unittest.skipUnless(HAVE, "scipy/statsmodels not installed on host")
class StatsHostTest(unittest.TestCase):
    def test_student_and_welch_match_scipy(self):
        for kind, eq in (("ttest", True), ("welch", False)):
            r = S.compute(kind, {"a": A, "b": B})
            ref = stats.ttest_ind(A, B, equal_var=eq)
            self.assertAlmostEqual(r["t"], ref.statistic, places=10)
            self.assertAlmostEqual(r["p"], ref.pvalue, places=12)
            lo, hi = ref.confidence_interval()
            self.assertAlmostEqual(r["mean_diff_ci"][0], lo, places=10)
            self.assertAlmostEqual(r["mean_diff_ci"][1], hi, places=10)

    def test_known_cohens_d(self):
        r = S.compute("cohens_d", {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        self.assertAlmostEqual(r["effect"]["cohens_d"], -1.2, places=12)  # (3-6)/sqrt((2.5+10)/2)
        self.assertEqual(r["effect"]["magnitude"], "large")

    def test_one_sample_and_paired(self):
        r = S.compute("ttest", {"a": A, "mu": 5})
        ref = stats.ttest_1samp(A, 5)
        self.assertAlmostEqual(r["p"], ref.pvalue, places=12)
        r = S.compute("paired_t", {"a": A, "b": B[:10]})
        self.assertAlmostEqual(r["t"], stats.ttest_rel(A, B[:10]).statistic, places=10)

    def test_mannwhitney(self):
        r = S.compute("mannwhitney", {"a": A, "b": B})
        ref = stats.mannwhitneyu(A, B)
        self.assertEqual(r["U"], ref.statistic)
        self.assertAlmostEqual(r["p"], ref.pvalue, places=12)
        self.assertAlmostEqual(r["effect"]["rank_biserial_r"], 2 * ref.statistic / (len(A) * len(B)) - 1)

    def test_chi2_and_cramers_v(self):
        t = [[10, 20], [30, 40]]
        r = S.compute("chi2", {"table": t})
        chi, p, dof, _ = stats.chi2_contingency(t)
        self.assertAlmostEqual(r["chi2"], chi)
        self.assertAlmostEqual(r["p"], p)
        self.assertAlmostEqual(r["effect"]["cramers_v"], (chi / 100) ** 0.5)
        self.assertIn("warning", S.compute("chi2", {"table": [[1, 2], [3, 4]]}))

    def test_mcnemar_exact_equals_binomial(self):
        r = S.compute("mcnemar", {"table": [[10, 5], [1, 20]]})
        self.assertAlmostEqual(r["p"], stats.binomtest(1, 6, 0.5).pvalue, places=12)  # 0.21875
        self.assertAlmostEqual(r["p"], 0.21875, places=12)
        self.assertEqual(r["effect"]["odds_ratio_b_over_c"], 5.0)

    def test_wilson_interval_known_value(self):
        r = S.compute("prop_ci", {"successes": 8, "n": 10})
        self.assertAlmostEqual(r["ci"][0], 0.4901624715, places=8)
        self.assertAlmostEqual(r["ci"][1], 0.9433178485, places=8)

    def test_holm_and_bh(self):
        p = [0.01, 0.04, 0.03, 0.20]
        r = S.compute("multitest", {"pvalues": p, "method": "holm"})
        self.assertEqual([round(x, 10) for x in r["p_adjusted"]], [0.04, 0.09, 0.09, 0.2])
        r = S.compute("multitest", {"pvalues": p, "method": "bh"})
        self.assertEqual([round(x, 10) for x in r["p_adjusted"]], [0.04, 0.0533333333, 0.0533333333, 0.2])
        self.assertEqual(r["n_rejected"], 1)

    def test_bad_input(self):
        for kind, data in (("nope", {}), ("welch", {"a": [1]}), ("prop_ci", {"successes": 5, "n": 3}),
                           ("mcnemar", {"table": [[1, 2, 3]]}), ("multitest", {"pvalues": [2]}),
                           ("welch", {"a": A, "b": B, "alpha": 2})):
            with self.assertRaises(ValueError):
                S.compute(kind, data)


@unittest.skipUnless(docker_available(), "Docker grader image not available")
class StatsSandboxTest(unittest.TestCase):
    def test_tool_runs_in_sandbox_and_reports_effect(self):
        from tools.mesh_mcp.sandbox_tools import stats_test
        out = stats_test("welch", {"a": A, "b": B})
        self.assertTrue(out.startswith("Welch two-sample t-test"))
        self.assertIn("EFFECT SIZE: {cohens_d=", out)
        p = float(re.search(r"^p: (\S+)", out, re.M).group(1))
        if HAVE:
            self.assertAlmostEqual(p, stats.ttest_ind(A, B, equal_var=False).pvalue, places=5)
        with self.assertRaisesRegex(ValueError, "unknown kind"):
            stats_test("bogus", {"a": A})


if __name__ == "__main__":
    unittest.main()
