"""E2 router report (ingest/router_eval.baseline_report): balanced accuracy, minority recall, McNemar vs each baseline, per-project no-regression. No binary needed."""
import unittest

from ingest.router_eval import baseline_report, e2_line


def row(i, proj, actual, mesh, top, rule_pick="fr-builder", fr=False):
    return {"i": i, "proj": proj, "actual": actual, "pick": mesh, "top": top, "rule_pick": rule_pick, "mesh": mesh == actual, "maj": top == actual,
            "rule": rule_pick == actual, "fr": fr}


class BaselineReport(unittest.TestCase):
    def test_majority_predictor_gets_chance_balanced_accuracy(self):
        rows = [row(i, "p", "general-purpose", "general-purpose", "general-purpose") for i in range(90)] + \
               [row(90 + i, "p", "other", "general-purpose", "general-purpose") for i in range(10)]
        s = baseline_report(rows)["e2_all"]
        self.assertAlmostEqual(s["systems"]["mesh"]["acc"], 0.9)
        self.assertAlmostEqual(s["systems"]["mesh"]["balanced_acc"], 0.5)
        self.assertEqual(s["systems"]["mesh"]["minority_recall"], 0.0)
        self.assertEqual(s["systems"]["mesh"]["minority_n"], 10)
        self.assertEqual(s["majority_class"], "general-purpose")
        self.assertEqual(s["mesh_vs_majority_so_far"]["mcnemar_exact_p"], 1.0)     # identical predictions: no discordant pair
        self.assertTrue(s["mesh_vs_majority_so_far"]["no_regression"])

    def test_learned_router_beating_majority_on_minority(self):
        rows = [row(i, "p", "a", "a", "a") for i in range(30)] + [row(30 + i, "p", "b", "b", "a") for i in range(12)]
        s = baseline_report(rows)["e2_all"]
        self.assertAlmostEqual(s["systems"]["mesh"]["balanced_acc"], 1.0)
        self.assertAlmostEqual(s["systems"]["majority_so_far"]["balanced_acc"], 0.5)
        self.assertEqual(s["mesh_vs_majority_so_far"]["mesh_only_right"], 12)
        self.assertAlmostEqual(s["mesh_vs_majority_so_far"]["mcnemar_exact_p"], round(2 / 4096, 4))
        self.assertEqual(s["systems"]["mesh"]["recall"]["b"], [12, 12])

    def test_per_project_no_regression_flags_the_worse_project(self):
        rows = [row(i, "good", "a", "a", "b") for i in range(5)] + [row(10 + i, "bad", "a", "b", "a") for i in range(3)]
        s = baseline_report(rows)["e2_all"]["mesh_vs_majority_so_far"]
        self.assertFalse(s["no_regression"])
        self.assertEqual(s["regressed_projects"], ["bad"])
        self.assertEqual(s["per_project"]["good"], {"n": 5, "mesh": 5, "base": 0})
        self.assertEqual(s["mesh_only_right"], 5)

    def test_in_domain_section_and_line(self):
        rows = [row(0, "fr", "fr-tester", "fr-tester", "fr-tester", "fr-tester", fr=True), row(1, "x", "general-purpose", "general-purpose", "general-purpose")]
        r = baseline_report(rows)
        self.assertEqual(r["e2_in_domain_fr"]["n"], 1)
        self.assertEqual(r["e2_all"]["n"], 2)
        self.assertIn("no-regression", e2_line(r))

    def test_empty_trace(self):
        r = baseline_report([])
        self.assertEqual(r["e2_all"]["n"], 0)
        self.assertIsNone(r["e2_all"]["mesh_vs_phase_rule"]["mde_at_observed"])


if __name__ == "__main__":
    unittest.main()
