"""E3 router harness: feature hygiene and the abstain-to-majority naive Bayes router (ingest/router_e3.py)."""
import unittest

from ingest import router_e3 as e3


def row(i, agent, text, project="p", kind="dispatch"):
    return {"id": f"o{i}", "ts": f"2026-10-0{1 + i // 100}T00:00:{i % 60:02d}Z", "kind": kind, "agent": agent, "project": project,
            "task_text": text, "features": [], "outcome": "success", "signals": []}


class FeatureHygieneTest(unittest.TestCase):
    def test_drops_paths_users_hashes_and_digit_tokens(self):
        t = r"Fix R117 merge 7a05bba in C:\Users\mariu\neural-mesh\src\hebbian.rs see https://x.io/a and ./scripts/run.sh then test"
        f = e3.features_clean(t)
        self.assertEqual(f, ["fix", "merge", "see", "test"])
        for bad in ("mariu", "users", "r117", "7a05bba", "hebbian", "rs", "scripts"):
            self.assertNotIn(bad, f)

    def test_phase_verb_uses_first_phase_keyword(self):
        self.assertEqual(e3.phase_verb("Review then build the thing"), "verb:reviewer")
        self.assertEqual(e3.phase_verb("Ponder quietly"), "verb:none")


class NaiveBayesRouterTest(unittest.TestCase):
    def test_margin_keeps_majority_until_evidence_is_strong(self):
        rows = [row(i, "general-purpose", "explore the code base") for i in range(20)]
        rows += [row(20 + i, "fr-tester", "write acceptance tests for parser") for i in range(6)]
        rows += [row(40, "fr-tester", "write acceptance tests for parser")]
        rows = e3.load_rows(rows)
        mask = e3.scored_mask(rows)
        loose = e3.sys_nb(rows, mask, "clean", alpha=0.05, proj_w=3.0, margin=0.0)
        strict = e3.sys_nb(rows, mask, "clean", alpha=0.05, proj_w=3.0, margin=1e9)
        self.assertEqual(loose[-1], "fr-tester")
        self.assertEqual(strict[-1], "general-purpose")


if __name__ == "__main__":
    unittest.main()
