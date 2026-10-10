"""labeler v2: structural outcome labels (chain successors, fix-task starts, env interruptions) + agreement with the evaluator's hand labels."""
import unittest
from pathlib import Path

from ingest import labeler as lb
from ingest.config import Config
from ingest.extract import DispatchItem


def item(i, ts, prompt, agent="statsclaw:builder", kind="dispatch", chain="c1", report="", desc="d", later=None, signals=None, session="s"):
    return DispatchItem(id=f"out:{i}", ts=ts, end_ts=ts, project="p", session=session, kind=kind, agent=agent, model=None, spawn_depth=1,
                        description=desc, prompt=prompt, prompt_secret=False, report=report, report_secret=False, tainted=False, result_head="",
                        signals=signals or {}, later=later or {"no_report": False, "redispatched": False, "continued": 0, "result_error": False},
                        tokens=None, settled=True, chain=chain)


class LabelerTest(unittest.TestCase):
    def test_resume_after_turn_limit_marks_previous_segment(self):
        a = item("a", "2026-10-08T10:00:00Z", "PHASE A", agent="statsclaw:tester", report="done")
        b = item("b", "2026-10-08T11:00:00Z", "You hit the 100-turn limit mid PHASE A. Continue.", agent="statsclaw:tester", kind="continuation", report="done")
        out = lb.label_items([a, b])
        self.assertEqual(out["out:a"][0], "re-dispatch")
        self.assertIn("resumed_after_interrupt", out["out:a"][1])
        self.assertEqual(out["out:b"][0], "success")           # the resume message itself is fine; it is the PREVIOUS segment that was cut off

    def test_fix_round_start_and_defects_followed(self):
        b1 = item("b1", "2026-10-08T10:00:00Z", "You are the StatsClaw **builder** for run REQ-20261008-x", report="BUILD_COMPLETE")
        b2 = item("b2", "2026-10-08T12:00:00Z", "Builder fix round 1 for REQ-20261008-x. D1-D5 from the tester.", kind="continuation", report="BUILD_COMPLETE")
        out = lb.label_items([b1, b2])
        self.assertEqual(out["out:b1"][0], "defect")
        self.assertEqual(out["out:b2"][0], "fix-round")

    def test_reviewer_defects_need_a_review_origin_fix(self):
        r = item("r", "2026-10-08T10:00:00Z", "You are the StatsClaw **reviewer**", agent="statsclaw:reviewer", chain="cr", report="PASS WITH NOTES")
        b = item("b", "2026-10-08T11:00:00Z", "Builder fix round 3 (from the independent reviewer)", agent="general-purpose", kind="dispatch", chain="cb",
                 desc="fix")
        b2 = item("b2", "2026-10-08T11:00:00Z", "You are acting as the StatsClaw **builder**", agent="general-purpose", chain="cb0", report="ok")
        self.assertEqual(lb.label_items([r, b, b2])["out:r"][0], "defect")
        ci = item("b3", "2026-10-08T11:00:00Z", "Builder fix round 4 (CI only)", agent="fr-builder", chain="cc")
        self.assertEqual(lb.label_items([r, ci])["out:r"][0], "success")      # a CI fix is not a review defect

    def test_product_rate_limit_text_is_not_an_environment_failure(self):
        a = item("a", "2026-10-08T10:00:00Z", "add rate limiting to the API", agent="general-purpose",
                 report="Added rate limiting with a per-key limit; API error paths return 429 JSON bodies in the tests.\n- done")
        self.assertEqual(lb.label_items([a])["out:a"][0], "re-dispatch")      # '429' alone is still an interruption cue in v2 ...
        a2 = item("a2", "2026-10-08T10:00:00Z", "add rate limiting to the API", agent="general-purpose", report="Added rate limiting. API error paths are covered. Done.")
        self.assertEqual(lb.label_items([a2])["out:a2"][0], "success")       # ... but 'API error paths' / 'rate limiting' are not

    def test_blocked_and_note(self):
        s = item("s", "2026-10-08T10:00:00Z", "ship it", agent="statsclaw:shipper", report="wrote ship.md and marked SHIP_BLOCKED (git push denied by permission classifier)")
        n = item("n", "2026-10-08T10:05:00Z", "Note from leader (no reply needed): wire X", agent="statsclaw:shipper", kind="continuation", report="whatever")
        out = lb.label_items([s, n])
        self.assertEqual(out["out:s"][0], "blocked")
        self.assertEqual((out["out:n"][0], out["out:n"][1]), ("success", ["note"]))

    def test_gate_negative_is_a_signal_not_a_defect(self):
        a = item("a", "2026-10-08T10:00:00Z", "run the cycle", agent="general-purpose", report="Real cycle: NOT promoted. The gate said no; nothing pushed.")
        oc, sig = lb.label_items([a])["out:a"]
        self.assertEqual(oc, "success")
        self.assertIn("verdict_gate_negative", sig)

    def test_agreement_with_hand_labels_on_real_states(self):
        cfg = Config()
        if not cfg.files_dir.exists() or not any(cfg.files_dir.glob("*.json")):
            self.skipTest("no ingest state on this checkout")
        res = lb.validate(cfg)
        if res["labelled"] < 40:
            self.skipTest("hand-labelled dispatches not present in this state")
        self.assertGreater(res["v2"]["exact"], res["v1"]["exact"])
        self.assertGreaterEqual(res["v2"]["exact"], 45)


if __name__ == "__main__":
    unittest.main()
