"""Tests for the cross-session scorecard metric functions (ingest/scorecard.py). Synthetic transcripts only; no network."""
import json
import math
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ingest.config import Config
from ingest.scorecard import cluster_bootstrap_delta, load_labels, median, p90, parse_transcript, quantile, run_scorecard

SECRET = "SECRET-PROMPT-TEXT-9f3a"


def user(ts, text):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": text}}


def asst(ts, mid, blocks, usage=None, model="m"):
    m = {"id": mid, "model": model, "content": blocks}
    if usage:
        m["usage"] = usage
    return {"type": "assistant", "timestamp": ts, "message": m}


def tool_use(i, name, inp=None):
    return {"type": "tool_use", "id": i, "name": name, "input": inp or {}}


def result(ts, tid):
    return {"type": "user", "timestamp": ts, "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": "x"}]}}


def hook(ts, tuid, ev, ms, typ="hook_success"):
    return {"type": "attachment", "timestamp": ts, "attachment": {"type": typ, "hookEvent": ev, "toolUseID": tuid, "durationMs": str(ms)}}


def U(i, o, cc, cr):
    return {"input_tokens": i, "output_tokens": o, "cache_creation_input_tokens": cc, "cache_read_input_tokens": cr}


class Stats(unittest.TestCase):
    def test_quantile_median_p90(self):
        self.assertEqual(median([1, 2, 3]), 2.0)
        self.assertEqual(median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(p90(list(range(11))), 9.0)
        self.assertTrue(math.isnan(median([])))
        self.assertEqual(quantile([5], 0.9), 5.0)
        self.assertEqual(median([1, None, 3]), 2.0)

    def test_bootstrap_detects_a_shift_and_not_noise(self):
        base = {f"b{i}": [10 + (i % 3), 11, 12] for i in range(8)}
        post = {f"p{i}": [20 + (i % 3), 21, 22] for i in range(8)}
        r = cluster_bootstrap_delta(base, post, reps=300)
        self.assertGreater(r["delta"], 8)
        self.assertTrue(r["sig"])
        self.assertGreater(r["lo"], 0)
        same = cluster_bootstrap_delta(base, {f"p{i}": v for i, v in enumerate(base.values())}, reps=300)
        self.assertFalse(same["sig"])
        self.assertEqual(same["delta"], 0.0)

    def test_bootstrap_needs_enough_sessions_and_is_deterministic(self):
        self.assertIsNone(cluster_bootstrap_delta({"a": [1]}, {"b": [2], "c": [2], "d": [2]}))
        b = {f"b{i}": [i, i + 1] for i in range(5)}
        p = {f"p{i}": [i + 3, i + 4] for i in range(5)}
        self.assertEqual(cluster_bootstrap_delta(b, p, reps=200), cluster_bootstrap_delta(b, p, reps=200))


class Parse(unittest.TestCase):
    def test_turn_tokens_dedupe_by_message_id_and_max_output(self):
        recs = [user("2026-10-01T00:00:00Z", SECRET),
                asst("2026-10-01T00:00:01Z", "m1", [{"type": "text", "text": "a"}], U(10, 1, 100, 1000)),
                asst("2026-10-01T00:00:02Z", "m1", [tool_use("t1", "Read")], U(10, 50, 100, 1000)),
                result("2026-10-01T00:00:03Z", "t1"),
                asst("2026-10-01T00:00:09Z", "m2", [{"type": "text", "text": "b"}], U(5, 7, 0, 2000)),
                user("2026-10-01T00:01:00Z", "second"),
                asst("2026-10-01T00:01:05Z", "m3", [{"type": "text", "text": "c"}], U(1, 1, 1, 1))]
        s = parse_transcript(recs, is_sub=False)
        self.assertEqual(len(s["turns"]), 2)
        self.assertEqual(s["turns"][0]["tok"], [15, 57, 100, 3000])
        self.assertEqual(s["turns"][1]["tok"], [1, 1, 1, 1])
        self.assertEqual(s["tokens"], [16, 58, 101, 3001])
        self.assertEqual(s["api_calls"], 3)
        self.assertEqual(s["turns"][0]["end"], "2026-10-01T00:00:09Z")

    def test_tool_counts_and_searching_per_turn_and_gaps(self):
        recs = [user("2026-10-01T00:00:00Z", "go"),
                asst("2026-10-01T00:00:01Z", "m1", [tool_use("a", "Read"), tool_use("b", "Grep"), tool_use("c", "Bash")]),
                asst("2026-10-01T00:00:01Z", "m1", [tool_use("a", "Read")]),
                result("2026-10-01T00:00:01.050Z", "a"), result("2026-10-01T00:00:03Z", "c")]
        s = parse_transcript(recs, False)
        self.assertEqual(s["turns"][0]["tools"], 3)
        self.assertEqual(s["turns"][0]["rgg"], 2)
        self.assertEqual(s["tool_counts"], {"Read": 1, "Grep": 1, "Bash": 1})
        self.assertEqual(s["gaps"]["Read"], [50])
        self.assertEqual(s["gaps"]["Bash"], [2000])

    def test_search_like_bash_is_counted_without_keeping_the_command(self):
        recs = [user("2026-10-01T00:00:00Z", "go"),
                asst("2026-10-01T00:00:01Z", "m1", [tool_use("a", "Bash", {"command": "cd x && grep -rn " + SECRET + " src"}),
                                                    tool_use("b", "Bash", {"command": "npm test"}), tool_use("c", "Bash", {"command": "ls -la"})])]
        s = parse_transcript(recs, False)
        self.assertEqual((s["bash_search"], s["turns"][0]["bs"]), (2, 2))
        self.assertNotIn(SECRET, json.dumps(s))

    def test_subagent_file_has_no_turns_but_totals(self):
        recs = [user("2026-10-01T00:00:00Z", "dispatch prompt"),
                asst("2026-10-01T00:00:05Z", "m1", [tool_use("a", "Glob"), tool_use("b", "Edit")], U(3, 4, 5, 6))]
        s = parse_transcript(recs, True)
        self.assertEqual(s["turns"], [])
        self.assertEqual(s["tokens"], [3, 4, 5, 6])
        self.assertEqual((s["n_tools"], s["rgg"]), (2, 1))

    def test_hooks_use_max_per_event_group_and_count_errors(self):
        recs = [user("2026-10-01T00:00:00Z", "go"),
                hook("2026-10-01T00:00:01Z", "t1", "PreToolUse", 100), hook("2026-10-01T00:00:01Z", "t1", "PreToolUse", 700),
                hook("2026-10-01T00:00:01Z", "t1", "PostToolUse", 50, "hook_non_blocking_error"),
                hook("2026-10-01T00:00:01Z", "s", "SessionStart", 9000),
                {"type": "system", "subtype": "stop_hook_summary", "timestamp": "2026-10-01T00:00:05Z",
                 "hookInfos": [{"command": "a", "durationMs": 300}, {"command": "b", "durationMs": 1200}]}]
        s = parse_transcript(recs, False)
        self.assertEqual(s["turns"][0]["hook_wall_ms"], 750)
        self.assertEqual(s["turns"][0]["stop_ms"], 1200)
        self.assertEqual(s["hook_events"]["PreToolUse"][:2], [800, 2])
        self.assertEqual(s["hook_events"]["PostToolUse"][2], 1)

    def test_compaction_and_dispatches_and_no_text_kept(self):
        recs = [user("2026-10-01T00:00:00Z", SECRET),
                asst("2026-10-01T00:00:01Z", "m1", [tool_use("d1", "Agent", {"subagent_type": "fr-builder", "prompt": SECRET})]),
                {"type": "system", "subtype": "compact_boundary", "timestamp": "2026-10-01T00:05:00Z"}]
        s = parse_transcript(recs, False)
        self.assertEqual(s["compactions"], ["2026-10-01T00:05:00Z"])
        self.assertEqual(s["dispatches"]["d1"]["type"], "fr-builder")
        self.assertNotIn(SECRET, json.dumps(s))

    def test_system_prefixed_user_messages_are_not_turns_and_synthetic_has_no_usage(self):
        recs = [user("2026-10-01T00:00:00Z", "<task-notification>x</task-notification>"), user("2026-10-01T00:00:01Z", "real"),
                asst("2026-10-01T00:00:02Z", "m1", [{"type": "text", "text": "hi"}], U(1, 1, 1, 1), model="<synthetic>")]
        s = parse_transcript(recs, False)
        self.assertEqual(len(s["turns"]), 1)
        self.assertEqual(s["api_calls"], 0)


class EndToEnd(unittest.TestCase):
    def _write(self, p: Path, recs):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    def test_baseline_split_deltas_and_no_text_in_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            proj, data, res = d / "projects", d / "data", d / "results"
            recs = []
            for i in range(6):   # 3 turns before 2026-10-10, 3 after, cache read rising
                day = "2026-10-05" if i < 3 else "2026-10-12"
                recs += [user(f"{day}T00:0{i}:00Z", SECRET),
                         asst(f"{day}T00:0{i}:05Z", f"m{i}", [{"type": "text", "text": SECRET}], U(10, 10, 0, 100 if i < 3 else 500))]
            self._write(proj / "C--x" / "11111111-1111-1111-1111-111111111111.jsonl", recs)
            recs2 = []
            for i in range(4):
                day = "2026-10-06" if i < 2 else "2026-10-13"
                recs2 += [user(f"{day}T01:0{i}:00Z", "q"), asst(f"{day}T01:0{i}:05Z", f"n{i}", [{"type": "text", "text": "a"}], U(10, 10, 0, 100 if i < 2 else 500))]
            self._write(proj / "C--x" / "22222222-2222-2222-2222-222222222222.jsonl", recs2)
            self._write(proj / "C--x" / "33333333-3333-3333-3333-333333333333.jsonl",
                        [user("2026-10-14T00:00:00Z", "q"), asst("2026-10-14T00:00:05Z", "z", [{"type": "text", "text": "a"}], U(1, 1, 0, 900))])
            self._write(data / "outcomes.jsonl", [])
            cfg = Config(projects=proj, data=data)
            t = datetime(2026, 10, 10, tzinfo=timezone.utc).timestamp()
            r1 = run_scorecard(cfg, res, now=t, reps=200)
            self.assertEqual((r1["turns"], r1["after_turns"]), (11, 6))
            r2 = run_scorecard(cfg, res, reps=200)       # the baseline cutoff is sticky across runs
            self.assertEqual(r2["cutoff"], r1["cutoff"])
            md = (res / "scorecard.md").read_text(encoding="utf-8")
            self.assertIn("After the cutoff", md)
            self.assertIn("tokens per turn (all, incl. cache read)", md)
            rows = [json.loads(x) for x in (data / "scorecard.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(any(r["baseline"] for r in rows) and any(not r["baseline"] for r in rows))
            html = (res / "scorecard.html").read_text(encoding="utf-8")
            self.assertNotIn(SECRET, md + html + json.dumps(rows))
            self.assertNotIn("<script", html)
            self.assertNotIn("http", html)

    def test_labels_are_read_from_the_ingest_outcomes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "outcomes.jsonl"
            p.write_text(json.dumps({"id": "out:toolu_1", "kind": "dispatch", "outcome": "re-dispatch"}) + "\n" +
                         json.dumps({"id": "out:toolu_2", "kind": "continuation", "outcome": "success"}) + "\n" + "{torn", encoding="utf-8")
            self.assertEqual(load_labels(p), {"toolu_1": "re-dispatch"})


if __name__ == "__main__":
    unittest.main()
