"""Fixture tests for ops/: the dashboard renders with missing or partial data, alerts dedup and rate-limit.

    py -3.12 -m unittest discover -s ops/tests -t .
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

os.environ["OPS_NOTIFY"] = "none"  # never pop real toasts from tests

from ops import alerts, dashboard, sources  # noqa: E402


def write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def jl(rows) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def nightly_run(run_id, promoted=False, exit_code=0, status="ok"):
    return {"run_id": run_id, "started": "2026-10-20T01:00:00+00:00", "status": status, "exit": exit_code,
            "stages": [{"name": "train", "status": "ok", "exit": 0, "detail": {"ver": run_id, "n_new": 5}},
                       {"name": "eval", "status": "ok", "exit": 0, "reason": "win rate 0.61, CI [0.5, 0.7]",
                        "detail": {"ver": run_id}},
                       {"name": "promote", "status": "ok" if promoted else "noop", "exit": 0,
                        "detail": {"promoted": promoted}}]}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.nm, self.ac, self.out = base / "nm", base / "ac", base / "out"
        self.over = {"nm": self.nm, "ac": self.ac, "out": self.out}

    def tearDown(self):
        self.tmp.cleanup()

    def populate(self):
        b = self.ac / "train" / "batch1"
        write(b / "leaderboard.json", json.dumps({"spent_usd": 12.5, "fields": {"math": [
            {"agent": "mesh-mathematician", "held_base": [8, 10], "held_with_pack": [9, 10], "pack_version": 2,
             "pack_status": "candidate", "pooled": {"n": 10, "base": 8, "pack": 9, "b_base_only": 0,
                                                    "c_pack_only": 1, "p": 1.0}}]}}))
        write(b / "packs" / "mesh-mathematician" / "math" / "status.json",
              json.dumps({"status": "candidate", "version": 2, "rounds": [{"kept": True}], "pooled": {"p": 1.0}}))
        write(b / "results.jsonl", jl([{"agent": "a", "phase": "held_base", "cost": 0.1, "error": None},
                                        {"agent": "a", "phase": "tune", "cost": 0.1, "error": "boom"}]) + "{torn")
        write(self.ac / "batch1_stdout.log",
              "10:00:00 start: 1 agents, rounds 0..2, budget $50, spent so far $0.00\n10:05:00 round 0 done, spent $5.00\n")
        L = self.nm / "data" / "lifelong"
        write(L / "outcomes.jsonl", jl([{"id": "o1", "ts": "2026-10-22T10:00:00Z", "project": "p"}]))
        write(L / "mesh" / "router_state.json", "{}")
        write(L / "mesh" / "experience.jsonl", jl([{"task": "t1", "agent": "a", "field": "math"}]))
        write(L / "runs.jsonl", jl([{"ts": "2026-10-24T06:00:00+00:00", "totals": {"outcomes": 1},
                                     "mesh": {"fed": 3}}]))
        write(L / "ingest-runs.log", "2026-10-24T08:00:00+02:00 ingest run exit 0\n")
        D = self.nm / "llm" / "models"
        write(D / "runs.jsonl", jl([nightly_run("r1")]))
        write(D / "state.json", json.dumps({"production": {"ver": "v1", "base": "Qwen"}, "trained_ids": ["x"]}))
        write(D / "nightly.log", "step 1 loss 0.5\nLoading weights: 50%|## | 1/2 [00:01<00:01, 1.0it/s]\n")


class DashboardTests(Fixture):
    def test_renders_with_everything_missing(self):
        t0 = time.time()
        st = dashboard.build(self.over)
        self.assertLess(time.time() - t0, 5)
        html = (self.out / "index.html").read_text(encoding="utf-8")
        self.assertIn("neural-mesh ops", html)
        self.assertIn("workspace.db not present yet", html)
        status = json.loads((self.out / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["worst"], "err")  # router_state.json missing
        self.assertEqual(status["training"], [])
        self.assertTrue(any(h["name"] == "router" for h in st["health"]))

    def test_renders_partial_data_and_escapes(self):
        self.populate()
        L = self.nm / "data" / "lifelong" / "mesh"
        write(L / "experience.jsonl", jl([{"task": "<script>alert(1)</script>", "agent": "a", "field": "math"}]))
        st = dashboard.build(self.over)
        html = (self.out / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertNotIn("http://", html.replace("http-equiv", ""))  # self-contained: no external resources
        b = st["training"]["batches"][0]
        self.assertEqual((b["spent"], b["budget"], b["results"], b["result_errors"]), (12.5, 50.0, 2, 1))
        self.assertEqual(st["mesh"]["router"]["fed_by_ingest"], 3)
        self.assertNotIn("it/s]", "\n".join(st["nightly"]["log_tail"]))

    def test_corrupt_json_is_reported_not_fatal(self):
        self.populate()
        write(self.ac / "train" / "batch1" / "leaderboard.json", "{not json")
        st = dashboard.build(self.over)
        self.assertTrue(any(h["name"] == "read error" for h in st["health"]))
        self.assertTrue((self.out / "index.html").exists())

    def test_workspace_db_read_only(self):
        self.populate()
        db = self.nm / "data" / "lifelong" / "mesh" / "workspace.db"
        make_workspace(db, [("t1", "a", 1, "solved"), ("t2", "b", 0, "<b>bold</b>")])
        st = dashboard.build(self.over)
        ws = st["mesh"]["workspace"]
        self.assertEqual((ws["entries"], ws["verified"]), (2, 1))
        self.assertIn("&lt;b&gt;bold", (self.out / "index.html").read_text(encoding="utf-8"))


def make_workspace(db: Path, rows):
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE IF NOT EXISTS entries (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, agent TEXT, "
                "kind TEXT, field TEXT, content TEXT, enc INTEGER DEFAULT 0, confidence REAL, verified INTEGER, "
                "sources TEXT, created REAL, s INTEGER DEFAULT 0, f INTEGER DEFAULT 0, superseded_by INTEGER, "
                "concepts TEXT, vec BLOB)")
    for task, agent, verified, content in rows:
        con.execute("INSERT INTO entries (task_id, agent, kind, content, confidence, verified, created) "
                    "VALUES (?,?,?,?,?,?,?)", (task, agent, "result", content, 0.9, verified, time.time()))
    con.commit()
    con.close()


class AlertTests(Fixture):
    def run_alerts(self, now):
        return alerts.run(self.over, dry_run=True, now=now)

    def types(self, events, outcome_prefix="sent"):
        return sorted(e["type"] for e in events if e["outcome"].startswith(outcome_prefix))

    def test_first_run_is_baseline_and_repeat_is_quiet(self):
        self.populate()
        write(self.ac / "batch1_stdout.log", "10:00 QUOTA: plan usage limit reached - stopping\n")
        self.assertEqual(self.run_alerts(1000), [])  # history is not re-announced
        self.assertEqual(self.run_alerts(2000), [])
        self.assertIn("baseline", (self.out / "alerts.log").read_text(encoding="utf-8"))

    def test_quota_dedup_and_hourly_rate_limit(self):
        self.populate()
        log = self.ac / "batch1_stdout.log"
        self.run_alerts(0)
        with open(log, "a", encoding="utf-8") as f:
            f.write("11:00 QUOTA: plan usage limit reached - stopping\n")
        ev = self.run_alerts(10_000)
        self.assertEqual(self.types(ev), ["quota_stop"])
        self.assertEqual(self.run_alerts(10_100), [])  # same line is not raised twice
        with open(log, "a", encoding="utf-8") as f:
            f.write("11:30 QUOTA: again\n")
        ev = self.run_alerts(10_200)  # within the hour: logged but suppressed
        self.assertEqual(self.types(ev, "suppressed"), ["quota_stop"])
        with open(log, "a", encoding="utf-8") as f:
            f.write("13:00 QUOTA: later\n")
        ev = self.run_alerts(10_000 + 3600)
        self.assertEqual(self.types(ev), ["quota_stop"])
        logged = (self.out / "alerts.log").read_text(encoding="utf-8")
        self.assertEqual(logged.count("\tquota_stop\t"), 3)
        self.assertIn("suppressed", logged)

    def test_other_events(self):
        self.populate()
        self.run_alerts(0)
        b = self.ac / "train" / "batch1"
        write(b / "packs" / "mesh-mathematician" / "math" / "status.json",
              json.dumps({"status": "promoted", "version": 3, "rounds": [], "pooled": {"p": 0.01}}))
        with open(self.ac / "batch1_stdout.log", "a", encoding="utf-8") as f:
            f.write("12:00 round 1 done, spent $9.00\n== stage 2 done 12:01\nTraceback (most recent call last):\n")
        D = self.nm / "llm" / "models"
        write(D / "runs.jsonl", jl([nightly_run("r1"), nightly_run("r2", promoted=True),
                                    nightly_run("r3", exit_code=2, status="failed")]))
        make_workspace(self.nm / "data" / "lifelong" / "mesh" / "workspace.db", [("t9", "solver", 1, "x")])
        write(self.nm / "data" / "lifelong" / "mesh" / "promotions.jsonl",
              jl([{"agent": "a", "field": "math", "version": 3, "result": "published"}]))
        ev = self.run_alerts(100_000)
        sent = self.types(ev)
        for kind in ("pack_promoted", "stage_done", "training_error", "nightly_promoted", "nightly_failed",
                     "workspace_verified"):
            self.assertIn(kind, sent)
        self.assertIn("stage_done", self.types(ev, "suppressed"))  # two stage lines in one pass: one toast
        self.assertEqual(self.run_alerts(200_000), [])

    def test_router_stale_fires_once_then_daily(self):
        self.populate()
        rs = self.nm / "data" / "lifelong" / "mesh" / "router_state.json"
        self.run_alerts(0)
        old = time.time() - 72 * 3600
        os.utime(rs, (old, old))
        self.assertEqual(self.types(self.run_alerts(10_000)), ["router_stale"])
        self.assertEqual(self.run_alerts(20_000), [])
        self.assertEqual(self.types(self.run_alerts(10_000 + 25 * 3600)), ["router_stale"])
        os.utime(rs, None)
        self.assertEqual(self.run_alerts(10_000 + 30 * 3600), [])

    def test_unreadable_state_file_rebaselines(self):
        self.populate()
        self.out.mkdir(parents=True)
        write(self.out / "alert_state.json", "garbage")
        self.assertEqual(self.run_alerts(0), [])
        self.assertIn("snapshot", json.loads((self.out / "alert_state.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()