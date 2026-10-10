"""End-to-end tests on a SYNTHETIC transcript tree: incremental reading, idempotent re-runs, taint handling, outcomes,
memory credit and pruning, judge parsing, lock. No network, no mesh binary, no encoder (all disabled via env)."""
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ingest import extract as ex
from ingest.config import Config
from ingest.judge import parse_grade
from ingest.memory import MemoryStore, hash_embed
from ingest.pipeline import run
from ingest.util import ProcessLock, iter_jsonl
from ingest.vectors import leader_clusters

OLD = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat().replace("+00:00", "Z")


def ts(minutes):
    return (datetime.now(timezone.utc) - timedelta(days=1) + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def user(text, t, sidechain=False):
    return {"type": "user", "timestamp": ts(t), "cwd": "C:\\work\\demo-proj", "isSidechain": sidechain, "message": {"role": "user", "content": text}}


def asst(blocks, t, mid, model="claude-test"):
    return {"type": "assistant", "timestamp": ts(t), "cwd": "C:\\work\\demo-proj",
            "message": {"id": mid, "model": model, "content": blocks, "usage": {"input_tokens": 10, "output_tokens": 5,
                                                                                 "cache_creation_input_tokens": 1, "cache_read_input_tokens": 2}}}


def tool_result(tid, text, t):
    return {"type": "user", "timestamp": ts(t), "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}}


class Env:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.projects = root / "projects"
        self.data = root / "data"
        self.cfg = Config()
        self.cfg.projects = self.projects
        self.cfg.data = self.data
        self.cfg.judge_budget = 0
        self.cfg.mesh_feedback = False
        self.cfg.settle_minutes = 1
        self.cfg.mesh_exe = root / "nope" / "neural-mesh.exe"
        self.cfg.meshstore_exe = root / "nope" / "meshstore.exe"

    def write(self, rel, records, mode="w"):
        p = self.projects / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, mode, encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        return p


def build_tree(env):
    sess = "sess-1"
    main = [
        user("Please add a retry helper to the demo service and test it", 0),
        asst([{"type": "tool_use", "id": "toolu_A", "name": "Agent", "input": {"subagent_type": "fr-builder", "description": "build retry helper",
                                                                              "prompt": "Implement a retry helper with backoff in src/retry.ts and add tests."}}], 1, "m1"),
        tool_result("toolu_A", "Done: retry helper implemented, 4 tests pass.", 30),
        asst([{"type": "tool_use", "id": "toolu_B", "name": "Agent", "input": {"subagent_type": "fr-tester", "description": "test retry helper",
                                                                              "prompt": "Run acceptance tests for the retry helper."}}], 31, "m2"),
        tool_result("toolu_B", "TEST_BLOCK: 2 tests failed", 40),
        asst([{"type": "tool_use", "id": "toolu_C", "name": "Bash", "input": {"command": "cat api/.env"}}], 41, "m3"),
        tool_result("toolu_C", "SHOULD_NEVER_BE_STORED=1", 42),
        asst([{"type": "text", "text": "The retry helper is in place; the root cause of the flaky test was a missing await in the backoff loop."}], 43, "m4"),
        user("thanks, looks good", 50),
    ]
    env.write(f"proj/{sess}.jsonl", main)
    sa = [user("Implement a retry helper with backoff in src/retry.ts and add tests.", 2, True),
          asst([{"type": "tool_use", "id": "x1", "name": "Edit", "input": {"file_path": "src/retry.ts"}}], 3, "s1"),
          tool_result("x1", "ok", 4),
          asst([{"type": "text", "text": "BUILD_COMPLETE. Implemented retry with exponential backoff; the fix was to cap the delay at 30 s. "
                                         "We decided to keep the helper generic instead of per-client. npm test: 4 passed, 0 failed."}], 29, "s2")]
    env.write(f"proj/{sess}/subagents/agent-aaa.jsonl", sa)
    (env.projects / f"proj/{sess}/subagents/agent-aaa.meta.json").write_text(json.dumps({"agentType": "fr-builder", "toolUseId": "toolu_A", "spawnDepth": 1}))
    sb = [user("Run acceptance tests for the retry helper.", 32, True),
          asst([{"type": "text", "text": "TEST_BLOCK r1: 2 tests failed (D1, D2). The bug was a missing await in the backoff loop."}], 39, "t1")]
    env.write(f"proj/{sess}/subagents/agent-bbb.jsonl", sb)
    (env.projects / f"proj/{sess}/subagents/agent-bbb.meta.json").write_text(json.dumps({"agentType": "fr-tester", "toolUseId": "toolu_B", "spawnDepth": 1}))
    return sess


class PipelineTests(unittest.TestCase):
    def setUp(self):
        os.environ["INGEST_EMBED_EXE"] = "Z:/missing/embed_texts.exe"  # vectors off: tests must not need the encoder
        self.env = Env()
        build_tree(self.env)

    def tearDown(self):
        os.environ.pop("INGEST_EMBED_EXE", None)
        self.env.tmp.cleanup()

    def test_outcomes_and_idempotency(self):
        r1 = run(self.env.cfg)
        outs = {o["id"]: o for o in iter_jsonl(self.env.cfg.outcomes)}
        self.assertEqual(set(outs), {"out:toolu_A", "out:toolu_B"})
        self.assertEqual(outs["out:toolu_A"]["outcome"], "success")
        self.assertEqual(outs["out:toolu_B"]["outcome"], "defect")
        self.assertEqual(outs["out:toolu_A"]["agent"], "fr-builder")
        self.assertEqual(outs["out:toolu_A"]["tokens"]["total"], 2 * 18)
        self.assertGreater(r1["bytes_read"], 0)
        # re-run: nothing new, nothing re-read
        r2 = run(self.env.cfg)
        self.assertEqual(r2["bytes_read"], 0)
        self.assertEqual(r2["new"]["outcomes"], 0)
        self.assertEqual(len(list(iter_jsonl(self.env.cfg.outcomes))), 2)
        self.assertEqual(r2["new"]["memory_new"], 0)

    def test_incremental_append(self):
        run(self.env.cfg)
        self.env.write("proj/sess-1.jsonl", [
            asst([{"type": "tool_use", "id": "toolu_D", "name": "Agent", "input": {"subagent_type": "fr-scriber", "description": "docs",
                                                                                  "prompt": "Document the retry helper."}}], 60, "m9")], mode="a")
        r = run(self.env.cfg)
        self.assertGreater(r["bytes_read"], 0)
        ids = [o["id"] for o in iter_jsonl(self.env.cfg.outcomes)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("out:toolu_D", ids)  # no sidechain and idle -> settled as re-dispatch (no final report)

    def test_partial_line_not_consumed(self):
        p = self.env.projects / "proj/sess-1.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write('{"type": "user", "timestamp": "')  # torn line (writer mid-flush)
        run(self.env.cfg)
        with open(p, "a", encoding="utf-8") as f:
            f.write(ts(70) + '", "message": {"role": "user", "content": "one more request about retries please"}}\n')
        run(self.env.cfg)
        turns = [s for s in self.env.cfg.files_dir.glob("*.json") if json.loads(s.read_text())["agent_id"] is None]
        st = json.loads(turns[0].read_text())
        self.assertEqual(st["turns"][-1]["user"]["t"], "one more request about retries please")
        self.assertEqual(st["signals"].get("bad_line", 0), 0)

    def test_taint_never_stored(self):
        run(self.env.cfg)
        blob = "".join(p.read_text(encoding="utf-8") for p in self.env.data.rglob("*") if p.is_file() and p.suffix in (".json", ".jsonl"))
        self.assertNotIn("SHOULD_NEVER_BE_STORED", blob)
        self.assertNotIn("api/.env", blob)
        main = [json.loads(p.read_text()) for p in self.env.cfg.files_dir.glob("*.json")]
        main = [s for s in main if s["agent_id"] is None][0]
        self.assertTrue(main["turns"][0]["tainted"])
        self.assertEqual(main["tainted_tools"], 1)

    def test_secret_in_report_is_scrubbed_and_dropped_from_sft(self):
        fake = "sk-ant-api03-" + "Ab3" * 15
        self.env.write("proj/sess-1/subagents/agent-bbb.jsonl", [asst([{"type": "text", "text": "leaked " + fake}], 39.5, "t2")], mode="a")
        run(self.env.cfg)
        blob = "".join(p.read_text(encoding="utf-8") for p in self.env.data.rglob("*") if p.is_file())
        self.assertNotIn(fake, blob)

    def test_lock_excludes_second_run(self):
        lock = ProcessLock(self.env.cfg.lock_file)
        self.assertTrue(lock.acquire())
        try:
            r = run(self.env.cfg)
            self.assertIn("skipped", r)
        finally:
            lock.release()
        self.assertNotIn("skipped", run(self.env.cfg))


class MemoryTests(unittest.TestCase):
    def setUp(self):
        os.environ["INGEST_EMBED_EXE"] = "Z:/missing/embed_texts.exe"
        self.env = Env()

    def tearDown(self):
        os.environ.pop("INGEST_EMBED_EXE", None)
        self.env.tmp.cleanup()

    def test_credit_and_prune(self):
        m = MemoryStore(self.env.cfg)
        m.add("The fix was to cap the retry delay at thirty seconds in the backoff loop.", "fix", "p", "s0", "src1", OLD, 1.0)
        m.add("The fix was to cap the retry delay at thirty seconds in the backoff loop.", "fix", "p", "s9", "src2", OLD, 1.0)
        self.assertEqual(len(m.items), 1)
        it = next(iter(m.items.values()))
        self.assertEqual(it["seen"], 2)
        m.save()
        for i in range(4):  # four retrievals in sessions that later failed
            m.recall("retry delay backoff fix", k=1, session=f"bad{i}")
        m = MemoryStore(self.env.cfg)
        future = datetime.now(timezone.utc).timestamp() + 60
        m.credit({f"bad{i}": [(future, 0.0)] for i in range(4)})
        m.refresh_and_prune()
        it = next(iter(m.items.values()))
        self.assertEqual(it["retrievals"], 4)
        self.assertLess(it["value"], 0.3)
        self.assertEqual(it["status"], "pruned")
        # crediting is idempotent
        before = (it["pos"], it["neg"])
        m.credit({f"bad{i}": [(future, 0.0)] for i in range(4)})
        self.assertEqual(before, (it["pos"], it["neg"]))

    def test_hash_embed_matches_rust_semantics(self):
        v = hash_embed("retry backoff retry", 64)
        self.assertAlmostEqual(sum(x * x for x in v), 1.0, places=6)
        self.assertEqual(hash_embed("", 8), [0.0] * 8)


class UnitTests(unittest.TestCase):
    def test_parse_grade(self):
        self.assertEqual(parse_grade('{"met_task":0.5,"verified":0.3,"clean":0.2,"reason":"ok"}')[0], 1.0)
        self.assertEqual(parse_grade('noise {"met_task":0.25,"verified":0,"clean":0.1} tail')[0], 0.35)
        self.assertEqual(parse_grade('{"met_task":9,"verified":9,"clean":9}')[0], 1.0)
        self.assertIsNone(parse_grade("not json"))

    def test_classify(self):
        self.assertEqual(ex.classify("build x", "", "", {}, {"no_report": True})[0], "re-dispatch")
        self.assertEqual(ex.classify("fix round 2 for D6", "done, tests pass", "", {}, {})[0], "fix-round")
        self.assertEqual(ex.classify("test", "TEST_BLOCK r1", "", {}, {})[0], "defect")
        self.assertEqual(ex.classify("test", "all good: 12 passed, 0 failed", "", {}, {})[0], "success")
        self.assertEqual(ex.classify("ship", "git push was blocked by the permission classifier", "", {}, {})[0], "blocked")

    def test_memory_chunks(self):
        ch = ex.memory_chunks("- The root cause was a race in the session store when two requests refresh at once.\n- short\n- hello world")
        self.assertEqual([t for t, _ in ch], ["root-cause"])

    def test_leader_clusters(self):
        a = np.array([[1, 0], [0.99, 0.141], [0, 1]], dtype="f4")
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        self.assertEqual(leader_clusters(a, 0.9), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
