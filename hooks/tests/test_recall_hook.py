"""Unit tests for hooks/recall_hook.py: gating, scrub-on-output, timeout, silence on error. Synthetic memories only; no network.
Run from the repo root:  python -m unittest hooks.tests.test_recall_hook -v"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "hooks"))
import recall_hook as rh  # noqa: E402
from ingest.config import Config  # noqa: E402

HOOK = ROOT / "hooks" / "recall_hook.py"
FAKE_KEY = "AKIA" + "ABCDEFGHIJKLMNOP"          # assembled at runtime: no secret-looking literal in the repo


def item(i, text, project="demo-proj", tag="decision", **kw):
    return {"id": f"mem:{i:016x}", "seq": i, "text": text, "tag": tag, "project": project, "sessions": ["s"], "sources": ["x"], "ts_first": "", "ts_last": "",
            "seen": 1, "retrievals": 0, "pos": 0.5, "neg": 0.0, "decay_ts": "", "value": 0.5, "status": "active", "vec_key": None, **kw}


ITEMS = [
    item(1, "The frobnicator_cache invalidation uses a generation counter in src/frob/cache.rs; never clear the map directly."),
    item(2, "Convention: every migration for the ledger_sync service gets a rollback script next to it in db/migrations."),
    item(3, "Payment gateway retries use exponential backoff capped at 30 seconds, configured in gateway_config.toml."),
    item(4, "frobnicator_cache invalidation in the OTHER project works differently, see src/frob/cache.rs there.", project="other-proj"),
] + [item(10 + k, f"Unrelated note number {k} about colour palettes and typography for the marketing site.") for k in range(12)]


def make_data(items=ITEMS):
    d = tempfile.mkdtemp(prefix="recall-test-")
    mem = Path(d, "memory")
    mem.mkdir()
    (mem / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items), encoding="utf-8")
    return d


def run_hook(payload, data, extra_env=None, raw=None):
    env = dict(os.environ, INGEST_DATA=data, **(extra_env or {}))
    env.pop("NEURAL_MESH_RECALL", None) if not (extra_env and "NEURAL_MESH_RECALL" in extra_env) else None
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-S", str(HOOK)], input=raw if raw is not None else json.dumps(payload), capture_output=True, text=True,
                       encoding="utf-8", env=env, timeout=20)
    return r, time.perf_counter() - t


class GateTests(unittest.TestCase):
    def test_skips_chitchat_slash_and_short(self):
        for p in ["", "thanks", "ok", "continue", "/plugin install foo", "yes please", "what a nice day today"]:
            self.assertLess(rh.gate_score(p, ["demo-proj"]), rh.MIN_GATE_SCORE, p)

    def test_passes_code_and_project_prompts(self):
        for p in ["why does src/frob/cache.rs clear the map here?", "what does `frobnicator_cache` do on invalidation",
                  "last time we decided how the demo-proj retries work, remind me", "fix the load_items() call in the parser module"]:
            self.assertGreaterEqual(rh.gate_score(p, ["demo-proj"]), rh.MIN_GATE_SCORE, p)

    def test_generic_project_names_do_not_count(self):
        self.assertLess(rh.gate_score("clean up the repo and tell me about it", ["repo", "mariu"]), rh.MIN_GATE_SCORE)


class BuildBlockTests(unittest.TestCase):
    def cfg(self, items=ITEMS):
        self.data = make_data(items)
        return Config(data=Path(self.data))

    def test_relevant_prompt_recalls_same_project_only(self):
        cfg = self.cfg()
        out = rh.build_block("how does the frobnicator_cache invalidation in src/frob/cache.rs work?", r"C:\work\demo-proj", "sess-1", cfg=cfg)
        self.assertIn("generation counter", out)
        self.assertNotIn("OTHER project", out)           # project scoping: no cross-project leakage
        self.assertLessEqual(len(out), rh.MAX_CHARS)
        self.assertLessEqual(out.count("\n- "), rh.MAX_K)

    def test_nothing_passes_prints_nothing_and_logs_nothing(self):
        cfg = self.cfg()
        out = rh.build_block("explain the quantum chromodynamics lagrangian in `qcd_lagrangian.py` please", r"C:\work\demo-proj", "sess-2", cfg=cfg)
        self.assertEqual(out, "")
        self.assertFalse((cfg.memory_dir / "retrievals.jsonl").exists())

    def test_only_printed_memories_are_logged_with_session(self):
        cfg = self.cfg()
        out = rh.build_block("how does the frobnicator_cache invalidation in src/frob/cache.rs work?", r"C:\work\demo-proj", "sess-3", cfg=cfg)
        rows = [json.loads(l) for l in (cfg.memory_dir / "retrievals.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session"], "sess-3")
        self.assertEqual(rows[0]["ids"], [ITEMS[0]["id"]])
        self.assertEqual(out.count("\n- "), 1)

    def test_no_project_match_means_no_pool(self):
        cfg = self.cfg()
        self.assertEqual(rh.build_block("how does the frobnicator_cache invalidation in src/frob/cache.rs work?", r"C:\work\nothing-here", "s", cfg=cfg), "")

    def test_missing_memory_dir_is_silent(self):
        cfg = Config(data=Path(tempfile.mkdtemp()) / "nope")
        self.assertEqual(rh.build_block("how does `frobnicator_cache` work in src/frob/cache.rs", r"C:\work\demo-proj", "s", cfg=cfg), "")


class ScrubOnOutputTests(unittest.TestCase):
    def test_secret_memory_is_never_printed(self):
        leaky = item(1, f"The frobnicator_cache invalidation deploy key is {FAKE_KEY} stored in src/frob/cache.rs for the demo-proj service.")
        clean = item(2, "The frobnicator_cache invalidation uses a generation counter in src/frob/cache.rs and never clears the map directly.")
        cfg = Config(data=Path(make_data([leaky, clean] + ITEMS[4:])))
        out = rh.build_block("how does the frobnicator_cache invalidation in src/frob/cache.rs work?", r"C:\work\demo-proj", "s", cfg=cfg)
        self.assertNotIn(FAKE_KEY, out)
        self.assertNotIn("deploy key", out)
        self.assertIn("generation counter", out)

    def test_pii_is_redacted_in_output(self):
        pii = item(1, "The frobnicator_cache invalidation owner is bob@example.com per src/frob/cache.rs, ask before touching the demo-proj map.")
        cfg = Config(data=Path(make_data([pii] + ITEMS[4:])))
        out = rh.build_block("who owns the frobnicator_cache invalidation in src/frob/cache.rs?", r"C:\work\demo-proj", "s", cfg=cfg)
        self.assertNotIn("bob@example.com", out)
        self.assertIn("[REDACTED:email]", out)

    def test_format_block_rechecks_whole_block(self):
        out, printed = rh.format_block([item(1, f"token {FAKE_KEY}")])
        self.assertEqual((out, printed), ("", []))


class TimeoutAndSilenceTests(unittest.TestCase):
    def test_run_with_deadline_gives_up(self):
        t = time.perf_counter()
        done, out = rh.run_with_deadline(lambda: time.sleep(5) or "late", 0.2)
        self.assertFalse(done)
        self.assertEqual(out, "")
        self.assertLess(time.perf_counter() - t, 1.0)

    def test_run_with_deadline_swallows_exceptions(self):
        def boom():
            raise RuntimeError("x")
        self.assertEqual(rh.run_with_deadline(boom, 1.0), (True, ""))

    def test_subprocess_prints_for_relevant_prompt(self):
        data = make_data()
        r, _ = run_hook({"prompt": "how does the frobnicator_cache invalidation in src/frob/cache.rs work?", "cwd": r"C:\work\demo-proj", "session_id": "t"}, data)
        self.assertEqual(r.returncode, 0)
        self.assertIn("generation counter", r.stdout)

    def test_subprocess_budget_exceeded_prints_nothing(self):
        data = make_data()
        r, dt = run_hook({"prompt": "how does the frobnicator_cache invalidation in src/frob/cache.rs work?", "cwd": r"C:\work\demo-proj", "session_id": "t"},
                         data, {"NEURAL_MESH_RECALL_BUDGET_S": "0"})
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        self.assertLess(dt, 1.5)

    def test_subprocess_garbage_and_empty_stdin_are_silent(self):
        data = make_data()
        for raw in ["not json {{{", "", "[1,2,3]", '{"prompt": null}', '{"prompt": 5, "cwd": 7}']:
            r, _ = run_hook(None, data, raw=raw)
            self.assertEqual((r.returncode, r.stdout), (0, ""), raw)

    def test_subprocess_corrupt_store_is_silent(self):
        data = make_data()
        Path(data, "memory", "items.jsonl").write_bytes(b"\xff\xfe garbage\n{broken")
        r, _ = run_hook({"prompt": "how does the frobnicator_cache invalidation in src/frob/cache.rs work?", "cwd": r"C:\work\demo-proj"}, data)
        self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_disabled_by_env_is_silent_and_fast(self):
        data = make_data()
        r, dt = run_hook({"prompt": "how does the frobnicator_cache invalidation in src/frob/cache.rs work?", "cwd": r"C:\work\demo-proj"}, data,
                         {"NEURAL_MESH_RECALL": "off"})
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        self.assertFalse(Path(data, "memory", "retrievals.jsonl").exists())
        self.assertLess(dt, 0.5)      # generous bound for CI noise; measured ~35 ms (hooks/README.md)


if __name__ == "__main__":
    unittest.main()
