"""S1: the UserPromptSubmit shadow hook (llm/hook.py) and the offline shadow-drain. No Ollama, no MiniLM, no judge, Rust stubbed."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from llm import frontend_live as fl, hook
from llm.tests.test_frontend_live import ROWS, Stub, decide_as

REPO = Path(__file__).resolve().parents[2]


class ShadowHook(unittest.TestCase):
    def test_enqueue_writes_one_line_and_rejects_garbage(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "q.jsonl"
            self.assertTrue(hook.enqueue(json.dumps({"prompt": "fix it", "session_id": "S", "cwd": "C:\\Users\\x\\proj\\"}), path=p, now=0))
            self.assertTrue(hook.enqueue("\ufeff" + json.dumps({"prompt": "with bom"}), path=p, now=0))
            for bad in ("", "not json", "[]", json.dumps({"prompt": ""}), json.dumps({"prompt": 5})):
                self.assertFalse(hook.enqueue(bad, path=p))
            recs = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[1]["prompt"], "with bom")
        self.assertEqual({k: recs[0][k] for k in ("ts", "session", "project", "prompt")},
                         {"ts": "1970-01-01T00:00:00.000Z", "session": "S", "project": "proj", "prompt": "fix it"})
        self.assertEqual(recs[0]["sha256"], fl.sha("fix it"))

    def test_ts_is_sub_second_and_old_lines_stay_readable(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "q.jsonl"
            hook.enqueue(json.dumps({"prompt": "a", "session_id": "S"}), path=p, now=1760000000.4567)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": "2025-10-09T08:53:20Z", "sha256": fl.sha("b"), "session": "S", "prompt": "b"}) + "\n")
            q = fl.read_shadow(p)
        self.assertEqual(q[0]["ts"], "2025-10-09T08:53:20.456Z")
        self.assertAlmostEqual(fl.parse_ts(q[0]["ts"]), 1760000000.456, places=3)
        self.assertEqual(fl.parse_ts(q[1]["ts"]), 1760000000.0)      # an old whole-second line

    def _run_hook(self, stdin: str, env_extra: dict):
        with tempfile.TemporaryDirectory() as d:
            env = {**os.environ, "MESH_LLM_MODELS": d, **env_extra}
            t = time.monotonic()
            r = subprocess.run([sys.executable, "-S", "-m", "llm", "hook", "--shadow"], input=stdin.encode(), capture_output=True, cwd=REPO, env=env, timeout=30)
            ms = (time.monotonic() - t) * 1000
            q = Path(d) / "frontend_shadow.jsonl"
            lines = q.read_text(encoding="utf-8").splitlines() if q.exists() else []
        return r, lines, ms

    def test_hook_prints_nothing_logs_and_exits_0(self):
        r, lines, ms = self._run_hook(json.dumps({"prompt": "héllo wörld", "session_id": "S1", "cwd": "C:/a/b"}), {})
        self.assertEqual((r.returncode, r.stdout, r.stderr), (0, b"", b""))
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["prompt"], "héllo wörld")
        self.assertLess(ms, 3000)      # the in-process budget is checked by the watchdog; wall time here includes interpreter start-up
        r, lines, _ = self._run_hook("garbage{", {})
        self.assertEqual((r.returncode, r.stdout, lines), (0, b"", []))

    def test_off_switch(self):
        r, lines, _ = self._run_hook(json.dumps({"prompt": "x y z"}), {"NEURAL_MESH_FE_SHADOW": "off"})
        self.assertEqual((r.returncode, r.stdout, lines), (0, b"", []))

    def test_proposed_settings_file(self):
        p = REPO / "hooks" / "settings-frontend-shadow.json"
        cfg = json.loads(p.read_text(encoding="utf-8"))
        h = cfg["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        self.assertEqual((h["type"], h["timeout"]), ("command", 5))
        self.assertIn("-m llm hook --shadow", h["command"])
        self.assertFalse((REPO / ".claude" / "settings.json").exists() and "llm hook --shadow" in (REPO / ".claude" / "settings.json").read_text(encoding="utf-8"))


class ShadowDrain(unittest.TestCase):
    def test_drain_joins_judges_and_keeps_unjoined_pending(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            q = d / "q.jsonl"
            hook.enqueue(json.dumps({"prompt": "Earlier Question", "session_id": "s0", "cwd": "/p"}), path=q, now=0)
            hook.enqueue(json.dumps({"prompt": "never answered", "session_id": "s9", "cwd": "/p"}), path=q, now=time.time())
            rows = [dict(ROWS[0], ts="1970-01-01T00:00:30Z", response="local answer")]
            kw = dict(backend_factory=Stub, decide=decide_as("local"), rows=rows, shadow_path=q, drained_path=d / "out.jsonl",
                      state_path=d / "state.json", log=lambda *a: None)
            st = fl.shadow_drain(**kw)
            self.assertEqual((st["drained"], st["joined"], st["waiting"]), (1, 1, 1))
            out = [json.loads(x) for x in (d / "out.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual((out[0]["would_be"], out[0]["claude_id"], out[0]["score"], out[0]["wrong_if_served"]), ("local", "sft:1", 1.0, False))
            self.assertNotIn("prompt", out[0])
            self.assertEqual(fl.shadow_drain(**kw)["drained"], 0)                       # idempotent
            st = fl.shadow_drain(**{**kw, "now": time.time() + fl.JOIN_WAIT_S + 10})   # the unjoined one expires
            self.assertEqual((st["drained"], st["unjoined_expired"]), (1, 1))

    def test_graded_answer_is_never_in_bank_chunks_or_cache(self):
        """F1: Claude's joined answer is 30 s OLDER than the hook ts (whole-second truncation / clock skew). It is a verified,
        same-project, same-text row, so without the exclusion it would be a cache hit, the max_sim neighbour and the top chunk."""
        hook_t = 1760000000.0
        ts = lambda dt: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(hook_t + dt))  # noqa: E731
        verified = {"project": "p", "outcome": "success", "judge_score": 1.0, "judge_model": "j"}
        claude = {"id": "sft:claude", "ts": ts(-30), "session": "s1", "instruction": "Fix The Parser", "response": "CLAUDE GRADED ANSWER", **verified}
        twin = {"id": "sft:twin", "ts": ts(-45), "session": "s1", "instruction": "fix the parser please", "response": "CLAUDE GRADED ANSWER", **verified}
        other = {"id": "sft:old", "ts": ts(-7200), "session": "s0", "instruction": "an older unrelated question", "response": "old answer", **verified}
        rows = [claude, twin, other]

        class Rec(Stub):
            def __init__(self):
                super().__init__()
                self.encoded = []

            def encode(self, texts):
                self.encoded += list(texts)
                return super().encode(texts)
        be = Rec()
        dec = decide_as("local")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            q = d / "q.jsonl"
            hook.enqueue(json.dumps({"prompt": "fix the parser", "session_id": "s1", "cwd": "/p"}), path=q, now=hook_t)
            e = fl.read_shadow(q)[0]
            self.assertEqual(fl.join_claude(e, rows)["id"], "sft:claude")
            bank = fl.shadow_bank(e, rows, fl.join_claude(e, rows))
            self.assertEqual([r["id"] for r in bank], ["sft:old"])
            # without the exclusion the strict ts rule alone would leak both rows into the cache
            leaky = fl.compute_signals(e["prompt"], "p", e["ts"], rows, Stub(), skip_local=True, local_timeout=1)
            self.assertEqual((leaky["cache_why"], leaky["answer"]), ("hit", "CLAUDE GRADED ANSWER"))
            fl.shadow_drain(backend_factory=lambda: be, decide=dec, rows=rows, shadow_path=q, drained_path=d / "out.jsonl",
                            state_path=d / "state.json", log=lambda *a: None)
            out = json.loads((d / "out.jsonl").read_text(encoding="utf-8"))
        cfg, sig, chunks = dec.calls[0]
        self.assertEqual(sig["cache-hit"], 0)
        self.assertEqual([c["id"] for c in chunks], ["sft:old"])
        self.assertTrue(all("CLAUDE GRADED ANSWER" not in c["text"] for c in chunks))
        self.assertTrue(all("CLAUDE GRADED ANSWER" not in t for t in be.encoded))          # not even embedded for max_sim
        self.assertEqual((out["cache_why"], out["claude_id"], out["score"]), ("no_match", "sft:claude", 0.0))   # graded the LOCAL answer


if __name__ == "__main__":
    unittest.main()
