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
                         {"ts": "1970-01-01T00:00:00Z", "session": "S", "project": "proj", "prompt": "fix it"})
        self.assertEqual(recs[0]["sha256"], fl.sha("fix it"))

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


if __name__ == "__main__":
    unittest.main()
