import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from agentkit import install  # noqa: E402
from train import runner  # noqa: E402
from agentkit import protocol  # noqa: E402

PY = "C:/Python312/python.exe"
SETTINGS = {"hooks": {"PreToolUse": [{"matcher": "Task|Agent", "hooks": [
    {"type": "command", "command": PY + " C:/Users/mariu/neural-mesh/hooks/mesh_assist.py", "timeout": 10}]}],
    "PostToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "x"}]}]}, "model": "opus"}


class InstallTest(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.p = self.d / "settings.json"
        self.p.write_text(json.dumps(SETTINGS), encoding="utf-8")

    def test_install_idempotent_with_backup(self):
        msg = install.install(self.p, Path("C:/repo"))
        self.assertIn("installed", msg)
        s = json.loads(self.p.read_text(encoding="utf-8"))
        post = s["hooks"]["PostToolUse"]
        self.assertEqual(post[0], SETTINGS["hooks"]["PostToolUse"][0])
        entry = post[1]
        self.assertEqual(entry["matcher"], "Task|Agent")
        self.assertEqual(entry["hooks"][0], {"type": "command", "command": PY + " C:/repo/hooks/mesh_capture.py",
                                             "timeout": 10})
        self.assertEqual(s["model"], "opus")
        self.assertEqual(s["hooks"]["PreToolUse"], SETTINGS["hooks"]["PreToolUse"])
        self.assertEqual(len(list(self.d.glob("settings.json.bak-agentkit-*"))), 1)
        before = self.p.read_text(encoding="utf-8")
        self.assertIn("already installed", install.install(self.p, Path("C:/repo")))
        self.assertEqual(self.p.read_text(encoding="utf-8"), before)

    def test_dry_run_and_invalid_json(self):
        before = self.p.read_text(encoding="utf-8")
        out = install.install(self.p, Path("C:/repo"), dry_run=True)
        self.assertIn("mesh_capture.py", out)
        self.assertEqual(self.p.read_text(encoding="utf-8"), before)
        self.p.write_text("{broken", encoding="utf-8")
        with self.assertRaises(SystemExit):
            install.install(self.p, Path("C:/repo"))
        self.assertEqual(self.p.read_text(encoding="utf-8"), "{broken")

    def test_joins_existing_task_agent_group_and_quotes_spaces(self):
        s = {"hooks": {"PostToolUse": [{"matcher": "Task|Agent", "hooks": [{"type": "command", "command": "other"}]}]}}
        new, changed = install.plan(s, Path("C:/my repo"), python="C:/Program Files/py.exe")
        self.assertTrue(changed)
        hooks = new["hooks"]["PostToolUse"][0]["hooks"]
        self.assertEqual(len(new["hooks"]["PostToolUse"]), 1)
        self.assertEqual(hooks[1]["command"], '"C:/Program Files/py.exe" "C:/my repo/hooks/mesh_capture.py"')
        self.assertEqual(len(s["hooks"]["PostToolUse"][0]["hooks"]), 1)      # input not mutated


class RunnerPromptTest(unittest.TestCase):
    def setUp(self):
        os.environ["MESHTRAIN_EXPERIENCE"] = "off"

    def test_protocol_per_field_and_identical_arms(self):
        for field, mode in runner.PROTOCOL_MODE.items():
            for kind in ("int", "numeric", "mcq", "code", "stdio"):
                t = {"id": "t", "field": field, "kind": kind, "prompt": "Solve it."}
                base, packed = runner.build_prompt(t, None), runner.build_prompt(t, "PACK NOTES")
                self.assertTrue(base.startswith(protocol.render(mode)), field)
                self.assertEqual(packed.replace(
                    "\n\n---\n\nKnowledge from the neural mesh for this kind of task (use what helps, ignore what does "
                    "not):\nPACK NOTES", ""), base)              # the arms differ only by the pack
                self.assertIn(runner.FORMAT[kind], base)
                self.assertIn("JSON block", runner.FORMAT[kind])

    def test_answer_line_still_required(self):
        for kind in ("int", "numeric", "mcq"):
            self.assertIn("ANSWER:", runner.FORMAT[kind])
        for kind in ("code", "stdio"):
            self.assertIn("python block", runner.FORMAT[kind])

    def test_protocol_off(self):
        os.environ["MESHTRAIN_PROTOCOL"] = "off"
        try:
            p = runner.build_prompt({"id": "t", "field": "math", "kind": "int", "prompt": "Solve it."}, None)
        finally:
            os.environ.pop("MESHTRAIN_PROTOCOL", None)
        self.assertNotIn(protocol.MARKER, p)


if __name__ == "__main__":
    unittest.main()