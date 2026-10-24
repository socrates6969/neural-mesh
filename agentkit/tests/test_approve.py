import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from agentkit import approve, trust  # noqa: E402

KEY = b"a" * 32
ESC = chr(27)


class TTY(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


class WS:
    def __init__(self, entry=None, can=True):
        self.entry, self.can, self.approved = entry, can, []

    def can_approve(self):
        return self.can

    def get(self, entry_id):
        return self.entry

    def approve(self, entry_id, by, approval_sig):
        self.approved.append((entry_id, by, approval_sig))
        return True


E = {"id": "7", "task_id": "S1", "agent": "coder", "kind": "fact", "content": "parser handles CRLF",
     "verified": False, "claimed_verified": True}


class Asker:
    """Reads the code from the prompt and types it back (or a fixed answer)."""
    def __init__(self, mode="echo"):
        self.mode, self.prompts = mode, []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        code = re.search(r"type this code: (\S+)", prompt).group(1)
        return {"echo": code, "lower": code.lower(), "word": "APPROVE", "empty": ""}.get(self.mode, self.mode)


class ApproveTest(unittest.TestCase):
    def run_main(self, ws, ask=None, tty=True, argv=("7",), environ=None, parents=None, key=KEY):
        out, err = io.StringIO(), io.StringIO()
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            rc = approve.main(list(argv), ws=ws, ask=ask or Asker(), stdin=TTY(tty), stdout=TTY(tty),
                              environ={} if environ is None else environ, parents=[] if parents is None else parents,
                              key=key)
        finally:
            sys.stdout, sys.stderr = old
        return rc, out.getvalue() + err.getvalue()

    def test_shows_then_approves_with_nonce_and_signature(self):
        ws = WS(dict(E))
        ask = Asker()
        rc, out = self.run_main(ws, ask=ask)
        self.assertEqual(rc, 0)
        self.assertIn("parser handles CRLF", out)
        (eid, by, sig), = ws.approved
        self.assertTrue(by.startswith("human:"))
        self.assertTrue(trust.human_ok(dict(E, verified=True, verified_by=by, approval_sig=sig), KEY))
        self.assertFalse(trust.human_ok(dict(E, content="other", verified=True, verified_by=by, approval_sig=sig), KEY))

    def test_nonce_is_random_per_run(self):
        codes = set()
        for _ in range(5):
            ask = Asker()
            self.run_main(WS(dict(E)), ask=ask)
            codes.add(re.search(r"type this code: (\S+)", ask.prompts[0]).group(1))
        self.assertEqual(len(codes), 5)
        self.assertTrue(all(len(c) == approve.NONCE_LEN for c in codes))

    def test_wrong_code_cancels(self):
        for mode in ("word", "lower", "empty", "XXXXXX"):
            ws = WS(dict(E))
            rc, _ = self.run_main(ws, ask=Asker(mode))
            self.assertEqual((rc, ws.approved), (1, []), mode)

    def test_refuses_inside_agent_or_claude_code(self):
        for env in ({"CLAUDECODE": "1"}, {"CLAUDE_CODE_ENTRYPOINT": "cli"}, {"CLAUDE_SESSION_ID": "x"},
                    {"claude_code_session_id": "x"}):
            ws = WS(dict(E))
            rc, out = self.run_main(ws, environ=env)
            self.assertEqual((rc, ws.approved), (5, []), env)
            self.assertIn("agent", out)
        for parents in (["bash.exe", "claude.exe", "explorer.exe"], ["node", "claude"]):
            ws = WS(dict(E))
            rc, _ = self.run_main(ws, parents=parents)
            self.assertEqual((rc, ws.approved), (5, []), parents)
        ws = WS(dict(E))
        orig = approve.parent_names
        approve.parent_names = lambda: None                  # ancestors cannot be inspected: fail closed
        try:
            rc = approve.main(["7"], ws=ws, ask=Asker(), stdin=TTY(True), stdout=TTY(True), environ={}, key=KEY)
        finally:
            approve.parent_names = orig
        self.assertEqual((rc, ws.approved), (5, []))

    def test_this_test_process_is_detected(self):
        if os.environ.get("CLAUDECODE") or any(k.startswith("CLAUDE_CODE_") for k in os.environ):
            self.assertTrue(approve.agent_reasons())

    def test_non_interactive_refused(self):
        ws = WS(dict(E))
        rc, _ = self.run_main(ws, tty=False)
        self.assertEqual((rc, ws.approved), (3, []))

    def test_backend_without_signed_approve_refused(self):
        rc, _ = self.run_main(WS(dict(E), can=False))
        self.assertEqual(rc, 4)

    def test_no_key_refused(self):
        orig = trust.load_key
        trust.load_key = lambda create=False: None
        try:
            ws = WS(dict(E))
            rc, _ = self.run_main(ws, key=None)
        finally:
            trust.load_key = orig
        self.assertEqual((rc, ws.approved), (6, []))

    def test_missing_entry_and_usage(self):
        self.assertEqual(self.run_main(WS(None))[0], 1)
        self.assertEqual(self.run_main(WS(dict(E)), argv=())[0], 2)

    def test_already_trusted_noop(self):
        e = dict(E, verified=True, verified_by="pipeline:grader")
        ws = WS(e)
        rc, out = self.run_main(ws)
        self.assertEqual((rc, ws.approved), (0, []))
        self.assertIn("already verified", out)

    def test_display_strips_controls_and_ansi_with_warning(self):
        evil = ("ok" + ESC + "[2J" + ESC + "]0;pwned" + chr(7) + " text" + chr(13) + "fake" + chr(0) + chr(0x9b)
                + "31m" + chr(0x85) + "\nline2\tTab" + ESC + "c")
        ws = WS(dict(E, content=evil, agent="a" + ESC + "[31mred"))
        rc, out = self.run_main(ws, ask=Asker("empty"))
        self.assertEqual(rc, 1)
        self.assertNotIn(ESC, out)
        for c in (chr(7), chr(13), chr(0), chr(0x9b), chr(0x85)):
            self.assertNotIn(c, out)
        self.assertIn("line2\tTab", out)
        self.assertIn("WARNING", out)
        clean, stripped = approve.sanitize("plain\ttext\nok")
        self.assertEqual((clean, stripped), ("plain\ttext\nok", False))


class TrustTest(unittest.TestCase):
    def test_allowlist(self):
        self.assertTrue(trust.trusted({"verified": True, "verified_by": "pipeline:grader"}))
        for e in ({"verified": True}, {"verified": True, "verified_by": ""}, {"verified": True, "verified_by": "grader"},
                  {"verified": True, "verified_by": "pipeline:other"}, {"verified": False, "verified_by": "pipeline:grader"},
                  {"verified": True, "verified_by": "pipeline-or-human"}, {"verified": True, "verified_by": "human:"},
                  {"verified": True, "verified_by": "human:bob"}):
            self.assertFalse(trust.trusted(e, KEY), e)
        self.assertTrue(trust.trusted({"source": "card", "provenance": "pipeline:grader"}))
        self.assertFalse(trust.trusted({"source": "card"}))
        self.assertFalse(trust.trusted({"source": "card", "provenance": "human:bob"}))

    def test_human_signature(self):
        e = {"id": "1", "task_id": "S", "agent": "a", "kind": "fact", "content": "c", "verified": True,
             "verified_by": "human:bob"}
        s = dict(e, approval_sig=trust.sign(e, "human:bob", KEY))
        self.assertTrue(trust.trusted(s, KEY))
        self.assertFalse(trust.trusted(s, b"b" * 32))
        self.assertFalse(trust.trusted(s, None))
        self.assertFalse(trust.trusted(dict(s, verified_by="human:eve"), KEY))
        self.assertFalse(trust.trusted(dict(s, id="2"), KEY))
        self.assertEqual(trust.verified_by(s, KEY), "human:bob")
        self.assertIn("unsigned", trust.verified_by(s, None))

    def test_key_created_atomically_owner_only(self):
        d = Path(tempfile.mkdtemp())
        p = d / "mesh-approve.key"
        os.environ["NEURAL_MESH_APPROVE_KEY"] = str(p)
        try:
            self.assertIsNone(trust.load_key(create=False))
            k = trust.load_key(create=True)
            self.assertEqual(len(k), trust.KEY_BYTES)
            self.assertEqual(trust.load_key(), k)
            self.assertEqual(trust.load_key(create=True), k)          # never overwritten
            if os.name == "nt":
                import subprocess
                acl = subprocess.run(["icacls", str(p)], capture_output=True, text=True).stdout
                self.assertNotIn("Everyone", acl)
                self.assertNotIn("BUILTIN\\Users", acl)
            else:
                self.assertEqual(p.stat().st_mode & 0o077, 0)
            p.write_bytes(b"short")
            self.assertIsNone(trust.load_key(create=True))           # malformed: fail closed, not replaced
            self.assertEqual(p.read_bytes(), b"short")
        finally:
            os.environ.pop("NEURAL_MESH_APPROVE_KEY", None)

    def test_acl_failure_fails_closed(self):
        d = Path(tempfile.mkdtemp())
        p = d / "k"
        os.environ["NEURAL_MESH_APPROVE_KEY"] = str(p)
        orig = trust._restrict_owner_only
        trust._restrict_owner_only = lambda path: (_ for _ in ()).throw(OSError("acl"))
        try:
            self.assertIsNone(trust.load_key(create=True))
            self.assertFalse(p.exists())
        finally:
            trust._restrict_owner_only = orig
            os.environ.pop("NEURAL_MESH_APPROVE_KEY", None)


if __name__ == "__main__":
    unittest.main()