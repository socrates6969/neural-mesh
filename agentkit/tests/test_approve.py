import io
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from agentkit import approve  # noqa: E402


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

    def approve(self, entry_id, by="human"):
        self.approved.append((entry_id, by))
        return True


E = {"id": "7", "task_id": "S1", "agent": "coder", "kind": "fact", "content": "parser handles CRLF",
     "verified": False, "claimed_verified": True}


class ApproveTest(unittest.TestCase):
    def run_main(self, ws, answer="APPROVE", tty=True, argv=("7",)):
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            rc = approve.main(list(argv), ws=ws, ask=lambda _p: answer, stdin=TTY(tty), stdout=TTY(tty))
        finally:
            sys.stdout = old
        return rc, out.getvalue()

    def test_shows_then_approves_on_exact_confirmation(self):
        ws = WS(dict(E))
        rc, out = self.run_main(ws)
        self.assertEqual(rc, 0)
        self.assertIn("parser handles CRLF", out)
        self.assertIn("agent claims verified", out)
        self.assertEqual(len(ws.approved), 1)
        self.assertTrue(ws.approved[0][1].startswith("human:"))

    def test_anything_else_cancels(self):
        for ans in ("", "y", "yes", "approve", " APPROVE!"):
            ws = WS(dict(E))
            rc, _ = self.run_main(ws, answer=ans)
            self.assertEqual((rc, ws.approved), (1, []), ans)

    def test_non_interactive_refused(self):
        ws = WS(dict(E))
        rc, _ = self.run_main(ws, tty=False)
        self.assertEqual((rc, ws.approved), (3, []))

    def test_backend_without_approve_refused(self):
        rc, _ = self.run_main(WS(dict(E), can=False))
        self.assertEqual(rc, 4)

    def test_missing_entry_and_usage(self):
        self.assertEqual(self.run_main(WS(None))[0], 1)
        self.assertEqual(self.run_main(WS(dict(E)), argv=())[0], 2)

    def test_already_verified_noop(self):
        ws = WS(dict(E, verified=True, verified_by="grader"))
        rc, out = self.run_main(ws)
        self.assertEqual((rc, ws.approved), (0, []))
        self.assertIn("already verified", out)


if __name__ == "__main__":
    unittest.main()