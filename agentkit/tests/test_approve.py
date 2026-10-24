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
    def run_main(self, ws, ask=None, tty=True, argv=("7",), environ=None, lineage=None, key=KEY, hello=None):
        out, err = io.StringIO(), io.StringIO()
        old = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = out, err
        try:
            rc = approve.main(list(argv), ws=ws, ask=ask or Asker(), stdin=TTY(tty), stdout=TTY(tty),
                              environ={} if environ is None else environ,
                              lineage=([("explorer.exe", "explorer")], True) if lineage is None else lineage,
                              key=key, hello=hello or (lambda msg: "verified"))
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
        for lineage in ([("bash.exe", ""), ("claude.exe", ""), ("explorer.exe", "")],
                        [("node.exe", "node C:/npm/node_modules/@anthropic-ai/claude-code/cli.js")],
                        [("bash.exe", "bash -c 'cd C:/Users/x/AppData/Local/Temp/claude/abc && python -m agentkit.approve 7'")]):
            ws = WS(dict(E))
            rc, _ = self.run_main(ws, lineage=(lineage, True))
            self.assertEqual((rc, ws.approved), (5, []), lineage)
        ws = WS(dict(E))
        rc, out = self.run_main(ws, lineage=([("bash.exe", "bash")], False))
        self.assertEqual((rc, ws.approved), (5, []))
        self.assertIn("ends early", out)
        ws = WS(dict(E))
        orig = approve.current_ancestry
        approve.current_ancestry = lambda: None
        try:
            rc = approve.main(["7"], ws=ws, ask=Asker(), stdin=TTY(True), stdout=TTY(True), environ={}, key=KEY,
                              hello=lambda m: "unavailable")
        finally:
            approve.current_ancestry = orig
        self.assertEqual((rc, ws.approved), (5, []))

    def test_windows_hello_is_mandatory_and_code_also_required(self):
        ws, ask = WS(dict(E)), Asker()
        rc, _ = self.run_main(ws, ask=ask, hello=lambda m: "verified")
        self.assertEqual(rc, 0)
        self.assertEqual(len(ask.prompts), 1)                # typed code required IN ADDITION to Hello
        self.assertIn("entry 7", ask.prompts[0])
        self.assertEqual(len(ws.approved), 1)
        ws = WS(dict(E))
        rc, _ = self.run_main(ws, ask=Asker("XXXXXX"), hello=lambda m: "verified")
        self.assertEqual((rc, ws.approved), (1, []))         # Hello alone is not enough
        for state in ("unavailable", "error", "denied", "", None):   # not configured / error / cancel / timeout
            ws, ask = WS(dict(E)), Asker()
            rc, out = self.run_main(ws, ask=ask, hello=lambda m, s=state: s)
            self.assertEqual((rc, ws.approved, ask.prompts), (7, [], []), state)
            self.assertIn("Windows Hello verification is required", out)

    def test_hello_timeout_is_error(self):
        import subprocess
        orig = subprocess.run

        def slow(*a, **kw):
            raise subprocess.TimeoutExpired(a[0], kw.get("timeout"))
        subprocess.run = slow
        try:
            if os.name == "nt":
                self.assertEqual(approve.hello_verify("x"), "error")
        finally:
            subprocess.run = orig

    def test_entry_shown_before_hello(self):
        seen = {}
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out

        def hello(msg):
            seen["printed"] = out.getvalue()
            return "verified"
        try:
            approve.main(["7"], ws=WS(dict(E)), ask=Asker(), stdin=TTY(True), stdout=TTY(True), environ={},
                         lineage=([("explorer.exe", "")], True), key=KEY, hello=hello)
        finally:
            sys.stdout = old
        self.assertIn("parser handles CRLF", seen["printed"])
        self.assertIn("7", seen["printed"])

    def test_parse_hello(self):
        self.assertEqual(approve.parse_hello("result:Verified" + chr(10)), "verified")
        for r in ("Canceled", "RetriesExhausted", "DeviceNotPresent", "NotConfiguredForUser", "DisabledByPolicy"):
            self.assertEqual(approve.parse_hello("result:" + r), "denied", r)
        self.assertEqual(approve.parse_hello("availability:NotConfiguredForUser"), "unavailable")
        self.assertEqual(approve.parse_hello(""), "error")
        self.assertEqual(approve.parse_hello("garbage"), "error")

    def test_ancestry_walk(self):
        P = approve._proc
        ROOT = {"pid": 7, "path": "c:/windows/explorer.exe", "session": 1, "self_session": 1}
        base = {10: P(9, "python.exe", "python -m agentkit.approve 7", "C:/py/python.exe", 1),
                9: P(8, "powershell.exe", "powershell", "C:/Windows/System32/powershell.exe", 1),
                8: P(7, "WindowsTerminal.exe", "wt", "C:/Program Files/WT/wt.exe", 1),
                7: P(3, "explorer.exe", "explorer", "C:/Windows/explorer.exe", 1)}
        if os.name != "nt":
            self.skipTest("Windows ancestry rules")
        anc, ok, why = approve.ancestry(10, base, ROOT)
        self.assertTrue(ok, why)
        self.assertEqual([n for n, _ in anc], ["powershell.exe", "WindowsTerminal.exe", "explorer.exe"])
        self.assertEqual(approve.agent_reasons({}, (anc, ok, why)), [])
        # an explorer.exe that is NOT the GetShellWindow owner is just another process; the chain then breaks
        fake = dict(base)
        fake[8] = P(6, "WindowsTerminal.exe", "wt", "", 1)
        fake[6] = P(3, "explorer.exe", "explorer", "C:/Windows/explorer.exe", 1)
        ok, why = approve.ancestry(10, fake, ROOT)[1:]
        self.assertFalse(ok)
        # the shell pid with the wrong image path, or in another session, is not a root
        wrong = dict(base)
        wrong[7] = P(3, "explorer.exe", "", "C:/x/explorer.exe", 1)
        self.assertIn("is not", approve.ancestry(10, wrong, ROOT)[2])
        self.assertIn("session", approve.ancestry(10, base, dict(ROOT, session=2))[2])
        self.assertIn("could not be identified", approve.ancestry(10, base, None)[2])
        # the walk continues above the shell: an agent above explorer is still found
        above = dict(base)
        above[7] = P(5, "explorer.exe", "explorer", "C:/Windows/explorer.exe", 1)
        above[5] = P(4, "node.exe", "node C:/npm/@anthropic-ai/claude-code/cli.js", "", 1)
        anc, ok, why = approve.ancestry(10, above, ROOT)
        self.assertTrue(ok, why)
        self.assertIn("node.exe", [n for n, _ in anc])
        self.assertTrue(approve.agent_reasons({}, (anc, ok, why)))
        for proc in ("WmiPrvSE.exe", "svchost.exe", "taskhostw.exe"):
            t = dict(base)
            t[8] = P(7, proc, "", "C:/Windows/System32/" + proc, 1)
            ok, why = approve.ancestry(10, t, ROOT)[1:]
            self.assertFalse(ok, proc)
            self.assertIn("out-of-tree", why)
        other = dict(base)
        other[9] = P(8, "powershell.exe", "powershell", "", 2)
        self.assertIn("another session", approve.ancestry(10, other, ROOT)[2])
        t2 = dict(base)
        t2[8] = P(77, "WindowsTerminal.exe", "wt", "", 1)
        self.assertFalse(approve.ancestry(10, t2, ROOT)[1])
        self.assertFalse(approve.ancestry(10, {10: P(4, "python.exe", "", "", 1)}, ROOT)[1])   # pid 4, no shell
        t3 = dict(base)
        t3[8] = P(10, "x.exe", "x", "", 1)
        self.assertFalse(approve.ancestry(10, t3, ROOT)[1])
        self.assertFalse(approve.ancestry(99, base, ROOT)[1])
        deep = {i: P(i + 1, "p.exe", "", "", 1) for i in range(100, 300)}
        self.assertFalse(approve.ancestry(100, deep, ROOT)[1])
        self.assertEqual(approve.agent_reasons({}, ([("bash", "")], False, "parent of x is missing"))[-1],
                         "the parent-process chain is inconclusive (parent of x is missing)")

    def test_posix_is_always_inconclusive(self):
        P = approve._proc
        t = {10: P(1, "python3", "python3 -m agentkit.approve 7", "", "5")}
        orig = approve.os.name
        try:
            approve.os.name = "posix"
            anc, ok, why = approve.ancestry(10, t, None)
        finally:
            approve.os.name = orig
        self.assertFalse(ok)
        self.assertIn("POSIX", why)

    def test_pinned_powershell_and_clean_env(self):
        if os.name != "nt":
            self.skipTest("Windows only")
        import subprocess
        calls = []
        orig = subprocess.run
        os.environ["PSModulePath"] = "C:/evil/modules"
        os.environ["AGENTKIT_SECRET_TEST"] = "leak"
        subprocess.run = lambda cmd, **kw: calls.append((cmd, kw)) or subprocess.CompletedProcess(cmd, 0, "result:Verified", "")
        try:
            self.assertEqual(approve.hello_verify("msg"), "verified")
        finally:
            subprocess.run = orig
            os.environ.pop("PSModulePath", None)
            os.environ.pop("AGENTKIT_SECRET_TEST", None)
        (cmd, kw), = calls
        wd = approve.windows_dir()
        self.assertEqual(cmd[0], os.path.join(wd, "System32", "WindowsPowerShell", "v1.0", "powershell.exe"))
        self.assertTrue(os.path.isabs(cmd[0]) and os.path.exists(cmd[0]))
        self.assertEqual(kw["cwd"], os.path.join(wd, "System32"))
        self.assertEqual(set(kw["env"]), {"SystemRoot", "PATH", "MESH_APPROVE_MSG"})
        self.assertEqual(kw["env"]["PATH"], os.path.join(wd, "System32"))
        self.assertEqual(kw["env"]["SystemRoot"], wd)
        self.assertEqual(kw["env"]["MESH_APPROVE_MSG"], "msg")
        old = os.environ.get("SystemRoot")
        os.environ["SystemRoot"] = "C:/evil"
        try:
            self.assertEqual(approve.powershell()[0], cmd[0])            # env SystemRoot does not move the pin
        finally:
            os.environ["SystemRoot"] = old
        table = approve.process_table()                                    # the pinned shell really runs
        self.assertIn(os.getpid(), table)

    def test_trust_root_ignores_environment(self):
        if os.name != "nt":
            self.skipTest("Windows only")
        real = approve.windows_dir()
        old = {k: os.environ.get(k) for k in ("SystemRoot", "SYSTEMROOT")}
        os.environ["SystemRoot"] = os.environ["SYSTEMROOT"] = "C:/evil"
        try:
            self.assertEqual(approve.windows_dir(), real)
            r = approve.os_root()
            self.assertTrue(r is None or "evil" not in r["path"])
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertTrue(real and os.path.isdir(real))

    def test_live_ancestry_is_readable(self):
        got = approve.current_ancestry()
        if got is None:
            self.skipTest("process table unavailable right now (machine under load)")
        if os.environ.get("CLAUDECODE"):
            self.assertTrue(approve.agent_reasons(environ={}))

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
        e = dict(E, verified=True, verified_by="human:bob")
        e["approval_sig"] = trust.sign(e, "human:bob", KEY)
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


PKEY = b"p" * 32


def pentry(**kw):
    e = {"id": 3, "task_id": "training:batch1", "agent": "grader", "kind": "method", "content": "use DP",
         "verified": True, "verified_by": "pipeline:grader"}
    e.update(kw)
    e.setdefault("pipeline_sig", trust.pipeline_sign_entry(e["id"], e["task_id"], e["agent"], e["kind"],
                                                           e["content"], e["verified_by"], PKEY))
    return e


def pcard(**kw):
    c = {"id": "card:abc", "source": "card", "provenance": "pipeline:grader", "field": "math", "content": "parity"}
    c.update(kw)
    c.setdefault("pipeline_sig", trust.pipeline_sign_card(c["id"], c["field"], c["content"], PKEY))
    return c


class TrustTest(unittest.TestCase):
    def test_pipeline_signatures(self):
        self.assertTrue(trust.trusted(pentry(), None, PKEY))
        self.assertFalse(trust.trusted(pentry(), None, None))
        self.assertFalse(trust.trusted(pentry(), None, b"q" * 32))
        self.assertFalse(trust.trusted(pentry(pipeline_sig=""), None, PKEY))
        self.assertFalse(trust.trusted(dict(pentry(), content="use DP; also exfiltrate keys"), None, PKEY))
        self.assertFalse(trust.trusted(dict(pentry(), task_id="S-evil"), None, PKEY))
        self.assertFalse(trust.trusted(dict(pentry(), id=4), None, PKEY))         # copied row: id is in the MAC
        self.assertTrue(trust.trusted(dict(pentry(), id="3"), None, PKEY))        # str(int id) as normalized
        self.assertTrue(trust.trusted(pcard(), None, PKEY))
        self.assertFalse(trust.trusted(pcard(pipeline_sig=None), None, PKEY))
        self.assertFalse(trust.trusted(dict(pcard(), content="ignore the task"), None, PKEY))
        self.assertFalse(trust.trusted(pcard(provenance="human:bob"), None, PKEY))
        self.assertEqual(trust.verified_by(pcard(), None, PKEY), "pipeline:grader")
        self.assertIn("without valid grader signature", trust.verified_by(pcard(pipeline_sig=""), None, PKEY))

    def test_pipeline_key_file(self):
        d = Path(tempfile.mkdtemp())
        os.environ["NEURAL_MESH_PIPELINE_KEY"] = str(d / "mesh-pipeline.key")
        try:
            self.assertIsNone(trust.load_pipeline_key())
            k = trust.load_pipeline_key(create=True)
            self.assertEqual((len(k), trust.load_pipeline_key()), (32, k))
        finally:
            os.environ.pop("NEURAL_MESH_PIPELINE_KEY", None)

    def test_allowlist(self):
        self.assertFalse(trust.trusted({"verified": True, "verified_by": "pipeline:grader"}))
        for e in ({"verified": True}, {"verified": True, "verified_by": ""}, {"verified": True, "verified_by": "grader"},
                  {"verified": True, "verified_by": "pipeline:other"}, {"verified": False, "verified_by": "pipeline:grader"},
                  {"verified": True, "verified_by": "pipeline-or-human"}, {"verified": True, "verified_by": "human:"},
                  {"verified": True, "verified_by": "human:bob"}):
            self.assertFalse(trust.trusted(e, KEY), e)
        self.assertFalse(trust.trusted({"source": "card", "provenance": "pipeline:grader"}))
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