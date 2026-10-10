"""Mirror job guard rails: db location, pinned CLI, exit codes, ingest lock. No network, no npx (a stub runner records the
commands that would have been run)."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from ingest import mirror as M
from ingest.config import Config
from ingest.util import ProcessLock


class Fake:
    """stub for subprocess.run: records (cmd, cwd); ok(cmd) decides the return code."""

    def __init__(self, ok=lambda cmd: True):
        self.calls = []
        self.ok = ok

    def __call__(self, cmd, cwd):
        self.calls.append((list(cmd), str(cwd)))
        return subprocess.CompletedProcess(cmd, 0 if self.ok(cmd) else 1, stdout="", stderr="")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.repo = root / "repo"
        (self.repo / ".swarm").mkdir(parents=True)
        self.outside = root / "elsewhere"
        self.outside.mkdir()
        self.cfg = Config()
        self.cfg.data = self.repo / "data" / "lifelong"
        (self.cfg.memory_dir).mkdir(parents=True)
        items = [{"id": f"mem-{i:04d}", "status": "active", "tag": "fact", "text": f"memory number {i}", "project": "p"} for i in range(5)]
        (self.cfg.memory_dir / "items.jsonl").write_text("".join(json.dumps(x) + "\n" for x in items), encoding="utf-8")
        outs = [{"id": f"o{i}", "agent": "a", "outcome": "success", "reward": 1.0, "task_text": f"task {i}"} for i in range(3)]
        self.cfg.outcomes.write_text("".join(json.dumps(x) + "\n" for x in outs), encoding="utf-8")
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        os.environ.pop("NEURAL_MESH_RUFLO_DB", None)
        os.environ.pop("NEURAL_MESH_RUFLO_CLI", None)
        self.which = mock.patch.object(M.shutil, "which", lambda name: "C:/fake/npx.cmd")
        self.which.start()

    def tearDown(self):
        self.which.stop()
        self.env.stop()
        self.tmp.cleanup()

    def run_mirror(self, fake=None, **kw):
        fake = fake or Fake()
        return M.mirror(self.cfg, repo=self.repo, runner=fake, **kw), fake


class DbLocation(Base):
    def test_default_db_is_the_repo_swarm(self):
        st, fake = self.run_mirror()
        self.assertEqual(os.path.normcase(st["db_path"]), os.path.normcase(os.path.realpath(self.repo / ".swarm" / "memory.db")))
        self.assertEqual(st["exit"], 0)
        cmd = fake.calls[0][0]
        self.assertEqual(os.path.normcase(cmd[cmd.index("--path") + 1]), os.path.normcase(st["db_path"]))
        self.assertEqual(os.path.normcase(fake.calls[0][1]), os.path.normcase(str(self.repo)), "cwd is the repo")

    def test_env_overrides_default(self):
        os.environ["NEURAL_MESH_RUFLO_DB"] = str(self.repo / "data" / "other.db")
        self.assertEqual(M.default_db(self.repo), self.repo / "data" / "other.db")

    def test_external_db_refused_with_exit_5(self):
        st, fake = self.run_mirror(db_path=str(self.outside / "memory.db"))
        self.assertEqual(st["exit"], 5)
        self.assertIn("outside the repo", st["error"])
        self.assertEqual(fake.calls, [])

    def test_external_db_allowed_with_flag(self):
        st, fake = self.run_mirror(db_path=str(self.outside / "memory.db"), allow_external=True)
        self.assertEqual(st["exit"], 0)
        self.assertTrue(fake.calls)

    def test_case_and_dotdot_tricks(self):
        sneaky = str(self.repo / ".swarm" / ".." / ".." / "elsewhere" / "memory.db")
        self.assertFalse(M.check_db(sneaky, repo=self.repo)[0])
        if sys.platform == "win32":
            self.assertTrue(M.check_db(str(self.repo).upper() + "\\.swarm\\memory.db", repo=self.repo)[0])
        self.assertFalse(M.check_db(str(self.repo) + "-evil/memory.db", repo=self.repo)[0], "prefix of the repo name is not inside")

    @unittest.skipUnless(sys.platform == "win32", "junctions are Windows-only")
    def test_junction_escape_refused(self):
        import _winapi
        j = self.repo / "linked"
        _winapi.CreateJunction(str(self.outside), str(j))
        try:
            ok, resolved, why = M.check_db(j / "memory.db", repo=self.repo)
            self.assertFalse(ok, why)
            st, fake = self.run_mirror(db_path=str(j / "memory.db"))
            self.assertEqual(st["exit"], 5)
            self.assertEqual(fake.calls, [])
        finally:
            os.rmdir(j)

    def test_unc_always_refused(self):
        for p in (r"\\server\share\memory.db", "//server/share/memory.db", r"\\?\UNC\server\share\memory.db"):
            ok, _, why = M.check_db(p, allow_external=True, repo=self.repo)
            self.assertFalse(ok, p)
            self.assertIn("UNC", why)

    def test_mapped_drive_refused(self):
        with mock.patch.object(M, "_is_remote_drive", lambda p: True):
            ok, _, why = M.check_db(self.repo / ".swarm" / "memory.db", allow_external=True, repo=self.repo)
        self.assertFalse(ok)
        self.assertIn("network", why)


class CliPin(Base):
    def test_cli_version_pinned(self):
        self.assertEqual(M.CLI_VERSION, "3.56.3")
        _, fake = self.run_mirror()
        for cmd, _ in fake.calls:
            self.assertIn("@claude-flow/cli@3.56.3", cmd)
            self.assertFalse(any("@latest" in c for c in cmd))

    def test_env_overrides_pin(self):
        os.environ["NEURAL_MESH_RUFLO_CLI"] = "3.57.0"
        self.assertEqual(M.cli_package(), "@claude-flow/cli@3.57.0")


class ExitCodes(Base):
    def test_all_stored_then_noop_is_0(self):
        st, _ = self.run_mirror()
        self.assertEqual((st["exit"], st["stored"], st["todo"]), (0, 8, 8))
        st, fake = self.run_mirror()
        self.assertEqual((st["exit"], st["todo"]), (0, 0))
        self.assertEqual(fake.calls, [])

    def test_partial_is_3(self):
        st, _ = self.run_mirror(Fake(ok=lambda cmd: "lifelong/outcome" not in " ".join(cmd)))
        self.assertEqual(st["exit"], 3)
        self.assertGreater(st["stored"], 0)
        self.assertGreater(st["failed"], 0)

    def test_none_stored_is_4(self):
        st, _ = self.run_mirror(Fake(ok=lambda cmd: False))
        self.assertEqual((st["exit"], st["stored"]), (4, 0))

    def test_npx_missing_is_4(self):
        with mock.patch.object(M.shutil, "which", lambda name: None):
            st, _ = self.run_mirror()
        self.assertEqual(st["exit"], 4)
        self.assertIn("npx", st["error"])

    def test_cli_returns_the_exit_code(self):
        from ingest.__main__ import main
        with mock.patch.object(M, "mirror", lambda *a, **k: {"exit": 3}), mock.patch("builtins.print"):
            self.assertEqual(main(["mirror"]), 3)


class Locking(Base):
    def test_busy_lock_exits_75(self):
        lk = ProcessLock(self.cfg.lock_file)
        self.assertTrue(lk.acquire())
        try:
            st, fake = self.run_mirror()
        finally:
            lk.release()
        self.assertEqual(st["exit"], 75)
        self.assertEqual(fake.calls, [])

    def test_waits_on_the_ingest_lock(self):
        lk = ProcessLock(self.cfg.lock_file)
        self.assertTrue(lk.acquire())
        t = threading.Timer(1.0, lk.release)
        t.start()
        t0 = time.monotonic()
        st, _ = self.run_mirror(wait_s=10)
        t.join()
        self.assertEqual(st["exit"], 0)
        self.assertGreaterEqual(time.monotonic() - t0, 0.9)


if __name__ == "__main__":
    unittest.main()
