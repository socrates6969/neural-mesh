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
    """stub for subprocess.run of `memory import -i <file>`: records (cmd, cwd, entries read from the chunk file) and answers like
    the CLI ("Entries: N"); ok(entry) decides which entries are stored; boom=True raises instead."""

    def __init__(self, ok=lambda entry: True, boom=False):
        self.calls = []
        self.files = []
        self.ok = ok
        self.boom = boom

    def __call__(self, cmd, cwd):
        f = cmd[cmd.index("-i") + 1]
        self.files.append(f)
        entries = json.loads(Path(f).read_text(encoding="utf-8"))["entries"]
        self.calls.append((list(cmd), str(cwd), entries))
        if self.boom:
            raise subprocess.TimeoutExpired(cmd, 180)
        n = sum(1 for e in entries if self.ok(e))
        return subprocess.CompletedProcess(cmd, 0, stdout=f"[OK] Imported from {f}\n  - Entries: {n}\n  - Vectors: {n}\n", stderr="")


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
        for cmd, _, _ in fake.calls:
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
        st, _ = self.run_mirror(Fake(ok=lambda e: "lifelong/outcome" not in e["key"]))
        self.assertEqual(st["exit"], 3)
        self.assertGreater(st["stored"], 0)
        self.assertGreater(st["failed"], 0)

    def test_none_stored_is_4(self):
        st, _ = self.run_mirror(Fake(ok=lambda e: False))
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


class ImportFile(Base):
    def test_argv_never_contains_values(self):
        st, fake = self.run_mirror()
        self.assertEqual(st["exit"], 0)
        batch = M.build_batch(self.cfg)
        for cmd, _, entries in fake.calls:
            self.assertIn("import", cmd)
            self.assertNotIn("--value", cmd)
            self.assertNotIn("store", cmd)
            joined = " ".join(cmd)
            for b in batch:
                self.assertNotIn(b["value_json"], joined)
                self.assertNotIn(b["key"], joined)
            self.assertTrue(all(e["namespace"] == M.NAMESPACE for e in entries))
        self.assertEqual(sorted(e["key"] for _, _, es in fake.calls for e in es), sorted(b["key"] for b in batch))

    def test_chunk_file_deleted_after_success_and_on_error(self):
        _, fake = self.run_mirror()
        self.assertTrue(fake.files)
        self.assertFalse(any(os.path.exists(f) for f in fake.files))
        (self.cfg.state / "mirror_done.txt").unlink()
        boom = Fake(boom=True)
        st, _ = self.run_mirror(boom)
        self.assertEqual(st["exit"], 4)
        self.assertTrue(boom.files)
        self.assertFalse(any(os.path.exists(f) for f in boom.files))

    def test_chunks_of_200_and_partial_chunk_not_marked_done(self):
        items = [{"id": f"mem-x{i:04d}", "status": "active", "tag": "fact", "text": f"bulk item {i}", "project": "p"} for i in range(450)]
        with open(self.cfg.memory_dir / "items.jsonl", "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(x) + "\n" for x in items))
        fake = Fake(ok=lambda e: e["key"] != "lifelong/mem/x0300")
        st, _ = self.run_mirror(fake)
        self.assertEqual([len(es) for _, _, es in fake.calls], [200, 200, 58])
        self.assertEqual((st["chunks"], st["stored"], st["failed"], st["exit"]), (3, 457, 1, 3))
        # the chunk with the failure is retried (upsert) next run; the others are done
        fake2 = Fake()
        st2, _ = self.run_mirror(fake2)
        self.assertEqual(st2["todo"], 200)
        self.assertEqual(st2["exit"], 0)

    def test_cli_failure_exit_code_counts_as_nothing_stored(self):
        class Bad(Fake):
            def __call__(self, cmd, cwd):
                super().__call__(cmd, cwd)
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="[ERROR] Import error")
        st, _ = self.run_mirror(Bad())
        self.assertEqual((st["stored"], st["exit"]), (0, 4))


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


class DryRunSecretDrop(Base):
    """AC-2: a record that still has a SECRET-class string after scrub never reaches the mirror batch. Runs the real CLI entry
    point in-process (``python -m ingest mirror --dry-run``); nothing is called (dry run), the db is the repo default."""

    # assembled at runtime so no literal key sits in the source; each matches a SECRET pattern in ingest/scrub.py
    ANT = "sk-ant-" + "api03-" + "Zq7Xw2Lp9Rt4Vb8Nm3Kc6Hd1"                    # api_key (sk-ant-...)
    AWS = "AKIA" + "QX7Z2LP9RT4VB8NM"                                         # api_key (AKIA + 16)
    PEM_BODY = "MIIEpAIBAAKCAQEAzq7Xw2Lp9Rt4Vb8Nm3Kc6Hd1Fg5Js0"
    PEM = "-----BEGIN RSA " + "PRIVATE KEY-----\n" + PEM_BODY + "\n-----END RSA " + "PRIVATE KEY-----"   # private_key

    def setUp(self):
        super().setUp()
        bad_items = [{"id": "mem-s001", "status": "active", "tag": "fact", "project": "p", "text": f"use key {self.ANT} for the api"},
                     {"id": "mem-s002", "status": "active", "tag": "fact", "project": "p", "text": f"deploy key:\n{self.PEM}"}]
        with open(self.cfg.memory_dir / "items.jsonl", "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(x) + "\n" for x in bad_items))
        with open(self.cfg.outcomes, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": "o-s1", "agent": "a", "outcome": "success", "reward": 1.0,
                                "task_text": f"export AWS_ACCESS_KEY_ID={self.AWS} and retry"}) + "\n")

    def cli_dry_run(self):
        import io
        from contextlib import redirect_stdout
        from ingest.__main__ import main
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main(["--data", str(self.cfg.data), "mirror", "--dry-run"])
        return code, json.loads(buf.getvalue())

    def test_secret_records_dropped_counted_and_absent_from_batch_file(self):
        from ingest.scrub import scrub
        for s in (self.ANT, self.AWS, self.PEM):   # precondition: each really is SECRET-class for the scrubber
            self.assertTrue(scrub(s).secret, "fixture must be classified secret by ingest/scrub.py")
        code, st = self.cli_dry_run()
        self.assertEqual(code, 0)
        self.assertEqual(st["exit"], 0)
        self.assertTrue(st["dry_run"])
        self.assertEqual(st["dropped_secret"], 3, "the stats count every dropped secret record")
        self.assertEqual(st["candidates"], 8, "the 5 memories + 3 outcomes without secrets are kept")
        batch_file = self.cfg.data / "mirror_batch.jsonl"
        text = batch_file.read_text(encoding="utf-8")
        rows = [json.loads(x) for x in text.splitlines() if x.strip()]
        keys = {r["key"] for r in rows}
        self.assertNotIn("lifelong/mem/s001", keys)
        self.assertNotIn("lifelong/mem/s002", keys)
        self.assertNotIn(f"lifelong/outcome/{M.sha16('o-s1')}", keys)
        for needle in (self.ANT, self.AWS, self.PEM_BODY, "sk-ant-", "AKIA", "PRIVATE KEY", "Zq7Xw2Lp9Rt4Vb8Nm3Kc6Hd1", "QX7Z2LP9RT4VB8NM"):
            self.assertNotIn(needle, text, "no secret substring anywhere in the batch file")

    def test_positive_control_normal_records_kept(self):
        _, st = self.cli_dry_run()
        rows = [json.loads(x) for x in (self.cfg.data / "mirror_batch.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        keys = {r["key"] for r in rows}
        self.assertEqual(len(rows), 8)
        for i in range(5):
            self.assertIn(f"lifelong/mem/{i:04d}", keys)
        for i in range(3):
            self.assertIn(f"lifelong/outcome/{M.sha16(f'o{i}')}", keys)
        self.assertIn("memory number 0", "".join(r["value"] for r in rows))


if __name__ == "__main__":
    unittest.main()
