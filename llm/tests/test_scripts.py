"""PowerShell nightly scripts (Windows PowerShell 5.1) with STUB pythons: stderr output must not be fatal, exit codes must
propagate (worst stage wins, llm 75 = busy = 0), logs rotate above 5 MB. Nothing real runs (no ingest, no llm, no npx).
Run: llm/.venv/Scripts/python -m unittest llm.tests.test_scripts"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PS = shutil.which("powershell")
INGEST = REPO / "ingest" / "scripts" / "run_ingest.ps1"
NIGHTLY = REPO / "llm" / "scripts" / "run_nightly.ps1"

# exit code: STUB_<KIND>_RC for `mirror`, `nightly` (llm) else STUB_RUN_RC; always writes to stdout AND stderr
STUB = r"""@echo off
echo stub-out %*
echo stub-err %* 1>&2
set RC=%STUB_RUN_RC%
echo %* | findstr /C:"mirror" >nul && set RC=%STUB_MIRROR_RC%
echo %* | findstr /C:"nightly" >nul && set RC=%STUB_LLM_RC%
if "%RC%"=="" set RC=0
exit /b %RC%
"""


@unittest.skipUnless(sys.platform == "win32" and PS, "Windows PowerShell only")
class Scripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.stub = self.d / "stubpy.cmd"
        self.stub.write_text(STUB, encoding="ascii")
        self.log = self.d / "logs" / "test.log"

    def tearDown(self):
        self.tmp.cleanup()

    def ps(self, script, *args, **rc):
        env = dict(os.environ)
        for k in ("RUN", "MIRROR", "LLM"):
            env.pop(f"STUB_{k}_RC", None)
        env.update({f"STUB_{k.upper()}_RC": str(v) for k, v in rc.items()})
        cmd = [PS, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), *args, "-Log", str(self.log)]
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)

    def text(self, p=None):
        return (p or self.log).read_text(encoding="utf-8-sig", errors="replace")

    def test_ingest_stderr_not_fatal_and_exit_code_propagates(self):
        r = self.ps(INGEST, "-Python", str(self.stub), "-NoMirror", run=3)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        t = self.text()
        self.assertIn("stub-err", t)
        self.assertIn("ingest run exit 3", t)
        self.assertNotIn("FATAL", t)

    def test_ingest_mirror_refused_is_reported(self):
        r = self.ps(INGEST, "-Python", str(self.stub), run=0, mirror=5)
        self.assertEqual(r.returncode, 5)
        self.assertIn("mirror REFUSED", self.text())

    def test_nightly_llm_failure_propagates_and_mirror_still_runs(self):
        r = self.ps(NIGHTLY, "-Python", str(self.stub), "-IngestPython", str(self.stub), run=0, llm=3, mirror=0)
        self.assertEqual(r.returncode, 3, r.stdout + r.stderr)
        t = self.text()
        self.assertIn("stub-err -u -m llm nightly", t)
        self.assertIn("llm nightly exit 3", t)
        self.assertIn("mirror ok", t)
        self.assertIn("nightly done (exit 3)", t)
        self.assertNotIn("FATAL", t)
        self.assertTrue(Path(str(self.log) + ".ingest").exists(), "ingest logs next to the test log, not the real one")

    def test_nightly_llm_busy_is_0_and_worst_code_wins(self):
        r = self.ps(NIGHTLY, "-Python", str(self.stub), "-IngestPython", str(self.stub), run=0, llm=75, mirror=0)
        self.assertEqual(r.returncode, 0)
        self.assertIn("llm busy", self.text())
        r = self.ps(NIGHTLY, "-Python", str(self.stub), "-IngestPython", str(self.stub), run=0, llm=0, mirror=4)
        self.assertEqual(r.returncode, 4)

    def test_nightly_ingest_failure_skips_llm(self):
        r = self.ps(NIGHTLY, "-Python", str(self.stub), "-IngestPython", str(self.stub), "-NoMirror", run=2, llm=0)
        self.assertEqual(r.returncode, 2)
        t = self.text()
        self.assertIn("skipping the llm round", t)
        self.assertNotIn("llm nightly exit", t)

    def test_missing_python_is_fatal_exit_1(self):
        r = self.ps(NIGHTLY, "-Python", str(self.d / "nope.exe"), "-IngestPython", str(self.stub), "-NoMirror", run=0)
        self.assertEqual(r.returncode, 1)
        self.assertIn("FATAL", self.text())

    def test_ingest_detach_appends_to_the_log(self):
        import time
        self.log.parent.mkdir(parents=True)
        self.log.write_text("earlier run line\n", encoding="utf-8")
        for _ in range(2):   # two detached runs: both appended, nothing truncated
            r = self.ps(INGEST, "-Python", str(self.stub), "-Detach", run=0)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                done = self.text().count("ingest run exit 0")
                if done >= _ + 1:
                    break
                time.sleep(0.25)
        time.sleep(0.5)
        t = self.text()
        self.assertIn("earlier run line", t, "-Detach must append, not truncate")
        self.assertEqual(t.count("ingest run exit 0"), 2)
        self.assertEqual(t.count("stub-out -m ingest run"), 2)
        self.assertNotIn("mirror", t, "the detached run is ingest only")
        self.assertNotIn("FATAL", t)

    def test_log_rotation(self):
        self.log.parent.mkdir(parents=True)
        self.log.write_bytes(b"x" * (5 * 2**20 + 10))
        Path(str(self.log) + ".1").write_text("older", encoding="utf-8")
        r = self.ps(INGEST, "-Python", str(self.stub), "-NoMirror", run=0)
        self.assertEqual(r.returncode, 0)
        self.assertGreater(Path(str(self.log) + ".1").stat().st_size, 5 * 2**20)
        self.assertEqual(Path(str(self.log) + ".2").read_text(encoding="utf-8"), "older")
        self.assertLess(self.log.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
