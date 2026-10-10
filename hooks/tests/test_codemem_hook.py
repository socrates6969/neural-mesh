"""codemem hook: end-to-end through the real exe (skipped when it is not built). Silence and exit 0 on every problem, toggle, scoping, size cap.
Run: python -m unittest hooks.tests.test_codemem_hook"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = next((p for p in (ROOT / "data" / "codemem" / "bin" / "codemem.exe", ROOT / "target" / "release" / "codemem.exe") if p.exists()), None)


@unittest.skipUnless(EXE, "codemem.exe not built (cargo rustc --release --bin codemem -- -C strip=symbols)")
class CodememHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="cm-hook-test-"))
        cls.repo = cls.tmp / "repo"
        (cls.repo / "src").mkdir(parents=True)
        (cls.repo / "src" / "tenant.ts").write_text(
            "export async function withTenant(id: string) {\n  await tx.$executeRaw`select set_config('app.tenant_id', ${id}, true)`;\n}\n", encoding="utf8")
        (cls.repo / "src" / "filler.ts").write_text("export function filler() { return 1; }\n", encoding="utf8")
        (cls.repo / ".env").write_text("API_KEY=sk-ant-api03-abcdefghijklmnopqrstuv\n", encoding="utf8")
        cls.data = cls.tmp / "data"
        cls.data.mkdir()
        cfg = {"split": True, "repos": [{"name": "t", "path": str(cls.repo).replace("\\", "/"), "terms": ["tenant"]}]}
        (cls.data / "config.json").write_text(json.dumps(cfg), encoding="utf8")
        r = cls.run_exe(["update"])
        assert r.returncode == 0, r.stderr

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def run_exe(cls, args, stdin=None, **env):
        e = dict(os.environ, NEURAL_MESH_CODEMEM_DIR=str(cls.data), NEURAL_MESH_CODEMEM_LOG="off")
        e.pop("NEURAL_MESH_CODEMEM", None)
        e.update(env)
        return subprocess.run([str(EXE), *args], input=stdin, capture_output=True, text=True, env=e, timeout=20)

    def hook(self, prompt, cwd=None, **env):
        payload = json.dumps({"prompt": prompt, "cwd": cwd or str(self.repo), "session_id": "t"})
        return self.run_exe(["hook"], stdin=payload, **env)

    Q = "where is `withTenant` and the set_config call defined in src/tenant.ts"

    def test_fires_with_pointer_and_stays_under_token_cap(self):
        r = self.hook(self.Q)
        self.assertEqual(r.returncode, 0)
        self.assertIn("src/tenant.ts:", r.stdout)
        self.assertLessEqual(len(r.stdout), 900)
        self.assertLessEqual(r.stdout.count("\n- "), 5)

    def test_secret_files_are_not_indexed_or_leaked(self):
        r = self.run_exe(["query", "sk-ant-api03 API_KEY abcdefghijklmnopqrstuv", "--repo", "t"])
        self.assertNotIn("abcdefghijklmnop", r.stdout)
        self.assertNotIn(".env", r.stdout.replace("codemem repo", ""))

    def test_silent_on_chatter_wrong_cwd_bad_json_and_empty_input(self):
        for prompt, cwd in (("continue", None), ("thanks, that looks good to me", None), (self.Q, str(self.tmp / "elsewhere"))):
            r = self.hook(prompt, cwd)
            self.assertEqual((r.returncode, r.stdout), (0, ""), (prompt, cwd))
        for raw in ("not json", "", "{}", '{"prompt": 5, "cwd": null}', "\x00\xff"):
            r = self.run_exe(["hook"], stdin=raw)
            self.assertEqual((r.returncode, r.stdout), (0, ""), raw)

    def test_off_switch(self):
        r = self.hook(self.Q, NEURAL_MESH_CODEMEM="off")
        self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_missing_index_or_config_is_silent(self):
        empty = self.tmp / "empty"
        empty.mkdir(exist_ok=True)
        r = self.hook(self.Q, NEURAL_MESH_CODEMEM_DIR=str(empty))
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        r = self.hook(self.Q, NEURAL_MESH_CODEMEM_DIR=str(self.tmp / "does-not-exist"))
        self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_zero_budget_never_blocks_or_errors(self):
        r = self.hook(self.Q, NEURAL_MESH_CODEMEM_BUDGET_MS="0")
        self.assertEqual(r.returncode, 0)
        self.assertIn(r.stdout[:9], ("", "[codemem]"))

    def test_corrupt_index_is_silent(self):
        bad = self.tmp / "bad"
        (bad / "t").mkdir(parents=True)
        shutil.copy(self.data / "config.json", bad / "config.json")
        (bad / "t" / "index.cmem").write_bytes(b"CMEMIDX1" + b"\x00" * 50)
        r = self.hook(self.Q, NEURAL_MESH_CODEMEM_DIR=str(bad))
        self.assertEqual((r.returncode, r.stdout), (0, ""))

    def test_incremental_update_reports_only_changed_files(self):
        r1 = json.loads(self.run_exe(["update", "--repo", "t"]).stdout.splitlines()[0])
        self.assertEqual((r1["changed"], r1["removed"]), (0, 0))
        (self.repo / "src" / "extra.ts").write_text("export function addedLater() { return 2; }\n", encoding="utf8")
        r2 = json.loads(self.run_exe(["update", "--repo", "t"]).stdout.splitlines()[0])
        self.assertEqual((r2["changed"], r2["removed"]), (1, 0))
        self.assertLess(r2["ms"], 5000)
        (self.repo / "src" / "extra.ts").unlink()
        r3 = json.loads(self.run_exe(["update", "--repo", "t"]).stdout.splitlines()[0])
        self.assertEqual(r3["removed"], 1)


if __name__ == "__main__":
    unittest.main()
