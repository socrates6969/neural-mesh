"""lean_check tests: verdict parsing offline; real checks need Docker and the meshtrain-lean images."""
import os
import unittest

from tools.mesh_mcp import lean_tools as L
from tools.mesh_mcp.common import docker_available

GOOD = "theorem add_comm' (a b : Nat) : a + b = b + a := by omega\n#eval 2 + 2\n"
BAD = "theorem wrong (a : Nat) : a + 1 = a := by omega\n"
SORRY = "theorem later (a b : Nat) : a * b = b * a := by sorry\n"
MATHLIB_PART = "import Mathlib.Analysis.SpecialFunctions.Pow.Real\nexample (x : Real) : 0 <= x ^ 2 := sq_nonneg x\n"
MATHLIB_FULL = "import Mathlib\nexample (x : Real) : 0 <= x ^ 2 := sq_nonneg x\n"
# measured cold start of a bare `import Mathlib`: 403-482 s (6 GB, 2 CPUs); far over a unit-test budget
FULL_MATHLIB_SECS = 480


class VerdictTest(unittest.TestCase):
    def test_verdicts(self):
        self.assertEqual(L.verdict(0, "4")[0], "VERIFIED")
        v, msgs = L.verdict(1, "/w/Main.lean:1:42: error: omega could not prove the goal\nmore")
        self.assertEqual(v, "FAILED")
        self.assertEqual(msgs, ["Main.lean:1:42: error: omega could not prove the goal"])
        for q in ("'sorry'", "`sorry`"):
            self.assertEqual(L.verdict(0, "/w/Main.lean:1:8: warning: declaration uses " + q)[0],
                             "INCOMPLETE (uses sorry)")
        self.assertEqual(L.verdict(124, "")[0], "TIMEOUT")
        self.assertEqual(L.kind_of(MATHLIB_FULL), "mathlib_full")
        self.assertEqual(L.kind_of(MATHLIB_PART), "mathlib_part")
        self.assertEqual(L.kind_of(GOOD), "core")
        with self.assertRaises(ValueError):
            L.lean_check("  ")


@unittest.skipUnless(docker_available(L.CORE), "meshtrain-lean:core not built")
class LeanSandboxTest(unittest.TestCase):
    def test_valid_proof_verified(self):
        out = L.lean_check(GOOD)
        self.assertTrue(out.startswith("VERIFIED"), out)

    def test_wrong_statement_fails_with_location(self):
        out = L.lean_check(BAD)
        self.assertTrue(out.startswith("FAILED"), out)
        self.assertIn("Main.lean:1:", out)
        self.assertIn("error", out)

    def test_sorry_is_incomplete(self):
        self.assertTrue(L.lean_check(SORRY).startswith("INCOMPLETE"))

    def test_mathlib_import(self):
        out = L.lean_check(MATHLIB_PART)  # default budget: 240 s; measured ~25 s
        if docker_available(L.MATHLIB):
            self.assertTrue(out.startswith("VERIFIED"), out)
        else:
            self.assertIn("Mathlib is not available", out)

    @unittest.skipUnless(os.environ.get("MESH_LEAN_FULL_MATHLIB") == "1",
                         "bare `import Mathlib` cold start measured at %d s; set MESH_LEAN_FULL_MATHLIB=1 to run"
                         % FULL_MATHLIB_SECS)
    def test_full_mathlib_import(self):
        out = L.lean_check(MATHLIB_FULL)  # default budget: 600 s
        self.assertTrue(out.startswith("VERIFIED"), out)


if __name__ == "__main__":
    unittest.main()
