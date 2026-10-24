"""lean_check tests: verdict parsing offline; real checks need Docker and the meshtrain-lean images."""
import unittest

from tools.mesh_mcp import lean_tools as L
from tools.mesh_mcp.common import docker_available

GOOD = "theorem add_comm' (a b : Nat) : a + b = b + a := by omega\n#eval 2 + 2\n"
BAD = "theorem wrong (a : Nat) : a + 1 = a := by omega\n"
SORRY = "theorem later (a b : Nat) : a * b = b * a := by sorry\n"
MATHLIB = "import Mathlib\nexample (x : Real) : 0 <= x ^ 2 := sq_nonneg x\n"


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
        out = L.lean_check(MATHLIB, timeout_s=300)
        if docker_available(L.MATHLIB):
            self.assertTrue(out.startswith("VERIFIED"), out)
        else:
            self.assertIn("Mathlib is not available", out)


if __name__ == "__main__":
    unittest.main()
