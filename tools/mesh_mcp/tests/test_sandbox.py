"""Sandbox tests (need Docker and the grader image): math_solve and causal_analysis on data with a known effect."""
import re
import tempfile
import unittest
from pathlib import Path

from tools.mesh_mcp.common import docker_available

try:
    import numpy as np
except ImportError:
    np = None


def synthetic_csv(path, n=2000, effect=2.0, seed=1):
    """Binary treatment confounded by z and w; true ATE = effect."""
    r = np.random.default_rng(seed)
    z, w = r.normal(size=n), r.normal(size=n)
    t = (0.8 * z + 0.5 * w + r.normal(size=n) > 0).astype(int)
    y = effect * t + 1.5 * z - 1.0 * w + r.normal(size=n)
    rows = ["z,w,t,y"] + ["%.6f,%.6f,%d,%.6f" % row for row in zip(z, w, t, y)]
    Path(path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def num(pattern, text):
    return [float(x) for x in re.search(pattern, text, re.M).groups()]


@unittest.skipUnless(docker_available(), "Docker grader image not available")
class MathSolveTest(unittest.TestCase):
    def test_sympy_integral(self):
        from tools.mesh_mcp.sandbox_tools import math_solve
        out = math_solve("import sympy as sp\nx = sp.symbols('x')\n"
                         "print(sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo)))\n"
                         "print(sp.integrate(x**3 * sp.sin(x), (x, 0, sp.pi)))")
        self.assertIn("exit code 0", out)
        self.assertIn("sqrt(pi)", out)
        self.assertIn("-6*pi + pi**3", out)

    def test_z3_sat_and_unsat(self):
        from tools.mesh_mcp.sandbox_tools import math_solve
        out = math_solve("from z3 import *\nx, y = Ints('x y')\ns = Solver()\ns.add(x + y == 10, x - y == 4)\n"
                         "print(s.check(), s.model()[x], s.model()[y])\n"
                         "s2 = Solver(); s2.add(x > 2, x < 1); print(s2.check())")
        self.assertIn("sat 7 3", out)
        self.assertIn("unsat", out)

    def test_no_network_and_errors_reported(self):
        from tools.mesh_mcp.sandbox_tools import math_solve
        out = math_solve("import socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
                         "    print('NET OPEN')\nexcept OSError as e:\n    print('blocked', type(e).__name__)")
        self.assertIn("blocked", out)
        self.assertNotIn("NET OPEN", out)
        out = math_solve("raise SystemExit('boom')")
        self.assertIn("exit code 1", out)
        self.assertIn("boom", out)


@unittest.skipUnless(docker_available() and np is not None, "Docker grader image or numpy not available")
class CausalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.csv = str(Path(cls.tmp.name, "syn.csv"))
        synthetic_csv(cls.csv)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, method):
        from tools.mesh_mcp.sandbox_tools import causal_analysis
        out = causal_analysis(self.csv, "t", "y", ["z", "w"], method)
        est = num(r"^ESTIMATE: (\S+)", out)[0]
        lo, hi = num(r"^95% CI: \[(\S+), (\S+)\]", out)
        self.assertAlmostEqual(est, 2.0, delta=0.15, msg=out)
        self.assertTrue(lo < 2.0 < hi, out)
        self.assertRegex(out, r"placebo_permutation: new_effect=\S+ p=\S+ -> PASS")
        self.assertRegex(out, r"random_common_cause: new_effect=\S+ p=\S+ -> PASS")
        self.assertIn("No unmeasured confounding", out)
        return out

    def test_backdoor_linear_recovers_known_effect(self):
        self.assertIn("OLS 95% interval", self.check("backdoor"))

    def test_propensity_weighting_recovers_known_effect(self):
        self.assertIn("bootstrap", self.check("propensity_score_weighting"))

    def test_unadjusted_estimate_is_confounded(self):
        from tools.mesh_mcp.sandbox_tools import causal_analysis
        est = num(r"^ESTIMATE: (\S+)", causal_analysis(self.csv, "t", "y", []))[0]
        self.assertGreater(est, 2.4)

    def test_bad_inputs(self):
        from tools.mesh_mcp.sandbox_tools import causal_analysis
        with self.assertRaisesRegex(ValueError, "columns not in CSV"):
            causal_analysis(self.csv, "t", "nope", [])
        with self.assertRaisesRegex(ValueError, "not found"):
            causal_analysis(self.csv + ".missing.csv", "t", "y", [])
        with self.assertRaisesRegex(ValueError, "binary"):
            causal_analysis(self.csv, "z", "y", ["w"], "propensity_score_weighting")

    def test_discover_graph_is_labelled_exploratory(self):
        from tools.mesh_mcp.sandbox_tools import discover_graph
        out = discover_graph(self.csv)
        self.assertTrue(out.startswith("EXPLORATORY"))
        self.assertIn("z --> y", out)


if __name__ == "__main__":
    unittest.main()
