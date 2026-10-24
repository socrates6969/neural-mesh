"""NOTEARS and front-door tests. Host tests need numpy+scipy; tool tests need Docker and the grader image."""
import re
import sys
import tempfile
import unittest
from pathlib import Path

from tools.mesh_mcp.common import docker_available

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sandbox"))
try:
    import numpy as np
    import notears_impl as N
    HAVE = True
except ImportError:
    np = None
    HAVE = False

# 5-node linear SEM (fixed seed): 0->1, 0->2, 1->3, 2->3, 3->4 with mixed-sign weights, unit Gaussian noise
TRUE_W = {(0, 1): 1.5, (0, 2): -1.0, (1, 3): 0.8, (2, 3): 1.2, (3, 4): -1.5}
NAMES = ["a", "b", "c", "d", "e"]
FD_EFFECT = 0.8 * 1.5  # t -> m (0.8), m -> y (1.5); u confounds t and y and is not in the CSV


def sem_data(n=1000, seed=0):
    W = np.zeros((5, 5))
    for (i, j), w in TRUE_W.items():
        W[i, j] = w
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 5))
    for j in range(5):
        X[:, j] = X @ W[:, j] + rng.normal(size=n)
    return X


def shd(true_edges, est_edges):
    """Structural Hamming distance between edge sets of (i, j): missing + extra, a reversed edge counts once."""
    pairs = {frozenset(e) for e in true_edges} | {frozenset(e) for e in est_edges}
    return sum(1 for p in pairs if {e for e in true_edges if frozenset(e) == p}
               != {e for e in est_edges if frozenset(e) == p})


def write_csv(path, header, rows):
    Path(path).write_text(",".join(header) + "\n" + "\n".join(",".join("%.6f" % v for v in r) for r in rows) + "\n",
                          encoding="utf-8")


def frontdoor_csv(path, n=3000, seed=0):
    r = np.random.default_rng(seed)
    u = r.normal(size=n)
    t = u + r.normal(size=n)
    m = 0.8 * t + r.normal(size=n)
    y = 1.5 * m + 2.0 * u + r.normal(size=n)
    write_csv(path, ["t", "m", "y"], zip(t, m, y))


@unittest.skipUnless(HAVE, "numpy/scipy not installed on host")
class NotearsHostTest(unittest.TestCase):
    def test_recovers_sem_dag(self):
        W, h, _ = N.notears_linear(sem_data(), lambda1=0.1, w_threshold=0.3)
        est = {(i, j) for i in range(5) for j in range(5) if W[i, j] != 0}
        self.assertLessEqual(shd(set(TRUE_W), est), 1, est)
        self.assertLess(h, 1e-8)
        for (i, j), w in TRUE_W.items():
            if (i, j) in est:
                self.assertAlmostEqual(W[i, j], w, delta=0.35)

    def test_h_is_zero_only_for_dags(self):
        self.assertEqual(N.h_acyclic(np.array([[0, 1.0], [0, 0]]))[0], 0.0)
        self.assertGreater(N.h_acyclic(np.array([[0, 1.0], [1.0, 0]]))[0], 0.1)

    def test_shd_helper(self):
        self.assertEqual(shd({(0, 1), (1, 2)}, {(0, 1), (1, 2)}), 0)
        self.assertEqual(shd({(0, 1), (1, 2)}, {(1, 0), (1, 2)}), 1)  # reversed
        self.assertEqual(shd({(0, 1)}, {(0, 1), (0, 2)}), 1)  # extra


@unittest.skipUnless(docker_available() and HAVE, "Docker grader image or numpy not available")
class CausalToolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sem = str(Path(cls.tmp.name, "sem.csv"))
        write_csv(cls.sem, NAMES, sem_data())
        cls.fd = str(Path(cls.tmp.name, "fd.csv"))
        frontdoor_csv(cls.fd)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_notears_tool_shd(self):
        from tools.mesh_mcp.sandbox_tools import discover_dag_notears
        out = discover_dag_notears(self.sem)
        self.assertTrue(out.startswith("EXPLORATORY"), out)
        self.assertIn("checked < 1e-8", out)
        est = {(NAMES.index(a), NAMES.index(b)) for a, b in re.findall(r"^\s+(\w) --> (\w)", out, re.M)}
        self.assertLessEqual(shd(set(TRUE_W), est), 1, out)

    def _estimate(self, out):
        est = float(re.search(r"^ESTIMATE: (\S+)", out, re.M).group(1))
        lo, hi = (float(x) for x in re.search(r"^95% CI: \[(\S+), (\S+)\]", out, re.M).groups())
        return est, lo, hi

    def test_frontdoor_recovers_effect_backdoor_does_not(self):
        from tools.mesh_mcp.sandbox_tools import causal_analysis
        fd = causal_analysis(self.fd, "t", "y", [], "frontdoor", mediator="m")
        est, lo, hi = self._estimate(fd)
        self.assertTrue(lo <= FD_EFFECT <= hi, fd)
        self.assertIn("frontdoor.two_stage_regression", fd)
        self.assertRegex(fd, r"placebo_permutation: new_effect=\S+ p=\S+ -> PASS")
        bd = causal_analysis(self.fd, "t", "y", [], "backdoor")  # u is hidden, so back-door adjustment is biased
        b_est, b_lo, b_hi = self._estimate(bd)
        self.assertFalse(b_lo <= FD_EFFECT <= b_hi, bd)
        self.__class__.numbers = (est, lo, hi, b_est, b_lo, b_hi)

    def test_frontdoor_not_identifiable_is_reported(self):
        from tools.mesh_mcp.sandbox_tools import causal_analysis
        direct = 'digraph {U[label="Unobserved"]; U->t; U->y; t->m; m->y; t->y;}'
        out = causal_analysis(self.fd, "t", "y", [], "frontdoor", mediator="m", graph=direct)
        self.assertTrue(out.startswith("NOT IDENTIFIABLE"), out)
        self.assertIn("direct t -> y edge", out)
        self.assertNotIn("ESTIMATE", out)
        out = causal_analysis(self.fd, "t", "y", [], "frontdoor")
        self.assertTrue(out.startswith("NOT IDENTIFIABLE") and "needs mediator" in out, out)
        confounded = 'digraph {U[label="Unobserved"]; U->t; U->y; U->m; t->m; m->y;}'
        out = causal_analysis(self.fd, "t", "y", [], "frontdoor", mediator="m", graph=confounded)
        self.assertTrue(out.startswith("NOT IDENTIFIABLE"), out)


if __name__ == "__main__":
    unittest.main()
