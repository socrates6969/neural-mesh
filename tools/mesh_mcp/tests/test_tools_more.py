"""ICP, phack_guard, market_series and code_run tests. Market tests use recorded responses (no network)."""
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from tools.mesh_mcp import code_run as CR
from tools.mesh_mcp import market as M
from tools.mesh_mcp import phack as P
from tools.mesh_mcp.common import docker_available
from tools.mesh_mcp.tests.test_literature import fetcher

FIX = Path(__file__).resolve().parent / "fixtures"
try:
    import numpy as np
except ImportError:
    np = None


def icp_csv(path, n_per=400, seed=0):
    """y = 1.0 x1 - 0.8 x2 + e; x3 = 0.9 y + shift(env) + noise(env) is a child of y that is intervened on;
    x4 is irrelevant noise. Environments shift x1's mean and x2's variance."""
    r = np.random.default_rng(seed)
    rows = []
    for e, (m1, s2, d3) in enumerate([(0, 1, 0), (2, 1, 1.5), (-1, 2.5, -2)]):
        x1, x2, x4 = r.normal(m1, 1, n_per), r.normal(0, s2, n_per), r.normal(0, 1, n_per)
        y = 1.0 * x1 - 0.8 * x2 + r.normal(0, 1, n_per)
        x3 = 0.9 * y + d3 + r.normal(0, 0.5 + 0.5 * e, n_per)
        rows += [(e, *v) for v in zip(x1, x2, x3, x4, y)]
    Path(path).write_text("env,x1,x2,x3,x4,y\n" + "\n".join("%d,%.6f,%.6f,%.6f,%.6f,%.6f" % r for r in rows),
                          encoding="utf-8")


class PhackTest(unittest.TestCase):
    def test_clean_single_test_is_ok(self):
        r = P.check([{"label": "primary", "p": 0.01}], {"outcomes": ["primary"]})
        self.assertEqual(r["verdict"], "OK", r)

    def test_uncorrected_multiple_testing_blocks(self):
        tests = [{"label": "o%d" % i, "p": p} for i, p in enumerate([0.03, 0.2, 0.4, 0.6, 0.04, 0.7])]
        r = P.check(tests)
        self.assertEqual(r["verdict"], "BLOCK")
        self.assertTrue(any("multiple testing without correction" in b for b in r["block"]), r)

    def test_correction_in_log_or_prereg_clears_it(self):
        ps = [0.03, 0.2, 0.4, 0.6, 0.04, 0.7]
        tests = [{"label": "o%d" % i, "p": p} for i, p in enumerate(ps)]
        self.assertEqual(P.check(tests + [{"kind": "multitest", "p_raw": ps}])["verdict"], "OK")
        self.assertEqual(P.check(tests, {"correction": "holm"})["verdict"], "OK")

    def test_strong_results_surviving_holm_only_warn(self):
        r = P.check([{"label": "a", "p": 0.001}, {"label": "b", "p": 0.002}, {"label": "c", "p": 0.5}])
        self.assertEqual(r["verdict"], "OK")
        self.assertTrue(r["warn"])

    def test_optional_stopping_blocks_unless_preregistered(self):
        looks = [{"label": "conv", "p": 0.21, "n": 100}, {"label": "conv", "p": 0.09, "n": 200},
                 {"label": "conv", "p": 0.04, "n": 300}]
        r = P.check(looks)
        self.assertEqual(r["verdict"], "BLOCK")
        self.assertTrue(any("stopped at the first significant look" in b for b in r["block"]), r)
        r = P.check(looks, {"outcomes": ["conv"], "looks": 3, "sequential": True})
        self.assertEqual(r["verdict"], "OK", r)

    def test_unregistered_outcome_blocks(self):
        r = P.check([{"label": "primary", "p": 0.2}, {"label": "secondary_x", "p": 0.01}],
                    {"outcomes": ["primary"], "correction": "holm"})
        self.assertEqual(r["verdict"], "BLOCK")
        self.assertTrue(any("not in the preregistration: secondary_x" in b for b in r["block"]), r)

    def test_caliper_flags_bunching_under_05(self):
        ps = [0.046, 0.047, 0.048, 0.049, 0.0495, 0.0499, 0.047, 0.052]
        r = P.check([{"label": "s%d" % i, "p": p} for i, p in enumerate(ps)], {"correction": "bh"})
        self.assertEqual(r["verdict"], "BLOCK")
        self.assertTrue(any(b.startswith("caliper test") for b in r["block"]), r)
        self.assertAlmostEqual(P._binom_sf(7, 8), 9 / 256)

    def test_tool_reads_session_log(self):
        os.environ["CLAUDE_SESSION_ID"] = "phack-test"
        try:
            for i, p in enumerate([0.03, 0.3, 0.04, 0.6]):
                P.record_test("welch", {}, {"p": p, "n_a": 10, "n_b": 10}, "outcome%d" % i)
            out = P.phack_guard()
            self.assertTrue(out.startswith("BLOCK (session log"), out)
            self.assertIn("4 tests checked", out)
            self.assertTrue(P.phack_guard('[{"label": "x", "p": 0.2}]').startswith("OK (tests_json"))
        finally:
            os.environ.pop("CLAUDE_SESSION_ID")


def market_fetch(url, headers=None):
    if "fred" in url:
        return 200, (FIX / "fred_DGS10.csv").read_bytes(), {}
    if "query1.finance.yahoo.com" in url and "/SPY?" in url:
        return 200, (FIX / "yahoo_SPY.json").read_bytes(), {}
    return 404, b"", {}


class MarketTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._dir = os.environ.get("MESH_MARKET_DIR")
        os.environ["MESH_MARKET_DIR"] = self.tmp.name

    def tearDown(self):
        os.environ["MESH_MARKET_DIR"] = self._dir
        self.tmp.cleanup()

    def test_fred_series_with_provenance_and_cache(self):
        out = M.market_series(fred_id="DGS10", start="2024-01-01", end="2024-01-31", fetch=fetcher(market_fetch))
        self.assertTrue(out.startswith("FRED DGS10 (value), 21 observations 2024-01-02 .. 2024-01-31"), out)
        self.assertIn("url=https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=2024-01-01&coed=2024-01-31",
                      out)
        sha = re.search(r"sha256=([0-9a-f]{64}) cached=False", out).group(1)
        meta = json.loads(next(Path(self.tmp.name).glob("fred_DGS10_*.json")).read_text())
        self.assertEqual(meta["sha256"], sha)
        self.assertTrue(meta["retrieved_utc"].endswith("Z"))
        self.assertEqual((Path(self.tmp.name) / ".gitignore").read_text(), "*\n")
        again = M.market_series(fred_id="DGS10", start="2024-01-01", end="2024-01-31",
                                fetch=fetcher(lambda u, h=None: (500, b"", {})))
        self.assertIn("cached=True", again)

    def test_tampered_cache_is_refetched(self):
        M.market_series(fred_id="DGS10", start="2024-01-01", end="2024-01-31", fetch=fetcher(market_fetch))
        f = next(Path(self.tmp.name).glob("fred_DGS10_*.csv"))
        f.write_text("date,value\n2024-01-02,999\n")
        out = M.market_series(fred_id="DGS10", start="2024-01-01", end="2024-01-31", fetch=fetcher(market_fetch))
        self.assertIn("cached=False", out)
        self.assertNotIn("999", out)

    def test_yahoo_adjusted_close(self):
        out = M.market_series(ticker="SPY", start="2024-01-01", end="2024-01-31", fetch=fetcher(market_fetch))
        self.assertTrue(out.startswith("YAHOO SPY (adjusted close), 21 observations"), out)

    def test_failure_returns_error_never_data(self):
        out = M.market_series(ticker="NOPE", start="2024-01-01", end="2024-01-31", fetch=fetcher(market_fetch))
        self.assertTrue(out.startswith("FETCH FAILED"), out)
        self.assertIn("nothing was synthesized", out)
        self.assertEqual(list(Path(self.tmp.name).glob("*.csv")), [])

    def test_validation(self):
        for kw in ({}, {"ticker": "SPY", "fred_id": "DGS10"}, {"ticker": "S P Y"}, {"ticker": "SPY", "start": "x"}):
            with self.assertRaises(ValueError):
                M.market_series(**dict({"start": "2024-01-01"}, **kw))


@unittest.skipUnless(docker_available(), "Docker grader image not available")
class CodeRunTest(unittest.TestCase):
    def setUp(self):
        CR.reset()

    def test_pass_with_tests(self):
        out = CR.code_run("python", "def sq(x):\n    return x * x\nprint(sq(3))", "assert sq(4) == 16", "ok")
        self.assertTrue(out.startswith("PASS (python, first run, task ok)"), out)
        self.assertIn("tests passed", out)
        self.assertIn("9", out)

    def test_failure_is_structured_and_rounds_are_capped(self):
        bad, tests = "def sq(x):\n    return x + x", "assert sq(3) == 9, 'sq wrong'"
        out = CR.code_run("python", bad, tests, "t")
        err = json.loads(re.search(r"^ERROR: (.*)$", out, re.M).group(1))
        self.assertEqual((err["phase"], err["type"], err["file"], err["line"]), ("tests", "AssertionError",
                                                                                 "tests.py", 1))
        self.assertIn("3 repair rounds left", out)
        heads = [CR.code_run("python", bad, tests, "t").splitlines()[0] for _ in range(3)]
        self.assertEqual(heads, ["FAIL (python, repair round %d of 3, task t)" % i for i in (1, 2, 3)])
        self.assertTrue(CR.code_run("python", bad, tests, "t").startswith("STOP"))
        fixed = CR.code_run("python", "def sq(x):\n    return x * x", tests, "t2")
        self.assertTrue(fixed.startswith("PASS"))

    def test_pass_resets_counter(self):
        CR.code_run("python", "raise ValueError('x')", None, "r")
        self.assertTrue(CR.code_run("python", "print(1)", None, "r").startswith("PASS (python, repair round 1"))
        self.assertTrue(CR.code_run("python", "print(1)", None, "r").startswith("PASS (python, first run"))

    def test_syntax_error_and_shell(self):
        out = CR.code_run("python", "def f(:\n    pass", None, "s")
        err = json.loads(re.search(r"^ERROR: (.*)$", out, re.M).group(1))
        self.assertEqual((err["type"], err["line"]), ("SyntaxError", 1))
        out = CR.code_run("sh", "echo hello; exit 3", None, "sh")
        self.assertTrue(out.startswith("FAIL (sh"), out)
        self.assertIn("hello", out)
        self.assertTrue(CR.code_run("sh", "echo ok", "test 1 -eq 1", "sh2").startswith("PASS"))

    def test_no_network(self):
        out = CR.code_run("python", "import urllib.request\nurllib.request.urlopen('http://1.1.1.1', timeout=3)",
                          None, "net")
        self.assertTrue(out.startswith("FAIL"), out)
        self.assertIn("URLError", out)


@unittest.skipUnless(docker_available() and np is not None, "Docker grader image or numpy not available")
class IcpToolTest(unittest.TestCase):
    def test_finds_parents_and_excludes_shifted_child(self):
        from tools.mesh_mcp.sandbox_tools import invariant_causal_prediction
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d, "icp.csv"))
            icp_csv(p)
            out = invariant_causal_prediction(p, "y", "env")
            strict = invariant_causal_prediction(p, "y", "env", correction="subsets")
        self.assertIn("ESTIMATED CAUSAL PARENTS: x1, x2\n", out)
        self.assertIn("predictors: x1, x2, x3, x4", out)
        self.assertNotIn("x3", out.split("ESTIMATED CAUSAL PARENTS:")[1].splitlines()[0])
        self.assertIn("16 subsets tested", out)
        self.assertIn("ESTIMATED CAUSAL PARENTS:", strict)

    def test_stats_test_calls_are_logged_for_phack_guard(self):
        from tools.mesh_mcp.sandbox_tools import stats_test
        os.environ["CLAUDE_SESSION_ID"] = "log-test"
        try:
            out = stats_test("welch", {"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]}, "speed", 1)
            self.assertIn("Logged for phack_guard as outcome 'speed'", out)
            rec = P.read_log()[-1]
            self.assertEqual((rec["kind"], rec["label"], rec["look"], rec["n"]), ("welch", "speed", 1, 10))
            self.assertAlmostEqual(rec["p"], 0.1075, places=3)
        finally:
            os.environ.pop("CLAUDE_SESSION_ID")


if __name__ == "__main__":
    unittest.main()
