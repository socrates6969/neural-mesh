"""Front-end (feature 1) refinement tests: fail-closed Rust calls, missing logprobs, py/Rust policy parity. No Ollama, no models.
Run: llm/.venv/Scripts/python -m unittest llm.tests.test_frontend"""
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from llm import config, frontend


def _proc(rc=0, out="", err=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=out, stderr=err)


class FreshBin:
    """a fake 'binary' newer than the Rust sources, so rust_decide reaches the (stubbed) subprocess call"""

    def __enter__(self):
        self.d = tempfile.TemporaryDirectory()
        self.p = Path(self.d.name) / "neural-mesh.exe"
        self.p.write_text("x")
        t = time.time() + 3600
        os.utime(self.p, (t, t))
        return self.p

    def __exit__(self, *a):
        self.d.cleanup()


class RustDecideFailsClosed(unittest.TestCase):
    def _call(self, proc=None, exc=None):
        with FreshBin() as b:
            fake = mock.Mock(side_effect=exc) if exc else mock.Mock(return_value=proc)
            with mock.patch.object(frontend, "_run", fake):
                return frontend.rust_decide({"enabled": 1}, {"local-conf": -0.1}, [{"id": "a", "score": 0.9, "tokens": 5}], bin_path=b), fake

    def assertForwardNoContext(self, d):
        self.assertEqual((d["decision"], d["context"], d["context_tokens"], d["ok"]), ("forward", [], 0, False))
        self.assertTrue(d["error"])

    def test_nonzero_exit_forwards_without_context(self):
        d, _ = self._call(_proc(1, json.dumps({"ok": False, "error": "x"})))
        self.assertForwardNoContext(d)
        d, _ = self._call(_proc(3, json.dumps({"ok": True, "decision": "local", "context": ["a"], "context_tokens": 5})))
        self.assertForwardNoContext(d)

    def test_garbage_or_not_ok_output_forwards_without_context(self):
        for out in ["", "not json", "[1,2]", json.dumps({"ok": False, "decision": "local"}), json.dumps({"ok": True, "decision": "explode"})]:
            d, _ = self._call(_proc(0, out))
            self.assertForwardNoContext(d)

    def test_timeout_and_oserror_forward_without_context(self):
        d, fake = self._call(exc=subprocess.TimeoutExpired("nm", 10))
        self.assertForwardNoContext(d)
        self.assertEqual(fake.call_args.kwargs["timeout"], frontend.RUST_TIMEOUT_S)
        self.assertEqual(frontend.RUST_TIMEOUT_S, 10.0)
        d, _ = self._call(exc=OSError("gone"))
        self.assertForwardNoContext(d)

    def test_good_answer_passes_through(self):
        good = {"ok": True, "schema": 2, "decision": "forward", "reason": "ood", "tool": None, "context": ["a"], "context_tokens": 5}
        d, fake = self._call(_proc(0, json.dumps(good)))
        self.assertEqual(d, good)
        args = fake.call_args.args[0]
        self.assertEqual(args[1], "frontend")
        self.assertEqual(json.loads(args[args.index("--chunks") + 1]), [{"id": "a", "score": 0.9, "tokens": 5}])

    def test_stale_or_missing_binary_is_not_run(self):
        fake = mock.Mock(return_value=_proc(0, json.dumps({"ok": True, "decision": "local"})))
        with tempfile.TemporaryDirectory() as d, mock.patch.object(frontend, "_run", fake):
            old = Path(d) / "nm.exe"
            old.write_text("x")
            os.utime(old, (1, 1))   # older than src/frontend.rs
            self.assertFalse(frontend.binary_fresh(old))
            self.assertForwardNoContext(frontend.rust_decide({"enabled": 1}, {}, [], bin_path=old))
            self.assertForwardNoContext(frontend.rust_decide({"enabled": 1}, {}, [], bin_path=Path(d) / "missing.exe"))
        fake.assert_not_called()


class MissingLogprobsForward(unittest.TestCase):
    def test_no_logprobs_gives_conf_none_and_the_policy_forwards(self):
        for lp in (None, [], [{"logprob": float("nan")}], [{"token": "x"}]):
            with mock.patch.object(frontend.judge, "_post", return_value={"message": {"content": " hi "}, "logprobs": lp, "eval_count": 3}):
                ans, conf, n = frontend.ask_local("m", "q")
            self.assertEqual((ans, conf, n), ("hi", None, 3))
            sig = {"cache-hit": 0, "local-conf": conf, "local-tokens": n, "max-sim": 0.99}
            self.assertEqual(frontend.py_decide({"enabled": 1, "min-conf": -99.0, "min-sim": 0.0}, sig), ("forward", "no_conf"))
        with mock.patch.object(frontend.judge, "_post", return_value={"message": {"content": "a"}, "logprobs": [{"logprob": -0.1}, {"logprob": -0.3}]}):
            ans, conf, n = frontend.ask_local("m", "q")
        self.assertAlmostEqual(conf, -0.2)
        self.assertEqual(n, 2)

    def test_sim_counts_cache_tool_local_separately(self):
        rows = [{"cache": True, "tool": None, "conf": None, "ntok": 5, "max_sim": 0.1, "score_used": 1.0, "tok_in": 10, "tok_out": 10},
                {"cache": False, "tool": "calc", "conf": None, "ntok": 0, "max_sim": None, "score_used": 1.0, "tok_in": 5, "tok_out": 5},
                {"cache": False, "tool": None, "conf": -0.05, "ntok": 5, "max_sim": 0.9, "score_used": 0.0, "tok_in": 7, "tok_out": 3},
                {"cache": False, "tool": None, "conf": None, "ntok": 5, "max_sim": 0.9, "score_used": 1.0, "tok_in": 1, "tok_out": 1}]
        g = frontend._sim(rows, [{"min-conf": -0.1, "min-sim": 0.5}])[0]
        self.assertEqual((g["cache"], g["tool"], g["local"], g["forward"], g["served"], g["wrong"]), (1, 1, 1, 1, 3, 1))
        self.assertEqual((g["saved_tokens"], g["total_tokens"]), (40, 42))


@unittest.skipUnless(frontend.binary_fresh(), "release binary missing or stale (cargo build --release)")
class PyMirrorsRust(unittest.TestCase):
    def test_py_decide_equals_rust_on_the_grid(self):
        # one process per call (~0.1 s on Windows): a branch-covering set rather than the full cross product
        sigs = [{"cache-hit": 0, "local-conf": c, "local-tokens": n, "max-sim": ms}
                for c, n, ms in [(None, 5, 0.9), (float("nan"), 5, 0.9), (-0.5, 5, 0.9), (-0.2, 5, 0.9), (-0.1, 5, 0.9), (-0.1, 0, 0.9), (-0.1, 401, 0.9),
                                 (-0.1, 400, 0.9), (-0.1, 5, None), (-0.1, 5, -0.4), (-0.1, 5, 0.0), (-0.1, 5, 0.3), (-0.1, 5, 0.29), (None, 0, None)]]
        sigs += [{"cache-hit": 1, "local-conf": None, "local-tokens": 0, "max-sim": None}, {"cache-hit": 0, "tool": "calc", "local-conf": None, "local-tokens": 0, "max-sim": -1.0}]
        n = 0
        for mc in (-0.2,):
            for msim in (0.0, 0.3):
                cfg = {"enabled": 1, "min-conf": mc, "min-sim": msim, "max-local-tokens": 400}
                for s in sigs:
                    s2 = {k: ("NaN" if isinstance(v, float) and v != v else v) for k, v in s.items()}
                    d = frontend.rust_decide(cfg, s2, [])
                    self.assertTrue(d["ok"], d)
                    self.assertEqual(frontend.py_decide(cfg, s), (d["decision"], d["reason"]), (cfg, s))
                    n += 1
        self.assertEqual(n, 32)
        # min-sim 0 with a NEGATIVE max_sim: in distribution on both sides
        cfg = {"enabled": 1, "min-conf": -1.0, "min-sim": 0.0}
        s = {"cache-hit": 0, "local-conf": -0.1, "local-tokens": 5, "max-sim": -0.4}
        self.assertEqual(frontend.py_decide(cfg, s), ("local", "local"))
        self.assertEqual(frontend.rust_decide(cfg, s, [])["decision"], "local")
        self.assertEqual(frontend.rust_decide({}, s, [])["reason"], "disabled")


if __name__ == "__main__":
    unittest.main()
