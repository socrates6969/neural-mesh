"""S2 `ask` wrapper: no Ollama, no MiniLM, no judge (stub backend), Rust stubbed."""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from llm import frontend, frontend_live as fl

REAL_BUSY = fl.nightly_busy

class Stub:
    def __init__(self, answer="local answer", conf=-0.05, fail=False):
        self.answer, self.conf, self.fail, self.asked = answer, conf, fail, 0

    def encode(self, texts):
        return np.array([[1.0, 0.1 * (len(t) % 3), 0.0] for t in texts], np.float32) / 1.01

    def ntok(self, s):
        return len((s or "").split())

    def ask(self, prompt, timeout):
        self.asked += 1
        if self.fail:
            raise TimeoutError("ollama")
        return self.answer, self.conf, 2

    def grade(self, i, r, a):
        return 1.0 if a == r else 0.0


ROWS = [{"id": "sft:1", "ts": "2026-10-01T00:00:00Z", "project": "p", "session": "s0", "instruction": "earlier question", "response": "earlier answer " * 5,
         "outcome": "success", "judge_score": 1.0, "judge_model": "j"}]


def decide_as(decision, context=("sft:1",)):
    def f(cfg, sig, chunks):
        f.calls.append((cfg, sig, chunks))
        return {"ok": True, "decision": decision, "reason": decision, "tool": None, "context": list(context) if decision == "forward" or decision == "tool" else [],
                "context_tokens": 10}
    f.calls = []
    return f


class Ask(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.log = Path(self.d.name) / "live.jsonl"
        self.lock = mock.patch.object(fl, "nightly_busy", return_value=False)
        self.lock.start()
        self.head = mock.patch.object(fl, "git_head", return_value=None)
        self.head.start()

    def tearDown(self):
        self.lock.stop()
        self.head.stop()
        self.d.cleanup()

    def run_ask(self, prompt="what is x?", **kw):
        out, fwd = io.StringIO(), []

        def forward(payload, cmd, o):
            fwd.append((payload, cmd))
            return 0
        kw.setdefault("rows_loader", lambda: ROWS)
        kw.setdefault("project", "p")
        rc = fl.ask(prompt, out=out, log_path=self.log, forward=forward, **kw)
        recs = [json.loads(x) for x in self.log.read_text(encoding="utf-8").splitlines()] if self.log.exists() else []
        return rc, out.getvalue(), fwd, recs

    def test_disabled_is_pure_pass_through(self):
        factory = mock.Mock(side_effect=AssertionError("no model when disabled"))
        dec = mock.Mock(side_effect=AssertionError("no policy when disabled"))
        rc, out, fwd, recs = self.run_ask("  raw prompt\n", backend_factory=factory, decide=dec, forward_cmd="claude -p")
        self.assertEqual((rc, fwd, recs), (0, [("  raw prompt\n", "claude -p")], []))
        factory.assert_not_called()
        dec.assert_not_called()
        # the real forwarder without a command writes the prompt unchanged to stdout
        out = io.StringIO()
        self.assertEqual(fl.ask("abc", out=out, log_path=self.log), 0)
        self.assertEqual(out.getvalue(), "abc\n")
        self.assertFalse(self.log.exists())

    def test_any_error_forwards_the_original_prompt_without_context(self):
        boom = mock.Mock(side_effect=RuntimeError("encoder exploded"))
        for kw in ({"backend_factory": boom}, {"backend_factory": Stub, "decide": mock.Mock(side_effect=ValueError("bad"))},
                   {"backend_factory": Stub, "rows_loader": mock.Mock(side_effect=OSError("disk"))}):
            for mode in ("cli", "advisory"):
                self.log.unlink(missing_ok=True)
                rc, out, fwd, recs = self.run_ask("the prompt", enabled=True, mode=mode, **kw)
                self.assertEqual(fwd, [("the prompt", None)], (kw, mode))
                self.assertEqual(out, "")
                self.assertEqual((recs[-1]["decision"], recs[-1]["reason"], recs[-1]["served"]), ("forward", "error", "forward"))
                self.assertTrue(recs[-1]["error"])
        # a failing Rust policy (fail-closed answer) also forwards without context
        rc, out, fwd, recs = self.run_ask("p2", enabled=True, mode="advisory", backend_factory=Stub, decide=lambda c, s, ch: frontend._fail("exit 3"))
        self.assertEqual(fwd, [("p2", None)])
        self.assertFalse(recs[-1]["rust_ok"])

    def test_log_holds_no_prompt_text_without_log_text(self):
        """privacy: without --log-text frontend_live.jsonl carries the prompt's sha256 only, in every mode and on the error path"""
        secret = "SECRET-PROMPT zebra-7731 my password is hunter2"
        answer = "LOCAL-ANSWER quokka-4410"
        cases = [dict(mode=m, decide=decide_as(d)) for m in ("shadow", "advisory", "cli") for d in ("local", "forward")]
        cases.append(dict(mode="cli", decide=mock.Mock(side_effect=ValueError(f"cannot parse {secret!r}"))))   # message echoes the prompt
        for kw in cases:
            self.run_ask(secret, enabled=True, backend_factory=lambda: Stub(answer=answer), **kw)
        raw = self.log.read_text(encoding="utf-8")
        recs = [json.loads(x) for x in raw.splitlines()]
        self.assertEqual(len(recs), len(cases))
        for frag in ("SECRET-PROMPT", "zebra-7731", "hunter2", "quokka-4410"):
            self.assertNotIn(frag, raw)
        self.assertTrue(all(r["sha256"] == fl.sha(secret) and "prompt" not in r and "local_answer" not in r for r in recs))
        self.assertEqual(recs[-1]["reason"], "error")
        self.assertIn("<prompt>", recs[-1]["error"])
        # --log-text is the only way text gets in
        self.run_ask(secret, enabled=True, mode="cli", backend_factory=lambda: Stub(answer=answer), decide=decide_as("local"), log_text=True)
        self.assertIn("zebra-7731", self.log.read_text(encoding="utf-8").splitlines()[-1])

    def test_shadow_logs_would_be_and_forwards_unchanged(self):
        rc, out, fwd, recs = self.run_ask("q", enabled=True, mode="shadow", backend_factory=Stub, decide=decide_as("local"))
        self.assertEqual(fwd, [("q", None)])
        r = recs[-1]
        self.assertEqual((r["served"], r["would_be"], r["decision"]), ("forward", "local", "local"))
        self.assertNotIn("prompt", r)
        self.assertEqual(r["sha256"], fl.sha("q"))

    def test_advisory_adds_untrusted_context_only(self):
        rc, out, fwd, recs = self.run_ask("q", enabled=True, mode="advisory", backend_factory=Stub, decide=decide_as("forward"))
        payload = fwd[0][0]
        self.assertTrue(payload.endswith("\n\nq"))
        self.assertIn("<retrieved untrusted id=sft:1", payload)
        rc, out, fwd, recs = self.run_ask("q", enabled=True, mode="advisory", backend_factory=Stub, decide=decide_as("local"))
        self.assertEqual(len(fwd), 1)            # advisory never serves a local answer
        self.assertEqual(recs[-1]["would_be"], "local")

    def test_cli_serves_local_and_forwards_otherwise(self):
        rc, out, fwd, recs = self.run_ask("q", enabled=True, mode="cli", backend_factory=Stub, decide=decide_as("local"), log_text=True)
        self.assertEqual((rc, out, fwd), (0, "local answer\n", []))
        self.assertEqual((recs[-1]["served"], recs[-1]["prompt"]), ("local", "q"))
        rc, out, fwd, recs = self.run_ask("q", enabled=True, mode="cli", backend_factory=Stub, decide=decide_as("forward"))
        self.assertEqual(out, "")
        self.assertIn("untrusted", fwd[0][0])

    def test_local_model_is_skipped_while_nightly_lock_is_fresh(self):
        st = Stub()
        dec = decide_as("forward")
        with mock.patch.object(fl, "nightly_busy", return_value=True):
            self.run_ask("q", enabled=True, mode="cli", backend_factory=lambda: st, decide=dec)
        self.assertEqual(st.asked, 0)
        self.assertIsNone(dec.calls[0][1]["local-conf"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "nightly.lock"
            self.assertFalse(REAL_BUSY(p))
            p.write_text("1")
            self.assertTrue(REAL_BUSY(p))
            os.utime(p, (1, 1))
            self.assertFalse(REAL_BUSY(p))   # a stale (crashed) lock does not block

    def test_overall_budget_exceeded_forwards_the_original_prompt(self):
        import time as _t

        class Slow(Stub):
            def ask(self, prompt, timeout):
                _t.sleep(3)
                return super().ask(prompt, timeout)
        t = _t.monotonic()
        rc, out, fwd, recs = self.run_ask("q", enabled=True, mode="cli", backend_factory=Slow, decide=decide_as("local"), budget_s=0.3)
        self.assertLess(_t.monotonic() - t, 2.0)
        self.assertEqual((rc, out, fwd), (0, "", [("q", None)]))                  # no local answer served, no context added
        self.assertEqual((recs[-1]["decision"], recs[-1]["reason"], recs[-1]["served"]), ("forward", "budget", "forward"))
        self.assertEqual(fl.DEFAULT_BUDGET_S, 30.0)
        # within budget the same call serves the local answer
        rc, out, fwd, recs = self.run_ask("q", enabled=True, mode="cli", backend_factory=Stub, decide=decide_as("local"), budget_s=5)
        self.assertEqual((out, fwd), ("local answer\n", []))

    def test_local_timeout_means_no_confidence(self):
        dec = decide_as("forward")
        self.run_ask("q", enabled=True, mode="cli", backend_factory=lambda: Stub(fail=True), decide=dec)
        self.assertIsNone(dec.calls[0][1]["local-conf"])

    def test_forward_cmd_gets_stdin_and_its_exit_code_is_returned(self):
        script = Path(self.d.name) / "fwd.py"
        got = Path(self.d.name) / "got.txt"
        script.write_text(f"import sys\nopen(r'{got}', 'w', encoding='utf-8').write(sys.stdin.read())\nsys.exit(7)\n")
        cmd = f'"{sys.executable}" "{script}"'
        rc = fl.ask("payload text", forward_cmd=cmd, out=io.StringIO(), log_path=self.log)
        self.assertEqual(rc, 7)
        self.assertEqual(got.read_text(encoding="utf-8"), "payload text")

    def test_cfg_file_only_counts_when_keep(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.json"
            self.assertEqual(fl.policy_cfg(cfg_file=p), {"enabled": 1})
            p.write_text(json.dumps({"verdict": "REJECTED", "min-conf": -0.5}))
            self.assertEqual(fl.policy_cfg(cfg_file=p), {"enabled": 1})
            p.write_text(json.dumps({"verdict": "KEEP", "min-conf": -0.3, "min-sim": 0.6, "enabled": True}))
            self.assertEqual(fl.policy_cfg(cfg_file=p), {"enabled": 1, "min-conf": -0.3, "min-sim": 0.6})
            self.assertEqual(fl.policy_cfg(min_sim=0.9, cfg_file=p)["min-sim"], 0.9)

    def test_cli_dispatch_happens_before_the_lock(self):
        from llm import __main__ as m
        with mock.patch.object(m, "lock", side_effect=AssertionError("ask must not take the lock")), \
                mock.patch("llm.frontend_live.cli_ask", return_value=5) as ca:
            self.assertEqual(m.main(["ask", "--mode", "cli"]), 5)
        ca.assert_called_once_with(["--mode", "cli"])


if __name__ == "__main__":
    unittest.main()
