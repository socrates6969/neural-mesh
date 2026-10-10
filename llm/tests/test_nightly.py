"""`python -m llm nightly` orchestration with train/eval/promote stubbed: one lock + stage records, failure -> exit 3 with later
stages skipped, battery skip, abandoned-adapter sweep. Run: llm/.venv/Scripts/python -m unittest llm.tests.test_nightly"""
import json
from unittest import mock

from llm import config, evaluate, nightly, promote, train
from llm.tests.test_release import Sandbox

QUIET = lambda *_: None  # noqa: E731


class Stubs(Sandbox):
    """train makes a new adapter (parent = production), eval writes eval.json, promote records a decision; each call also
    checks that the nightly lock is held and fresh."""

    def setUp(self):
        super().setUp()
        self.calls = []
        self.lock_mtimes = []
        self.train_new = True
        self.eval_fails = False
        self.promote_exit = 0
        self.gate = {"promote": False, "reason": "stub gate: keep production"}
        p = [mock.patch.object(train, "run", self.fake_train), mock.patch.object(evaluate, "run", self.fake_eval),
             mock.patch.object(promote, "run", self.fake_promote)]
        for x in p:
            x.start()
            self.addCleanup(x.stop)

    def _seen(self, name):
        self.calls.append(name)
        if config.LOCK.exists():
            self.lock_mtimes.append(config.LOCK.stat().st_mtime)

    def fake_train(self, synthetic=False, epochs=2, log=print, **kw):
        self._seen("train")
        if not self.train_new:
            return None
        prod = (config.load_state().get("production") or {}).get("ver")
        ver = f"v{len(list(config.ADAPTERS.iterdir())) + 1:02d}"
        self.adapter(ver, parent=prod, new_ids=[ver + "-row"], evaluated=False)
        return {"ver": ver, "n_new": 1, "n_replay": 0}

    def fake_eval(self, synthetic=False, ver=None, judge_budget=None, log=print, **kw):
        self._seen(f"eval:{ver}")
        if self.eval_fails:
            raise RuntimeError("CUDA out of memory (stub)")
        (config.ADAPTERS / ver / "eval.json").write_text(json.dumps({"gate": self.gate, "loss": {"diff_cluster_ci": {}}, "production": None}))
        return {"gate": self.gate}

    def fake_promote(self, synthetic=False, ver=None, log=print, **kw):
        self._seen(f"promote:{ver}")
        st = config.load_state()
        st.setdefault("history", []).append({"ver": ver, "promoted": False, "action": "reject" if not self.promote_exit else "promote_failed"})
        config.save_state(st)
        return {"ver": ver, "promoted": False, "exit": self.promote_exit, "reason": "stub"}

    def nightly(self, power=True):
        from llm.__main__ import lock
        with lock():
            return nightly.run(power=lambda: power, log=QUIET)

    def runs(self):
        return [json.loads(x) for x in config.RUNS.read_text(encoding="utf-8").splitlines()]


class NightlyRun(Stubs):
    def test_single_lock_and_stage_records(self):
        config.LOCK.parent.mkdir(parents=True, exist_ok=True)
        rec = self.nightly()
        self.assertEqual(rec["exit"], 0)
        self.assertEqual([s["name"] for s in rec["stages"]], ["preflight", "train", "eval", "promote"])
        self.assertEqual([s["status"] for s in rec["stages"]], ["noop", "ok", "ok", "noop"])
        self.assertEqual(self.calls, ["train", "eval:v01", "promote:v01"])
        self.assertEqual(len(self.lock_mtimes), 3, "lock held during every stage (one lock for the whole round)")
        self.assertFalse(config.LOCK.exists(), "released at the end")
        for s in rec["stages"]:
            self.assertTrue({"name", "status", "exit", "reason", "started", "ended", "detail"} <= set(s))
        self.assertEqual(self.runs()[-1]["run_id"], rec["run_id"])
        self.assertEqual(config.load_state()["last_nightly"]["status"], "ok")
        self.assertEqual(config.load_state()["eval_attempts"], {"v01": 1})

    def test_lock_mtime_refreshed_per_stage(self):
        config.LOCK.parent.mkdir(parents=True, exist_ok=True)
        with mock.patch.object(nightly, "touch_lock", wraps=nightly.touch_lock) as t:
            self.nightly()
        self.assertEqual(t.call_count, 4, "refreshed before preflight, train, eval and promote")

    def test_busy_lock_exits_75(self):
        from llm.__main__ import main
        config.LOCK.write_text("1 0")
        with self.assertRaises(SystemExit) as e:
            main(["nightly"])
        self.assertEqual(e.exception.code, 75)
        self.assertEqual(self.calls, [])

    def test_stage_failure_exit_3_and_later_stages_skipped(self):
        self.eval_fails = True
        rec = self.nightly()
        self.assertEqual(rec["exit"], 3)
        st = {s["name"]: s for s in rec["stages"]}
        self.assertEqual(st["eval"]["status"], "failed")
        self.assertIn("out of memory", st["eval"]["reason"])
        self.assertEqual(st["promote"]["status"], "skipped")
        self.assertNotIn("promote:v01", self.calls)
        self.assertEqual(self.runs()[-1]["status"], "failed")

    def test_promote_failure_is_exit_3(self):
        self.promote_exit = 3
        rec = self.nightly()
        self.assertEqual((rec["exit"], rec["stages"][-1]["status"]), (3, "failed"))

    def test_cli_returns_the_run_exit_code(self):
        from llm.__main__ import main
        self.eval_fails = True
        with mock.patch.object(nightly, "on_ac_power", lambda: True):
            self.assertEqual(main(["nightly"]), 3)

    def test_battery_skips_llm_stages(self):
        rec = self.nightly(power=False)
        self.assertEqual(rec["exit"], 0)
        self.assertEqual(rec["host_power"], "battery")
        self.assertEqual([(s["name"], s["status"]) for s in rec["stages"][1:]], [("train", "skipped"), ("eval", "skipped"), ("promote", "skipped")])
        self.assertEqual(self.calls, [])

    def test_nothing_new_is_noop_exit_0(self):
        self.train_new = False
        rec = self.nightly()
        self.assertEqual(rec["exit"], 0)
        self.assertEqual([s["status"] for s in rec["stages"]], ["noop", "noop", "noop", "noop"])

    def test_pending_candidate_is_resumed_not_retrained(self):
        self.adapter("v01", evaluated=False)
        rec = self.nightly()
        self.assertEqual(rec["stages"][1]["status"], "skipped")
        self.assertEqual(self.calls, ["eval:v01", "promote:v01"])

    def test_power_probe_runs(self):
        self.assertIn(nightly.on_ac_power(), (True, False, None))


class AbandonedSweep(Sandbox):
    def sweep(self, st):
        return nightly.abandoned_sweep(st, log=QUIET)

    def test_torn_adapter_removed(self):
        (config.ADAPTERS / "v01").mkdir()
        (config.ADAPTERS / "v01" / "adapter_model.safetensors").write_text("half")
        out = self.sweep({"history": []})
        self.assertEqual(out["removed_torn"], ["v01"])
        self.assertFalse((config.ADAPTERS / "v01").exists())

    def test_older_unevaluated_superseded(self):
        for v in ("v01", "v02", "v03"):
            self.adapter(v, evaluated=False)
        st = {"history": [], "trained_ids": ["x"]}
        out = self.sweep(st)
        self.assertEqual(out["abandoned"], [{"ver": "v01", "reason": "superseded"}, {"ver": "v02", "reason": "superseded"}])
        self.assertEqual(out["pending"], "v03")
        self.assertEqual([h["action"] for h in st["history"]], ["abandoned", "abandoned"])
        self.assertEqual(st["trained_ids"], ["x"], "trained_ids only change on promote")
        self.assertTrue((config.ADAPTERS / "v01").exists(), "abandoned adapters are kept on disk (versioned)")

    def test_stale_parent(self):
        self.adapter("v01")
        self.adapter("v02", parent="v00", evaluated=False)
        st = {"production": {"ver": "v01"}, "history": [{"ver": "v01", "promoted": True, "action": "promote"}]}
        out = self.sweep(st)
        self.assertEqual(out["abandoned"][0]["ver"], "v02")
        self.assertIn("stale parent", out["abandoned"][0]["reason"])
        self.assertIsNone(out["pending"])

    def test_eval_failed_twice(self):
        self.adapter("v01", evaluated=False)
        out = self.sweep({"history": [], "eval_attempts": {"v01": 1}})
        self.assertEqual((out["abandoned"], out["pending"]), ([], "v01"))
        st = {"history": [], "eval_attempts": {"v01": 2}}
        out = self.sweep(st)
        self.assertEqual(out["abandoned"], [{"ver": "v01", "reason": "eval failed twice"}])
        self.assertIsNone(out["pending"])

    def test_decided_and_production_untouched_and_abandoned_skipped_by_latest_candidate(self):
        self.adapter("v01")
        self.adapter("v02", parent="v01")
        st = {"production": {"ver": "v01"}, "history": [{"ver": "v01", "promoted": True, "action": "promote"},
                                                        {"ver": "v02", "promoted": False, "action": "reject"}]}
        out = self.sweep(st)
        self.assertEqual((out["abandoned"], out["removed_torn"], out["pending"]), ([], [], None))
        self.adapter("v03", parent="v00", evaluated=False)
        st2 = dict(st, history=list(st["history"]))
        self.sweep(st2)
        config.save_state(st2)
        self.assertEqual(evaluate.latest_candidate(False), "v02", "abandoned v03 is not picked up by a standalone eval/promote")


if __name__ == "__main__":
    import unittest
    unittest.main()
