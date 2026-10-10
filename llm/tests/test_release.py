"""Release-side tests (no GPU, no Ollama, no judge): promote cleanup on failure, rollback (promote --to), and the shared
fixtures used by the nightly tests. Ollama is replaced by an in-memory fake behind promote.ollama().
Run: llm/.venv/Scripts/python -m unittest llm.tests.test_release"""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llm import config, evaluate, promote


class FakeOllama:
    """in-memory model registry answering `ollama list|create|cp|rm`."""

    def __init__(self, models=None, fail=()):
        self.models = dict(models or {})
        self.fail = set(fail)       # subcommands that fail ("create", "cp", ...)
        self.calls = []
        self.n = 0

    def __call__(self, *args):
        self.calls.append(args)
        cmd = args[0]
        ok = lambda out="": subprocess.CompletedProcess(args, 0, stdout=out, stderr="")  # noqa: E731
        if cmd in self.fail:
            if cmd == "create":
                self.models[args[1]] = "half-written"   # a failed create can leave a tag behind
            return subprocess.CompletedProcess(args, 1, stdout="", stderr=f"{cmd} failed")
        if cmd == "list":
            return ok("NAME ID SIZE MODIFIED\n" + "".join(f"{k} {v} 1.6GB now\n" for k, v in self.models.items()))
        if cmd == "create":
            self.n += 1
            self.models[args[1]] = f"id{self.n:04d}"
            return ok()
        if cmd == "cp":
            if args[1] not in self.models:
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="not found")
            self.models[args[2]] = self.models[args[1]]
            return ok()
        if cmd == "rm":
            self.models.pop(args[1], None)
            return ok()
        raise AssertionError(args)


class Sandbox(unittest.TestCase):
    """config.* pointed at a temp dir; merge / GGUF conversion stubbed (they only create files)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.saved = {k: getattr(config, k) for k in ("MODELS", "ADAPTERS", "MERGED", "STATE", "LOCK", "BASE_DIR")}
        config.MODELS = root
        config.ADAPTERS = root / "adapters"
        config.MERGED = root / "merged"
        config.STATE = root / "state.json"
        config.LOCK = root / "nightly.lock"
        config.BASE_DIR = root / "base"
        config.ADAPTERS.mkdir(parents=True)

        def fake_merge(base, adapter, out, log=print):
            out.mkdir(parents=True, exist_ok=True)
            (out / "model.safetensors").write_text("x")
            return out

        def fake_gguf(src, outtype="q8_0", log=print):
            f = src / f"model-{outtype}.gguf"
            f.write_text("gguf")
            return f

        self.patches = [mock.patch.object(promote, "merge", fake_merge), mock.patch.object(promote, "to_gguf", fake_gguf)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        for k, v in self.saved.items():
            setattr(config, k, v)
        self.tmp.cleanup()

    def adapter(self, ver, parent=None, new_ids=(), gate=None, evaluated=True):
        d = config.ADAPTERS / ver
        d.mkdir(parents=True)
        (d / "adapter_model.safetensors").write_text("a")
        (d / "meta.json").write_text(json.dumps({"ver": ver, "base": "B", "parent": parent, "new_ids": list(new_ids), "replay_ids": []}))
        if evaluated:
            g = gate if gate is not None else {"promote": True, "reason": "test pass"}
            (d / "eval.json").write_text(json.dumps({"gate": g, "loss": {"diff_cluster_ci": {}}, "production": parent}))
        return d

    def fake(self, **kw):
        f = FakeOllama(**kw)
        p = mock.patch.object(promote, "ollama", f)
        p.start()
        self.addCleanup(p.stop)
        return f


class PromoteCleanup(Sandbox):
    def test_success_points_latest_and_removes_merged(self):
        self.adapter("v1", new_ids=["a", "b"])
        ol = self.fake()
        rec = promote.run(log=lambda *_: None)
        self.assertEqual((rec["promoted"], rec["exit"]), (True, 0))
        self.assertEqual(ol.models["mesh-llm:latest"], ol.models["mesh-llm:v1"])
        st = config.load_state()
        self.assertEqual(st["production"]["ver"], "v1")
        self.assertEqual(st["trained_ids"], ["a", "b"])
        self.assertEqual(st["history"][-1]["action"], "promote")
        self.assertFalse((config.MERGED / "v1").exists())

    def test_create_failure_removes_merged_dir_and_tag(self):
        self.adapter("v1")
        ol = self.fake(fail={"create"})
        rec = promote.run(log=lambda *_: None)
        self.assertEqual((rec["promoted"], rec["exit"], rec["action"]), (False, 3, "promote_failed"))
        self.assertNotIn("mesh-llm:v1", ol.models)
        self.assertFalse((config.MERGED / "v1").exists())
        self.assertNotIn(("cp", "mesh-llm:v1", "mesh-llm:latest"), ol.calls)
        st = config.load_state()
        self.assertIsNone(st.get("production"))
        self.assertEqual(st["history"][-1]["action"], "promote_failed")

    def test_keep_merged_keeps_dir_even_on_failure(self):
        self.adapter("v1")
        self.fake(fail={"create"})
        promote.run(keep_merged=True, log=lambda *_: None)
        self.assertTrue((config.MERGED / "v1").exists())

    def test_cp_failure_restores_previous_latest(self):
        self.adapter("v0")
        self.adapter("v1", parent="v0")
        config.save_state({"trained_ids": ["old"], "production": {"ver": "v0", "ollama": "mesh-llm:v0"},
                           "history": [{"ver": "v0", "promoted": True, "action": "promote", "synthetic": False}]})
        ol = self.fake(models={"mesh-llm:v0": "id-old", "mesh-llm:latest": "id-old"}, fail={"cp"})
        rec = promote.run(ver="v1", log=lambda *_: None)
        self.assertEqual(rec["exit"], 3)
        self.assertIn(("cp", "mesh-llm:v0", "mesh-llm:latest"), ol.calls, "previous production restored as latest")
        self.assertNotIn("mesh-llm:v1", ol.models)
        st = config.load_state()
        self.assertEqual(st["production"]["ver"], "v0")
        self.assertEqual(st["trained_ids"], ["old"])

    def test_digest_mismatch_after_cp_is_a_failure(self):
        self.adapter("v1")
        ol = self.fake()
        orig = ol.__call__

        def lying(*args):
            r = orig(*args)
            if args[0] == "cp":
                ol.models[args[2]] = "stale"
            return r

        with mock.patch.object(promote, "ollama", lying):
            rec = promote.run(log=lambda *_: None)
        self.assertEqual(rec["exit"], 3)
        self.assertIn("digest", rec["error"])

    def test_failed_promotion_is_retried_once_then_decided(self):
        self.adapter("v1")
        self.fake(fail={"create"})
        self.assertEqual(promote.run(log=lambda *_: None)["exit"], 3)
        self.assertEqual(promote.run(log=lambda *_: None)["exit"], 3)
        self.assertEqual(promote.run(log=lambda *_: None)["reason"], "already decided")

    def test_cli_exit_code(self):
        from llm.__main__ import main
        self.adapter("v1")
        self.fake(fail={"create"})
        with mock.patch("builtins.print"):
            self.assertEqual(main(["promote"]), 3)


class Rollback(Sandbox):
    def setUp(self):
        super().setUp()
        self.adapter("v1", new_ids=["a"])
        self.adapter("v2", parent="v1", new_ids=["b"])
        self.adapter("v3", parent="v2", new_ids=["c", "d"])
        self.adapter("v9-syn", new_ids=["s"])
        h = lambda v, **kw: {"ver": v, "promoted": True, "action": "promote", "synthetic": False, **kw}  # noqa: E731
        config.save_state({"trained_ids": ["a", "b", "c", "d"], "production": {"ver": "v3", "ollama": "mesh-llm:v3"},
                           "history": [h("v1"), h("v2"), h("v3"), {"ver": "v9-syn", "promoted": True, "synthetic": True},
                                       {"ver": "v4", "promoted": False, "action": "reject"}]})

    def test_rollback_resets_lineage_ids_and_latest(self):
        ol = self.fake(models={"mesh-llm:v2": "id2", "mesh-llm:v3": "id3", "mesh-llm:latest": "id3"})
        rec = promote.rollback("v2", log=lambda *_: None)
        self.assertEqual((rec["exit"], rec["action"], rec["reimported"]), (0, "rollback", False))
        self.assertEqual(ol.models["mesh-llm:latest"], "id2")
        st = config.load_state()
        self.assertEqual(st["production"]["ver"], "v2")
        self.assertEqual(st["production"]["rolled_back_from"], "v3")
        self.assertEqual(st["trained_ids"], ["a", "b"], "union of new_ids along v2 -> v1")
        self.assertEqual(st["history"][-1]["action"], "rollback")

    def test_missing_tag_is_reimported(self):
        ol = self.fake(models={"mesh-llm:v3": "id3", "mesh-llm:latest": "id3"})
        rec = promote.rollback("v1", log=lambda *_: None)
        self.assertEqual((rec["exit"], rec["reimported"]), (0, True))
        self.assertIn("mesh-llm:v1", ol.models)
        self.assertEqual(ol.models["mesh-llm:latest"], ol.models["mesh-llm:v1"])
        self.assertFalse((config.MERGED / "v1").exists())
        self.assertEqual(config.load_state()["trained_ids"], ["a"])

    def test_unknown_unpromoted_or_synthetic_version_exits_4(self):
        ol = self.fake(models={"mesh-llm:latest": "id3"})
        for v in ("nope", "v4", "v9-syn"):
            self.assertEqual(promote.rollback(v, log=lambda *_: None)["exit"], 4, v)
        self.assertEqual(ol.models["mesh-llm:latest"], "id3")
        self.assertEqual(config.load_state()["production"]["ver"], "v3")

    def test_cli_rollback_exit_codes(self):
        from llm.__main__ import main
        self.fake(models={"mesh-llm:v2": "id2", "mesh-llm:latest": "id3"})
        with mock.patch("builtins.print"):
            self.assertEqual(main(["promote", "--to", "nope"]), 4)
            self.assertEqual(main(["promote", "--to", "v2"]), 0)

    def test_failed_rollback_restores_latest(self):
        ol = self.fake(models={"mesh-llm:v2": "id2", "mesh-llm:v3": "id3", "mesh-llm:latest": "id3"}, fail={"cp"})
        rec = promote.rollback("v2", log=lambda *_: None)
        self.assertEqual(rec["exit"], 3)
        self.assertIn(("cp", "mesh-llm:v3", "mesh-llm:latest"), ol.calls)
        self.assertEqual(config.load_state()["production"]["ver"], "v3")


class GateConstants(unittest.TestCase):
    def test_gate_constants_unchanged(self):
        self.assertEqual((evaluate.N_LOSS, evaluate.N_TASK, evaluate.MIN_PAIR_ITEMS, evaluate.N_DET, evaluate.DET_MAX_NEW), (200, 60, 40, 20, 768))
        self.assertEqual((evaluate.CAT_MAX_DROP, evaluate.POOLED_MAX_DROP, evaluate.CAT_SIGN_ALPHA), (0.10, 0.05, 0.10))
        self.assertEqual((evaluate.MIN_N, evaluate.MIN_SESSIONS), (30, 3))


if __name__ == "__main__":
    unittest.main()
