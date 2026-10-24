"""llm v2: continual lineage (D), DPO pair eligibility (C), rejection-sampling self-training (B), budget accounting.
GPU-free: model / training calls are stubbed. Run: llm/.venv/Scripts/python -m pytest -q llm/tests/test_v2.py"""
import json
from datetime import datetime
from unittest import mock

from llm import config, contract, dpo, evaluate, lineage, nightly, promote, selftrain, train
from llm.tests.test_release import Sandbox

QUIET = lambda *_: None  # noqa: E731


def ev(win=0.5, n=60, regressed=(), loss_hi=0.1, trusted=True, promote_=False):
    return {"gate": {"promote": promote_, "reason": "stub"}, "judge_trusted": {"ok": trusted, "why": ""},
            "pairwise": {"n_judged": n, "winrate_ci": {"mean": win, "lo": win - 0.05, "hi": win + 0.05}},
            "deterministic": {"regressed": list(regressed)}, "loss": {"diff_cluster_ci": {"n": 50, "lo": loss_hi - 0.2, "hi": loss_hi}},
            "production": None}


class V2(Sandbox):
    def setUp(self):
        super().setUp()
        self.saved_v2 = {k: getattr(config, k) for k in ("DATA", "SFT", "EXTERNAL", "EVAL_EXTERNAL", "SELF_VERIFIED", "DPO_PAIRS",
                                                          "CONTINUAL", "MAX_CHAIN")}
        root = config.MODELS / "data"
        root.mkdir()
        config.DATA, config.SFT, config.EXTERNAL = root, root / "sft.jsonl", root / "sft_external.jsonl"
        config.EVAL_EXTERNAL, config.SELF_VERIFIED, config.DPO_PAIRS = root / "eval.jsonl", root / "self.jsonl", root / "pairs.jsonl"
        config.CONTINUAL, config.MAX_CHAIN = True, 3
        self._frac = selftrain.CONTRACT_FRAC
        selftrain.CONTRACT_FRAC = 0.0          # plain format unless a test asks for the JSON contract

    def tearDown(self):
        for k, v in self.saved_v2.items():
            setattr(config, k, v)
        selftrain.CONTRACT_FRAC = self._frac
        super().tearDown()

    def cand(self, ver, parent=None, e=None, base="B"):
        d = self.adapter(ver, parent=parent, evaluated=False)
        m = json.loads((d / "meta.json").read_text())
        m["base"] = base
        (d / "meta.json").write_text(json.dumps(m))
        if e is not None:
            (d / "eval.json").write_text(json.dumps(e))
        return d


class Lineage(V2):
    def test_not_worse_rules(self):
        self.assertTrue(lineage.not_worse(ev(0.5))[0], "a statistical tie at 0.5 is not worse")
        self.assertFalse(lineage.not_worse(ev(0.49))[0], "point estimate below the incumbent")
        self.assertFalse(lineage.not_worse(ev(0.6, regressed=["math"]))[0], "battery regression")
        self.assertFalse(lineage.not_worse(ev(0.6, loss_hi=-0.01))[0], "loss significantly worse")
        self.assertFalse(lineage.not_worse(ev(0.6, n=10))[0], "too few judged items")
        self.assertFalse(lineage.not_worse(ev(0.6, trusted=False))[0], "untrusted judge")

    def test_parent_fresh_then_head_then_dead_end(self):
        st = {"history": []}
        self.assertEqual(lineage.parent_for_training(st, "B")[0], None)
        self.cand("v01", e=ev(0.52))
        lineage.record(st, "v01", promoted=False, log=QUIET)
        self.assertEqual(st["lineage"]["ver"], "v01")
        self.assertEqual(lineage.parent_for_training(st, "B")[0], "v01", "not-worse reject becomes the parent")
        self.cand("v02", parent="v01", e=ev(0.40))
        lineage.record(st, "v02", promoted=False, log=QUIET)
        self.assertEqual(st["lineage"]["ver"], "v01", "a worse candidate is a dead end")
        self.assertFalse(st["lineage_log"][-1]["became_head"])
        self.assertEqual(lineage.parent_for_training(st, "B")[0], "v01")
        self.assertEqual(lineage.parent_for_training(st, "OTHER")[0], None, "head on another base is never a parent")

    def test_chain_depth_resets_to_production(self):
        self.cand("p0")
        st = {"production": {"ver": "p0", "base": "B"}, "history": []}
        parent = "p0"
        for i in range(1, 4):
            v = f"v{i:02d}"
            self.cand(v, parent=parent, e=ev(0.5))
            lineage.record(st, v, promoted=False, log=QUIET)
            self.assertEqual(st["lineage"]["depth"], i)
            parent = v
        p, why = lineage.parent_for_training(st, "B")
        self.assertEqual(p, "p0", why)
        self.assertIn("chain reset", why)

    def test_promotion_sets_head_depth_0_and_rollback_resets(self):
        st = {"history": []}
        self.cand("v01", e=ev(0.7, promote_=True))
        lineage.record(st, "v01", promoted=True, log=QUIET)
        self.assertEqual((st["lineage"]["ver"], st["lineage"]["depth"]), ("v01", 0))
        lineage.reset_to(st, "v00", "B", "rollback")
        self.assertEqual(st["lineage"]["ver"], "v00")

    def test_continual_off_uses_production_only(self):
        config.CONTINUAL = False
        self.cand("v01", e=ev(0.52))
        st = {"history": []}
        lineage.record(st, "v01", promoted=False, log=QUIET)
        self.assertIsNone(lineage.parent_for_training(st, "B")[0])

    def test_sweep_keeps_candidate_trained_from_head(self):
        self.cand("v01", e=ev(0.55))
        st = {"history": [{"ver": "v01", "promoted": False, "action": "reject"}]}
        lineage.record(st, "v01", promoted=False, log=QUIET)
        self.cand("v02", parent="v01")
        out = nightly.abandoned_sweep(st, log=QUIET)
        self.assertEqual((out["abandoned"], out["pending"]), ([], "v02"))
        self.cand("v03", parent="v99")
        out = nightly.abandoned_sweep(st, log=QUIET)
        self.assertEqual(out["abandoned"][0], {"ver": "v02", "reason": "superseded"})
        self.assertIn("stale parent", out["abandoned"][1]["reason"])

    def test_promote_reject_records_lineage(self):
        self.cand("v01", e=ev(0.51, promote_=False))
        promote.run(ver="v01", log=QUIET)
        st = config.load_state()
        self.assertEqual(st["lineage"]["ver"], "v01")
        self.assertEqual(st["history"][-1]["action"], "reject")

    def test_chain_ids(self):
        self.adapter("a1", new_ids=["x"], evaluated=False)
        self.adapter("a2", parent="a1", new_ids=["y"], evaluated=False)
        self.assertEqual(train.chain_ids("a2", "new_ids"), {"x", "y"})
        self.assertEqual(train.chain_ids(None, "new_ids"), set())


def jl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def row(i, instr, session="s"):
    return {"id": i, "ts": "2026-10-01T00:00:00Z", "session": session, "instruction": instr, "response": "r " + i}


class DpoPairs(V2):
    def judged(self, ver, items, winners, how="both-orders"):
        d = self.cand(ver)
        detail = [{"id": i, "winner": w, "how": how} for i, w in zip(items, winners)]
        (d / "eval.json").write_text(json.dumps({"pairwise": {"detail": detail}}))
        (d / "answers.json").write_text(json.dumps({"answers": {"task_cand": [f"cand {i}" for i in items],
                                                                "task_prod": [f"prod {i}" for i in items]}}))

    def test_only_current_train_rows_count(self):
        tr = [row("t1", "alpha beta gamma delta one"), row("t2", "unrelated train request two")]
        held = [row("h1", "a held out request about something else entirely")]
        jl(config.EVAL_EXTERNAL, [row("e1", "an open source eval question about rivers and lakes")])
        self.judged("v01", ["t1", "t2", "h1", "e1"], ["x", "tie", "y", "x"])
        with mock.patch("llm.anchors.battery_shingles", return_value=set()):
            pairs = dpo.eligible_pairs(tr, held, log=QUIET)
        self.assertEqual([(p["row_id"], p["chosen"], p["rejected"]) for p in pairs], [("t1", "cand t1", "prod t1")])

    def test_order_disagreement_and_shingle_overlap_dropped(self):
        long_q = "what is the boiling point of water at sea level in degrees celsius please"
        tr = [row("t1", long_q), row("t2", "short different request")]
        held = [row("h1", "tell me " + long_q)]
        d = self.cand("v01")
        (d / "eval.json").write_text(json.dumps({"pairwise": {"detail": [{"id": "t1", "winner": "x", "how": "both-orders"},
                                                                         {"id": "t2", "winner": "y", "how": "order-disagree"}]}}))
        (d / "answers.json").write_text(json.dumps({"answers": {"task_cand": ["a", "b"], "task_prod": ["c", "d"]}}))
        with mock.patch("llm.anchors.battery_shingles", return_value=set()):
            self.assertEqual(dpo.eligible_pairs(tr, held, log=QUIET), [], "t1 paraphrases a held-out row; t2 was a tie")

    def test_checker_pairs_and_switch(self):
        jl(config.DPO_PAIRS, [{"id": "dpos:1", "row_id": "bench:math:1", "instruction": "q one", "chosen": "a", "rejected": "b",
                               "source": "checker"}])
        with mock.patch("llm.anchors.battery_shingles", return_value=set()):
            self.assertEqual(len(dpo.eligible_pairs([], [], log=QUIET)), 1)
            with mock.patch.object(config, "DPO_SELF_PAIRS", False):
                self.assertEqual(len(dpo.eligible_pairs([], [], log=QUIET)), 0)


class SelfTrain(V2):
    def test_checker_kinds(self):
        C = selftrain.checker()
        self.assertTrue(selftrain.grade(C, {"kind": "int", "answer": "42"}, "so...\nANSWER: 42"))
        self.assertFalse(selftrain.grade(C, {"kind": "int", "answer": "42"}, "ANSWER: 41"))
        self.assertTrue(selftrain.grade(C, {"kind": "mcq", "answer": "C"}, "ANSWER: (C)"))
        self.assertTrue(selftrain.grade(C, {"kind": "cwe", "answer": "CWE-79"}, "ANSWER: CWE-79"))
        self.assertTrue(selftrain.grade(C, {"kind": "numeric", "answer": "3.14"}, "ANSWER: 3.141"))

    def test_mmqa_gold_and_pool_excludes_eval(self):
        jl(config.EXTERNAL, [
            {"id": "ext:1", "source": "oss:metamathqa", "instruction": "Tom has 3 apples and buys 4 more apples at the store today",
             "response": "3+4=7\nThe answer is: 7"},
            {"id": "ext:2", "source": "oss:metamathqa", "instruction": "Sara reads 12 pages a day for a whole week of holidays",
             "response": "The answer is: one half"},
            {"id": "ext:3", "source": "oss:metamathqa", "instruction": "Compute the value asked in the rivers and lakes question now",
             "response": "The answer is: 1,200"}])
        jl(config.EVAL_EXTERNAL, [row("e1", "Compute the value asked in the rivers and lakes question now please")])
        jl(config.SFT, [])
        with mock.patch.object(selftrain, "BENCH", config.MODELS / "nobench"), mock.patch("llm.anchors.battery_shingles", return_value=set()):
            p = selftrain.pool(log=QUIET)
        self.assertEqual([(t["id"], t["kind"], t["answer"]) for t in p], [("mmqa:ext:1", "int", "7")])
        self.assertTrue(selftrain.instruction(p[0]).endswith("`ANSWER: <answer>`."))

    def test_bench_tune_split_only(self):
        b = config.MODELS / "bench"
        b.mkdir()
        jl(b / "physics.jsonl", [{"id": "p1", "kind": "mcq", "split": "tune", "prompt": "q1", "answer": "A"},
                                 {"id": "p2", "kind": "mcq", "split": "held", "prompt": "q2", "answer": "B"}])
        with mock.patch.object(selftrain, "BENCH", b):
            self.assertEqual([t["id"] for t in selftrain._bench(docker=False)], ["bench:physics:p1"])

    def test_run_keeps_verified_and_writes_pairs(self):
        tasks = [{"id": "t1", "kind": "int", "answer": "7", "prompt": "q1", "origin": "x"},
                 {"id": "t2", "kind": "int", "answer": "9", "prompt": "q2", "origin": "x"}]
        outs = ["ANSWER: 7", "long wrong\nANSWER: 8", "ANSWER: 1", "ANSWER: 2"]
        with mock.patch.object(selftrain, "CONTRACT_FRAC", 0.0), mock.patch("llm.model.tokenizer"), \
                mock.patch("llm.model.load_4bit", return_value=mock.MagicMock()), \
                mock.patch("llm.model.sample_batch", return_value=outs), mock.patch("llm.judge.unload"), \
                mock.patch("torch.cuda.empty_cache"), mock.patch.object(selftrain, "SEEN", config.MODELS / "seen.json"):
            st = selftrain.run(minutes=5, k=2, gen_bs=4, tasks=tasks, log=QUIET)
        self.assertEqual((st["prompts"], st["correct"], st["kept"], st["pairs"]), (2, 1, 1, 1))
        rows = [json.loads(x) for x in config.SELF_VERIFIED.read_text().splitlines()]
        self.assertEqual((rows[0]["source"], rows[0]["response"], rows[0]["pass_k"]), ("self_verified", "ANSWER: 7", "1/2"))
        pr = json.loads(config.DPO_PAIRS.read_text().splitlines()[0])
        self.assertEqual((pr["chosen"], pr["rejected"], pr["source"]), ("ANSWER: 7", "long wrong\nANSWER: 8", "checker"))

    def test_standalone_refuses_in_night_window(self):
        self.assertTrue(selftrain.in_night_window(datetime(2026, 10, 22, 3, 0)))
        self.assertFalse(selftrain.in_night_window(datetime(2026, 10, 22, 8, 30)))
        with mock.patch.object(selftrain, "in_night_window", return_value=True):
            self.assertIn("refused", selftrain.run(minutes=5, nightly=False, log=QUIET)["reason"])

    def test_sample_self_prefers_unseen(self):
        jl(config.SELF_VERIFIED, [{"id": f"s{i}", "instruction": "q", "response": "a"} for i in range(4)])
        got = train.sample_self(2, {"s0", "s1"}, seed=1)
        self.assertEqual({r["id"] for r in got}, {"s2", "s3"})


class NightlyBudget(V2):
    def test_selftrain_minutes_come_out_of_the_training_budget(self):
        seen = {}

        def fake_self(minutes, log=print, **kw):
            seen["self_budget"] = minutes
            return {"kept": 3, "minutes": 12.0, "reason": "stub"}

        def fake_train(synthetic=False, epochs=2, log=print, budget_minutes=None, **kw):
            seen["train_budget"] = budget_minutes
            return None
        with mock.patch.object(config, "SELFTRAIN_MINUTES", 20.0), mock.patch.object(config, "TRAIN_MINUTES", 110.0), \
                mock.patch.object(selftrain, "run", fake_self), mock.patch.object(train, "run", fake_train):
            rec = nightly.run(power=lambda: True, log=QUIET)
        self.assertEqual([s["name"] for s in rec["stages"]], ["preflight", "selftrain", "train", "eval", "promote"])
        self.assertEqual((seen["self_budget"], seen["train_budget"]), (20.0, 98.0))

class JsonContract(V2):
    GOOD = '{"answer": 42, "confidence": 0.8, "evidence": ["6*7=42"], "assumptions": []}'

    def test_parse_valid_and_invalid(self):
        self.assertEqual(contract.parse(self.GOOD)["answer"], 42)
        self.assertIsNotNone(contract.parse("```json\n" + self.GOOD + "\n```"), "json fence tolerated")
        self.assertIsNotNone(contract.parse("Here you go: " + self.GOOD), "surrounding text tolerated")
        self.assertIsNone(contract.parse('{"answer": 1, "confidence": 1.5, "evidence": [], "assumptions": []}'), "confidence > 1")
        self.assertIsNone(contract.parse('{"answer": 1, "confidence": 0.5, "evidence": "x", "assumptions": []}'), "evidence not a list")
        self.assertIsNone(contract.parse('{"answer": 1, "confidence": 0.5}'), "missing keys")
        self.assertIsNone(contract.parse("ANSWER: 42"))

    def test_normalize_to_plain_formats(self):
        self.assertTrue(contract.normalize(self.GOOD).endswith("ANSWER: 42"))
        code = '{"answer": "def f(x):\\n    return x", "confidence": 0.9, "evidence": [], "assumptions": ["ints"]}'
        self.assertEqual(contract.normalize(code), "```python\ndef f(x):\n    return x\n```")
        self.assertEqual(contract.normalize("plain\nANSWER: 3"), "plain\nANSWER: 3", "plain text unchanged")

    def test_battery_scoring_accepts_both_formats(self):
        bench = mock.MagicMock()
        bench.score = lambda item, text: {"correct": text.strip().splitlines()[-1] == "ANSWER: 42", "infra": False}
        self.assertTrue(evaluate.det_correct(bench, {"id": "m1"}, "ANSWER: 42"))
        self.assertTrue(evaluate.det_correct(bench, {"id": "m1"}, self.GOOD))
        self.assertFalse(evaluate.det_correct(bench, {"id": "m1"}, self.GOOD.replace("42", "41")))

    def test_selftrain_contract_prompts_need_a_valid_contract(self):
        C = selftrain.checker()
        t = {"id": "x", "kind": "int", "answer": "42", "prompt": "q"}
        with mock.patch.object(selftrain, "CONTRACT_FRAC", 1.0):
            self.assertIn('"confidence"', selftrain.instruction(t))
            self.assertTrue(selftrain.grade(C, t, self.GOOD))
            self.assertFalse(selftrain.grade(C, t, "ANSWER: 42"), "asked for the contract, answered plain")
        with mock.patch.object(selftrain, "CONTRACT_FRAC", 0.0):
            self.assertTrue(selftrain.grade(C, t, "ANSWER: 42"))
            self.assertNotIn('"confidence"', selftrain.instruction(t))
