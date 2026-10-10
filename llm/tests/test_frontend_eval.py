"""Front-end eval v2 (llm/frontend_eval.py) with a stub backend: no tokenizer, no MiniLM, no Ollama, no judge, no Rust binary."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from llm import frontend, frontend_eval as fe


def make_rows(n_sessions=12, per=10, worlds=("agent_report", "human_prompt")):
    rows = []
    for s in range(n_sessions):
        for j in range(per):
            k = s * per + j
            rows.append({"id": f"r{k:03d}", "ts": f"2026-10-{1 + s:02d}T{8 + j:02d}:00:00Z", "project": "p", "session": f"s{s:02d}",
                         "source": worlds[s % len(worlds)], "instruction": f"question number {k} about topic {k % 7}", "response": f"answer {k}",
                         "outcome": "success", "judge_score": 1.0, "judge_model": "j"})
    return rows


class StubBackend:
    """near-duplicate vectors (cos ~ .9) so nothing is OOD; the judge says 1 iff the answer equals the reference"""

    def __init__(self, rows, wrong_ids=(), conf=-0.05):
        self.ref = {r["instruction"]: r["response"] for r in rows}
        self.wrong, self.conf = set(wrong_ids), conf
        self.by_instr = {r["instruction"]: r["id"] for r in rows}
        self.calls = {"ask": 0, "grade": 0}
        self.name = {"local_model": "stub", "judge": "stub"}

    def ntok(self, s):
        return len((s or "").split())

    def encode(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            v = np.ones(8, np.float32) + 0.2 * np.random.default_rng(seed).standard_normal(8).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)

    def ask(self, instruction):
        self.calls["ask"] += 1
        rid = self.by_instr[instruction]
        return ("wrong" if rid in self.wrong else self.ref[instruction]), self.conf, 3

    def tool(self, vec):
        return None

    def grade(self, instruction, reference, answer):
        self.calls["grade"] += 1
        return 1.0 if answer == reference else 0.0

    def done_asking(self):
        pass

    def close(self):
        pass


def py_rust(cfg, sig, chunks):
    d, why = frontend.py_decide(cfg, sig)
    return {"ok": True, "decision": d, "reason": why, "context": [], "context_tokens": 0 if d != "forward" else 5}


class ClusterSplit(unittest.TestCase):
    def test_disjoint_time_ordered_and_no_trained_leak(self):
        rows = make_rows(6, 6)
        meta = {r["id"]: {"cluster": i // 2} for i, r in enumerate(rows)}       # clusters of 2 consecutive rows
        del meta["r035"]                                                         # no meta -> its session is the unit
        trained = {"r004", "r011"}                                               # clusters 2 and 5
        sp = fe.cluster_split(rows, meta, trained)
        ids = {k: {r["id"] for r in sp[k]} for k in ("tune", "held", "conf")}
        self.assertFalse(ids["tune"] & ids["held"] or ids["tune"] & ids["conf"] or ids["held"] & ids["conf"])
        everything = ids["tune"] | ids["held"] | ids["conf"]
        self.assertFalse(everything & {"r004", "r005", "r010", "r011"})          # trained rows AND their cluster mates
        self.assertEqual(set(sp["dropped_ids"]), {"r004", "r005", "r010", "r011"})
        self.assertEqual(len(everything) + 4, len(rows))
        units = {k: {fe.unit_of(r, meta) for r in sp[k]} for k in ("tune", "held", "conf")}
        self.assertFalse(units["tune"] & units["held"] or units["held"] & units["conf"] or units["tune"] & units["conf"])
        first = lambda k: [min(r["ts"] for r in sp[k] if fe.unit_of(r, meta) == u) for u in sp["units"][k]]  # noqa: E731
        self.assertLessEqual(max(first("tune")), min(first("held")))
        self.assertLessEqual(max(first("held")), min(first("conf")))
        for k in ("tune", "held", "conf"):
            self.assertEqual([r["ts"] for r in sp[k]], sorted(r["ts"] for r in sp[k]))
        self.assertTrue(all(abs(len(sp["units"][a]) - len(sp["units"][b])) <= 1 for a in sp["units"] for b in sp["units"]))


class EvalV2(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.out = Path(self.d.name)

    def tearDown(self):
        self.d.cleanup()

    def run_eval(self, rows, backend=None, meta=None):
        made = []

        def factory():
            made.append(backend or StubBackend(rows))
            return made[-1]
        meta = meta if meta is not None else {r["id"]: {"cluster": i} for i, r in enumerate(rows)}
        res = fe.evaluate_v2(rows, meta, [], factory, decide=py_rust, out_dir=self.out, log=lambda *a: None, reps=400)
        return res, made

    def test_one_session_is_insufficient_and_calls_no_model(self):
        rows = make_rows(1, 120)
        res, made = self.run_eval(rows)
        self.assertEqual(res["verdict"], "INSUFFICIENT")
        self.assertEqual(made, [])
        self.assertIn("sessions", res["gates"]["tune"]["failed"])
        self.assertEqual(json.loads((self.out / "frontend_eval.json").read_text())["schema"], 2)
        self.assertFalse((self.out / "frontend_cfg.json").exists())

    def test_one_world_is_insufficient(self):
        rows = make_rows(12, 10, worlds=("agent_report",))
        res, _ = self.run_eval(rows)
        self.assertEqual(res["verdict"], "INSUFFICIENT")
        self.assertIn("worlds", res["gates"]["held"]["failed"])

    def test_good_local_answers_keep_and_write_cfg(self):
        rows = make_rows(12, 10)
        res, made = self.run_eval(rows)
        self.assertEqual(res["verdict"], "KEEP", res.get("why"))
        cfg = json.loads((self.out / "frontend_cfg.json").read_text())
        self.assertNotIn("enabled", cfg)
        self.assertGreaterEqual(cfg["min-conf"], -1.0)
        self.assertGreaterEqual(cfg["min-sim"], 0.3)
        self.assertNotIn(-99.0, [g["min-conf"] for g in res["sweep_on_tune"]])
        self.assertTrue(res["worlds_saved_both"])
        self.assertTrue(all("p_holm" in t for t in res["tests"]))
        self.assertEqual(len(res["tests"]), 2 * 2 * 3)   # (held, conf) x 2 worlds x (saved, loss, tost)

    def test_non_keep_never_writes_cfg(self):
        rows = make_rows(12, 10)
        # 1) no tuning point (30% wrong at every threshold) -> REJECTED
        bad = StubBackend(rows, wrong_ids={r["id"] for i, r in enumerate(rows) if i % 3 == 0})
        res, _ = self.run_eval(rows, backend=bad)
        self.assertEqual(res["verdict"], "REJECTED")
        self.assertFalse((self.out / "frontend_cfg.json").exists())
        # 2) tuning works on TUNE, but HELD/CONF answers are wrong -> REJECTED by the wrong-rate bound
        later = {r["id"] for r in rows[40:] if int(r["id"][1:]) % 2 == 0}
        res, _ = self.run_eval(rows, backend=StubBackend(rows, wrong_ids=later))
        self.assertEqual(res["verdict"], "REJECTED")
        self.assertTrue(any("upper bound" in w for w in res["why"]))
        self.assertFalse((self.out / "frontend_cfg.json").exists())
        # 3) the Rust policy failing closed is never a KEEP
        res = fe.evaluate_v2(rows, {r["id"]: {"cluster": i} for i, r in enumerate(rows)}, [], lambda: StubBackend(rows),
                             decide=lambda c, s, ch: frontend._fail("boom"), out_dir=self.out, log=lambda *a: None, reps=200)
        self.assertEqual(res["verdict"], "REJECTED")
        self.assertFalse((self.out / "frontend_cfg.json").exists())

    def test_rolling_origin_bank(self):
        rows = make_rows(3, 4)
        be = StubBackend(rows)
        recs = fe.signals(rows, rows, be, log=lambda *a: None)
        self.assertIsNone(recs[0]["max_sim"])                  # the first row has no earlier rows
        self.assertEqual(recs[0]["chunks"], [])
        later = [c["id"] for c in recs[5]["chunks"]]
        self.assertTrue(later and all(c < "r005" for c in later))


if __name__ == "__main__":
    unittest.main()
