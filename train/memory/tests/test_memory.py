import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)
from train.memory import api, cards, client, config as C, daemon, search, store  # noqa: E402


def _card(task, field, text, vec=None):
    return {"id": "card:" + task, "task": task, "field": field, "agent": "a", "verified": True, "problem": text,
            "text": "Type: t\nMethod: " + text}


def _vec(*xs):
    v = np.zeros(384, np.float32)
    for i, x in enumerate(xs):
        v[i] = x
    return v / (np.linalg.norm(v) or 1)


class _TmpStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._old = C.MEM_DIR
        C.MEM_DIR = Path(self.tmp)

    def tearDown(self):
        C.MEM_DIR = self._old


class FusionTest(unittest.TestCase):
    def test_rrf_rewards_agreement(self):
        f = search.rrf([[1, 2, 3], [2, 3, 1]], kk=60)
        self.assertAlmostEqual(f[2], 1 / 62 + 1 / 61)
        self.assertEqual(max(f, key=f.get), 2)

    def test_rrf_single_list_keeps_order(self):
        f = search.rrf([[5, 4, 9]])
        self.assertEqual(sorted(f, key=lambda i: -f[i]), [5, 4, 9])


class GateFilterTest(unittest.TestCase):
    def setUp(self):
        cs = [_card("a", "math", "modular arithmetic residues"), _card("b", "math", "graph coloring"),
              _card("c", "physics", "modular arithmetic residues")]
        self.idx = search.Index(cs, np.stack([_vec(1, 0), _vec(0.3, 1), _vec(1, 0.05)]))

    def test_threshold_field_exclude_and_k(self):
        q = _vec(1, 0)
        hits = self.idx.search("modular residues", q, field="math", k=2, mode="dense", gate=0.9)
        self.assertEqual([c["task"] for _, c in hits], ["a"])                  # b is below the gate, c other field
        self.assertEqual(self.idx.search("x", q, field="math", k=2, mode="dense", gate=0.9, exclude={"a"}), [])
        self.assertEqual(len(self.idx.search("x", q, k=1, mode="dense", gate=-1)), 1)

    def test_hybrid_and_bm25_modes(self):
        q = _vec(1, 0)
        self.assertEqual(self.idx.search("graph coloring", q, field="math", k=1, mode="bm25", gate=0)[0][1]["task"], "b")
        hy = self.idx.search("modular residues", q, field="math", k=2, mode="hybrid", gate=0.5)
        self.assertEqual([c["task"] for _, c in hy], ["a"])

    def test_rerank_uses_cross_encoder_order(self):
        class CE:
            def score(self, q, ps):
                return np.array([10.0 if "graph" in p else 0.0 for p in ps], np.float32)
        hits = self.idx.search("anything", _vec(1, 0), field="math", k=2, mode="rerank", ce=CE(), gate=5.0)
        self.assertEqual([c["task"] for _, c in hits], ["b"])

    def test_low_usefulness_card_is_not_broadcast(self):
        st = {"cards": {"card:a": {"s": 0, "f": 6}}}
        hits = self.idx.search("x", _vec(1, 0), field="math", k=2, mode="dense", gate=-1, stats=st)
        self.assertNotIn("a", [c["task"] for _, c in hits])


class CardTest(unittest.TestCase):
    def test_scrub_drops_card_with_secret(self):
        t = {"id": "t1", "prompt": "Compute the sum of the first n odd numbers.", "field": "math"}
        r = {"text": "Use induction since 1+3+...=n^2. My key is sk-ant-api03-" + "A" * 40 + " ok", "agent": "x"}
        self.assertIsNone(cards.make_card(t, r, distill="extractive"))

    def test_extractive_card_is_capped_and_has_provenance(self):
        t = {"id": "t2", "prompt": "Compute the sum of the first n odd numbers. " * 5, "field": "math"}
        r = {"text": "We use induction since the sum telescopes. " * 30 + "\nCheck: verified for n=1..5.\nANSWER: 42", "agent": "x"}
        c = cards.make_card(t, r, distill="extractive", batch="b")
        self.assertTrue(c["verified"])
        self.assertEqual((c["task"], c["agent"], c["batch"]), ("t2", "x", "b"))
        self.assertLessEqual(C.approx_tokens(c["text"]), C.CARD_TOKENS)
        self.assertNotIn("ANSWER: 42", c["text"])

    def test_render_respects_injection_budget(self):
        hits = [(1.0, {"id": str(i), "text": "w" * 470}) for i in range(5)]
        out = api.render(hits)
        self.assertLessEqual(C.approx_tokens(out), C.INJECT_TOKENS)
        self.assertIn("[1]", out)


class UsefulnessTest(_TmpStore):
    def test_observe_counts(self):
        api.observe(["card:a", "card:b"], True)
        api.observe(["card:a"], False)
        st = store.load_stats()
        self.assertEqual(st["cards"]["card:a"]["s"], 1)
        self.assertEqual(st["cards"]["card:a"]["f"], 1)
        self.assertAlmostEqual(store.usefulness(st, "card:a")[0], 0.5)

    def test_batch_updater_is_incremental_and_skips_errors_and_ambiguous(self):
        api.log_injection("t1", ["card:x"])
        api.log_injection("t2", ["card:y"])
        api.log_injection("t2", ["card:z"])            # t2 got different cards at different times -> ambiguous
        res = Path(self.tmp) / "results.jsonl"
        rows = [{"task": "t1", "passed": True}, {"task": "t1", "passed": False}, {"task": "t1", "passed": True, "error": "quota"},
                {"task": "t2", "passed": True}, {"task": "t3", "passed": True}]
        res.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.assertEqual(api.update_from_results([res]), 2)
        self.assertEqual(api.update_from_results([res]), 0)         # cursor: nothing counted twice
        st = store.load_stats()
        self.assertEqual((st["cards"]["card:x"]["s"], st["cards"]["card:x"]["f"]), (1, 1))
        self.assertNotIn("card:y", st["cards"])
        with open(res, "a", encoding="utf-8") as f:
            f.write(json.dumps({"task": "t1", "passed": True}) + "\n")
        self.assertEqual(api.update_from_results([res]), 1)

    def test_store_add_dedups_and_requires_verified(self):
        enc = lambda texts: np.stack([_vec(1) for _ in texts])
        cs = [_card("a", "math", "x"), _card("a", "math", "x"), dict(_card("b", "math", "y"), verified=False)]
        self.assertEqual(store.add(cs, enc), 1)
        self.assertEqual(store.add(cs, enc), 0)
        got, vecs = store.load()
        self.assertEqual((len(got), vecs.shape), (1, (1, 384)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DaemonTest(_TmpStore):
    def setUp(self):
        super().setUp()
        self._port, self._key = C.PORT, C.KEY
        C.PORT, C.KEY = _free_port(), Path(self.tmp) / "k"
        self._ret, self._warm = api.retrieve, api.warm
        api.retrieve = lambda text, **kw: [(0.9, {"id": "card:z", "task": "z", "text": "Method: stub for " + text})]
        api.warm = lambda: None
        os.environ["NEURAL_MESH_MEMORY_DAEMON"] = "off"

    def tearDown(self):
        api.retrieve, api.warm = self._ret, self._warm
        C.PORT, C.KEY = self._port, self._key
        os.environ.pop("NEURAL_MESH_MEMORY_DAEMON", None)
        super().tearDown()

    def test_fallback_in_process_when_daemon_down(self):
        out = client.lookup("some problem text", timeout=0.3)
        self.assertIn("stub for some problem text", out)
        self.assertEqual(client.lookup("x", timeout=0.3, in_process=False), "")

    def test_daemon_answers_and_requires_key(self):
        th = threading.Thread(target=daemon.serve, kwargs={"port": C.PORT, "idle_s": 2.0}, daemon=True)
        th.start()
        for _ in range(50):
            if client._request({"op": "ping"}, 0.2):
                break
            time.sleep(0.05)
        api.retrieve = lambda text, **kw: [(0.9, {"id": "card:d", "task": "d", "text": "Method: from daemon"})]
        self.assertIn("from daemon", client.lookup("q", timeout=1.0))
        self.assertEqual(daemon.handle({"op": "ping", "key": "wrong"}, daemon.key())["error"], "auth")
        th.join(10)                                    # idle exit
        self.assertFalse(th.is_alive())


class HookBudgetTest(unittest.TestCase):
    def test_hook_never_blocks_past_budget_and_exits_zero(self):
        env = dict(os.environ, NEURAL_MESH_ASSIST_BUDGET_S="0.3", NEURAL_MESH_MEMORY_DAEMON="off",
                   NEURAL_MESH_MEMORY_PORT=str(_free_port()), NEURAL_MESH_PACKS=os.path.join(tempfile.gettempdir(), "no-packs.json"))
        p = {"tool_name": "Agent", "tool_input": {"subagent_type": "x", "description": "d",
                                                  "prompt": "Find all integers n such that n^2 + 1 is divisible by n + 1, with proof."}}
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "hooks", "mesh_assist.py")], input=json.dumps(p),
                           capture_output=True, text=True, env=env, timeout=30)
        self.assertEqual(r.returncode, 0)
        self.assertLess(time.monotonic() - t0, 3.0)
        if r.stdout:
            self.assertIn("updatedInput", json.loads(r.stdout)["hookSpecificOutput"])


if __name__ == "__main__":
    unittest.main()