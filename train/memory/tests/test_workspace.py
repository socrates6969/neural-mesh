import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, ROOT)
from train.memory import api, config as C, workspace as W  # noqa: E402


def fake_embed(texts):
    """Deterministic bag-of-words hashing vectors: similar words -> similar vectors, no model load."""
    out = []
    for t in texts:
        v = np.zeros(384, np.float32)
        for w in str(t).lower().split():
            w = w.strip(".,;:!?()")
            if len(w) > 2:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 384] += 1
        out.append(v / (np.linalg.norm(v) or 1))
    return np.stack(out)


class WorkspaceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Path(self.tmp) / "ws.db"
        self._db, self._embed, self._key, self._mem = W.DB, W._embed, W.KEY, C.MEM_DIR
        W.DB, W._embed, W.KEY, C.MEM_DIR = self.db, fake_embed, Path(self.tmp) / "ws.key", Path(self.tmp) / "mem"
        os.environ.pop("MESH_WORKSPACE_ENCRYPT", None)

    def tearDown(self):
        W.DB, W._embed, W.KEY, C.MEM_DIR = self._db, self._embed, self._key, self._mem
        os.environ.pop("MESH_WORKSPACE_ENCRYPT", None)

    def test_post_validates_kind_and_drops_secrets(self):
        self.assertIsInstance(W.post("t1", "a", "fact", "the integral converges by comparison"), int)
        with self.assertRaises(ValueError):
            W.post("t1", "a", "gossip", "x")
        self.assertIsNone(W.post("t1", "a", "fact", "token sk-ant-api03-" + "B" * 40))
        self.assertIsNone(W.post("t1", "a", "fact", "   "))

    def test_read_ranks_own_task_and_surfaces_relevant_other_tasks(self):
        hi = W.post("t1", "a", "result", "dynamic programming over subsets solves the tour", confidence=0.9, verified=True)
        lo = W.post("t1", "b", "hypothesis", "maybe a greedy tour works", confidence=0.2)
        far = W.post("t2", "c", "method", "dynamic programming over subsets with bitmask states", confidence=0.9)
        W.post("t3", "d", "fact", "photosynthesis converts light into chemical energy", confidence=0.9)
        got = W.read("t1", query="dynamic programming over subsets", k=8)
        ids = [g["id"] for g in got]
        self.assertEqual(ids[0], hi)
        self.assertIn(far, ids)
        self.assertEqual(next(g for g in got if g["id"] == far)["scope"], "other")
        self.assertTrue(all(g["task_id"] != "t3" for g in got))      # irrelevant other task is not broadcast
        self.assertLess(next(g for g in got if g["id"] == far)["salience"], got[0]["salience"])
        self.assertIn(lo, ids)

    def test_failures_are_kept_as_avoid_items(self):
        for i in range(6):
            W.post("t1", "a", "result", "step %d of the proof is fine" % i, confidence=0.9, verified=True)
        f = W.post("t1", "a", "failure", "induction on n alone fails for even n", confidence=0.3)
        got = W.read("t1", k=4)
        self.assertEqual(len(got), 4)
        self.assertIn(f, [g["id"] for g in got])
        self.assertTrue(next(g for g in got if g["id"] == f)["avoid"])

    def test_observe_updates_usefulness_and_routes_cards(self):
        a = W.post("t1", "a", "method", "binary search on the answer")
        b = W.post("t1", "a", "method", "binary search on the answer value")
        seen = []
        orig = api.observe
        api.observe = lambda ids, passed: seen.append((list(ids), passed))
        try:
            W.observe([a, "card:x"], True)
            W.observe([b], False)
            W.observe([b], False)
        finally:
            api.observe = orig
        self.assertEqual(seen, [(["card:x"], True)])
        got = {g["id"]: g["salience"] for g in W.read("t1", k=8)}
        self.assertGreater(got[a], got[b])

    def test_compress_folds_low_salience_entries(self):
        for i in range(10):
            W.post("t1", "a", "hypothesis", "guess number %d about the bound" % i, confidence=0.2)
        v = W.post("t1", "a", "result", "the bound is tight", confidence=0.9, verified=True)
        fl = W.post("t1", "a", "failure", "the naive bound fails", confidence=0.2)
        sid = W.compress("t1", keep=3)
        self.assertIsInstance(sid, int)
        live = W.read("t1", k=50)
        ids = [g["id"] for g in live]
        self.assertIn(sid, ids)
        self.assertIn(v, ids)
        self.assertIn(fl, ids)
        self.assertLessEqual(len(live), 3 + 1 + 2)
        summ = next(g for g in live if g["id"] == sid)
        self.assertEqual(summ["kind"], "summary")
        self.assertIn("Summary of", summ["content"])
        self.assertIsNone(W.compress("t1", keep=50))

    def test_concepts_and_cross_field_analogies(self):
        cs = W.concepts("Conservation of energy: kinetic energy plus potential energy stays constant", "physics")
        self.assertIn("energy", cs)
        self.assertIn("field:physics", cs)
        W.post("p1", "a", "method", "conserved quantity energy invariant stays constant along the motion", field="physics")
        W.post("m1", "a", "method", "an invariant conserved quantity stays constant under each move", field="math")
        W.post("m2", "a", "fact", "unrelated geometry of circles", field="math")
        got = W.analogies("find an invariant conserved quantity that stays constant under each move", k=3, field="math")
        self.assertTrue(got)
        self.assertTrue(all(g["field"] != "math" for g in got))
        self.assertEqual(got[0]["field"], "physics")
        st = W.stats()
        self.assertGreater(st["edges"], 0)
        self.assertEqual(st["entries"], 3)

    def test_search_returns_cards(self):
        orig = api.retrieve
        api.retrieve = lambda text, field=None, k=3: [(0.8, {"id": "card:q", "task": "q", "field": "math", "text": "Method: x"})]
        try:
            got = W.search("problem", field="math")
        finally:
            api.retrieve = orig
        self.assertEqual(got[0]["id"], "card:q")

    def test_encryption_roundtrip_and_fail_closed(self):
        os.environ["MESH_WORKSPACE_ENCRYPT"] = "1"
        eid = W.post("t1", "a", "fact", "secret-free but private plan text")
        import sqlite3
        raw = sqlite3.connect(str(self.db)).execute("SELECT content, enc FROM entries WHERE id = ?", (eid,)).fetchone()
        self.assertEqual(raw[1], 1)
        self.assertNotIn("private plan", raw[0])
        self.assertEqual(W.read("t1")[0]["content"], "secret-free but private plan text")
        self.assertEqual(len(W.KEY.read_bytes()), 32)
        W.KEY.unlink()
        self.assertEqual(W.read("t1"), [])                         # cannot decrypt -> never shown
        W.KEY = Path(self.tmp) / "nodir" / "x" / "\0bad"
        with self.assertRaises(Exception):
            W.post("t1", "a", "fact", "must not be stored in plaintext")
        n = sqlite3.connect(str(self.db)).execute("SELECT COUNT(*) FROM entries").fetchone()[0]
        self.assertEqual(n, 1)

    def test_two_processes_write_concurrently(self):
        code = ("import sys; sys.path.insert(0, %r); from pathlib import Path; "
                "from train.memory import workspace as W; from train.memory.tests.test_workspace import fake_embed; "
                "W.DB = Path(%r); W._embed = fake_embed; "
                "[W.post('c', sys.argv[1], 'fact', 'concurrent fact %%d from %%s' %% (i, sys.argv[1])) for i in range(40)]"
                % (ROOT, str(self.db)))
        ps = [subprocess.Popen([sys.executable, "-c", code, name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
              for name in ("p1", "p2")]
        errs = [p.communicate(timeout=120)[1] for p in ps]
        self.assertEqual([p.returncode for p in ps], [0, 0], errs)
        st = W.stats()
        self.assertEqual(st["entries"], 80)

    def test_cli(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            W.main(["post", "--task", "t9", "--kind", "plan", "--content", "first prove the lemma"])
            W.main(["read", "--task", "t9"])
            W.main(["stats"])
        self.assertIn("first prove the lemma", buf.getvalue())


if __name__ == "__main__":
    unittest.main()