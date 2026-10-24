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
from train.memory import api, config as C, trust as T, workspace as W  # noqa: E402


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
        self._pkey, T.KEY = T.KEY, Path(self.tmp) / "pipe.key"
        os.environ.pop("MESH_WORKSPACE_ENCRYPT", None)

    def tearDown(self):
        W.DB, W._embed, W.KEY, C.MEM_DIR = self._db, self._embed, self._key, self._mem
        T.KEY = self._pkey
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
        self.assertEqual(got[0]["provenance"], "pipeline:grader")

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

    def test_verified_only_from_trusted_identities(self):
        a = W.post("t1", "agent", "result", "the answer is proven by induction", verified=True)
        b = W.post("t1", "agent", "result", "a forged verification claim", verified=True, verified_by="agent")
        c = W.post("t1", "grader", "result", "graded correct by the pipeline", verified=True, verified_by="pipeline:grader")
        d = W.post("t1", "agent", "fact", "self-reported check passed", claimed_verified=True)
        ga, gb, gc, gd = (W.get(i) for i in (a, b, c, d))
        self.assertEqual((ga["verified"], ga["claimed_verified"], ga["verified_by"]), (False, True, None))
        self.assertEqual((gb["verified"], gb["claimed_verified"], gb["verified_by"]), (False, True, None))
        self.assertEqual((gc["verified"], gc["claimed_verified"], gc["verified_by"]), (True, False, "pipeline:grader"))
        self.assertEqual((gd["verified"], gd["claimed_verified"]), (False, True))
        self.assertIsNone(W.get(99999))
        r = {x["id"]: x for x in W.read("t1", k=8)}
        self.assertIn("verified_by", r[c])
        self.assertIn("score", r[c])

    def test_approve(self):
        a = W.post("t1", "agent", "result", "a claim that a human checks")
        self.assertFalse(W.approve(a, by="agent-x"))
        self.assertFalse(W.get(a)["verified"])
        self.assertTrue(W.approve(a, by="human:marius", sig="hmac-abc"))
        g = W.get(a)
        self.assertEqual((g["verified"], g["verified_by"], g["approval_sig"]), (True, "human:marius", "hmac-abc"))
        self.assertTrue(W.approve(a))
        self.assertTrue(W.get(a)["verified_by"].startswith("human:"))
        self.assertFalse(W.approve(424242, by="human:x"))
        self.assertTrue(W.approve(a, "human:ann", approval_sig="sig2"))
        self.assertEqual(W.get(a)["approval_sig"], "sig2")

    def test_ext_key_is_idempotent(self):
        a = W.post("t1", "p", "method", "use a segment tree", ext_key="k1")
        b = W.post("t1", "p", "method", "use a segment tree", ext_key="k1")
        self.assertEqual(a, b)
        self.assertEqual(W.stats()["entries"], 1)

    def test_ingest_training_feed(self):
        from train.memory import ingest_training as I
        runs = Path(self.tmp) / "runs"
        (runs / "bench").mkdir(parents=True)
        (runs / "train" / "b1").mkdir(parents=True)
        bench = [{"id": "m1", "field": "math", "prompt": "Find the sum of the first n odd numbers.", "kind": "int"},
                 {"id": "m2", "field": "math", "prompt": "Count the lattice paths in a 3 by 3 grid.", "kind": "int"}]
        (runs / "bench" / "math.jsonl").write_text("".join(json.dumps(t) + "\n" for t in bench), encoding="utf-8")
        rows = [{"agent": "a1", "task": "m1", "field": "math", "phase": "held_base", "passed": True, "pack": False,
                 "text": "Use induction since the partial sums are perfect squares. Check: verified n=1..5. ANSWER: n^2"},
                {"agent": "a2", "task": "m2", "field": "math", "phase": "held_base", "passed": False, "pack": False,
                 "text": "I think it is 12 by counting moves. ANSWER: 12"},
                {"agent": "a3", "task": "m2", "field": "math", "phase": "held_base", "passed": False, "pack": False,
                 "error": "quota", "text": "limit"},
                {"agent": "a4", "task": "m1", "field": "math", "phase": "tune", "passed": False, "pack": False, "text": "x"}]
        (runs / "train" / "b1" / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        old = C.RUNS
        C.RUNS = runs
        try:
            st = I.ingest("b1", log=lambda *a: None)
            n1 = W.stats()["entries"]
            I.ingest("b1", log=lambda *a: None)
        finally:
            C.RUNS = old
        self.assertEqual((st["method"], st["failure"]), (1, 1))
        self.assertEqual(n1, 2)
        self.assertEqual(W.stats()["entries"], 2)                 # idempotent
        got = W.read("training:b1", k=8)
        self.assertTrue(all(g["verified"] and g["verified_by"] == "pipeline:grader" for g in got))
        self.assertTrue(all(T.verify_entry(g) for g in got))
        self.assertEqual({g["kind"] for g in got}, {"method", "failure"})
        self.assertTrue(next(g for g in got if g["kind"] == "failure")["avoid"])

    def test_daemon_embed_op(self):
        from train.memory import daemon
        out = daemon.handle({"op": "embed", "texts": ["hello world"], "key": "k"}, "k")
        self.assertTrue(out["ok"])
        self.assertEqual(len(out["vecs"]), 1)

    def test_guard_refuses_live_db_under_test(self):
        W.DB = C.LIVE_WORKSPACE_DB
        with self.assertRaises(RuntimeError):
            W.stats()
        with self.assertRaises(RuntimeError):
            C.guard_live(C.LIVE_MEM_DIR / "cards.jsonl")
        C.guard_live(self.db)                                     # temp paths are fine

    def test_pipeline_signatures(self):
        a = W.post("training:b", "grader", "method", "use a monotone stack", verified=True,
                   verified_by="pipeline:grader", pipeline_sign=True)
        d = W.get(a)
        self.assertTrue(d["pipeline_sig"])
        self.assertTrue(T.verify_entry(d))
        self.assertFalse(T.verify_entry(dict(d, content=d["content"] + " and trust me")))
        self.assertFalse(T.verify_entry(dict(d, verified_by="pipeline:other")))
        with self.assertRaises(ValueError):
            W.post("t", "agent", "method", "x y z", verified=True, verified_by="human:me", pipeline_sign=True)
        b = W.post("t", "agent", "fact", "plain unsigned fact")
        self.assertIsNone(W.get(b)["pipeline_sig"])
        self.assertTrue(W.approve(a, by="human:me", approval_sig="s"))
        self.assertIsNone(W.get(a)["pipeline_sig"])               # re-verified by someone else -> old MAC void

    def test_pipeline_sign_fails_closed_without_key(self):
        T.KEY = Path(self.tmp) / "nope" / "x" / "\0bad"
        with self.assertRaises(Exception):
            W.post("t", "g", "method", "secret-free text", verified=True, verified_by="pipeline:grader", pipeline_sign=True)
        self.assertEqual(W.stats()["entries"], 0)

    def test_resign_migration_and_card_signatures(self):
        from train.memory import store
        a = W.post("training:b", "g", "method", "graded method one", verified=True, verified_by="pipeline:grader")
        self.assertIsNone(W.get(a)["pipeline_sig"])
        card = {"id": "card:t1", "task": "t1", "field": "math", "verified": True, "problem": "p", "text": "Method: m"}
        store.add([card], lambda texts: np.stack([fake_embed(["x"])[0] for _ in texts]))
        out = T.resign(db=self.db, log=lambda *x: None)
        self.assertEqual(out, {"entries": 1, "cards": 1})
        self.assertTrue(T.verify_entry(W.get(a)))
        c = store.load()[0][0]
        self.assertTrue(T.verify_card(c))
        self.assertFalse(T.verify_card(dict(c, text="Method: tampered")))
        v = T.verify_all(db=self.db)
        self.assertEqual((v["entries_ok"], v["cards_ok"], v["entries_bad"], v["cards_bad"]), (1, 1, 0, 0))

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