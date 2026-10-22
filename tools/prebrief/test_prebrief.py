"""Unit tests: python -m unittest prebrief.test_prebrief (run from tools/)."""
import os
import unittest

import numpy as np

from prebrief.evaluate import mcnemar_exact, mde_sign, ranks
from prebrief.index import Index, anchors, relevant, tokens
from prebrief.scrub import scrub

os.environ["PREBRIEF_LOG"] = "off"


def _doc(i, ts, text, session="s1", agent="general-purpose", kind="run"):
    return {"id": f"d{i}", "kind": kind, "ts": ts, "session": session, "agent": agent, "title": text[:30], "text": text,
            "report": text, "prompt": text}


class TestPrebrief(unittest.TestCase):
    def test_scrub(self):
        self.assertNotIn("sk-abcdefghijklmnopqrstu", scrub("key sk-abcdefghijklmnopqrstu end"))
        self.assertIn("[EMAIL]", scrub("mail someone@example.com now"))
        self.assertIn("password=[REDACTED]", scrub("password: hunter2secret"))
        self.assertEqual(scrub("plain text 123"), "plain text 123")

    def test_tokens_and_anchors(self):
        t = tokens("late_agent_init in routerState and 8f3352b3abcdef01 the")
        self.assertIn("late_agent_init", t)
        self.assertIn("agent", t)
        self.assertIn("routerstate", t)
        self.assertNotIn("the", t)
        self.assertNotIn("8f3352b3abcdef01", t)
        a = anchors("edit src/hebbian.rs at 7a05bba, set `late_agent_init`, call evaluateCandidate")
        self.assertTrue({"p:hebbian.rs", "h:7a05bba", "t:late_agent_init", "i:evaluatecandidate"} <= a)

    def test_prequential_and_relevance(self):
        docs = [_doc(0, "2026-10-01", "router_eval.py replay of hebbian_router bug"),
                _doc(1, "2026-10-02", "unrelated tariff parser work", session="s2"),
                _doc(2, "2026-10-03", "router_eval.py hebbian_router fix again", session="s3")]
        idx = Index(docs)
        av = idx.available("2026-10-03", exclude_session="s3")
        self.assertEqual(av.tolist(), [True, True, False])
        rel = relevant(idx, "touched router_eval.py and hebbian_router", av)
        self.assertEqual(rel, {0})
        s = idx.search("hebbian router replay", None, av, method="B")
        self.assertEqual(int(np.argmax(s)), 0)
        self.assertTrue(np.isneginf(s[2]))
        self.assertEqual(ranks(s, rel), 1)

    def test_stats(self):
        self.assertAlmostEqual(mcnemar_exact(7, 1), 0.0703125)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)
        m10, m40 = mde_sign(10), mde_sign(40)
        self.assertIsNotNone(m40)
        self.assertTrue(m10 is None or m10 > m40)

    def test_brief_budget(self):
        from prebrief.brief import prebrief
        docs = [_doc(i, f"2026-10-0{1 + i % 9}", ("replay of router_eval.py found the hebbian late agent bug failed badly. " * 20) + str(i),
                     session=f"s{i}") for i in range(30)]
        idx = Index(docs)
        r = prebrief("hebbian late agent bug replay", "general-purpose", idx=idx, max_tokens=150)
        self.assertLessEqual(r["tokens"], 150)
        self.assertIn("Prior solutions", r["markdown"])


class TestPlantedSecrets(unittest.TestCase):
    """Fake secrets planted across every truncation boundary must not survive in any corpus field (scrub-before-truncate)."""

    FAKE = "sk-PLANTEDfakeKEY0123456789abcdefXYZ"   # synthetic, matches the sk- rule only when whole
    FRAG = "PLANTEDfakeKEY"

    def _straddle(self, cut: int, filler: str = "x ") -> str:
        head = (filler * cut)[: cut - 10]  # secret starts 10 chars before the cut, ends after it
        return head + self.FAKE + " tail text after the secret " * 20

    def test_no_fragment_survives(self):
        import json
        import tempfile
        from prebrief import corpus
        with tempfile.TemporaryDirectory() as root:
            sub = os.path.join(root, "proj", "sess1", "subagents")
            os.makedirs(sub)
            prompt = self._straddle(3000) + self._straddle(3000)          # crosses 3000 and ~6000
            report = self._straddle(4000) + " " + self._straddle(4000)     # crosses 4000 and ~8000
            lines = [
                {"type": "user", "timestamp": "2026-10-01T00:00:00Z", "message": {"content": prompt}},
                {"type": "assistant", "timestamp": "2026-10-01T00:01:00Z", "message": {"content": [{"type": "text", "text": report}], "usage": {}}},
            ]
            with open(os.path.join(sub, "agent-a1.jsonl"), "w", encoding="utf-8") as f:
                f.write("\n".join(json.dumps(x) for x in lines))
            with open(os.path.join(sub, "agent-a1.meta.json"), "w", encoding="utf-8") as f:
                json.dump({"agentType": "general-purpose", "description": ("d" * 95) + self.FAKE}, f)
            hive = os.path.join(root, "hive")
            os.makedirs(os.path.join(hive, "findings"))
            with open(os.path.join(hive, "findings", "f.md"), "w", encoding="utf-8") as f:
                f.write("# head\n" + self._straddle(5000))
            with open(os.path.join(hive, "journal.jsonl"), "w", encoding="utf-8") as f:
                f.write(json.dumps({"ts": "2026-10-01T00:00:00", "kind": "finding", "text": ("j" * 95) + " " + self.FAKE, "ref": self.FAKE}) + "\n")
            docs = corpus.runs(root) + corpus.findings(hive) + corpus.journal(hive)
            self.assertEqual(len(docs), 3)
            blob = json.dumps(docs)
            self.assertNotIn(self.FRAG, blob)
            self.assertNotIn(self.FRAG.lower(), blob.lower())
            self.assertIn("[KEY]", blob)


if __name__ == "__main__":
    unittest.main()
