import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from hooks import mesh_capture as mc  # noqa: E402

F = chr(96) * 3


def fenced(obj):
    return F + "json\n" + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n" + F


class StubWS:
    def __init__(self, raise_=False):
        self.posts, self.raise_ = [], raise_

    def post(self, task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None):
        if self.raise_:
            raise RuntimeError("down")
        self.posts.append(dict(task_id=task_id, agent=agent, kind=kind, content=content, confidence=confidence,
                               verified=verified, sources=sources))
        return "id%d" % len(self.posts)


def payload(text, agent="mesh-mathematician", sid="S1", tool="Agent", shape="content"):
    if shape == "content":
        resp = {"content": [{"type": "text", "text": text}], "totalTokens": 10}
    elif shape == "str":
        resp = text
    else:
        resp = {"result": text}
    d = {"tool_name": tool, "tool_input": {"subagent_type": agent, "prompt": "p"}, "tool_response": resp}
    if sid:
        d["session_id"] = sid
    return d


FULL = {"answer": 42, "confidence": 0.9, "evidence": ["brute force over n<=10^4 in python"],
        "hypotheses": [{"claim": "f is periodic", "test_plan": "check 10^5 terms", "alternatives": ["eventually periodic"]}],
        "open_questions": ["does it hold for n<0?"],
        "workspace_posts": [{"kind": "method", "content": "reduce mod 7 first", "verified": True},
                            {"kind": "fact", "content": "f(1)=1", "verified": True, "evidence": ["computed"]},
                            {"kind": "rumor", "content": "ignored"}]}


class CaptureTest(unittest.TestCase):
    def test_full_block(self):
        ws = StubWS()
        ids = mc.capture(payload("work\nANSWER: 42\n" + fenced(FULL)), ws)
        kinds = [p["kind"] for p in ws.posts]
        self.assertEqual(kinds, ["result", "method", "fact", "hypothesis", "question"])
        self.assertEqual(len(ids), 5)
        self.assertTrue(all(p["task_id"] == "S1" and p["agent"] == "mesh-mathematician" for p in ws.posts))
        by = {p["kind"]: p for p in ws.posts}
        self.assertFalse(by["result"]["verified"])           # the answer itself was not marked verified
        self.assertTrue(by["method"]["verified"])            # marked verified + block-level evidence
        self.assertTrue(by["fact"]["verified"])              # marked verified + own evidence
        self.assertFalse(by["hypothesis"]["verified"])
        self.assertIn("test: check 10^5 terms", by["hypothesis"]["content"])
        self.assertEqual(by["result"]["content"], "42")
        self.assertAlmostEqual(by["result"]["confidence"], 0.9)

    def test_verified_requires_evidence(self):
        ws = StubWS()
        d = {"answer": "x", "workspace_posts": [{"kind": "result", "content": "claimed", "verified": True}]}
        mc.capture(payload(fenced(d)), ws)
        self.assertEqual([p["verified"] for p in ws.posts], [False, False])

    def test_needs_human_posts_failure(self):
        ws = StubWS()
        mc.capture(payload(fenced({"answer": None, "needs_human": True, "failures": ["tried A", "tried B"]})), ws)
        kinds = [p["kind"] for p in ws.posts]
        self.assertEqual(kinds.count("failure"), 3)
        self.assertTrue(any("needs_human" in p["content"] for p in ws.posts))

    def test_consequence_check_becomes_approval_question(self):
        ws = StubWS()
        d = {"answer": "draft ready, not sent", "consequences": ["emails 300 customers", "cannot be recalled"],
             "needs_human": {"value": True, "reason": "approve sending the email"}}
        mc.capture(payload(fenced(d)), ws)
        q = [p for p in ws.posts if p["kind"] == "question"]
        self.assertEqual(len(q), 1)
        self.assertIn("approve sending the email", q[0]["content"])
        self.assertIn("emails 300 customers", q[0]["content"])

    def test_no_json_fallback(self):
        ws = StubWS()
        text = "plain answer " * 100
        mc.capture(payload(text, shape="str"), ws)
        self.assertEqual(len(ws.posts), 1)
        p = ws.posts[0]
        self.assertEqual(p["kind"], "result")
        self.assertLessEqual(len(p["content"]), mc.FALLBACK_CHARS)
        self.assertLess(p["confidence"], 0.5)
        self.assertFalse(p["verified"])

    def test_response_shapes(self):
        for shape in ("content", "str", "result"):
            ws = StubWS()
            mc.capture(payload(fenced({"answer": 7}), shape=shape), ws)
            self.assertEqual(ws.posts[0]["content"], "7", shape)

    def test_secrets_are_dropped(self):
        ws = StubWS()
        key = "sk-ant-" + "a1B2c3D4e5F6g7H8i9J0kLmN"
        d = {"answer": "the key is " + key, "workspace_posts": [{"kind": "fact", "content": "safe fact"}]}
        mc.capture(payload(fenced(d)), ws)
        self.assertEqual([p["content"] for p in ws.posts], ["safe fact"])
        self.assertFalse(any(key in json.dumps(p) for p in ws.posts))

    def test_pii_is_redacted(self):
        ws = StubWS()
        mc.capture(payload(fenced({"answer": "mail jane.doe@example.com for access"})), ws)
        self.assertNotIn("jane.doe@example.com", ws.posts[0]["content"])

    def test_ignored_and_failing(self):
        ws = StubWS()
        self.assertEqual(mc.capture(payload("x", tool="Bash"), ws), [])
        self.assertEqual(mc.capture(payload("x", sid=None), ws), [])
        self.assertEqual(mc.capture({"tool_name": "Agent", "session_id": "S"}, ws), [])
        self.assertEqual(ws.posts, [])
        self.assertEqual(mc.capture(payload(fenced(FULL)), StubWS(raise_=True)), [])
        self.assertEqual(mc.capture(None, ws), [])

    def test_post_cap(self):
        ws = StubWS()
        d = {"answer": 1, "open_questions": ["q%d" % i for i in range(30)]}
        mc.capture(payload(fenced(d)), ws)
        self.assertEqual(len(ws.posts), mc.MAX_POSTS)

    def test_default_agent(self):
        ws = StubWS()
        p = payload(fenced({"answer": 1}))
        p["tool_input"] = {"prompt": "p"}
        mc.capture(p, ws)
        self.assertEqual(ws.posts[0]["agent"], "general-purpose")

    def _run(self, raw, env):
        t0 = time.monotonic()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "hooks", "mesh_capture.py")], input=raw,
                           capture_output=True, text=True, env=dict(os.environ, **env), timeout=60)
        return r, time.monotonic() - t0

    def test_subprocess_posts_prints_nothing_exit0(self):
        log = os.path.join(tempfile.mkdtemp(), "posts.jsonl")
        env = {"NEURAL_MESH_WORKSPACE_MODULE": "hooks.tests.slow_ws", "SLOW_WS_LOG": log}
        r, _ = self._run(json.dumps(payload(fenced(FULL))), env)
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        with open(log, encoding="utf-8") as fh:
            rows = [json.loads(x) for x in fh]
        self.assertEqual(rows[0]["kind"], "result")

    def test_subprocess_budget_and_garbage(self):
        env = {"NEURAL_MESH_WORKSPACE_MODULE": "hooks.tests.slow_ws", "SLOW_WS_POST_SLEEP": "20",
               "NEURAL_MESH_CAPTURE_BUDGET_S": "1.0"}
        r, secs = self._run(json.dumps(payload(fenced(FULL))), env)
        self.assertEqual((r.returncode, r.stdout), (0, ""))
        self.assertLess(secs, 4.0)
        for raw in ("", "{bad", "[]", "null", "\x00"):
            r, _ = self._run(raw, env)
            self.assertEqual((r.returncode, r.stdout), (0, ""), repr(raw))
        r, _ = self._run(json.dumps(payload("x")), {"NEURAL_MESH_CAPTURE": "off"})
        self.assertEqual((r.returncode, r.stdout), (0, ""))


if __name__ == "__main__":
    unittest.main()