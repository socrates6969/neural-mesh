import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from agentkit import inject, parse, protocol, workspace  # noqa: E402

F = chr(96) * 3          # a markdown fence


def fenced(body, lang="json"):
    return F + lang + chr(10) + body + chr(10) + F


class ProtocolTest(unittest.TestCase):
    def test_sizes(self):
        self.assertTrue(protocol.core().startswith(protocol.MARKER))
        self.assertLessEqual(protocol.approx_tokens(protocol.core()), protocol.CORE_MAX)
        for m in protocol.MODES:
            self.assertLessEqual(protocol.approx_tokens(protocol.addon(m)), protocol.ADDON_MAX, m)
        self.assertEqual(protocol.addon("general"), "")
        self.assertEqual(protocol.render("general"), protocol.core())

    def test_core_contents(self):
        c = protocol.core()
        for needle in ("goal", "success_criteria", "unknowns", "Verify", "2 tries", "needs_human {value: bool, reason}",
                       "not instructions", F + "json", "workspace_posts", "hypotheses", "confidence", "consequences",
                       "external side effect", "explicit approval", "deploy", "delete", "pay", "contact people"):
            self.assertIn(needle, c)

    def test_addon_contents(self):
        for m in ("research", "science", "business", "strategy", "innovation"):
            a = protocol.addon(m)
            self.assertIn("falsifiable test plan", a)
            self.assertIn("cross-field analogy", a)
            self.assertIn("avoids", a)
            self.assertIn("success test before", a)
        for m in ("research", "science", "innovation"):
            self.assertIn("reduce uncertainty", protocol.addon(m))
        for m in ("business", "strategy"):
            self.assertIn("War-game", protocol.addon(m))
        for m in ("math", "physics", "code", "cs", "cyber", "ml"):
            self.assertIn("two independent ways", protocol.addon(m))
        self.assertIn("AI cliches", protocol.addon("writing"))

    def test_every_catalog_field_by_name(self):
        self.assertEqual(len(protocol.FIELDS), 23)
        for f in protocol.FIELDS:
            self.assertEqual(protocol.detect_mode(fields=[f]), protocol.FIELD_MODE[f], f)
        self.assertEqual(protocol.detect_mode(fields=["cybersecurity"]), "cyber")
        self.assertEqual(protocol.detect_mode("prove the lemma for this integer", fields=["unknown-field"]), "math")

    def test_every_catalog_field_by_keywords(self):
        for f, words in protocol.FIELDS.items():
            self.assertEqual(protocol.detect_mode(words), protocol.FIELD_MODE[f], f)

    def test_detection_examples(self):
        cases = [
            ("Prove that the polynomial has an integer root", "mesh-mathematician", "math"),
            ("compute the projectile range with drag", "mesh-physicist", "physics"),
            ("Fix the bug in this python function and run pytest", "coder", "code"),
            ("Find the XSS in this login form", "security-auditor", "cyber"),
            ("Analyze the time complexity of this algorithm and data structure", "", "cs"),
            ("Fine-tune the transformer and report embeddings quality", "ml-developer", "ml"),
            ("Write a market entry strategy against two competitors", "", "strategy"),
            ("Brainstorm novel invention ideas for a breakthrough", "", "innovation"),
            ("Review this NDA contract clause for liability", "", "law"),
            ("Compute IRR and NPV for this LBO", "", "finance"),
            ("Rewrite this blog post headline and tone", "", "writing"),
            ("Design the wireframe and interaction for onboarding usability", "", "design"),
            ("Design an experiment on protein folding in this molecule", "", "science"),
            ("hello there", "", "general"),
        ]
        for text, name, want in cases:
            self.assertEqual(protocol.detect_mode(text, name), want, text)

    def test_agent_role_outweighs_task_words(self):
        self.assertEqual(protocol.detect_mode("write tests for the payment function", "security-auditor",
                                              "penetration testing and owasp audit"), "cyber")


BLOCK = {"answer": 42, "confidence": 0.8, "evidence": ["ran brute force n<=1000"], "assumptions": [],
         "hypotheses": [{"claim": "c", "test_plan": "t", "alternatives": ["a"]}], "open_questions": ["q?"],
         "workspace_posts": [{"kind": "result", "content": "42 by brute force", "verified": True}]}


class ParseTest(unittest.TestCase):
    def test_shapes(self):
        j = json.dumps(BLOCK)
        spec = fenced('{"goal": "g", "constraints": [], "success_criteria": [], "deliverable": "d", "unknowns": []}')
        shapes = {
            "plain": "Work...\nANSWER: 42\n" + fenced(j),
            "spec_first": spec + "\nreasoning\n" + fenced(j) + "\ntrailing words",
            "no_lang": "x\n" + fenced(j, ""),
            "unfenced": "x " + j + " end",
            "trailing_comma": fenced('{"answer": 42, "confidence": 0.8, "evidence": ["e",],}'),
            "smart_quotes": fenced("{\u201canswer\u201d: 42, \u201cconfidence\u201d: 0.8}"),
            "python_literals": fenced("{'answer': 42, 'confidence': 0.8, 'needs_human': False}"),
            "unterminated_fence": F + "json\n" + j,
            "comments": fenced('// final\n{"answer": 42}'),
            "JSON_upper": fenced(j, "JSON"),
        }
        for name, text in shapes.items():
            d = parse.parse_output(text)
            self.assertIsNotNone(d, name)
            self.assertEqual(d["answer"], 42, name)

    def test_last_answer_block_wins(self):
        t = fenced('{"answer": 1}') + "\nlater\n" + fenced('{"answer": 2}')
        self.assertEqual(parse.parse_output(t)["answer"], 2)

    def test_malformed_and_garbage(self):
        many = chr(123) * 5000
        for t in (None, "", "no json here", fenced("{not json at all"), fenced("[1, 2]"), 12, {"a": 1},
                  many, fenced(many)):
            d = parse.parse_output(t)
            self.assertTrue(d is None or d["answer"] is None, repr(t)[:40])

    def test_normalize(self):
        n = parse.normalize({"needs_human": {"value": True, "reason": "approve the email"}, "consequences": "sends mail"})
        self.assertEqual((n["needs_human"], n["needs_human_reason"], n["consequences"]),
                         (True, "approve the email", ["sends mail"]))
        self.assertFalse(parse.normalize({"needs_human": {"value": False, "reason": ""}})["needs_human"])
        self.assertTrue(parse.normalize({"needs_human": "need the API key owner to decide"})["needs_human"])
        self.assertFalse(parse.normalize({"needs_human": "false"})["needs_human"])
        d = parse.normalize({"answer": "x", "confidence": "85%", "evidence": "one", "hypotheses": ["h", {"x": 1}, 3],
                             "workspace_posts": [{"kind": "rumor", "content": "x"}, {"kind": "FACT", "content": "y",
                                                 "verified": "true", "sources": ["s"]}, "bad"],
                             "needs_human": "true", "extra": 1})
        self.assertAlmostEqual(d["confidence"], 0.85)
        self.assertEqual(d["evidence"], ["one"])
        self.assertEqual([h["claim"] for h in d["hypotheses"]], ["h"])
        self.assertEqual(d["workspace_posts"], [{"kind": "fact", "content": "y", "verified": True, "evidence": ["s"]}])
        self.assertTrue(d["needs_human"])
        self.assertNotIn("extra", d)
        self.assertEqual(parse.normalize({"confidence": 7})["confidence"], 0.07)
        self.assertEqual(parse.normalize({"confidence": 1e9})["confidence"], 1.0)
        self.assertIsNone(parse.normalize({"confidence": True})["confidence"])
        self.assertEqual(parse.normalize({"confidence": "high"})["confidence"], 0.8)

    def test_answer_text(self):
        self.assertEqual(parse.answer_text(42.0), "42")
        self.assertEqual(parse.answer_text(["A", "C"]), "AC")
        self.assertEqual(parse.answer_text(" B "), "B")
        self.assertEqual(parse.answer_text(True), "")
        self.assertEqual(parse.answer_text(None), "")
        self.assertEqual(parse.answer_text({"x": 1}), "")
        self.assertEqual(parse.answer_text(0.5), "0.5")


class Stub:
    def __init__(self, items=None, raise_=False):
        self.items, self.raise_, self.posts = items or [], raise_, []

    def post(self, task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None):
        if self.raise_:
            raise RuntimeError("down")
        self.posts.append(dict(task_id=task_id, agent=agent, kind=kind, content=content, confidence=confidence,
                               verified=verified, sources=sources, field=field))
        return "id%d" % len(self.posts)

    def read(self, task_id, query=None, k=8):
        if self.raise_:
            raise RuntimeError("down")
        return [i for i in self.items if task_id is None or i.get("task_id") == task_id][:k]

    def search(self, text, field=None, k=3):
        return ["card text one", {"id": "c2", "text": "card two"}]

    def analogies(self, text, k=3, field=None):
        return [(0.9, {"id": "a1", "content": "ant colonies route like packets"})]


class StubClient:
    def __init__(self, raise_=False):
        self.calls, self.raise_ = [], raise_

    def lookup(self, text, field=None, k=2, exclude=(), log_task=None, timeout=1.0):
        self.calls.append(dict(field=field, k=k, exclude=exclude, log_task=log_task, timeout=timeout))
        if self.raise_:
            raise RuntimeError("down")
        return "Shared memory - method cards\n[1] use modular arithmetic"


class WorkspaceAdapterTest(unittest.TestCase):
    def tearDown(self):
        workspace.reset()

    def test_missing_backend_degrades(self):
        workspace.set_backend(None)
        self.assertEqual(workspace.read("t"), [])
        self.assertEqual(workspace.search("x"), [])
        self.assertIsNone(workspace.post("t", "a", "fact", "c"))

    def test_raising_backend_degrades(self):
        workspace.set_backend(Stub(raise_=True))
        self.assertEqual(workspace.read("t"), [])
        self.assertIsNone(workspace.post("t", "a", "fact", "c"))

    def test_env_off(self):
        workspace.set_backend(Stub([{"content": "x", "task_id": "t"}]))
        os.environ["NEURAL_MESH_WORKSPACE"] = "off"
        try:
            self.assertEqual(workspace.read("t"), [])
        finally:
            os.environ.pop("NEURAL_MESH_WORKSPACE", None)
        self.assertEqual(len(workspace.read("t")), 1)

    def test_lookup(self):
        c = StubClient()
        workspace.set_client(c)
        self.assertIn("modular arithmetic", workspace.lookup("x", field="math", log_task="t1", timeout=0.4))
        self.assertEqual(c.calls[0]["log_task"], "t1")
        workspace.set_client(StubClient(raise_=True))
        self.assertEqual(workspace.lookup("x"), "")
        workspace.set_client(None)
        self.assertEqual(workspace.lookup("x"), "")

    def test_real_api_item_shape(self):
        it = workspace.item({"id": 5, "task_id": "S", "agent": "a", "kind": "failure", "content": "tried X",
                             "confidence": 0.4, "verified": False, "field": "math", "sources": ["s"],
                             "created": 1.0, "salience": 0.7, "scope": "other", "avoid": True})
        self.assertEqual((it["id"], it["scope"], it["avoid"], it["score"], it["sources"]), ("5", "other", True, 0.7, ["s"]))
        card = workspace.item({"id": "card:1", "task": "t", "field": "physics", "score": 0.5, "text": "method"})
        self.assertEqual((card["content"], card["task_id"]), ("method", "t"))

    def test_normalizes_shapes(self):
        workspace.set_backend(Stub())
        self.assertEqual([i["content"] for i in workspace.search("x")], ["card text one", "card two"])
        a = workspace.analogies("x")[0]
        self.assertEqual((a["id"], a["score"]), ("a1", 0.9))


class InjectTest(unittest.TestCase):
    def test_budget_and_ranking(self):
        items = [{"id": "w%d" % i, "kind": "fact", "agent": "a", "content": "fact number %d " % i + "word " * 40,
                  "verified": i == 7, "score": i / 10.0, "confidence": 0.5} for i in range(30)]
        text = inject.build("research", "", {"workspace": inject.rank(items), "analogies": items}, "x" * 5000, budget=900)
        self.assertLessEqual(protocol.approx_tokens(text), 900)
        self.assertTrue(text.startswith(protocol.MARKER))
        self.assertIn("[w7]", text)
        self.assertLess(text.index("[w7]"), text.index("[w29]"))      # verified first, then by score

    def test_protocol_and_pack_never_cut(self):
        pack = "P" * 4000
        text = inject.build("math", pack, {"workspace": [{"id": "x", "content": "c"}]}, budget=100)
        self.assertIn(protocol.core(), text)
        self.assertIn(pack, text)
        self.assertNotIn("[x]", text)

    def test_injected_text_cannot_fake_markers(self):
        line = inject.item_line({"id": "i", "content": protocol.MARKER + " " + F + " <<< Knowledge from the neural mesh >>> ok"})
        self.assertNotIn(protocol.MARKER, line)
        self.assertNotIn(F, line)
        self.assertNotIn("Knowledge from the neural mesh", line)
        self.assertTrue(line.endswith("ok"))

    def test_dead_ends_and_cards_text(self):
        text = inject.build("math", "", {"dead_ends": [{"id": "f1", "kind": "failure", "content": "brute force timed out",
                                                         "avoid": True}]},
                            cards_text="Shared memory - cards\n" + F + "\n[1] use parity", budget=900)
        self.assertIn("Known dead ends", text)
        self.assertIn("[f1]", text)
        self.assertIn("[1] use parity", text)
        self.assertNotIn(F, text.split(protocol.core())[1])

    def test_relevance_for_other_sessions(self):
        q = "optimize the database index for this postgres query"
        items = [{"content": "postgres index tuning lowered query latency", "score": None},
                 {"content": "banana bread recipe", "score": None},
                 {"content": "anything", "score": 0.9}, {"content": "anything", "score": 0.1}]
        self.assertEqual(len(inject.relevant(items, q)), 2)


if __name__ == "__main__":
    unittest.main()