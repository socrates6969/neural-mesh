"""final_answer compatibility: every bench gold answer must still grade correct in old and new output shapes."""
import json
import os
import random
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from train import check  # noqa: E402

BENCH = Path(os.environ.get("MESHTRAIN_BENCH_DIR", r"C:\Users\mariu\ruv\hive\runs\agent-catalog\bench"))
F = chr(96) * 3
CHECKERS = {"int": check.check_int, "numeric": check.check_numeric, "mcq": check.check_mcq}
if hasattr(check, "check_cwe"):
    CHECKERS["cwe"] = check.check_cwe


def block(answer, **extra):
    d = {"answer": answer, "confidence": 0.7, "evidence": ["ran it"], "assumptions": [], "hypotheses": [],
         "open_questions": [], "workspace_posts": []}
    d.update(extra)
    return F + "json\n" + json.dumps(d) + "\n" + F


def shapes(gold):
    spec = F + "json\n" + json.dumps({"goal": "g", "constraints": [], "success_criteria": ["ANSWER: <integer>"],
                                     "deliverable": "ANSWER: <integer>", "unknowns": []}) + "\n" + F
    return {
        "legacy": "reasoning...\nANSWER: " + gold,
        "legacy_boxed": "so \\boxed{" + gold + "}",
        "answer_then_json": spec + "\nwork\nANSWER: " + gold + "\n" + block(gold),
        "json_only": spec + "\nwork\n" + block(gold),
        "json_wrong_answer_line_wins": "ANSWER: " + gold + "\n" + block("WRONG-0"),
        "placeholder_in_json_strings": "ANSWER: " + gold + "\n" + block(gold, assumptions=["ANSWER: <integer>"]),
    }


def load(name):
    p = BENCH / name
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


@unittest.skipUnless(BENCH.exists(), "bench dir not present")
class FinalAnswerCompat(unittest.TestCase):
    def test_all_gold_answers_grade_correct(self):
        n = 0
        for p in sorted(BENCH.glob("*.jsonl")):
            for t in load(p.name):
                chk = CHECKERS.get(t.get("kind"))
                if not chk:
                    continue
                gold = str(t["answer"])
                for sname, text in shapes(gold).items():
                    if sname == "legacy_boxed" and ("{" in gold or "}" in gold):
                        continue
                    self.assertTrue(chk(check.final_answer(text), t["answer"]), (p.name, t["id"], sname, gold))
                n += 1
        self.assertGreater(n, 1000)

    def test_json_number_and_letter_lists(self):
        self.assertTrue(check.check_int(check.final_answer(block(-1)), "-1"))
        self.assertTrue(check.check_numeric(check.final_answer(block(2.5)), "2.50"))
        self.assertTrue(check.check_mcq(check.final_answer(block(["A", "C"])), "AC"))

    def test_wrong_json_answer_still_fails(self):
        self.assertFalse(check.check_int(check.final_answer(block(7)), "8"))
        self.assertEqual(check.final_answer("no answer at all"), "")


@unittest.skipUnless(BENCH.exists(), "bench dir not present")
class CodeExtractCompat(unittest.TestCase):
    def test_code_block_still_extracted_next_to_json(self):
        rng = random.Random(0)
        for name in ("computer-science.jsonl", "coding_v1_saturated.jsonl", "coding.jsonl"):
            rows = [t for t in load(name) if t.get("kind") in ("code", "stdio") and t.get("answer")]
            for t in rng.sample(rows, min(40, len(rows))):
                code = t["answer"]
                text = "plan\n" + F + "python\n" + code + "\n" + F + "\n" + block("see python block")
                self.assertEqual(check.extract_code(text).strip(), code.strip(), (name, t["id"]))


if __name__ == "__main__":
    unittest.main()