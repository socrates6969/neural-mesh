"""The agents' structured JSON output contract: {"answer": ..., "confidence": 0..1, "evidence": [...], "assumptions": [...]}.

The local model answers in it only when a prompt asks (INSTRUCTION); otherwise the plain `ANSWER: <x>` / fenced-code formats
stay the default, so the deterministic battery and the pairwise judge prompts are unchanged.
* parse(text): the contract object, or None when the text is not a valid contract (a ```json fence is tolerated).
* normalize(text): a valid contract becomes the plain format the existing graders read (bench/run.py strict parse: last line
  `ANSWER: <answer>`; code answers as one ```python block), anything else is returned unchanged. Every grader in llm/ goes
  through it, so both formats score the same.
Training: llm/selftrain.py asks for the contract on a share of its prompts (CONTRACT_FRAC) and keeps a sample only if it is a
valid contract AND its answer passes the checker, so the format is learnt from verified answers, never from made-up evidence."""
from __future__ import annotations

import json
import re

KEYS = ("answer", "confidence", "evidence", "assumptions")
INSTRUCTION = ("\n\nReply with ONLY a JSON object, no other text: "
               '{"answer": <final answer>, "confidence": <number from 0 to 1>, "evidence": [<short facts or steps that support '
               'the answer>], "assumptions": [<assumptions you made>]}.')
CODE_INSTRUCTION = INSTRUCTION[:-1] + ' Put the complete code as a string in "answer".'
_FENCE = re.compile(r"^```(?:json)?\s*\n(.*?)\n?```\s*$", re.S)


def _candidates(text: str):
    t = (text or "").strip()
    m = _FENCE.match(t)
    if m:
        yield m.group(1).strip()
    yield t
    i, j = t.find("{"), t.rfind("}")
    if 0 <= i < j:
        yield t[i:j + 1]


def parse(text: str) -> dict | None:
    for c in _candidates(text):
        try:
            o = json.loads(c)
        except ValueError:
            continue
        if not isinstance(o, dict) or not all(k in o for k in KEYS):
            continue
        conf = o["confidence"]
        if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            continue
        if not isinstance(o["evidence"], list) or not isinstance(o["assumptions"], list):
            continue
        if o["answer"] is None or (isinstance(o["answer"], str) and not o["answer"].strip()):
            continue
        return o
    return None


def _is_code(a: str) -> bool:
    return "```" in a or bool(re.search(r"^\s*(def |class |import |from \w+ import )", a, re.M))


def normalize(text: str) -> str:
    o = parse(text)
    if o is None:
        return text
    a = o["answer"]
    a = a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
    if _is_code(a):
        return a if "```" in a else "```python\n" + a.strip("\n") + "\n```"
    steps = "\n".join(str(e) for e in o["evidence"] if str(e).strip())
    return (steps + "\n" if steps else "") + "ANSWER: " + " ".join(a.split())