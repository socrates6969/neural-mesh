"""$0 structural pre-check: obvious junk never costs a judge call (local or Haiku). Pure stdlib, deterministic.

check(text) returns None (looks structurally fine) or one reason code:
  empty       nothing (or a few characters) of substance
  refusal     a short answer that opens with a canned refusal / "as an AI" line
  truncation  an unclosed ``` code fence (the text was cut mid-block)
  repetition  a degenerate loop: one line or one 6-word phrase repeated many times
  parse       expect="json" and the text is not JSON, or a ```python block that does not compile
It is deliberately conservative: it only rejects what is unambiguous, because a false reject silently discards a good row.
Idea source: ecosystem-scout's "$0 pre-judge" recommendation (hive/ecosystem-scout.md); reimplemented, no code copied.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter

MIN_CHARS = 3
REFUSAL_MAX_CHARS = 600
REFUSAL = re.compile(
    r"^\s*(?:(?:i['’]?m sorry|i am sorry|sorry|i apologi[sz]e)[,.]?\s*(?:but\s+)?i\s+(?:can(?:no|['’]?)t|am unable|['’]?m unable|won['’]?t)|"
    r"i can(?:no|['’]?)t (?:help|assist|comply|do that|provide)|i cannot (?:help|assist|comply|provide|do)|"
    r"i['’]?m unable to|i am unable to|as an ai(?: language model)?|unfortunately,? i (?:can(?:no|['’]?)t|am unable|do not have))", re.I)
FENCE = re.compile(r"^\s*```", re.M)
PY_BLOCK = re.compile(r"```(?:python|py)[ \t]*\r?\n(.*?)```", re.S | re.I)
SHINGLE = 6
SHINGLE_REPEATS = 8       # the same 6-word phrase this many times = a loop
LINE_REPEATS = 6          # the same non-trivial line this many times = a loop


def _repeats(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 12]
    if lines and Counter(lines).most_common(1)[0][1] >= LINE_REPEATS:
        return True
    words = text.split()
    if len(words) < SHINGLE * SHINGLE_REPEATS:
        return False
    grams = Counter(tuple(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1))
    return grams.most_common(1)[0][1] >= SHINGLE_REPEATS


def _bad_python(text: str) -> bool:
    for blk in PY_BLOCK.findall(text):
        if not blk.strip():
            continue
        try:
            ast.parse(blk)
        except (SyntaxError, ValueError):
            return True
    return False


def check(text: str | None, expect: str | None = None) -> str | None:
    """None if the text passes, else the reason code (see module doc). expect: None | 'json' | 'python'."""
    t = (text or "").strip()
    if len(t) < MIN_CHARS:
        return "empty"
    if len(t) <= REFUSAL_MAX_CHARS and REFUSAL.match(t):
        return "refusal"
    if len(FENCE.findall(t)) % 2 == 1:
        return "truncation"
    if _repeats(t):
        return "repetition"
    if expect == "json":
        try:
            json.loads(t)
        except ValueError:
            return "parse"
    if expect == "python" and PY_BLOCK.search(t) and _bad_python(t):
        return "parse"
    return None


def expect_of(reference: str) -> str | None:
    """what the reference looks like, so a candidate can be held to the same shape (python fence -> must compile)."""
    return "python" if PY_BLOCK.search(reference or "") else None
