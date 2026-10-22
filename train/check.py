"""Answer extraction and checking. A task passes only if the checker says so; nothing here is graded by an LLM."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

_ANS = re.compile(r"ANSWER\s*[:=]\s*(.+)", re.I)


def final_answer(text):
    hits = _ANS.findall(text or "")
    if hits:
        return hits[-1].strip().strip("`*$ ").rstrip(".")
    m = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    return m[-1].strip() if m else ""


def _num(s):
    s = (s or "").replace(",", "").replace("\\!", "")
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", s)
    return float(m.group(0)) if m else None


def check_int(pred, gold):
    v = _num(pred)
    return v is not None and abs(v - round(v)) < 1e-9 and int(round(v)) == int(gold)


def check_numeric(pred, gold):
    v, g = _num(pred), _num(gold)
    if v is None or g is None:
        return False
    return abs(v - g) <= max(0.011, 0.01 * abs(g))     # 2-decimal answers; 1 % relative for large values


def _letters(s):
    s = (s or "").upper()
    found = re.findall(r"\(([A-D])\)|\b([A-D])\b", s)
    got = {a or b for a, b in found}
    if not got:
        got = set(re.findall(r"[A-D]", re.sub(r"[^A-D]", "", s)))
    return "".join(sorted(got))


def check_mcq(pred, gold):
    return bool(pred) and _letters(pred) == _letters(gold)


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?[ \t]*\r?\n(.*?)```", text or "", re.S)
    return max(blocks, key=len) if blocks else ""


def check_code(text, task, timeout=90.0):
    code = extract_code(text)
    if not code or task["entry_point"] not in code:
        return False
    harness = code + "\n\n" + task["test"] + "\n\n"
    if "def check(" in task["test"]:
        harness += "check(%s)\n" % task["entry_point"]
    with tempfile.TemporaryDirectory(prefix="meshtrain-") as d:
        p = os.path.join(d, "t.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(harness)
        try:
            r = subprocess.run([sys.executable, p], cwd=d, capture_output=True, timeout=timeout, env={**os.environ, "PYTHONHASHSEED": "0"})
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def grade(task, text):
    if task["kind"] == "code":
        return check_code(text, task)
    return {"int": check_int, "numeric": check_numeric, "mcq": check_mcq}[task["kind"]](final_answer(text), task["answer"])
