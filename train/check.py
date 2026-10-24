"""Answer extraction and checking. A task passes only if the checker says so; nothing here is graded by an LLM."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

_ANS = re.compile(r"ANSWER\s*[:=]\s*(.+)", re.I)


_JSON_FENCE = re.compile(r"```[ \t]*(?:json|json5)[ \t]*\r?\n.*?(?:```|\Z)", re.S | re.I)


def _json_answer(text):
    try:
        from agentkit import parse
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from agentkit import parse
    d = parse.parse_output(text)
    return parse.answer_text(d["answer"]) if d else ""


def final_answer(text):
    """The ANSWER: line (outside ```json blocks; template placeholders such as <integer> are skipped), else the
    agent-kit JSON block's "answer", else the last \\boxed{}. The ANSWER line wins when both exist."""
    hits = [h for h in _ANS.findall(_JSON_FENCE.sub("", text or "")) if not h.strip().startswith("<")]
    if hits:
        return hits[-1].strip().strip("`*$ ").rstrip(".")
    try:
        ja = _json_answer(text or "")
    except Exception:
        ja = ""
    if ja:
        return ja.strip().strip("`*$ ").rstrip(".")
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


GRADER_IMAGE = os.environ.get("MESHTRAIN_GRADER_IMAGE", "meshtrain-grader:1")


def _run_sandboxed(d, timeout):
    """Run d/t.py in a container: no network, read-only code mount, 1 CPU, 1 GB, 128 pids, unprivileged user.
    Without Docker, grading refuses to run unless MESHTRAIN_ALLOW_UNSANDBOXED=1 (then: scrubbed env, temp cwd)."""
    import shutil
    import uuid
    if shutil.which("docker"):
        name = "mtg-" + uuid.uuid4().hex[:12]
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "1", "--memory", "1g",
               "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp:rw,size=64m", "-v", d + ":/code:ro", "-w", "/tmp",
               "-e", "PYTHONHASHSEED=0", GRADER_IMAGE, "python", "/code/t.py"]
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return False
    if os.environ.get("MESHTRAIN_ALLOW_UNSANDBOXED") != "1":
        raise RuntimeError("docker not found: refusing to execute model-written code unsandboxed")
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "TEMP", "TMP") if k in os.environ}
    env["PYTHONHASHSEED"] = "0"
    try:
        return subprocess.run([sys.executable, os.path.join(d, "t.py")], cwd=d, capture_output=True, timeout=timeout, env=env).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def check_code(text, task, timeout=120.0):
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
        return _run_sandboxed(d, timeout)


JUDGE = Path(__file__).with_name("stdio_judge.py")


def _same(out, exp):
    a, b = out.split(), exp.split()
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        try:
            fx, fy = float(x), float(y)
        except ValueError:
            return False
        if not (math.isfinite(fx) and math.isfinite(fy)) or not abs(fx - fy) <= 1e-6 * max(1.0, abs(fy)):
            return False
    return True


def check_stdio(text, task, code=None):
    """Competitive-programming task: full program reading stdin; every stored test must match.
    Expected outputs stay on the host; the container only sees the inputs and returns what the program printed."""
    import shutil
    import uuid
    code = code if code is not None else extract_code(text)
    if not code:
        return False
    ins, exp = task["tests"]["inputs"], task["tests"]["outputs"]
    nonce = "@@" + uuid.uuid4().hex + "@@"
    with tempfile.TemporaryDirectory(prefix="meshtrain-") as d:
        Path(d, "sol.py").write_text(code, encoding="utf-8")
        Path(d, "t.py").write_text(JUDGE.read_text(encoding="utf-8"), encoding="utf-8")
        Path(d, "inputs.json").write_text(json.dumps({"inputs": ins, "time_limit": task["time_limit"]}), encoding="utf-8")
        total = min(600.0, 30.0 + task["time_limit"] * len(ins))
        if not shutil.which("docker"):
            raise RuntimeError("docker not found: refusing to execute model-written code unsandboxed")
        name = "mtg-" + uuid.uuid4().hex[:12]
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "1", "--memory", "1g",
               "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp:rw,size=64m", "-v", d + ":/code:ro", "-w", "/tmp",
               "-e", "JUDGE_NONCE=" + nonce, GRADER_IMAGE, "python", "/code/t.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=total)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return False
    if nonce not in r.stdout:
        return False
    try:
        res = json.loads(r.stdout.rsplit(nonce, 1)[1])
        if not isinstance(res, list) or len(res) != len(exp):
            return False
        for x, e in zip(res, exp):
            if not (isinstance(x, dict) and type(x.get("rc")) is int and type(x.get("timeout")) is bool
                    and isinstance(x.get("out"), str)):
                return False
            if x["rc"] != 0 or x["timeout"] or not _same(x["out"], e):
                return False
        return True
    except Exception:
        return False


def grade(task, text):
    if task["kind"] == "code":
        return check_code(text, task)
    if task["kind"] == "stdio":
        return check_stdio(text, task)
    return {"int": check_int, "numeric": check_numeric, "mcq": check_mcq}[task["kind"]](final_answer(text), task["answer"])
