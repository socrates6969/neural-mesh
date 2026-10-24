"""code_run: run Python or shell code (plus optional tests) in the network-less grader sandbox and return
stdout/stderr/exit code. A failure comes back as a structured error so the calling agent can repair the code and
call again; the tool never edits code itself. Repairs are capped at MAX_REPAIRS rounds per (session, task_id); the
round counter is kept here, not trusted from the caller, and is reset when a run passes.
"""
from __future__ import annotations

import json
import os
import threading

from tools.mesh_mcp.common import cap, run_sandbox

MAX_REPAIRS = 3
LANGS = ("python", "sh")
_failures: dict[tuple[str, str], int] = {}
_lock = threading.Lock()

HARNESS = r'''
import json, os, subprocess, sys, traceback
a = json.load(open("args.json"))
os.chdir("/tmp")
open("/tmp/solution." + ("py" if a["lang"] == "python" else "sh"), "w").write(a["code"])
if a.get("tests"):
    open("/tmp/tests." + ("py" if a["lang"] == "python" else "sh"), "w").write(a["tests"])

def report(phase, exc):
    tb = traceback.extract_tb(exc.__traceback__)
    frames = [f for f in tb if f.filename.endswith(("solution.py", "tests.py"))] or tb[-1:]
    f = frames[-1] if frames else None
    err = {"phase": phase, "type": type(exc).__name__, "message": str(exc)[:500],
           "file": os.path.basename(f.filename) if f else None, "line": f.lineno if f else None,
           "code_line": (f.line or "").strip() if f else None}
    if isinstance(exc, SyntaxError):
        err.update({"file": os.path.basename(exc.filename or "solution.py"), "line": exc.lineno,
                    "code_line": (exc.text or "").strip()})
    traceback.print_exception(exc, file=sys.stderr)
    print("@@ERROR@@" + json.dumps(err), file=sys.stderr)
    sys.exit(1)

if a["lang"] == "python":
    sys.path.insert(0, "/tmp")
    import runpy
    try:
        ns = runpy.run_path("/tmp/solution.py", run_name="solution" if a.get("tests") else "__main__")
    except SystemExit as e:
        if e.code not in (0, None):
            report("solution", e)
        ns = {}
    except BaseException as e:
        report("solution", e)
    if a.get("tests"):
        try:
            exec(compile(open("/tmp/tests.py").read(), "/tmp/tests.py", "exec"), dict(ns, __name__="tests"))
        except BaseException as e:
            report("tests", e)
        print("@@TESTS_PASSED@@")
else:
    for phase, name in (("solution", "/tmp/solution.sh"), ("tests", "/tmp/tests.sh")):
        if phase == "tests" and not a.get("tests"):
            break
        r = subprocess.run(["bash", "-e", name], capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode:
            print("@@ERROR@@" + json.dumps({"phase": phase, "type": "ExitCode", "message": "exit %d" % r.returncode,
                                            "stderr_tail": r.stderr[-400:]}), file=sys.stderr)
            sys.exit(r.returncode)
    if a.get("tests"):
        print("@@TESTS_PASSED@@")
'''


def _key(task_id):
    return (os.environ.get("CLAUDE_SESSION_ID") or "default", str(task_id or "default"))


def code_run(lang: str, code: str, tests: str | None = None, task_id: str | None = None,
             timeout_s: int = 60) -> str:
    lang = str(lang or "python").lower().replace("python3", "python").replace("bash", "sh")
    if lang not in LANGS:
        raise ValueError("lang must be one of %s" % ", ".join(LANGS))
    if not str(code or "").strip():
        raise ValueError("code is required")
    key = _key(task_id)
    with _lock:
        failed = _failures.get(key, 0)
    if failed > MAX_REPAIRS:
        return cap("STOP (task %s): the initial run and %d repair rounds all failed. The repair budget is used up; "
                   "report the failure and the last error instead of retrying. Use a new task_id only for a "
                   "genuinely different task." % (key[1], MAX_REPAIRS))
    rnd = failed  # 0 = first run, 1..MAX_REPAIRS = repair rounds
    out, err, rc = run_sandbox(HARNESS, {"lang": lang, "code": code, "tests": tests or ""}, timeout_s=timeout_s)
    structured = None
    keep = []
    for line in err.splitlines():
        if line.startswith("@@ERROR@@"):
            try:
                structured = json.loads(line[len("@@ERROR@@"):])
            except ValueError:
                pass
        else:
            keep.append(line)
    err = "\n".join(keep)
    tests_passed = "@@TESTS_PASSED@@" in out
    out = out.replace("@@TESTS_PASSED@@\n", "").replace("@@TESTS_PASSED@@", "")
    ok = rc == 0 and (tests_passed or not tests)
    with _lock:
        if ok:
            _failures.pop(key, None)
        else:
            _failures[key] = failed + 1
    label = "first run" if rnd == 0 else "repair round %d of %d" % (rnd, MAX_REPAIRS)
    head = "%s (%s, %s, task %s)" % ("PASS" if ok else "FAIL", lang, label, key[1])
    if rc == 124:
        structured = {"phase": "solution", "type": "Timeout", "message": err.strip()[:200]}
    lines = [head, "exit code %d%s" % (rc, "; tests passed" if ok and tests else ""),
             "--- stdout ---", out[-2500:].rstrip(), "--- stderr ---", err[-1500:].rstrip()]
    if not ok:
        if structured is None:
            structured = {"phase": "solution", "type": "ExitCode", "message": "exit %d" % rc}
        lines.append("ERROR: " + json.dumps(structured))
        left = MAX_REPAIRS - rnd
        lines.append("NEXT: fix the code yourself and call code_run again with task_id=%r (%d repair round%s left)"
                     % (key[1], left, "" if left == 1 else "s") if left > 0 else
                     "STOP: that was the last repair round; report the failure and this error.")
    return cap("\n".join(lines))


def reset(task_id: str | None = None) -> None:
    """Testing hook: forget the failure count for a task (or all tasks)."""
    with _lock:
        if task_id is None:
            _failures.clear()
        else:
            _failures.pop(_key(task_id), None)
