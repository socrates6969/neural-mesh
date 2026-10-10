"""S1 shadow hook for Claude Code's UserPromptSubmit event: `python -S -m llm hook --shadow` (proposed config:
hooks/settings-frontend-shadow.json; NOT installed by this repo).

It only ENQUEUES: one line {ts, sha256, session, project, prompt} appended to models/frontend_shadow.jsonl (millisecond ts; at most
MAX_LINE_BYTES; written with ONE os.write on an O_APPEND descriptor, plus a sidecar lock on Windows, so concurrent sessions never tear
a line). It prints NOTHING (so it
never adds context and never changes the prompt), always exits 0 (never 2, which would block the prompt), loads no model, and is
bounded by a watchdog (BUDGET_S, os._exit(0)). Off switch: NEURAL_MESH_FE_SHADOW=off (checked before anything else is read).
`python -m llm shadow-drain` does the expensive part offline. Coexists with hooks/recall_hook.py (that one prints recalled memories;
this one prints nothing) and with the router shadow hooks/route_shadow.py (PreToolUse, data/lifelong/route_shadow.jsonl)."""
import os
import sys

BUDGET_S = 1.0
MAX_PROMPT_CHARS = 20000
MAX_LINE_BYTES = 64 * 1024      # one queue line, UTF-8, newline included
LOCK_WAIT_S = 0.25


def _watchdog(budget_s: float) -> None:
    import threading
    t = threading.Timer(budget_s, lambda: os._exit(0))
    t.daemon = True
    t.start()


def enqueue(raw: str, path=None, now=None) -> bool:
    """parse the hook JSON and append one queue line; False (and no write) on anything unusable. Never raises."""
    import hashlib
    import json
    import time
    try:
        raw = (raw or "").lstrip("\ufeff")   # a BOM from a Windows pipe must not lose the prompt
        d = json.loads(raw) if raw.strip() else {}
        prompt = d.get("prompt") if isinstance(d, dict) else None
        if not isinstance(prompt, str) or not prompt.strip():
            return False
        prompt = prompt[:MAX_PROMPT_CHARS]
        cwd = str(d.get("cwd") or "")
        project = [x for x in cwd.replace("\\", "/").rstrip("/").split("/") if x][-1:] or [""]
        now = time.time() if now is None else float(now)
        # millisecond ts (old whole-second lines stay readable: parse_ts accepts both)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(now)) + ".%03dZ" % min(999, int((now % 1) * 1000))
        rec = {"ts": ts, "sha256": hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest(),
               "session": d.get("session_id"), "project": project[0], "prompt": prompt}
        if path is None:
            from . import config
            path = config.MODELS / "frontend_shadow.jsonl"
        line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8", "replace")
        while len(line) > MAX_LINE_BYTES and rec["prompt"]:   # cap the line (multi-byte text / JSON escapes can exceed the char cap)
            rec["prompt"], rec["truncated"] = rec["prompt"][:int(len(rec["prompt"]) * MAX_LINE_BYTES / len(line) * 0.95)], True
            line = (json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8", "replace")
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        _append_line(str(path), line)
        return True
    except Exception:
        return False


def _append_line(path: str, line: bytes) -> None:
    """ONE os.write of the whole line on an O_APPEND descriptor, so concurrent sessions never interleave inside a line. On Windows
    the CRT emulates O_APPEND (seek-to-end, then write), so writers also serialise on a sidecar lock file (bounded spin, never
    blocks the hook past LOCK_WAIT_S; past it the write still happens)."""
    lock = None
    if os.name == "nt":
        lock = _lock_nt(path + ".lock")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
        try:
            n = os.write(fd, line)
            if n != len(line):   # a short write on a regular file means disk full or similar: finish the same line
                os.write(fd, line[n:])
        finally:
            os.close(fd)
    finally:
        if lock is not None:
            _unlock_nt(lock)


def _lock_nt(lock_path: str):
    import msvcrt
    import time
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    except OSError:
        return None
    end = time.monotonic() + LOCK_WAIT_S
    while True:
        try:
            os.lseek(fd, 0, 0)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return fd
        except OSError:
            if time.monotonic() > end:
                os.close(fd)
                return None
            time.sleep(0.002)


def _unlock_nt(fd) -> None:
    import msvcrt
    try:
        os.lseek(fd, 0, 0)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    finally:
        os.close(fd)


def cli(argv) -> None:
    """never returns: exit 0 always"""
    try:
        if os.environ.get("NEURAL_MESH_FE_SHADOW", "").strip().lower() != "off" and "--shadow" in argv:
            _watchdog(float(os.environ.get("NEURAL_MESH_FE_SHADOW_BUDGET_S", BUDGET_S)))
            raw = sys.stdin.buffer.read().decode("utf-8", "replace")
            enqueue(raw)
    except BaseException:
        pass
    os._exit(0)
