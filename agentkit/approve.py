"""Human approval: mark ONE Global Workspace entry verified, after showing it. Interactive, human-only.

    python -m agentkit.approve <workspace_id>

This is the only way an agent-produced entry becomes verified=True (grader pipelines vouch for their own posts as
pipeline:<name>). Before anything is changed it refuses when:
  * stdin or stdout is not an interactive terminal;
  * it runs inside an agent / Claude Code process (CLAUDECODE, CLAUDE_CODE_*, CLAUDE_SESSION_ID, CLAUDE_PID in the
    environment, or a parent process named claude / claude.exe) - a TTY alone is no proof of a human;
  * the approval key (~/.claude/mesh-approve.key, created atomically and owner-only) is missing and cannot be made;
  * the backend cannot store an approval signature.
The entry is printed with control characters and ANSI/ESC sequences removed (with a warning if any were), and the
human must type back a random per-run code. The approval is stored as verified_by="human:<os user>" plus
approval_sig = HMAC-SHA256 over the entry (agentkit/trust.py), which the hook checks before trusting the entry.
"""
from __future__ import annotations

import getpass
import os
import re
import secrets
import sys

from agentkit import trust

NONCE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"     # no 0/O/1/I/L look-alikes
NONCE_LEN = 6
AGENT_ENV_EXACT = ("CLAUDECODE", "CLAUDE_SESSION_ID", "CLAUDE_PID")
AGENT_ENV_PREFIX = ("CLAUDE_CODE_",)
AGENT_PROCS = ("claude", "claude.exe")

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[PX^_][^\x1b]*\x1b\\|\x1b.?|\x9b[0-?]*[ -/]*[@-~]")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def sanitize(text) -> tuple[str, bool]:
    """(printable text, whether anything was removed): ANSI/ESC sequences dropped, C0/C1 controls except \\n and \\t
    removed (\\r included, so a line cannot overwrite what was printed before it)."""
    s = str(text if text is not None else "")
    out = _CTRL.sub("", _ANSI.sub("", s))
    return out, out != s


def agent_env(environ=None) -> list[str]:
    env = os.environ if environ is None else environ
    return sorted(k for k in env if k.upper() in AGENT_ENV_EXACT or k.upper().startswith(AGENT_ENV_PREFIX))


def _parent_names_windows(pid: int, depth: int = 12) -> list[str]:
    import ctypes
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                    ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                    ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]
    k32 = ctypes.windll.kernel32
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    if snap in (-1, 0xFFFFFFFF, None):
        raise OSError("snapshot failed")
    procs = {}
    try:
        e = PE()
        e.dwSize = ctypes.sizeof(PE)
        ok = k32.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            procs[e.th32ProcessID] = (e.th32ParentProcessID, e.szExeFile)
            ok = k32.Process32NextW(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    names, cur, seen = [], pid, set()
    for _ in range(depth):
        if cur not in procs or cur in seen:
            break
        seen.add(cur)
        ppid, _name = procs[cur]
        if ppid in procs:
            names.append(procs[ppid][1])
        cur = ppid
    return names


def _parent_names_posix(pid: int, depth: int = 12) -> list[str]:
    import subprocess
    names, cur = [], pid
    for _ in range(depth):
        r = subprocess.run(["ps", "-o", "ppid=", "-p", str(cur)], capture_output=True, text=True, timeout=5)
        ppid = int((r.stdout or "0").strip() or 0)
        if ppid <= 1:
            break
        n = subprocess.run(["ps", "-o", "comm=", "-p", str(ppid)], capture_output=True, text=True, timeout=5)
        names.append(os.path.basename((n.stdout or "").strip()))
        cur = ppid
    return names


def parent_names() -> list[str] | None:
    """Names of the ancestor processes, or None when they cannot be determined (treated as a refusal)."""
    try:
        return (_parent_names_windows if os.name == "nt" else _parent_names_posix)(os.getpid())
    except Exception:
        return None


def agent_reasons(environ=None, parents=None) -> list[str]:
    """Why this looks like an agent / Claude Code process ([] = looks like a human's own shell)."""
    why = ["environment variable " + k for k in agent_env(environ)]
    names = parent_names() if parents is None else parents
    if names is None:
        why.append("parent processes could not be inspected")
    else:
        why += ["parent process " + n for n in names if n.lower() in AGENT_PROCS]
    return why


def _interactive(stdin=None, stdout=None) -> bool:
    i, o = stdin or sys.stdin, stdout or sys.stdout
    try:
        return bool(i.isatty() and o.isatty())
    except Exception:
        return False


def show(e: dict) -> tuple[str, bool]:
    rows = [("id", e.get("id")), ("task/session", e.get("task_id")), ("agent", e.get("agent")), ("kind", e.get("kind")),
            ("field", e.get("field")), ("confidence", e.get("confidence")), ("verified", e.get("verified")),
            ("verified_by", e.get("verified_by")), ("agent claims verified", e.get("claimed_verified")),
            ("sources", "; ".join(str(s) for s in e.get("sources") or []))]
    text = "\n".join("%-22s %s" % (k + ":", v) for k, v in rows if v not in (None, "", []))
    text += "\n\ncontent:\n" + str(e.get("content") or "")
    return sanitize(text)


def nonce(n: int = NONCE_LEN) -> str:
    return "".join(secrets.choice(NONCE_ALPHABET) for _ in range(n))


def main(argv=None, ws=None, ask=input, stdin=None, stdout=None, environ=None, parents=None, key=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 2
    why = agent_reasons(environ, parents)
    if why:
        print("refusing: this looks like an agent or Claude Code process (%s). Run it in your own terminal."
              % ", ".join(why), file=sys.stderr)
        return 5
    if not _interactive(stdin, stdout):
        print("refusing: approval needs an interactive terminal (a human at the keyboard).", file=sys.stderr)
        return 3
    if ws is None:
        from agentkit import workspace as ws
    if not ws.can_approve():
        print("refusing: the workspace backend cannot store a signed approval (needs get() and "
              "approve(entry_id, by=, approval_sig=)); nothing changed.", file=sys.stderr)
        return 4
    key = key if key is not None else trust.load_key(create=True)
    if not key:
        print("refusing: no usable approval key at %s (could not create it owner-only); nothing changed."
              % trust.key_path(), file=sys.stderr)
        return 6
    entry_id = argv[0]
    e = ws.get(entry_id)
    if not e:
        print("no such workspace entry: %s" % sanitize(entry_id)[0], file=sys.stderr)
        return 1
    text, stripped = show(e)
    print(text)
    if stripped:
        print("\nWARNING: control characters / terminal escape sequences were removed from this entry before "
              "display. The stored text contains them; treat it with suspicion.")
    if e.get("verified") and trust.trusted(e, key):
        print("\nalready verified (by %s); nothing changed." % sanitize(e.get("verified_by"))[0])
        return 0
    code = nonce()
    try:
        answer = ask("\nTo mark this entry verified, type this code: %s (anything else cancels): " % code)
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip() != code:
        print("cancelled; nothing changed.")
        return 1
    by = "human:" + (getpass.getuser() or "unknown")
    sig = trust.sign(dict(e, id=entry_id if e.get("id") in (None, "") else e.get("id")), by, key)
    ok = ws.approve(entry_id, by=by, approval_sig=sig)
    print("approved (%s)." % by if ok else "approval failed; nothing changed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())