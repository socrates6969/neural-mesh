r"""Human approval: mark ONE Global Workspace entry verified, after showing it. Interactive, human-only.

    python -m agentkit.approve <workspace_id>

This is the only way an agent-produced entry becomes verified=True (grader pipelines vouch for their own posts as
pipeline:<name>). Before anything is changed it refuses when:
  * stdin or stdout is not an interactive terminal;
  * it runs inside an agent / Claude Code process (CLAUDECODE, CLAUDE_CODE_*, CLAUDE_SESSION_ID, CLAUDE_PID in the
    environment; an ancestor named claude(.exe), node/bun/deno running a claude CLI, or any ancestor with "claude" in
    its command line) - a TTY alone is no proof of a human;
  * the parent chain is inconclusive: it must reach the OS shell (the GetShellWindow owner, image
    <GetSystemWindowsDirectoryW>\explorer.exe, same session), with no missing parent, no ancestor in another session
    and no WmiPrvSE / svchost / taskhostw launch below it;
  * the approval key (~/.claude/mesh-approve.key, created atomically and owner-only) is missing and cannot be made;
  * the backend cannot store an approval signature.
The entry (id and content) is printed with control characters and ANSI/ESC sequences removed (with a warning if any
were). Then Windows Hello is MANDATORY (not configured, unavailable, error, cancel or timeout all refuse) and the human
must ALSO type back a random per-run code. The process-ancestry checks are defense in depth, not the proof of a human:
Hello is. On POSIX there is no Hello, so approval always refuses. The approval is stored as verified_by="human:<os user>" plus
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


OUT_OF_TREE = frozenset({"wmiprvse.exe", "svchost.exe", "taskhostw.exe"})   # reparented / service-launched
MAX_DEPTH = 64

_PS_PROCS = ("Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,Name,CommandLine,"
             "ExecutablePath,SessionId | ConvertTo-Json -Compress")


def _proc(ppid, name, cmd="", path="", session=None) -> dict:
    return {"ppid": int(ppid or 0), "name": str(name or ""), "cmd": str(cmd or ""), "path": str(path or ""),
            "session": session}


def process_table() -> dict:
    """{pid: {ppid, name, cmd, path, session}} for every process (raises on failure; callers fail closed)."""
    import json
    import subprocess
    if os.name == "nt":
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_PROCS],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        rows = json.loads(r.stdout)
        rows = rows if isinstance(rows, list) else [rows]
        return {int(x["ProcessId"]): _proc(x["ParentProcessId"], x.get("Name"), x.get("CommandLine"),
                                           x.get("ExecutablePath"), x.get("SessionId")) for x in rows}
    r = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,sess=,comm=,args="], capture_output=True, text=True, timeout=30)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 4:
            out[int(parts[0])] = _proc(parts[1], os.path.basename(parts[3]), parts[4] if len(parts) > 4 else "",
                                       parts[3], parts[2])
    if not out:
        raise OSError("empty process table")
    return out


def windows_dir() -> str | None:
    """The Windows directory from the OS (GetSystemWindowsDirectoryW); SystemRoot/SYSTEMROOT env vars are ignored."""
    if os.name != "nt":
        return None
    import ctypes
    buf = ctypes.create_unicode_buffer(520)
    n = ctypes.windll.kernel32.GetSystemWindowsDirectoryW(buf, 520)
    return buf.value if 0 < n < 520 else None


def session_of(pid: int) -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    sid = wintypes.DWORD()
    ok = ctypes.windll.kernel32.ProcessIdToSessionId(wintypes.DWORD(pid), ctypes.byref(sid))
    return int(sid.value) if ok else None


def shell_pid() -> int | None:
    """pid owning the desktop shell window (GetShellWindow + GetWindowThreadProcessId), or None."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    u32 = ctypes.windll.user32
    u32.GetShellWindow.restype = wintypes.HWND
    hwnd = u32.GetShellWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    u32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
    return int(pid.value) or None


def os_root() -> dict | None:
    """The trust root from the OS: {pid, path, session} of the shell's explorer.exe, or None when unavailable."""
    try:
        wd, sp = windows_dir(), shell_pid()
        if not wd or not sp:
            return None
        return {"pid": sp, "path": os.path.join(wd, "explorer.exe").lower(), "session": session_of(sp),
                "self_session": session_of(os.getpid())}
    except Exception:
        return None


def ancestry(pid: int, table: dict, root: dict | None):
    r"""([(name, command line)] of the ancestors, nearest first; complete; reason).
    Windows: complete=True only when the walk reaches the OS shell: the pid that owns GetShellWindow(), whose image
    path is <GetSystemWindowsDirectoryW>\explorer.exe and whose session is this process's session. Above the shell the
    walk continues (names still checked for agents) to pid 0/4 or the first unreadable parent. Below the shell, a
    missing parent, a loop, MAX_DEPTH, a parent in another session, or a WmiPrvSE / svchost / taskhostw launch is
    inconclusive (refuse). A name alone never makes a root. POSIX is always inconclusive: ppid 1 (or a subreaper)
    means the process was reparented, and there is no OS shell anchor."""
    out, cur, seen = [], pid, {pid}
    me = table.get(pid)
    if me is None:
        return out, False, "this process is not in the process table"
    if os.name != "nt":
        return out, False, "POSIX: no OS shell anchor (ppid 1 / subreaper launches are inconclusive)"
    if not root:
        return out, False, "the OS shell (GetShellWindow) could not be identified"
    above = False
    for _ in range(MAX_DEPTH):
        e = table.get(cur)
        if e is None:
            return (out, True, "") if above else (out, False, "process %d vanished" % cur)
        if cur != pid and not above:
            n = e["name"].lower()
            if cur == root["pid"]:
                if e["path"].lower() != root["path"]:
                    return out, False, "the shell process is not %s" % root["path"]
                if root.get("self_session") is None or root.get("session") != root.get("self_session"):
                    return out, False, "the shell is not in this process's session"
                above = True                                     # verified shell; keep walking for agent names
            elif n in OUT_OF_TREE:
                return out, False, "launched through %s (reparented or out-of-tree)" % e["name"]
            elif me["session"] is not None and e["session"] is not None and str(e["session"]) != str(me["session"]):
                return out, False, "parent %s is in another session" % e["name"]
        ppid = e["ppid"]
        if ppid in (0, 4):
            return (out, True, "") if above else (out, False, "reached pid %d without passing the shell" % ppid)
        if ppid not in table:
            return (out, True, "") if above else (out, False, "parent of %s is missing" % (e["name"] or cur))
        if ppid in seen:
            return out, False, "parent chain loops"
        seen.add(ppid)
        out.append((table[ppid]["name"], table[ppid]["cmd"]))
        cur = ppid
    return out, False, "parent chain deeper than %d" % MAX_DEPTH


def current_ancestry():
    """(ancestors, complete, reason) for this process, or None when the process table cannot be read."""
    try:
        return ancestry(os.getpid(), process_table(), os_root())
    except Exception:
        return None


def _agentlike(name: str, cmd: str) -> str | None:
    n, c = (name or "").lower(), (cmd or "").lower()
    if n in AGENT_PROCS:
        return "parent process " + name
    if n in ("node", "node.exe", "bun", "bun.exe", "deno", "deno.exe") and "claude" in c:
        return "parent process %s running a claude CLI" % name
    if "claude" in c:
        return "parent process %s with claude in its command line" % name
    return None


def agent_reasons(environ=None, lineage=None) -> list[str]:
    """Why this looks like an agent / Claude Code process ([] = looks like a human's own shell).
    lineage: (ancestors, complete) as from ancestry(); None = inspect this process."""
    why = ["environment variable " + k for k in agent_env(environ)]
    got = current_ancestry() if lineage is None else lineage
    if got is None:
        why.append("parent processes could not be inspected")
        return why
    ancestors, complete = got[0], got[1]
    reason = got[2] if len(got) > 2 else ""
    for name, cmd in ancestors:
        r = _agentlike(name, cmd)
        if r:
            why.append(r)
    if not complete:
        why.append("the parent-process chain is inconclusive (%s)" % (reason or "ends early"))
    return why


_PS_HELLO = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and
  $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [Type]$t) { $task = $asTask.MakeGenericMethod($t).Invoke($null, @($op)); [void]$task.Wait(-1); $task.Result }
$null = [Windows.Security.Credentials.UI.UserConsentVerifier, Windows.Security.Credentials.UI, ContentType = WindowsRuntime]
$a = Await ([Windows.Security.Credentials.UI.UserConsentVerifier]::CheckAvailabilityAsync()) ([Windows.Security.Credentials.UI.UserConsentVerifierAvailability])
if ("$a" -ne 'Available') { Write-Output ("availability:" + $a); exit 0 }
$r = Await ([Windows.Security.Credentials.UI.UserConsentVerifier]::RequestVerificationAsync($env:MESH_APPROVE_MSG)) ([Windows.Security.Credentials.UI.UserConsentVerificationResult])
Write-Output ("result:" + $r)
"""


def parse_hello(out: str) -> str:
    """'verified' | 'unavailable' | 'denied' | 'error' from the PowerShell script's last line."""
    last = (out or "").strip().splitlines()[-1:] or [""]
    line = last[0].strip()
    if line.startswith("availability:"):
        return "unavailable"
    if line.startswith("result:"):
        return "verified" if line.split(":", 1)[1].strip() == "Verified" else "denied"
    return "error"


HELLO_TIMEOUT_S = 120


def hello_verify(message: str) -> str:
    """Windows Hello (UserConsentVerifier) prompt: 'verified', 'denied' (cancelled/failed), 'unavailable' (not
    configured / not Windows), or 'error' (WinRT could not be reached)."""
    if os.name != "nt":
        return "unavailable"
    import subprocess
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_HELLO], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=HELLO_TIMEOUT_S,
                           env=dict(os.environ, MESH_APPROVE_MSG=message[:200]))
    except Exception:
        return "error"
    return parse_hello(r.stdout)


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


def main(argv=None, ws=None, ask=input, stdin=None, stdout=None, environ=None, lineage=None, key=None,
         hello=hello_verify) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 2
    why = agent_reasons(environ, lineage)
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
    # the entry (id and content) is already on screen; Hello, when configured, is required IN ADDITION to the code
    # the entry (id and content) is already on screen. Windows Hello is MANDATORY: anything but a successful
    # verification (not configured, unavailable, error, cancel, timeout) refuses. The typed code is required as well.
    state = hello("Approve workspace entry %s as verified" % sanitize(entry_id)[0])
    if state != "verified":
        print("refusing: Windows Hello verification is required and did not succeed (%s); nothing changed." % state,
              file=sys.stderr)
        return 7
    print("Windows Hello: verified.")
    code = nonce()
    try:
        answer = ask("\nTo mark entry %s verified, type this code: %s (anything else cancels): "
                     % (sanitize(entry_id)[0], code))
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