"""Human approval: mark ONE Global Workspace entry verified, after showing it. Interactive only.

    python -m agentkit.approve <workspace_id>

This is the only way an agent-produced entry becomes verified=True (grader pipelines set it out of band when they
post). It refuses to run without an interactive terminal on both stdin and stdout, so no agent, hook or script can
approve by piping input. The approval is recorded as verified_by="human:<os user>".
"""
from __future__ import annotations

import getpass
import sys

CONFIRM = "APPROVE"


def _interactive(stdin=None, stdout=None) -> bool:
    i, o = stdin or sys.stdin, stdout or sys.stdout
    try:
        return bool(i.isatty() and o.isatty())
    except Exception:
        return False


def show(e: dict) -> str:
    rows = [("id", e.get("id")), ("task/session", e.get("task_id")), ("agent", e.get("agent")), ("kind", e.get("kind")),
            ("field", e.get("field")), ("confidence", e.get("confidence")), ("verified", e.get("verified")),
            ("verified_by", e.get("verified_by")), ("agent claims verified", e.get("claimed_verified")),
            ("sources", "; ".join(e.get("sources") or []))]
    head = "\n".join("%-22s %s" % (k + ":", v) for k, v in rows if v not in (None, "", []))
    return head + "\n\ncontent:\n" + str(e.get("content") or "")


def main(argv=None, ws=None, ask=input, stdin=None, stdout=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 2
    if not _interactive(stdin, stdout):
        print("refusing: approval needs an interactive terminal (a human at the keyboard).", file=sys.stderr)
        return 3
    if ws is None:
        from agentkit import workspace as ws
    if not ws.can_approve():
        print("refusing: the workspace backend has no get()/approve() (train.memory.workspace); nothing changed.",
              file=sys.stderr)
        return 4
    entry_id = argv[0]
    e = ws.get(entry_id)
    if not e:
        print("no such workspace entry: %s" % entry_id, file=sys.stderr)
        return 1
    print(show(e))
    if e.get("verified"):
        print("\nalready verified (by %s); nothing changed." % (e.get("verified_by") or "?"))
        return 0
    try:
        answer = ask("\nType %s to mark this entry verified (anything else cancels): " % CONFIRM)
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer.strip() != CONFIRM:
        print("cancelled; nothing changed.")
        return 1
    ok = ws.approve(entry_id, by="human:" + getpass.getuser())
    print("approved." if ok else "approval failed; nothing changed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())