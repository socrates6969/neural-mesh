"""Install the agent-kit capture hook into Claude Code's user settings (run by the lead AFTER merge, not by agents).

    python -m agentkit.install [--settings PATH] [--root REPO] [--python EXE] [--dry-run]

Adds a PostToolUse entry {matcher "Task|Agent", command "<python> <repo>/hooks/mesh_capture.py", timeout 10}.
The Python is the one the existing mesh_assist PreToolUse hook uses (Python 3.12 on this machine); --python overrides.
Idempotent: nothing changes when a mesh_capture.py command is already registered. The file is backed up first
(settings.json.bak-agentkit-<timestamp>) and written atomically; invalid JSON is never overwritten.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = Path.home() / ".claude" / "settings.json"
MATCHER = "Task|Agent"
SCRIPT = "mesh_capture.py"


def _commands(settings: dict, event: str):
    for group in (settings.get("hooks") or {}).get(event) or []:
        for h in group.get("hooks") or [] if isinstance(group, dict) else []:
            if isinstance(h, dict) and isinstance(h.get("command"), str):
                yield group, h


def assist_python(settings: dict) -> str | None:
    """The interpreter of the registered mesh_assist hook (the first token of its command), if any."""
    for _, h in _commands(settings, "PreToolUse"):
        cmd = h["command"]
        if "mesh_assist.py" in cmd:
            exe = cmd.split(" -")[0].split(" ")[0] if not cmd.startswith('"') else cmd[1:].split('"')[0]
            return exe
    return None


def plan(settings: dict, root: Path, python: str | None = None) -> tuple[dict, bool]:
    """(new settings, changed). Pure: the input dict is not modified."""
    new = json.loads(json.dumps(settings))
    if any(SCRIPT in h["command"] for _, h in _commands(new, "PostToolUse")):
        return new, False
    exe = python or assist_python(new) or sys.executable
    exe = exe.replace("\\", "/")
    script = (root / "hooks" / SCRIPT).as_posix()
    q = lambda s: '"' + s + '"' if " " in s else s
    entry = {"type": "command", "command": q(exe) + " " + q(script), "timeout": 10}
    post = new.setdefault("hooks", {}).setdefault("PostToolUse", [])
    for group in post:
        if isinstance(group, dict) and group.get("matcher") == MATCHER:
            group.setdefault("hooks", []).append(entry)
            break
    else:
        post.append({"matcher": MATCHER, "hooks": [entry]})
    return new, True


def install(settings_path: Path = SETTINGS, root: Path = ROOT, python: str | None = None, dry_run: bool = False) -> str:
    try:
        cur = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    except ValueError as e:
        raise SystemExit("refusing to touch %s: it is not valid JSON (%s)" % (settings_path, e))
    new, changed = plan(cur, root, python)
    if not changed:
        return "already installed: " + str(settings_path)
    if dry_run:
        return json.dumps(new["hooks"]["PostToolUse"], indent=2)
    if settings_path.exists():
        bak = settings_path.with_name(settings_path.name + ".bak-agentkit-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.copy2(settings_path, bak)
    tmp = settings_path.with_name(settings_path.name + ".tmp-agentkit")
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, settings_path)
    return "installed mesh_capture into " + str(settings_path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--settings", type=Path, default=SETTINGS)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--python")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    print(install(a.settings, a.root, a.python, a.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())