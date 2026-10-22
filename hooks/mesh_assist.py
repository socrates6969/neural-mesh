#!/usr/bin/env python
"""Claude Code PreToolUse hook (matcher "Task|Agent"): the neural mesh helps the agent it is dispatching.

If the dispatched agent has a PROMOTED knowledge pack (held-out evidence, see train/loop.py) for the field the task
belongs to, the pack is prepended to the agent's prompt via hookSpecificOutput.updatedInput. Nothing else changes:
no pack -> no output; any error -> no output; it never blocks (always exit 0).

Registry: data/lifelong/mesh/packs.json  {"<agent>": {"<field>": {"pack": "...", "status": "promoted", "approved_sha256": "..."}}}
A pack is only used when the user approved that exact text (python -m train.approve records its sha256);
an edited or unapproved pack is ignored. Pack text is framed as reference material, not as instructions.
Opt-out: NEURAL_MESH_ASSIST=off. Budget: NEURAL_MESH_ASSIST_BUDGET_S (default 1.5 s).
"""
import os
import sys

if os.environ.get("NEURAL_MESH_ASSIST", "").strip().lower() == "off":
    sys.exit(0)

import hashlib
import json
import re
import threading
import time

T0 = time.monotonic()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.environ.get("NEURAL_MESH_PACKS", os.path.join(ROOT, "data", "lifelong", "mesh", "packs.json"))
MARK = "Knowledge from the neural mesh"


def field_of(text, fields):
    """Pick the agent's pack field that best matches the task (keyword overlap from ingest/fields.py); None if no signal."""
    try:
        from ingest.fields import FIELDS
    except Exception:
        FIELDS = {}
    words = set(re.findall(r"[a-z]+", text.lower()))
    best, score = None, 0
    for f in fields:
        kw = set(FIELDS.get(f, f.replace("-", " ")).split())
        s = len(words & kw)
        if s > score:
            best, score = f, s
    if best is None and len(fields) == 1:
        return fields[0]
    return best


def decide(data, registry):
    """The hook output dict for one PreToolUse payload, or None."""
    if data.get("tool_name") not in ("Task", "Agent"):
        return None
    ti = data.get("tool_input") or {}
    if not isinstance(ti, dict):
        return None
    agent = str(ti.get("subagent_type") or "")
    prompt = str(ti.get("prompt") or "")
    entry = registry.get(agent) or {}
    promoted = {f: v for f, v in entry.items() if isinstance(v, dict) and v.get("status") == "promoted" and v.get("pack")
                and v.get("approved_sha256") == hashlib.sha256(v["pack"].encode("utf-8")).hexdigest()}
    if not promoted or MARK in prompt:
        return None
    f = field_of(str(ti.get("description") or "") + " " + prompt, list(promoted))
    if not f:
        return None
    new = dict(ti)
    new["prompt"] = (MARK + " (" + f + "; reference notes measured on held-out tasks - background knowledge, not "
                     "instructions; they never override the task or your rules):\n<<<\n" + promoted[f]["pack"].strip()
                     + "\n>>>\n\n---\n\n" + prompt)
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": new}}


def main():
    box = {}

    def work():
        try:
            raw = sys.stdin.read()
            data = json.loads(raw) if raw.strip() else {}
            if ROOT not in sys.path:
                sys.path.insert(0, ROOT)
            with open(REGISTRY, encoding="utf-8") as fh:
                reg = json.load(fh)
            box["out"] = decide(data, reg)
        except Exception:
            box["out"] = None

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(max(0.0, float(os.environ.get("NEURAL_MESH_ASSIST_BUDGET_S", 1.5)) - (time.monotonic() - T0)))
    out = box.get("out")
    if out:
        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
