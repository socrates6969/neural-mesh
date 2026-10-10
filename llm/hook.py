"""S1 shadow hook for Claude Code's UserPromptSubmit event: `python -S -m llm hook --shadow` (proposed config:
hooks/settings-frontend-shadow.json; NOT installed by this repo).

It only ENQUEUES: one line {ts, sha256, session, project, prompt} appended to models/frontend_shadow.jsonl. It prints NOTHING (so it
never adds context and never changes the prompt), always exits 0 (never 2, which would block the prompt), loads no model, and is
bounded by a watchdog (BUDGET_S, os._exit(0)). Off switch: NEURAL_MESH_FE_SHADOW=off (checked before anything else is read).
`python -m llm shadow-drain` does the expensive part offline. Coexists with hooks/recall_hook.py (that one prints recalled memories;
this one prints nothing) and with the router shadow hooks/route_shadow.py (PreToolUse, data/lifelong/route_shadow.jsonl)."""
import os
import sys

BUDGET_S = 1.0
MAX_PROMPT_CHARS = 20000


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
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)), "sha256": hashlib.sha256(prompt.encode("utf-8", "replace")).hexdigest(),
               "session": d.get("session_id"), "project": project[0], "prompt": prompt}
        if path is None:
            from . import config
            path = config.MODELS / "frontend_shadow.jsonl"
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:   # one write of one line: concurrent sessions do not interleave lines
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


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
