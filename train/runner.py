"""Run one catalog agent on one task, headless and lean (no user plugins, MCP servers, skills or session files).

The agent's own definition (system prompt body, tools, model) is passed with --agents, so it behaves as installed.
Optional mesh knowledge (a pack) is prepended to the task - this is exactly what the live mesh_assist hook does.
Web access is disabled during evaluation so answers cannot be looked up; packs are researched separately.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

CLAUDE = shutil.which("claude") or "claude"
EVAL_TOOLS = "Bash,Read,Write,Edit"
# file tools are limited to the per-run temp directory; Bash only for python (no web tools during evaluation)
ALLOWED = ["Bash(python:*)", "Bash(python3:*)", "Bash(py:*)", "Read(./**)", "Write(./**)", "Edit(./**)"]
FORMAT = {
    "int": "Finish with one final line: ANSWER: <integer>",
    "numeric": "Finish with one final line: ANSWER: <number>",
    "mcq": "Finish with one final line: ANSWER: <letter(s)>, e.g. ANSWER: B or ANSWER: ACD",
    "code": "Put the complete final solution in a single ```python block (it is run against hidden tests).",
}
_FM = re.compile(r"^\ufeff?---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
_lock = threading.Lock()

# Child processes get only what they need to run and to authenticate Claude Code (OAuth lives in ~/.claude);
# API tokens for other services in the parent environment (GitHub, AWS, Pinecone, ...) are NOT passed on.
_ENV_KEEP = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "USERPROFILE", "HOME", "HOMEDRIVE",
             "HOMEPATH", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
             "LANG", "PYTHONUTF8", "PYTHONIOENCODING", "CLAUDE_CONFIG_DIR"}


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if k.upper() in _ENV_KEEP or k.upper().startswith("ANTHROPIC_")}
    env.update(extra or {})
    return env


def load_agent(entry: dict) -> dict:
    """{name, prompt, description, tools, model} from a catalog row (reads the agent file; nothing is modified)."""
    text = Path(entry["path"]).read_text(encoding="utf-8", errors="replace") if entry.get("path") else ""
    m = _FM.match(text)
    body = (m.group(2) if m else text).strip() or entry["description"]
    model = (entry.get("model") or "").strip().strip("\"'")
    return {"name": entry["name"], "prompt": body, "description": entry["description"][:300],
            "model": model if model in ("opus", "sonnet", "haiku", "fable") else ""}


def build_prompt(task: dict, pack: str | None) -> str:
    parts = []
    if pack:
        parts.append("Knowledge from the neural mesh for this kind of task (use what helps, ignore what does not):\n" + pack.strip())
    parts.append(task["prompt"].strip())
    parts.append(FORMAT[task["kind"]])
    return "\n\n---\n\n".join(parts)


def run(agent: dict, task: dict, pack: str | None, workdir: Path, timeout: float = 900, max_turns: int = 30,
        budget_usd: float = 3.0) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    key = "a" + str(abs(hash(agent["name"])) % 10**8)
    spec = {key: {"description": agent["description"] or "agent", "prompt": agent["prompt"], "tools": EVAL_TOOLS.split(",")}}
    if agent.get("model"):
        spec[key]["model"] = agent["model"]
    tmp = Path(tempfile.mkdtemp(prefix="mt-", dir=workdir))
    (tmp / "agents.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp / "mcp.json").write_text('{"mcpServers":{}}', encoding="utf-8")
    cmd = [CLAUDE, "-p", "--no-session-persistence", "--setting-sources", "project", "--strict-mcp-config",
           "--mcp-config", str(tmp / "mcp.json"), "--disable-slash-commands", "--exclude-dynamic-system-prompt-sections",
           "--tools", EVAL_TOOLS, "--allowedTools", *ALLOWED, "--agents", str(tmp / "agents.json"), "--agent", key,
           "--output-format", "json", "--max-turns", str(max_turns), "--max-budget-usd", str(budget_usd)]
    t0 = time.time()
    rec = {"agent": agent["name"], "task": task["id"], "field": task["field"], "pack": bool(pack), "ok": False,
           "cost": 0.0, "secs": 0.0, "error": None, "text": ""}
    try:
        r = subprocess.run(cmd, input=build_prompt(task, pack), capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout, cwd=str(tmp), env=clean_env())
        try:
            d = json.loads(r.stdout)
            rec["text"] = d.get("result") or ""
            rec["cost"] = float(d.get("total_cost_usd") or 0)
            if d.get("is_error"):
                rec["error"] = (d.get("subtype") or "error") + ": " + rec["text"][:200]
        except Exception:
            rec["error"] = "unparsed output rc=%s: %s %s" % (r.returncode, r.stdout[-300:], r.stderr[-300:])
    except subprocess.TimeoutExpired:
        rec["error"] = "timeout"
    finally:
        rec["secs"] = round(time.time() - t0, 1)
        shutil.rmtree(tmp, ignore_errors=True)
    return rec
