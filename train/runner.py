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
import sys
import tempfile
import threading
import time
from pathlib import Path

CLAUDE = shutil.which("claude") or "claude"
EVAL_TOOLS = "Read,Write,Edit"
SANDBOX_TOOL = "mcp__sandbox__run_python"
# No shell: code runs only through the sandbox MCP tool (network-less container). File tools stay in the run dir.
ALLOWED = ["Read(./**)", "Write(./**)", "Edit(./**)", SANDBOX_TOOL]
SANDBOX_SERVER = str(Path(__file__).with_name("sandbox_mcp.py"))
TOOL_NOTE = ("Tools: there is no shell. To run Python (numpy + standard library, no network) call the run_python tool; "
             "Read/Write/Edit work only inside your task folder.")
# The answer line / code block stays the graded output; the protocol's closing ```json block comes after it.
FORMAT = {
    "int": "Give the final answer on its own line: ANSWER: <integer> (then the closing JSON block, same answer).",
    "numeric": "Give the final answer on its own line: ANSWER: <number> (then the closing JSON block, same answer).",
    "mcq": ("Give the final answer on its own line: ANSWER: <letter(s)>, e.g. ANSWER: B or ANSWER: ACD (then the closing "
            "JSON block, same answer)."),
    "code": ("Put the complete final solution in a single ```python block (it is run against hidden tests). In the closing "
             "JSON block set \"answer\" to a one-line summary; do not repeat the code."),
    "stdio": ("Write a complete Python 3 program that reads standard input and writes standard output. Put the final "
              "program in a single ```python block; it is run against hidden tests with the time limit above (CPython). "
              "In the closing JSON block set \"answer\" to a one-line summary; do not repeat the code."),
}
# Bench field -> agent-kit protocol mode (math, physics, coding, computer-science, cybersecurity, neural-networks).
PROTOCOL_MODE = {"math": "math", "physics": "physics", "coding": "code", "computer-science": "cs",
                 "cybersecurity": "cyber", "neural-networks": "ml"}


def protocol_block(task: dict) -> str:
    """The agent-kit protocol for a bench task; "" with MESHTRAIN_PROTOCOL=off (for runs comparable to pre-kit rounds).
    It depends only on the task, never on the arm, so the pack and base arms of a round see identical prompts."""
    if os.environ.get("MESHTRAIN_PROTOCOL", "on") == "off":
        return ""
    try:
        from agentkit import protocol
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from agentkit import protocol
    field = str(task.get("field") or "")
    mode = PROTOCOL_MODE.get(field) or protocol.detect_mode(task.get("prompt", ""), fields=[field])
    return protocol.render(mode)


_FM = re.compile(r"^\ufeff?---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)
_lock = threading.Lock()

# Child processes get only what they need to run and to authenticate Claude Code (OAuth lives in ~/.claude);
# API tokens for other services in the parent environment (GitHub, AWS, Pinecone, ...) are NOT passed on.
_ENV_KEEP = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "USERPROFILE", "HOME", "HOMEDRIVE",
             "HOMEPATH", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
             "LANG", "PYTHONUTF8", "PYTHONIOENCODING", "CLAUDE_CONFIG_DIR"}


QUOTA_RE = re.compile(r"hit your (weekly|daily|usage|session) limit|usage limit reached|rate limit(ed)? (exceeded|reached)|"
                      r"credit balance is too low|limit . resets", re.I)


def is_quota(text: str) -> bool:
    """Plan/usage-limit replies are not answers: they must never be graded as a wrong answer."""
    return bool(text) and len(text) < 600 and bool(QUOTA_RE.search(text))


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


def method_cards(task: dict) -> str:
    """Shared method cards (train.memory.client.lookup; warm daemon first). Depends only on the task, so both arms get
    the same block; log_task records which cards were shown so their usefulness can be learned. MESHTRAIN_CARDS=off."""
    if os.environ.get("MESHTRAIN_CARDS", "on") == "off":
        return ""
    try:
        from train.memory import client
        return client.lookup(task["prompt"], field=task.get("field"), k=2, exclude={task["id"]}, log_task=task["id"],
                             timeout=10.0) or ""
    except Exception:
        return ""


def build_prompt(task: dict, pack: str | None) -> str:
    parts = []
    proto = protocol_block(task)
    if proto:
        parts.append(proto)
    if pack:
        parts.append("Knowledge from the neural mesh for this kind of task (use what helps, ignore what does not):\n" + pack.strip())
    if os.environ.get("MESHTRAIN_EXPERIENCE", "on") != "off":
        try:
            from . import experience
            mem = experience.render(experience.retrieve(task["prompt"], field=task.get("field"), k=2, min_sim=0.45,
                                                        exclude={task["id"]}))
            if mem:
                parts.append(mem)
        except Exception:
            pass
    cards = method_cards(task)
    if cards:
        parts.append(cards)
    parts.append(task["prompt"].strip())
    parts.append(FORMAT[task["kind"]] + "\n" + TOOL_NOTE)
    return "\n\n---\n\n".join(parts)


def run(agent: dict, task: dict, pack: str | None, workdir: Path, timeout: float = 900, max_turns: int = 30,
        budget_usd: float = 3.0) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    key = "a" + str(abs(hash(agent["name"])) % 10**8)
    spec = {key: {"description": agent["description"] or "agent", "prompt": agent["prompt"], "tools": EVAL_TOOLS.split(",") + [SANDBOX_TOOL]}}
    if agent.get("model"):
        spec[key]["model"] = agent["model"]
    tmp = Path(tempfile.mkdtemp(prefix="mt-", dir=workdir))
    (tmp / "agents.json").write_text(json.dumps(spec), encoding="utf-8")
    mcp = {"mcpServers": {"sandbox": {"command": sys.executable, "args": [SANDBOX_SERVER, str(tmp)]}}}
    (tmp / "mcp.json").write_text(json.dumps(mcp), encoding="utf-8")
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
            if is_quota(rec["text"]):
                rec["error"] = "quota: " + rec["text"][:200]
            elif d.get("is_error"):
                rec["error"] = (d.get("subtype") or "error") + ": " + rec["text"][:200]
        except Exception:
            rec["error"] = "unparsed output rc=%s: %s %s" % (r.returncode, r.stdout[-300:], r.stderr[-300:])
    except subprocess.TimeoutExpired:
        rec["error"] = "timeout"
    finally:
        rec["secs"] = round(time.time() - t0, 1)
        shutil.rmtree(tmp, ignore_errors=True)
    return rec
