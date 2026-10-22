"""Minimal MCP stdio server with one tool, run_python, that executes code in a network-less container.

Evaluation agents get this tool INSTEAD of a shell, so model-written code never runs on the host.
Launched by claude via --mcp-config with the run directory as argv[1]; only that directory is mounted (at /w).
Protocol: newline-delimited JSON-RPC 2.0 (initialize, tools/list, tools/call); no third-party packages.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

IMAGE = os.environ.get("MESHTRAIN_GRADER_IMAGE", "meshtrain-grader:1")
MAX_OUT = 20000
TOOL = {
    "name": "run_python",
    "description": "Run a Python 3.12 script in an isolated sandbox (numpy + standard library, no network, 1 CPU, 1 GB). "
                   "The working directory /w is your task folder (files you Write are visible there). Returns stdout, stderr "
                   "and the exit code. Use print() to see results.",
    "inputSchema": {"type": "object", "properties": {
        "code": {"type": "string", "description": "Python source to run"},
        "timeout_s": {"type": "integer", "description": "seconds before the run is killed (max 300, default 120)"}},
        "required": ["code"]},
}


def run_code(workdir: str, code: str, timeout_s: int) -> tuple[str, bool]:
    timeout_s = max(5, min(int(timeout_s or 120), 300))
    fn = "_run_%s.py" % uuid.uuid4().hex[:8]
    path = os.path.join(workdir, fn)
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    name = "mts-" + uuid.uuid4().hex[:12]
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "1", "--memory", "1g",
           "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp:rw,size=64m", "-v", workdir + ":/w", "-w", "/w",
           "-e", "PYTHONHASHSEED=0", "-e", "PYTHONUNBUFFERED=1", IMAGE, "python", fn]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_s + 20)
        out = "exit code %d\n--- stdout ---\n%s\n--- stderr ---\n%s" % (r.returncode, r.stdout[-MAX_OUT:], r.stderr[-MAX_OUT // 4:])
        return out, r.returncode != 0
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        return "killed: exceeded %d s" % timeout_s, True
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def main() -> None:
    workdir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        mid, method = msg.get("id"), msg.get("method")
        if mid is None:  # notification
            continue
        if method == "initialize":
            ver = (msg.get("params") or {}).get("protocolVersion", "2024-11-05")
            res = {"protocolVersion": ver, "capabilities": {"tools": {}}, "serverInfo": {"name": "sandbox", "version": "1.0"}}
        elif method == "tools/list":
            res = {"tools": [TOOL]}
        elif method == "tools/call":
            p = msg.get("params") or {}
            if p.get("name") != "run_python":
                err = {"code": -32602, "message": "unknown tool"}
                sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "error": err}) + "\n")
                sys.stdout.flush()
                continue
            a = p.get("arguments") or {}
            text, is_err = run_code(workdir, str(a.get("code", "")), a.get("timeout_s", 120))
            res = {"content": [{"type": "text", "text": text}], "isError": is_err}
        elif method == "ping":
            res = {}
        else:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}}) + "\n")
            sys.stdout.flush()
            continue
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": mid, "result": res}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()