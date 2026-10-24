"""Shared helpers: output capping, argument coercion and the network-less Docker sandbox runner."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

MAX_OUT = 6000
IMAGE = os.environ.get("MESH_TOOLS_IMAGE", "meshtrain-grader:2")
SANDBOX_DIR = Path(__file__).resolve().parent / "sandbox"
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def cap(text: str, limit: int = MAX_OUT) -> str:
    """Trim text to at most `limit` chars, keeping the head and noting the cut."""
    text = _CTRL.sub("", text or "")
    if len(text) <= limit:
        return text
    note = "\n... [truncated %d chars]" % (len(text) - limit)
    return text[: limit - len(note)] + note


def clean(text, limit: int | None = None) -> str:
    """Collapse whitespace and strip control chars from external text; optionally shorten it."""
    s = " ".join(_CTRL.sub(" ", str(text or "")).split())
    if limit and len(s) > limit:
        s = s[: limit - 3].rstrip() + "..."
    return s


def as_json(value, name: str):
    """Accept either a JSON string or an already-decoded value for a *_json argument."""
    if isinstance(value, (dict, list)):
        return value
    if value is None or value == "":
        raise ValueError("%s is required" % name)
    try:
        return json.loads(value)
    except ValueError as e:
        raise ValueError("%s is not valid JSON: %s" % (name, e)) from None


def run_sandbox(script: str, args: dict | None = None, mounts: list[tuple[str, str]] | None = None,
                timeout_s: int = 120, memory: str = "1g") -> tuple[str, str, int]:
    """Run a Python script in the network-less grader container.

    The script and args.json sit in a fresh temp dir mounted read-only at /w; `mounts` are extra
    (host_path, container_path) files, also read-only. Same isolation flags as train/sandbox_mcp.py.
    Returns (stdout, stderr, exit code); exit code 124 means killed on timeout.
    """
    timeout_s = max(5, min(int(timeout_s or 120), 300))
    work = tempfile.mkdtemp(prefix="meshmcp-")
    try:
        Path(work, "main.py").write_text(script, encoding="utf-8")
        Path(work, "args.json").write_text(json.dumps(args or {}), encoding="utf-8")
        for f in ("stats_impl.py", "causal_impl.py"):
            shutil.copy(SANDBOX_DIR / f, Path(work, f))
        name = "mesh-" + uuid.uuid4().hex[:12]
        cmd = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", "1", "--memory", memory,
               "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
               "--mount", "type=bind,src=%s,dst=/w,readonly" % work, "-w", "/w"]
        for host, cont in mounts or []:
            cmd += ["--mount", "type=bind,src=%s,dst=%s,readonly" % (host, cont)]
        cmd += ["-e", "PYTHONHASHSEED=0", "-e", "PYTHONUNBUFFERED=1", "-e", "OMP_NUM_THREADS=1",
                "-e", "OPENBLAS_NUM_THREADS=1", "-e", "PYTHONDONTWRITEBYTECODE=1", IMAGE, "python", "main.py"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout_s + 30)
            return r.stdout, r.stderr, r.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
            return "", "killed: exceeded %d s" % timeout_s, 124
        except FileNotFoundError:
            return "", "docker not found on PATH", 127
    finally:
        shutil.rmtree(work, ignore_errors=True)


def docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
