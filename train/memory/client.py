"""Client used by the hook and the runner: ask the warm daemon; if it is down, start it (detached, at most once per
SPAWN_COOLDOWN_S) and answer this call in-process. Imports only the standard library on the warm path."""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import config as C

SPAWN_COOLDOWN_S = 30.0
ROOT = str(Path(__file__).resolve().parents[2])


def _request(req: dict, timeout: float) -> dict | None:
    from .daemon import key
    req = dict(req, key=key())
    try:
        with socket.create_connection(("127.0.0.1", C.PORT), timeout=min(timeout, 0.25)) as s:
            s.settimeout(timeout)
            s.sendall((json.dumps(req) + "\n").encode("utf-8"))
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(65536)
                if not chunk:
                    break
                buf += chunk
        out = json.loads(buf)
        return out if out.get("ok") else None
    except (OSError, ValueError):
        return None


def spawn() -> bool:
    """Start the daemon detached unless disabled or recently attempted. Returns True if a start was attempted."""
    if os.environ.get("NEURAL_MESH_MEMORY_DAEMON", "on") == "off":
        return False
    stamp = C.MEM_DIR / ".daemon-spawn"
    try:
        if time.time() - stamp.stat().st_mtime < SPAWN_COOLDOWN_S:
            return False
    except OSError:
        pass
    try:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(time.time()), encoding="utf-8")
        flags = 0
        if os.name == "nt":
            flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | NO_WINDOW
        subprocess.Popen([sys.executable, "-m", "train.memory", "serve"], cwd=ROOT, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True, creationflags=flags)
        return True
    except OSError:
        return False


def lookup(text: str, field: str | None = None, k: int = 2, exclude=(), log_task: str | None = None,
           timeout: float = 1.0, budget: int = C.INJECT_TOKENS, in_process: bool = True) -> str:
    """Rendered memory block for `text` ("" when nothing passes the gate)."""
    req = {"op": "lookup", "text": text[:C.MAX_PROBLEM * 2], "field": field, "k": k, "exclude": list(exclude or ()),
           "log_task": log_task, "budget": budget}
    out = _request(req, timeout)
    if out is not None:
        return out.get("text") or ""
    spawn()
    if not in_process:
        return ""
    from . import api
    return api.render(api.retrieve(text, field=field, k=k, exclude=exclude, log_task=log_task), budget)


def observe(card_ids, passed: bool, timeout: float = 1.0) -> None:
    if _request({"op": "observe", "cards": list(card_ids), "passed": bool(passed)}, timeout) is None:
        from . import api
        api.observe(card_ids, passed)

def embed(texts: list[str], timeout: float = 1.0):
    """MiniLM vectors from the warm daemon as an (n, 384) float32 array, or None if it is down (it is then started
    for the next caller). Used by the workspace so read()/post() avoid loading the encoder in every process."""
    import base64
    import numpy as np
    out = _request({"op": "embed", "texts": list(texts)}, timeout)
    if out is None:
        spawn()
        return None
    return np.stack([np.frombuffer(base64.b64decode(b), dtype="<f4") for b in out["vecs"]]).astype(np.float32) \
        if out["vecs"] else np.zeros((0, 384), np.float32)
