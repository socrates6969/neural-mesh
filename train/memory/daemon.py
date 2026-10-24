"""Tiny localhost memory daemon: keeps the MiniLM encoder, cross-encoder and card index loaded so a lookup costs
milliseconds instead of ~1 s of model loading. Started on demand by the client, exits after IDLE_EXIT_S idle.

Protocol: one JSON line per TCP connection on 127.0.0.1:PORT, answered by one JSON line. Every request must carry the
shared key from ~/.claude/mesh-memory.key (random, created on first use), so other local processes cannot read the
memory or forge outcome feedback through the daemon. Ops: ping | lookup | observe."""
from __future__ import annotations

import base64
import json
import os
import secrets
import socketserver
import threading
import time
import hmac

from . import config as C

MAX_REQ = 256 * 1024


def key() -> str:
    """The shared secret; created (random 32 bytes hex) on first use."""
    try:
        k = C.KEY.read_text(encoding="utf-8").strip()
        if k:
            return k
    except OSError:
        pass
    C.KEY.parent.mkdir(parents=True, exist_ok=True)
    k = secrets.token_hex(32)
    try:
        fd = os.open(C.KEY, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(k)
        return k
    except FileExistsError:
        return C.KEY.read_text(encoding="utf-8").strip()


def handle(req: dict, secret: str) -> dict:
    """Dispatch one request (pure function of the request; also used by tests)."""
    if not hmac.compare_digest(str(req.get("key", "")), secret):
        return {"ok": False, "error": "auth"}
    from . import api
    op = req.get("op")
    if op == "ping":
        return {"ok": True, "pid": os.getpid()}
    if op == "lookup":
        hits = api.retrieve(str(req.get("text") or ""), field=req.get("field") or None, k=int(req.get("k") or 2),
                            exclude=req.get("exclude") or (), mode=req.get("mode") or None, log_task=req.get("log_task") or None)
        return {"ok": True, "text": api.render(hits, int(req.get("budget") or C.INJECT_TOKENS)),
                "hits": [{"score": round(s, 4), "id": c["id"], "task": c.get("task")} for s, c in hits]}
    if op == "observe":
        api.observe(req.get("cards") or [], bool(req.get("passed")))
        return {"ok": True}
    if op == "embed":
        from .. import experience
        texts = [str(t)[:C.MAX_PROBLEM] for t in (req.get("texts") or [])][:64]
        vecs = experience.encoder().encode(texts) if texts else []
        return {"ok": True, "vecs": [base64.b64encode(v.astype("<f4").tobytes()).decode("ascii") for v in vecs]}
    return {"ok": False, "error": "op"}


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = False


def _code_stamp() -> float:
    from pathlib import Path
    return max(p.stat().st_mtime for p in Path(__file__).resolve().parent.glob("*.py"))


def serve(port: int = C.PORT, idle_s: float = C.IDLE_EXIT_S) -> None:
    """Serve until idle for idle_s, or until the memory code changes on disk (the next client starts a fresh one)."""
    secret = key()
    stamp = _code_stamp()
    from . import api
    api.warm()
    last = [time.monotonic()]

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            last[0] = time.monotonic()
            try:
                line = self.rfile.readline(MAX_REQ)
                out = handle(json.loads(line), secret)
            except Exception as e:  # never let a bad request kill the daemon
                out = {"ok": False, "error": type(e).__name__}
            self.wfile.write((json.dumps(out) + "\n").encode("utf-8"))
            last[0] = time.monotonic()

    srv = _Server(("127.0.0.1", port), H)

    def watchdog():
        while True:
            time.sleep(min(10.0, idle_s / 4 + 0.05))
            try:
                stale = _code_stamp() != stamp
            except OSError:
                stale = False
            if stale or time.monotonic() - last[0] > idle_s:
                srv.shutdown()
                return
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        srv.serve_forever(poll_interval=0.5)
    finally:
        srv.server_close()