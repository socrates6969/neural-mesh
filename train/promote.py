"""Automatic promotion of knowledge packs into the live neural-mesh registry (no manual step).

A pack goes live only when ALL of these hold:
  1. training marked it "promoted": significant held-out gain (pooled exact McNemar p < 0.05, more pack-only wins);
  2. it is at most MAX_WORDS words (the content screen was removed at the user's request);
  3. it is signed with an HMAC (key in ~/.claude/mesh-pack.key, created on first use, owner-only ACL). The live hook
     re-checks the signature, so a pack edited after promotion is ignored.
Every promotion and every screen rejection is appended to promotions.jsonl next to the registry.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from pathlib import Path

from hooks.mesh_assist import mac

KEY_PATH = Path(os.environ.get("NEURAL_MESH_PACK_KEY", str(Path.home() / ".claude" / "mesh-pack.key")))
REGISTRY = Path(os.environ.get("NEURAL_MESH_PACKS", r"C:\Users\mariu\neural-mesh\data\lifelong\mesh\packs.json"))
MAX_WORDS = 900


def _key() -> bytes:
    """Signing key: created once, atomically, owner-only; fail closed if permissions cannot be restricted."""
    if not KEY_PATH.exists():
        KEY_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
        try:
            os.write(fd, secrets.token_bytes(32))
        finally:
            os.close(fd)
        if os.name == "nt":
            r = subprocess.run(["icacls", str(KEY_PATH), "/inheritance:r", "/grant:r", os.environ.get("USERNAME", "") + ":F"],
                               capture_output=True)
            if r.returncode != 0:
                KEY_PATH.unlink()
                raise RuntimeError("could not restrict permissions on the pack signing key")
    if os.name != "nt" and os.stat(KEY_PATH).st_mode & 0o077:
        raise RuntimeError("pack signing key is readable by other users; refusing to use it")
    return KEY_PATH.read_bytes()


def _log(rec: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY.with_name("promotions.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def publish(agent: str, field: str, st: dict) -> str:
    """Publish a training status entry if it qualifies. Returns 'published', 'rejected: ...' or 'not promoted'."""
    if st.get("status") != "promoted" or not st.get("current"):
        return "not promoted"
    pack = st["current"]
    why = ["too long"] if len(pack.split()) > MAX_WORDS else []
    rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "agent": agent, "field": field, "version": st.get("version"),
           "pooled": st.get("pooled"), "sha256": hashlib.sha256(pack.encode("utf-8")).hexdigest()}
    if why:
        _log({**rec, "result": "rejected", "why": why})
        return "rejected: " + ", ".join(why)
    reg = json.loads(REGISTRY.read_text(encoding="utf-8")) if REGISTRY.exists() else {}
    reg.setdefault(agent, {})[field] = {"pack": pack, "status": "promoted", "pooled": st.get("pooled"),
                                       "version": st.get("version"), "sha256": rec["sha256"],
                                       "approved_mac": mac(_key(), agent, field, pack)}
    tmp = REGISTRY.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1), encoding="utf-8")
    tmp.replace(REGISTRY)
    _log({**rec, "result": "published"})
    return "published"