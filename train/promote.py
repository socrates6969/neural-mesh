"""Automatic promotion of knowledge packs into the live neural-mesh registry (no manual step).

A pack goes live only when ALL of these hold:
  1. training marked it "promoted": significant held-out gain (pooled exact McNemar p < 0.05, more pack-only wins);
  2. it passes the content screen below (no links, shell/install/network instructions, credentials, file paths,
     or instruction-override phrasing);
  3. it is signed with an HMAC (key in ~/.claude/mesh-pack.key, created on first use, owner-only ACL). The live hook
     re-checks the signature, so a pack edited after promotion is ignored.
Every promotion and every screen rejection is appended to promotions.jsonl next to the registry.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import time
from pathlib import Path

from hooks.mesh_assist import mac

KEY_PATH = Path(os.environ.get("NEURAL_MESH_PACK_KEY", str(Path.home() / ".claude" / "mesh-pack.key")))
REGISTRY = Path(os.environ.get("NEURAL_MESH_PACKS", r"C:\Users\mariu\neural-mesh\data\lifelong\mesh\packs.json"))
MAX_WORDS = 900

BLOCK = [
    (r"https?://|www\.|\]\(|\b[a-z0-9-]+\.(com|net|org|io|dev|ru|cn|xyz|sh)\b", "link or domain"),
    (r"\b(curl|wget|invoke-webrequest|iwr|powershell|cmd\.exe|bash|chmod|sudo)\b|\bsh\s+-c\b|\brm\s+-r|\bdel\s+[/\\]", "shell command"),
    (r"\b(pip|npm|npx|uv|conda|apt|brew|cargo|go)\s+(install|add|get|i)\b", "package install"),
    (r"\b(subprocess|os\.system|os\.popen|eval\(|exec\(|__import__|socket|requests\.|urllib|http\.client)", "code execution / network API"),
    (r"\b(api[_ -]?key|secret|password|passwd|token|credential|ssh)\b|\.env\b", "credential reference"),
    (r"\b[a-z]:[\\/]|~/|/etc/|/home/|/users/|%appdata%|\$home", "file path"),
    (r"ignore (all |any |the )?(previous|prior|above|earlier)|disregard|system prompt|you are now|override|jailbreak|do not tell",
     "instruction override"),
    (r"\b(fetch|download|upload|send|post) (the |a |this |your )?(file|data|result|output|url|page)", "data movement"),
]


def screen(pack: str) -> list[str]:
    """Return the reasons a pack is unsafe to inject (empty list = passes)."""
    low = pack.lower()
    why = [label for pat, label in BLOCK if re.search(pat, low)]
    if len(pack.split()) > MAX_WORDS:
        why.append("too long")
    # allowlist on top of the denylist: printable ASCII + newline/tab only, so homoglyphs, zero-width and bidi
    # characters cannot be used to slip a blocked word past the patterns above
    if any(not (32 <= ord(c) < 127 or ord(c) in (9, 10)) for c in pack):
        why.append("non-ASCII or control characters")
    return why


def _key() -> bytes:
    if not KEY_PATH.exists():
        KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        KEY_PATH.write_bytes(secrets.token_bytes(32))
        if os.name == "nt":
            subprocess.run(["icacls", str(KEY_PATH), "/inheritance:r", "/grant:r", os.environ.get("USERNAME", "") + ":F"],
                           capture_output=True)
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
    why = screen(pack)
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