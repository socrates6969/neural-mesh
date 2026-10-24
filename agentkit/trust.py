"""Trust rules for Global Workspace entries: who may vouch for an entry and how a human approval is proven.

verified_by allowlist:
  * "pipeline:<name>" for names in PIPELINES only (out-of-band pipelines that post grader-verified results);
  * "human:<user>" only with a valid approval_sig: HMAC-SHA256, keyed with ~/.claude/mesh-approve.key, over the entry's
    id, task, agent, kind, content and verified_by (so an edited entry or a copied signature does not verify).
Everything else is untrusted: no verified_by, unknown pipelines, unsigned or wrongly signed human approvals, and legacy
verified=True rows. A missing or malformed key fails closed (no human approval verifies, approve refuses).
Method cards are trusted only when their provenance is "pipeline:grader".
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import subprocess
from pathlib import Path

PIPELINES = frozenset({"pipeline:grader"})
CARD_PROVENANCE = "pipeline:grader"
KEY_BYTES = 32


def key_path() -> Path:
    return Path(os.environ.get("NEURAL_MESH_APPROVE_KEY", str(Path.home() / ".claude" / "mesh-approve.key")))


def _restrict_owner_only(p: Path) -> None:
    """Owner-only ACL; raises on failure (the caller then deletes the key: fail closed)."""
    if os.name == "nt":
        user = os.environ.get("USERNAME") or ""
        if not user:
            raise OSError("no USERNAME for the ACL")
        r = subprocess.run(["icacls", str(p), "/inheritance:r", "/grant:r", user + ":F"], capture_output=True,
                           text=True, timeout=30)
        if r.returncode != 0:
            raise OSError("icacls failed: " + (r.stderr or r.stdout).strip()[:200])
    else:
        os.chmod(p, 0o600)


def load_key(create: bool = False) -> bytes | None:
    """The approval key, or None (missing/malformed). create=True makes it atomically (O_EXCL) and owner-only."""
    p = key_path()
    try:
        k = p.read_bytes()
        return k if len(k) == KEY_BYTES else None
    except FileNotFoundError:
        if not create:
            return None
    except OSError:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    k = secrets.token_bytes(KEY_BYTES)
    try:
        fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError:                      # another process won the race: use its key
        return load_key(create=False)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(k)
            fh.flush()
            os.fsync(fh.fileno())
        _restrict_owner_only(p)
    except Exception:
        try:
            p.unlink()
        except OSError:
            pass
        return None
    return k


def _msg(entry: dict, by: str) -> bytes:
    parts = [str(entry.get("id") or ""), str(entry.get("task_id") or ""), str(entry.get("agent") or ""),
             str(entry.get("kind") or ""), str(entry.get("content") or ""), by]
    return "\x1f".join(parts).encode("utf-8")


def sign(entry: dict, by: str, key: bytes) -> str:
    return hmac.new(key, _msg(entry, by), hashlib.sha256).hexdigest()


def human_ok(entry: dict, key: bytes | None) -> bool:
    by = str(entry.get("verified_by") or "")
    sig = entry.get("approval_sig")
    if not (key and by.startswith("human:") and len(by) > len("human:") and isinstance(sig, str) and sig):
        return False
    return hmac.compare_digest(sig, sign(entry, by, key))


def trusted(entry: dict, key: bytes | None = None) -> bool:
    """May this entry cross a session boundary (be shown to agents of other sessions)?"""
    if entry.get("source") == "card":
        return (entry.get("provenance") or entry.get("verified_by")) == CARD_PROVENANCE
    if entry.get("verified") is not True:
        return False
    by = str(entry.get("verified_by") or "")
    if by in PIPELINES:
        return True
    return human_ok(entry, key)


def verified_by(entry: dict, key: bytes | None = None) -> str:
    """Display label; only an allowlisted, checkable voucher is shown as a verifier."""
    if entry.get("source") == "card":
        return CARD_PROVENANCE if trusted(entry, key) else "unverified (card without grader provenance)"
    if trusted(entry, key):
        return str(entry.get("verified_by"))
    if entry.get("verified") is True:
        return "unverified (unrecognized or unsigned verifier)"
    return "unverified (agent claim)" if entry.get("claimed_verified") else "unverified"