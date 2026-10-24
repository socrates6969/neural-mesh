"""Trust rules for Global Workspace entries: who may vouch for an entry and how that is proven.

verified_by allowlist (anything else is untrusted across sessions):
  * "pipeline:<name>" for names in PIPELINES, only with a valid pipeline_sig: HMAC-SHA256 keyed with
    ~/.claude/mesh-pipeline.key, written by the trusted writers (train.memory.ingest_training, the card builder) over
    task_id|agent|kind|content|verified_by of the STORED (scrubbed) entry - see pipeline_sign_entry();
  * "human:<user>" only with a valid approval_sig: HMAC-SHA256 keyed with ~/.claude/mesh-approve.key over
    id|task_id|agent|kind|content|verified_by (agentkit.approve writes it after a human-presence check);
  * method cards (source="card") only with provenance "pipeline:grader" AND a valid pipeline_sig over
    card id|field|text (pipeline_sign_card()).
No verified_by, unknown pipelines, unsigned or wrongly signed rows are untrusted. Missing or malformed keys fail
closed. Threat model and limits: agentkit/SECURITY.md (same-user code execution can read these keys).
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
SEP = "\x1f"


def key_path() -> Path:
    return Path(os.environ.get("NEURAL_MESH_APPROVE_KEY", str(Path.home() / ".claude" / "mesh-approve.key")))


def pipeline_key_path() -> Path:
    return Path(os.environ.get("NEURAL_MESH_PIPELINE_KEY", str(Path.home() / ".claude" / "mesh-pipeline.key")))


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


def _load(p: Path, create: bool) -> bytes | None:
    """A 32-byte key, or None (missing/malformed). create=True makes it atomically (O_EXCL) and owner-only."""
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
        return _load(p, create=False)
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


def load_key(create: bool = False) -> bytes | None:
    """The human-approval key (~/.claude/mesh-approve.key)."""
    return _load(key_path(), create)


def load_pipeline_key(create: bool = False) -> bytes | None:
    """The pipeline key (~/.claude/mesh-pipeline.key); only trusted writers create it (create=True)."""
    return _load(pipeline_key_path(), create)


def _mac(key: bytes, parts) -> str:
    return hmac.new(key, SEP.join(str(p if p is not None else "") for p in parts).encode("utf-8"),
                    hashlib.sha256).hexdigest()


# --- human approvals ---------------------------------------------------------------------------------------------
def sign(entry: dict, by: str, key: bytes) -> str:
    return _mac(key, (entry.get("id"), entry.get("task_id"), entry.get("agent"), entry.get("kind"),
                      entry.get("content"), by))


def human_ok(entry: dict, key: bytes | None) -> bool:
    by = str(entry.get("verified_by") or "")
    sig = entry.get("approval_sig")
    if not (key and by.startswith("human:") and len(by) > len("human:") and isinstance(sig, str) and sig):
        return False
    return hmac.compare_digest(sig, sign(entry, by, key))


# --- pipeline writers (train.memory.ingest_training, the card builder) --------------------------------------------
def pipeline_sign_entry(task_id, agent, kind, content, verified_by, key: bytes) -> str:
    """pipeline_sig for a workspace entry; `content` must be the text as STORED (after scrubbing)."""
    return _mac(key, ("entry", task_id, agent, kind, content, verified_by))


def pipeline_sign_card(card_id, field, text, key: bytes) -> str:
    """pipeline_sig for a method card (id as served, e.g. "card:..."; text as served)."""
    return _mac(key, ("card", card_id, field, text, CARD_PROVENANCE))


def _sig_ok(sig, want) -> bool:
    return isinstance(sig, str) and bool(sig) and hmac.compare_digest(sig, want)


def pipeline_ok(entry: dict, pkey: bytes | None) -> bool:
    if not pkey:
        return False
    sig = entry.get("pipeline_sig")
    if entry.get("source") == "card":
        if (entry.get("provenance") or entry.get("verified_by")) != CARD_PROVENANCE:
            return False
        return _sig_ok(sig, pipeline_sign_card(entry.get("id"), entry.get("field"), entry.get("content"), pkey))
    by = str(entry.get("verified_by") or "")
    if by not in PIPELINES:
        return False
    return _sig_ok(sig, pipeline_sign_entry(entry.get("task_id"), entry.get("agent"), entry.get("kind"),
                                            entry.get("content"), by, pkey))


def trusted(entry: dict, key: bytes | None = None, pipeline_key: bytes | None = None) -> bool:
    """May this entry cross a session boundary (be shown to agents of other sessions)?"""
    if entry.get("source") == "card":
        return pipeline_ok(entry, pipeline_key)
    if entry.get("verified") is not True:
        return False
    by = str(entry.get("verified_by") or "")
    if by in PIPELINES:
        return pipeline_ok(entry, pipeline_key)
    return human_ok(entry, key)


def verified_by(entry: dict, key: bytes | None = None, pipeline_key: bytes | None = None) -> str:
    """Display label; only an allowlisted voucher with a valid signature is shown as a verifier."""
    if entry.get("source") == "card":
        return CARD_PROVENANCE if trusted(entry, key, pipeline_key) else "unverified (card without valid grader signature)"
    if trusted(entry, key, pipeline_key):
        return str(entry.get("verified_by"))
    if entry.get("verified") is True:
        return "unverified (unrecognized or unsigned verifier)"
    return "unverified (agent claim)" if entry.get("claimed_verified") else "unverified"