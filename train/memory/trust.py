"""Pipeline signatures: trusted writers (train.memory.ingest_training, the card builder) sign what they write, so a
reader (agentkit/trust.py) can tell grader-pipeline knowledge from anything else that merely claims it.

  key      ~/.claude/mesh-pipeline.key (env MESH_PIPELINE_KEY): 32 random bytes, created atomically (O_EXCL + hard
           link), restricted to the owner; any failure raises (fail closed - nothing is written unsigned-as-signed)
  MAC      HMAC-SHA256, lowercase hex, over the UTF-8 payload (fields joined with "\x1f", unit separator):
             workspace entry:  "mesh-pipeline-v1", "entry", str(id), task_id, kind, content, verified_by
             method card:      "mesh-pipeline-v1", "card",  card id, task, text, "pipeline:grader"
           content/text is the stored scrubbed plaintext (what get()/read() return), never ciphertext.

  python -m train.memory.trust resign      sign all pipeline:* workspace entries and all cards (migration)
  python -m train.memory.trust verify      count valid / invalid / unsigned
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

KEY = Path(os.environ.get("MESH_PIPELINE_KEY", str(Path.home() / ".claude" / "mesh-pipeline.key")))
VERSION = "mesh-pipeline-v1"
SEP = "\x1f"


def restrict_owner_only(path: Path) -> None:
    if os.name != "nt":
        os.chmod(path, 0o600)
        return
    user = os.environ.get("USERNAME") or os.getlogin()
    r = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", user + ":F"], capture_output=True, text=True)
    if r.returncode != 0:
        raise PermissionError("cannot restrict key to the owner: " + (r.stderr or r.stdout).strip()[:200])


def load_key(path: Path, create: bool) -> bytes:
    """32-byte secret at `path`; created atomically and owner-only when create=True. Raises on any problem."""
    path = Path(path)
    try:
        k = path.read_bytes()
    except FileNotFoundError:
        if not create:
            raise
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".%d.tmp" % os.getpid())
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(secrets.token_bytes(32))
            restrict_owner_only(tmp)
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        k = path.read_bytes()
    if len(k) != 32:
        raise ValueError("key at %s must be 32 bytes" % path)
    return k


def _mac(key: bytes, parts) -> str:
    return hmac.new(key, SEP.join([VERSION] + [str(p) for p in parts]).encode("utf-8"), hashlib.sha256).hexdigest()


def entry_payload(entry_id, task_id, kind, content, verified_by) -> list:
    return ["entry", entry_id, task_id, kind, content, verified_by]


def card_payload(card_id, task, text) -> list:
    return ["card", card_id, task, text, "pipeline:grader"]


def sign_entry(entry_id, task_id, kind, content, verified_by, key: bytes | None = None) -> str:
    return _mac(key or load_key(KEY, create=True), entry_payload(entry_id, task_id, kind, content, verified_by))


def sign_card(card: dict, key: bytes | None = None) -> str:
    return _mac(key or load_key(KEY, create=True), card_payload(card["id"], card.get("task"), card.get("text")))


def verify_entry(d: dict, key: bytes | None = None) -> bool:
    try:
        key = key or load_key(KEY, create=False)
    except (OSError, ValueError):
        return False
    sig = d.get("pipeline_sig")
    return isinstance(sig, str) and hmac.compare_digest(
        sig, _mac(key, entry_payload(d["id"], d["task_id"], d["kind"], d["content"], d.get("verified_by"))))


def verify_card(card: dict, key: bytes | None = None) -> bool:
    try:
        key = key or load_key(KEY, create=False)
    except (OSError, ValueError):
        return False
    sig = card.get("pipeline_sig")
    return isinstance(sig, str) and hmac.compare_digest(sig, sign_card(card, key))


def resign(db=None, log=print) -> dict:
    """Migration: (re)sign every pipeline:* workspace entry and every card with the current key."""
    from . import config as C
    from . import store
    from . import workspace as W
    key = load_key(KEY, create=True)
    con = W._connect(db)
    n_e = 0
    try:
        rows = con.execute("SELECT * FROM entries WHERE verified = 1 AND verified_by LIKE 'pipeline:%'").fetchall()
        upd = []
        for r in rows:
            c = W._content(r)
            if c is None:
                continue
            upd.append((sign_entry(r["id"], r["task_id"], r["kind"], c, r["verified_by"], key), r["id"]))
        con.execute("BEGIN IMMEDIATE")
        con.executemany("UPDATE entries SET pipeline_sig = ? WHERE id = ?", upd)
        con.execute("COMMIT")
        n_e = len(upd)
    finally:
        con.close()
    p = C.cards_path()
    n_c = 0
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        out = []
        for l in lines:
            c = json.loads(l)
            if c.get("verified") is True:
                c["pipeline_sig"] = sign_card(c, key)
                n_c += 1
            out.append(json.dumps(c, ensure_ascii=False))
        tmp = p.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(x + "\n" for x in out), encoding="utf-8")
        os.replace(tmp, p)                      # same line order, so cards.f32 stays aligned
        store._cache.clear()
    log("signed %d workspace entries, %d cards" % (n_e, n_c))
    return {"entries": n_e, "cards": n_c}


def verify_all(db=None) -> dict:
    from . import store
    from . import workspace as W
    key = load_key(KEY, create=False)
    con = W._connect(db)
    try:
        rows = con.execute("SELECT * FROM entries WHERE verified_by LIKE 'pipeline:%'").fetchall()
    finally:
        con.close()
    out = {"entries_ok": 0, "entries_bad": 0, "entries_unsigned": 0, "cards_ok": 0, "cards_bad": 0, "cards_unsigned": 0}
    for r in rows:
        d = W._as_dict(r, W._content(r))
        k = "entries_unsigned" if not d.get("pipeline_sig") else ("entries_ok" if verify_entry(d, key) else "entries_bad")
        out[k] += 1
    for c in store.load()[0]:
        k = "cards_unsigned" if not c.get("pipeline_sig") else ("cards_ok" if verify_card(c, key) else "cards_bad")
        out[k] += 1
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "resign":
        resign()
    elif cmd == "verify":
        print(json.dumps(verify_all(), indent=1))
    else:
        print(__doc__)