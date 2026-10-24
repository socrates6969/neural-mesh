"""Global Workspace for the neural mesh (Global Workspace Theory style): agents post what they know about a task, and
read() broadcasts the most salient items back - this task's own entries plus, at lower weight, relevant entries from
other tasks/sessions. Storage: SQLite in WAL mode (local, concurrent-safe across processes, no server).

  post(task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None,
       claimed_verified=False, verified_by=None, approval_sig=None, ext_key=None) -> int | None
  get(entry_id) -> dict | None                          one entry, decrypted
  approve(entry_id, by="human", sig=None) -> bool       mark verified (trusted out-of-band callers only)
  read(task_id, query=None, k=8) -> list[dict]          salience = relevance x recency x confidence x verified bonus
                                                          (x kind weight x usefulness); failures come back avoid=True
  search(text, field=None, k=3) -> list[dict]           method cards (train.memory card store)
  observe(entry_ids, passed) -> None                    usefulness of workspace entries and/or cards ("card:...")
  compress(task_id, keep=8) -> id | None                summary entry replacing old low-salience entries
  analogies(text, k=3, field=None) -> list[dict]        relevant items from DIFFERENT fields (concept graph bridge)
  stats() -> dict

Trust model: `verified` is never taken from model output. post() stores verified=True ONLY together with a
verified_by of "pipeline:<name>" (grader pipelines, e.g. train.memory.ingest_training) or "human:<user>" (an approval,
optionally HMAC-signed: approval_sig is stored as given). A bare verified=True (e.g. relayed from an agent) is stored as
claimed_verified=True, verified=False - a display-only, unchecked claim. Only out-of-band code (grader pipelines, the
human approval CLI) may pass verified_by; tool wrappers exposed to agents must not forward it.

Everything is scrubbed with ingest.scrub; content with a SECRET-class hit is dropped (post returns None).
Optional encryption: MESH_WORKSPACE_ENCRYPT=1 encrypts content with AES-GCM; the key lives in
~/.claude/mesh-workspace.key (created atomically, owner-only). It fails closed: if the key cannot be created, read or
restricted, post raises instead of writing plaintext. Embeddings and concept keywords stay unencrypted so retrieval
keeps working; they reveal topics, not the text.
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from . import config as C

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DB = Path(os.environ.get("MESH_WORKSPACE_DB", str(C.NM / "data" / "lifelong" / "mesh" / "workspace.db")))
KEY = Path(os.environ.get("MESH_WORKSPACE_KEY", str(Path.home() / ".claude" / "mesh-workspace.key")))
KINDS = ("fact", "hypothesis", "result", "question", "plan", "failure", "method")
KIND_W = {"result": 1.0, "method": 1.0, "failure": 1.0, "fact": 0.9, "summary": 0.9, "plan": 0.8, "question": 0.7,
          "hypothesis": 0.6}
HALF_LIFE_S = float(os.environ.get("MESH_WORKSPACE_HALF_LIFE_S", 3 * 86400))
VERIFIED_BONUS = 1.5
OTHER_W = 0.4            # entries of other tasks/sessions count less than this task's own
OTHER_MIN_COS = 0.45     # ... and only when clearly relevant
ANALOGY_MIN_COS = 0.30
MAX_CONTENT = 4000
MAX_OTHER_SCAN = 3000    # most recent entries of other tasks considered per read
DIM = 384

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL, agent TEXT, kind TEXT NOT NULL, field TEXT,
  content TEXT NOT NULL, enc INTEGER NOT NULL DEFAULT 0, confidence REAL NOT NULL, verified INTEGER NOT NULL,
  sources TEXT, created REAL NOT NULL, s INTEGER NOT NULL DEFAULT 0, f INTEGER NOT NULL DEFAULT 0,
  superseded_by INTEGER, concepts TEXT, vec BLOB, claimed_verified INTEGER NOT NULL DEFAULT 0, verified_by TEXT,
  approval_sig TEXT, ext_key TEXT, pipeline_sig TEXT, bench_task TEXT, split TEXT, problem_key TEXT);
CREATE INDEX IF NOT EXISTS entries_task ON entries(task_id, superseded_by);
CREATE INDEX IF NOT EXISTS entries_created ON entries(created);
CREATE TABLE IF NOT EXISTS concepts (entry_id INTEGER NOT NULL, concept TEXT NOT NULL, PRIMARY KEY (entry_id, concept));
CREATE INDEX IF NOT EXISTS concepts_c ON concepts(concept);
CREATE TABLE IF NOT EXISTS edges (a TEXT NOT NULL, b TEXT NOT NULL, w INTEGER NOT NULL, PRIMARY KEY (a, b));
"""


# ---------------------------------------------------------------- plumbing

def _connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or DB)
    C.guard_live(path)                                               # tests must use MESH_WORKSPACE_DB=<temp file>
    path.parent.mkdir(parents=True, exist_ok=True)
    delay = 0.02
    for attempt in range(12):
        con = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        try:
            _setup(con)
            return con
        except sqlite3.OperationalError as e:
            # switching a fresh file to WAL / creating the schema can report "locked" without waiting on the busy
            # handler when another process does the same at that moment: back off and retry
            con.close()
            if "locked" not in str(e) and "busy" not in str(e) or attempt == 11:
                raise
            time.sleep(delay)
            delay = min(1.0, delay * 2)
    raise sqlite3.OperationalError("workspace database stayed locked")


def _setup(con: sqlite3.Connection) -> None:
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    if con.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
        con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(_SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(entries)")}
    for col, decl in (("claimed_verified", "INTEGER NOT NULL DEFAULT 0"), ("verified_by", "TEXT"), ("approval_sig", "TEXT"),
                      ("ext_key", "TEXT"), ("pipeline_sig", "TEXT"), ("bench_task", "TEXT"), ("split", "TEXT"),
                      ("problem_key", "TEXT")):
        if col not in have:                                          # databases created before the trust model
            try:
                con.execute("ALTER TABLE entries ADD COLUMN %s %s" % (col, decl))
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e):                 # another process added it first: fine
                    raise
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS entries_ext ON entries(ext_key) WHERE ext_key IS NOT NULL")


def _embed(texts: list[str]) -> np.ndarray:
    """MiniLM sentence vectors: from the warm memory daemon when it runs (~25 ms), else the in-process encoder (and
    the daemon is started for next time). Tests may replace this function."""
    texts = [t[:C.MAX_PROBLEM] for t in texts]
    if os.environ.get("MESH_WORKSPACE_DAEMON", "on") != "off":
        from . import client
        v = client.embed(texts)
        if v is not None:
            return v
    from .. import experience
    return experience.encoder().encode(texts)


def _scrub(text: str):
    from ingest import scrub as S
    return S.scrub(text)


def concepts(text: str, field: str | None = None, n_uni: int = 6, n_bi: int = 4) -> list[str]:
    """Keyword concepts of a text: frequent content words and adjacent-word phrases (a cheap stand-in for noun
    phrases), plus field:<field>. Uses the prebrief tokenizer (stopwords, camel/snake splitting)."""
    from tools.prebrief.index import tokens
    toks = [t for t in tokens(text or "") if not t.isdigit()]
    uni = Counter(t for t in toks if len(t) >= 4)
    bi = Counter(a + " " + b for a, b in zip(toks, toks[1:]) if len(a) >= 3 and len(b) >= 3 and a != b)
    out = [t for t, _ in sorted(uni.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))[:n_uni]]
    out += [t for t, c in sorted(bi.items(), key=lambda x: (-x[1], x[0])) if c >= 1][:n_bi]
    if field:
        out.append("field:" + field)
    return out


# ---------------------------------------------------------------- encryption (optional, fail closed)

def _encrypt_on() -> bool:
    return os.environ.get("MESH_WORKSPACE_ENCRYPT", "0").strip() == "1"


def _restrict_owner_only(path: Path) -> None:
    if os.name != "nt":
        os.chmod(path, 0o600)
        return
    user = os.environ.get("USERNAME") or os.getlogin()
    r = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r", user + ":F"], capture_output=True, text=True)
    if r.returncode != 0:
        raise PermissionError("cannot restrict workspace key to the owner: " + (r.stderr or r.stdout).strip()[:200])


def _key(create: bool) -> bytes:
    """The 32-byte AES key. Created atomically (O_EXCL) and restricted to the owner; any failure raises."""
    try:
        k = KEY.read_bytes()
    except FileNotFoundError:
        if not create:
            raise
        KEY.parent.mkdir(parents=True, exist_ok=True)
        tmp = KEY.with_name(KEY.name + ".%d.tmp" % os.getpid())
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(secrets.token_bytes(32))
            _restrict_owner_only(tmp)
            try:
                os.link(tmp, KEY)            # atomic, fails if another process won the race
            except FileExistsError:
                pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        k = KEY.read_bytes()
    if len(k) != 32:
        raise ValueError("workspace key must be 32 bytes")
    return k


def _seal(plain: str, aad: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    nonce = secrets.token_bytes(12)
    ct = AESGCM(_key(create=True)).encrypt(nonce, plain.encode("utf-8"), aad.encode("utf-8"))
    return base64.b64encode(nonce + ct).decode("ascii")


def _open(blob: str, aad: str) -> str:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    raw = base64.b64decode(blob)
    return AESGCM(_key(create=False)).decrypt(raw[:12], raw[12:], aad.encode("utf-8")).decode("utf-8")


# ---------------------------------------------------------------- API

TRUSTED_PREFIXES = ("pipeline:", "human:")


def post(task_id, agent, kind, content, confidence=0.5, verified=False, sources=None, field=None, db=None,
         claimed_verified=False, verified_by=None, approval_sig=None, ext_key=None, pipeline_sign=False,
         bench_task=None, split=None, problem=None):
    """Store one entry; returns its id (int), or None when the content is empty or has a secret (scrub drop).
    verified=True is kept only with verified_by "pipeline:<name>" or "human:<user>"; otherwise it becomes an unchecked
    claimed_verified. ext_key (optional) makes the post idempotent: a second post with the same key returns the
    existing id and changes nothing. bench_task / split / problem (the solved problem's text; only its fingerprint
    is stored) tag knowledge derived from a benchmark task, so read()/search() never return it for that same problem
    and held-out splits can be hidden. pipeline_sign=True (trusted pipelines only, e.g. ingest_training) adds a
    pipeline_sig HMAC (train/memory/trust.py) over id, task, kind, content and verified_by; it requires a
    verified_by "pipeline:*" and raises if the pipeline key is unavailable (fail closed)."""
    kind = str(kind).strip().lower()
    if kind not in KINDS and kind != "summary":
        raise ValueError("kind must be one of " + "|".join(KINDS))
    sc = _scrub(str(content or ""))
    if sc.secret or not sc.text.strip():
        return None
    text = sc.text[:MAX_CONTENT]
    src = []
    for x in sources or ():
        ss = _scrub(str(x))
        if not ss.secret:
            src.append(ss.text[:300])
    task_id, agent = str(task_id), _scrub(str(agent or "")).text[:120]
    cs = concepts(text, field)
    vec = np.asarray(_embed([text])[0], dtype=np.float32).tobytes()
    stored, enc = text, 0
    if _encrypt_on():
        stored, enc = _seal(text, task_id + "|" + kind), 1          # raises (fails closed) on any key/cipher problem
    conf = float(min(1.0, max(0.0, confidence)))
    vby = str(verified_by) if verified_by else None
    trusted = bool(verified) and bool(vby) and vby.startswith(TRUSTED_PREFIXES)
    claimed = bool(claimed_verified) or (bool(verified) and not trusted)
    pkey = None
    if pipeline_sign:
        if not (trusted and vby.startswith("pipeline:")):
            raise ValueError("pipeline_sign requires verified=True and verified_by='pipeline:<name>'")
        from . import trust
        pkey = trust.load_key(trust.KEY, create=True)
    con = _connect(db)
    try:
        con.execute("BEGIN IMMEDIATE")
        if ext_key is not None:
            row = con.execute("SELECT id FROM entries WHERE ext_key = ?", (str(ext_key),)).fetchone()
            if row:
                con.execute("COMMIT")
                return row[0]
        cur = con.execute("INSERT INTO entries (task_id, agent, kind, field, content, enc, confidence, verified, sources, "
                          "created, concepts, vec, claimed_verified, verified_by, approval_sig, ext_key, bench_task, split, "
                          "problem_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (task_id, agent, kind, field, stored, enc, conf, int(trusted), json.dumps(src), time.time(),
                           json.dumps(cs), vec, int(claimed), vby if trusted else None,
                           str(approval_sig) if (trusted and approval_sig) else None,
                           str(ext_key) if ext_key is not None else None, bench_task, split,
                           C.problem_key(_scrub(problem).text) if problem else None))
        eid = cur.lastrowid
        if pkey is not None:
            from . import trust
            con.execute("UPDATE entries SET pipeline_sig = ? WHERE id = ?",
                        (trust.sign_entry(eid, task_id, agent, kind, text, vby, pkey), eid))
        con.executemany("INSERT OR IGNORE INTO concepts (entry_id, concept) VALUES (?,?)", [(eid, c) for c in cs])
        pairs = sorted({(a, b) for a in cs for b in cs if a < b})
        con.executemany("INSERT INTO edges (a, b, w) VALUES (?,?,1) ON CONFLICT(a, b) DO UPDATE SET w = w + 1", pairs)
        con.execute("COMMIT")
        return eid
    except BaseException:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def _content(row) -> str | None:
    if not row["enc"]:
        return row["content"]
    try:
        return _open(row["content"], row["task_id"] + "|" + row["kind"])
    except Exception:
        return None                                                  # no key / tampered: never shown


def _salience(row, rel: float, now: float, own: bool) -> float:
    age = max(0.0, now - row["created"])
    recency = max(0.1, 0.5 ** (age / HALF_LIFE_S))
    use = 2.0 * (1 + row["s"]) / (2 + row["s"] + row["f"])
    bonus = VERIFIED_BONUS if row["verified"] else 1.0
    return (rel * recency * max(0.05, row["confidence"]) * bonus * KIND_W.get(row["kind"], 0.8) * use
            * (1.0 if own else OTHER_W))


def _as_dict(row, content, sal=None, own=True, rel=None):
    d = {"id": row["id"], "task_id": row["task_id"], "agent": row["agent"], "kind": row["kind"], "content": content,
         "confidence": row["confidence"], "verified": bool(row["verified"]), "verified_by": row["verified_by"],
         "claimed_verified": bool(row["claimed_verified"]), "approval_sig": row["approval_sig"],
         "pipeline_sig": row["pipeline_sig"], "field": row["field"], "bench_task": row["bench_task"],
         "split": row["split"],
         "sources": json.loads(row["sources"] or "[]"), "created": row["created"], "scope": "task" if own else "other",
         "avoid": row["kind"] == "failure", "usefulness": {"s": row["s"], "f": row["f"]},
         "superseded_by": row["superseded_by"]}
    if sal is not None:
        d.update(salience=round(sal, 5), score=round(sal, 5), relevance=round(rel, 4))
    return d


def get(entry_id, db=None) -> dict | None:
    """One entry by id, decrypted (None if missing or not decryptable)."""
    con = _connect(db)
    try:
        row = con.execute("SELECT * FROM entries WHERE id = ?", (int(entry_id),)).fetchone()
    finally:
        con.close()
    if row is None:
        return None
    c = _content(row)
    return None if c is None else _as_dict(row, c)


def approve(entry_id, by="human", sig=None, db=None, approval_sig=None) -> bool:
    """Mark an entry verified by a trusted out-of-band party: by="human:<user>" (with an optional HMAC approval_sig,
    checked by the caller) or "pipeline:<name>". A bare "human" becomes "human:<USERNAME>". Returns False if the entry
    does not exist or `by` is not a trusted identity. Never expose this to agents."""
    sig = approval_sig if approval_sig is not None else sig
    by = str(by or "")
    if by == "human":
        by = "human:" + (os.environ.get("USERNAME") or os.environ.get("USER") or "unknown")
    if not by.startswith(TRUSTED_PREFIXES):
        return False
    con = _connect(db)
    try:
        con.execute("BEGIN IMMEDIATE")
        n = con.execute("UPDATE entries SET verified = 1, verified_by = ?, approval_sig = ?, pipeline_sig = NULL "
                        "WHERE id = ?",
                        (by, str(sig) if sig else None, int(entry_id))).rowcount
        con.execute("COMMIT")
    finally:
        con.close()
    return n == 1


def _vec(row) -> np.ndarray:
    return np.frombuffer(row["vec"], dtype=np.float32) if row["vec"] else np.zeros(DIM, np.float32)


def _hidden(row, qnorm: str, exclude) -> bool:
    """Never broadcast an entry derived from the very problem being asked (it would leak its answer), entries of
    excluded bench tasks, or entries of excluded splits (MESH_MEMORY_EXCLUDE_SPLITS)."""
    if exclude and row["bench_task"] and row["bench_task"] in exclude:
        return True
    if C.EXCLUDE_SPLITS and row["split"] in C.EXCLUDE_SPLITS:
        return True
    k = row["problem_key"]
    return bool(k and qnorm and k in qnorm)


def read(task_id, query=None, k=8, db=None, exclude=None) -> list[dict]:
    """The broadcast for a task: most salient entries, this task first-class, other tasks down-weighted and gated.
    Failures are negative memory: returned with avoid=True, and up to max(1, k // 4) of them are always kept.
    exclude: bench task ids whose derived entries must not be returned; entries derived from the problem in `query`
    itself are always left out."""
    task_id = str(task_id)
    con = _connect(db)
    try:
        own = con.execute("SELECT * FROM entries WHERE task_id = ? AND superseded_by IS NULL", (task_id,)).fetchall()
        other = con.execute("SELECT * FROM entries WHERE task_id != ? AND superseded_by IS NULL ORDER BY created DESC "
                            "LIMIT ?", (task_id, MAX_OTHER_SCAN)).fetchall()
    finally:
        con.close()
    if query:
        q = np.asarray(_embed([str(query)])[0], dtype=np.float32)
    elif own:
        q = np.mean([_vec(r) for r in own], axis=0)
        q = q / (np.linalg.norm(q) or 1.0)
    else:
        q = None
    qnorm = C.norm_text(str(query)) if query else ""
    exclude = set(exclude or ())
    own = [r for r in own if not _hidden(r, qnorm, exclude)]
    other = [r for r in other if not _hidden(r, qnorm, exclude)]
    now, items = time.time(), []
    for r in own:
        rel = 1.0 if q is None or not query else 0.5 + 0.5 * max(0.0, float(_vec(r) @ q))
        items.append((_salience(r, rel, now, True), r, True, rel))
    if q is not None:
        for r in other:
            cos = float(_vec(r) @ q)
            if cos >= OTHER_MIN_COS:
                items.append((_salience(r, cos, now, False), r, False, cos))
    items.sort(key=lambda x: -x[0])
    out, fails = [], []
    for sal, r, is_own, rel in items:
        c = _content(r)
        if c is None:
            continue
        (fails if r["kind"] == "failure" else out).append(_as_dict(r, c, sal, is_own, rel))
    n_fail = min(len(fails), max(1, k // 4))
    picked = sorted(out[:max(0, k - n_fail)] + fails[:n_fail], key=lambda d: -d["salience"])
    rest = sorted(out[max(0, k - n_fail):] + fails[n_fail:], key=lambda d: -d["salience"])
    return (picked + rest)[:k]


def search(text, field=None, k=3, low_confidence=False, exclude=None) -> list[dict]:
    """Method cards for a problem (the shared card memory; see train/memory/search.py for the retrieval chain).
    Pass the full problem text: queries under 20 words return nothing unless low_confidence=True (then the hits are
    flagged band="low", expected_precision ~.25). Each dict has band and expected_precision."""
    from . import api
    # cards are built only from grader-verified, pack-free runs (train/memory/cards.py), hence the provenance
    return [{"id": c["id"], "task": c.get("task"), "field": c.get("field"), "score": round(s, 4), "text": c["text"],
             "content": c["text"], "source": "card", "provenance": "pipeline:grader", "verified": True,
             "pipeline_sig": c.get("pipeline_sig"), "band": c.get("band"), "expected_precision": c.get("expected_precision")}
            for s, c in api.retrieve(str(text), field=field, k=k, low_confidence=low_confidence, exclude=exclude)]


def observe(entry_ids, passed, db=None) -> None:
    """Outcome feedback: these entries/cards were used and the task passed (True) or failed (False)."""
    ws, cards = [], []
    for e in entry_ids or ():
        if isinstance(e, str) and e.startswith("card:"):
            cards.append(e)
        else:
            try:
                ws.append(int(e))
            except (TypeError, ValueError):
                continue
    if ws:
        col = "s" if passed else "f"
        con = _connect(db)
        try:
            con.execute("BEGIN IMMEDIATE")
            con.executemany("UPDATE entries SET %s = %s + 1 WHERE id = ?" % (col, col), [(i,) for i in ws])
            con.execute("COMMIT")
        finally:
            con.close()
    if cards:
        from . import api
        api.observe(cards, bool(passed))


def _first_words(text: str, n: int = 25) -> str:
    w = " ".join(str(text).split()).split(" ")
    return " ".join(w[:n]) + ("..." if len(w) > n else "")


def compress(task_id, keep=8, db=None):
    """Context compression: all but the `keep` most salient entries of a task (verified entries and failures are
    always kept) are folded into one 'summary' entry; the originals are marked superseded (hidden from read(), kept
    on disk). Returns the summary id, or None when there is nothing to fold."""
    task_id = str(task_id)
    con = _connect(db)
    try:
        rows = con.execute("SELECT * FROM entries WHERE task_id = ? AND superseded_by IS NULL", (task_id,)).fetchall()
    finally:
        con.close()
    now = time.time()
    ranked = sorted(rows, key=lambda r: -_salience(r, 1.0, now, True))
    fold = [r for r in ranked[keep:] if not r["verified"] and r["kind"] != "failure"]
    if len(fold) < 2:
        return None
    fold.sort(key=lambda r: r["created"])
    lines = []
    for r in fold:
        c = _content(r)
        if c is not None:
            lines.append("- [%s] %s" % (r["kind"], _first_words(c)))
    body = "Summary of %d earlier entries:\n" % len(lines) + "\n".join(lines)
    fields = Counter(r["field"] for r in fold if r["field"])
    sid = post(task_id, "workspace", "summary", body[:MAX_CONTENT], confidence=float(np.mean([r["confidence"] for r in fold])),
               sources=["ws:%d" % r["id"] for r in fold], field=fields.most_common(1)[0][0] if fields else None, db=db)
    if sid is None:
        return None
    con = _connect(db)
    try:
        con.execute("BEGIN IMMEDIATE")
        con.executemany("UPDATE entries SET superseded_by = ? WHERE id = ?", [(sid, r["id"]) for r in fold])
        con.execute("COMMIT")
    finally:
        con.close()
    return sid


def _neighbours(con, cs: list[str], per: int = 5) -> set:
    out = set(cs)
    for c in cs:
        for r in con.execute("SELECT b AS n, w FROM edges WHERE a = ? UNION ALL SELECT a AS n, w FROM edges WHERE b = ? "
                             "ORDER BY w DESC LIMIT ?", (c, c, per)):
            if not r["n"].startswith("field:"):
                out.add(r["n"])
    return out


def analogies(text, k=3, field=None, db=None, exclude=None) -> list[dict]:
    """Cross-field analogy: the most relevant workspace entries and method cards from fields OTHER than the query's
    (inferred from the closest card when not given). Score = cosine x (1 + 0.5 x concept-graph overlap), where the
    query's concepts are expanded with their strongest co-occurring concepts in the workspace graph."""
    from . import store
    q = np.asarray(_embed([str(text)])[0], dtype=np.float32)
    cards, vecs = store.load()
    if field is None and len(cards):
        field = cards[int(np.argmax(vecs @ q))].get("field")
    qc = [c for c in concepts(str(text)) if not c.startswith("field:")]
    con = _connect(db)
    try:
        expanded = _neighbours(con, qc)
        rows = con.execute("SELECT * FROM entries WHERE superseded_by IS NULL AND field IS NOT NULL AND field != ? "
                           "ORDER BY created DESC LIMIT ?", (field or "", MAX_OTHER_SCAN)).fetchall()
    finally:
        con.close()
    qnorm, exclude = C.norm_text(str(text)), set(exclude or ())
    rows = [r for r in rows if not _hidden(r, qnorm, exclude)]
    cand = []
    for r in rows:
        cos = float(_vec(r) @ q)
        if cos >= ANALOGY_MIN_COS:
            cand.append((cos, set(json.loads(r["concepts"] or "[]")), "ws", r))
    for i, c in enumerate(cards):
        ck = c.get("problem_key") or C.problem_key(c.get("problem") or "")
        if (ck and ck in qnorm) or c.get("task") in exclude or (C.EXCLUDE_SPLITS and c.get("split") in C.EXCLUDE_SPLITS):
            continue
        if c.get("field") and c.get("field") != field:
            cos = float(vecs[i] @ q)
            if cos >= ANALOGY_MIN_COS:
                cand.append((cos, set(concepts(c.get("text", ""))), "card", c))
    out = []
    for cos, cs, src, obj in cand:
        shared = (cs - {x for x in cs if x.startswith("field:")}) & expanded
        bridge = len(shared) / max(1, len(expanded))
        score = cos * (1 + 0.5 * bridge)
        if src == "ws":
            content = _content(obj)
            if content is None:
                continue
            item = {"id": obj["id"], "source": "workspace", "task_id": obj["task_id"], "kind": obj["kind"],
                    "field": obj["field"], "content": content}
        else:
            item = {"id": obj["id"], "source": "card", "task_id": obj.get("task"), "kind": "method",
                    "field": obj.get("field"), "content": obj.get("text")}
        if src == "ws":
            item.update(agent=obj["agent"], confidence=obj["confidence"], verified=bool(obj["verified"]),
                        verified_by=obj["verified_by"], claimed_verified=bool(obj["claimed_verified"]),
                        approval_sig=obj["approval_sig"], pipeline_sig=obj["pipeline_sig"], provenance=obj["verified_by"])
        else:
            item.update(agent=obj.get("agent"), confidence=1.0, verified=True, verified_by="pipeline:grader",
                        claimed_verified=False, provenance="pipeline:grader", approval_sig=None,
                        pipeline_sig=obj.get("pipeline_sig"), task=obj.get("task"), text=obj.get("text"))
        item.update(score=round(score, 4), relevance=round(cos, 4), concepts=sorted(shared), query_field=field)
        out.append(item)
    out.sort(key=lambda d: -d["score"])
    return out[:k]


def stats(db=None) -> dict:
    con = _connect(db)
    try:
        kinds = dict(con.execute("SELECT kind, COUNT(*) FROM entries GROUP BY kind").fetchall())
        tot = con.execute("SELECT COUNT(*), COUNT(DISTINCT task_id), SUM(superseded_by IS NOT NULL), SUM(enc), "
                          "SUM(verified) FROM entries").fetchone()
        nc = con.execute("SELECT COUNT(DISTINCT concept) FROM concepts").fetchone()[0]
        ne = con.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    finally:
        con.close()
    return {"entries": tot[0], "tasks": tot[1], "superseded": tot[2] or 0, "encrypted": tot[3] or 0,
            "verified": tot[4] or 0, "kinds": kinds, "concepts": nc, "edges": ne, "db": str(DB)}


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m train.memory.workspace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("post")
    p.add_argument("--task", required=True)
    p.add_argument("--agent", default="cli")
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--content", required=True)
    p.add_argument("--confidence", type=float, default=0.5)
    p.add_argument("--verified", action="store_true")
    p.add_argument("--field")
    p.add_argument("--source", action="append", default=[])
    p = sub.add_parser("read")
    p.add_argument("--task", required=True)
    p.add_argument("--query")
    p.add_argument("-k", type=int, default=8)
    p = sub.add_parser("search")
    p.add_argument("text")
    p.add_argument("--field")
    p.add_argument("-k", type=int, default=3)
    p.add_argument("--low-confidence", action="store_true")
    p = sub.add_parser("analogies")
    p.add_argument("text")
    p.add_argument("--field")
    p.add_argument("-k", type=int, default=3)
    p = sub.add_parser("compress")
    p.add_argument("--task", required=True)
    p.add_argument("--keep", type=int, default=8)
    p = sub.add_parser("observe")
    p.add_argument("ids", nargs="+")
    p.add_argument("--failed", action="store_true")
    p = sub.add_parser("get")
    p.add_argument("id", type=int)
    sub.add_parser("stats")
    a = ap.parse_args(argv)
    if a.cmd == "post":
        out = post(a.task, a.agent, a.kind, a.content, a.confidence, a.verified, a.source, a.field)
    elif a.cmd == "read":
        out = read(a.task, a.query, a.k)
    elif a.cmd == "search":
        out = search(a.text, a.field, a.k, a.low_confidence)
    elif a.cmd == "analogies":
        out = analogies(a.text, a.k, a.field)
    elif a.cmd == "compress":
        out = compress(a.task, a.keep)
    elif a.cmd == "get":
        out = get(a.id)
    elif a.cmd == "observe":
        observe(a.ids, not a.failed)
        out = "ok"
    else:
        out = stats()
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())