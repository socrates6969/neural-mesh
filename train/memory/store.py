"""The shared card store: cards.jsonl (append-only, one card per line) + cards.f32 (MiniLM vectors of each card's
index text, same order) + usefulness.json (outcome counts per card). Every agent reads the same store; only the
builder (grader-verified runs) appends cards, and only observe()/the batch updater change usefulness counts."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import config as C

DIM = 384
_lock = threading.Lock()
_cache: dict = {}


def index_text(card: dict) -> str:
    """What retrieval matches against: the (scrubbed) solved problem plus the card's method text."""
    return (card.get("problem") or "") + "\n" + (card.get("text") or "")


def _vec_path(p: Path) -> Path:
    return p.with_suffix(".f32")


def load(path: Path | None = None) -> tuple[list[dict], np.ndarray]:
    """Cards + their vectors, cached per file mtime (cheap to call on every lookup)."""
    path = path or C.cards_path()
    vp = _vec_path(path)
    if not path.exists():
        return [], np.zeros((0, DIM), np.float32)
    key = (str(path), path.stat().st_mtime_ns, vp.stat().st_mtime_ns if vp.exists() else 0)
    with _lock:
        if _cache.get("key") != key:
            cards = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            vecs = np.fromfile(vp, dtype=np.float32).reshape(-1, DIM) if vp.exists() else np.zeros((0, DIM), np.float32)
            n = min(len(cards), len(vecs))
            _cache.update(key=key, cards=cards[:n], vecs=vecs[:n])
        return _cache["cards"], _cache["vecs"]


def add(cards: list[dict], encode, path: Path | None = None) -> int:
    """Append cards not yet stored (dedup by id) with their index-text embeddings. encode(list[str]) -> (n, 384)."""
    path = path or C.cards_path()
    old, _ = load(path)
    have = {c["id"] for c in old}
    new, seen = [], set()
    for c in cards:
        if c["id"] not in have and c["id"] not in seen and c.get("verified") is True:
            new.append(c)
            seen.add(c["id"])
    if not new:
        return 0
    vecs = np.asarray(encode([index_text(c) for c in new]), dtype=np.float32).reshape(-1, DIM)
    C.guard_live(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        _repair(path, len(old))
        with open(path, "a", encoding="utf-8") as f:
            for c in new:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        with open(_vec_path(path), "ab") as f:
            vecs.tofile(f)
        _cache.clear()
    return len(new)


def remove(card_ids, path: Path | None = None) -> int:
    """Drop cards by id, rewriting cards.jsonl and cards.f32 together (same order). Returns the number removed."""
    path = path or C.cards_path()
    C.guard_live(path)
    ids = set(card_ids)
    cards, vecs = load(path)
    keep = [i for i, c in enumerate(cards) if c["id"] not in ids]
    if len(keep) == len(cards):
        return 0
    with _lock:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(cards[i], ensure_ascii=False) + "\n" for i in keep), encoding="utf-8")
        vtmp = _vec_path(path).with_suffix(".f32.tmp")
        vecs[keep].astype(np.float32).tofile(vtmp)
        os.replace(vtmp, _vec_path(path))
        os.replace(tmp, path)
        _cache.clear()
    return len(cards) - len(keep)


def rewrite(cards: list[dict], path: Path | None = None) -> None:
    """Replace card metadata in place (same ids, same order; vectors untouched)."""
    path = path or C.cards_path()
    C.guard_live(path)
    old, _ = load(path)
    assert [c["id"] for c in cards] == [c["id"] for c in old], "rewrite must keep ids and order"
    with _lock:
        tmp = path.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cards), encoding="utf-8")
        os.replace(tmp, path)
        _cache.clear()


def _repair(path: Path, n: int) -> None:
    """Truncate cards.jsonl / cards.f32 to their first n aligned entries (an interrupted append can leave one longer)."""
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []
    vp = _vec_path(path)
    nv = vp.stat().st_size // (4 * DIM) if vp.exists() else 0
    if len(lines) != n:
        path.write_text("".join(l + "\n" for l in lines[:n]), encoding="utf-8")
    if nv != n:
        v = np.fromfile(vp, dtype=np.float32)[:n * DIM] if vp.exists() else np.zeros(0, np.float32)
        v.tofile(vp)


# ---- usefulness (outcome feedback) ----

class _FileLock:
    """Cross-process lock via an exclusive lock file (stale after 30 s)."""

    def __init__(self, path: Path, wait: float = 5.0):
        self.path, self.wait = Path(str(path) + ".lock"), wait

    def __enter__(self):
        t0 = time.monotonic()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > 30:
                        self.path.unlink()
                        continue
                except OSError:
                    pass
                if time.monotonic() - t0 > self.wait:
                    raise TimeoutError("usefulness lock busy")
                time.sleep(0.02)

    def __exit__(self, *a):
        try:
            self.path.unlink()
        except OSError:
            pass


def load_stats(path: Path | None = None) -> dict:
    path = path or C.stats_path()
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    d.setdefault("cards", {})
    d.setdefault("cursors", {})
    return d


def _save_stats(d: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp%d" % os.getpid())
    tmp.write_text(json.dumps(d, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def update_stats(fn, path: Path | None = None) -> dict:
    """Read-modify-write usefulness.json under the cross-process lock; fn(stats) mutates in place."""
    path = path or C.stats_path()
    C.guard_live(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, _FileLock(path):
        d = load_stats(path)
        fn(d)
        _save_stats(d, path)
    return d


def usefulness(stats: dict, card_id: str) -> tuple[float, int]:
    """(posterior mean pass rate with a Beta(1,1) prior, number of observations)."""
    e = stats.get("cards", {}).get(card_id) or {}
    s, f = int(e.get("s", 0)), int(e.get("f", 0))
    return (1 + s) / (2 + s + f), s + f
