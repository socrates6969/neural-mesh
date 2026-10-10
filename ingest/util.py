"""Small helpers: atomic writes, JSONL, process lock (released by the OS if the process dies), hashing."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator


def sha16(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def atomic_write_text(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, separators=(",", ":")))


def read_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Read a JSONL file, skipping torn / invalid lines (a crash can leave a partial last line)."""
    try:
        with open(path, "rb") as f:
            for raw in f:
                try:
                    yield json.loads(raw.decode("utf-8", "replace"))
                except (json.JSONDecodeError, ValueError):
                    continue
    except FileNotFoundError:
        return


def append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Append rows; terminates a torn last line first so a previous crash cannot corrupt the next row."""
    rows = list(rows)
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(path, "ab") as g:
                    g.write(b"\n")
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return len(rows)


class ProcessLock:
    """Exclusive non-blocking lock on a file. The OS releases it when the process exits or crashes, so there is no
    stale-lock problem. ``acquire(wait_s)`` returns False if another run holds it after ``wait_s`` seconds."""

    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def acquire(self, wait_s: float = 0.0) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + wait_s
        while True:
            fh = open(self.path, "a+b")
            try:
                if sys.platform == "win32":
                    import msvcrt
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.fh = fh
                try:
                    fh.seek(0)
                    fh.truncate()
                    fh.write(f"{os.getpid()} {time.time():.0f}\n".encode())
                    fh.flush()
                except OSError:
                    pass
                return True
            except OSError:
                fh.close()
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.5)

    def release(self) -> None:
        if not self.fh:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.fh.close()
        self.fh = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.release()
