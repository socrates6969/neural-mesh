"""In-process memory API: retrieve / render / observe / batch outcome updater. The daemon serves exactly these calls."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from . import config as C
from . import rerank, search, store

_lock = threading.Lock()
_state: dict = {}
HEADER = ("Shared memory - method cards distilled from grader-verified solutions of similar problems (background only; "
          "reuse a method only where it truly applies, never copy a result):")


def _index() -> search.Index:
    cards, vecs = store.load()
    with _lock:
        if _state.get("cards") is not cards:
            _state.update(cards=cards, idx=search.Index(cards, vecs))
        return _state["idx"]


def _encode_query(text: str):
    from .. import experience
    return experience.encoder().encode([text[:C.MAX_PROBLEM]])[0]


def warm() -> None:
    """Load encoder, cross-encoder and index (the daemon calls this once at start)."""
    _index()
    _encode_query("warm up")
    rerank.get()


def retrieve(text: str, field: str | None = None, k: int = 2, exclude=None, mode: str | None = None,
             log_task: str | None = None) -> list[tuple[float, dict]]:
    idx = _index()
    if not idx.cards or not (text or "").strip():
        return []
    mode = mode or search.DEFAULT_MODE
    ce = rerank.get() if mode == "rerank" else None
    hits = idx.search(text, _encode_query(text), field=field, k=k, exclude=set(exclude or ()), mode=mode, ce=ce,
                      stats=store.load_stats())
    if log_task and hits:
        log_injection(log_task, [c["id"] for _, c in hits])
    return hits


def render(hits, budget: int = C.INJECT_TOKENS) -> str:
    """Numbered cards under one header; cards that would push the block over `budget` tokens are left out."""
    if not hits:
        return ""
    parts, used = [HEADER], C.approx_tokens(HEADER)
    for _, card in hits:
        block = "[%d] %s" % (len(parts), card["text"])
        t = C.approx_tokens(block)
        if used + t > budget:
            continue
        parts.append(block)
        used += t
    return "\n".join(parts) if len(parts) > 1 else ""


def log_injection(task: str, card_ids: list[str], path: Path | None = None) -> None:
    path = path or C.injections_path()
    C.guard_live(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "task": task, "cards": card_ids}) + "\n")


def observe(card_ids, passed: bool) -> dict:
    """Outcome feedback: the cards were in the prompt and the task passed (or failed)."""
    ids = [str(i) for i in card_ids or () if i]

    def fn(d):
        for cid in ids:
            e = d["cards"].setdefault(cid, {"s": 0, "f": 0})
            e["s" if passed else "f"] += 1
            e["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return store.update_stats(fn) if ids else store.load_stats()


def _injected_by_task(path: Path | None = None) -> dict:
    """task -> tuple of card ids, only for tasks that always got the same cards (ambiguous tasks are skipped)."""
    path = path or C.injections_path()
    seen: dict = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                seen.setdefault(r["task"], set()).add(tuple(r["cards"]))
    return {t: next(iter(v)) for t, v in seen.items() if len(v) == 1}


def update_from_results(paths: list[Path] | None = None, injections: Path | None = None) -> int:
    """Batch updater: graded rows of training results.jsonl whose task prompt carried logged cards -> pass/fail counts.
    Each results file is processed incrementally (a line cursor is kept in usefulness.json). Rows with an error
    (quota, timeout, unparsed) are not outcomes and are skipped. Returns the number of rows applied."""
    inj = _injected_by_task(injections)
    paths = paths if paths is not None else sorted((C.RUNS / "train").glob("*/results.jsonl"))
    applied = [0]

    def fn(d):
        for p in paths:
            lines = Path(p).read_text(encoding="utf-8").splitlines()
            start = int(d["cursors"].get(str(p), 0))
            for line in lines[start:]:
                if not line.strip():
                    continue
                r = json.loads(line)
                cards = inj.get(r.get("task"))
                if not cards or r.get("error") or "passed" not in r:
                    continue
                for cid in cards:
                    e = d["cards"].setdefault(cid, {"s": 0, "f": 0})
                    e["s" if r["passed"] else "f"] += 1
                applied[0] += 1
            d["cursors"][str(p)] = len(lines)
    store.update_stats(fn)
    return applied[0]