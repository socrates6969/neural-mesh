"""Trusted pipeline feed: grader-verified training results -> Global Workspace (cross-session trusted knowledge).

  python -m train.memory.ingest_training <batch> [<batch> ...]      e.g. batch1 cyber1

For each results.jsonl row with phase held_base and no error:
  passed=True  -> one kind="method" entry per task (the method card of the shortest verified solution)
  passed=False -> one kind="failure" entry per (task, agent): negative memory ("this attempt was graded wrong")
All entries: verified=True, verified_by="pipeline:grader", task_id="training:<batch>", field set. This module is
out-of-band (it reads grader output, never model claims), which is what makes verified_by trustworthy.
Idempotent: every entry has an ext_key, so re-running posts nothing new.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import cards as K
from . import config as C
from . import workspace as W
from .builder import _tasks, ambiguous_ids, legacy_remapper


def ingest(batch: str, db=None, log=print) -> dict:
    rp = C.RUNS / "train" / batch / "results.jsonl"
    rows = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = _tasks(C.RUNS / "bench")
    remap = legacy_remapper(tasks)
    if remap is not None:
        rows = [dict(r, task=t) for r in rows for t in [remap(r)] if t is not None]
    tid = "training:" + batch
    stats = {"method": 0, "failure": 0, "skipped": 0}
    for task, r in K.verified_rows(rp, remap).items():
        if task not in tasks:
            stats["skipped"] += 1
            continue
        c = K.make_card(tasks[task], r, distill="extractive", batch=batch)
        if c is None:
            stats["skipped"] += 1
            continue
        eid = W.post(tid, r.get("agent", ""), "method", c["text"], confidence=0.9, verified=True,
                     verified_by="pipeline:grader", sources=["task:" + task, "card:" + task], field=c["field"], db=db,
                     ext_key="%s:method:%s" % (tid, task), pipeline_sign=True, bench_task=task,
                     split=tasks[task].get("split"), problem=tasks[task]["prompt"])
        stats["method" if eid else "skipped"] += 1
    seen = set()
    for r in rows:
        if r.get("phase") != "held_base" or r.get("error") or r.get("passed") is not False or r.get("pack"):
            continue
        key = (r["task"], r.get("agent", ""))
        if key in seen or r["task"] not in tasks:
            continue
        seen.add(key)
        t = tasks[r["task"]]
        tail = " ".join((r.get("text") or "").split())[-240:]
        content = ("Graded WRONG: attempt by %s on: %s\nWhat it answered (tail): %s"
                   % (r.get("agent", "?"), K.gist_of(t["prompt"], 24), tail))
        eid = W.post(tid, r.get("agent", ""), "failure", content, confidence=0.8, verified=True,
                     verified_by="pipeline:grader", sources=["task:" + r["task"]], field=r.get("field") or t.get("field"),
                     db=db, ext_key="%s:failure:%s:%s" % (tid, r["task"], r.get("agent", "")), pipeline_sign=True,
                     bench_task=r["task"], split=t.get("split"), problem=t["prompt"])
        stats["failure" if eid else "skipped"] += 1
    log("%s: %s" % (tid, stats))
    return stats


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for b in argv:
        ingest(b)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))