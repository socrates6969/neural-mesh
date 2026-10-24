"""Data hygiene for the shared memory (idempotent):  python -m train.memory hygiene

* drops cards and workspace entries derived from AMBIGUOUS bench ids (one bench file using the same id for different
  problems, e.g. physics.jsonl jee-phy-N): the solution cannot be matched to its problem, so the item may be wrong;
* drops cards / entries recorded under a RETIRED bench id (a current row has it as legacy_id): they are rebuilt
  under the current id by `build` / `ingest_training`, which map legacy results only when certain;
* tags every remaining card / training entry with bench_task, split and problem_key (the fingerprint that keeps an
  item from ever being returned for its own problem - held-out leakage protection).
"""
from __future__ import annotations

import json
import re

from . import config as C
from . import store
from . import workspace as W
from .builder import _tasks, ambiguous_ids

_EXT = re.compile(r"^training:[^:]+:(?:method|failure):([^:]+)")


def run(db=None, log=print) -> dict:
    tasks = _tasks(C.RUNS / "bench")
    amb = ambiguous_ids() | {t["legacy_id"] for t in tasks.values() if t.get("legacy_id") and t["legacy_id"] != t["id"]
                             and t["legacy_id"] not in tasks}
    cards, _ = store.load()
    removed_cards = store.remove([c["id"] for c in cards if c.get("task") in amb])
    cards, _ = store.load()
    tagged = []
    for c in cards:
        c = dict(c)
        t = tasks.get(c.get("task"))
        if t is not None:
            c["split"] = t.get("split", "")
        c["problem_key"] = C.problem_key(c.get("problem") or "")
        tagged.append(c)
    if tagged:
        store.rewrite(tagged)
    con = W._connect(db)
    removed_e = tagged_e = 0
    try:
        rows = con.execute("SELECT id, ext_key FROM entries WHERE ext_key LIKE 'training:%'").fetchall()
        drop, upd = [], []
        for r in rows:
            m = _EXT.match(r["ext_key"] or "")
            if not m:
                continue
            task = m.group(1)
            if task in amb:
                drop.append((r["id"],))
            elif task in tasks:
                upd.append((task, tasks[task].get("split"), C.problem_key(W._scrub(tasks[task]["prompt"]).text), r["id"]))
        con.execute("BEGIN IMMEDIATE")
        con.executemany("DELETE FROM concepts WHERE entry_id = ?", drop)
        con.executemany("DELETE FROM entries WHERE id = ?", drop)
        con.executemany("UPDATE entries SET bench_task = ?, split = ?, problem_key = ? WHERE id = ?", upd)
        con.execute("COMMIT")
        removed_e, tagged_e = len(drop), len(upd)
    finally:
        con.close()
    out = {"ambiguous_ids": len(amb), "cards_removed": removed_cards, "cards_tagged": len(tagged),
           "entries_removed": removed_e, "entries_tagged": tagged_e}
    log(json.dumps(out))
    return out