"""`python -m train.memory build`: cards from every training batch's grader-verified, pack-free held_base runs, plus a
migration of the legacy experience.jsonl records that no batch result covers. Incremental: existing card ids are skipped."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import cards as K
from . import config as C
from . import store
from . import trust


def _tasks(bench: Path) -> dict:
    """task id -> bench row. An id that one bench file uses for DIFFERENT problems (physics.jsonl reuses jee-phy-N)
    is ambiguous: a result row cannot be matched to its problem, so such ids are left out entirely."""
    out, bad = {}, set()
    for p in sorted(bench.glob("*.jsonl")):
        seen = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                if t["id"] in seen and C.norm_text(seen[t["id"]]) != C.norm_text(t["prompt"]):
                    bad.add(t["id"])
                seen.setdefault(t["id"], t["prompt"])
                out.setdefault(t["id"], t)       # current bench files sort before *_vN_saturated archives
    for b in bad:
        out.pop(b, None)
    return out


def legacy_remapper(tasks: dict):
    """Map result rows recorded under a retired bench id (`legacy_id`, e.g. physics jee-phy-N before the re-id) to
    the current id, ONLY when the mapping is certain - duplicate-free beats coverage:
      * the legacy id names exactly one current row -> that row (passed and failed rows alike);
      * otherwise, for a PASSED row: re-grade its text with the harness grader (train.check.grade) against every
        candidate row; exactly one correct candidate -> that row;
      * anything else (failed row of a shared legacy id, zero or several correct candidates) -> dropped.
    Rows whose task is a current id pass through unchanged."""
    by_legacy = {}
    for t in tasks.values():
        if t.get("legacy_id") and t["legacy_id"] != t["id"]:
            by_legacy.setdefault(t["legacy_id"], []).append(t)
    if not by_legacy:
        return None
    from .. import check
    cache = {}

    def remap(r):
        tid = r.get("task")
        if tid in tasks:
            return tid
        cands = by_legacy.get(tid)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]["id"]
        if not r.get("passed") or r.get("error"):
            return None
        key = (tid, r.get("text") or "")
        if key not in cache:
            ok = []
            for t in cands:
                try:
                    if check.grade(t, r.get("text") or ""):
                        ok.append(t["id"])
                except Exception:
                    pass
            cache[key] = ok[0] if len(ok) == 1 else None
        return cache[key]
    return remap


def ambiguous_ids(bench: Path | None = None) -> set:
    bench = bench or (C.RUNS / "bench")
    ids = set()
    for p in sorted(bench.glob("*.jsonl")):
        seen = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                t = json.loads(line)
                if t["id"] in seen and C.norm_text(seen[t["id"]]) != C.norm_text(t["prompt"]):
                    ids.add(t["id"])
                seen.setdefault(t["id"], t["prompt"])
    return ids


def _encode(texts):
    from .. import experience
    return experience.encoder().encode(texts)


def build(batches: list[str] | None = None, distill: str | None = None, migrate: bool = True, log=print) -> int:
    distill = distill or C.DISTILL
    have = {c["id"] for c in store.load()[0]}
    tasks = _tasks(C.RUNS / "bench")
    dirs = [C.RUNS / "train" / b for b in batches] if batches else sorted(p for p in (C.RUNS / "train").iterdir() if p.is_dir())
    new, dropped = [], 0
    for d in dirs:
        rp = d / "results.jsonl"
        if not rp.exists():
            continue
        for tid, r in K.verified_rows(rp, legacy_remapper(tasks)).items():
            if "card:" + tid in have or tid not in tasks:
                continue
            c = K.make_card(tasks[tid], r, distill=distill, batch=d.name)
            if c is None:
                dropped += 1
                continue
            c["pipeline_sig"] = trust.sign_card(c)          # the builder is a trusted pipeline (grader output only)
            new.append(c)
            have.add(c["id"])
            if len(new) % 10 == 0:
                log("  %d cards distilled (%s)" % (len(new), c["distiller"]))
                store.add(new, _encode)
    if migrate:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from .. import experience
        amb = ambiguous_ids() | {t["legacy_id"] for t in tasks.values()       # retired ids: their legacy records
                                 if t.get("legacy_id") and t["legacy_id"] not in tasks}   # cannot be matched safely
        for rec in experience.load()[0]:
            if "card:" + rec["task"] in have or rec["task"] in amb:
                continue
            c = K.make_card({"id": rec["task"], "prompt": rec["problem"], "field": rec.get("field")},
                            {"text": rec["solution"], "agent": rec.get("agent", "")}, distill=distill, batch="experience")
            if c is None:
                dropped += 1
                continue
            c["pipeline_sig"] = trust.sign_card(c)
            new.append(c)
            have.add(c["id"])
    n = store.add(new, _encode)
    log("cards: %d new (dropped %d on scrub), store %s" % (len(new), dropped, C.cards_path()))
    return n
