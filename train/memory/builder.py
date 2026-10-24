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
        for tid, r in K.verified_rows(rp).items():
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
        amb = ambiguous_ids()
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
