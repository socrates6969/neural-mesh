"""Export verified agent-training results into the live neural mesh (user-authorized 2026-10-22: the router and the
nightly LLM must learn from this data).

1. Router: each held-out problem's grader-verified solver(s) become outcome rows labelled with that agent; fed now through
   the ingest pipeline's MeshFeedback (at-most-once via state/mesh_fed.txt) and appended to outcomes.jsonl.
2. Nightly LLM: one grader-verified correct answer per problem appended to sft.jsonl (source="verified_benchmark").
Idempotent by id; backups of both files are written first.
"""
from __future__ import annotations

import collections
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

NM = Path(r"C:\Users\mariu\neural-mesh")
import os  # noqa: E402
os.environ.setdefault("INGEST_DATA", str(NM / "data" / "lifelong"))   # write to the LIVE mesh data
os.environ.setdefault("INGEST_MESH_EXE", str(NM / "target" / "release" / "neural-mesh.exe"))
os.environ.setdefault("INGEST_MESHSTORE_EXE", str(NM / "target" / "release" / "meshstore.exe"))
from ingest.config import Config  # noqa: E402
from ingest.meshio import MeshFeedback  # noqa: E402
from ingest.extract import features  # noqa: E402

from . import data  # noqa: E402
from .loop import ROOT  # noqa: E402


def _ids(p: Path) -> set:
    return {json.loads(l)["id"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()} if p.exists() else set()


def main(batch="batch1"):
    cfg = Config()
    res = [json.loads(l) for l in (ROOT / "train" / batch / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    res = [r for r in res if not r.get("error")]
    tasks = {}
    for f in {r["field"] for r in res}:
        tasks.update({t["id"]: t for t in data.load(ROOT / "bench", f)})
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for p in (cfg.outcomes, cfg.sft):
        shutil.copyfile(p, p.with_name(p.name + ".bak-meshtrain-" + stamp))
    now = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())

    by = collections.defaultdict(dict)
    for r in res:
        if r["phase"] == "held_base" and r["task"] in tasks:
            by[(r["field"], r["task"], r["round"])][r["agent"]] = r["passed"]
    have, rows = _ids(cfg.outcomes), []
    for (field, tid, rnd), d in by.items():
        for agent in [a for a, ok in d.items() if ok]:
            oid = "out:meshtrain:" + hashlib.sha256(("%s|%s|%s|%s" % (batch, agent, tid, rnd)).encode()).hexdigest()[:16]
            if oid in have:
                continue
            text = tasks[tid]["prompt"][:1500]
            rows.append({"id": oid, "ts": now, "end_ts": now, "project": "meshtrain", "session": "meshtrain-" + batch,
                         "kind": "benchmark", "agent": agent, "model": None, "spawn_depth": 1, "task_text": text,
                         "features": ["field_" + field.replace("-", "_")] + [x for x in features(text[:600]) if "," not in x][:15],
                         "tokens": None, "outcome": "success", "reward": 1.0, "judge_score": 1.0,
                         "signals": ["grader_verified", "solved_by:%d/%d" % (sum(d.values()), len(d))], "vec_key": None})
    with open(cfg.outcomes, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mf = MeshFeedback(cfg)
    for r in rows:
        mf.feed(r)
    print("router: %d outcomes appended and fed %s" % (len(rows), mf.stats))

    best = {}
    for r in res:
        if r["passed"] and r["task"] in tasks and 40 < len(r.get("text") or "") < 4000:
            if r["task"] not in best or len(r["text"]) < len(best[r["task"]]["text"]):
                best[r["task"]] = r
    have, srows = _ids(cfg.sft), []
    for tid, r in best.items():
        sid = "sft:meshtrain:" + hashlib.sha256(tid.encode()).hexdigest()[:16]
        if sid in have:
            continue
        ins = tasks[tid]["prompt"]
        srows.append({"id": sid, "ts": now, "project": "meshtrain", "session": "meshtrain-" + r["field"],
                      "source": "verified_benchmark", "agent": r["agent"], "model": None, "instruction": ins,
                      "response": r["text"], "outcome": "success", "judge_score": 1.0, "judge_model": "grader:exact",
                      "rubric_version": "grader-v1", "scrub_version": "public-benchmark",
                      "chars": {"instruction": len(ins), "response": len(r["text"])}})
    with open(cfg.sft, "a", encoding="utf-8") as f:
        for r in srows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("llm: %d verified SFT rows appended %s" % (len(srows), dict(collections.Counter(best[t]["field"] for t in best))))


if __name__ == "__main__":
    main(*(sys.argv[1:] or []))