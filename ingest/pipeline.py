"""`run` and `status`: the incremental ingestion pipeline (read -> scrub -> extract -> judge -> emit -> memory -> mesh)."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone

from . import extract as ex
from .judge import RUBRIC_VERSION, Judge
from .labeler import label_items
from .memory import MemoryStore
from .meshio import MeshFeedback
from .reader import discover, load_all_states, update_file
from .scrub import SCRUB_VERSION
from .util import ProcessLock, append_jsonl, atomic_write_text, iter_jsonl, read_json, sha16
from .vectors import VectorStore, leader_clusters, vec_key

MIN_SFT_RESP = 80
MIN_SFT_USER = 15
SFT_NEAR_DUP = (0.97, 0.95)   # instruction AND response MiniLM cosine at/above these => near-duplicate, not written
SFT_CLUSTER = 0.80            # leader-clustering threshold on instruction vectors for the held-out split
HELDOUT_FRAC = 0.20


def _reward(outcome: str, judge: float | None) -> float:
    h = ex.OUTCOME_REWARD[outcome]
    return round(h if judge is None else 0.5 * h + 0.5 * judge, 4)


def run(cfg, wait_s: float = 0.0, quiet: bool = False) -> dict:
    lock = ProcessLock(cfg.lock_file)
    if not lock.acquire(wait_s):
        return {"ok": True, "skipped": "another ingest run holds the lock"}
    try:
        return _run_locked(cfg)
    finally:
        lock.release()


def _run_locked(cfg) -> dict:
    t0 = time.monotonic()
    now = time.time()
    settle_s = cfg.settle_minutes * 60
    summary: dict = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scrub_version": SCRUB_VERSION,
                     "rubric_version": RUBRIC_VERSION}

    # 1. read new bytes (incremental, resumable)
    hits: Counter = Counter()
    files = discover(cfg.projects)
    bytes_read = changed = errors = 0
    for path, rel in files:
        try:
            _, n, ch, delta = update_file(path, rel, cfg.files_dir)
        except (OSError, ValueError) as e:  # a locked / vanished file is retried next run
            errors += 1
            continue
        bytes_read += n
        changed += int(ch)
        hits.update(delta)
    summary.update(files=len(files), files_changed=changed, bytes_read=bytes_read, read_errors=errors)

    # 2. assemble
    sessions = ex.group_sessions(load_all_states(cfg.files_dir))
    out_ids = {r["id"] for r in iter_jsonl(cfg.outcomes) if "id" in r}
    sft_ids = {r["id"] for r in iter_jsonl(cfg.sft) if "id" in r}
    judge = Judge(cfg.judge_cache, cfg.ollama_url, cfg.judge_model, cfg.judge_budget, cfg.judge_timeout_s)
    dispatches = [d for s in sessions for d in ex.dispatch_items(s, now, settle_s)]
    turns = [t for s in sessions for t in ex.turn_items(s, now, settle_s)]
    dispatches.sort(key=lambda d: (d.ts or "", d.id))
    v2_labels = label_items(dispatches) if os.environ.get("INGEST_LABELER", "v2") != "v1" else {}   # R122 structural labeler (labeler.py)
    turns.sort(key=lambda t: (t.ts or "", t.id))

    new_out, new_sft, mem_sources = [], [], []
    drop = Counter()

    # 3. outcomes (judge first, so the reward can blend the grade)
    for d in dispatches:
        if not d.settled or d.id in out_ids:
            continue
        task = (d.description + "\n" + d.prompt).strip()
        g = judge.grade(d.id, task, d.report) if d.report and not d.report_secret else None
        outcome, sig = v2_labels.get(d.id) or ex.classify(task, d.report, d.result_head, d.signals, d.later)
        js = g["score"] if g else None
        task_text = d.description + (" :: " + d.prompt[:400] if d.prompt else "")
        if d.prompt_secret:
            task_text, sig = d.description, sig + ["prompt_secret_dropped"]
        row = {"id": d.id, "ts": d.ts, "end_ts": d.end_ts, "project": d.project, "session": d.session, "kind": d.kind, "agent": d.agent,
               "model": d.model, "spawn_depth": d.spawn_depth, "task_text": task_text, "features": ex.features(d.description + " " + d.prompt[:400]),
               "tokens": d.tokens, "outcome": outcome, "reward": _reward(outcome, js), "judge_score": js, "signals": sig}
        new_out.append(row)
        out_ids.add(d.id)
        # sft: agent prompt -> final report, only good outcomes with a high grade, never tainted / secret
        if outcome == "success" and js is not None and js >= cfg.sft_min_judge and not d.tainted and not d.prompt_secret \
                and not d.report_secret and len(d.report) >= MIN_SFT_RESP and len(d.prompt) >= MIN_SFT_USER:
            sid = "sft:" + sha16(d.prompt, d.report)
            if sid not in sft_ids:
                new_sft.append(_sft_row(sid, d.end_ts or d.ts, d.project, d.session, "agent_report", d.agent, d.model, d.prompt, d.report, outcome, g))
                sft_ids.add(sid)
        elif d.tainted:
            drop["sft_tainted"] += 1
        elif d.report_secret or d.prompt_secret:
            drop["sft_secret"] += 1
        if outcome in ("success", "defect") and (js is None or js >= 0.5) and not d.tainted and not d.report_secret:
            mem_sources.append((d.report, d.project, d.session, d.id, d.end_ts or d.ts, js if js is not None else ex.OUTCOME_REWARD[outcome]))

    # 4. sft from user turns (user request -> final answer), judged
    for t in turns:
        if not t.settled:
            continue
        if t.tainted:
            drop["turn_tainted"] += 1
            continue
        if t.user_secret or t.resp_secret:
            drop["turn_secret"] += 1
            continue
        if len(t.resp) < MIN_SFT_RESP or len(t.user) < MIN_SFT_USER:
            continue
        sid = "sft:" + sha16(t.user, t.resp)
        if sid in sft_ids:
            continue
        if ex.negative_feedback(t.feedback):
            drop["turn_negative_feedback"] += 1
            continue
        g = judge.grade(t.id, t.user, t.resp)
        if g is None or g["score"] < cfg.sft_min_judge:
            continue
        new_sft.append(_sft_row(sid, t.end_ts or t.ts, t.project, t.session, "user_turn", None, t.model, t.user, t.resp, None, g))
        sft_ids.add(sid)
        mem_sources.append((t.resp, t.project, t.session, t.id, t.end_ts or t.ts, g["score"]))

    # 5. vectors (one MiniLM space, cached by content hash) + sft near-duplicate GC, then write (append-only, de-duplicated by id)
    vs = VectorStore(cfg)
    vec_ok = vs.available()
    if vec_ok:
        old_sft = list(iter_jsonl(cfg.sft))
        texts = [r["task_text"] for r in new_out] + [x for r in old_sft + new_sft for x in (r["instruction"], r["response"])]
        vec_ok = vs.ensure(texts)
    if vec_ok:
        for r in new_out:
            r["vec_key"] = vec_key(r["task_text"])
        import numpy as np
        acc_i = [vs.get(r["instruction"]) for r in old_sft]
        acc_r = [vs.get(r["response"]) for r in old_sft]
        kept = []
        for r in new_sft:
            vi, vr = vs.get(r["instruction"]), vs.get(r["response"])
            if acc_i and float(np.max(np.stack(acc_i) @ vi)) >= SFT_NEAR_DUP[0] and float(np.max(np.stack(acc_r) @ vr)) >= SFT_NEAR_DUP[1]:
                drop["sft_near_dup"] += 1
                continue
            r["vec_key_instruction"], r["vec_key_response"] = vec_key(r["instruction"]), vec_key(r["response"])
            acc_i.append(vi)
            acc_r.append(vr)
            kept.append(r)
        new_sft = kept
    append_jsonl(cfg.outcomes, new_out)
    append_jsonl(cfg.sft, new_sft)
    if vec_ok:
        _write_sft_meta(cfg, vs)

    # 6. memory with usefulness credit
    mem = MemoryStore(cfg, vs)
    chunks = [(tag, chunk, project, session, src, ts, score) for text, project, session, src, ts, score in mem_sources
              for tag, chunk in ex.memory_chunks(text)]
    if mem.vs:
        mem.vs.ensure([c[1] for c in chunks])
    for tag, chunk, project, session, src, ts, score in chunks:
        mem.add(chunk, tag, project, session, src, ts or "", score)
    rewards: dict[str, list] = {}
    for r in iter_jsonl(cfg.outcomes):
        rewards.setdefault(r.get("session") or "", []).append((ex.parse_ts(r.get("end_ts") or r.get("ts")), float(r.get("reward", 0))))
    for r in iter_jsonl(cfg.sft):
        if r.get("source") == "user_turn":
            rewards.setdefault(r.get("session") or "", []).append((ex.parse_ts(r.get("ts")), float(r.get("judge_score") or 0)))
    mem.credit(rewards)
    mem.refresh_and_prune()
    mem.save()
    mem.export_mesh()

    # 7. mesh router feedback
    mf = MeshFeedback(cfg)
    if cfg.mesh_feedback and mf.available():
        for r in new_out:
            mf.feed(r)

    summary.update(
        new={"outcomes": len(new_out), "sft": len(new_sft), "memory_new": mem.stats["new"], "memory_reinforced": mem.stats["reinforced"],
             "memory_pruned": mem.stats["pruned"], "memory_exported": mem.stats["exported"]},
        totals={"outcomes": len(out_ids), "sft": len(sft_ids), "memory_active": sum(1 for i in mem.items.values() if i["status"] == "active"),
                "memory_items": len(mem.items)},
        pending={"dispatches_unsettled": sum(1 for d in dispatches if not d.settled), "turns_unsettled": sum(1 for t in turns if not t.settled)},
        scrub_hits=dict(sorted(hits.items())), dropped=dict(sorted(drop.items())), judge=judge.stats, mesh=mf.stats, vectors=vs.stats,
        memory={k: v for k, v in mem.stats.items() if k not in ("new", "reinforced", "pruned", "exported")},
        outcome_counts=dict(Counter(r["outcome"] for r in new_out)), runtime_s=round(time.monotonic() - t0, 1))
    append_jsonl(cfg.runs, [summary])
    return summary


def _write_sft_meta(cfg, vs) -> None:
    """Sidecar sft_meta.jsonl (rewritten): semantic cluster per row (leader clustering of instruction vectors in time order),
    held-out flag (whole clusters whose FIRST row is in the latest 20% by time, so paraphrases never straddle train/eval),
    and replay_weight = 1/cluster_size for diversity-aware replay sampling."""
    rows = sorted(iter_jsonl(cfg.sft), key=lambda r: (r.get("ts") or "", r["id"]))
    if not rows:
        return
    m = vs.matrix([r["instruction"] for r in rows])
    if m is None:
        return
    cl = leader_clusters(m, SFT_CLUSTER)
    first = {}
    for i, c in enumerate(cl):
        first.setdefault(c, i)
    cut = int(len(rows) * (1 - HELDOUT_FRAC))
    size = Counter(cl)
    lines = [json.dumps({"id": r["id"], "cluster": c, "cluster_size": size[c], "heldout": first[c] >= cut,
                         "replay_weight": round(1 / size[c], 4)}) + "\n" for r, c in zip(rows, cl)]
    atomic_write_text(cfg.data / "sft_meta.jsonl", "".join(lines))


def _sft_row(sid, ts, project, session, source, agent, model, instruction, response, outcome, g) -> dict:
    return {"id": sid, "ts": ts, "project": project, "session": session, "source": source, "agent": agent, "model": model,
            "instruction": instruction, "response": response, "outcome": outcome, "judge_score": g["score"], "judge_model": g["model"],
            "rubric_version": g["rubric_version"], "scrub_version": SCRUB_VERSION,
            "chars": {"instruction": len(instruction), "response": len(response)}}


def status(cfg) -> dict:
    files = discover(cfg.projects)
    cursors = {}
    for st in load_all_states(cfg.files_dir):
        cursors[st["rel"]] = st["cursor"]["offset"]
    total = sum(p.stat().st_size for p, _ in files)
    done = sum(min(cursors.get(rel, 0), p.stat().st_size) for p, rel in files)
    runs = list(iter_jsonl(cfg.runs))
    lock = ProcessLock(cfg.lock_file)
    held = not lock.acquire(0)
    lock.release()
    oc = Counter(r.get("outcome") for r in iter_jsonl(cfg.outcomes))
    return {"ok": True, "data": str(cfg.data), "files": len(files), "files_tracked": len(cursors), "bytes_total": total, "bytes_ingested": done,
            "bytes_pending": total - done, "outcomes": sum(oc.values()), "outcome_counts": dict(oc),
            "sft": sum(1 for _ in iter_jsonl(cfg.sft)), "judged": sum(1 for _ in iter_jsonl(cfg.judge_cache)),
            "memory": read_json(cfg.memory_dir / "meta.json", {}).get("seq", 0), "runs": len(runs), "lock_held": held,
            "last_run": {k: runs[-1].get(k) for k in ("ts", "new", "runtime_s", "judge")} if runs else None}
