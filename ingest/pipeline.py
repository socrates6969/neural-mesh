"""`run` and `status`: the incremental ingestion pipeline (read -> scrub -> extract -> judge -> emit -> memory -> mesh)."""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone

from . import extract as ex
from . import trajectory as tj
from .judge import RUBRIC_VERSION, Judge
from .labeler import label_items
from .memory import MemoryStore
from .meshio import MeshFeedback
from .reader import SecretRegistry, discover, discover_report, load_all_states, update_file
from .schedule import Admission, Cand, grade_all
from .scrub import SCRUB_VERSION
from .util import ProcessLock, append_jsonl, atomic_write_text, iter_jsonl, read_json, sha16
from .vectors import VectorStore, leader_clusters, vec_key

MIN_SFT_RESP = 80
MIN_SFT_USER = 15
SFT_NEAR_DUP = (0.97, 0.95)   # instruction AND response MiniLM cosine at/above these => near-duplicate, not written
SFT_CLUSTER = 0.80            # leader-clustering threshold on instruction vectors for the held-out split
HELDOUT_FRAC = 0.20
ADMISSION = os.environ.get("INGEST_ADMISSION", "exec")   # "exec" (gap C: execution evidence, no judge) | "judge" (legacy local judge + decoy)
EXEC_GRADE = {"score": None, "model": "exec-evidence", "rubric_version": "exec-evidence-v1"}
MAX_TASK_STEP = 1500          # task text kept in a tool-use step row
MAX_HISTORY_STEP = 12         # previous actions shown in a tool-use step row


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
    files, skipped = discover_report(cfg.projects)
    bytes_read = changed = errors = 0
    reg = SecretRegistry(cfg.state / "secret_fps.json")
    for path, rel in files:
        try:
            _, n, ch, delta = update_file(path, rel, cfg.files_dir, reg)
        except (OSError, ValueError) as e:  # a locked / vanished file is retried next run
            errors += 1
            continue
        bytes_read += n
        changed += int(ch)
        hits.update(delta)
    reg.save()
    summary.update(files=len(files), files_changed=changed, bytes_read=bytes_read, read_errors=errors,
                   files_skipped=sum(skipped.values()), files_skipped_by=skipped)

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
    labels = {d.id: v2_labels.get(d.id) or ex.classify(_task_of(d), d.report, d.result_head, d.signals, d.later) for d in dispatches}
    apply_verdicts(dispatches)

    # 2b. SFT admission (R122, schedule.py): the judge budget goes first to SFT candidates, spread over sessions and sources, with a decoy contrast;
    #     candidates are looked at on EVERY run (not only when their outcome row is written), so a row skipped for lack of budget is retried
    vs = VectorStore(cfg)
    if ADMISSION == "exec":
        cands, adm, negatives = _exec_admission(dispatches, turns, labels, sft_ids)
    else:
        cands = _candidates(cfg, dispatches, turns, labels, sft_ids)
        if cands and vs.available():
            vs.ensure([c.instruction for c in cands] + [c.output for c in cands])
        adm = grade_all(judge, cands, vs if vs.available() else None, cfg.sft_min_judge)
        negatives = []

    new_out, new_sft, mem_sources = [], [], []
    drop = Counter()

    # 3. outcomes (judge first, so the reward can blend the grade)
    for d in dispatches:
        if not d.settled:
            continue
        is_new = d.id not in out_ids
        task = _task_of(d)
        outcome, sig = labels[d.id]
        a = adm.get(d.id)
        # sft: agent prompt -> final report, only good outcomes admitted by schedule.grade_all, never tainted / secret. Considered on every run
        # (an outcome row written while the judge budget was exhausted must not lose its SFT chance for good).
        if a is not None and a.accepted:
            sid = "sft:" + sha16(d.prompt, d.report)
            if sid not in sft_ids:
                new_sft.append(_sft_row(sid, d.end_ts or d.ts, d.project, d.session, "agent_report", d.agent, d.model, d.prompt, d.report, outcome, a.grade, a,
                                        ev=d.ev))
                sft_ids.add(sid)
        elif a is not None and a.accepted is False and a.decoy_needed:
            drop["sft_decoy_not_rejected"] += 1
        if not is_new:
            continue
        g = (a.grade if a is not None else None) or (judge.grade(d.id, task, d.report)
                                                      if ADMISSION != "exec" and d.report and not d.report_secret else None)
        js = g["score"] if g else None
        task_text = d.description + (" :: " + d.prompt[:400] if d.prompt else "")
        if d.prompt_secret:
            task_text, sig = d.description, sig + ["prompt_secret_dropped"]
        row = {"id": d.id, "ts": d.ts, "end_ts": d.end_ts, "project": d.project, "session": d.session, "kind": d.kind, "agent": d.agent,
               "model": d.model, "spawn_depth": d.spawn_depth, "task_text": task_text, "features": ex.features(d.description + " " + d.prompt[:400]),
               "tokens": d.tokens, "outcome": outcome, "reward": _reward(outcome, js), "judge_score": js, "signals": sig,
               "traj": tj.profile(d.calls), "verified": d.ev.get("verified"), "evidence": d.ev.get("why")}
        new_out.append(row)
        out_ids.add(d.id)
        if outcome == "success" and d.tainted:
            drop["sft_tainted"] += 1
        elif outcome == "success" and (d.report_secret or d.prompt_secret):
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
        a = adm.get(t.id)
        if a is None or not a.accepted:
            if a is not None and a.accepted is False and a.decoy_needed:
                drop["sft_decoy_not_rejected"] += 1
            continue
        g = a.grade
        new_sft.append(_sft_row(sid, t.end_ts or t.ts, t.project, t.session, "user_turn" if t.agent_id is None else "agent_turn",
                                t.agent, t.model, t.user, t.resp, None, g, a, ev=t.ev))
        sft_ids.add(sid)
        mem_sources.append((t.resp, t.project, t.session, t.id, t.end_ts or t.ts, g["score"] if g["score"] is not None else 1.0))

    # 5. vectors (one MiniLM space, cached by content hash) + sft near-duplicate GC, then write (append-only, de-duplicated by id)
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
    neg_ids = {r["id"] for r in iter_jsonl(cfg.data / "sft_negatives.jsonl") if "id" in r}
    new_neg = [r for r in negatives if r["id"] not in neg_ids and r["id"] not in sft_ids]
    append_jsonl(cfg.data / "sft_negatives.jsonl", new_neg)
    traj_stats = _write_trajectories(cfg, dispatches, turns, labels)
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
        admission=ADMISSION, taint_mode=ex.TAINT_MODE, trajectories=traj_stats, negatives_new=len(new_neg),
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


def _task_of(d) -> str:
    return (d.description + chr(10) + d.prompt).strip()


def _candidates(cfg, dispatches, turns, labels, sft_ids) -> list[Cand]:
    """Everything that could become an SFT row this run (same gates as before; the judge is not consulted here)."""
    out = []
    for d in dispatches:
        if d.kind != "dispatch" or not d.settled or labels[d.id][0] != "success" or d.tainted or d.prompt_secret or d.report_secret or not d.report:
            continue      # a SendMessage continuation is not a (request, answer) pair: its "report" is the agent's final report of the whole chain
        if len(d.report) < MIN_SFT_RESP or len(d.prompt) < MIN_SFT_USER or ("sft:" + sha16(d.prompt, d.report)) in sft_ids:
            continue
        out.append(Cand(d.id, _task_of(d), d.prompt, d.report, d.session, d.end_ts or d.ts or "", "agent_report"))
    for t in turns:
        if not t.settled or t.tainted or t.user_secret or t.resp_secret or len(t.resp) < MIN_SFT_RESP or len(t.user) < MIN_SFT_USER:
            continue
        if ("sft:" + sha16(t.user, t.resp)) in sft_ids or ex.negative_feedback(t.feedback):
            continue
        out.append(Cand(t.id, t.user, t.user, t.resp, t.session, t.end_ts or t.ts or "", "user_turn"))
    return out


def _sft_row(sid, ts, project, session, source, agent, model, instruction, response, outcome, g, adm=None, ev=None) -> dict:
    row = {"id": sid, "ts": ts, "project": project, "session": session, "source": source, "agent": agent, "model": model,
           "instruction": instruction, "response": response, "outcome": outcome, "judge_score": g["score"], "judge_model": g["model"],
           "rubric_version": g["rubric_version"], "scrub_version": SCRUB_VERSION,
           "chars": {"instruction": len(instruction), "response": len(response)}}
    if ev is not None:
        row["verified"] = ev.get("verified")
        row["evidence"] = {k: ev.get(k) for k in ("why", "edits", "checks_after", "check", "tests_passed", "tests_failed", "verdict") if ev.get(k) is not None}
    if adm is not None:     # decoy contrast (schedule.py): null decoy_score = no decoy could be built, admitted on the real grade alone
        row["decoy_score"] = adm.decoy_score
        row["decoy_margin"] = None if adm.decoy_score is None else round(g["score"] - adm.decoy_score, 4)
    return row


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


# ------------------------------------------------------------------------------------------------ gap C: later verdicts
_VERIFIER = re.compile(r"(?i)review|verif|\btest|tester|\bqa\b|audit|evaluat|validat|critic|judge|checker")
_PASS_VERDICT = re.compile(r"(?im)^[\W_]*(?:final\s+)?(?:verdict|result|status|overall|gate|decision|outcome)\b[\W_]{0,4}\s*[:=-]?\s*[\W_]{0,3}"
                           r"(PASS(?:ED)?|APPROVED?|ACCEPT(?:ED)?|LGTM|SHIP(?:PABLE)?)\b")
_FAIL_VERDICT = re.compile(r"(?im)^[\W_]*(?:final\s+)?(?:verdict|result|status|overall|gate|decision|outcome)\b[\W_]{0,4}\s*[:=-]?\s*[\W_]{0,3}"
                           r"(FAIL(?:ED)?|REJECT(?:ED)?|CHANGES[ _]REQUESTED|BLOCK(?:ED|ING)?|NEEDS[ _]FIX(?:ES)?|NO[- ]GO|NOT[ _]READY)\b")
VERDICT_WINDOW_S = 6 * 3600


def verdict_of(report: str) -> bool | None:
    """True (an explicit PASS verdict line), False (explicit FAIL/REJECT verdict line), None (no verdict line, or both)."""
    p, f = bool(_PASS_VERDICT.search(report or "")), bool(_FAIL_VERDICT.search(report or ""))
    return None if p == f else p


def apply_verdicts(dispatches: list) -> None:
    """A builder dispatch without its own execution evidence inherits the explicit verdict of the NEXT verifier dispatch
    (reviewer/tester/QA by agent type or description) in the same session, if that verifier started after the builder ended and
    within 6 h. Every builder since the previous verifier is credited. A builder's own exec evidence is never overridden."""
    by_s: dict = {}
    for d in dispatches:
        if d.kind == "dispatch":
            by_s.setdefault(d.session, []).append(d)
    for ds in by_s.values():
        ds.sort(key=lambda d: (d.ts or "", d.id))
        open_builders = []
        for d in ds:
            if not _VERIFIER.search(f"{d.agent} {d.description}"):
                open_builders.append(d)
                continue
            v = verdict_of(d.report)
            if v is not None:
                for b in open_builders:
                    gap = ex.parse_ts(d.ts) - ex.parse_ts(b.end_ts or b.ts)
                    if b.ev.get("verified") is None and 0 <= gap <= VERDICT_WINDOW_S:
                        b.ev = dict(b.ev, verified=v, why="later_pass_verdict" if v else "later_fail_verdict", verdict=d.id)
            open_builders = []


def _exec_admission(dispatches, turns, labels, sft_ids):
    """gap C admission: a candidate is admitted iff its trajectory has execution evidence of success (verified=True); verified=False
    candidates become NEGATIVE rows (sft_negatives.jsonl, for DPO / contrastive use); no evidence = not admitted. Same structural gates
    as before (settled, untainted, no secret, minimum lengths, no negative user feedback); the local judge is not consulted."""
    cands, adm, negs = [], {}, []
    for d in dispatches:
        if d.kind != "dispatch" or not d.settled or d.tainted or d.prompt_secret or d.report_secret or not d.report:
            continue
        if labels[d.id][0] in ("re-dispatch", "blocked") or len(d.report) < MIN_SFT_RESP or len(d.prompt) < MIN_SFT_USER:
            continue
        sid = "sft:" + sha16(d.prompt, d.report)
        if sid in sft_ids:
            continue
        v = d.ev.get("verified")
        cands.append(Cand(d.id, _task_of(d), d.prompt, d.report, d.session, d.end_ts or d.ts or "", "agent_report"))
        adm[d.id] = Admission(grade=EXEC_GRADE, accepted=bool(v), why=d.ev.get("why", ""))
        if v is False:
            negs.append(_sft_row(sid, d.end_ts or d.ts, d.project, d.session, "agent_report", d.agent, d.model, d.prompt, d.report,
                                 labels[d.id][0], EXEC_GRADE, ev=d.ev))
    for t in turns:
        if not t.settled or t.tainted or t.user_secret or t.resp_secret or len(t.resp) < MIN_SFT_RESP or len(t.user) < MIN_SFT_USER:
            continue
        sid = "sft:" + sha16(t.user, t.resp)
        if sid in sft_ids or ex.negative_feedback(t.feedback):
            continue
        v = t.ev.get("verified")
        src = "user_turn" if t.agent_id is None else "agent_turn"
        cands.append(Cand(t.id, t.user, t.user, t.resp, t.session, t.end_ts or t.ts or "", src))
        adm[t.id] = Admission(grade=EXEC_GRADE, accepted=bool(v), why=t.ev.get("why", ""))
        if v is False:
            negs.append(_sft_row(sid, t.end_ts or t.ts, t.project, t.session, src, t.agent, t.model, t.user, t.resp, None, EXEC_GRADE, ev=t.ev))
    return cands, adm, negs


# ------------------------------------------------------------------------------------------------ gap B: trajectories + tool-use rows
def _step_rows(traj_id: str, ts, project, session, agent, task: str, calls: list[dict], ev: dict) -> list[dict]:
    """One tool-use SFT row per call: instruction = task + the previous actions (content-free), response = the scrubbed rationale the
    agent wrote before the call + the action (tool, verbs/extension, argument keys). Calls whose rationale was secret are skipped."""
    rows, hist = [], []
    task_c = task if len(task) <= MAX_TASK_STEP else task[:MAX_TASK_STEP] + "\n[...]"
    for i, c in enumerate(calls):
        why = c.get("why") or {}
        if not why.get("s"):
            shown = hist[-MAX_HISTORY_STEP:]
            prog = "\n".join(f"- {h}" for h in shown) if shown else "(none yet)"
            more = f" ({len(hist) - len(shown)} earlier not shown)" if len(hist) > len(shown) else ""
            instr = f"TASK:\n{task_c}\n\nACTIONS SO FAR{more}:\n{prog}\n\nWhat is the next action?"
            resp = ((why.get("t") or "").strip() + "\n" if why.get("t") else "") + "ACTION: " + tj.action_text(c)
            rows.append({"id": "tool:" + sha16(traj_id, str(i)), "traj": traj_id, "step": i, "ts": c.get("ts") or ts, "project": project,
                         "session": session, "source": "tool_step", "agent": agent, "instruction": instr, "response": resp,
                         "action": {"n": c.get("n"), "k": c.get("k"), "v": c.get("v"), "x": c.get("x")},
                         "verified": ev.get("verified"), "evidence": ev.get("why"), "scrub_version": SCRUB_VERSION})
        hist.append(tj.step_line(c))
    return rows


def _write_trajectories(cfg, dispatches, turns, labels) -> dict:
    """trajectories.jsonl: one content-free record per settled subagent dispatch / turn with tool calls (router + analysis input).
    tooluse_sft.jsonl: per-call next-action rows from UNTAINTED trajectories whose task text is not secret (verified flag kept; a
    trainer admits verified=True only, verified=False are negatives)."""
    tp, sp = cfg.data / "trajectories.jsonl", cfg.data / "tooluse_sft.jsonl"
    have_t = {r["id"] for r in iter_jsonl(tp) if "id" in r}
    have_s = {r["id"] for r in iter_jsonl(sp) if "id" in r}
    new_t, new_s = [], []
    items = [(d.id, d.ts, d.project, d.session, d.agent, labels[d.id][0], _task_of(d), d.prompt_secret, d.tainted, d.calls, d.ev, d.settled)
             for d in dispatches if d.kind == "dispatch"]
    items += [(t.id, t.ts, t.project, t.session, t.agent or ("main" if t.agent_id is None else None), None, t.user, t.user_secret, t.tainted,
               t.calls, t.ev, t.settled) for t in turns]
    for tid, ts, project, session, agent, outcome, task, task_secret, tainted, calls, ev, settled in items:
        if not settled or not calls or tid in have_t:
            continue
        new_t.append({"id": tid, "ts": ts, "project": project, "session": session, "agent": agent, "outcome": outcome,
                      "verified": ev.get("verified"), "evidence": ev.get("why"), "tainted": tainted, "profile": tj.profile(calls),
                      "calls": [tj.compact(c) for c in calls]})
        have_t.add(tid)
        if tainted or task_secret:
            continue
        for r in _step_rows(tid, ts, project, session, agent, task, calls, ev):
            if r["id"] not in have_s:
                new_s.append(r)
                have_s.add(r["id"])
    append_jsonl(tp, new_t)
    append_jsonl(sp, new_s)
    return {"trajectories_new": len(new_t), "steps_new": len(new_s), "trajectories_total": len(have_t), "steps_total": len(have_s)}
