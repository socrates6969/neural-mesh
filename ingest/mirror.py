"""Mirror job: copy memory chunks and outcome summaries (TEXT + METADATA only, re-scrubbed) into ruflo memory, namespace
"vector-patterns" (ruflo embeds them with its own bundled MiniLM; we never ship vectors there).

Runnable later (AgentDB needs a Claude Code restart before its bridge is live): ``python -m ingest mirror``.
Resumable: ``state/mirror_done.txt`` holds ``key<TAB>sha16(value)``; a changed value is re-mirrored (upsert).
``--dry-run`` only writes the batch to ``mirror_batch.jsonl`` and calls nothing.
"""
from __future__ import annotations

import json
import shutil
import subprocess

from .scrub import scrub
from .util import append_jsonl, atomic_write_text, iter_jsonl, sha16

NAMESPACE = "vector-patterns"
CLI = ["-y", "@claude-flow/cli@latest", "memory", "store"]


def build_batch(cfg) -> list[dict]:
    out = []
    for it in iter_jsonl(cfg.memory_dir / "items.jsonl"):
        if it.get("status") != "active":
            continue
        val = {"kind": "memory", "id": it["id"], "tag": it["tag"], "text": it["text"], "project": it.get("project"),
               "value": it.get("value"), "seen": it.get("seen"), "retrievals": it.get("retrievals"), "source": "neural-mesh/ingest"}
        out.append({"key": f"lifelong/mem/{it['id'][4:]}", "tags": f"lifelong,memory,{it['tag']}", "value": val})
    for r in iter_jsonl(cfg.outcomes):
        val = {"kind": "outcome", "id": r["id"], "agent": r.get("agent"), "outcome": r.get("outcome"), "reward": r.get("reward"),
               "judge_score": r.get("judge_score"), "project": r.get("project"), "ts": r.get("ts"), "task": (r.get("task_text") or "")[:300],
               "source": "neural-mesh/ingest"}
        out.append({"key": f"lifelong/outcome/{sha16(r['id'])}", "tags": f"lifelong,outcome,{r.get('outcome')}", "value": val})
    clean = []
    for b in out:
        s = json.dumps(b["value"], ensure_ascii=False)
        r = scrub(s)
        if r.secret:  # belt and braces: everything was scrubbed at ingest; if anything still looks secret, drop it
            continue
        b["value_json"] = r.text
        clean.append(b)
    return clean


def mirror(cfg, dry_run: bool = False, limit: int | None = None, db_path: str | None = None) -> dict:
    batch = build_batch(cfg)
    done_path = cfg.state / "mirror_done.txt"
    done = set()
    if done_path.exists():
        done = {x.strip() for x in done_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [b for b in batch if f"{b['key']}\t{sha16(b['value_json'])}" not in done]
    if limit is not None:
        todo = todo[:limit]
    stats = {"candidates": len(batch), "todo": len(todo), "stored": 0, "failed": 0, "dry_run": dry_run, "namespace": NAMESPACE}
    if dry_run:
        atomic_write_text(cfg.data / "mirror_batch.jsonl", "".join(json.dumps({"key": b["key"], "tags": b["tags"], "value": b["value_json"]},
                                                                              ensure_ascii=False) + "\n" for b in todo))
        return stats
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        stats["error"] = "npx not found"
        return stats
    for b in todo:
        cmd = [npx, *CLI, "-k", b["key"], "--value", b["value_json"], "-n", NAMESPACE, "--tags", b["tags"], "--provenance", "agent_output"]
        if db_path:
            cmd += ["--path", db_path]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            ok = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if ok:
            stats["stored"] += 1
            done_path.parent.mkdir(parents=True, exist_ok=True)
            with open(done_path, "a", encoding="utf-8") as f:
                f.write(f"{b['key']}\t{sha16(b['value_json'])}\n")
        else:
            stats["failed"] += 1
            if stats["failed"] >= 5 and stats["stored"] == 0:
                stats["error"] = "first 5 stores failed; stopping (is the ruflo CLI reachable?)"
                break
    append_jsonl(cfg.data / "mirror_runs.jsonl", [stats])
    return stats
