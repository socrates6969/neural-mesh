"""Mirror job: copy memory chunks and outcome summaries (TEXT + METADATA only, re-scrubbed) into ruflo memory, namespace
"vector-patterns" (ruflo embeds them with its own bundled MiniLM; we never ship vectors there).

Runnable later (AgentDB needs a Claude Code restart before its bridge is live): ``python -m ingest mirror``.
Resumable: ``state/mirror_done.txt`` holds ``key<TAB>sha16(value)``; a changed value is re-mirrored (upsert).
``--dry-run`` only writes the batch to ``mirror_batch.jsonl`` and calls nothing.

Guard rails:
* the target db defaults to ``<repo>/.swarm/memory.db`` (env ``NEURAL_MESH_RUFLO_DB``); a db that does not RESOLVE inside the
  repo (symlinks / junctions followed, case-insensitive on Windows) is refused (exit 5) unless ``allow_external``; UNC paths
  and mapped network drives are always refused;
* the ruflo CLI is pinned (``CLI_VERSION``; env ``NEURAL_MESH_RUFLO_CLI`` overrides) instead of ``@latest``;
* the ingest lock (``data/lifelong/.ingest.lock``) is held for the whole job (busy -> exit 75), so items.jsonl / outcomes.jsonl
  are not appended while being read;
* ``stats["exit"]``: 0 everything stored (or nothing to do), 3 partial, 4 nothing stored / npx missing, 5 refused, 75 busy.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config import REPO
from .scrub import scrub
from .util import ProcessLock, append_jsonl, atomic_write_text, iter_jsonl, sha16

NAMESPACE = "vector-patterns"
CLI_VERSION = "3.56.3"   # the locally cached @claude-flow/cli (npm-cache/_npx), pinned 2026-10-10
EXIT_OK, EXIT_PARTIAL, EXIT_NONE, EXIT_REFUSED, EXIT_BUSY = 0, 3, 4, 5, 75


def cli_package() -> str:
    v = (os.environ.get("NEURAL_MESH_RUFLO_CLI") or CLI_VERSION).strip()
    return v if v.startswith("@") else f"@claude-flow/cli@{v}"


def default_db(repo: Path = REPO) -> Path:
    env = os.environ.get("NEURAL_MESH_RUFLO_DB")
    return Path(env) if env else Path(repo) / ".swarm" / "memory.db"


def _is_remote_drive(path: str) -> bool:
    if sys.platform != "win32":
        return False
    drive = os.path.splitdrive(path)[0]
    if len(drive) != 2 or drive[1] != ":":
        return False
    import ctypes
    return ctypes.windll.kernel32.GetDriveTypeW(drive + "\\") == 4   # DRIVE_REMOTE


def _is_unc(path: str) -> bool:
    p = path.replace("/", "\\")
    return p.startswith("\\\\") and not p.startswith("\\\\?\\") or p.upper().startswith("\\\\?\\UNC\\")


def check_db(db_path, allow_external: bool = False, repo: Path = REPO) -> tuple[bool, str, str]:
    """(ok, resolved path, reason). Compares normcase'd fully resolved paths, so a junction / symlink inside the repo that
    points outside it counts as outside."""
    raw = os.path.abspath(str(db_path))
    if _is_unc(str(db_path)) or _is_unc(raw):
        return False, raw, "UNC path refused"
    if _is_remote_drive(raw):
        return False, raw, "mapped network drive refused"
    resolved = os.path.realpath(raw)
    if _is_unc(resolved) or _is_remote_drive(resolved):
        return False, resolved, "resolves to a network location; refused"
    root = os.path.normcase(os.path.realpath(str(repo)))
    inside = os.path.normcase(resolved).startswith(root.rstrip("\\/") + os.sep)
    if not inside and not allow_external:
        return False, resolved, f"db resolves outside the repo ({resolved}); pass --allow-external-db to write there"
    return True, resolved, "inside repo" if inside else "external db allowed by --allow-external-db"


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


def _run(cmd: list[str], cwd: Path):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, cwd=str(cwd))


def _exit_code(stats: dict) -> int:
    if stats["todo"] == 0:
        return EXIT_OK
    if stats["stored"] == 0:
        return EXIT_NONE
    return EXIT_OK if stats["stored"] >= stats["todo"] else EXIT_PARTIAL


def mirror(cfg, dry_run: bool = False, limit: int | None = None, db_path: str | None = None, allow_external: bool = False,
           wait_s: float = 0.0, repo: Path = REPO, runner=_run) -> dict:
    db = Path(db_path) if db_path else default_db(repo)
    ok, resolved, why = check_db(db, allow_external, repo)
    stats = {"candidates": 0, "todo": 0, "stored": 0, "failed": 0, "dry_run": dry_run, "namespace": NAMESPACE,
             "db_path": resolved, "cli_version": cli_package()}
    if not ok:
        stats.update(error=why, exit=EXIT_REFUSED)
        return stats
    lock = ProcessLock(cfg.lock_file)
    if not lock.acquire(wait_s):
        stats.update(error="another ingest run holds the lock", exit=EXIT_BUSY)
        return stats
    try:
        return _mirror_locked(cfg, stats, dry_run, limit, resolved, repo, runner)
    finally:
        lock.release()


def _mirror_locked(cfg, stats: dict, dry_run: bool, limit, db: str, repo: Path, runner) -> dict:
    batch = build_batch(cfg)
    done_path = cfg.state / "mirror_done.txt"
    done = set()
    if done_path.exists():
        done = {x.strip() for x in done_path.read_text(encoding="utf-8").splitlines() if x.strip()}
    todo = [b for b in batch if f"{b['key']}\t{sha16(b['value_json'])}" not in done]
    if limit is not None:
        todo = todo[:limit]
    stats.update(candidates=len(batch), todo=len(todo))
    if dry_run:
        atomic_write_text(cfg.data / "mirror_batch.jsonl", "".join(json.dumps({"key": b["key"], "tags": b["tags"], "value": b["value_json"]},
                                                                              ensure_ascii=False) + "\n" for b in todo))
        stats["exit"] = EXIT_OK
        return stats
    if not todo:
        stats["exit"] = EXIT_OK
        return stats
    npx = shutil.which("npx") or shutil.which("npx.cmd")
    if not npx:
        stats.update(error="npx not found", exit=EXIT_NONE)
        return stats
    for b in todo:
        cmd = [npx, "-y", cli_package(), "memory", "store", "-k", b["key"], "--value", b["value_json"], "-n", NAMESPACE,
               "--tags", b["tags"], "--provenance", "agent_output", "--path", db]
        try:
            ok = runner(cmd, repo).returncode == 0
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
    stats["exit"] = _exit_code(stats)
    append_jsonl(cfg.data / "mirror_runs.jsonl", [stats])
    return stats
