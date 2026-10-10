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
* records go through ``memory import -i <temp file>`` in chunks of ``CHUNK`` (cwd = repo), never as ``--value`` on argv
  (process list, ~32K command-line limit); the temp file is deleted in ``finally``. Limitation: import stores key/value/
  namespace only (no ruflo tags / provenance);
* the ingest lock (``data/lifelong/.ingest.lock``) is held for the whole job (busy -> exit 75), so items.jsonl / outcomes.jsonl
  are not appended while being read;
* ``stats["exit"]``: 0 everything stored (or nothing to do), 3 partial, 4 nothing stored / npx missing, 5 refused, 75 busy;
  every non-dry run that reaches an import, and every refused / busy run, appends a row to ``mirror_runs.jsonl``.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import REPO
from .scrub import scrub
from .util import ProcessLock, append_jsonl, atomic_write_text, iter_jsonl, sha16

NAMESPACE = "vector-patterns"
CLI_VERSION = "latest"   # user policy (2026-10-10): run tools via npx @latest; override with env NEURAL_MESH_RUFLO_CLI
CHUNK = 200            # entries per `memory import` file
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


def build_batch(cfg, stats: dict | None = None) -> list[dict]:
    """Active memories + outcomes, re-scrubbed. A record whose value still has a SECRET-class hit after scrub is dropped; the
    number of such drops is written to ``stats["dropped_secret"]`` when ``stats`` is given (counts only, never content)."""
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
    dropped = 0
    for b in out:
        s = json.dumps(b["value"], ensure_ascii=False)
        r = scrub(s)
        if r.secret:  # belt and braces: everything was scrubbed at ingest; if anything still looks secret, drop it
            dropped += 1
            continue
        b["value_json"] = r.text
        clean.append(b)
    if stats is not None:
        stats["dropped_secret"] = dropped
    return clean


def _run(cmd: list[str], cwd: Path):
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180, cwd=str(cwd))


_ENTRIES = re.compile(r"Entries:\s*(\d+)")


def import_chunk(npx: str, chunk: list[dict], db: str, repo: Path, runner) -> int:
    """`memory import` of one chunk from a temp JSON file (values never appear on argv / in the process list). Returns the number
    of entries the CLI reports as imported (the CLI exits 0 even when some writes fail, so its "Entries: N" line is the truth).
    The temp file is always deleted. NOTE: `memory import` keeps key/value/namespace only; ruflo tags/provenance are not set."""
    doc = {"entries": [{"key": b["key"], "value": b["value_json"], "namespace": NAMESPACE} for b in chunk]}
    fd, tmp = tempfile.mkstemp(prefix="mirror-chunk-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        cmd = [npx, "-y", cli_package(), "memory", "import", "-i", tmp, "-n", NAMESPACE, "--path", db]
        try:
            r = runner(cmd, repo)
        except (OSError, subprocess.SubprocessError):
            return 0
        m = _ENTRIES.search(r.stdout or "")
        return min(len(chunk), int(m.group(1))) if r.returncode == 0 and m else 0
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _exit_code(stats: dict) -> int:
    if stats["todo"] == 0:
        return EXIT_OK
    if stats["stored"] == 0:
        return EXIT_NONE
    return EXIT_OK if stats["stored"] >= stats["todo"] else EXIT_PARTIAL


def _log_run(cfg, stats: dict) -> None:
    """one row per run in mirror_runs.jsonl, including refused (5) and busy (75) runs (exit code + reason in "error")."""
    from datetime import datetime, timezone
    append_jsonl(cfg.data / "mirror_runs.jsonl", [{"ts": datetime.now(timezone.utc).isoformat(), **stats}])


def mirror(cfg, dry_run: bool = False, limit: int | None = None, db_path: str | None = None, allow_external: bool = False,
           wait_s: float = 0.0, repo: Path = REPO, runner=_run) -> dict:
    db = Path(db_path) if db_path else default_db(repo)
    ok, resolved, why = check_db(db, allow_external, repo)
    stats = {"candidates": 0, "todo": 0, "stored": 0, "failed": 0, "dropped_secret": 0, "dry_run": dry_run, "namespace": NAMESPACE,
             "db_path": resolved, "cli_version": cli_package()}
    if not ok:
        stats.update(error=why, exit=EXIT_REFUSED)
        _log_run(cfg, stats)
        return stats
    lock = ProcessLock(cfg.lock_file)
    if not lock.acquire(wait_s):
        stats.update(error="another ingest run holds the lock", exit=EXIT_BUSY)
        _log_run(cfg, stats)
        return stats
    try:
        return _mirror_locked(cfg, stats, dry_run, limit, resolved, repo, runner)
    finally:
        lock.release()


def _mirror_locked(cfg, stats: dict, dry_run: bool, limit, db: str, repo: Path, runner) -> dict:
    batch = build_batch(cfg, stats)
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
    done_path.parent.mkdir(parents=True, exist_ok=True)
    stats["chunks"] = 0
    dead = 0
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        stats["chunks"] += 1
        n = import_chunk(npx, chunk, db, repo, runner)
        stats["stored"] += n
        stats["failed"] += len(chunk) - n
        if n == len(chunk):   # all-or-nothing bookkeeping: a partial chunk is re-imported (upsert) next run
            with open(done_path, "a", encoding="utf-8") as f:
                f.writelines(f"{b['key']}\t{sha16(b['value_json'])}\n" for b in chunk)
        dead = dead + 1 if n == 0 else 0
        if dead >= 2 and stats["stored"] == 0:
            stats["error"] = "first 2 import chunks stored nothing; stopping (is the ruflo CLI reachable?)"
            break
    stats["exit"] = _exit_code(stats)
    _log_run(cfg, stats)
    return stats
