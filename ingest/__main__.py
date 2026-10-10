"""CLI: python -m ingest run | status | recall "<query>"   (run from the neural-mesh repo root)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import Config


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m ingest", description="Lifelong ingestion of Claude Code transcripts into neural-mesh.")
    p.add_argument("--projects", help="transcript root (default ~/.claude/projects)")
    p.add_argument("--data", help="output root (default <repo>/data/lifelong)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="incremental ingestion")
    r.add_argument("--settle-min", type=float, help="minutes of idleness before an agent/turn counts as finished (default 20)")
    r.add_argument("--judge-budget", type=int, help="max NEW judge calls this run (default 300)")
    r.add_argument("--no-judge", action="store_true")
    r.add_argument("--no-mesh", action="store_true", help="do not call neural-mesh feedback")
    r.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the lock (default 0: skip if busy)")
    r.add_argument("--quiet", action="store_true")
    sub.add_parser("status")
    q = sub.add_parser("recall", help="retrieve useful memories (logged for usefulness credit)")
    q.add_argument("query")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--session", default=os.environ.get("CLAUDE_SESSION_ID"))
    q.add_argument("--no-log", action="store_true")
    m = sub.add_parser("mirror", help="copy scrubbed memories/outcomes into ruflo's LOCAL memory (namespace vector-patterns)")
    m.add_argument("--dry-run", action="store_true", help="write mirror_batch.jsonl only; call nothing")
    m.add_argument("--limit", type=int, help="max new records this run")
    m.add_argument("--db-path", default=None,
                   help="ruflo memory.db to write (default: env NEURAL_MESH_RUFLO_DB, else <repo>/.swarm/memory.db); must resolve inside the repo")
    m.add_argument("--allow-external-db", action="store_true", help="allow a db that resolves outside the repo (UNC / network drives stay refused)")
    m.add_argument("--wait", type=float, default=0.0, help="seconds to wait for the ingest lock (default 0: exit 75 if busy)")
    a = p.parse_args(argv)

    cfg = Config()
    if a.projects:
        cfg.projects = Path(a.projects)
    if a.data:
        cfg.data = Path(a.data)

    if a.cmd == "run":
        from .pipeline import run
        if a.settle_min is not None:
            cfg.settle_minutes = a.settle_min
        if a.judge_budget is not None:
            cfg.judge_budget = a.judge_budget
        if a.no_judge:
            cfg.judge_budget = 0
        if a.no_mesh:
            cfg.mesh_feedback = False
        res = run(cfg, wait_s=a.wait, quiet=a.quiet)
        if not a.quiet:
            print(json.dumps(res, indent=1))
        return 0
    if a.cmd == "status":
        from .pipeline import status
        print(json.dumps(status(cfg), indent=1))
        return 0
    if a.cmd == "recall":
        from .memory import MemoryStore
        print(json.dumps(MemoryStore(cfg).recall(a.query, a.k, a.session, log=not a.no_log), indent=1, ensure_ascii=False))
        return 0
    if a.cmd == "mirror":
        from .mirror import mirror
        stats = mirror(cfg, dry_run=a.dry_run, limit=a.limit, db_path=a.db_path, allow_external=a.allow_external_db, wait_s=a.wait)
        print(json.dumps(stats, indent=1))
        return int(stats.get("exit", 0))   # 0 all/noop, 3 partial, 4 none stored / npx missing, 5 refused, 75 busy
    return 2


if __name__ == "__main__":
    sys.exit(main())
