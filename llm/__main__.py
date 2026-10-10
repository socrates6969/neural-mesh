"""python -m llm {bench|train|eval|promote|status|frontend-eval|nightly|calibrate} [--synthetic] [--ver V] [--epochs N] [--judge-budget N]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

from . import config


@contextmanager
def lock(stale_s: float = 12 * 3600):
    """exclusive lock file (O_EXCL); a lock older than stale_s (crashed run) is taken over. Exit 75 if busy."""
    config.MODELS.mkdir(parents=True, exist_ok=True)
    p = config.LOCK
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time():.0f}\n".encode()); os.close(fd)
            break
        except FileExistsError:
            try:
                if time.time() - p.stat().st_mtime > stale_s:
                    p.unlink(); continue
            except FileNotFoundError:
                continue
            print(f"busy: {p} exists (another llm run); exiting", file=sys.stderr)
            sys.exit(75)
    try:
        yield
    finally:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def status() -> dict:
    from . import data
    st = config.load_state()
    rows = data.read_jsonl(config.SFT)
    train, held = data.split(rows, data.load_meta())
    advs = sorted(p.name for p in config.ADAPTERS.iterdir()) if config.ADAPTERS.exists() else []
    return {"base": config.base_name(), "sft_rows": len(rows), "train_rows": len(train), "heldout_rows": len(held),
            "trained_in_production_lineage": len(st.get("trained_ids", [])), "production": st.get("production"),
            "synthetic_production": st.get("synthetic_production"), "adapters": advs, "last_history": st.get("history", [])[-3:],
            "locked": config.LOCK.exists()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m llm")
    ap.add_argument("cmd", choices=["bench", "train", "eval", "promote", "status", "frontend-eval", "nightly", "calibrate"])
    ap.add_argument("--synthetic", action="store_true", help="use the LABELLED synthetic sample instead of data/lifelong/sft.jsonl")
    ap.add_argument("--ver")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--judge-budget", type=int, default=None, help="hard cap on Haiku judge calls today (eval default 120, calibrate default 200)")
    ap.add_argument("--to", metavar="VER", help="promote: roll production back to an earlier promoted version")
    ap.add_argument("--keep-merged", action="store_true", help="promote: keep models/merged/<ver> (default: removed, also on failure)")
    a = ap.parse_args(argv)
    if a.cmd == "status":
        print(json.dumps(status(), indent=1)); return 0
    rc = 0
    with lock():
        if a.cmd == "bench":
            from . import bench; bench.run()
        elif a.cmd == "train":
            from . import train; train.run(synthetic=a.synthetic, epochs=a.epochs)
        elif a.cmd == "eval":
            from . import evaluate; evaluate.run(synthetic=a.synthetic, ver=a.ver, judge_budget=a.judge_budget)
        elif a.cmd == "promote":
            from . import promote
            if a.to:
                rc = promote.rollback(a.to, keep_merged=a.keep_merged).get("exit", 0)
            else:
                rc = promote.run(synthetic=a.synthetic, ver=a.ver, keep_merged=a.keep_merged).get("exit", 0)
        elif a.cmd == "calibrate":
            from . import calibrate; calibrate.run(budget=a.judge_budget or 200)
        elif a.cmd == "frontend-eval":
            from . import frontend; frontend.evaluate(synthetic=a.synthetic, n=a.n)
        elif a.cmd == "nightly":   # preflight sweep -> train -> eval -> promote under ONE lock; RunRecord -> models/runs.jsonl
            from . import nightly
            rc = nightly.run(synthetic=a.synthetic, epochs=a.epochs, judge_budget=a.judge_budget)["exit"]
    return rc


if __name__ == "__main__":
    sys.exit(main())
