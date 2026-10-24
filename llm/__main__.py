"""python -m llm {bench|train|eval|promote|status|frontend-eval|nightly|calibrate|tool-exemplars} [--synthetic] [--ver V] [--epochs N] [--judge-budget N]
   python -m llm ask ...           front-end wrapper (S2, llm/frontend_live.py; pass-through unless --enabled; never takes the lock)
   python -m llm hook --shadow     UserPromptSubmit shadow hook (S1, llm/hook.py; prints nothing, exit 0)
   python -m llm shadow-drain      offline drain of the shadow queue (S1, under the lock)"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__" and sys.argv[1:2] == ["hook"]:   # S1 hook fast path: nothing else is imported, never returns
    from .hook import cli as _hook_cli
    _hook_cli(sys.argv[2:])

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["hook"]:
        from .hook import cli as hook_cli
        hook_cli(argv[1:])
    if argv[:1] == ["ask"]:          # dispatched BEFORE lock(): ask must never wait for or block the nightly
        from .frontend_live import cli_ask
        return cli_ask(argv[1:])
    if argv[:1] == ["external"]:      # refresh the train-only open-source data (data/lifelong/sft_external.jsonl)
        from . import external
        if "--if-stale" in argv and not external.stale():
            print("external: fresh (< 7 days); skipped")
            return 0
        external.build_eval() if "--eval" in argv else external.build()
        return 0
    if argv[:1] == ["shadow-drain"]:
        from .frontend_live import shadow_drain
        with lock():
            shadow_drain()
        return 0
    ap = argparse.ArgumentParser(prog="python -m llm")
    ap.add_argument("cmd", choices=["bench", "train", "eval", "promote", "status", "frontend-eval", "nightly", "calibrate", "tool-exemplars", "anchors", "learning-curve", "smoke", "selftrain", "dpo-pairs"])
    ap.add_argument("--synthetic", action="store_true", help="use the LABELLED synthetic sample instead of data/lifelong/sft.jsonl")
    ap.add_argument("--ver")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--battery", action="store_true", help="learning-curve: also score the deterministic battery (slow)")
    ap.add_argument("--anchor-ratio", type=float, default=None, help="train: anchor rows per new row (default config.ANCHOR_RATIO; 0 disables)")
    ap.add_argument("--judge-budget", type=int, default=None, help="hard cap on Haiku judge calls today (eval default 120, calibrate default 200)")
    ap.add_argument("--to", metavar="VER", help="promote: roll production back to an earlier promoted version")
    ap.add_argument("--base", help="smoke: candidate base under models/base/ (default: the configured base)")
    ap.add_argument("--steps", type=int, default=30, help="smoke: optimizer steps")
    ap.add_argument("--gen-bs", type=int, default=8, help="smoke: generation batch size")
    ap.add_argument("--out", help="smoke: result JSON path (default models/smoke_<base>.json)")
    ap.add_argument("--minutes", type=float, default=30.0, help="selftrain: GPU minute budget")
    ap.add_argument("--k", type=int, default=4, help="selftrain: samples per prompt")
    ap.add_argument("--keep-merged", action="store_true", help="promote: keep models/merged/<ver> (default: removed, also on failure)")
    a = ap.parse_args(argv)
    if a.cmd == "smoke":            # one candidate base, a real short QLoRA run + generation pass (llm/smoke.py)
        from . import smoke
        with lock():
            smoke.run(a.base or config.base_name(), steps=a.steps, gen_bs=a.gen_bs, out=Path(a.out) if a.out else None)
        return 0
    if a.cmd == "dpo-pairs":         # count the eligible preference pairs (no GPU, no lock)
        from . import data, dpo
        rows = data.read_jsonl(config.SFT)
        tr, held = data.split(rows, data.load_meta())
        n = len(dpo.eligible_pairs(tr, held))
        print(json.dumps({"eligible": n, "threshold": config.DPO_MIN_PAIRS, "dpo_on": n >= config.DPO_MIN_PAIRS}))
        return 0
    if a.cmd == "status":
        print(json.dumps(status(), indent=1)); return 0
    if a.cmd == "tool-exemplars":   # MiniLM only, no GPU / Ollama: no lock
        from . import frontend
        frontend.build_tool_exemplars(); return 0
    rc = 0
    with lock():
        if a.cmd == "bench":
            from . import bench; bench.run()
        elif a.cmd == "train":
            from . import config as C, train
            train.run(synthetic=a.synthetic, epochs=a.epochs, anchor_ratio=C.ANCHOR_RATIO if a.anchor_ratio is None else a.anchor_ratio)
        elif a.cmd == "selftrain":   # refuses inside 03:00-08:30 (the nightly runs it as a stage instead)
            from . import selftrain
            r = selftrain.run(minutes=a.minutes, k=a.k, nightly=False)
            rc = 0 if not str(r.get("reason", "")).startswith("refused") else 75
        elif a.cmd == "learning-curve":
            from . import learning_curve; learning_curve.run(battery=a.battery)
        elif a.cmd == "anchors":
            from . import anchors; anchors.build(n=a.n if a.n != 80 else 320)
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
            from . import frontend; frontend.evaluate(synthetic=a.synthetic)
        elif a.cmd == "nightly":   # preflight sweep -> train -> eval -> promote under ONE lock; RunRecord -> models/runs.jsonl
            from . import nightly
            rc = nightly.run(synthetic=a.synthetic, epochs=a.epochs, judge_budget=a.judge_budget)["exit"]
    return rc


if __name__ == "__main__":
    sys.exit(main())
