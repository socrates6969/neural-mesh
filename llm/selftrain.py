"""(B) Rejection-sampling self-training: `python -m llm selftrain [--minutes N] [--k K]`, or the optional nightly stage
(config.SELFTRAIN_MINUTES > 0, minutes taken out of the training budget).

The current model (the adapter the next round starts from: llm/lineage.py) samples k answers per prompt; an AUTOMATIC
checker grades each one; only verified-correct answers become training rows (source "self_verified", train-only, never held
out, data/lifelong/sft_self_verified.jsonl). A prompt with both a correct and a wrong sample also yields a checker-verified
preference pair for DPO (data/lifelong/dpo_pairs.jsonl, llm/dpo.py). Nothing here is graded by an LLM.

Prompt pool (pool()), all with gold answers:
* MetaMathQA rows of the train-only external set whose reference ends "The answer is: <number>" (numeric check);
* agent-catalog benchmarks (MESH_SELFTRAIN_BENCH, default the hive run dir), "tune" split ONLY (its "held" split stays
  untouched): int / numeric / mcq / cwe answers via train/check.py, and function-level code via its Docker sandbox
  (skipped without Docker; never run unsandboxed). Competitive stdio tasks are left out (too slow to grade per sample).
Excluded: every request the gate can score on (current held-out rows, the fixed open-source eval set) and anything sharing an
8-word shingle with them or with the deterministic battery (dpo.eval_bank)."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config, data

BENCH = Path(os.environ.get("MESH_SELFTRAIN_BENCH", "C:/Users/mariu/ruv/hive/runs/agent-catalog/bench"))
BENCH_FILES = ("math", "math_v1_saturated", "math_v2_saturated", "physics", "neural-networks", "cybersecurity",
               "computer-science", "coding_v1_saturated")
KINDS = ("int", "numeric", "mcq", "cwe", "code")
SEEN = None   # default config.MODELS / "selftrain_seen.json": task id -> times sampled (least-tried prompts go first)


def _seen_path() -> Path:
    return SEEN or config.MODELS / "selftrain_seen.json"
FORMAT = "\n\nThink step by step. Give the final answer on the last line in the form `ANSWER: <answer>`."
CODE_FORMAT = "\n\nReturn the complete solution in one ```python code block."
_ANS_MMQA = re.compile(r"The answer is:?\s*\$?(-?[\d,]*\.?\d+)\$?\s*$")
NIGHT = (3 * 60, 8 * 60 + 30)                    # the nightly owns the GPU 03:00-08:30 local


def checker():
    """train/check.py (agent-catalog graders); adds the cwe checker when this checkout's copy predates it."""
    if str(config.REPO) not in sys.path:
        sys.path.insert(0, str(config.REPO))
    from train import check as C
    if not hasattr(C, "check_cwe"):
        def check_cwe(pred, gold):
            m = re.search(r"CWE[-\s_]?(\d+)", (pred or "").upper())
            g = re.search(r"CWE[-\s_]?(\d+)", (gold or "").upper())
            return bool(m and g and m.group(1) == g.group(1))
        C.check_cwe = check_cwe
    return C


def grade(C, task: dict, text: str) -> bool:
    k = task["kind"]
    if k == "code":
        return bool(C.check_code(text, task))
    fn = {"int": C.check_int, "numeric": C.check_numeric, "mcq": C.check_mcq, "cwe": C.check_cwe}[k]
    return bool(fn(C.final_answer(text), task["answer"]))


def _mmqa() -> list[dict]:
    out = []
    for r in data.read_jsonl(config.EXTERNAL):
        if r.get("source") != "oss:metamathqa":
            continue
        m = _ANS_MMQA.search(r["response"].strip())
        if not m:
            continue
        g = m.group(1).replace(",", "")
        out.append({"id": "mmqa:" + r["id"], "kind": "int" if re.fullmatch(r"-?\d+", g) else "numeric", "answer": g,
                    "prompt": r["instruction"], "origin": "oss:metamathqa"})
    return out


def _bench(docker: bool) -> list[dict]:
    out = []
    for f in BENCH_FILES:
        p = BENCH / (f + ".jsonl")
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            t = json.loads(line)
            if t.get("split") != "tune" or t.get("kind") not in KINDS or (t["kind"] == "code" and not docker):
                continue
            out.append(dict(t, id=f"bench:{f}:{t['id']}", origin=f"agent-catalog:{f}"))
    return out


def instruction(task: dict) -> str:
    p = task["prompt"]
    if task["kind"] == "code":
        return p if "```" in p else p + CODE_FORMAT
    return p if "ANSWER" in p else p + FORMAT


def pool(seed: int = config.SEED, log=print) -> list[dict]:
    import shutil
    from . import anchors as A, dpo
    rows = data.read_jsonl(config.SFT)
    _, held = data.split(rows, data.load_meta())
    ids, bank = dpo.eval_bank(held)
    docker = bool(shutil.which("docker"))
    raw = _bench(docker) + _mmqa()
    out = [t for t in raw if t["id"] not in ids and not A.leaks(t["prompt"], bank)]
    log(f"selftrain pool: {len(out)} prompts ({len(raw) - len(out)} dropped as eval/battery overlap; "
        f"docker {'on' if docker else 'off: code tasks skipped'})")
    try:
        seen = json.loads(_seen_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    random.Random(seed).shuffle(out)
    out.sort(key=lambda t: seen.get(t["id"], 0))          # least-sampled first, random within a tie
    return out


def in_night_window(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    m = now.hour * 60 + now.minute
    return NIGHT[0] <= m < NIGHT[1]


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def run(minutes: float, k: int = 4, max_new: int = 768, temperature: float = 0.8, gen_bs: int = 8, nightly: bool = True,
        tasks: list[dict] | None = None, log=print) -> dict:
    """samples until the minute budget is spent. Returns stats incl. 'minutes' actually used (the nightly subtracts them)."""
    t0 = time.time()
    if minutes <= 0:
        return {"kept": 0, "minutes": 0.0, "reason": "selftrain off (0 minutes)"}
    if not nightly and in_night_window():
        return {"kept": 0, "minutes": 0.0, "reason": "refused: 03:00-08:30 belongs to the nightly"}
    import torch
    from . import judge, lineage as L, model as M
    C = checker()
    tasks = tasks if tasks is not None else pool(log=log)
    if not tasks:
        return {"kept": 0, "minutes": round((time.time() - t0) / 60, 2), "reason": "empty prompt pool"}
    base = config.base_path()
    parent, why = L.parent_for_training(config.load_state(), base.name)
    judge.unload()
    tok = M.tokenizer(base)
    m = M.load_4bit(base, adapter=(config.ADAPTERS / parent) if parent else None)
    model_tag = f"{base.name}+{parent or 'base'}"
    log(f"selftrain: sampling {model_tag} [{why}], k={k}, T={temperature}, budget {minutes} min")
    try:
        seen = json.loads(_seen_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        seen = {}
    have = {r["id"] for r in data.read_jsonl(config.SELF_VERIFIED)}
    per = max(1, gen_bs // k)
    stats = {"prompts": 0, "samples": 0, "correct": 0, "kept": 0, "pairs": 0, "by_kind": {}}
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for i in range(0, len(tasks), per):
        if time.time() - t0 > minutes * 60:
            break
        chunk = tasks[i:i + per]
        texts = M.sample_batch(m, tok, [instruction(t) for t in chunk], k=k, max_new=max_new, temperature=temperature, bs=gen_bs)
        rows, pairs = [], []
        for j, t in enumerate(chunk):
            outs = texts[j * k:(j + 1) * k]
            ok = [grade(C, t, o) for o in outs]
            seen[t["id"]] = seen.get(t["id"], 0) + 1
            b = stats["by_kind"].setdefault(t["kind"], {"prompts": 0, "samples": 0, "correct": 0, "kept": 0})
            b["prompts"] += 1
            b["samples"] += len(outs)
            b["correct"] += sum(ok)
            stats["prompts"] += 1
            stats["samples"] += len(outs)
            stats["correct"] += sum(ok)
            good = [o for o, g in zip(outs, ok) if g]
            bad = [o for o, g in zip(outs, ok) if not g and o.strip()]
            if not good:
                continue
            best = min(good, key=len)                       # shortest verified answer: least room for wrong reasoning
            rid = "selfv:" + _hash(t["id"], best)
            if rid not in have:
                have.add(rid)
                rows.append({"id": rid, "ts": ts, "project": "self", "session": "self_verified", "source": "self_verified",
                             "agent": None, "model": model_tag, "instruction": instruction(t), "response": best, "outcome": "success",
                             "judge_score": None, "checker": t["kind"], "task_id": t["id"], "origin": t["origin"],
                             "gold": t.get("answer"), "pass_k": f"{sum(ok)}/{len(outs)}"})
                b["kept"] += 1
                stats["kept"] += 1
            if bad:
                pairs.append({"id": "dpos:" + _hash(t["id"], best, bad[0]), "row_id": t["id"], "instruction": instruction(t),
                              "chosen": best, "rejected": bad[0], "source": "checker", "checker": t["kind"], "model": model_tag, "ts": ts})
        for path, recs in ((config.SELF_VERIFIED, rows), (config.DPO_PAIRS, pairs)):
            if recs:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        stats["pairs"] += len(pairs)
        if stats["prompts"] % (per * 10) == 0:
            log(f"selftrain: {stats['prompts']} prompts, pass {stats['correct']}/{stats['samples']}, kept {stats['kept']}, pairs {stats['pairs']}")
    _seen_path().parent.mkdir(parents=True, exist_ok=True)
    _seen_path().write_text(json.dumps(seen), encoding="utf-8")
    del m
    torch.cuda.empty_cache()
    stats.update(minutes=round((time.time() - t0) / 60, 2), model=model_tag, k=k, temperature=temperature,
                 reason=f"pass@1 {stats['correct'] / max(1, stats['samples']):.3f} on {stats['prompts']} prompts")
    log(f"selftrain: {stats}")
    return stats
