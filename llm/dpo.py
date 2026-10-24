"""(C) Preference training after SFT: DPO (Rafailov et al. 2023) on the LoRA adapter, implemented directly on peft (no trl).

Reference model = the adapter as it stands after tonight's SFT pass: its log-probs for every pair are computed once, before
the first DPO update, so policy and reference share one set of weights in VRAM.
loss = -log sigmoid(beta * ((logp_pi(chosen) - logp_ref(chosen)) - (logp_pi(rejected) - logp_ref(rejected)))),
log-probs summed over response tokens only (prompt masked), bs 1, gradient accumulation, time-boxed.

Pairs (eligible_pairs), every one TRAIN-ONLY:
1. judge pairs: pairwise-judge verdicts stored next to old candidates (adapters/<ver>/eval.json detail + answers.json).
   A verdict counts only when both orders agreed ("both-orders") or the $0 structural check settled it; ties never.
   EVERY eval item is excluded: a pair is used only when its request is a CURRENT train row (it was held out when it was
   judged and has since rolled out of the time-based held-out window). Requests from the current held-out split, from the
   fixed open-source eval set (sft_eval_external.jsonl), or sharing an 8-word shingle with any of those or with the
   deterministic battery are dropped, so the gate never sees a request it was optimised on.
2. checker pairs: (verified-correct, verified-wrong) samples of the same prompt from llm/selftrain.py (dpo_pairs.jsonl),
   filtered the same way.
The step switches itself on at config.DPO_MIN_PAIRS eligible pairs (train.py)."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time

from . import config, data

BETA, LR, GRAD_ACCUM, MAX_PAIRS = 0.1, 2e-5, 8, 600


def _h(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8", "replace")).hexdigest()[:16]


def eval_bank(held: list[dict]) -> tuple[set[str], set]:
    """ids and 8-word shingles of everything the gate can be scored on (held-out rows, open-source eval set, battery)."""
    from . import anchors as A
    ev = data.read_jsonl(config.EVAL_EXTERNAL)
    ids = {r["id"] for r in held} | {r["id"] for r in ev}
    bank = A.battery_shingles()
    for r in held + ev:
        bank |= A._shingles(r["instruction"])
    return ids, bank


def judge_pairs(train_by_id: dict[str, dict]) -> list[dict]:
    out = []
    if not config.ADAPTERS.exists():
        return out
    for d in sorted(config.ADAPTERS.iterdir()):
        try:
            ev = json.loads((d / "eval.json").read_text(encoding="utf-8"))
            ans = json.loads((d / "answers.json").read_text(encoding="utf-8"))["answers"]
        except (OSError, ValueError, KeyError):
            continue
        detail = (ev.get("pairwise") or {}).get("detail") or []
        cand, prod = ans.get("task_cand") or [], ans.get("task_prod") or []
        if len(detail) != len(cand) or len(cand) != len(prod):
            continue
        for x, a, b in zip(detail, cand, prod):
            if x.get("winner") not in ("x", "y") or x.get("how") not in ("both-orders", "structural"):
                continue
            row = train_by_id.get(x.get("id"))
            if row is None:
                continue
            ch, rj = (a, b) if x["winner"] == "x" else (b, a)
            if ch.strip() and ch.strip() != rj.strip():
                out.append({"id": "dpoj:" + _h(row["id"], ch, rj), "row_id": row["id"], "instruction": row["instruction"],
                            "chosen": ch, "rejected": rj, "source": "judge", "from_eval_of": d.name})
    return out


def read_pairs(path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue     # torn last line of a concurrent writer
                if isinstance(r, dict) and r.get("id") and r.get("instruction") and r.get("chosen") and r.get("rejected"):
                    out.append(r)
    except FileNotFoundError:
        pass
    return out


def eligible_pairs(train_rows: list[dict], held: list[dict], log=print) -> list[dict]:
    from . import anchors as A
    ids, bank = eval_bank(held)
    raw = (judge_pairs({r["id"]: r for r in train_rows}) if config.DPO_JUDGE_PAIRS else []) + \
          (read_pairs(config.DPO_PAIRS) if config.DPO_SELF_PAIRS else [])
    out, seen, dropped = [], set(), 0
    for p in raw:
        if p["id"] in seen or p.get("row_id") in ids or A.leaks(p["instruction"], bank):
            dropped += 1
            continue
        seen.add(p["id"])
        out.append(p)
    by = {}
    for p in out:
        by[p["source"]] = by.get(p["source"], 0) + 1
    log(f"dpo: {len(out)} eligible pairs {by} ({dropped} dropped: duplicate, eval item or eval/battery overlap)")
    return out


def _seq_logp(model, ids, lab):
    """sum of log-probs of the labelled (tail) tokens, logits only for those positions."""
    import torch
    n = int((lab[0] != -100).sum().item())
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=n + 1).logits[:, :-1].float()
    tgt = lab[:, -n:]
    return torch.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum()


def train(model, tok, pairs: list[dict], minutes: float, beta: float = BETA, lr: float = LR, grad_accum: int = GRAD_ACCUM,
          max_pairs: int = MAX_PAIRS, seed: int = config.SEED, log=print) -> dict:
    import torch
    import bitsandbytes as bnb
    from . import model as M
    t0 = time.time()
    budget = minutes * 60
    rng = random.Random(seed)
    pairs = pairs[:]
    rng.shuffle(pairs)
    pairs = pairs[:max_pairs]
    enc = []
    for p in pairs:
        c = M.encode(tok, p["instruction"], p["chosen"])
        r = M.encode(tok, p["instruction"], p["rejected"])
        enc.append([torch.tensor([x], device="cuda") for x in (*c, *r)])
    model.eval()
    ref = []
    with torch.no_grad():
        for cid, clab, rid, rlab in enc:
            if time.time() - t0 > budget / 3:      # never spend more than a third of the budget on the reference pass
                break
            ref.append((_seq_logp(model, cid, clab).item(), _seq_logp(model, rid, rlab).item()))
    enc = enc[:len(ref)]
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=0.0)
    losses, margins, micro, step = [], [], 0, 0
    for (cid, clab, rid, rlab), (rc, rr) in zip(enc, ref):
        if time.time() - t0 > budget:
            break
        z = beta * ((_seq_logp(model, cid, clab) - rc) - (_seq_logp(model, rid, rlab) - rr))
        loss = -torch.nn.functional.logsigmoid(z)
        (loss / grad_accum).backward()
        losses.append(loss.item())
        margins.append(z.item() / beta)
        micro += 1
        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
    if micro % grad_accum:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
    tail = margins[-max(1, len(margins) // 4):]
    st = {"status": "ok" if micro else "noop", "eligible": len(pairs), "reference_pairs": len(ref), "pairs_trained": micro,
          "optimizer_steps": step if micro else 0, "beta": beta, "lr": lr, "seconds": round(time.time() - t0, 1),
          "mean_loss": round(sum(losses) / len(losses), 4) if losses else None, "start_loss": round(math.log(2), 4),
          "reward_acc_last_quarter": round(sum(m > 0 for m in tail) / len(tail), 3) if margins else None,
          "by_source": {s: sum(1 for p in pairs[:len(ref)] if p["source"] == s) for s in ("judge", "checker")}}
    log(f"dpo: {st}")
    return st
