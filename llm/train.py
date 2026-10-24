"""`python -m llm train`: continual QLoRA round = new rows + replay sample of rows the parent's lineage already saw.
The new adapter starts from the lineage head (llm/lineage.py: the best not-worse candidate, else production, else a fresh
LoRA) and is saved to models/adapters/<ver>/ (versioned, never overwritten). Held-out sessions (latest by time) are never
trained on. Train-only extras: anchors, external open-source rows, self_verified rows (llm/selftrain.py). After SFT, a DPO
pass runs when at least config.DPO_MIN_PAIRS eligible preference pairs exist (llm/dpo.py); its minutes come out of the
same budget."""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from datetime import datetime, timezone

from . import config, data

MAX_NEW_PER_ROUND = 1500   # oldest-first cap so one nightly round stays bounded (~1-2 h on the 3B)
LR, CONTINUE_LR = 2e-4, 1e-4   # a round on top of an existing adapter moves it less (forgetting)


def chain_ids(ver: str | None, key: str) -> set[str]:
    """union of meta[key] along ver's parent chain (key new_ids: what the lineage was trained on as new rows)."""
    ids: set[str] = set()
    seen: set[str] = set()
    while ver and ver not in seen:
        seen.add(ver)
        try:
            meta = json.loads((config.ADAPTERS / ver / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            break
        ids |= set(meta.get(key) or [])
        ver = meta.get("parent")
    return ids


def sample_self(n: int, seen: set[str], seed: int) -> list[dict]:
    """up to n self_verified rows (train-only), the ones this lineage has not seen first."""
    rows = data.read_jsonl(config.SELF_VERIFIED)
    if n <= 0 or not rows:
        return []
    random.Random(seed).shuffle(rows)
    rows.sort(key=lambda r: r["id"] in seen)
    return rows[:n]


def run(synthetic: bool = False, epochs: int = 2, max_new: int = MAX_NEW_PER_ROUND, anchor_ratio: float = config.ANCHOR_RATIO,
        budget_minutes: float | None = None, log=print) -> dict | None:
    """budget_minutes: GPU minutes for SFT + DPO (default config.TRAIN_MINUTES; the nightly passes what selftrain left)."""
    from . import anchors as A, lineage as L
    rows, source = data.load_rows(synthetic)
    train_rows, held = data.split(rows, data.load_meta() if not synthetic else None)
    st = config.load_state()
    base = config.base_path()
    if synthetic:
        parent, why = (st.get("synthetic_production") or {}).get("ver"), "synthetic production"
        lineage = set(st.get("synthetic_trained_ids", []))
    else:
        parent, why = L.parent_for_training(st, base.name)
        lineage = chain_ids(parent, "new_ids")
    meta = data.load_meta() if not synthetic else {}
    new, replay = data.select_for_round(train_rows, lineage, weights={k: v.get('replay_weight', 1.0) for k, v in meta.items()} or None)
    new.sort(key=lambda r: r.get("ts") or "")
    new = new[:max_new]
    anc = [] if synthetic else A.sample(int(round(len(new) * anchor_ratio)), seed=config.SEED + len(lineage))
    ext = [] if synthetic else data.sample_external(config.EXTERNAL_PER_NIGHT, seed=config.SEED + len(lineage) + len(new))
    slf = [] if synthetic else sample_self(config.SELF_PER_NIGHT, chain_ids(parent, "self_ids"), seed=config.SEED + len(lineage))
    log(f"data: {source}; rows {len(rows)} (train {len(train_rows)}, held-out {len(held)} in later sessions); new {len(new)}, replay {len(replay)}, "
        f"anchors {len(anc)} (ratio {anchor_ratio}, pool {len(A.load())}); external {len(ext)}, self_verified {len(slf)} (train-only)")
    if not new and not ext and not slf:
        log("nothing new to train on; no adapter written")
        return None
    from . import dpo, judge, model as M
    judge.unload()  # free the judge's VRAM
    ver = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ("-syn" if synthetic else "")
    out = config.ADAPTERS / ver
    tok = M.tokenizer(base)
    enc = [M.encode(tok, r["instruction"], r["response"]) for r in new + replay + anc + ext + slf]
    pairs = [] if synthetic else dpo.eligible_pairs(train_rows, held, log=log)
    budget = None if synthetic else (budget_minutes if budget_minutes is not None else config.TRAIN_MINUTES)
    dpo_min = min(config.DPO_MINUTES, budget / 3) if budget and len(pairs) >= config.DPO_MIN_PAIRS else 0.0
    lr = CONTINUE_LR if parent else LR
    log(f"base {base.name}; parent adapter {parent or '(fresh LoRA)'} [{why}]; {len(enc)} examples, mean {sum(len(e[0]) for e in enc) / len(enc):.0f} tokens; "
        f"lr {lr}; budget {budget} min (DPO {dpo_min}: {len(pairs)} eligible pairs, threshold {config.DPO_MIN_PAIRS})")
    t0 = time.time()
    m = M.load_4bit(base, adapter=(config.ADAPTERS / parent) if parent else None, trainable=True)
    stats = M.train_loop(m, enc, epochs=max(epochs, config.NIGHTLY_EPOCHS), lr=lr, log=log,
                         time_budget_s=None if budget is None else (budget - dpo_min) * 60)
    if dpo_min > 0:
        dpo_stats = dpo.train(m, tok, pairs, minutes=dpo_min, log=log)
    else:
        dpo_stats = {"status": "off", "eligible": len(pairs),
                     "reason": "synthetic" if synthetic else f"{len(pairs)} eligible pairs < {config.DPO_MIN_PAIRS}"}
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out)
    meta = {"ver": ver, "base": base.name, "parent": parent, "parent_why": why, "data_source": source, "synthetic": synthetic,
            "n_new": len(new), "n_replay": len(replay), "new_ids": [r["id"] for r in new], "replay_ids": [r["id"] for r in replay],
            "n_anchor": len(anc), "anchor_ratio": anchor_ratio, "anchor_ids": [r["id"] for r in anc],
            "n_external": len(ext), "external_sources": dict(Counter(r.get("source") for r in ext)),
            "n_self": len(slf), "self_ids": [r["id"] for r in slf], "self_sources": dict(Counter(r.get("checker") for r in slf)),
            "heldout_sessions": sorted({r.get("session") for r in held}), "epochs": epochs, "lr": lr, "budget_minutes": budget,
            "train": stats, "dpo": dpo_stats, "wall_s": round(time.time() - t0, 1), "created": datetime.now(timezone.utc).isoformat()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"adapter {out} ({stats}; dpo {dpo_stats.get('status')})")
    return meta
