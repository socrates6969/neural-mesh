"""`python -m llm train`: continual QLoRA round = new rows + replay sample of rows the production lineage already saw.
The new adapter starts from the production adapter (or a fresh LoRA) and is saved to models/adapters/<ver>/ (versioned,
never overwritten). Held-out sessions (latest by time) are never trained on."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from . import config, data

MAX_NEW_PER_ROUND = 1500   # oldest-first cap so one nightly round stays bounded (~1-2 h on the 3B)


def run(synthetic: bool = False, epochs: int = 2, max_new: int = MAX_NEW_PER_ROUND, anchor_ratio: float = config.ANCHOR_RATIO,
        log=print) -> dict | None:
    rows, source = data.load_rows(synthetic)
    train_rows, held = data.split(rows, data.load_meta() if not synthetic else None)
    st = config.load_state()
    lineage = set(st.get("trained_ids", [])) if not synthetic else set(st.get("synthetic_trained_ids", []))
    meta = data.load_meta() if not synthetic else {}
    new, replay = data.select_for_round(train_rows, lineage, weights={k: v.get('replay_weight', 1.0) for k, v in meta.items()} or None)
    new.sort(key=lambda r: r.get("ts") or "")
    new = new[:max_new]
    from . import anchors as A
    anc = [] if synthetic else A.sample(int(round(len(new) * anchor_ratio)), seed=config.SEED + len(lineage))
    ext = [] if synthetic else data.sample_external(config.EXTERNAL_PER_NIGHT, seed=config.SEED + len(lineage) + len(new))
    log(f"data: {source}; rows {len(rows)} (train {len(train_rows)}, held-out {len(held)} in later sessions); new {len(new)}, replay {len(replay)}, "
        f"anchors {len(anc)} (ratio {anchor_ratio}, pool {len(A.load())}); external {len(ext)} (train-only)")
    if not new and not ext:
        log("nothing new to train on; no adapter written")
        return None
    from . import judge, model as M
    judge.unload()  # free the judge's VRAM
    base = config.base_path()
    prod = (st.get("production") or {}) if not synthetic else (st.get("synthetic_production") or {})
    parent = prod.get("ver")
    ver = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ("-syn" if synthetic else "")
    out = config.ADAPTERS / ver
    tok = M.tokenizer(base)
    enc = [M.encode(tok, r["instruction"], r["response"]) for r in new + replay + anc + ext]
    log(f"base {base.name}; parent adapter {parent or '(fresh LoRA)'}; {len(enc)} examples, mean {sum(len(e[0]) for e in enc) / len(enc):.0f} tokens")
    t0 = time.time()
    m = M.load_4bit(base, adapter=(config.ADAPTERS / parent) if parent else None, trainable=True)
    stats = M.train_loop(m, enc, epochs=max(epochs, config.NIGHTLY_EPOCHS), log=log,
                         time_budget_s=None if synthetic else config.TRAIN_MINUTES * 60)
    out.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out)
    meta = {"ver": ver, "base": base.name, "parent": parent, "data_source": source, "synthetic": synthetic,
            "n_new": len(new), "n_replay": len(replay), "new_ids": [r["id"] for r in new], "replay_ids": [r["id"] for r in replay],
            "n_anchor": len(anc), "anchor_ratio": anchor_ratio, "anchor_ids": [r["id"] for r in anc],
            "n_external": len(ext), "external_sources": dict(__import__("collections").Counter(r.get("source") for r in ext)),
            "heldout_sessions": sorted({r.get("session") for r in held}), "epochs": epochs, "train": stats,
            "wall_s": round(time.time() - t0, 1), "created": datetime.now(timezone.utc).isoformat()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"adapter {out} ({stats})")
    return meta
