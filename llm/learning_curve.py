"""`python -m llm learning-curve`: does the local model get better as it trains on more real session data?

Every real (non-synthetic) adapter plus the base model is scored on ONE common held-out set: the current held-out rows
(latest sessions, cluster-aware split from data.split) minus every row any compared adapter was trained on. For each model:
token NLL / perplexity on the held-out responses, and the per-example improvement over the base model with a 95%
cluster-bootstrap CI (clusters = the ingestor's paraphrase clusters, else sessions). Optional deterministic battery
(--battery) scores general reasoning with the fixed scorer (evaluate.det_correct).
Output: models/learning_curve.json (numbers only, no text). Takes the llm lock like every GPU stage.
"""
from __future__ import annotations

import gc
import json
import math
from datetime import datetime, timezone

from . import config, data, stats


def _real_adapters() -> list[str]:
    if not config.ADAPTERS.exists():
        return []
    return sorted(p.name for p in config.ADAPTERS.iterdir() if (p / "meta.json").exists() and not p.name.endswith("-syn"))


def run(vers: list[str] | None = None, battery: bool = False, n_det: int = 20, log=print) -> dict:
    import torch
    from peft import PeftModel
    from . import evaluate as E, model as M
    vers = vers or _real_adapters()
    metas = {v: json.loads((config.ADAPTERS / v / "meta.json").read_text(encoding="utf-8")) for v in vers}
    rows, source = data.load_rows(False)
    smeta = data.load_meta()
    _, held = data.split(rows, smeta)
    trained = set().union(*[set(m["new_ids"]) | set(m["replay_ids"]) for m in metas.values()]) if metas else set()
    held = [r for r in held if r["id"] not in trained]
    if len(held) < 20:
        raise SystemExit(f"only {len(held)} held-out rows untouched by every compared adapter; need >= 20")
    unit = lambda r: (smeta.get(r["id"]) or {}).get("cluster") or r.get("session")  # noqa: E731
    base_name = {m["base"] for m in metas.values()}
    if len(base_name) > 1:
        raise SystemExit(f"adapters use different bases: {base_name}")
    base = config.base_path(next(iter(base_name)) if base_name else None)
    tok = M.tokenizer(base)
    enc = [M.encode(tok, r["instruction"], r["response"]) for r in held]
    log(f"learning curve: {len(vers)} adapters + base on {len(held)} common held-out rows from "
        f"{len({r.get('session') for r in held})} sessions ({source})")
    m = M.load_4bit(base)
    first = True
    for v in vers:
        if first:
            m = PeftModel.from_pretrained(m, config.ADAPTERS / v, adapter_name=v); first = False
        else:
            m.load_adapter(config.ADAPTERS / v, adapter_name=v)
    items = E.det_items(n_det) if battery else []
    bench = E._bench_module() if battery else None

    def score(name: str) -> dict:
        losses = M.example_losses(m, enc)
        out = {"losses": losses, "nll": sum(a for a, _ in losses) / max(1, sum(b for _, b in losses))}
        if battery:
            ans = M.generate_batch(m, tok, [it["prompt"] for it in items], max_new=E.DET_MAX_NEW)   # same settings as eval
            ok = [E.det_correct(bench, it, t) for it, t in zip(items, ans)]
            cats: dict = {}
            for it, o in zip(items, ok):
                c = cats.setdefault(it["cat"], [0, 0]); c[0] += int(o); c[1] += 1
            out["battery"] = {k: {"correct": a, "n": n} for k, (a, n) in cats.items()}
        log(f"  {name}: held-out NLL {out['nll']:.4f} (ppl {math.exp(out['nll']):.2f})"
            + (f"; battery {out['battery']}" if battery else ""))
        return out

    results = {}
    if vers:
        with m.disable_adapter():
            results["BASE"] = score("BASE")
        for v in vers:
            m.set_adapter(v)
            results[v] = score(v)
    else:
        results["BASE"] = score("BASE")
    base_l = results["BASE"]["losses"]
    curve = []
    for name, res in results.items():
        mt = metas.get(name, {})
        groups: dict = {}
        for r, b, c in zip(held, base_l, res["losses"]):
            groups.setdefault(unit(r), []).append(b[0] / max(1, b[1]) - c[0] / max(1, c[1]))
        ci = stats.cluster_bootstrap_ci(list(groups.values()))
        curve.append({"model": name, "train_session_rows": mt.get("n_new", 0) + mt.get("n_replay", 0),
                      "train_anchor_rows": mt.get("n_anchor", 0), "epochs": mt.get("epochs"), "created": mt.get("created"),
                      "heldout_nll": res["nll"], "heldout_ppl": math.exp(res["nll"]),
                      "improvement_vs_base": {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"]},
                      **({"battery": res["battery"]} if battery else {})})
    curve.sort(key=lambda c: (c["train_session_rows"], c["created"] or ""))
    out = {"ts": datetime.now(timezone.utc).isoformat(), "data_source": source, "heldout_rows": len(held),
           "heldout_sessions": len({r.get("session") for r in held}), "clusters": len({unit(r) for r in held}),
           "base": base.name, "curve": curve,
           "note": "per-example improvement = base NLL - model NLL per response token (positive = better); 95% cluster bootstrap"}
    p = config.REPO / "llm" / "models" / "learning_curve.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    del m
    gc.collect()
    torch.cuda.empty_cache()
    log(f"wrote {p}")
    return out
