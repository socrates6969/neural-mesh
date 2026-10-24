"""(D) Continual lineage: each night trains from the best previous candidate instead of from the base.

Terms. INCUMBENT = the production adapter (or the bare base when nothing was ever promoted). Every candidate is evaluated
against the INCUMBENT, never against its parent (evaluate.py is unchanged), and promotion still needs the unchanged gate.
HEAD = state["lineage"]: the adapter the next round starts from.

Which candidate becomes HEAD (record(), called by promote.run once a candidate is decided):
* promoted -> HEAD = it, depth 0 (it is production now).
* rejected -> HEAD = it only if it is NOT WORSE than the incumbent, judged by its own eval.json:
    - the pairwise judge was trusted and judged >= evaluate.MIN_PAIR_ITEMS items, and
    - the pairwise win rate POINT ESTIMATE vs the incumbent is >= 0.5 (it may be a statistical tie, never a loss), and
    - no deterministic-battery regression (the gate's own per-category + pooled rule: evaluate.det_regressions), and
    - the held-out loss is not significantly worse (cluster-bootstrap CI upper bound >= 0).
  Otherwise HEAD stays where it was (the worse candidate is a dead end; its adapter is kept on disk for the record).
Because each HEAD was itself compared with the INCUMBENT, not with the previous HEAD, quality cannot drift down step by step:
every HEAD is individually not-worse than production. MAX_CHAIN caps the number of unpromoted generations stacked on top of
production; after that the next round starts from production again (depth reset), so the chain cannot run away for weeks
on statistical ties.

Forgetting: a HEAD round trains on the rows its chain never saw (new) plus a replay sample of rows it did see
(data.select_for_round, REPLAY_RATIO), the self-distillation anchors, and the train-only external / self-verified rows,
at a lower learning rate (train.CONTINUE_LR)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def head(st: dict) -> dict | None:
    return st.get("lineage") or None


def _adapter_ok(ver: str | None) -> bool:
    return bool(ver) and (config.ADAPTERS / ver / "meta.json").exists()


def parent_for_training(st: dict, base: str) -> tuple[str | None, str]:
    """(parent ver or None for a fresh LoRA, why)."""
    prod = st.get("production") or {}
    prod_ok = bool(prod.get("ver")) and prod.get("base", base) == base and _adapter_ok(prod.get("ver"))
    if config.CONTINUAL:
        h = head(st)
        if h and h.get("base") == base and _adapter_ok(h.get("ver")) and h.get("ver") != prod.get("ver"):
            if int(h.get("depth", 0)) < config.MAX_CHAIN:
                return h["ver"], f"lineage head {h['ver']} (depth {h.get('depth', 0)}, {h.get('reason')})"
            return (prod["ver"] if prod_ok else None), f"chain reset: head {h['ver']} reached depth {h.get('depth')} >= {config.MAX_CHAIN}"
    if prod_ok:
        return prod["ver"], "production"
    return None, "fresh LoRA (no production on this base)" if not prod.get("ver") else f"fresh LoRA (production is on {prod.get('base')})"


def valid_parent(st: dict, parent: str | None, base: str) -> bool:
    """a pending candidate is still worth evaluating when it was trained from what a round would start from today, or from
    production itself."""
    prod = (st.get("production") or {}).get("ver")
    return parent == parent_for_training(st, base)[0] or (parent is not None and parent == prod)


def not_worse(ev: dict) -> tuple[bool, str]:
    from .evaluate import MIN_PAIR_ITEMS
    if not (ev.get("judge_trusted") or {}).get("ok", False):
        return False, "pairwise judge not trusted"
    pair = ev.get("pairwise") or {}
    if pair.get("n_judged", 0) < MIN_PAIR_ITEMS:
        return False, f"too few judged items ({pair.get('n_judged', 0)} < {MIN_PAIR_ITEMS})"
    w = (pair.get("winrate_ci") or {}).get("mean")
    if w is None or w < 0.5:
        return False, f"pairwise win rate {w} < 0.5 vs the incumbent"
    reg = (ev.get("deterministic") or {}).get("regressed")
    if reg is None or reg:
        return False, f"deterministic battery regression in {reg}"
    lo = (ev.get("loss") or {}).get("diff_cluster_ci") or {}
    if lo.get("n") and lo.get("hi") is not None and lo["hi"] < 0:
        return False, "held-out loss significantly worse"
    return True, f"not worse: win rate {w:.3f} >= 0.5, no battery regression, loss not significantly worse"


def record(st: dict, ver: str, promoted: bool, log=print) -> dict:
    """mutates st: maybe moves HEAD to ver; appends st['lineage_log']. Call once per decided candidate."""
    try:
        meta = json.loads((config.ADAPTERS / ver / "meta.json").read_text(encoding="utf-8"))
        ev = json.loads((config.ADAPTERS / ver / "eval.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta, ev = {}, {}
    old = head(st) or {}
    if promoted:
        ok, why = True, "promoted"
        depth = 0
    else:
        ok, why = not_worse(ev)
        depth = (int(old.get("depth", 0)) + 1) if meta.get("parent") and meta.get("parent") == old.get("ver") else 1
    rec = {"ver": ver, "parent": meta.get("parent"), "base": meta.get("base"), "became_head": ok, "why": why, "ts": _now()}
    if ok:
        st["lineage"] = {"ver": ver, "base": meta.get("base"), "depth": depth, "reason": why, "since": rec["ts"]}
        log(f"lineage: head -> {ver} (depth {depth}; {why})")
    else:
        log(f"lineage: head stays {old.get('ver') or '(none)'}; {ver} is a dead end ({why})")
    st.setdefault("lineage_log", []).append(rec)
    return rec


def reset_to(st: dict, ver: str, base: str, reason: str) -> None:
    """after a rollback: the next round starts from the rolled-back production."""
    st["lineage"] = {"ver": ver, "base": base, "depth": 0, "reason": reason, "since": _now()}
    st.setdefault("lineage_log", []).append({"ver": ver, "became_head": True, "why": reason, "ts": _now()})
