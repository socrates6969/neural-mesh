"""`python -m llm eval`: candidate adapter vs the current production model on held-out LATER sessions.

(1) held-out loss: per-example mean response-token NLL, paired (production - candidate; > 0 = candidate better),
    95% CI by session-cluster bootstrap (examples inside a session are correlated) and a plain paired t-CI for reference.
(2) task battery: greedy answers to held-out instructions from both models, each graded 0/0.5/1 against Claude's
    actual answer by the local judge (Ollama, temperature 0); paired (candidate - production) with the same CIs.
Gate (promote.py applies it): loss cluster-CI lower bound > 0 AND battery not significantly worse (cluster-CI upper
bound >= 0) AND battery mean diff >= -0.05 AND >= 30 loss examples from >= 3 held-out sessions. Otherwise: not promoted.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
from datetime import datetime, timezone

from . import config, data, stats

N_LOSS = 200      # max held-out examples for the loss test
N_TASK = 40       # max held-out instructions for the judge battery
MIN_N, MIN_SESSIONS = 30, 3   # MIN_SESSIONS counts bootstrap units (ingestor clusters, else sessions)


def _pick(rows: list[dict], n: int, salt: str) -> list[dict]:
    return sorted(rows, key=lambda r: hashlib.sha256((salt + r["id"]).encode()).hexdigest())[:n]


def latest_candidate(synthetic: bool) -> str | None:
    if not config.ADAPTERS.exists():
        return None
    vs = sorted(p.name for p in config.ADAPTERS.iterdir() if (p / "meta.json").exists() and p.name.endswith("-syn") == synthetic)
    return vs[-1] if vs else None


MIN_DECOY_REJECT = 0.7   # the judge must reject >= 70% of hard decoys, else the battery cannot certify "no regression"


def gate(loss_ci: dict, task_ci: dict, n_sessions: int, decoy_reject: float | None = 1.0) -> tuple[bool, str]:
    if decoy_reject is not None and decoy_reject < MIN_DECOY_REJECT:
        return False, f"judge unreliable: rejects only {decoy_reject:.2f} of nearest-but-wrong decoys (< {MIN_DECOY_REJECT})"
    if loss_ci["n"] < MIN_N or n_sessions < MIN_SESSIONS:
        return False, f"insufficient held-out data (n={loss_ci['n']} < {MIN_N} or sessions={n_sessions} < {MIN_SESSIONS})"
    if not loss_ci["lo"] > 0:
        return False, f"no significant held-out loss win (CI [{loss_ci['lo']:.4f}, {loss_ci['hi']:.4f}])"
    if task_ci["n"] and task_ci["hi"] < 0:
        return False, f"significant task-battery regression (CI [{task_ci['lo']:.3f}, {task_ci['hi']:.3f}])"
    if task_ci["n"] and task_ci["mean"] < -0.05:
        return False, f"task-battery mean regression {task_ci['mean']:.3f} < -0.05"
    return True, "significant held-out loss win, no task-battery regression"


def run(synthetic: bool = False, ver: str | None = None, n_loss: int = N_LOSS, n_task: int = N_TASK, log=print) -> dict | None:
    import torch
    from peft import PeftModel
    from . import judge, model as M
    explicit = ver is not None
    ver = ver or latest_candidate(synthetic)
    if not ver:
        log("no candidate adapter to evaluate"); return None
    cand = config.ADAPTERS / ver
    if (cand / "eval.json").exists() and not explicit:
        log(f"{ver} already evaluated; nothing to do"); return None
    meta = json.loads((cand / "meta.json").read_text(encoding="utf-8"))
    rows, source = data.load_rows(synthetic)
    _, held = data.split(rows, data.load_meta() if not synthetic else None)
    trained = set(meta["new_ids"]) | set(meta["replay_ids"])
    held = [r for r in held if r["id"] not in trained]   # paranoia: never score on trained rows
    st = config.load_state()
    prod = (st.get("synthetic_production") if synthetic else st.get("production")) or {}
    prod_ver = prod.get("ver")
    loss_rows = _pick(held, n_loss, "loss")
    task_rows = _pick(held, n_task, "task")
    log(f"eval {ver} vs production {prod_ver or 'BASE (no adapter)'}; held-out {len(held)} rows from {len({r.get('session') for r in held})} sessions; loss n={len(loss_rows)}, battery n={len(task_rows)}")
    judge.unload()
    base = config.base_path(meta["base"])
    tok = M.tokenizer(base)
    m = M.load_4bit(base)
    m = PeftModel.from_pretrained(m, cand, adapter_name="cand")
    if prod_ver:
        m.load_adapter(config.ADAPTERS / prod_ver, adapter_name="prod")

    def with_prod(fn):
        if prod_ver:
            m.set_adapter("prod"); return fn()
        with m.disable_adapter():
            return fn()

    enc = [M.encode(tok, r["instruction"], r["response"]) for r in loss_rows]
    m.set_adapter("cand"); lc = M.example_losses(m, enc)
    lp = with_prod(lambda: M.example_losses(m, enc))
    smeta = data.load_meta() if not synthetic else {}
    unit = lambda r: (smeta.get(r["id"]) or {}).get("cluster") or r.get("session")  # noqa: E731  bootstrap unit: paraphrase cluster, else session
    per = [(unit(r), p[0] / max(1, p[1]) - c[0] / max(1, c[1])) for r, p, c in zip(loss_rows, lp, lc)]
    groups: dict = {}
    for s, d in per:
        groups.setdefault(s, []).append(d)
    loss_ci = stats.cluster_bootstrap_ci(list(groups.values()))
    loss_t = stats.paired_ci([d for _, d in per])
    nll_c = sum(c[0] for c in lc) / max(1, sum(c[1] for c in lc)); nll_p = sum(p[0] for p in lp) / max(1, sum(p[1] for p in lp))
    log(f"held-out token NLL: production {nll_p:.4f} (ppl {math.exp(nll_p):.2f}) -> candidate {nll_c:.4f} (ppl {math.exp(nll_c):.2f}); "
        f"per-example diff {loss_ci['mean']:+.4f} cluster-CI [{loss_ci['lo']:+.4f}, {loss_ci['hi']:+.4f}]")
    instr = [r["instruction"] for r in task_rows]
    m.set_adapter("cand"); ans_c = M.generate_batch(m, tok, instr)
    ans_p = with_prod(lambda: M.generate_batch(m, tok, instr))
    del m; gc.collect(); torch.cuda.empty_cache()
    sc_c = [judge.grade(r["instruction"], r["response"], a) for r, a in zip(task_rows, ans_c)]
    sc_p = [judge.grade(r["instruction"], r["response"], a) for r, a in zip(task_rows, ans_p)]
    # judge calibration: Claude's answer to the nearest-but-different held-out request (MiniLM-384) is a hard decoy
    from .vectors import Encoder, nearest_wrong
    hv = Encoder().encode([r["instruction"][:2000] for r in task_rows])
    dec = [judge.grade(r["instruction"], r["response"], task_rows[j]["response"]) for i, r in enumerate(task_rows)
           if (j := nearest_wrong(i, hv, [x["response"] for x in task_rows])) >= 0]
    decoy_reject = sum(1 for x in dec if x < 0.5) / len(dec) if dec else None
    log(f"judge decoy check: rejected {decoy_reject} of {len(dec)} nearest-but-wrong decoys")
    judge.unload()
    tg: dict = {}
    for r, a, b in zip(task_rows, sc_c, sc_p):
        tg.setdefault(unit(r), []).append(a - b)
    task_ci = stats.cluster_bootstrap_ci(list(tg.values()))
    log(f"battery (judge {config.JUDGE_MODEL}): production {sum(sc_p) / max(1, len(sc_p)):.3f} -> candidate {sum(sc_c) / max(1, len(sc_c)):.3f}; "
        f"diff {task_ci['mean']:+.3f} cluster-CI [{task_ci['lo']:+.3f}, {task_ci['hi']:+.3f}]")
    ok, why = gate(loss_ci, task_ci, len(groups), decoy_reject)
    res = {"ver": ver, "production": prod_ver, "data_source": source, "synthetic": synthetic, "base": meta["base"],
           "heldout_rows": len(held), "heldout_sessions": len(groups),
           "loss": {"nll_production": nll_p, "nll_candidate": nll_c, "ppl_production": math.exp(nll_p), "ppl_candidate": math.exp(nll_c),
                    "diff_cluster_ci": loss_ci, "diff_paired_t": loss_t},
           "battery": {"judge": config.JUDGE_MODEL, "rubric": judge.RUBRIC_VERSION, "mean_production": sum(sc_p) / max(1, len(sc_p)),
                       "mean_candidate": sum(sc_c) / max(1, len(sc_c)), "diff_cluster_ci": task_ci, "diff_paired_t": stats.paired_ci([a - b for a, b in zip(sc_c, sc_p)]),
                       "decoy_reject_rate": decoy_reject, "n_decoys": len(dec)},
           "gate": {"promote": ok, "reason": why}, "evaluated": datetime.now(timezone.utc).isoformat()}
    (cand / "eval.json").write_text(json.dumps(res, indent=1, default=lambda x: None if isinstance(x, float) and math.isinf(x) else x), encoding="utf-8")
    log(f"gate: {'PROMOTE' if ok else 'KEEP PRODUCTION'} - {why}")
    return res
